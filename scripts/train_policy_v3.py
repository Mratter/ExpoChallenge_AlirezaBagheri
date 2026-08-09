from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_native_thread_count = (
    "1" if os.environ.get("CITY_RECOVERY_V3_SPAWN_WORKER") == "1" else "12"
)
for _parent_thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_parent_thread_variable] = _native_thread_count

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.scenarios_v3 import (  # noqa: E402
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
)
from backend.app.simulator_core import canonical_hash  # noqa: E402
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_ORDER_V3,
    ENGINE_V3_SPEC_SHA256,
    OBSERVATION_ORDER_V3,
    SOLVED_DEFINITION_V3_SHA256,
    CityRecoveryEnvV3,
    CyclingScenarioEnvV3,
    public_preparedness_curriculum_action_v3,
)
from scripts.v3_protocol import (  # noqa: E402
    PROTOCOL_PATH,
    SOURCE_SEAL_PATH,
    TRAINING_AUTHORIZATION_PATH,
    scientific_source_identity,
    verify_documents,
    verify_training_authorization,
)

CONFIG_PATH = ROOT / "training" / "v3" / "config.json"
RUN_ROOT = ROOT / "internal" / "training_runs" / "v3"
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
LEDGER_PATH = RUN_ROOT / "checkpoint-ledger.jsonl"
SB3_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.zip"
ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.onnx"
METADATA_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.metadata.json"
MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.v3.json"
RECEIPT_PATH = ROOT / "training" / "v3" / "training-receipt.json"
PARITY_PATH = ROOT / "training" / "v3" / "onnx-parity.json"
TRAINING_USE_PATH = ROOT / "training" / "v3" / "training-use-receipt.json"
TERMINAL_MARKER_PATH = ROOT / "training" / "v3" / "training-terminal.json"
DEVELOPMENT_ROOT = ROOT / "internal" / "developmental_runs" / "v3"
STAGING_ROOT = RUN_ROOT / "staging"
POLICY_ID = "city-recovery-sb3-ppo-v3"
POLICY_VERSION = "3.0.0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"write-once artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    fsync_parent(path)


def publish_new(staged: Path, destination: Path) -> None:
    if not staged.is_file():
        raise RuntimeError(f"staged release artifact is missing: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(staged):
            raise RuntimeError(
                f"immutable release artifact conflicts with staging: {destination}"
            )
        return
    temporary = destination.with_suffix(
        destination.suffix + f".publish.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with staged.open("rb") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != sha256_file(staged):
            raise RuntimeError(f"staged release copy failed verification: {staged}")
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or sha256_file(destination) != sha256_file(staged):
            raise RuntimeError(
                f"immutable release artifact appeared with different bytes: {destination}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    fsync_parent(destination)


def _sealed_release_pairs(stage_root: Path) -> tuple[tuple[Path, Path], ...]:
    return (
        (stage_root / SB3_PATH.name, SB3_PATH),
        (stage_root / ONNX_PATH.name, ONNX_PATH),
        (stage_root / METADATA_PATH.name, METADATA_PATH),
        (stage_root / MANIFEST_PATH.name, MANIFEST_PATH),
        (stage_root / PARITY_PATH.name, PARITY_PATH),
        (stage_root / RECEIPT_PATH.name, RECEIPT_PATH),
    )


def _load_staged_release_receipt(
    stage_root: Path,
    campaign_identity: str,
    campaign_provenance: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a complete staged release before first or resumed publication."""

    pairs = _sealed_release_pairs(stage_root)
    missing = [str(staged) for staged, _ in pairs if not staged.is_file()]
    if missing:
        raise RuntimeError(
            "partial release cannot resume because staged artifacts are missing: "
            + ", ".join(missing)
        )
    for staged, destination in pairs:
        if destination.exists() and (
            not destination.is_file() or sha256_file(destination) != sha256_file(staged)
        ):
            raise RuntimeError(
                f"published release bytes conflict with staging: {destination}"
            )
    try:
        receipt = json.loads(
            (stage_root / RECEIPT_PATH.name).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("staged training receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("staged training receipt root must be an object")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_semantic_sha256"
    }
    artifacts = receipt.get("artifacts")
    terminal_checkpoint = receipt.get("terminal_checkpoint")
    if (
        receipt.get("status") != "completed"
        or receipt.get("execution_class") != "sealed_authorized_campaign"
        or receipt.get("campaign_provenance") != campaign_provenance
        or receipt.get("campaign_provenance_sha256") != campaign_identity
        or receipt.get("receipt_semantic_sha256") != canonical_hash(unsigned_receipt)
        or receipt.get("completed_transitions") != int(config["total_transitions"])
        or receipt.get("completed_stage_count")
        != int(config["production_stages"]["count"])
        or receipt.get("checkpoint_ledger_sha256") != sha256_file(LEDGER_PATH)
        or not isinstance(artifacts, dict)
        or not isinstance(terminal_checkpoint, dict)
    ):
        raise RuntimeError("staged training receipt provenance drifted")
    expected_bindings = {
        "terminal_checkpoint": (
            terminal_checkpoint,
            SB3_PATH,
            stage_root / SB3_PATH.name,
        ),
        "training_onnx": (
            artifacts.get("training_onnx"),
            ONNX_PATH,
            stage_root / ONNX_PATH.name,
        ),
        "metadata": (
            artifacts.get("metadata"),
            METADATA_PATH,
            stage_root / METADATA_PATH.name,
        ),
        "manifest": (
            artifacts.get("manifest"),
            MANIFEST_PATH,
            stage_root / MANIFEST_PATH.name,
        ),
        "parity": (
            artifacts.get("parity"),
            PARITY_PATH,
            stage_root / PARITY_PATH.name,
        ),
    }
    for label, (binding, released, staged) in expected_bindings.items():
        expected_path = str(released.relative_to(ROOT)).replace("\\", "/")
        if (
            not isinstance(binding, dict)
            or binding.get("path") != expected_path
            or binding.get("sha256") != sha256_file(staged)
            or (
                label == "terminal_checkpoint"
                and binding.get("bytes") != staged.stat().st_size
            )
        ):
            raise RuntimeError(f"staged {label} binding drifted")
    if receipt.get("manifest_sha256") != sha256_file(
        stage_root / MANIFEST_PATH.name
    ) or receipt.get("parity_sha256") != sha256_file(stage_root / PARITY_PATH.name):
        raise RuntimeError("staged manifest or parity receipt binding drifted")
    return receipt


def _publish_sealed_release(
    stage_root: Path,
    campaign_identity: str,
    campaign_provenance: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Idempotently finish an identity-matching release and its terminal marker."""

    receipt = _load_staged_release_receipt(
        stage_root, campaign_identity, campaign_provenance, config
    )
    for staged, destination in _sealed_release_pairs(stage_root):
        publish_new(staged, destination)
    terminal_base = {
        "schema_version": 1,
        "status": "sealed_training_release_complete",
        "campaign_identity": campaign_identity,
        "campaign_provenance_sha256": canonical_hash(campaign_provenance),
        "configured_transitions": int(config["total_transitions"]),
        "actual_completed_transitions": int(receipt["completed_transitions"]),
        "completed_stage_count": int(config["production_stages"]["count"]),
        "training_use_receipt_sha256": sha256_file(TRAINING_USE_PATH),
        "training_receipt_sha256": sha256_file(RECEIPT_PATH),
        "checkpoint_ledger_sha256": sha256_file(LEDGER_PATH),
        "checkpoint_ledger_terminal_record_sha256": receipt[
            "checkpoint_ledger_terminal_record_sha256"
        ],
        "release_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                SB3_PATH,
                ONNX_PATH,
                METADATA_PATH,
                MANIFEST_PATH,
                PARITY_PATH,
                RECEIPT_PATH,
            )
        },
    }
    terminal = {
        **terminal_base,
        "terminal_semantic_sha256": canonical_hash(terminal_base),
    }
    if TERMINAL_MARKER_PATH.exists():
        try:
            existing_terminal = json.loads(
                TERMINAL_MARKER_PATH.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError("training terminal marker is invalid") from exc
        if existing_terminal != terminal:
            raise RuntimeError("training terminal marker identity drifted")
    else:
        write_new_json(TERMINAL_MARKER_PATH, terminal)
    return receipt


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["algorithm"] != "Stable-Baselines3 PPO":
        raise RuntimeError("training config algorithm drifted")
    if config["device"] != "cpu":
        raise RuntimeError("training config device drifted")
    if config["environment"] != "CityRecoveryEnv-v3":
        raise RuntimeError("training config environment drifted")
    if config["observation_count"] != len(OBSERVATION_ORDER_V3):
        raise RuntimeError("training config observation count drifted")
    if config["action_count"] != len(ACTION_ORDER_V3):
        raise RuntimeError("training config action count drifted")
    curriculum = config.get("behavior_cloning")
    if not isinstance(curriculum, dict) or curriculum.get("teacher") != (
        "public-preparedness-curriculum-v3"
    ):
        raise RuntimeError("training curriculum teacher drifted")
    if curriculum.get("teacher_version") != "1.0.0":
        raise RuntimeError("training curriculum teacher version drifted")
    if not curriculum.get("training_split_only"):
        raise RuntimeError("behavior cloning must remain training-split-only")
    if (
        int(curriculum.get("iterations", 0)) < 1
        or int(curriculum.get("epochs_per_iteration", 0)) < 1
    ):
        raise RuntimeError("behavior-cloning iteration schedule is invalid")
    expected_beta = [1.0] + [0.0] * (int(curriculum["iterations"]) - 1)
    if curriculum.get("dagger_beta_schedule") != expected_beta:
        raise RuntimeError("behavior-cloning DAgger beta schedule drifted")
    if config["network"] != {
        "activation": "SiLU",
        "actor": [384, 256, 128],
        "critic": [384, 256, 128],
        "orthogonal_initialization": True,
    }:
        raise RuntimeError("training network topology drifted")
    rollout_batch = int(config["n_steps_per_lane"]) * int(config["simulator_lanes"])
    if rollout_batch % int(config["batch_size"]) != 0:
        raise RuntimeError("PPO rollout batch must divide into exact minibatches")
    if int(config["expected_trainable_parameters"]) != 322_733:
        raise RuntimeError("expected trainable parameter count drifted")
    stages = config.get("production_stages")
    vectorization = config.get("vectorization")
    if (
        not isinstance(stages, dict)
        or stages.get("count") != 7
        or stages.get("rollouts_per_stage") != 30
        or stages.get("transitions_per_stage") != 30 * rollout_batch
        or stages.get("episode_steps_per_lane") != 30
        or (stages["transitions_per_stage"] // int(config["simulator_lanes"]))
        % stages["episode_steps_per_lane"]
        != 0
        or config["checkpoint_interval_transitions"] != stages["transitions_per_stage"]
        or config["total_transitions"]
        != stages["count"] * stages["transitions_per_stage"]
        or stages.get("resume_at_stage_boundaries") is not True
        or stages.get("fresh_environment_per_stage") is not True
        or vectorization
        != {
            "backend": "SubprocVecEnv",
            "parent_torch_inter_op_threads": 1,
            "parent_torch_intra_op_threads": 12,
            "start_method": "spawn",
            "worker_blas_threads": 1,
        }
    ):
        raise RuntimeError("production stage/resume contract drifted")
    return config


def source_identity() -> dict[str, Any]:
    return scientific_source_identity()


def training_scenarios() -> list[tuple[Any, int]]:
    return [
        (family.build(seed), family.tape_seed(seed))
        for family in TRAINING_FAMILIES_V3
        for seed in TRAINING_SEEDS_V3
    ]


class LedgerCheckpointCallback(BaseCallback):
    def __init__(
        self,
        interval: int,
        campaign_provenance: dict[str, Any],
        checkpoint_root: Path,
        ledger_path: Path,
    ):
        super().__init__(verbose=0)
        self.interval = interval
        self.next_transition = interval
        self.campaign_provenance = campaign_provenance
        self.campaign_provenance_sha256 = canonical_hash(campaign_provenance)
        self.checkpoint_root = checkpoint_root
        self.ledger_path = ledger_path

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_transition:
            return True
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_root / f"transition-{self.num_timesteps:09d}"
        self.model.save(str(checkpoint))
        path = checkpoint.with_suffix(".zip")
        fsync_file(path)
        fsync_parent(path)
        record = {
            "schema_version": 1,
            "record_type": "v3_training_checkpoint",
            "campaign_provenance": self.campaign_provenance,
            "campaign_provenance_sha256": self.campaign_provenance_sha256,
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "num_timesteps": self.num_timesteps,
            "registered_threshold": self.next_transition,
            "sha256": sha256_file(path),
            "wall_time_unix": time.time(),
        }
        previous_hash = "0" * 64
        if self.ledger_path.exists():
            lines = [
                line
                for line in self.ledger_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if lines:
                previous_hash = json.loads(lines[-1])["record_sha256"]
        record["previous_record_sha256"] = previous_hash
        record["record_sha256"] = canonical_hash(record)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        fsync_parent(self.ledger_path)
        while self.next_transition <= self.num_timesteps:
            self.next_transition += self.interval
        return True


def _stage_checkpoint_path(
    checkpoint_root: Path, stage_number: int, completed_transitions: int
) -> Path:
    return checkpoint_root / (
        f"stage-{stage_number:02d}-transition-{completed_transitions:09d}.zip"
    )


def _load_completed_stage_records(
    config: dict[str, Any],
    campaign_provenance: dict[str, Any],
    checkpoint_root: Path,
    ledger_path: Path,
) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    if any(not line for line in lines):
        raise RuntimeError("production stage ledger contains a blank record")
    stage_transitions = int(config["production_stages"]["transitions_per_stage"])
    stage_count = int(config["production_stages"]["count"])
    if len(lines) > stage_count:
        raise RuntimeError("production stage ledger exceeds its registered stage count")
    records: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    provenance_sha256 = canonical_hash(campaign_provenance)
    for stage_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise RuntimeError("production stage ledger is invalid JSON") from exc
        if not isinstance(record, dict):
            raise RuntimeError("production stage ledger record must be an object")
        completed = stage_number * stage_transitions
        expected_path = _stage_checkpoint_path(checkpoint_root, stage_number, completed)
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if (
            record.get("schema_version") != 1
            or record.get("record_type") != "v3_training_stage_checkpoint"
            or record.get("stage_number") != stage_number
            or record.get("stage_start_transitions")
            != (stage_number - 1) * stage_transitions
            or record.get("stage_transitions") != stage_transitions
            or record.get("num_timesteps") != completed
            or record.get("stage_seed")
            != int(config["policy_seed"]) + (stage_number - 1) * 10_007
            or record.get("fresh_environment_reset") is not True
            or record.get("campaign_provenance") != campaign_provenance
            or record.get("campaign_provenance_sha256") != provenance_sha256
            or record.get("checkpoint")
            != str(expected_path.relative_to(ROOT)).replace("\\", "/")
            or record.get("previous_record_sha256") != previous_hash
            or record.get("record_sha256") != canonical_hash(unsigned)
            or not expected_path.is_file()
            or record.get("sha256") != sha256_file(expected_path)
        ):
            raise RuntimeError("production stage checkpoint provenance drifted")
        records.append(record)
        previous_hash = record["record_sha256"]
    return records


def _commit_completed_stage(
    model: PPO,
    config: dict[str, Any],
    campaign_provenance: dict[str, Any],
    stage_number: int,
    checkpoint_root: Path,
    ledger_path: Path,
    previous_hash: str,
) -> dict[str, Any]:
    stage_transitions = int(config["production_stages"]["transitions_per_stage"])
    completed = stage_number * stage_transitions
    if int(model.num_timesteps) != completed:
        raise RuntimeError("PPO stage did not end on its exact registered boundary")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = _stage_checkpoint_path(checkpoint_root, stage_number, completed)
    temporary_base = checkpoint_root / (
        f".stage-{stage_number:02d}-pid-{os.getpid()}.tmp"
    )
    temporary_zip = Path(str(temporary_base) + ".zip")
    temporary_zip.unlink(missing_ok=True)
    # Stable-Baselines3 only appends ``.zip`` when the supplied path has no
    # suffix.  ``temporary_base`` intentionally ends in ``.tmp``, so passing
    # that path directly would create the extensionless file and leave the
    # durability check looking for a file that does not exist.  Give SB3 the
    # exact ``.zip`` path we fsync and atomically publish.
    model.save(temporary_zip)
    fsync_file(temporary_zip)
    os.replace(temporary_zip, destination)
    fsync_parent(destination)
    record = {
        "schema_version": 1,
        "record_type": "v3_training_stage_checkpoint",
        "stage_number": stage_number,
        "stage_start_transitions": (stage_number - 1) * stage_transitions,
        "stage_transitions": stage_transitions,
        "num_timesteps": completed,
        "stage_seed": int(config["policy_seed"]) + (stage_number - 1) * 10_007,
        "fresh_environment_reset": True,
        "campaign_provenance": campaign_provenance,
        "campaign_provenance_sha256": canonical_hash(campaign_provenance),
        "checkpoint": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(destination),
        "previous_record_sha256": previous_hash,
    }
    record["record_sha256"] = canonical_hash(record)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_parent(ledger_path)
    return record


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.policy._predict(observation, deterministic=True)  # type: ignore[attr-defined]
        return torch.clamp(action, -1.0, 1.0)


def _set_stage_seed(config: dict[str, Any], stage_index: int) -> int:
    seed = int(config["policy_seed"]) + stage_index * 10_007
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed, using_cuda=False)
    return seed


@dataclass(frozen=True)
class _TrainingLaneFactory:
    """Pickle-safe constructor for one deterministic spawned training lane."""

    lane: int
    stage_index: int
    lane_count: int
    worker_blas_threads: int

    def __call__(self) -> CyclingScenarioEnvV3:
        expected_threads = str(self.worker_blas_threads)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if os.environ.get(variable) != expected_threads:
                raise RuntimeError(
                    f"spawned worker thread contract drifted: {variable}"
                )
        torch.set_num_threads(self.worker_blas_threads)
        torch.set_num_interop_threads(1)
        scenarios = training_scenarios()
        offset = (self.stage_index * self.lane_count + self.lane) % len(scenarios)
        rotated = scenarios[offset:] + scenarios[:offset]
        return CyclingScenarioEnvV3(rotated)


def _spawn_training_environment(
    factories: list[Callable[[], CyclingScenarioEnvV3]], worker_threads: int
) -> SubprocVecEnv:
    """Spawn workers while they inherit one-thread native-library limits."""

    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {variable: os.environ.get(variable) for variable in thread_variables}
    previous_worker_marker = os.environ.get("CITY_RECOVERY_V3_SPAWN_WORKER")
    for variable in thread_variables:
        os.environ[variable] = str(worker_threads)
    os.environ["CITY_RECOVERY_V3_SPAWN_WORKER"] = "1"
    try:
        return SubprocVecEnv(factories, start_method="spawn")
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value
        if previous_worker_marker is None:
            os.environ.pop("CITY_RECOVERY_V3_SPAWN_WORKER", None)
        else:
            os.environ["CITY_RECOVERY_V3_SPAWN_WORKER"] = previous_worker_marker


def build_environment(
    config: dict[str, Any], stage_index: int, *, check_contract: bool = False
) -> VecEnv:
    _set_stage_seed(config, stage_index)
    scenarios = training_scenarios()
    if check_contract:
        check_env(CityRecoveryEnvV3(*scenarios[0]), warn=True, skip_render_check=True)
    lane_count = int(config["simulator_lanes"])
    worker_threads = int(config["vectorization"]["worker_blas_threads"])
    factories: list[Callable[[], CyclingScenarioEnvV3]] = [
        _TrainingLaneFactory(lane, stage_index, lane_count, worker_threads)
        for lane in range(lane_count)
    ]
    environment = _spawn_training_environment(factories, worker_threads)
    environment.seed(int(config["policy_seed"]) + stage_index * 10_007)
    return environment


def build_model(config: dict[str, Any]) -> tuple[PPO, VecEnv]:
    seed = _set_stage_seed(config, 0)
    torch.set_num_threads(int(config["vectorization"]["parent_torch_intra_op_threads"]))
    torch.set_num_interop_threads(
        int(config["vectorization"]["parent_torch_inter_op_threads"])
    )
    torch.use_deterministic_algorithms(True)
    environment = build_environment(config, 0, check_contract=True)
    network = config["network"]
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps_per_lane"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs={
            "activation_fn": torch.nn.SiLU,
            "net_arch": {"pi": network["actor"], "vf": network["critic"]},
            "ortho_init": bool(network["orthogonal_initialization"]),
            "log_std_init": float(config["log_std_init"]),
        },
        seed=seed,
        device="cpu",
        verbose=1,
    )
    parameter_count = sum(parameter.numel() for parameter in model.policy.parameters())
    if parameter_count != int(config["expected_trainable_parameters"]):
        raise RuntimeError(f"v3 policy parameter count drifted: {parameter_count}")
    return model, environment


def behavior_clone_actor(model: PPO, config: dict[str, Any]) -> dict[str, Any]:
    curriculum = config["behavior_cloning"]
    if not curriculum["enabled"]:
        return {"enabled": False, "observation_count": 0, "iterations": 0}
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    actor_parameters = [
        *model.policy.mlp_extractor.policy_net.parameters(),
        *model.policy.action_net.parameters(),
    ]
    optimizer = torch.optim.Adam(
        actor_parameters,
        lr=float(curriculum["learning_rate"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["policy_seed"]) ^ 0xBC37017)
    batch_size = int(curriculum["batch_size"])
    final_loss = float("nan")
    iteration_reports: list[dict[str, Any]] = []
    for iteration in range(int(curriculum["iterations"])):
        for scenario, tape_seed in training_scenarios():
            environment = CityRecoveryEnvV3(scenario, tape_seed)
            observation, _ = environment.reset(seed=tape_seed)
            terminated = False
            while not terminated:
                teacher_action, evidence = public_preparedness_curriculum_action_v3(
                    observation
                )
                teacher_action = np.asarray(teacher_action, dtype=np.float32)
                if observation.shape != (len(OBSERVATION_ORDER_V3),) or not np.all(
                    np.isfinite(observation)
                ):
                    raise RuntimeError("behavior-cloning observation contract drifted")
                if teacher_action.shape != (len(ACTION_ORDER_V3),) or not np.all(
                    np.isfinite(teacher_action)
                ):
                    raise RuntimeError(
                        "behavior-cloning teacher action contract drifted"
                    )
                if np.any(teacher_action < -1.0) or np.any(teacher_action > 1.0):
                    raise RuntimeError(
                        "behavior-cloning teacher action is out of range"
                    )
                if (
                    evidence.get("teacher_id") != curriculum["teacher"]
                    or evidence.get("teacher_version") != curriculum["teacher_version"]
                ):
                    raise RuntimeError("behavior-cloning teacher identity drifted")
                if evidence["future_tape_visible"]:
                    raise RuntimeError(
                        "behavior-cloning teacher exposed the future tape"
                    )
                observations.append(observation.copy())
                targets.append(teacher_action)
                if float(curriculum["dagger_beta_schedule"][iteration]) == 1.0:
                    rollout_action = teacher_action
                else:
                    rollout_action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, _, _ = environment.step(rollout_action)
        observation_tensor = torch.as_tensor(
            np.asarray(observations), dtype=torch.float32
        )
        target_tensor = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        model.policy.train()
        for _ in range(int(curriculum["epochs_per_iteration"])):
            permutation = torch.randperm(
                observation_tensor.shape[0], generator=generator
            )
            for start in range(0, observation_tensor.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                distribution = model.policy.get_distribution(
                    observation_tensor[indices]
                )
                predicted = distribution.distribution.mean
                loss = torch.nn.functional.mse_loss(predicted, target_tensor[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                optimizer.step()
                final_loss = float(loss.detach().cpu())
        model.policy.eval()
        with torch.no_grad():
            distribution = model.policy.get_distribution(observation_tensor)
            predicted = distribution.distribution.mean
            iteration_loss = float(
                torch.nn.functional.mse_loss(predicted, target_tensor).detach().cpu()
            )
        iteration_reports.append(
            {
                "iteration": iteration + 1,
                "cumulative_observation_count": int(observation_tensor.shape[0]),
                "full_dataset_mse": iteration_loss,
            }
        )
    model.policy.eval()
    return {
        "enabled": True,
        "teacher": curriculum["teacher"],
        "teacher_version": curriculum["teacher_version"],
        "training_split_only": bool(curriculum["training_split_only"]),
        "dagger_beta_schedule": curriculum["dagger_beta_schedule"],
        "observation_order_sha256": canonical_hash(list(OBSERVATION_ORDER_V3)),
        "action_order_sha256": canonical_hash(list(ACTION_ORDER_V3)),
        "observation_count": int(observation_tensor.shape[0]),
        "iterations": int(curriculum["iterations"]),
        "epochs_per_iteration": int(curriculum["epochs_per_iteration"]),
        "batch_size": batch_size,
        "final_batch_mse": final_loss,
        "full_dataset_mse": iteration_reports[-1]["full_dataset_mse"],
        "iteration_reports": iteration_reports,
    }


def _load_curriculum_receipt(path: Path, campaign_identity: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("curriculum receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("curriculum receipt root must be an object")
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_semantic_sha256"
    }
    if (
        set(receipt)
        != {
            "schema_version",
            "status",
            "campaign_provenance_sha256",
            "deterministic_replay_permitted_before_first_stage",
            "curriculum",
            "receipt_semantic_sha256",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "behavior_cloning_complete"
        or receipt.get("campaign_provenance_sha256") != campaign_identity
        or receipt.get("deterministic_replay_permitted_before_first_stage") is not True
        or not isinstance(receipt.get("curriculum"), dict)
        or receipt.get("receipt_semantic_sha256") != canonical_hash(unsigned)
    ):
        raise RuntimeError("curriculum receipt provenance drifted")
    return receipt


def export_onnx(model: PPO, onnx_path: Path) -> None:
    wrapper = OnnxablePolicy(model.policy).eval()
    dummy = torch.zeros((1, len(OBSERVATION_ORDER_V3)), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def parity_report(model: PPO, onnx_path: Path) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        onnx_path.read_bytes(), options, providers=["CPUExecutionProvider"]
    )
    max_error = 0.0
    cases = 0
    for scenario, seed in training_scenarios()[:8]:
        env = CityRecoveryEnvV3(scenario, seed)
        observation, _ = env.reset(seed=seed)
        terminated = False
        while not terminated and cases < 64:
            pytorch_action, _ = model.predict(observation, deterministic=True)
            onnx_action = session.run(
                ["action"],
                {"observation": observation.reshape(1, -1).astype(np.float32)},
            )[0][0]
            max_error = max(
                max_error,
                float(
                    np.max(np.abs(np.asarray(pytorch_action) - np.asarray(onnx_action)))
                ),
            )
            observation, _, terminated, _, _ = env.step(pytorch_action)
            cases += 1
        if cases >= 64:
            break
    return {
        "cases": cases,
        "max_action_absolute_error": max_error,
        "passed": cases == 64 and max_error <= 1e-5,
        "providers": session.get_providers(),
        "onnx_sha256": sha256_file(onnx_path),
    }


def vectorization_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Exercise the exact spawned trainer runtime without training or writing files."""

    environment: VecEnv | None = None
    worker_processes: list[Any] = []
    try:
        environment = build_environment(config, 0, check_contract=True)
        if not isinstance(environment, SubprocVecEnv):
            raise RuntimeError("training vectorization backend drifted")
        worker_processes = list(environment.processes)
        if len(worker_processes) != int(config["simulator_lanes"]) or not all(
            process.is_alive() for process in worker_processes
        ):
            raise RuntimeError("spawned trainer worker tree is incomplete")
        observation = np.asarray(environment.reset())
        action = np.zeros(
            (int(config["simulator_lanes"]), len(ACTION_ORDER_V3)),
            dtype=np.float32,
        )
        next_observation, reward, done, infos = environment.step(action)
        next_observation = np.asarray(next_observation)
        reward = np.asarray(reward)
        done = np.asarray(done)
        if (
            observation.shape
            != (int(config["simulator_lanes"]), len(OBSERVATION_ORDER_V3))
            or next_observation.shape != observation.shape
            or reward.shape != (int(config["simulator_lanes"]),)
            or done.shape != (int(config["simulator_lanes"]),)
            or len(infos) != int(config["simulator_lanes"])
            or not np.all(np.isfinite(observation))
            or not np.all(np.isfinite(next_observation))
            or not np.all(np.isfinite(reward))
        ):
            raise RuntimeError("spawned trainer transition contract drifted")
        receipt = {
            "schema_version": 1,
            "status": "spawn_vectorization_preflight_passed_nonauthorizing",
            "backend": "SubprocVecEnv",
            "start_method": "spawn",
            "lane_count": len(worker_processes),
            "worker_pids": [int(process.pid) for process in worker_processes],
            "observation_shape": list(observation.shape),
            "action_shape": list(action.shape),
            "next_observation_shape": list(next_observation.shape),
            "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
            "next_observation_sha256": hashlib.sha256(
                next_observation.tobytes()
            ).hexdigest(),
            "reward_sha256": hashlib.sha256(reward.tobytes()).hexdigest(),
            "done_sha256": hashlib.sha256(done.tobytes()).hexdigest(),
            "writes_artifacts": False,
        }
    finally:
        if environment is not None:
            environment.close()
    if any(process.is_alive() for process in worker_processes):
        raise RuntimeError("spawned trainer worker tree did not close")
    receipt["all_workers_closed"] = True
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CityRecoveryEnv-v3 PPO candidate."
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--developmental-unsealed",
        action="store_true",
        help=(
            "permit one explicitly nonauthorizing shortened learning-dynamics run "
            "only while protocol and source seal are absent"
        ),
    )
    parser.add_argument(
        "--vectorization-preflight-only",
        action="store_true",
        help="spawn/reset/step/close the registered lanes without training or writes",
    )
    args = parser.parse_args()
    config = load_config()
    if args.vectorization_preflight_only:
        if args.developmental_unsealed or args.timesteps is not None:
            raise RuntimeError(
                "vectorization preflight cannot be combined with training"
            )
        print(json.dumps(vectorization_preflight(config), sort_keys=True))
        return
    protocol_exists = PROTOCOL_PATH.exists()
    seal_exists = SOURCE_SEAL_PATH.exists()
    governance: dict[str, Any] | None = None
    if protocol_exists != seal_exists:
        raise RuntimeError("partial V3 freeze detected; refusing to train")
    if protocol_exists:
        if args.developmental_unsealed or args.timesteps is not None:
            raise RuntimeError(
                "the frozen campaign forbids developmental mode and timestep overrides"
            )
        verification = verify_documents(require_final_unused=True)
        verify_training_authorization()
        governance = {
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "source_seal_sha256": sha256_file(SOURCE_SEAL_PATH),
            "training_authorization_sha256": sha256_file(TRAINING_AUTHORIZATION_PATH),
            "scientific_source_sha256": verification["scientific_source_sha256"],
        }
        execution_class = "sealed_authorized_campaign"
    else:
        if not args.developmental_unsealed:
            raise RuntimeError(
                "unsealed V3 training requires explicit --developmental-unsealed"
            )
        if args.timesteps is None:
            raise RuntimeError(
                "an unsealed developmental run requires an explicit shortened --timesteps"
            )
        if args.timesteps >= int(config["total_transitions"]):
            raise RuntimeError(
                "an unsealed developmental run must be shorter than the production budget"
            )
        execution_class = "developmental_unsealed_nonauthorizing"
    if args.timesteps is not None:
        if args.timesteps <= 0 or args.timesteps > int(config["total_transitions"]):
            raise ValueError(
                "--timesteps must be positive and no larger than the configured budget"
            )
        config = {**config, "total_transitions": args.timesteps}
    source = source_identity()
    runtime_provenance = {
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
            "torch_intra_op_threads": int(
                config["vectorization"]["parent_torch_intra_op_threads"]
            ),
            "torch_inter_op_threads": int(
                config["vectorization"]["parent_torch_inter_op_threads"]
            ),
            "worker_blas_threads": int(config["vectorization"]["worker_blas_threads"]),
        },
    }
    campaign_provenance = {
        "schema_version": 1,
        "execution_class": execution_class,
        "governance": governance,
        "scientific_source_sha256": source["semantic_sha256"],
        "registered_config_sha256": sha256_file(CONFIG_PATH),
        "effective_config_sha256": canonical_hash(config),
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "policy_seed": int(config["policy_seed"]),
        "configured_transitions": int(config["total_transitions"]),
        "simulator_lanes": int(config["simulator_lanes"]),
        "checkpoint_interval_transitions": int(
            config["checkpoint_interval_transitions"]
        ),
        "production_stages": config["production_stages"],
        "vectorization": config["vectorization"],
        "runtime": runtime_provenance,
    }
    campaign_identity = canonical_hash(campaign_provenance)
    if execution_class == "sealed_authorized_campaign":
        if TERMINAL_MARKER_PATH.exists():
            raise RuntimeError("sealed training campaign is already terminal")
        run_output_root = STAGING_ROOT / campaign_identity
        use_receipt = {
            "schema_version": 1,
            "status": "sealed_training_campaign_claimed",
            "campaign_identity": campaign_identity,
            "campaign_provenance": campaign_provenance,
            "campaign_provenance_sha256": campaign_identity,
            "single_use": True,
            "stage_resume_only": True,
            "behavior_cloning_resume": ("deterministic_replay_before_first_stage_only"),
        }
        if not TRAINING_USE_PATH.exists() and (
            LEDGER_PATH.exists() or CHECKPOINT_ROOT.exists() or run_output_root.exists()
        ):
            raise RuntimeError(
                "unclaimed preexisting checkpoint state must be preserved outside production paths"
            )
        if TRAINING_USE_PATH.exists():
            try:
                existing_use = json.loads(TRAINING_USE_PATH.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise RuntimeError("sealed training use receipt is invalid") from exc
            if existing_use != use_receipt:
                raise RuntimeError("sealed training use receipt identity drifted")
        else:
            write_new_json(TRAINING_USE_PATH, use_receipt)
        run_output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root = CHECKPOINT_ROOT
        ledger_path = LEDGER_PATH
        output_sb3_path = run_output_root / SB3_PATH.name
        output_onnx_path = run_output_root / ONNX_PATH.name
        output_metadata_path = run_output_root / METADATA_PATH.name
        output_manifest_path = run_output_root / MANIFEST_PATH.name
        output_receipt_path = run_output_root / RECEIPT_PATH.name
        output_parity_path = run_output_root / PARITY_PATH.name
        release_sb3_path = SB3_PATH
        release_onnx_path = ONNX_PATH
        release_metadata_path = METADATA_PATH
        release_manifest_path = MANIFEST_PATH
        release_parity_path = PARITY_PATH
        release_pairs = _sealed_release_pairs(run_output_root)
        any_published = any(destination.exists() for _, destination in release_pairs)
        complete_staging = all(staged.is_file() for staged, _ in release_pairs)
        if any_published or complete_staging:
            recovered_receipt = _publish_sealed_release(
                run_output_root,
                campaign_identity,
                campaign_provenance,
                config,
            )
            print(json.dumps(recovered_receipt, sort_keys=True))
            return
    else:
        run_output_root = DEVELOPMENT_ROOT / campaign_identity
        if run_output_root.exists():
            raise RuntimeError(
                "developmental run identity already exists; never overwrite it"
            )
        run_output_root.mkdir(parents=True)
        checkpoint_root = run_output_root / "checkpoints"
        ledger_path = run_output_root / "checkpoint-ledger.jsonl"
        output_sb3_path = run_output_root / SB3_PATH.name
        output_onnx_path = run_output_root / ONNX_PATH.name
        output_metadata_path = run_output_root / METADATA_PATH.name
        output_manifest_path = run_output_root / MANIFEST_PATH.name
        output_receipt_path = run_output_root / RECEIPT_PATH.name
        output_parity_path = run_output_root / PARITY_PATH.name
        release_sb3_path = output_sb3_path
        release_onnx_path = output_onnx_path
        release_metadata_path = output_metadata_path
        release_manifest_path = output_manifest_path
        release_parity_path = output_parity_path
    started = time.perf_counter()
    resumed_from_stage = 0
    if execution_class == "sealed_authorized_campaign":
        completed_stage_records = _load_completed_stage_records(
            config, campaign_provenance, checkpoint_root, ledger_path
        )
        completed_stage_count = len(completed_stage_records)
        resumed_from_stage = completed_stage_count
        curriculum_path = run_output_root / "behavior-cloning-receipt.json"
        if completed_stage_records:
            if not curriculum_path.is_file():
                raise RuntimeError(
                    "resumable stage state is missing its curriculum receipt"
                )
            curriculum_receipt = _load_curriculum_receipt(
                curriculum_path, campaign_identity
            )
            curriculum = curriculum_receipt["curriculum"]
            environment = build_environment(config, completed_stage_count)
            try:
                model = PPO.load(
                    ROOT / completed_stage_records[-1]["checkpoint"],
                    env=environment,
                    device="cpu",
                )
                expected_completed = completed_stage_count * int(
                    config["production_stages"]["transitions_per_stage"]
                )
                if int(model.num_timesteps) != expected_completed:
                    raise RuntimeError("resumable PPO checkpoint timestep drifted")
            except BaseException:
                environment.close()
                raise
        else:
            model, environment = build_model(config)
            try:
                curriculum = behavior_clone_actor(model, config)
                existing_curriculum_receipt = (
                    _load_curriculum_receipt(curriculum_path, campaign_identity)
                    if curriculum_path.exists()
                    else None
                )
                curriculum_receipt_base = {
                    "schema_version": 1,
                    "status": "behavior_cloning_complete",
                    "campaign_provenance_sha256": campaign_identity,
                    "deterministic_replay_permitted_before_first_stage": True,
                    "curriculum": curriculum,
                }
                curriculum_receipt = {
                    **curriculum_receipt_base,
                    "receipt_semantic_sha256": canonical_hash(curriculum_receipt_base),
                }
                if existing_curriculum_receipt is not None:
                    if existing_curriculum_receipt != curriculum_receipt:
                        raise RuntimeError(
                            "deterministic behavior-cloning replay drifted before stage one"
                        )
                else:
                    write_new_json(curriculum_path, curriculum_receipt)
            except BaseException:
                environment.close()
                raise
        parameter_count = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        stage_count = int(config["production_stages"]["count"])
        stage_transitions = int(config["production_stages"]["transitions_per_stage"])
        previous_hash = (
            completed_stage_records[-1]["record_sha256"]
            if completed_stage_records
            else "0" * 64
        )
        if completed_stage_count == stage_count:
            environment.close()
        for stage_index in range(completed_stage_count, stage_count):
            if stage_index > completed_stage_count:
                environment = build_environment(config, stage_index)
                model.set_env(environment, force_reset=True)
            try:
                stage_seed = _set_stage_seed(config, stage_index)
                model.set_random_seed(stage_seed)
                model.learn(
                    total_timesteps=stage_transitions,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                stage_record = _commit_completed_stage(
                    model,
                    config,
                    campaign_provenance,
                    stage_index + 1,
                    checkpoint_root,
                    ledger_path,
                    previous_hash,
                )
                previous_hash = stage_record["record_sha256"]
            finally:
                environment.close()
    else:
        model, environment = build_model(config)
        parameter_count = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        try:
            curriculum = behavior_clone_actor(model, config)
            callback = LedgerCheckpointCallback(
                int(config["checkpoint_interval_transitions"]),
                campaign_provenance,
                checkpoint_root,
                ledger_path,
            )
            model.learn(
                total_timesteps=int(config["total_transitions"]),
                callback=callback,
                progress_bar=False,
            )
        finally:
            environment.close()
    elapsed = time.perf_counter() - started
    model.save(output_sb3_path)
    fsync_file(output_sb3_path)
    fsync_parent(output_sb3_path)
    export_onnx(model, output_onnx_path)
    fsync_file(output_onnx_path)
    fsync_parent(output_onnx_path)
    parity = parity_report(model, output_onnx_path)
    if not parity["passed"]:
        raise RuntimeError(f"v3 ONNX parity failed: {parity}")
    write_json(output_parity_path, parity)
    manifest = {
        "schema_version": 1,
        "model": {
            "id": POLICY_ID,
            "version": POLICY_VERSION,
            "algorithm": "PPO",
            "environment": "CityRecoveryEnv-v3",
            "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
            "solved_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "trainable_parameters": parameter_count,
            "completed_training_transitions": int(model.num_timesteps),
            "training_seed": int(config["policy_seed"]),
        },
        "files": [],
    }
    for file_id, path, released_path in (
        ("onnx", output_onnx_path, release_onnx_path),
        ("sb3_checkpoint", output_sb3_path, release_sb3_path),
    ):
        manifest["files"].append(
            {
                "id": file_id,
                "path": str(released_path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(output_manifest_path, manifest)
    metadata = {
        "id": POLICY_ID,
        "version": POLICY_VERSION,
        "artifact_type": "stable_baselines3_ppo",
        "observation_order": list(OBSERVATION_ORDER_V3),
        "action_order": list(ACTION_ORDER_V3),
        "architecture": {
            "type": "separate_actor_critic_mlp",
            "actor": [
                len(OBSERVATION_ORDER_V3),
                *config["network"]["actor"],
                len(ACTION_ORDER_V3),
            ],
            "critic": [len(OBSERVATION_ORDER_V3), *config["network"]["critic"], 1],
            "activation": config["network"]["activation"],
            "trainable_parameters": parameter_count,
        },
        "training": {
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "library_version": stable_baselines3.__version__,
            "device": "cpu",
            "environment": "CityRecoveryEnv-v3",
            "completed_transitions": int(model.num_timesteps),
            "seed": int(config["policy_seed"]),
            "simulator_lanes": int(config["simulator_lanes"]),
            "scenario_unit_count": len(training_scenarios()),
            "curriculum": curriculum,
            "elapsed_seconds": elapsed,
        },
        "export": {
            "format": "ONNX",
            "opset": 17,
            "input_name": "observation",
            "output_name": "action",
            "runtime_provider": "CPUExecutionProvider",
            "onnx_sha256": sha256_file(output_onnx_path),
        },
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "solved_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "source_identity": source,
    }
    write_json(output_metadata_path, metadata)
    ledger_lines = (
        [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
        if ledger_path.exists()
        else []
    )
    terminal_checkpoint = {
        "path": str(release_sb3_path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": output_sb3_path.stat().st_size,
        "sha256": sha256_file(output_sb3_path),
    }
    receipt_base = {
        "schema_version": 1,
        "status": (
            "completed"
            if execution_class == "sealed_authorized_campaign"
            else "developmental_unsealed_completed_nonauthorizing"
        ),
        "execution_class": execution_class,
        "governance": governance,
        "campaign_provenance": campaign_provenance,
        "campaign_provenance_sha256": canonical_hash(campaign_provenance),
        "runtime": runtime_provenance,
        "production_stage_contract": (
            config["production_stages"]
            if execution_class == "sealed_authorized_campaign"
            else None
        ),
        "completed_stage_count": (
            len(ledger_lines)
            if execution_class == "sealed_authorized_campaign"
            else None
        ),
        "invocation_resumed_from_stage": resumed_from_stage,
        "config": config,
        "source_identity": source,
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "solved_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "completed_transitions": int(model.num_timesteps),
        "trainable_parameters": parameter_count,
        "curriculum": curriculum,
        "elapsed_seconds": elapsed,
        "checkpoint_ledger_sha256": sha256_file(ledger_path)
        if ledger_path.exists()
        else None,
        "checkpoint_ledger_record_count": len(ledger_lines),
        "checkpoint_ledger_terminal_record_sha256": (
            json.loads(ledger_lines[-1])["record_sha256"] if ledger_lines else None
        ),
        "terminal_checkpoint": terminal_checkpoint,
        "artifacts": {
            "training_onnx": {
                "path": str(release_onnx_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(output_onnx_path),
            },
            "metadata": {
                "path": str(release_metadata_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(output_metadata_path),
            },
            "manifest": {
                "path": str(release_manifest_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(output_manifest_path),
            },
            "parity": {
                "path": str(release_parity_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(output_parity_path),
            },
        },
        "manifest_sha256": sha256_file(output_manifest_path),
        "parity_sha256": sha256_file(output_parity_path),
    }
    receipt = {
        **receipt_base,
        "receipt_semantic_sha256": canonical_hash(receipt_base),
    }
    write_json(output_receipt_path, receipt)
    if execution_class == "sealed_authorized_campaign":
        _publish_sealed_release(
            run_output_root,
            campaign_identity,
            campaign_provenance,
            config,
        )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
