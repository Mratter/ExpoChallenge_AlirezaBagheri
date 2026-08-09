"""Select one V3 PPO checkpoint using the frozen development suite only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any

for _parent_thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_parent_thread_variable] = "12"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.scenarios_v3 import (  # noqa: E402
    DEVELOPMENT_FAMILIES_V3,
    DEVELOPMENT_SEEDS_V3,
)
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_ORDER_V3,
    ENGINE_V3_SPEC_SHA256,
    OBSERVATION_ORDER_V3,
    SOLVED_DEFINITION_V3_SHA256,
    rollout_candidate_v3,
)
from scripts.v3_protocol import (  # noqa: E402
    PROTOCOL_PATH,
    SOURCE_SEAL_PATH,
    TRAINING_AUTHORIZATION_PATH,
    canonical_hash,
    scientific_source_identity,
    verify_documents,
    verify_training_authorization,
)

CONFIG_PATH = ROOT / "training" / "v3" / "config.json"
TRAINING_RECEIPT_PATH = ROOT / "training" / "v3" / "training-receipt.json"
SELECTION_RECEIPT_PATH = ROOT / "training" / "v3" / "checkpoint-selection-receipt.json"
LEDGER_PATH = ROOT / "internal" / "training_runs" / "v3" / "checkpoint-ledger.jsonl"
TERMINAL_CHECKPOINT_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.zip"
TRAINING_MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.v3.json"
TRAINING_PARITY_PATH = ROOT / "training" / "v3" / "onnx-parity.json"
TRAINING_METADATA_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.metadata.json"
SELECTED_ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.selected.onnx"
SELECTED_PARITY_PATH = ROOT / "training" / "v3" / "selected-onnx-parity.json"
SELECTED_MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.v3.selected.json"
SELECTION_LOCK_PATH = ROOT / "internal" / "training_runs" / "v3" / "selection.lock"
TRAINING_USE_PATH = ROOT / "training" / "v3" / "training-use-receipt.json"
TRAINING_TERMINAL_PATH = ROOT / "training" / "v3" / "training-terminal.json"


class SelectionError(RuntimeError):
    """Raised when checkpoint selection cannot be reproduced safely."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_parent(path: Path) -> None:
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


def _durable_write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SelectionError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise SelectionError(f"write-once artifact appeared concurrently: {path}")
    os.replace(temporary, path)
    _fsync_parent(path)


def _publish_exact_file(source: Path, destination: Path) -> None:
    """Atomically publish one immutable file while retaining staging for resume."""

    if not source.is_file():
        raise SelectionError(f"selected deployment staging file is missing: {source}")
    source_sha256 = _sha256_file(source)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_sha256:
            raise SelectionError(
                f"selected deployment publication conflicts with staging: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".publish.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target:
            shutil.copyfileobj(source_handle, target)
            target.flush()
            os.fsync(target.fileno())
        if _sha256_file(temporary) != source_sha256:
            raise SelectionError(
                f"selected deployment copy failed verification: {destination}"
            )
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or _sha256_file(destination) != source_sha256:
            raise SelectionError(
                f"selected deployment appeared concurrently with drift: {destination}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_parent(destination)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise SelectionError("another checkpoint selector owns the lock") from exc
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class _OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.policy._predict(observation, deterministic=True)  # type: ignore[attr-defined]
        return torch.clamp(action, -1.0, 1.0)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SelectionError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"{label} root must be an object")
    return value


def _validate_training_evidence(
    verification: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_receipt = _read_object(TRAINING_RECEIPT_PATH, "V3 training receipt")
    config = _read_object(CONFIG_PATH, "V3 training config")
    receipt_unsigned = {
        key: value
        for key, value in training_receipt.items()
        if key != "receipt_semantic_sha256"
    }
    if training_receipt.get("receipt_semantic_sha256") != canonical_hash(
        receipt_unsigned
    ):
        raise SelectionError("V3 training receipt semantic hash drifted")
    if training_receipt.get("schema_version") != 1:
        raise SelectionError("V3 training receipt schema drifted")
    if training_receipt.get("status") != "completed":
        raise SelectionError("V3 training campaign is not complete")
    if training_receipt.get("execution_class") != "sealed_authorized_campaign":
        raise SelectionError("developmental V3 artifacts are not selection evidence")
    if training_receipt.get("config") != config:
        raise SelectionError("V3 training receipt config drifted")
    completed_transitions = int(training_receipt.get("completed_transitions", -1))
    if completed_transitions != int(config["total_transitions"]):
        raise SelectionError("V3 training campaign did not end on its exact budget")
    if training_receipt.get("source_identity") != scientific_source_identity():
        raise SelectionError("V3 training receipt source identity drifted")
    if training_receipt.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256:
        raise SelectionError("V3 training receipt engine binding drifted")
    if training_receipt.get("solved_definition_sha256") != SOLVED_DEFINITION_V3_SHA256:
        raise SelectionError("V3 training receipt outcome binding drifted")
    expected_governance = {
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "source_seal_sha256": _sha256_file(SOURCE_SEAL_PATH),
        "training_authorization_sha256": _sha256_file(TRAINING_AUTHORIZATION_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
    }
    if training_receipt.get("governance") != expected_governance:
        raise SelectionError("training receipt governance binding drifted")
    expected_provenance = {
        "schema_version": 1,
        "execution_class": "sealed_authorized_campaign",
        "governance": expected_governance,
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "registered_config_sha256": _sha256_file(CONFIG_PATH),
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
        "runtime": {
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "onnxruntime_available_providers": ort.get_available_providers(),
            "thread_settings": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "12"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "12"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "12"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "12"),
                "torch_intra_op_threads": int(
                    config["vectorization"]["parent_torch_intra_op_threads"]
                ),
                "torch_inter_op_threads": int(
                    config["vectorization"]["parent_torch_inter_op_threads"]
                ),
                "worker_blas_threads": int(
                    config["vectorization"]["worker_blas_threads"]
                ),
            },
        },
    }
    if training_receipt.get("campaign_provenance") != expected_provenance:
        raise SelectionError("training campaign provenance drifted")
    if training_receipt.get("runtime") != expected_provenance["runtime"]:
        raise SelectionError("training receipt runtime provenance drifted")
    if (
        training_receipt.get("production_stage_contract") != config["production_stages"]
        or training_receipt.get("completed_stage_count")
        != config["production_stages"]["count"]
        or not isinstance(training_receipt.get("invocation_resumed_from_stage"), int)
        or not 0
        <= training_receipt["invocation_resumed_from_stage"]
        <= config["production_stages"]["count"]
    ):
        raise SelectionError("training receipt stage-resume evidence drifted")
    if training_receipt.get("campaign_provenance_sha256") != canonical_hash(
        expected_provenance
    ):
        raise SelectionError("training campaign provenance hash drifted")
    use_receipt = _read_object(TRAINING_USE_PATH, "V3 training use receipt")
    expected_use_receipt = {
        "schema_version": 1,
        "status": "sealed_training_campaign_claimed",
        "campaign_identity": canonical_hash(expected_provenance),
        "campaign_provenance": expected_provenance,
        "campaign_provenance_sha256": canonical_hash(expected_provenance),
        "single_use": True,
        "stage_resume_only": True,
        "behavior_cloning_resume": "deterministic_replay_before_first_stage_only",
    }
    if use_receipt != expected_use_receipt:
        raise SelectionError("training use receipt drifted")

    terminal = training_receipt.get("terminal_checkpoint")
    if not isinstance(terminal, dict):
        raise SelectionError("terminal checkpoint receipt binding is missing")
    if terminal.get("path") != str(TERMINAL_CHECKPOINT_PATH.relative_to(ROOT)).replace(
        "\\", "/"
    ):
        raise SelectionError("terminal checkpoint path drifted")
    if (
        not TERMINAL_CHECKPOINT_PATH.is_file()
        or terminal.get("sha256") != _sha256_file(TERMINAL_CHECKPOINT_PATH)
        or terminal.get("bytes") != TERMINAL_CHECKPOINT_PATH.stat().st_size
    ):
        raise SelectionError("terminal checkpoint receipt bytes drifted")

    expected_artifact_paths = {
        "training_onnx": ROOT / "artifacts" / "city_recovery_ppo.v3.onnx",
        "metadata": TRAINING_METADATA_PATH,
        "manifest": TRAINING_MANIFEST_PATH,
        "parity": TRAINING_PARITY_PATH,
    }
    artifacts = training_receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SelectionError("training artifact receipt bindings are missing")
    for artifact_id, path in expected_artifact_paths.items():
        binding = artifacts.get(artifact_id)
        if not isinstance(binding, dict):
            raise SelectionError(f"training {artifact_id} binding is missing")
        if binding.get("path") != str(path.relative_to(ROOT)).replace("\\", "/"):
            raise SelectionError(f"training {artifact_id} path drifted")
        if not path.is_file() or binding.get("sha256") != _sha256_file(path):
            raise SelectionError(f"training {artifact_id} bytes drifted")
    if training_receipt.get("manifest_sha256") != _sha256_file(TRAINING_MANIFEST_PATH):
        raise SelectionError("training manifest hash drifted")
    if training_receipt.get("parity_sha256") != _sha256_file(TRAINING_PARITY_PATH):
        raise SelectionError("training parity hash drifted")
    training_parity = _read_object(TRAINING_PARITY_PATH, "V3 training ONNX parity")
    if training_parity.get("passed") is not True:
        raise SelectionError("training ONNX parity did not pass")
    if training_parity.get("onnx_sha256") != artifacts["training_onnx"]["sha256"]:
        raise SelectionError("training ONNX parity binding drifted")
    manifest = _read_object(TRAINING_MANIFEST_PATH, "V3 training manifest")
    model_binding = manifest.get("model")
    if not isinstance(model_binding, dict):
        raise SelectionError("training manifest model binding is missing")
    if (
        model_binding.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or model_binding.get("solved_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or model_binding.get("observation_count") != len(OBSERVATION_ORDER_V3)
        or model_binding.get("action_count") != len(ACTION_ORDER_V3)
        or model_binding.get("completed_training_transitions") != completed_transitions
        or model_binding.get("trainable_parameters")
        != training_receipt.get("trainable_parameters")
        or model_binding.get("trainable_parameters")
        != config.get("expected_trainable_parameters")
    ):
        raise SelectionError("training manifest scientific binding drifted")
    expected_manifest_files = [
        {
            "id": "onnx",
            "path": artifacts["training_onnx"]["path"],
            "bytes": expected_artifact_paths["training_onnx"].stat().st_size,
            "sha256": artifacts["training_onnx"]["sha256"],
        },
        {
            "id": "sb3_checkpoint",
            "path": terminal["path"],
            "bytes": terminal["bytes"],
            "sha256": terminal["sha256"],
        },
    ]
    if manifest.get("files") != expected_manifest_files:
        raise SelectionError("training manifest release-file binding drifted")
    metadata = _read_object(TRAINING_METADATA_PATH, "V3 training metadata")
    if (
        metadata.get("source_identity") != scientific_source_identity()
        or metadata.get("observation_order") != list(OBSERVATION_ORDER_V3)
        or metadata.get("action_order") != list(ACTION_ORDER_V3)
        or metadata.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or metadata.get("solved_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or metadata.get("training", {}).get("completed_transitions")
        != completed_transitions
        or metadata.get("architecture", {}).get("trainable_parameters")
        != config.get("expected_trainable_parameters")
        or metadata.get("export", {}).get("onnx_sha256")
        != artifacts["training_onnx"]["sha256"]
    ):
        raise SelectionError("training metadata scientific binding drifted")
    terminal = _read_object(TRAINING_TERMINAL_PATH, "V3 training terminal marker")
    terminal_unsigned = {
        key: value
        for key, value in terminal.items()
        if key != "terminal_semantic_sha256"
    }
    expected_release_files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in (
            TERMINAL_CHECKPOINT_PATH,
            ROOT / "artifacts" / "city_recovery_ppo.v3.onnx",
            TRAINING_METADATA_PATH,
            TRAINING_MANIFEST_PATH,
            TRAINING_PARITY_PATH,
            TRAINING_RECEIPT_PATH,
        )
    }
    if (
        terminal.get("terminal_semantic_sha256") != canonical_hash(terminal_unsigned)
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "sealed_training_release_complete"
        or terminal.get("campaign_identity") != canonical_hash(expected_provenance)
        or terminal.get("campaign_provenance_sha256")
        != canonical_hash(expected_provenance)
        or terminal.get("configured_transitions") != config["total_transitions"]
        or terminal.get("actual_completed_transitions") != config["total_transitions"]
        or terminal.get("completed_stage_count") != config["production_stages"]["count"]
        or terminal.get("training_use_receipt_sha256")
        != _sha256_file(TRAINING_USE_PATH)
        or terminal.get("training_receipt_sha256")
        != _sha256_file(TRAINING_RECEIPT_PATH)
        or terminal.get("checkpoint_ledger_sha256") != _sha256_file(LEDGER_PATH)
        or terminal.get("checkpoint_ledger_terminal_record_sha256")
        != training_receipt.get("checkpoint_ledger_terminal_record_sha256")
        or terminal.get("release_files") != expected_release_files
    ):
        raise SelectionError("sealed training terminal marker drifted")
    return training_receipt, config, expected_provenance


def _checkpoint_candidates(
    training_receipt: dict[str, Any],
    config: dict[str, Any],
    expected_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    if not LEDGER_PATH.is_file():
        raise SelectionError("sealed V3 checkpoint ledger is missing")
    if training_receipt.get("checkpoint_ledger_sha256") != _sha256_file(LEDGER_PATH):
        raise SelectionError("checkpoint ledger does not match the training receipt")
    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise SelectionError("checkpoint ledger is empty or contains blank records")
    interval = int(config["checkpoint_interval_transitions"])
    thresholds = list(range(interval, int(config["total_transitions"]) + 1, interval))
    if len(lines) != len(thresholds):
        raise SelectionError("checkpoint ledger record count drifted")
    if training_receipt.get("checkpoint_ledger_record_count") != len(lines):
        raise SelectionError("training receipt ledger count drifted")
    for line_number, (line, registered_threshold) in enumerate(
        zip(lines, thresholds), start=1
    ):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise SelectionError(
                f"checkpoint ledger line {line_number} is invalid"
            ) from exc
        if record.get("previous_record_sha256") != previous_hash:
            raise SelectionError("checkpoint ledger hash chain is broken")
        claimed_hash = record.get("record_sha256")
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if claimed_hash != canonical_hash(unsigned):
            raise SelectionError("checkpoint ledger record hash drifted")
        if (
            record.get("schema_version") != 1
            or record.get("record_type") != "v3_training_stage_checkpoint"
            or record.get("stage_number") != line_number
            or record.get("stage_start_transitions") != (line_number - 1) * interval
            or record.get("stage_transitions") != interval
            or record.get("campaign_provenance") != expected_provenance
            or record.get("campaign_provenance_sha256")
            != canonical_hash(expected_provenance)
            or record.get("stage_seed")
            != int(config["policy_seed"]) + (line_number - 1) * 10_007
            or record.get("fresh_environment_reset") is not True
        ):
            raise SelectionError("checkpoint ledger provenance drifted")
        completed_transitions = int(record["num_timesteps"])
        expected_relative = (
            "internal/training_runs/v3/checkpoints/"
            f"stage-{line_number:02d}-transition-{completed_transitions:09d}.zip"
        )
        if record.get("checkpoint") != expected_relative:
            raise SelectionError("checkpoint ledger path is not canonical")
        checkpoint = ROOT / expected_relative
        if not checkpoint.is_file() or _sha256_file(checkpoint) != record.get("sha256"):
            raise SelectionError("checkpoint ledger file binding drifted")
        if completed_transitions != registered_threshold:
            raise SelectionError("checkpoint ledger contains an unregistered interval")
        candidates.append(
            {
                "path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "sha256": record["sha256"],
                "completed_transitions": completed_transitions,
                "source": "checkpoint_ledger",
            }
        )
        previous_hash = str(claimed_hash)
    if (
        training_receipt.get("checkpoint_ledger_terminal_record_sha256")
        != previous_hash
    ):
        raise SelectionError("training receipt ledger terminal hash drifted")

    if not TERMINAL_CHECKPOINT_PATH.is_file():
        raise SelectionError("terminal V3 checkpoint is missing")
    candidates.append(
        {
            "path": str(TERMINAL_CHECKPOINT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(TERMINAL_CHECKPOINT_PATH),
            "completed_transitions": int(training_receipt["completed_transitions"]),
            "source": "terminal_checkpoint",
        }
    )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        unique[(candidate["sha256"], candidate["completed_transitions"])] = candidate
    return sorted(
        unique.values(),
        key=lambda item: (item["completed_transitions"], item["sha256"]),
    )


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    checkpoint = ROOT / candidate["path"]
    model = PPO.load(checkpoint, device="cpu")

    def action_provider(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    replay_mismatches = 0
    for family in DEVELOPMENT_FAMILIES_V3:
        for seed in DEVELOPMENT_SEEDS_V3:
            scenario = family.build(seed)
            tape_seed = family.tape_seed(seed)
            first = rollout_candidate_v3(scenario, tape_seed, action_provider)
            second = rollout_candidate_v3(scenario, tape_seed, action_provider)
            if first["trajectory_sha256"] != second["trajectory_sha256"]:
                replay_mismatches += 1
            outcome = first["absolute_outcome"]
            rows.append(
                {
                    "family_id": family.id,
                    "case_seed": seed,
                    "tape_seed": tape_seed,
                    "solved": bool(outcome["solved"]),
                    "resilience_auc": first["rauc"],
                    "critical_service_days": first["critical_service_days"],
                    "hard_violation_count": first["hard_violation_count"],
                    "max_conservation_residual": first[
                        "max_logistics_conservation_residual"
                    ],
                    "reason_codes": outcome["reason_codes"],
                    "trajectory_sha256": first["trajectory_sha256"],
                }
            )
    if len(rows) != 40:
        raise SelectionError("development checkpoint evaluation is incomplete")
    maximum_residual = max(row["max_conservation_residual"] for row in rows)
    hard_violations = sum(row["hard_violation_count"] for row in rows)
    if replay_mismatches or hard_violations or maximum_residual > 1e-6:
        raise SelectionError(
            "development checkpoint violated the frozen safety contract"
        )
    return {
        **candidate,
        "development_case_count": len(rows),
        "candidate_solved_count": sum(row["solved"] for row in rows),
        "candidate_mean_resilience_auc": round(
            fmean(row["resilience_auc"] for row in rows), 10
        ),
        "candidate_critical_service_days": sum(
            row["critical_service_days"] for row in rows
        ),
        "hard_violation_count": hard_violations,
        "max_conservation_residual": maximum_residual,
        "deterministic_replay_mismatches": replay_mismatches,
        "rows_sha256": canonical_hash(rows),
    }


def _onnx_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        path.read_bytes(), options, providers=["CPUExecutionProvider"]
    )


def _onnx_interface_receipt(
    path: Path, session: ort.InferenceSession
) -> dict[str, Any]:
    """Fail closed unless the selected export is the registered CPU interface."""

    model = onnx.load_model_from_string(path.read_bytes())
    opsets = [
        {"domain": item.domain or "", "version": int(item.version)}
        for item in model.opset_import
    ]
    graph_inputs = list(model.graph.input)
    graph_outputs = list(model.graph.output)

    def graph_shape(value_info: Any) -> list[str | int | None]:
        dimensions: list[str | int | None] = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(str(dimension.dim_param))
            else:
                dimensions.append(None)
        return dimensions

    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    expected_input_shape = ["batch", len(OBSERVATION_ORDER_V3)]
    expected_output_shape = ["batch", len(ACTION_ORDER_V3)]
    if (
        session.get_providers() != ["CPUExecutionProvider"]
        or opsets != [{"domain": "", "version": 17}]
        or len(graph_inputs) != 1
        or len(graph_outputs) != 1
        or graph_inputs[0].name != "observation"
        or graph_outputs[0].name != "action"
        or graph_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or graph_outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or graph_shape(graph_inputs[0]) != expected_input_shape
        or graph_shape(graph_outputs[0]) != expected_output_shape
        or len(runtime_inputs) != 1
        or len(runtime_outputs) != 1
        or runtime_inputs[0].name != "observation"
        or runtime_outputs[0].name != "action"
        or runtime_inputs[0].type != "tensor(float)"
        or runtime_outputs[0].type != "tensor(float)"
        or list(runtime_inputs[0].shape) != expected_input_shape
        or list(runtime_outputs[0].shape) != expected_output_shape
    ):
        raise SelectionError("selected ONNX provider or interface contract drifted")
    return {
        "execution_providers": ["CPUExecutionProvider"],
        "opset": 17,
        "input_name": "observation",
        "input_type": "tensor(float)",
        "input_shape": expected_input_shape,
        "output_name": "action",
        "output_type": "tensor(float)",
        "output_shape": expected_output_shape,
        "observation_count": len(OBSERVATION_ORDER_V3),
        "action_count": len(ACTION_ORDER_V3),
    }


def _onnx_action(session: ort.InferenceSession, observation: np.ndarray) -> np.ndarray:
    output = session.run(
        ["action"],
        {"observation": np.asarray(observation, dtype=np.float32).reshape(1, -1)},
    )[0]
    action = np.asarray(output, dtype=np.float64)
    if action.shape != (1, len(ACTION_ORDER_V3)) or not np.all(np.isfinite(action)):
        raise SelectionError("selected ONNX produced an invalid action tensor")
    return action[0]


def _export_selected_onnx(model: PPO, output_path: Path) -> None:
    if output_path.exists():
        raise SelectionError("selected ONNX stage is write-once")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise SelectionError("stale selected ONNX temporary file exists")
    wrapper = _OnnxablePolicy(model.policy).eval()
    dummy = torch.zeros((1, len(OBSERVATION_ORDER_V3)), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        temporary,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    # The legacy Torch ONNX exporter can leave the non-batch action dimension
    # symbolically named (for example ``Clipaction_dim_1``), even though ONNX
    # Runtime resolves it to the fixed 22-action contract.  Pin that graph
    # value-info dimension before validation/publication so the deployable
    # artifact itself—not only the runtime inference result—advertises 73→22.
    exported = onnx.load_model_from_string(temporary.read_bytes())
    graph_inputs = list(exported.graph.input)
    graph_outputs = list(exported.graph.output)
    if (
        len(graph_inputs) != 1
        or len(graph_outputs) != 1
        or graph_inputs[0].name != "observation"
        or graph_outputs[0].name != "action"
        or len(graph_inputs[0].type.tensor_type.shape.dim) != 2
        or len(graph_outputs[0].type.tensor_type.shape.dim) != 2
    ):
        temporary.unlink(missing_ok=True)
        raise SelectionError("selected ONNX export structure drifted")
    output_dimension = graph_outputs[0].type.tensor_type.shape.dim[1]
    output_dimension.ClearField("dim_param")
    output_dimension.dim_value = len(ACTION_ORDER_V3)
    shape_pinned = temporary.with_suffix(temporary.suffix + ".shape-pinned")
    if shape_pinned.exists():
        temporary.unlink(missing_ok=True)
        raise SelectionError("stale shape-pinned ONNX temporary file exists")
    onnx.save_model(exported, shape_pinned)
    with shape_pinned.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(shape_pinned, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    if output_path.exists():
        temporary.unlink(missing_ok=True)
        raise SelectionError("selected ONNX appeared concurrently")
    os.replace(temporary, output_path)
    _fsync_parent(output_path)


def _selected_onnx_parity(
    model: PPO,
    session: ort.InferenceSession,
    selected: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    interface = _onnx_interface_receipt(onnx_path, session)
    rows: list[dict[str, Any]] = []
    maximum_action_error = 0.0
    action_samples = 0
    outcome_mismatches = 0
    deterministic_replay_mismatches = 0
    maximum_rauc_error = 0.0
    hard_violations = 0
    maximum_residual = 0.0
    for family in DEVELOPMENT_FAMILIES_V3:
        for seed in DEVELOPMENT_SEEDS_V3:
            scenario = family.build(seed)
            tape_seed = family.tape_seed(seed)

            def reference_provider(observation: np.ndarray) -> np.ndarray:
                nonlocal maximum_action_error, action_samples
                action, _ = model.predict(observation, deterministic=True)
                reference = np.asarray(action, dtype=np.float64)
                deployed = _onnx_action(session, observation)
                if reference.shape != (len(ACTION_ORDER_V3),):
                    raise SelectionError("selected SB3 checkpoint action shape drifted")
                maximum_action_error = max(
                    maximum_action_error,
                    float(np.max(np.abs(reference - deployed))),
                )
                action_samples += 1
                return reference

            def deployed_provider(observation: np.ndarray) -> np.ndarray:
                return _onnx_action(session, observation)

            reference = rollout_candidate_v3(scenario, tape_seed, reference_provider)
            deployed_first = rollout_candidate_v3(
                scenario, tape_seed, deployed_provider
            )
            deployed_second = rollout_candidate_v3(
                scenario, tape_seed, deployed_provider
            )
            reference_outcome = reference["absolute_outcome"]
            deployed_outcome = deployed_first["absolute_outcome"]
            outcome_matches = (
                reference_outcome["solved"] == deployed_outcome["solved"]
                and reference_outcome["status"] == deployed_outcome["status"]
                and reference_outcome["reason_codes"]
                == deployed_outcome["reason_codes"]
                and reference["critical_service_days"]
                == deployed_first["critical_service_days"]
            )
            outcome_mismatches += int(not outcome_matches)
            deterministic_replay_mismatches += int(
                deployed_first["trajectory_sha256"]
                != deployed_second["trajectory_sha256"]
            )
            rauc_error = abs(float(reference["rauc"]) - float(deployed_first["rauc"]))
            maximum_rauc_error = max(maximum_rauc_error, rauc_error)
            hard_violations += int(deployed_first["hard_violation_count"])
            maximum_residual = max(
                maximum_residual,
                float(deployed_first["max_logistics_conservation_residual"]),
            )
            rows.append(
                {
                    "family_id": family.id,
                    "case_seed": seed,
                    "tape_seed": tape_seed,
                    "outcome_matches": outcome_matches,
                    "rauc_absolute_error": rauc_error,
                    "deployment_trajectory_sha256": deployed_first["trajectory_sha256"],
                }
            )
    passed = (
        len(rows) == 40
        and action_samples > 0
        and maximum_action_error <= 1e-5
        and maximum_rauc_error <= 1e-6
        and outcome_mismatches == 0
        and deterministic_replay_mismatches == 0
        and hard_violations == 0
        and maximum_residual <= 1e-6
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint_sha256": selected["sha256"],
        "onnx_sha256": _sha256_file(onnx_path),
        "interface": interface,
        "development_case_count": len(rows),
        "action_sample_count": action_samples,
        "maximum_action_absolute_error": maximum_action_error,
        "maximum_rauc_absolute_error": maximum_rauc_error,
        "outcome_mismatches": outcome_mismatches,
        "deterministic_replay_mismatches": deterministic_replay_mismatches,
        "hard_violation_count": hard_violations,
        "maximum_conservation_residual": maximum_residual,
        "rows_sha256": canonical_hash(rows),
    }


def _materialize_selected_deployment(
    selected: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    stage_root = (
        ROOT
        / "internal"
        / "training_runs"
        / "v3"
        / "selection-staging"
        / selected["sha256"]
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    claim_path = stage_root / "selection-claim.json"
    claim_base = {
        "schema_version": 1,
        "status": "selected_deployment_staging_claimed",
        "selected_checkpoint": selected,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "source_seal_sha256": _sha256_file(SOURCE_SEAL_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "observation_order_sha256": verification["observation_order_sha256"],
        "action_order_sha256": verification["action_order_sha256"],
        "destinations": {
            "onnx": str(SELECTED_ONNX_PATH.relative_to(ROOT)).replace("\\", "/"),
            "parity": str(SELECTED_PARITY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "manifest": str(SELECTED_MANIFEST_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
        },
    }
    claim = {**claim_base, "claim_semantic_sha256": canonical_hash(claim_base)}
    if claim_path.exists():
        if _read_object(claim_path, "selected deployment staging claim") != claim:
            raise SelectionError("selected deployment staging claim identity drifted")
    else:
        unexpected = [
            item
            for item in stage_root.iterdir()
            if not item.name.startswith(claim_path.name + ".tmp.")
        ]
        if unexpected:
            raise SelectionError(
                "unclaimed selected deployment staging state already exists"
            )
        _durable_write_new_json(claim_path, claim)

    staged_onnx = stage_root / SELECTED_ONNX_PATH.name
    staged_parity = stage_root / SELECTED_PARITY_PATH.name
    staged_manifest = stage_root / SELECTED_MANIFEST_PATH.name
    stage_receipt_path = stage_root / "selected-deployment-stage-receipt.json"
    checkpoint = ROOT / selected["path"]
    model = PPO.load(checkpoint, device="cpu")
    recomputed_root = Path(
        tempfile.mkdtemp(prefix=".recomputed-", dir=str(stage_root))
    )
    try:
        expected_onnx = recomputed_root / SELECTED_ONNX_PATH.name
        expected_parity = recomputed_root / SELECTED_PARITY_PATH.name
        expected_manifest = recomputed_root / SELECTED_MANIFEST_PATH.name
        _export_selected_onnx(model, expected_onnx)
        session = _onnx_session(expected_onnx)
        parity = _selected_onnx_parity(model, session, selected, expected_onnx)
        if parity["passed"] is not True:
            raise SelectionError(f"selected ONNX parity failed: {parity}")
        _durable_write_new_json(expected_parity, parity)
        manifest = {
            "schema_version": 1,
            "status": "selected_deployable",
            "policy": {
                "id": "city-recovery-sb3-ppo-v3-selected",
                "algorithm": "PPO",
                "observation_count": len(OBSERVATION_ORDER_V3),
                "action_count": len(ACTION_ORDER_V3),
                "completed_transitions": selected["completed_transitions"],
            },
            "bindings": {
                "protocol_sha256": _sha256_file(PROTOCOL_PATH),
                "source_seal_sha256": _sha256_file(SOURCE_SEAL_PATH),
                "scientific_source_sha256": verification[
                    "scientific_source_sha256"
                ],
                "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
                "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
                "observation_order_sha256": verification[
                    "observation_order_sha256"
                ],
                "action_order_sha256": verification["action_order_sha256"],
                "selected_checkpoint_sha256": selected["sha256"],
            },
            "files": {
                "onnx": {
                    "path": str(SELECTED_ONNX_PATH.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "bytes": expected_onnx.stat().st_size,
                    "sha256": _sha256_file(expected_onnx),
                    "input_name": "observation",
                    "output_name": "action",
                    "opset": 17,
                    "execution_providers": ["CPUExecutionProvider"],
                    "input_type": "tensor(float)",
                    "input_shape": ["batch", len(OBSERVATION_ORDER_V3)],
                    "output_type": "tensor(float)",
                    "output_shape": ["batch", len(ACTION_ORDER_V3)],
                },
                "parity": {
                    "path": str(SELECTED_PARITY_PATH.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "sha256": _sha256_file(expected_parity),
                },
            },
        }
        _durable_write_new_json(expected_manifest, manifest)
        expected_artifacts = (
            (expected_onnx, staged_onnx, SELECTED_ONNX_PATH),
            (expected_parity, staged_parity, SELECTED_PARITY_PATH),
            (expected_manifest, staged_manifest, SELECTED_MANIFEST_PATH),
        )
        for expected, staged, destination in expected_artifacts:
            expected_sha256 = _sha256_file(expected)
            for existing in (staged, destination):
                if existing.exists() and (
                    not existing.is_file()
                    or _sha256_file(existing) != expected_sha256
                ):
                    raise SelectionError(
                        f"selected deployment recovery bytes drifted: {existing}"
                    )
            if not staged.exists():
                _publish_exact_file(expected, staged)

        stage_receipt_base = {
            "schema_version": 1,
            "status": "selected_deployment_staging_complete",
            "claim_sha256": _sha256_file(claim_path),
            "selected_checkpoint_sha256": selected["sha256"],
            "artifacts": {
                artifact_id: {
                    "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": staged.stat().st_size,
                    "sha256": _sha256_file(staged),
                }
                for artifact_id, staged, destination in (
                    ("onnx", staged_onnx, SELECTED_ONNX_PATH),
                    ("parity", staged_parity, SELECTED_PARITY_PATH),
                    ("manifest", staged_manifest, SELECTED_MANIFEST_PATH),
                )
            },
        }
        stage_receipt = {
            **stage_receipt_base,
            "receipt_semantic_sha256": canonical_hash(stage_receipt_base),
        }
        if stage_receipt_path.exists():
            if (
                _read_object(
                    stage_receipt_path, "selected deployment staging receipt"
                )
                != stage_receipt
            ):
                raise SelectionError("selected deployment staging receipt drifted")
        else:
            _durable_write_new_json(stage_receipt_path, stage_receipt)

        for _, staged, destination in expected_artifacts:
            _publish_exact_file(staged, destination)
    finally:
        shutil.rmtree(recomputed_root, ignore_errors=True)
    return {
        "checkpoint": {
            "path": selected["path"],
            "sha256": selected["sha256"],
            "completed_transitions": selected["completed_transitions"],
        },
        "onnx": {
            "path": str(SELECTED_ONNX_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": SELECTED_ONNX_PATH.stat().st_size,
            "sha256": _sha256_file(SELECTED_ONNX_PATH),
        },
        "parity": {
            "path": str(SELECTED_PARITY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(SELECTED_PARITY_PATH),
        },
        "manifest": {
            "path": str(SELECTED_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(SELECTED_MANIFEST_PATH),
        },
    }


def build_selection_receipt() -> dict[str, Any]:
    verification = verify_documents(require_final_unused=True)
    verify_training_authorization()
    training_receipt, config, provenance = _validate_training_evidence(verification)
    candidates = _checkpoint_candidates(training_receipt, config, provenance)
    reports = [_evaluate_candidate(item) for item in candidates]
    selected = min(
        reports,
        key=lambda item: (
            -item["candidate_solved_count"],
            -item["candidate_mean_resilience_auc"],
            item["completed_transitions"],
            item["sha256"],
        ),
    )
    selected_deployment = _materialize_selected_deployment(selected, verification)
    receipt_base = {
        "schema_version": 1,
        "status": "checkpoint_selected_on_development_only",
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "source_seal_sha256": _sha256_file(SOURCE_SEAL_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "training_config_sha256": _sha256_file(CONFIG_PATH),
        "training_receipt_sha256": _sha256_file(TRAINING_RECEIPT_PATH),
        "training_use_receipt_sha256": _sha256_file(TRAINING_USE_PATH),
        "training_terminal_sha256": _sha256_file(TRAINING_TERMINAL_PATH),
        "campaign_provenance_sha256": training_receipt["campaign_provenance_sha256"],
        "checkpoint_ledger_sha256": (
            _sha256_file(LEDGER_PATH) if LEDGER_PATH.exists() else None
        ),
        "selection_dataset": "development",
        "development_case_count": 40,
        "ranking": [
            "candidate_solved_count:descending",
            "candidate_mean_resilience_auc:descending",
            "completed_transitions:ascending",
            "checkpoint_sha256:ascending",
        ],
        "candidate_reports": reports,
        "selected_checkpoint": selected,
        "selected_deployment": selected_deployment,
        "final_split_accessed": False,
    }
    return {**receipt_base, "receipt_semantic_sha256": canonical_hash(receipt_base)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-new", action="store_true")
    args = parser.parse_args()
    if not args.write_new:
        raise SelectionError(
            "explicit --write-new is required because selection materializes the pinned deployment"
        )
    with _exclusive_lock(SELECTION_LOCK_PATH):
        if SELECTION_RECEIPT_PATH.exists():
            raise SelectionError("checkpoint selection receipt is write-once")
        receipt = build_selection_receipt()
        _durable_write_new_json(SELECTION_RECEIPT_PATH, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
