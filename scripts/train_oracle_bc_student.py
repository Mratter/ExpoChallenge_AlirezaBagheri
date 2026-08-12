#!/usr/bin/env python3
"""Fit a causal oracle-label BC student with a matched public-rule control.

The student receives exactly the 73 public observations emitted by the
environment and imitates 22-action targets produced by the privileged oracle.
The full future tape is never an input.  The last four training seeds in every
family are a preregistered trajectory-level holdout.  Two actors with identical
initialization, normalization, architecture, minibatches, and training schedule
are fit on the other trajectories: one to oracle actions and one to the
original public preparedness rule.  Only the oracle-label actor is persisted
and evaluated once on development.  This tool never imports or accesses the
final split and never runs DAgger or PPO updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Keep numerical execution deterministic and avoid nested native thread pools.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    VecNormalize,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    ACTION_ORDER,
    ACTION_SIZE,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    CityRecoveryEnv,
)
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.city.planners import preparedness_teacher_action  # noqa: E402
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    function_source_sha256,
    load_json_object,
)
from scripts.train_policy import (  # noqa: E402
    build_model,
    evaluate_development_frozen,
    normalize_observations,
    reset_policy_seed,
    rms_digest,
    rms_state,
)
from scripts.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    actor_state_sha256,
    apply_normalization_state,
    checkpoint_bundle_reference,
    load_checkpoint_bundle,
    persist_checkpoint_bundle,
)

TOOL_ID = "train_oracle_bc_student.py"
SCHEMA_VERSION = 1
DATASET_TOOL_ID = "run_training_oracle_trajectories.py"
TRAINING_CASE_COUNT = 192
HORIZON_DAYS = 30
DATASET_ROW_COUNT = TRAINING_CASE_COUNT * HORIZON_DAYS
DEVELOPMENT_CASE_COUNT = 200
FIT_SEEDS = tuple(range(810000, 810028))
HOLDOUT_SEEDS = tuple(range(810028, 810032))
FIT_CASE_COUNT = 168
HOLDOUT_CASE_COUNT = 24
FIT_ROW_COUNT = FIT_CASE_COUNT * HORIZON_DAYS
HOLDOUT_ROW_COUNT = HOLDOUT_CASE_COUNT * HORIZON_DAYS

# Preregistered before inspecting any oracle-labelled student result.  Seed
# 67017 is the already selected release seed, whose old BC initialization was
# 152/200 on the expanded development roster.
POLICY_SEED = 67017
BC_EPOCHS = 15
BC_BATCH_SIZE = 512
BC_LEARNING_RATE = 1e-3
MODEL_N_STEPS = 250
MODEL_BATCH_SIZE = 250
MODEL_LEARNING_RATE = 7.5e-5
MODEL_TARGET_KL = 0.02
MODEL_ENT_COEF = 0.003
OLD_BC_SOLVED_COUNT = 152
DEVELOPMENT_CATASTROPHIC_FLOOR = 140
HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR = 0.01
COMPLETED_NEGATIVE_GATE_EXIT_CODE = 4
OLD_BC_RECEIPT_SHA256 = (
    "37bc81cd677ae86458c207a99758c1c295b411906e7fe1ce2ea22d26bb22398f"
)

DEMONSTRATION_FIELDS = frozenset(
    {
        "row_id",
        "input_contract",
        "student_input_future_tape_visible",
        "teacher_target_uses_full_future_tape",
        "observation_dtype",
        "target_dtype",
        "observation_shape",
        "target_shape",
        "observations",
        "targets",
        "observations_sha256",
        "targets_sha256",
        "dataset_sha256",
    }
)
DATASET_SOURCE_PATHS = (
    "scripts/run_training_oracle_trajectories.py",
    "scripts/headroom.py",
    "scripts/train_policy.py",
    "scripts/training_artifacts.py",
    "backend/app/shared_evidence.py",
    "backend/app/city/environment.py",
    "backend/app/city/scenarios.py",
    "backend/app/city/outcome.py",
    "backend/app/city/physics.py",
    "backend/app/city/planners.py",
    "backend/app/city/optimizer.py",
)
EXPECTED_ORACLE_CONFIG = {
    "population": 512,
    "elite_fraction": 0.10,
    "min_iterations": 20,
    "max_iterations": 40,
    "patience": 6,
    "initial_std": 0.25,
    "std_floor": 0.03,
    "smoothing": 0.75,
}


class OracleBCError(RuntimeError):
    """Raised when the preregistered causal BC contract cannot be honored."""


@dataclass(frozen=True)
class OracleDataset:
    """The validated public observations and privileged action labels."""

    observations: np.ndarray
    oracle_targets: np.ndarray
    hand_rule_targets: np.ndarray
    row_ids: tuple[str, ...]
    fit_indices: np.ndarray
    holdout_indices: np.ndarray
    split_contract: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    contract_sha256: str
    dataset_index_sha256: str


@dataclass
class StudentFit:
    """In-memory BC student and frozen preprocessing state."""

    model: PPO
    normalizer: VecNormalize
    report: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _worktree_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise OracleBCError(f"refusing to overwrite existing evidence: {path}")
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_parent(path)
    except FileExistsError as exc:
        raise OracleBCError(f"refusing to overwrite existing evidence: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=OracleBCError)


def _require_external_root(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise OracleBCError(f"{label} must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise OracleBCError(f"{label} must be outside the repository")
    if resolved == Path(resolved.anchor):
        raise OracleBCError(f"{label} cannot be a filesystem root")
    return resolved


def validate_old_bc_anchor(
    path: Path,
    *,
    expected_sha256: str = OLD_BC_RECEIPT_SHA256,
) -> dict[str, Any]:
    """Validate the selected-seed 152/200 comparison anchor."""

    actual_sha256 = file_sha256(
        path,
        label="selected-seed old BC receipt",
        error_type=OracleBCError,
    )
    if actual_sha256 != expected_sha256:
        raise OracleBCError("selected-seed old BC receipt hash mismatch")
    receipt = _load_json(path, "selected-seed old BC receipt")
    try:
        seed = int(receipt["config"]["policy_seed"])
        development = receipt["development_curve"]["bc_initialization"]
        solved_count = int(development["solved_count"])
        case_count = int(development["case_count"])
        behavior_cloning = receipt["behavior_cloning"]
        dagger_iterations = int(behavior_cloning["iterations"])
        observation_count = int(behavior_cloning["observation_count"])
        dagger_beta_schedule = list(behavior_cloning["dagger_beta_schedule"])
        epochs_per_iteration = int(behavior_cloning["epochs_per_iteration"])
        batch_size = int(behavior_cloning["batch_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleBCError("selected-seed old BC evidence is incomplete") from exc
    if (
        receipt.get("tool") != "train_policy.py"
        or receipt.get("training_split") != "train"
        or receipt.get("final_split_used") is not False
        or seed != POLICY_SEED
        or solved_count != OLD_BC_SOLVED_COUNT
        or case_count != DEVELOPMENT_CASE_COUNT
        or behavior_cloning.get("teacher") != "preparedness_teacher_action"
        or behavior_cloning.get("training_split_only") is not True
        or behavior_cloning.get("observation_normalization") is not True
        or dagger_iterations != 4
        or dagger_beta_schedule != [1.0, 0.0, 0.0, 0.0]
        or observation_count != 23040
        or epochs_per_iteration != 15
        or batch_size != 512
    ):
        raise OracleBCError("selected-seed old BC evidence drifted")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "policy_seed": seed,
        "development_case_count": case_count,
        "development_solved_count": solved_count,
        "training_method": "behavior_cloning_with_dagger",
        "dagger_iterations": dagger_iterations,
        "dagger_beta_schedule": dagger_beta_schedule,
        "training_observation_count": observation_count,
        "epochs_per_iteration": epochs_per_iteration,
        "batch_size": batch_size,
    }


def _safe_shard_path(dataset_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise OracleBCError("dataset shard path is invalid")
    supplied = Path(relative)
    if supplied.is_absolute():
        raise OracleBCError("dataset shard path must be relative")
    resolved = (dataset_root / supplied).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise OracleBCError("dataset shard escapes the dataset root") from exc
    return resolved


def _validate_demonstration(
    value: Any,
    *,
    row_id: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(value, dict) or set(value) != DEMONSTRATION_FIELDS:
        raise OracleBCError(f"{row_id} demonstration fields drifted")
    if (
        value.get("row_id") != row_id
        or value.get("input_contract") != "73_public_causal_observations"
        or value.get("student_input_future_tape_visible") is not False
        or value.get("teacher_target_uses_full_future_tape") is not True
        or value.get("observation_dtype") != "float32"
        or value.get("target_dtype") != "float32"
        or value.get("observation_shape") != [HORIZON_DAYS, OBSERVATION_SIZE]
        or value.get("target_shape") != [HORIZON_DAYS, ACTION_SIZE]
    ):
        raise OracleBCError(f"{row_id} causal demonstration contract drifted")
    raw_observations = value.get("observations")
    raw_targets = value.get("targets")
    if (
        canonical_hash(raw_observations) != value.get("observations_sha256")
        or canonical_hash(raw_targets) != value.get("targets_sha256")
        or canonical_hash(
            {"observations": raw_observations, "targets": raw_targets}
        )
        != value.get("dataset_sha256")
    ):
        raise OracleBCError(f"{row_id} demonstration hash mismatch")
    observations = np.asarray(raw_observations, dtype=np.float32)
    targets = np.asarray(raw_targets, dtype=np.float32)
    if (
        observations.shape != (HORIZON_DAYS, OBSERVATION_SIZE)
        or targets.shape != (HORIZON_DAYS, ACTION_SIZE)
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(targets))
        or np.any(np.abs(targets) > 1.0)
    ):
        raise OracleBCError(f"{row_id} demonstration tensor is invalid")
    return observations, targets, str(value["dataset_sha256"])


def _array_sha256(value: np.ndarray) -> str:
    """Hash one numeric array with its exact dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _hand_rule_targets(observations: np.ndarray) -> np.ndarray:
    """Apply the public preparedness teacher independently to every row."""

    targets: list[np.ndarray] = []
    for index, observation in enumerate(observations):
        action, evidence = preparedness_teacher_action(observation)
        action_array = np.asarray(action, dtype=np.float32)
        if (
            evidence.get("teacher_id")
            != "public-preparedness-curriculum-v3"
            or evidence.get("teacher_version") != "1.0.0"
            or evidence.get("uses_exact_public_observation") is not True
            or evidence.get("future_tape_visible") is not False
            or action_array.shape != (ACTION_SIZE,)
            or not np.all(np.isfinite(action_array))
            or np.any(np.abs(action_array) > 1.0)
        ):
            raise OracleBCError(
                f"preparedness teacher causal API drifted at row {index}"
            )
        targets.append(action_array)
    result = np.asarray(targets, dtype=np.float32)
    if result.shape != (DATASET_ROW_COUNT, ACTION_SIZE):
        raise OracleBCError("preparedness teacher target shape drifted")
    result.setflags(write=False)
    return result


def _trajectory_split_contract(
    row_ids: Sequence[str],
    observations: np.ndarray,
    oracle_targets: np.ndarray,
    hand_rule_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build and bind the preregistered 168/24 trajectory-level split."""

    expected = _expected_training_case_contracts()
    expected_row_ids = tuple(case["row_id"] for case in expected)
    if tuple(row_ids) != expected_row_ids:
        raise OracleBCError("trajectory split received a noncanonical roster")
    fit_case_indices = [
        index
        for index, case in enumerate(expected)
        if case["case_seed"] in FIT_SEEDS
    ]
    holdout_case_indices = [
        index
        for index, case in enumerate(expected)
        if case["case_seed"] in HOLDOUT_SEEDS
    ]

    def step_indices(case_indices: Sequence[int]) -> np.ndarray:
        return np.asarray(
            [
                case_index * HORIZON_DAYS + day
                for case_index in case_indices
                for day in range(HORIZON_DAYS)
            ],
            dtype=np.int64,
        )

    fit_indices = step_indices(fit_case_indices)
    holdout_indices = step_indices(holdout_case_indices)

    def split_record(
        name: str,
        case_indices: Sequence[int],
        indices: np.ndarray,
    ) -> dict[str, Any]:
        trajectory_ids = [expected[index]["row_id"] for index in case_indices]
        step_ids = [
            f"{expected[index]['row_id']}:day-{day:02d}"
            for index in case_indices
            for day in range(HORIZON_DAYS)
        ]
        return {
            "id": name,
            "trajectory_count": len(case_indices),
            "row_count": len(indices),
            "trajectory_row_ids": trajectory_ids,
            "trajectory_row_ids_sha256": canonical_hash(trajectory_ids),
            "step_row_ids_sha256": canonical_hash(step_ids),
            "observations_sha256": _array_sha256(observations[indices]),
            "oracle_targets_sha256": _array_sha256(oracle_targets[indices]),
            "hand_rule_targets_sha256": _array_sha256(
                hand_rule_targets[indices]
            ),
        }

    if (
        len(fit_case_indices) != FIT_CASE_COUNT
        or len(holdout_case_indices) != HOLDOUT_CASE_COUNT
        or fit_indices.shape != (FIT_ROW_COUNT,)
        or holdout_indices.shape != (HOLDOUT_ROW_COUNT,)
        or set(fit_indices) & set(holdout_indices)
        or sorted(np.concatenate((fit_indices, holdout_indices)).tolist())
        != list(range(DATASET_ROW_COUNT))
    ):
        raise OracleBCError("preregistered trajectory split drifted")
    contract = {
        "strategy": "last_four_training_seeds_per_family_held_out",
        "family_count": 6,
        "fit_case_seeds": list(FIT_SEEDS),
        "holdout_case_seeds": list(HOLDOUT_SEEDS),
        "fit": split_record("fit", fit_case_indices, fit_indices),
        "holdout": split_record(
            "trajectory_level_holdout",
            holdout_case_indices,
            holdout_indices,
        ),
    }
    fit_indices.setflags(write=False)
    holdout_indices.setflags(write=False)
    return fit_indices, holdout_indices, contract


def _expected_training_case_contracts() -> list[dict[str, Any]]:
    """Recompute the complete ordered training roster independently."""

    cases: list[dict[str, Any]] = []
    for family in TRAINING_FAMILIES:
        for case_seed in TRAINING_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape(scenario, tape_seed)
            cases.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "family_id": family.id,
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "scenario_sha256": canonical_hash(
                        scenario.model_dump(mode="json")
                    ),
                    "tape_sha256": canonical_hash(
                        [asdict(shock) for shock in schedule]
                    ),
                }
            )
    if (
        len(cases) != TRAINING_CASE_COUNT
        or len({case["row_id"] for case in cases}) != TRAINING_CASE_COUNT
        or len(TRAINING_FAMILIES) != 6
        or TRAINING_SEEDS != tuple(range(810000, 810032))
    ):
        raise OracleBCError("canonical training roster must remain 6 x 32")
    return cases


def load_oracle_dataset(dataset_root: Path) -> OracleDataset:
    """Load exactly 5,760 public-state/action pairs from validated shards."""

    protocol_path = dataset_root / "protocol.json"
    receipt_path = dataset_root / "training" / "receipt.json"
    protocol = _load_json(protocol_path, "training oracle protocol")
    receipt = _load_json(receipt_path, "training oracle receipt")
    contract = protocol.get("contract")
    contract_sha256 = protocol.get("contract_sha256")
    split = contract.get("split") if isinstance(contract, dict) else None
    demonstration_contract = (
        contract.get("demonstration_contract")
        if isinstance(contract, dict)
        else None
    )
    source_identity = (
        contract.get("source_identity") if isinstance(contract, dict) else None
    )
    expected_cases = _expected_training_case_contracts()
    if (
        not isinstance(contract, dict)
        or canonical_hash(contract) != contract_sha256
        or contract.get("tool") != DATASET_TOOL_ID
        or not isinstance(split, dict)
        or split.get("id") != "train"
        or split.get("family_count") != 6
        or split.get("family_ids")
        != [family.id for family in TRAINING_FAMILIES]
        or split.get("cartesian_case_count") != TRAINING_CASE_COUNT
        or split.get("seed_interval")
        != {"first": 810000, "last": 810031, "count": 32}
        or contract.get("access_contract")
        != {
            "training_split_used": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": False,
        }
        or contract.get("oracle_config") != EXPECTED_ORACLE_CONFIG
        or not isinstance(demonstration_contract, dict)
        or demonstration_contract.get("case_count") != TRAINING_CASE_COUNT
        or demonstration_contract.get("horizon_days") != HORIZON_DAYS
        or demonstration_contract.get("row_count") != DATASET_ROW_COUNT
        or demonstration_contract.get("observation_count") != OBSERVATION_SIZE
        or demonstration_contract.get("action_count") != ACTION_SIZE
        or demonstration_contract.get("observation_order")
        != list(OBSERVATION_ORDER)
        or demonstration_contract.get("action_order") != list(ACTION_ORDER)
        or demonstration_contract.get("student_input_future_tape_visible")
        is not False
        or demonstration_contract.get("teacher_target_uses_full_future_tape")
        is not True
        or not isinstance(source_identity, dict)
        or set(source_identity) != set(DATASET_SOURCE_PATHS)
        or contract.get("ordered_case_contract_sha256")
        != canonical_hash(expected_cases)
    ):
        raise OracleBCError("training oracle protocol contract drifted")
    for relative_path in DATASET_SOURCE_PATHS:
        if file_sha256(
            ROOT / relative_path,
            label=f"dataset source {relative_path}",
            error_type=OracleBCError,
        ) != source_identity[relative_path]:
            raise OracleBCError(f"training oracle source hash drifted: {relative_path}")
    rows = receipt.get("rows")
    invariants = receipt.get("invariants")
    if (
        receipt.get("tool") != DATASET_TOOL_ID
        or receipt.get("status") != "complete_training_oracle_demonstrations"
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("case_count") != TRAINING_CASE_COUNT
        or receipt.get("demonstration_row_count") != DATASET_ROW_COUNT
        or receipt.get("observation_count") != OBSERVATION_SIZE
        or receipt.get("action_count") != ACTION_SIZE
        or receipt.get("student_trained") is not False
        or not isinstance(rows, list)
        or len(rows) != TRAINING_CASE_COUNT
        or canonical_hash(rows) != receipt.get("rows_sha256")
        or not isinstance(invariants, dict)
        or not all(
            invariants.get(name) is expected
            for name, expected in {
                "case_count_exactly_192": True,
                "demonstration_rows_exactly_5760": True,
                "row_ids_unique": True,
                "observation_dimension_exactly_73": True,
                "action_dimension_exactly_22": True,
                "all_hard_violation_counts_zero": True,
                "all_conservation_residuals_exactly_zero": True,
                "development_split_used": False,
                "final_split_used": False,
                "learned_policy_loaded_or_run": False,
            }.items()
        )
    ):
        raise OracleBCError("training oracle receipt contract drifted")

    observation_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    row_ids: list[str] = []
    dataset_index: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OracleBCError(f"dataset row {index} is invalid")
        expected_case = expected_cases[index]
        row_id = expected_case["row_id"]
        if any(
            row.get(key) != expected_value
            for key, expected_value in expected_case.items()
        ):
            raise OracleBCError(
                f"dataset row {index} canonical case identity drifted"
            )
        shard_path = _safe_shard_path(dataset_root, row.get("shard"))
        if file_sha256(
            shard_path,
            label=f"oracle shard {index}",
            error_type=OracleBCError,
        ) != row.get("shard_sha256"):
            raise OracleBCError(f"oracle shard {index} hash mismatch")
        shard = _load_json(shard_path, f"oracle shard {index}")
        payload = shard.get("payload")
        case = shard.get("case")
        if (
            shard.get("contract_sha256") != contract_sha256
            or shard.get("split") != "train"
            or shard.get("phase") != "oracle"
            or shard.get("index") != index
            or case != expected_case
            or not isinstance(payload, dict)
            or payload.get("row_id") != row_id
        ):
            raise OracleBCError(f"oracle shard {index} binding drifted")
        observations, targets, dataset_sha256 = _validate_demonstration(
            payload.get("demonstration"),
            row_id=row_id,
        )
        if dataset_sha256 != row.get("dataset_sha256"):
            raise OracleBCError(f"dataset row {index} digest mismatch")
        observation_batches.append(observations)
        target_batches.append(targets)
        row_ids.append(row_id)
        dataset_index.append(
            {"row_id": row_id, "dataset_sha256": dataset_sha256}
        )
    if (
        len(set(row_ids)) != TRAINING_CASE_COUNT
        or canonical_hash(dataset_index) != receipt.get("dataset_index_sha256")
    ):
        raise OracleBCError("training oracle dataset index drifted")

    observations = np.concatenate(observation_batches).astype(np.float32, copy=False)
    oracle_targets = np.concatenate(target_batches).astype(
        np.float32, copy=False
    )
    if (
        observations.shape != (DATASET_ROW_COUNT, OBSERVATION_SIZE)
        or oracle_targets.shape != (DATASET_ROW_COUNT, ACTION_SIZE)
    ):
        raise OracleBCError("training oracle aggregate tensor shape drifted")
    observations.setflags(write=False)
    oracle_targets.setflags(write=False)
    hand_rule_targets = _hand_rule_targets(observations)
    fit_indices, holdout_indices, split_contract = _trajectory_split_contract(
        row_ids,
        observations,
        oracle_targets,
        hand_rule_targets,
    )
    return OracleDataset(
        observations=observations,
        oracle_targets=oracle_targets,
        hand_rule_targets=hand_rule_targets,
        row_ids=tuple(row_ids),
        fit_indices=fit_indices,
        holdout_indices=holdout_indices,
        split_contract=split_contract,
        receipt_path=receipt_path,
        receipt_sha256=file_sha256(receipt_path),
        contract_sha256=str(contract_sha256),
        dataset_index_sha256=str(receipt["dataset_index_sha256"]),
    )


def _source_identity() -> dict[str, str]:
    paths = (
        "scripts/train_oracle_bc_student.py",
        "scripts/train_policy.py",
        "scripts/training_artifacts.py",
        "backend/app/shared_evidence.py",
        "backend/app/city/environment.py",
        "backend/app/city/scenarios.py",
        "backend/app/city/outcome.py",
        "backend/app/city/planners.py",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def student_contract(
    dataset: OracleDataset,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Bind the single fit and one development decision before execution."""

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "source_identity": _source_identity(),
        "dataset": {
            "receipt_path": str(dataset.receipt_path),
            "receipt_sha256": dataset.receipt_sha256,
            "contract_sha256": dataset.contract_sha256,
            "dataset_index_sha256": dataset.dataset_index_sha256,
            "case_count": TRAINING_CASE_COUNT,
            "row_count": DATASET_ROW_COUNT,
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "observation_order": list(OBSERVATION_ORDER),
            "action_order": list(ACTION_ORDER),
            "trajectory_split": dataset.split_contract,
            "label_sources": {
                "oracle": (
                    "privileged_same_budget_CEM_actions; future tape is used "
                    "only by the label generator, never as student input"
                ),
                "matched_hand_rule_control": (
                    "preparedness_teacher_action applied independently to "
                    "each exact public observation"
                ),
                "matched_hand_rule_targets_sha256": _array_sha256(
                    dataset.hand_rule_targets
                ),
            },
        },
        "baseline_anchor": baseline,
        "baseline_comparability_disclosure": {
            "old_selected_seed_bc": {
                "method": "behavior_cloning_with_4_dagger_rounds",
                "training_observation_count": 23040,
                "development_solved_count": OLD_BC_SOLVED_COUNT,
            },
            "new_oracle_student": {
                "method": "single_fixed_offline_behavior_cloning_fit",
                "training_observation_count": FIT_ROW_COUNT,
                "development_solved_count": "measured_once_after_fit",
            },
            "matched_hand_rule_control": {
                "method": "single_fixed_offline_behavior_cloning_fit",
                "label_source": "preparedness_teacher_action",
                "training_observation_count": FIT_ROW_COUNT,
                "heldout_observation_count": HOLDOUT_ROW_COUNT,
                "development_evaluated": False,
                "purpose": (
                    "matched label-source diagnostic, not the historical "
                    "four-round DAgger baseline"
                ),
            },
            "like_for_like_training_volume_claimed": False,
        },
        "offline_distillation_disclosure": {
            "teacher_forced_off_policy_dataset_collection_passes": 1,
            "static_dataset_optimization_epochs": BC_EPOCHS,
            "interactive_relabelling": False,
            "dagger": False,
            "oracle_relabelled_student_states": False,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
            "why_no_oracle_dagger": (
                "Full CEM oracle relabelling at student-visited states is "
                "not computationally affordable or safely supported by this "
                "fixed trajectory dataset."
            ),
        },
        "fit": {
            "method": "matched_behavior_cloning_only",
            "policy_seed": POLICY_SEED,
            "epochs": BC_EPOCHS,
            "batch_size": BC_BATCH_SIZE,
            "learning_rate": BC_LEARNING_RATE,
            "loss": "mean_squared_error_of_deterministic_action_mean",
            "dataset_passes": BC_EPOCHS,
            "dagger_iterations": 0,
            "ppo_updates": 0,
            "actor_architecture": [384, 256, 128],
            "activation": "SiLU",
            "orthogonal_initialization": True,
            "log_standard_deviation_initialization": -1.5,
            "critic_trained": False,
            "fit_trajectory_count_per_student": FIT_CASE_COUNT,
            "fit_row_count_per_student": FIT_ROW_COUNT,
            "heldout_trajectory_count_per_student": HOLDOUT_CASE_COUNT,
            "heldout_row_count_per_student": HOLDOUT_ROW_COUNT,
            "label_treatments": [
                "privileged_oracle_actions",
                "preparedness_teacher_action_public_rule",
            ],
            "matched_initial_actor_state_required": True,
            "matched_minibatch_order_required": True,
        },
        "normalization": {
            "implementation": "VecNormalize observation RunningMeanStd",
            "fit_once_on_5040_fit_observations": True,
            "holdout_observations_excluded_from_rms": True,
            "identical_for_oracle_and_hand_rule_students": True,
            "epsilon": 1e-8,
            "clip_observation": 10.0,
            "frozen_before_fit_and_development_evaluation": True,
            "reward_normalization": False,
        },
        "catastrophic_gate": {
            "evaluation_count": 1,
            "case_count": DEVELOPMENT_CASE_COUNT,
            "development_condition": {
                "metric": "solved_count",
                "operator": ">=",
                "threshold": DEVELOPMENT_CATASTROPHIC_FLOOR,
            },
            "oracle_holdout_condition": {
                "metric": "relative_mse_improvement",
                "operator": ">",
                "threshold": HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR,
            },
            "on_pass": "record_eligible_for_separately_authorized_3_seed_ppo",
            "on_fail": "abort_before_ppo",
            "nonfinite_or_invalid_metric_is_failure": True,
            "interpretation": (
                "This is a catastrophic fit/generalization diagnostic on "
                "held-out training trajectories, not proof of optimality or "
                "expected future performance."
            ),
        },
        "access_contract": {
            "training_split_used_for_fit": True,
            "development_split_used_once_for_gate": True,
            "final_split_imported_or_used": False,
            "future_tape_fields_used_as_student_inputs": False,
        },
        "implementation_references": {
            name: function_source_sha256(
                ROOT,
                "scripts/train_policy.py",
                name,
                error_type=OracleBCError,
            )
            for name in (
                "build_model",
                "normalize_observations",
                "evaluate_development_frozen",
            )
        },
        "matched_control_reference": function_source_sha256(
            ROOT,
            "backend/app/city/planners.py",
            "preparedness_teacher_action",
            error_type=OracleBCError,
        ),
    }


def _model_environment() -> VecNormalize:
    family = TRAINING_FAMILIES[0]
    case_seed = TRAINING_SEEDS[0]
    scenario = family.build(case_seed)
    tape_seed = family.tape_seed(case_seed)
    schedule = generate_disaster_tape(scenario, tape_seed)

    def make_environment() -> CityRecoveryEnv:
        return CityRecoveryEnv(
            scenario,
            tape_seed,
            schedule,
            collect_evidence=False,
        )

    return VecNormalize(
        DummyVecEnv([make_environment]),
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        epsilon=1e-8,
        gamma=0.99,
    )


def _parameter_digest(parameters: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _critic_parameters(model: PPO) -> list[tuple[str, torch.Tensor]]:
    return [
        *(
            (f"value_net.{name}", parameter)
            for name, parameter in model.policy.mlp_extractor.value_net.named_parameters()
        ),
        *(
            (f"value_head.{name}", parameter)
            for name, parameter in model.policy.value_net.named_parameters()
        ),
    ]


def _actor_parameters(model: PPO) -> list[torch.Tensor]:
    return [
        *model.policy.mlp_extractor.policy_net.parameters(),
        *model.policy.action_net.parameters(),
    ]


def _prediction_metrics(
    model: PPO,
    observations: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    model.policy.eval()
    with torch.no_grad():
        prediction = model.policy.get_distribution(
            observations
        ).distribution.mean
        error = prediction - targets
        mse = float(torch.mean(torch.square(error)).cpu())
        per_dimension = torch.mean(torch.abs(error), dim=0).cpu().numpy()
    result = {
        "mse": mse,
        "mean_absolute_error_by_dimension": [
            float(value) for value in per_dimension
        ],
        "mean_absolute_error": float(np.mean(per_dimension)),
    }
    if (
        not np.isfinite(result["mse"])
        or result["mse"] < 0.0
        or len(result["mean_absolute_error_by_dimension"]) != ACTION_SIZE
        or not np.all(
            np.isfinite(result["mean_absolute_error_by_dimension"])
        )
    ):
        raise OracleBCError("BC prediction metrics are nonfinite or invalid")
    return result


def _relative_mse_improvement(
    initial: float, trained: float
) -> float | None:
    if (
        not np.isfinite(initial)
        or not np.isfinite(trained)
        or initial <= 0.0
        or trained < 0.0
    ):
        return None
    return (initial - trained) / initial


def _fit_actor(
    model: PPO,
    observations: torch.Tensor,
    targets: torch.Tensor,
    permutations: Sequence[torch.Tensor],
) -> tuple[float, str, str]:
    actor_parameters = _actor_parameters(model)
    critic_parameters = _critic_parameters(model)
    critic_before = _parameter_digest(critic_parameters)
    optimizer = torch.optim.Adam(actor_parameters, lr=BC_LEARNING_RATE)
    final_batch_mse = float("nan")
    model.policy.train()
    for permutation in permutations:
        for start in range(0, FIT_ROW_COUNT, BC_BATCH_SIZE):
            indices = permutation[start : start + BC_BATCH_SIZE]
            predicted = model.policy.get_distribution(
                observations[indices]
            ).distribution.mean
            loss = torch.nn.functional.mse_loss(predicted, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            final_batch_mse = float(loss.detach().cpu())
    model.policy.eval()
    critic_after = _parameter_digest(critic_parameters)
    if (
        critic_after != critic_before
        or int(model.num_timesteps) != 0
        or not np.isfinite(final_batch_mse)
    ):
        raise OracleBCError("BC-only fit changed critic, PPO counters, or loss")
    return final_batch_mse, critic_before, critic_after


def fit_student(dataset: OracleDataset) -> StudentFit:
    """Fit matched oracle-label and hand-rule-label actors on 168 trajectories."""

    if (
        dataset.observations.shape != (DATASET_ROW_COUNT, OBSERVATION_SIZE)
        or dataset.oracle_targets.shape != (DATASET_ROW_COUNT, ACTION_SIZE)
        or dataset.hand_rule_targets.shape
        != (DATASET_ROW_COUNT, ACTION_SIZE)
        or dataset.fit_indices.shape != (FIT_ROW_COUNT,)
        or dataset.holdout_indices.shape != (HOLDOUT_ROW_COUNT,)
    ):
        raise OracleBCError("student fit received a noncanonical dataset")
    oracle_normalizer = _model_environment()
    control_normalizer = _model_environment()
    try:
        fit_observations = dataset.observations[dataset.fit_indices]
        oracle_normalizer.obs_rms.update(fit_observations)
        control_normalizer.obs_rms.update(fit_observations)
        observation_rms = rms_state(oracle_normalizer.obs_rms)
        control_observation_rms = rms_state(control_normalizer.obs_rms)
        observation_rms_sha256 = rms_digest(observation_rms)
        if rms_digest(control_observation_rms) != observation_rms_sha256:
            raise OracleBCError("matched students received different observation RMS")
        normalized = normalize_observations(
            dataset.observations, observation_rms
        )
        oracle_normalizer.training = False
        oracle_normalizer.norm_reward = False
        control_normalizer.training = False
        control_normalizer.norm_reward = False

        reset_policy_seed(POLICY_SEED)
        oracle_model = build_model(
            oracle_normalizer,
            seed=POLICY_SEED,
            n_steps=MODEL_N_STEPS,
            batch_size=MODEL_BATCH_SIZE,
            learning_rate=MODEL_LEARNING_RATE,
            target_kl=MODEL_TARGET_KL,
            ent_coef=MODEL_ENT_COEF,
        )
        reset_policy_seed(POLICY_SEED)
        control_model = build_model(
            control_normalizer,
            seed=POLICY_SEED,
            n_steps=MODEL_N_STEPS,
            batch_size=MODEL_BATCH_SIZE,
            learning_rate=MODEL_LEARNING_RATE,
            target_kl=MODEL_TARGET_KL,
            ent_coef=MODEL_ENT_COEF,
        )
        oracle_actor_before = actor_state_sha256(oracle_model)
        control_actor_before = actor_state_sha256(control_model)
        oracle_critic_before = _parameter_digest(
            _critic_parameters(oracle_model)
        )
        control_critic_before = _parameter_digest(
            _critic_parameters(control_model)
        )
        if (
            oracle_actor_before != control_actor_before
            or oracle_critic_before != control_critic_before
        ):
            raise OracleBCError("matched student initialization hashes differ")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(POLICY_SEED ^ 0xBC37017)
        permutations = [
            torch.randperm(FIT_ROW_COUNT, generator=generator)
            for _ in range(BC_EPOCHS)
        ]
        observation_tensor = torch.as_tensor(normalized, dtype=torch.float32)
        fit_index_tensor = torch.as_tensor(
            dataset.fit_indices.copy(), dtype=torch.long
        )
        holdout_index_tensor = torch.as_tensor(
            dataset.holdout_indices.copy(), dtype=torch.long
        )
        fit_observation_tensor = observation_tensor[fit_index_tensor]
        holdout_observation_tensor = observation_tensor[holdout_index_tensor]
        oracle_fit_targets = torch.as_tensor(
            dataset.oracle_targets[dataset.fit_indices], dtype=torch.float32
        )
        oracle_holdout_targets = torch.as_tensor(
            dataset.oracle_targets[dataset.holdout_indices], dtype=torch.float32
        )
        control_fit_targets = torch.as_tensor(
            dataset.hand_rule_targets[dataset.fit_indices], dtype=torch.float32
        )
        control_holdout_targets = torch.as_tensor(
            dataset.hand_rule_targets[dataset.holdout_indices], dtype=torch.float32
        )
        oracle_untrained_holdout = _prediction_metrics(
            oracle_model, holdout_observation_tensor, oracle_holdout_targets
        )
        control_untrained_holdout = _prediction_metrics(
            control_model, holdout_observation_tensor, control_holdout_targets
        )
        oracle_initial_fit = _prediction_metrics(
            oracle_model, fit_observation_tensor, oracle_fit_targets
        )
        control_initial_fit = _prediction_metrics(
            control_model, fit_observation_tensor, control_fit_targets
        )
        oracle_final_batch, oracle_critic_before, oracle_critic_after = (
            _fit_actor(
                oracle_model,
                fit_observation_tensor,
                oracle_fit_targets,
                permutations,
            )
        )
        control_final_batch, control_critic_before, control_critic_after = (
            _fit_actor(
                control_model,
                fit_observation_tensor,
                control_fit_targets,
                permutations,
            )
        )
        oracle_trained_fit = _prediction_metrics(
            oracle_model, fit_observation_tensor, oracle_fit_targets
        )
        control_trained_fit = _prediction_metrics(
            control_model, fit_observation_tensor, control_fit_targets
        )
        oracle_trained_holdout = _prediction_metrics(
            oracle_model, holdout_observation_tensor, oracle_holdout_targets
        )
        control_trained_holdout = _prediction_metrics(
            control_model, holdout_observation_tensor, control_holdout_targets
        )

        def treatment_report(
            *,
            label_source: str,
            model: PPO,
            actor_before: str,
            critic_before: str,
            critic_after: str,
            final_batch_mse: float,
            initial_fit: dict[str, Any],
            trained_fit: dict[str, Any],
            untrained_holdout: dict[str, Any],
            trained_holdout: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "label_source": label_source,
                "actor_state_sha256_before": actor_before,
                "actor_state_sha256_after": actor_state_sha256(model),
                "critic_state_sha256_before": critic_before,
                "critic_state_sha256_after": critic_after,
                "critic_unchanged": critic_before == critic_after,
                "final_batch_mse": final_batch_mse,
                "fit": {
                    "untrained": initial_fit,
                    "trained": trained_fit,
                },
                "heldout": {
                    "untrained": untrained_holdout,
                    "trained": trained_holdout,
                    "relative_mse_improvement": _relative_mse_improvement(
                        untrained_holdout["mse"], trained_holdout["mse"]
                    ),
                },
            }

        oracle_report = treatment_report(
            label_source="privileged_same_budget_cem_oracle",
            model=oracle_model,
            actor_before=oracle_actor_before,
            critic_before=oracle_critic_before,
            critic_after=oracle_critic_after,
            final_batch_mse=oracle_final_batch,
            initial_fit=oracle_initial_fit,
            trained_fit=oracle_trained_fit,
            untrained_holdout=oracle_untrained_holdout,
            trained_holdout=oracle_trained_holdout,
        )
        control_report = treatment_report(
            label_source="preparedness_teacher_action_public_rule",
            model=control_model,
            actor_before=control_actor_before,
            critic_before=control_critic_before,
            critic_after=control_critic_after,
            final_batch_mse=control_final_batch,
            initial_fit=control_initial_fit,
            trained_fit=control_trained_fit,
            untrained_holdout=control_untrained_holdout,
            trained_holdout=control_trained_holdout,
        )
        if not oracle_report["critic_unchanged"] or not control_report[
            "critic_unchanged"
        ]:
            raise OracleBCError("matched BC fit changed a critic")
        report = {
            "method": "matched_behavior_cloning_only",
            "policy_seed": POLICY_SEED,
            "epochs": BC_EPOCHS,
            "batch_size": BC_BATCH_SIZE,
            "learning_rate": BC_LEARNING_RATE,
            "training_row_count_per_student": FIT_ROW_COUNT,
            "heldout_row_count_per_student": HOLDOUT_ROW_COUNT,
            "trajectory_split": dataset.split_contract,
            "matched_initialization": {
                "actor_state_sha256": oracle_actor_before,
                "critic_state_sha256": oracle_critic_before,
                "actor_hashes_equal_before_fit": True,
                "critic_hashes_equal_before_fit": True,
                "observation_rms_hashes_equal": True,
                "minibatch_permutations_identical": True,
            },
            "oracle_label_student": oracle_report,
            "matched_hand_rule_control": control_report,
            "dagger_iterations": 0,
            "ppo_updates": 0,
            "active_actor_critic_transitions": 0,
            "observation_rms_sha256": observation_rms_sha256,
            "observation_rms_count": float(observation_rms["count"]),
            "normalization_frozen": not oracle_normalizer.training,
            "holdout_excluded_from_normalization_fit": True,
        }
        control_normalizer.close()
        return StudentFit(
            model=oracle_model,
            normalizer=oracle_normalizer,
            report=report,
        )
    except Exception:
        oracle_normalizer.close()
        control_normalizer.close()
        raise


def _persist_fit(
    output_root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    dataset: OracleDataset,
) -> tuple[StudentFit, dict[str, Any]]:
    claim_path = output_root / "fit.claim.json"
    success_path = output_root / "fit.success.json"
    _atomic_create_json(
        claim_path,
        {
            "created_at_utc": _utc_now(),
            "contract_sha256": contract_sha256,
            "dataset_receipt_sha256": dataset.receipt_sha256,
            "policy_seed": POLICY_SEED,
            "oracle_label_fit_count": 1,
            "matched_hand_rule_fit_count": 1,
            "actor_fit_count": 2,
            "trajectory_split_sha256": canonical_hash(
                dataset.split_contract
            ),
            "development_evaluated": False,
        },
    )
    fit = fit_student(dataset)
    bundle_path = output_root / "bc-checkpoint"
    bundle_reference = persist_checkpoint_bundle(
        bundle_path,
        model=fit.model,
        normalizer=fit.normalizer,
        training_config={
            "tool": TOOL_ID,
            "method": "oracle_label_behavior_cloning_only",
            "contract_sha256": contract_sha256,
            "dataset_receipt_sha256": dataset.receipt_sha256,
            **contract["fit"],
            "normalization": contract["normalization"],
        },
        seed=POLICY_SEED,
        milestone="oracle-bc-only",
        checkpoint_id="oracle-bc-heldout-seed-67017",
        active_actor_critic_transitions=0,
    )
    success = {
        "created_at_utc": _utc_now(),
        "status": "complete_matched_bc_only_fits",
        "contract_sha256": contract_sha256,
        "dataset_receipt_sha256": dataset.receipt_sha256,
        "fit": fit.report,
        "baseline_anchor": contract["baseline_anchor"],
        "baseline_comparability_disclosure": contract[
            "baseline_comparability_disclosure"
        ],
        "offline_distillation_disclosure": contract[
            "offline_distillation_disclosure"
        ],
        "checkpoint_bundle": bundle_reference,
        "development_evaluated": False,
        "final_split_used": False,
    }
    _atomic_create_json(success_path, success)
    return fit, success


def _resume_fit(
    output_root: Path,
    contract_sha256: str,
) -> tuple[StudentFit, dict[str, Any]]:
    success_path = output_root / "fit.success.json"
    success = _load_json(success_path, "BC fit success")
    if (
        success.get("status") != "complete_matched_bc_only_fits"
        or success.get("contract_sha256") != contract_sha256
        or success.get("development_evaluated") is not False
        or success.get("final_split_used") is not False
    ):
        raise OracleBCError("persisted BC fit contract drifted")
    bundle = checkpoint_bundle_reference(output_root / "bc-checkpoint")
    if bundle != success.get("checkpoint_bundle"):
        raise OracleBCError("persisted BC bundle reference drifted")
    normalizer = _model_environment()
    try:
        loaded = load_checkpoint_bundle(
            output_root / "bc-checkpoint",
            env=normalizer,
            device="cpu",
        )
        apply_normalization_state(normalizer, loaded.normalization)
        normalizer.training = False
        normalizer.norm_reward = False
        return StudentFit(
            model=loaded.model,
            normalizer=normalizer,
            report=success["fit"],
        ), success
    except Exception:
        normalizer.close()
        raise


def catastrophic_gate(
    solved_count: Any,
    oracle_holdout_relative_mse_improvement: Any,
) -> dict[str, Any]:
    """Return the preregistered catastrophic-only continuation decision."""

    development_valid = (
        isinstance(solved_count, int)
        and not isinstance(solved_count, bool)
        and 0 <= solved_count <= DEVELOPMENT_CASE_COUNT
    )
    improvement_valid = bool(
        isinstance(oracle_holdout_relative_mse_improvement, (int, float))
        and not isinstance(oracle_holdout_relative_mse_improvement, bool)
        and np.isfinite(float(oracle_holdout_relative_mse_improvement))
    )
    development_passed = bool(
        development_valid
        and solved_count >= DEVELOPMENT_CATASTROPHIC_FLOOR
    )
    improvement_passed = bool(
        improvement_valid
        and float(oracle_holdout_relative_mse_improvement)
        > HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR
    )
    passed = development_passed and improvement_passed
    failed_conditions: list[str] = []
    if not development_valid:
        failed_conditions.append("development_solved_count_invalid")
    elif not development_passed:
        failed_conditions.append("development_solved_count_below_140")
    if not improvement_valid:
        failed_conditions.append("oracle_holdout_mse_improvement_invalid")
    elif not improvement_passed:
        failed_conditions.append("oracle_holdout_mse_improvement_at_or_below_1pct")
    return {
        "kind": "catastrophic_only",
        "conditions": {
            "development_solved_count": {
                "operator": ">=",
                "threshold": DEVELOPMENT_CATASTROPHIC_FLOOR,
                "observed": solved_count if development_valid else None,
                "valid": development_valid,
                "passed": development_passed,
            },
            "oracle_holdout_relative_mse_improvement": {
                "operator": ">",
                "threshold": HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR,
                "observed": (
                    float(oracle_holdout_relative_mse_improvement)
                    if improvement_valid
                    else None
                ),
                "valid": improvement_valid,
                "passed": improvement_passed,
            },
        },
        "passed": passed,
        "failed_conditions": failed_conditions,
        "decision": (
            "eligible_for_separately_authorized_3_seed_ppo"
            if passed
            else "abort_before_ppo"
        ),
        "completed": True,
        "retry_recommended": False,
    }


def _expected_development_identity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape(scenario, tape_seed)
            rows.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "tape_sha256": canonical_hash(
                        [asdict(shock) for shock in schedule]
                    ),
                }
            )
    if len(rows) != DEVELOPMENT_CASE_COUNT:
        raise OracleBCError("canonical development roster must contain 200 cases")
    return rows


def _validate_development(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleBCError("development evaluation is invalid")
    rows = value.get("rows")
    if (
        value.get("case_count") != DEVELOPMENT_CASE_COUNT
        or not isinstance(value.get("solved_count"), int)
        or isinstance(value.get("solved_count"), bool)
        or not isinstance(value.get("solve_rate"), (int, float))
        or isinstance(value.get("solve_rate"), bool)
        or not isinstance(rows, list)
        or len(rows) != DEVELOPMENT_CASE_COUNT
        or len({row.get("row_id") for row in rows if isinstance(row, dict)})
        != DEVELOPMENT_CASE_COUNT
        or int(value.get("hard_violation_count", -1)) != 0
        or float(value.get("maximum_conservation_residual", -1.0)) != 0.0
    ):
        raise OracleBCError("development evaluation contract drifted")
    expected_identity = _expected_development_identity()
    recomputed_solved_count = 0
    recomputed_hard_violation_count = 0
    recomputed_maximum_conservation_residual = 0.0
    for index, (row, expected) in enumerate(zip(rows, expected_identity, strict=True)):
        if not isinstance(row, dict) or any(
            row.get(key) != expected_value
            for key, expected_value in expected.items()
        ):
            raise OracleBCError(
                f"development row/tape identity drifted at index {index}"
            )
        solved = row.get("solved")
        hard_violation_count = row.get("hard_violation_count")
        conservation_residual = row.get("max_conservation_residual")
        if (
            not isinstance(solved, bool)
            or not isinstance(hard_violation_count, int)
            or isinstance(hard_violation_count, bool)
            or hard_violation_count < 0
            or not isinstance(conservation_residual, (int, float))
            or isinstance(conservation_residual, bool)
            or not np.isfinite(float(conservation_residual))
            or float(conservation_residual) < 0.0
        ):
            raise OracleBCError(
                f"development row outcome drifted at index {index}"
            )
        recomputed_solved_count += int(solved)
        recomputed_hard_violation_count += hard_violation_count
        recomputed_maximum_conservation_residual = max(
            recomputed_maximum_conservation_residual,
            float(conservation_residual),
        )
    recomputed_solve_rate = recomputed_solved_count / DEVELOPMENT_CASE_COUNT
    if (
        value["solved_count"] != recomputed_solved_count
        or float(value["solve_rate"]) != recomputed_solve_rate
        or value["hard_violation_count"] != recomputed_hard_violation_count
        or float(value["maximum_conservation_residual"])
        != recomputed_maximum_conservation_residual
        or recomputed_hard_violation_count != 0
        or recomputed_maximum_conservation_residual != 0.0
    ):
        raise OracleBCError("development aggregates do not match canonical rows")
    return value


def result_exit_code(receipt: dict[str, Any]) -> int:
    """Map a completed negative gate to exit 4, never to a retry signal."""

    gate = receipt.get("catastrophic_gate")
    if not isinstance(gate, dict) or gate.get("completed") is not True:
        raise OracleBCError("student receipt has no completed catastrophic gate")
    return 0 if gate.get("passed") is True else COMPLETED_NEGATIVE_GATE_EXIT_CODE


def evaluate_once(
    *,
    output_root: Path,
    contract_sha256: str,
    fit: StudentFit,
    fit_success: dict[str, Any],
    evaluator: Callable[[PPO, VecNormalize], dict[str, Any]] = (
        evaluate_development_frozen
    ),
) -> dict[str, Any]:
    """Claim and perform the single allowed development evaluation."""

    claim_path = output_root / "development-evaluation.claim.json"
    receipt_path = output_root / "student-receipt.json"
    failure_path = output_root / "development-evaluation.failure.json"
    if receipt_path.exists() or failure_path.exists():
        raise OracleBCError("development evaluation already has a terminal receipt")
    normalization_before = rms_digest(rms_state(fit.normalizer.obs_rms))
    oracle_holdout = fit.report.get("oracle_label_student", {}).get(
        "heldout", {}
    )
    oracle_holdout_improvement = oracle_holdout.get(
        "relative_mse_improvement"
    )
    claim_holdout_improvement = (
        float(oracle_holdout_improvement)
        if isinstance(oracle_holdout_improvement, (int, float))
        and not isinstance(oracle_holdout_improvement, bool)
        and np.isfinite(float(oracle_holdout_improvement))
        else None
    )
    _atomic_create_json(
        claim_path,
        {
            "created_at_utc": _utc_now(),
            "contract_sha256": contract_sha256,
            "fit_success_sha256": file_sha256(output_root / "fit.success.json"),
            "checkpoint_bundle": fit_success["checkpoint_bundle"],
            "evaluation_split": "dev",
            "evaluation_case_count": DEVELOPMENT_CASE_COUNT,
            "evaluation_count": 1,
            "catastrophic_gate": {
                "development_solved_count": {
                    "operator": ">=",
                    "threshold": DEVELOPMENT_CATASTROPHIC_FLOOR,
                },
                "oracle_holdout_relative_mse_improvement": {
                    "operator": ">",
                    "threshold": HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR,
                    "observed_before_development_claim": (
                        claim_holdout_improvement
                    ),
                },
            },
            "final_split_used": False,
        },
    )
    try:
        development = _validate_development(evaluator(fit.model, fit.normalizer))
        normalization_after = rms_digest(rms_state(fit.normalizer.obs_rms))
        if normalization_after != normalization_before:
            raise OracleBCError("development evaluation changed observation RMS")
        gate = catastrophic_gate(
            development["solved_count"], oracle_holdout_improvement
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "tool": TOOL_ID,
            "status": (
                "complete_eligible_for_separately_authorized_3_seed_ppo"
                if gate["passed"]
                else "complete_aborted_before_ppo"
            ),
            "completed": True,
            "retry_recommended": False,
            "created_at_utc": _utc_now(),
            "contract_sha256": contract_sha256,
            "fit": fit_success,
            "development": development,
            "development_rows_sha256": canonical_hash(development["rows"]),
            "comparison": {
                "old_selected_seed_bc": {
                    "development_solved_count": OLD_BC_SOLVED_COUNT,
                    "method": "behavior_cloning_with_4_dagger_rounds",
                    "training_observation_count": 23040,
                },
                "new_oracle_bc": {
                    "development_solved_count": development["solved_count"],
                    "method": "single_fixed_offline_behavior_cloning_fit",
                    "training_observation_count": FIT_ROW_COUNT,
                    "heldout_observation_count": HOLDOUT_ROW_COUNT,
                },
                "delta": development["solved_count"] - OLD_BC_SOLVED_COUNT,
                "like_for_like_training_volume_claimed": False,
                "matched_hand_rule_control": {
                    "development_evaluated": False,
                    "method": "single_fixed_offline_behavior_cloning_fit",
                    "label_source": "preparedness_teacher_action",
                    "training_observation_count": FIT_ROW_COUNT,
                    "heldout_observation_count": HOLDOUT_ROW_COUNT,
                    "heldout": fit.report.get(
                        "matched_hand_rule_control", {}
                    ).get("heldout"),
                    "distinct_from_historical_old_bc": True,
                },
            },
            "catastrophic_gate": gate,
            "ppo_started": False,
            "development_evaluation_count": 1,
            "development_split_used": True,
            "final_split_imported_or_used": False,
            "invariants": {
                "case_count_exactly_200": True,
                "hard_violation_count_zero": True,
                "conservation_residual_exactly_zero": True,
                "observation_rms_frozen": True,
                "dagger_iterations_zero": True,
                "ppo_updates_zero": True,
                "trajectory_holdout_excluded_from_fit": True,
                "matched_control_not_evaluated_on_development": True,
            },
        }
        _atomic_create_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        _atomic_create_json(
            failure_path,
            {
                "status": "terminal_development_evaluation_failure",
                "created_at_utc": _utc_now(),
                "contract_sha256": contract_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retry_permitted": False,
                "final_split_used": False,
            },
        )
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-bc-receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume-after-fit", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_root = _require_external_root(args.dataset_root, "--dataset-root")
    output_root = _require_external_root(args.output_root, "--output-root")
    old_bc_receipt = args.old_bc_receipt.resolve()
    if dataset_root == output_root or dataset_root in output_root.parents:
        raise OracleBCError("--output-root must be separate from the dataset root")
    if args.execute and output_root.exists():
        raise OracleBCError("--execute requires a new output root")
    if args.resume_after_fit and not output_root.is_dir():
        raise OracleBCError("--resume-after-fit requires an existing output root")
    return dataset_root, output_root, old_bc_receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_root, output_root, old_bc_receipt = _validate_args(args)
    baseline = validate_old_bc_anchor(old_bc_receipt)
    dataset = load_oracle_dataset(dataset_root)
    contract = student_contract(dataset, baseline)
    contract_sha256 = canonical_hash(contract)
    clean = _worktree_is_clean()
    if args.preflight:
        print(
            json.dumps(
                {
                    "tool": TOOL_ID,
                    "status": "ready" if clean else "blocked_dirty_worktree",
                    "filesystem_written": False,
                    "bc_fit_started": False,
                    "development_evaluated": False,
                    "final_split_imported_or_used": False,
                    "dataset_receipt_sha256": dataset.receipt_sha256,
                    "dataset_row_count": int(dataset.observations.shape[0]),
                    "fit_row_count": len(dataset.fit_indices),
                    "heldout_row_count": len(dataset.holdout_indices),
                    "policy_seed": POLICY_SEED,
                    "old_bc_anchor": baseline,
                    "catastrophic_gate": contract["catastrophic_gate"],
                    "contract_sha256": contract_sha256,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if clean else 3
    if not clean:
        raise OracleBCError("refusing fit or development evaluation from a dirty worktree")

    protocol_path = output_root / "protocol.json"
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=False)
        _atomic_create_json(
            protocol_path,
            {
                "created_at_utc": _utc_now(),
                "contract_sha256": contract_sha256,
                "contract": contract,
            },
        )
        fit, fit_success = _persist_fit(
            output_root,
            contract,
            contract_sha256,
            dataset,
        )
    else:
        protocol = _load_json(protocol_path, "BC student protocol")
        if (
            protocol.get("contract_sha256") != contract_sha256
            or protocol.get("contract") != contract
        ):
            raise OracleBCError("resume protocol differs from preregistered contract")
        if (output_root / "development-evaluation.claim.json").exists():
            raise OracleBCError("development evaluation was already claimed; no retry")
        fit, fit_success = _resume_fit(output_root, contract_sha256)

    started = time.perf_counter()
    try:
        receipt = evaluate_once(
            output_root=output_root,
            contract_sha256=contract_sha256,
            fit=fit,
            fit_success=fit_success,
        )
    finally:
        fit.normalizer.close()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output_root / "student-receipt.json"),
                "development_solved_count": receipt["development"][
                    "solved_count"
                ],
                "old_bc_solved_count": OLD_BC_SOLVED_COUNT,
                "delta": receipt["comparison"]["delta"],
                "catastrophic_gate": receipt["catastrophic_gate"],
                "elapsed_development_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "ppo_started": False,
                "final_split_imported_or_used": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result_exit_code(receipt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleBCError, TrainingArtifactError) as error:
        print(f"oracle BC student failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
