#!/usr/bin/env python3
"""Run the DEV-only large-network plus oracle-distillation experiment.

The existing oracle-distilled actor is 384/256/128 and therefore cannot be
loaded into the 768/512/256 treatment without an arbitrary shape transplant.
This runner instead fits a new large actor to the exact already-approved
5,040/720 oracle trajectory split, using the exact frozen observation moments
from the approved distillation run.  A matched large hand-rule control is fit
with identical initialization and minibatches.  The persisted oracle actor is
then combined with a fresh seeded large critic, a fixed 50k critic-only warm-up,
and 2M active PPO transitions for each of the three registered seeds.

There is no final-split import or evaluation in this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Callable

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    ACTION_SIZE,
    OBSERVATION_SIZE,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
)
from scripts import run_distilled_ppo_study as distilled  # noqa: E402
from scripts import run_large_architecture_study as capacity  # noqa: E402
from scripts import train_oracle_bc_student as oracle_bc  # noqa: E402
from scripts import train_policy  # noqa: E402
from scripts.training_artifacts import (  # noqa: E402
    LoadedCheckpointBundle,
    TrainingArtifactError,
    actor_state_sha256,
    apply_normalization_state,
    checkpoint_bundle_reference,
    load_checkpoint_bundle,
    persist_checkpoint_bundle,
    verify_checkpoint_bundle,
)

TOOL_ID = "run_combined_distilled_large_study.py"
SCHEMA_VERSION = 1
DEFAULT_DATASET_ROOT = Path(
    r"E:\city-recovery-training-oracle-v4-attempt-01"
)
DEFAULT_SOURCE_STUDENT_ROOT = Path(
    r"E:\city-recovery-oracle-bc-student-v4-attempt-01"
)
DISTILLATION_EVIDENCE_PATH = (
    ROOT / "internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json"
)
DISTILLATION_EVIDENCE_SHA256 = (
    "aee2df40263f892fb8d979ae190a483a91711564169bbac45336f32a24bb5e0d"
)
CAPACITY_EVIDENCE_PATH = (
    ROOT / "internal/developmental_runs/v4/network-capacity-study-200.json"
)
CAPACITY_EVIDENCE_SHA256 = (
    "fd27e39b3b4868e43231b91f879e1830f1b2380f37bd03c3b23b9e5510564304"
)

POLICY_SEEDS = (37_017, 47_017, 57_017)
FIT_POLICY_SEED = 67_017
HIDDEN_LAYERS = (768, 512, 256)
LEARNING_RATE = 3.0e-5
BC_LEARNING_RATE = 1.0e-3
BC_EPOCHS = 15
BC_BATCH_SIZE = 512
FIT_ROW_COUNT = 5_040
HOLDOUT_ROW_COUNT = 720
ACTIVE_TRANSITIONS = 2_000_000
FIXED_CRITIC_WARMUP_TRANSITIONS = 50_000
SELECTION_MILESTONES = (500_000, 1_000_000, 2_000_000)
TRAINER_MILESTONES = (200_000, *SELECTION_MILESTONES)
DEVELOPMENT_CASE_COUNT = 200
INCUMBENT_ENDPOINTS = (172, 171, 171)
LARGE_ONLY_ENDPOINTS = (178, 176, 175)
INCUMBENT_FIVE_SEED_MEAN = 171.4
SHIPPED_DEVELOPMENT_SOLVED_COUNT = 178
PROMOTION_BEST_SOLVES = 183
PROMOTION_ENDPOINT_SOLVES = 172
PROMOTION_ENDPOINT_SEED_COUNT = 2
ARM = capacity.LearningRateArm("combined_large_lr_3e_5_oracle_bc", LEARNING_RATE)


class CombinedStudyError(RuntimeError):
    """Raised when the registered combined-study contract cannot be honored."""


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=CombinedStudyError)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CombinedStudyError(f"refusing to overwrite evidence: {path}")
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
        raise CombinedStudyError(
            f"refusing to overwrite evidence: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_external_root(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CombinedStudyError(f"{label} must be absolute")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise CombinedStudyError(f"{label} must be outside the repository")
    if resolved == Path(resolved.anchor):
        raise CombinedStudyError(f"{label} cannot be a filesystem root")
    return resolved


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CombinedStudyError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_recorded_sources(source_contract: Any, label: str) -> None:
    if not isinstance(source_contract, dict):
        raise CombinedStudyError(f"{label} source contract is missing")
    sources = source_contract.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise CombinedStudyError(f"{label} source files are missing")
    for relative, expected in sources.items():
        if not isinstance(relative, str):
            raise CombinedStudyError(f"{label} source path is invalid")
        _sha256(expected, f"{label} source {relative}")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise CombinedStudyError(
                f"{label} source escapes the repository"
            ) from exc
        if not path.is_file() or file_sha256(path) != expected:
            raise CombinedStudyError(f"{label} source drifted: {relative}")


def load_portable_upstream_evidence() -> dict[str, Any]:
    """Validate and bind the published distillation and capacity receipts."""

    if file_sha256(DISTILLATION_EVIDENCE_PATH) != DISTILLATION_EVIDENCE_SHA256:
        raise CombinedStudyError("portable distillation evidence drifted")
    if file_sha256(CAPACITY_EVIDENCE_PATH) != CAPACITY_EVIDENCE_SHA256:
        raise CombinedStudyError("portable capacity evidence drifted")
    distillation = _load_json(
        DISTILLATION_EVIDENCE_PATH, "portable distillation evidence"
    )
    capacity_receipt = _load_json(
        CAPACITY_EVIDENCE_PATH, "portable capacity evidence"
    )
    distillation_endpoints = distillation.get("endpoint_summary", {})
    distillation_upstream = distillation.get("upstream_evidence", {})
    source_student = distillation_upstream.get("oracle_bc_student", {})
    large_arm = capacity_receipt.get("arm_endpoint_summaries", {}).get(
        "large_lr_3e_5", {}
    )
    capacity_comparison = capacity_receipt.get("comparison", {}).get(
        "large_lr_3e_5_vs_incumbent_same_seed_2m", {}
    )
    architecture = capacity_receipt.get("architecture", {})
    capacity_study_runs = capacity_receipt.get("study_runs")
    large_only_runs = (
        sorted(
            (
                row
                for row in capacity_study_runs
                if isinstance(row, dict)
                and row.get("arm_id") == "large_lr_3e_5"
            ),
            key=lambda row: int(row.get("policy_seed", -1)),
        )
        if isinstance(capacity_study_runs, list)
        else []
    )
    if (
        distillation.get("schema_version")
        != "city-recovery-oracle-distilled-ppo-dev-evidence-v1"
        or distillation.get("status") != "complete_not_promoted"
        or distillation.get("split") != "dev"
        or distillation.get("final_split_imported_or_used") is not False
        or distillation.get("registered_policy_seeds") != list(POLICY_SEEDS)
        or distillation_endpoints.get("solved_counts_by_seed")
        != {"37017": 178, "47017": 174, "57017": 170}
        or distillation_endpoints.get("mean_solved_count") != 174.0
        or source_student.get("receipt_sha256")
        != distilled.APPROVED_STUDENT_RECEIPT_SHA256
        or source_student.get("observation_rms_sha256")
        != "cb7b9a46369a0c225c3a6254433f6ef37e52b822ef44598fa4311b64e63a4ba4"
    ):
        raise CombinedStudyError("portable distillation result contract drifted")
    if (
        capacity_receipt.get("schema_version")
        != "city-recovery-network-capacity-dev-evidence-v1"
        or capacity_receipt.get("status") != "complete_not_promoted"
        or capacity_receipt.get("split") != "dev"
        or capacity_receipt.get("final_split_imported_or_used") is not False
        or capacity_receipt.get("registered_policy_seeds") != list(POLICY_SEEDS)
        or architecture.get("actor_hidden_layers") != list(HIDDEN_LAYERS)
        or architecture.get("critic_hidden_layers") != list(HIDDEN_LAYERS)
        or large_arm.get("learning_rate") != LEARNING_RATE
        or large_arm.get("solved_counts_by_seed")
        != {"37017": 178, "47017": 176, "57017": 175}
        or large_arm.get("mean_solved_count") != fmean(LARGE_ONLY_ENDPOINTS)
        or capacity_comparison.get("policy_seeds") != list(POLICY_SEEDS)
        or capacity_comparison.get("incumbent_solved_counts")
        != list(INCUMBENT_ENDPOINTS)
        or capacity_comparison.get("challenger_solved_counts")
        != list(LARGE_ONLY_ENDPOINTS)
        or [row.get("policy_seed") for row in large_only_runs]
        != list(POLICY_SEEDS)
        or [row.get("critic_warmup_transitions") for row in large_only_runs]
        != [50_000, 50_000, 60_000]
        or len(
            {
                row.get("bc_initialization_identity", {}).get(
                    "observation_rms_sha256"
                )
                for row in large_only_runs
            }
        )
        != len(POLICY_SEEDS)
        or len(
            {
                row.get("bc_initialization_identity", {}).get(
                    "dataset_sha256"
                )
                for row in large_only_runs
            }
        )
        != len(POLICY_SEEDS)
    ):
        raise CombinedStudyError("portable capacity result contract drifted")
    _validate_recorded_sources(
        distillation.get("source_contract"), "distillation"
    )
    _validate_recorded_sources(
        capacity_receipt.get("source_contract"), "capacity"
    )
    return {
        "distillation": {
            "path": str(DISTILLATION_EVIDENCE_PATH),
            "sha256": DISTILLATION_EVIDENCE_SHA256,
            "source_contract": distillation["source_contract"],
            "endpoint_solved_counts": [178, 174, 170],
            "source_student": source_student,
        },
        "capacity": {
            "path": str(CAPACITY_EVIDENCE_PATH),
            "sha256": CAPACITY_EVIDENCE_SHA256,
            "source_contract": capacity_receipt["source_contract"],
            "large_lr_3e_5_endpoints": list(LARGE_ONLY_ENDPOINTS),
            "architecture": architecture,
            "large_lr_3e_5_run_bindings": [
                {
                    "policy_seed": row["policy_seed"],
                    "critic_warmup_transitions": row[
                        "critic_warmup_transitions"
                    ],
                    "bc_initialization_identity": row[
                        "bc_initialization_identity"
                    ],
                    "training_receipt": row["training_receipt"],
                }
                for row in large_only_runs
            ],
            "historical_initialization_method": {
                "teacher": "preparedness_teacher_action_public_rule",
                "behavior_cloning_iterations": 4,
                "dagger_iterations": 4,
                "normalization": "fit_independently_per_policy_seed",
                "derivation": (
                    "hash-bound run_large_architecture_study.py delegated "
                    "unchanged initialization to hash-bound train_policy.py; "
                    "the portable per-seed dataset/actor/RMS identities and "
                    "training-receipt hashes are bound above"
                ),
            },
        },
        "comparison_confounds": {
            "comparison_is_nonfactorial": True,
            "large_only_initialization": (
                "preparedness teacher BC plus four DAgger iterations"
            ),
            "combined_initialization": (
                "single-pass offline oracle BC with zero DAgger iterations"
            ),
            "large_only_observation_rms": (
                "independently fit per seed; three distinct bound RMS hashes"
            ),
            "combined_observation_rms": (
                "one shared frozen RMS imported from the distillation run"
            ),
            "critic_warmup_transitions_by_seed": {
                "large_only": {
                    "37017": 50_000,
                    "47017": 50_000,
                    "57017": 60_000,
                },
                "combined": {
                    "37017": 50_000,
                    "47017": 50_000,
                    "57017": 50_000,
                },
            },
            "causal_increment_of_distilled_initialization_isolated": False,
        },
    }


def _source_identity() -> dict[str, str]:
    paths = (
        "scripts/run_combined_distilled_large_study.py",
        "scripts/run_distilled_ppo_study.py",
        "scripts/run_large_architecture_study.py",
        "scripts/train_oracle_bc_student.py",
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
    return {path: file_sha256(ROOT / path) for path in paths}


def _load_source_student_reference(student_root: Path) -> dict[str, Any]:
    try:
        reference = distilled.load_student_reference(student_root)
    except (distilled.DistilledPPOStudyError, TrainingArtifactError) as exc:
        raise CombinedStudyError(
            "approved source distillation checkpoint failed validation"
        ) from exc
    expected_rms = load_portable_upstream_evidence()["distillation"][
        "source_student"
    ]["observation_rms_sha256"]
    if reference["checkpoint"]["observation_rms_sha256"] != expected_rms:
        raise CombinedStudyError("source distillation observation RMS drifted")
    return reference


def _fit_contract(
    dataset: oracle_bc.OracleDataset,
    upstream: dict[str, Any],
    source_student: dict[str, Any],
) -> dict[str, Any]:
    return {
        **distilled._torch_runtime_binding(),
        "method": "new_large_single_pass_oracle_behavior_cloning",
        "architecture": list(HIDDEN_LAYERS),
        "parameter_counts": capacity.EXPECTED_PARAMETER_COUNTS,
        "policy_seed": FIT_POLICY_SEED,
        "fit_rows": FIT_ROW_COUNT,
        "heldout_rows": HOLDOUT_ROW_COUNT,
        "epochs": BC_EPOCHS,
        "batch_size": BC_BATCH_SIZE,
        "learning_rate": BC_LEARNING_RATE,
        "matched_hand_rule_control": True,
        "identical_initialization_and_minibatches": True,
        "dagger_iterations": 0,
        "ppo_updates": 0,
        "dataset": {
            "receipt_path": str(dataset.receipt_path),
            "receipt_sha256": dataset.receipt_sha256,
            "contract_sha256": dataset.contract_sha256,
            "dataset_index_sha256": dataset.dataset_index_sha256,
            "trajectory_split": dataset.split_contract,
        },
        "normalization": {
            "source": "approved_oracle_distillation_run",
            "source_receipt_sha256": upstream["distillation"]["sha256"],
            "source_checkpoint_manifest_sha256": source_student[
                "checkpoint"
            ]["manifest_sha256"],
            "observation_rms_sha256": source_student["checkpoint"][
                "observation_rms_sha256"
            ],
            "frozen": True,
            "holdout_excluded_from_fit": True,
        },
        "causal_input_contract": {
            "training_split_only": True,
            "student_input": "73_public_causal_observation_channels",
            "student_input_count": OBSERVATION_SIZE,
            "student_output": "22_continuous_action_targets",
            "student_output_count": ACTION_SIZE,
            "student_input_future_tape_visible": False,
            "teacher_target_uses_full_future_tape": True,
            "future_tape_use": "teacher_label_generator_only",
        },
    }


def _validate_fit_runtime(value: Mapping[str, Any]) -> None:
    """Fail closed unless the fit runs under its byte-defining Torch runtime."""

    try:
        distilled._validate_torch_runtime_binding(
            value, label="combined large oracle BC fit"
        )
    except distilled.DistilledPPOStudyError as exc:
        raise CombinedStudyError("large BC Torch runtime binding drifted") from exc


def base_contract(
    dataset: oracle_bc.OracleDataset,
    upstream: dict[str, Any],
    source_student: dict[str, Any],
    *,
    dataset_root: Path,
    source_student_root: Path,
) -> dict[str, Any]:
    sources = _source_identity()
    fit = _fit_contract(dataset, upstream, source_student)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "source_identity": sources,
        "source_identity_sha256": canonical_hash(sources),
        "input_roots": {
            "dataset_root": str(dataset_root),
            "source_student_root": str(source_student_root),
        },
        "upstream_evidence": upstream,
        "large_oracle_bc_fit": fit,
        "ppo_plan": {
            "architecture": list(HIDDEN_LAYERS),
            "learning_rate": LEARNING_RATE,
            "policy_seeds": list(POLICY_SEEDS),
            "active_actor_critic_transitions": ACTIVE_TRANSITIONS,
            "critic_warmup_transitions": FIXED_CRITIC_WARMUP_TRANSITIONS,
            "selection_milestones": list(SELECTION_MILESTONES),
            "fresh_seeded_critics": True,
            "frozen_source_observation_rms": True,
            "reward_profile": "v3_equivalent",
            "final_split_imported_or_used": False,
        },
        "promotion_rule": {
            "all_conditions_required": True,
            "best_dev_solved_count_at_least": PROMOTION_BEST_SOLVES,
            "three_seed_mean_strictly_above": INCUMBENT_FIVE_SEED_MEAN,
            "at_least_two_endpoints_at_or_above": PROMOTION_ENDPOINT_SOLVES,
            "required_endpoint_seed_count": PROMOTION_ENDPOINT_SEED_COUNT,
            "final_evaluation_authorized": False,
        },
    }


def _model_environment_with_source_rms(
    source_bundle: LoadedCheckpointBundle,
) -> Any:
    normalizer = oracle_bc._model_environment()
    apply_normalization_state(normalizer, source_bundle.normalization)
    normalizer.training = False
    normalizer.norm_reward = False
    if train_policy.rms_digest(train_policy.rms_state(normalizer.obs_rms)) != (
        source_bundle.normalization.observation_rms_sha256
    ):
        normalizer.close()
        raise CombinedStudyError("source observation RMS could not be restored")
    return normalizer


def fit_large_student(
    dataset: oracle_bc.OracleDataset,
    source_student: dict[str, Any],
    *,
    fit_contract: Mapping[str, Any],
    checkpoint_loader: Callable[..., LoadedCheckpointBundle] = (
        load_checkpoint_bundle
    ),
) -> oracle_bc.StudentFit:
    """Fit matched large actors while importing only the approved B RMS."""

    _validate_fit_runtime(fit_contract)
    if (
        dataset.observations.shape
        != (oracle_bc.DATASET_ROW_COUNT, OBSERVATION_SIZE)
        or dataset.oracle_targets.shape
        != (oracle_bc.DATASET_ROW_COUNT, ACTION_SIZE)
        or dataset.hand_rule_targets.shape
        != (oracle_bc.DATASET_ROW_COUNT, ACTION_SIZE)
        or dataset.fit_indices.shape != (FIT_ROW_COUNT,)
        or dataset.holdout_indices.shape != (HOLDOUT_ROW_COUNT,)
    ):
        raise CombinedStudyError("large BC fit received a noncanonical dataset")
    try:
        source_bundle = checkpoint_loader(
            Path(source_student["checkpoint"]["root"]),
            algorithm_class=train_policy.InstrumentedPPO,
            device="cpu",
        )
    except (TrainingArtifactError, OSError) as exc:
        raise CombinedStudyError("source BC checkpoint could not be loaded") from exc
    source_rms_sha256 = source_student["checkpoint"][
        "observation_rms_sha256"
    ]
    if source_bundle.normalization.observation_rms_sha256 != source_rms_sha256:
        raise CombinedStudyError("strong-loaded source RMS drifted")
    oracle_normalizer = _model_environment_with_source_rms(source_bundle)
    control_normalizer = _model_environment_with_source_rms(source_bundle)
    try:
        observation_rms = train_policy.rms_state(oracle_normalizer.obs_rms)
        normalized = train_policy.normalize_observations(
            dataset.observations, observation_rms
        )
        train_policy.reset_policy_seed(FIT_POLICY_SEED)
        oracle_model = capacity.build_large_model(
            oracle_normalizer,
            seed=FIT_POLICY_SEED,
            n_steps=250,
            batch_size=250,
            learning_rate=LEARNING_RATE,
            target_kl=0.02,
            ent_coef=0.003,
        )
        train_policy.reset_policy_seed(FIT_POLICY_SEED)
        control_model = capacity.build_large_model(
            control_normalizer,
            seed=FIT_POLICY_SEED,
            n_steps=250,
            batch_size=250,
            learning_rate=LEARNING_RATE,
            target_kl=0.02,
            ent_coef=0.003,
        )
        oracle_actor_before = actor_state_sha256(oracle_model)
        control_actor_before = actor_state_sha256(control_model)
        oracle_critic_before = oracle_bc._parameter_digest(
            oracle_bc._critic_parameters(oracle_model)
        )
        control_critic_before = oracle_bc._parameter_digest(
            oracle_bc._critic_parameters(control_model)
        )
        if (
            oracle_actor_before != control_actor_before
            or oracle_critic_before != control_critic_before
        ):
            raise CombinedStudyError("matched large initialization drifted")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(FIT_POLICY_SEED ^ 0xBC37017)
        permutations = [
            torch.randperm(FIT_ROW_COUNT, generator=generator)
            for _ in range(BC_EPOCHS)
        ]
        all_observations = torch.as_tensor(normalized, dtype=torch.float32)
        fit_indices = torch.as_tensor(
            dataset.fit_indices.copy(), dtype=torch.long
        )
        holdout_indices = torch.as_tensor(
            dataset.holdout_indices.copy(), dtype=torch.long
        )
        fit_observations = all_observations[fit_indices]
        holdout_observations = all_observations[holdout_indices]
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

        def metrics(model: Any, observations: Any, targets: Any) -> dict[str, Any]:
            return oracle_bc._prediction_metrics(model, observations, targets)

        oracle_untrained_fit = metrics(
            oracle_model, fit_observations, oracle_fit_targets
        )
        oracle_untrained_holdout = metrics(
            oracle_model, holdout_observations, oracle_holdout_targets
        )
        control_untrained_fit = metrics(
            control_model, fit_observations, control_fit_targets
        )
        control_untrained_holdout = metrics(
            control_model, holdout_observations, control_holdout_targets
        )
        oracle_final_batch, oracle_critic_before, oracle_critic_after = (
            oracle_bc._fit_actor(
                oracle_model,
                fit_observations,
                oracle_fit_targets,
                permutations,
            )
        )
        control_final_batch, control_critic_before, control_critic_after = (
            oracle_bc._fit_actor(
                control_model,
                fit_observations,
                control_fit_targets,
                permutations,
            )
        )

        def treatment(
            label_source: str,
            model: Any,
            actor_before: str,
            critic_before: str,
            critic_after: str,
            final_batch_mse: float,
            fit_targets: Any,
            holdout_targets: Any,
            untrained_fit: dict[str, Any],
            untrained_holdout: dict[str, Any],
        ) -> dict[str, Any]:
            trained_fit = metrics(model, fit_observations, fit_targets)
            trained_holdout = metrics(
                model, holdout_observations, holdout_targets
            )
            return {
                "label_source": label_source,
                "actor_state_sha256_before": actor_before,
                "actor_state_sha256_after": actor_state_sha256(model),
                "critic_state_sha256_before": critic_before,
                "critic_state_sha256_after": critic_after,
                "critic_unchanged": critic_before == critic_after,
                "final_batch_mse": final_batch_mse,
                "fit": {"untrained": untrained_fit, "trained": trained_fit},
                "heldout": {
                    "untrained": untrained_holdout,
                    "trained": trained_holdout,
                    "relative_mse_improvement": (
                        oracle_bc._relative_mse_improvement(
                            untrained_holdout["mse"], trained_holdout["mse"]
                        )
                    ),
                },
            }

        oracle_report = treatment(
            "privileged_same_budget_cem_oracle",
            oracle_model,
            oracle_actor_before,
            oracle_critic_before,
            oracle_critic_after,
            oracle_final_batch,
            oracle_fit_targets,
            oracle_holdout_targets,
            oracle_untrained_fit,
            oracle_untrained_holdout,
        )
        control_report = treatment(
            "preparedness_teacher_action_public_rule",
            control_model,
            control_actor_before,
            control_critic_before,
            control_critic_after,
            control_final_batch,
            control_fit_targets,
            control_holdout_targets,
            control_untrained_fit,
            control_untrained_holdout,
        )
        if not oracle_report["critic_unchanged"] or not control_report[
            "critic_unchanged"
        ]:
            raise CombinedStudyError("large BC fit changed a critic")
        report = {
            "method": "matched_large_behavior_cloning_only",
            "architecture": list(HIDDEN_LAYERS),
            "parameter_counts": capacity.EXPECTED_PARAMETER_COUNTS,
            "policy_seed": FIT_POLICY_SEED,
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
            "observation_rms_sha256": source_rms_sha256,
            "observation_rms_count": float(observation_rms["count"]),
            "observation_rms_imported_from_distillation_run": True,
            "normalization_frozen": True,
            "holdout_excluded_from_fit": True,
            "causal_input_contract": fit_contract["causal_input_contract"],
        }
        control_normalizer.close()
        return oracle_bc.StudentFit(
            model=oracle_model,
            normalizer=oracle_normalizer,
            report=report,
        )
    except Exception:
        oracle_normalizer.close()
        control_normalizer.close()
        raise


def large_bc_fit_gate(fit: Any) -> dict[str, Any]:
    """Apply the established catastrophic fit gate before any PPO starts."""

    oracle_report = fit.get("oracle_label_student") if isinstance(fit, dict) else None
    heldout = oracle_report.get("heldout") if isinstance(oracle_report, dict) else None
    untrained = heldout.get("untrained") if isinstance(heldout, dict) else None
    trained = heldout.get("trained") if isinstance(heldout, dict) else None
    relative = heldout.get("relative_mse_improvement") if isinstance(heldout, dict) else None
    metrics: list[Any] = []
    if isinstance(untrained, dict) and isinstance(trained, dict):
        metrics = [
            untrained.get("mse"),
            untrained.get("mean_absolute_error"),
            trained.get("mse"),
            trained.get("mean_absolute_error"),
        ]
        for report in (untrained, trained):
            per_dimension = report.get("mean_absolute_error_by_dimension")
            if isinstance(per_dimension, list):
                metrics.extend(per_dimension)
            else:
                metrics.append(None)
    metrics_finite = bool(
        len(metrics) == 4 + 2 * ACTION_SIZE
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in metrics
        )
    )
    actor_changed = bool(
        isinstance(oracle_report, dict)
        and isinstance(oracle_report.get("actor_state_sha256_before"), str)
        and isinstance(oracle_report.get("actor_state_sha256_after"), str)
        and oracle_report.get("actor_state_sha256_after")
        != oracle_report.get("actor_state_sha256_before")
    )
    relative_valid = bool(
        isinstance(relative, (int, float))
        and not isinstance(relative, bool)
        and math.isfinite(float(relative))
    )
    untrained_mse = (
        untrained.get("mse") if isinstance(untrained, dict) else None
    )
    trained_mse = trained.get("mse") if isinstance(trained, dict) else None
    recomputed_relative = (
        oracle_bc._relative_mse_improvement(
            float(untrained_mse), float(trained_mse)
        )
        if isinstance(untrained_mse, (int, float))
        and not isinstance(untrained_mse, bool)
        and isinstance(trained_mse, (int, float))
        and not isinstance(trained_mse, bool)
        else None
    )
    relative_consistent = bool(
        relative_valid
        and recomputed_relative is not None
        and float(relative) == recomputed_relative
    )
    improvement_passed = bool(
        relative_consistent
        and recomputed_relative
        > oracle_bc.HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR
    )
    passed = metrics_finite and actor_changed and improvement_passed
    return {
        "kind": "catastrophic_oracle_action_fit_gate",
        "established_floor_source": "train_oracle_bc_student.py",
        "conditions": {
            "heldout_oracle_action_metrics_finite": {
                "passed": metrics_finite,
            },
            "oracle_actor_changed_during_fit": {
                "before": (
                    oracle_report.get("actor_state_sha256_before")
                    if isinstance(oracle_report, dict)
                    else None
                ),
                "after": (
                    oracle_report.get("actor_state_sha256_after")
                    if isinstance(oracle_report, dict)
                    else None
                ),
                "passed": actor_changed,
            },
            "heldout_oracle_relative_mse_improvement": {
                "operator": ">",
                "threshold": oracle_bc.HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR,
                "observed": float(relative) if relative_valid else None,
                "recomputed_from_heldout_mse": recomputed_relative,
                "reported_matches_recomputed": relative_consistent,
                "passed": improvement_passed,
            },
        },
        "passed": passed,
        "decision": "eligible_for_ppo" if passed else "abort_before_ppo",
        "ppo_started": False,
        "development_evaluated": False,
        "final_split_imported_or_used": False,
    }


def _persist_large_fit(
    output_root: Path,
    contract: dict[str, Any],
    dataset: oracle_bc.OracleDataset,
    source_student: dict[str, Any],
) -> dict[str, Any] | None:
    fit_root = output_root / "large-oracle-bc"
    fit_root.mkdir(parents=True, exist_ok=False)
    _atomic_create_json(
        fit_root / "fit.claim.json",
        {
            "created_at_utc": _utc_now(),
            "base_contract_sha256": canonical_hash(contract),
            "fit_contract": contract["large_oracle_bc_fit"],
            "development_evaluated": False,
            "final_split_imported_or_used": False,
        },
    )
    _validate_fit_runtime(contract["large_oracle_bc_fit"])
    fit = fit_large_student(
        dataset,
        source_student,
        fit_contract=contract["large_oracle_bc_fit"],
    )
    try:
        reference = persist_checkpoint_bundle(
            fit_root / "checkpoint",
            model=fit.model,
            normalizer=fit.normalizer,
            training_config={
                "tool": TOOL_ID,
                "phase": "large_oracle_bc_only",
                "base_contract_sha256": canonical_hash(contract),
                **contract["large_oracle_bc_fit"],
            },
            seed=FIT_POLICY_SEED,
            milestone="large-oracle-bc-only",
            checkpoint_id="large-oracle-bc-heldout-seed-67017",
            active_actor_critic_transitions=0,
        )
    finally:
        fit.normalizer.close()
    gate = large_bc_fit_gate(fit.report)
    success = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": (
            "complete_large_oracle_bc_fit_eligible_for_ppo"
            if gate["passed"]
            else "complete_large_oracle_bc_fit_aborted_before_ppo"
        ),
        "created_at_utc": _utc_now(),
        "base_contract_sha256": canonical_hash(contract),
        "fit": fit.report,
        "causal_input_contract": contract["large_oracle_bc_fit"][
            "causal_input_contract"
        ],
        "catastrophic_fit_gate": gate,
        "checkpoint_bundle": reference,
        "ppo_started": False,
        "development_evaluated": False,
        "final_split_imported_or_used": False,
    }
    _atomic_create_json(fit_root / "fit.success.json", success)
    return load_large_fit_reference(output_root, contract) if gate["passed"] else None


def load_large_fit_gate(
    output_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate the terminal fit decision without assuming PPO eligibility."""

    success = _load_json(
        output_root / "large-oracle-bc/fit.success.json",
        "large oracle BC fit success",
    )
    gate = success.get("catastrophic_fit_gate")
    if (
        success.get("schema_version") != SCHEMA_VERSION
        or success.get("tool") != TOOL_ID
        or success.get("base_contract_sha256") != canonical_hash(contract)
        or success.get("causal_input_contract")
        != contract.get("large_oracle_bc_fit", {}).get(
            "causal_input_contract"
        )
        or success.get("ppo_started") is not False
        or success.get("development_evaluated") is not False
        or success.get("final_split_imported_or_used") is not False
        or not isinstance(gate, dict)
        or gate != large_bc_fit_gate(success.get("fit"))
        or success.get("status")
        != (
            "complete_large_oracle_bc_fit_eligible_for_ppo"
            if gate.get("passed") is True
            else "complete_large_oracle_bc_fit_aborted_before_ppo"
        )
    ):
        raise CombinedStudyError("large oracle BC terminal fit gate drifted")
    return gate


def _validate_large_fit_report(
    fit: Any,
    fit_contract: Mapping[str, Any],
    *,
    checkpoint_actor_sha256: Any,
    normalization_rms_sha256: Any,
) -> None:
    """Recompute the semantic fit/bundle binding from persisted evidence."""

    if not isinstance(fit, dict):
        raise CombinedStudyError("large oracle BC fit report is missing")
    oracle_report = fit.get("oracle_label_student")
    control_report = fit.get("matched_hand_rule_control")
    matched = fit.get("matched_initialization")
    trajectory_split = fit_contract.get("dataset", {}).get("trajectory_split")
    expected_rms = fit_contract.get("normalization", {}).get(
        "observation_rms_sha256"
    )
    if (
        not isinstance(oracle_report, dict)
        or not isinstance(control_report, dict)
        or not isinstance(matched, dict)
        or fit.get("method") != "matched_large_behavior_cloning_only"
        or fit.get("architecture") != list(HIDDEN_LAYERS)
        or fit.get("parameter_counts") != capacity.EXPECTED_PARAMETER_COUNTS
        or fit.get("policy_seed") != FIT_POLICY_SEED
        or fit.get("epochs") != BC_EPOCHS
        or fit.get("batch_size") != BC_BATCH_SIZE
        or fit.get("learning_rate") != BC_LEARNING_RATE
        or fit.get("training_row_count_per_student") != FIT_ROW_COUNT
        or fit.get("heldout_row_count_per_student") != HOLDOUT_ROW_COUNT
        or fit.get("trajectory_split") != trajectory_split
        or fit.get("dagger_iterations") != 0
        or fit.get("ppo_updates") != 0
        or fit.get("active_actor_critic_transitions") != 0
        or fit.get("observation_rms_sha256") != expected_rms
        or normalization_rms_sha256 != expected_rms
        or fit.get("observation_rms_imported_from_distillation_run") is not True
        or fit.get("normalization_frozen") is not True
        or fit.get("holdout_excluded_from_fit") is not True
        or fit.get("causal_input_contract")
        != fit_contract.get("causal_input_contract")
        or matched.get("actor_hashes_equal_before_fit") is not True
        or matched.get("critic_hashes_equal_before_fit") is not True
        or matched.get("observation_rms_hashes_equal") is not True
        or matched.get("minibatch_permutations_identical") is not True
        or oracle_report.get("label_source")
        != "privileged_same_budget_cem_oracle"
        or control_report.get("label_source")
        != "preparedness_teacher_action_public_rule"
        or oracle_report.get("critic_unchanged") is not True
        or control_report.get("critic_unchanged") is not True
        or oracle_report.get("critic_state_sha256_before")
        != oracle_report.get("critic_state_sha256_after")
        or control_report.get("critic_state_sha256_before")
        != control_report.get("critic_state_sha256_after")
        or oracle_report.get("actor_state_sha256_before")
        != matched.get("actor_state_sha256")
        or control_report.get("actor_state_sha256_before")
        != matched.get("actor_state_sha256")
        or oracle_report.get("critic_state_sha256_before")
        != matched.get("critic_state_sha256")
        or control_report.get("critic_state_sha256_before")
        != matched.get("critic_state_sha256")
        or checkpoint_actor_sha256
        != oracle_report.get("actor_state_sha256_after")
    ):
        raise CombinedStudyError("large oracle BC fit report contract drifted")


def load_large_fit_reference(
    output_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    fit_root = output_root / "large-oracle-bc"
    fit_contract = contract.get("large_oracle_bc_fit")
    if not isinstance(fit_contract, dict):
        raise CombinedStudyError("large oracle BC fit contract is missing")
    _validate_fit_runtime(fit_contract)
    success_path = fit_root / "fit.success.json"
    success = _load_json(success_path, "large oracle BC fit success")
    if (
        success.get("schema_version") != SCHEMA_VERSION
        or success.get("tool") != TOOL_ID
        or success.get("status")
        != "complete_large_oracle_bc_fit_eligible_for_ppo"
        or success.get("base_contract_sha256") != canonical_hash(contract)
        or success.get("causal_input_contract")
        != fit_contract.get("causal_input_contract")
        or success.get("catastrophic_fit_gate")
        != large_bc_fit_gate(success.get("fit"))
        or success.get("catastrophic_fit_gate", {}).get("passed") is not True
        or success.get("ppo_started") is not False
        or success.get("development_evaluated") is not False
        or success.get("final_split_imported_or_used") is not False
    ):
        raise CombinedStudyError("large oracle BC fit success drifted")
    verified = verify_checkpoint_bundle(fit_root / "checkpoint")
    manifest = verified.manifest
    training = manifest.get("training", {})
    checkpoint = manifest.get("checkpoint", {})
    normalization = manifest.get("normalization", {})
    fit = success.get("fit", {})
    expected_training_config = {
        "tool": TOOL_ID,
        "phase": "large_oracle_bc_only",
        "base_contract_sha256": canonical_hash(contract),
        **fit_contract,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or training.get("seed") != FIT_POLICY_SEED
        or training.get("milestone") != "large-oracle-bc-only"
        or training.get("config") != expected_training_config
        or training.get("config_sha256")
        != canonical_hash(expected_training_config)
        or checkpoint.get("id") != "large-oracle-bc-heldout-seed-67017"
        or checkpoint.get("active_actor_critic_transitions") != 0
        or checkpoint.get("counters", {}).get("num_timesteps") != 0
        or normalization.get("observation_rms_sha256")
        != fit_contract["normalization"]["observation_rms_sha256"]
    ):
        raise CombinedStudyError("large oracle BC checkpoint contract drifted")
    _validate_large_fit_report(
        fit,
        fit_contract,
        checkpoint_actor_sha256=checkpoint.get("actor_state_sha256"),
        normalization_rms_sha256=normalization.get(
            "observation_rms_sha256"
        ),
    )
    recorded = success.get("checkpoint_bundle")
    actual = checkpoint_bundle_reference(fit_root / "checkpoint")
    if recorded != actual:
        raise CombinedStudyError("large oracle BC bundle reference drifted")
    reference = {
        "root": str(verified.root),
        "fit_success": {
            "path": str(success_path),
            "sha256": file_sha256(success_path),
        },
        "checkpoint": {
            **actual,
            "root": str(verified.root),
            "observation_rms_sha256": actual["obs_rms_sha256"],
        },
        "fit_report": fit,
        "dataset_receipt_sha256": contract["large_oracle_bc_fit"]["dataset"][
            "receipt_sha256"
        ],
        "dataset_components": {
            "trajectory_split_sha256": canonical_hash(
                contract["large_oracle_bc_fit"]["dataset"]["trajectory_split"]
            ),
            "dataset_index_sha256": contract["large_oracle_bc_fit"]["dataset"][
                "dataset_index_sha256"
            ],
        },
        "final_split_imported_or_used": False,
    }
    return {**reference, "reference_sha256": canonical_hash(reference)}


def architecture_config() -> dict[str, Any]:
    return {
        "study_tool": TOOL_ID,
        "actor_hidden_layers": list(HIDDEN_LAYERS),
        "critic_hidden_layers": list(HIDDEN_LAYERS),
        "activation": "SiLU",
        "parameter_counts": capacity.EXPECTED_PARAMETER_COUNTS,
        "learning_rate": LEARNING_RATE,
        "registered_selection_milestones": list(SELECTION_MILESTONES),
        "selection_primary_metric": "development_solved_count",
        "resilience_auc_used_for_selection": False,
    }


def combined_config(
    fit_reference: dict[str, Any], upstream: dict[str, Any]
) -> dict[str, Any]:
    return {
        **distilled._torch_runtime_binding(),
        "study_tool": TOOL_ID,
        "initialization_method": "new_large_oracle_distilled_bc_actor",
        "single_pass_behavior_cloning": True,
        "matched_hand_rule_control_fit": True,
        "dagger_iterations": 0,
        "interactive_relabelling": False,
        "distribution_shift_resolved": False,
        "distribution_shift_is_a_separate_confound": True,
        "legacy_bc_or_dagger_dataset_collected_by_ppo_worker": False,
        "distilled_actor_byte_identical_across_seeds": True,
        "fresh_critic_initialized_from_each_policy_seed": True,
        "critic_warmup_transitions_fixed": FIXED_CRITIC_WARMUP_TRANSITIONS,
        "observation_rms_imported_from_distillation_and_frozen": True,
        "architecture": architecture_config(),
        "source_large_bc_fit_reference_sha256": fit_reference[
            "reference_sha256"
        ],
        "source_large_bc_fit_success_sha256": fit_reference["fit_success"][
            "sha256"
        ],
        "source_actor_state_sha256": fit_reference["checkpoint"][
            "actor_state_sha256"
        ],
        "source_observation_rms_sha256": fit_reference["checkpoint"][
            "observation_rms_sha256"
        ],
        "source_dataset_receipt_sha256": fit_reference[
            "dataset_receipt_sha256"
        ],
        "source_distillation_evidence_sha256": upstream["distillation"][
            "sha256"
        ],
        "source_capacity_evidence_sha256": upstream["capacity"]["sha256"],
    }


def seed_directory(output_root: Path, seed: int) -> Path:
    return output_root / f"seed-{seed}"


def trainer_arguments(output_root: Path, seed: int) -> list[str]:
    directory = seed_directory(output_root, seed)
    return [
        "--transitions",
        str(ACTIVE_TRANSITIONS),
        "--lanes",
        "20",
        "--n-steps",
        "250",
        "--batch-size",
        "500",
        "--policy-seed",
        str(seed),
        "--bc-epochs",
        str(BC_EPOCHS),
        "--learning-rate",
        format(LEARNING_RATE, ".12g"),
        "--target-kl",
        "0.02",
        "--ent-coef",
        "0.003",
        "--reward-profile",
        "v3_equivalent",
        "--preparedness-alignment-coefficient",
        "10.0",
        "--bc-warm-start",
        "--vec-normalize",
        "--critic-warmup-min-transitions",
        str(FIXED_CRITIC_WARMUP_TRANSITIONS),
        "--critic-warmup-max-transitions",
        str(FIXED_CRITIC_WARMUP_TRANSITIONS),
        "--freeze-observation-rms",
        "--checkpoint-dir",
        str(directory / "checkpoints"),
        "--json-output",
        str(directory / "training-receipt.json"),
    ]


def _fresh_large_policy_state(seed: int) -> dict[str, torch.Tensor]:
    if seed not in POLICY_SEEDS:
        raise CombinedStudyError(f"unregistered policy seed: {seed}")
    with capacity._inject_large_architecture(ARM):
        return train_policy.untrained_policy_state(
            seed=seed,
            n_steps=250,
            batch_size=500,
            learning_rate=LEARNING_RATE,
            target_kl=0.02,
            ent_coef=0.003,
        )


def fresh_critic_state_sha256(seed: int) -> str:
    state = _fresh_large_policy_state(seed)
    return train_policy.state_digest(distilled._critic_entries(state))


@contextmanager
def inject_combined_initialization(
    fit_reference: dict[str, Any],
    upstream: dict[str, Any],
    *,
    checkpoint_loader: Callable[..., LoadedCheckpointBundle] = (
        load_checkpoint_bundle
    ),
) -> Iterator[None]:
    """Bypass legacy BC/DAgger and inject the large distilled actor + B RMS."""

    try:
        loaded = checkpoint_loader(
            Path(fit_reference["checkpoint"]["root"]),
            algorithm_class=train_policy.InstrumentedPPO,
            device="cpu",
        )
    except (TrainingArtifactError, OSError) as exc:
        raise CombinedStudyError("large BC checkpoint could not be loaded") from exc
    distilled_state = {
        name: value.detach().cpu().clone()
        for name, value in loaded.model.policy.state_dict().items()
    }
    distilled_actor_sha256 = train_policy.state_digest(
        distilled._actor_entries(distilled_state)
    )
    if distilled_actor_sha256 != fit_reference["checkpoint"][
        "actor_state_sha256"
    ]:
        raise CombinedStudyError("large distilled actor hash drifted")
    observation_rms = distilled._normalization_state_for_trainer(loaded)
    if train_policy.rms_digest(observation_rms) != fit_reference["checkpoint"][
        "observation_rms_sha256"
    ]:
        raise CombinedStudyError("large distilled observation RMS drifted")

    original_dataset = train_policy.behavior_cloning_dataset
    original_clone = train_policy.behavior_clone_policy
    original_config = train_policy.resolved_training_config

    def injected_dataset() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
            np.zeros((1, ACTION_SIZE), dtype=np.float32),
        )

    def injected_clone(
        observations: np.ndarray,
        targets: np.ndarray,
        *,
        seed: int,
        n_steps: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        target_kl: float,
        ent_coef: float,
        normalize_observation: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        distilled._canonicalize_torch_runtime()
        if (
            seed not in POLICY_SEEDS
            or n_steps != 250
            or batch_size != 500
            or epochs != BC_EPOCHS
            or learning_rate != LEARNING_RATE
            or target_kl != 0.02
            or ent_coef != 0.003
            or normalize_observation is not True
            or observations.shape != (1, OBSERVATION_SIZE)
            or targets.shape != (1, ACTION_SIZE)
            or observations.dtype != np.float32
            or targets.dtype != np.float32
            or np.any(observations)
            or np.any(targets)
        ):
            raise CombinedStudyError(
                "combined initialization called outside registered config"
            )
        fresh_state = train_policy.untrained_policy_state(
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            target_kl=target_kl,
            ent_coef=ent_coef,
        )
        fresh_critic_sha256 = train_policy.state_digest(
            distilled._critic_entries(fresh_state)
        )
        merged = distilled.merge_distilled_actor(
            fresh_state, distilled_state
        )
        return merged, {
            "mean": observation_rms["mean"].copy(),
            "var": observation_rms["var"].copy(),
            "count": observation_rms["count"],
        }, {
            "teacher": "privileged_same_budget_cem_oracle",
            "training_split_only": True,
            "method": "approved_new_large_single_pass_oracle_bc",
            "dagger_beta_schedule": [],
            "iterations": 1,
            "dagger_iterations": 0,
            "interactive_relabelling": False,
            "legacy_hand_rule_demonstrations_recollected": False,
            "legacy_bc_or_dagger_dataset_collected_by_ppo_worker": False,
            "initialization_placeholder_observation_count": 1,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
            "epochs_per_iteration": BC_EPOCHS,
            "observation_count": FIT_ROW_COUNT,
            "dataset_sha256": canonical_hash(
                fit_reference["dataset_components"]
            ),
            "dataset_component_hashes": fit_reference["dataset_components"],
            "source_dataset_receipt_sha256": fit_reference[
                "dataset_receipt_sha256"
            ],
            "source_large_bc_fit_success_sha256": fit_reference[
                "fit_success"
            ]["sha256"],
            "source_checkpoint_manifest_sha256": fit_reference["checkpoint"][
                "manifest_sha256"
            ],
            "source_checkpoint_model_sha256": fit_reference["checkpoint"][
                "model_sha256"
            ],
            "actor_state_sha256": distilled_actor_sha256,
            "policy_state_sha256": train_policy.state_digest(merged),
            "fresh_critic_state_sha256": fresh_critic_sha256,
            "fresh_critic_policy_seed": seed,
            "critic_imported_from_bc_checkpoint": False,
            "observation_normalization": True,
            "observation_rms_sha256": train_policy.rms_digest(
                observation_rms
            ),
            "observation_rms_count": observation_rms["count"],
            "actor_byte_identical_to_large_bc_checkpoint": True,
            "critic_fresh_for_registered_seed": True,
        }

    def injected_config(
        args: argparse.Namespace,
        *,
        rollout_size: int,
        preparedness_alignment_coefficient: float,
    ) -> dict[str, Any]:
        value = original_config(
            args,
            rollout_size=rollout_size,
            preparedness_alignment_coefficient=(
                preparedness_alignment_coefficient
            ),
        )
        return {
            **value,
            "combined_distillation_capacity_experiment": combined_config(
                fit_reference, upstream
            ),
        }

    train_policy.behavior_cloning_dataset = injected_dataset
    train_policy.behavior_clone_policy = injected_clone
    train_policy.resolved_training_config = injected_config
    try:
        yield
    finally:
        train_policy.behavior_cloning_dataset = original_dataset
        train_policy.behavior_clone_policy = original_clone
        train_policy.resolved_training_config = original_config


def expected_training_config(
    seed: int,
    fit_reference: dict[str, Any],
    upstream: dict[str, Any],
) -> dict[str, Any]:
    args = train_policy.parse_args(trainer_arguments(Path("X:/unused"), seed))
    with capacity._inject_large_architecture(ARM):
        value = train_policy.resolved_training_config(
            args,
            rollout_size=20 * 250,
            preparedness_alignment_coefficient=10.0,
        )
    value["combined_distillation_capacity_experiment"] = combined_config(
        fit_reference, upstream
    )
    return value


def ppo_contract(
    base: dict[str, Any], fit_reference: dict[str, Any]
) -> dict[str, Any]:
    upstream = base["upstream_evidence"]
    configs = {
        str(seed): expected_training_config(seed, fit_reference, upstream)
        for seed in POLICY_SEEDS
    }
    critic_hashes = {
        str(seed): fresh_critic_state_sha256(seed) for seed in POLICY_SEEDS
    }
    if len(set(critic_hashes.values())) != len(POLICY_SEEDS):
        raise CombinedStudyError("registered fresh critics are not distinct")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "base_contract_sha256": canonical_hash(base),
        "upstream_evidence": upstream,
        "fit_reference": fit_reference,
        "registered_training_configs": configs,
        "registered_training_config_sha256_by_seed": {
            seed: canonical_hash(config) for seed, config in configs.items()
        },
        "registered_fresh_critic_state_sha256_by_seed": critic_hashes,
        "policy_seeds": list(POLICY_SEEDS),
        "selection_milestones": list(SELECTION_MILESTONES),
        "final_split_imported_or_used": False,
    }


def worker_command(output_root: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--_worker-seed",
        str(seed),
    ]


def _development_result_with_family_counts(
    value: Any,
    label: str,
    *,
    expected_active_transitions: int,
    expected_total_transitions: int,
) -> dict[str, Any]:
    """Validate canonical rows and derive the five family counts from them."""

    validated = distilled._development_result(
        value,
        label,
        expected_active_transitions=expected_active_transitions,
        expected_total_transitions=expected_total_transitions,
    )
    rows = value.get("rows") if isinstance(value, dict) else None
    expected_family_ids = [family.id for family in distilled.DEVELOPMENT_FAMILIES]
    counts = {family_id: 0 for family_id in expected_family_ids}
    row_counts = {family_id: 0 for family_id in expected_family_ids}
    family_rows: dict[str, list[dict[str, Any]]] = {
        family_id: [] for family_id in expected_family_ids
    }
    if not isinstance(rows, list):
        raise CombinedStudyError(f"{label} canonical rows are missing")
    for index, row in enumerate(rows):
        row_id = row.get("row_id") if isinstance(row, dict) else None
        family_id = row_id.rsplit(":", 1)[0] if isinstance(row_id, str) else None
        if family_id not in counts or not isinstance(row.get("solved"), bool):
            raise CombinedStudyError(f"{label} family row {index} drifted")
        row_counts[family_id] += 1
        counts[family_id] += int(row["solved"])
        family_rows[family_id].append(
            {"row_id": row_id, "solved": row["solved"]}
        )
    if (
        row_counts != {family_id: 40 for family_id in expected_family_ids}
        or sum(counts.values()) != validated["solved_count"]
    ):
        raise CombinedStudyError(f"{label} family aggregates drifted")
    return {
        **validated,
        "per_family_solved_count": counts,
        "per_family_rows_sha256": {
            family_id: canonical_hash(family_rows[family_id])
            for family_id in expected_family_ids
        },
    }


def _validate_training_receipt(
    path: Path,
    seed: int,
    contract: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_json(path, f"combined seed {seed} receipt")
    config = receipt.get("config")
    expected_config = contract["registered_training_configs"][str(seed)]
    behavior = receipt.get("behavior_cloning")
    initialization = receipt.get("initialization")
    normalization = receipt.get("normalization")
    warmup = receipt.get("critic_warmup")
    counts = receipt.get("transition_counts")
    checks = receipt.get("checks")
    milestone_states = receipt.get("milestone_states")
    fit_checkpoint = contract["fit_reference"]["checkpoint"]
    if (
        receipt.get("schema_version") != 1
        or receipt.get("tool") != train_policy.TOOL_ID
        or receipt.get("status") != "complete"
        or receipt.get("training_split") != "train"
        or receipt.get("evaluation_split") != "dev"
        or receipt.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or receipt.get("final_split_used") is not False
        or config != expected_config
        or canonical_hash(config)
        != contract["registered_training_config_sha256_by_seed"][str(seed)]
        or not isinstance(behavior, dict)
        or not isinstance(initialization, dict)
        or not isinstance(normalization, dict)
        or not isinstance(warmup, dict)
        or not isinstance(counts, dict)
        or not isinstance(checks, dict)
        or not isinstance(milestone_states, dict)
        or set(milestone_states) != {str(value) for value in TRAINER_MILESTONES}
        or counts.get("active_actor_critic") != ACTIVE_TRANSITIONS
        or counts.get("critic_warmup") != FIXED_CRITIC_WARMUP_TRANSITIONS
        or counts.get("total_environment")
        != ACTIVE_TRANSITIONS + FIXED_CRITIC_WARMUP_TRANSITIONS
        or any(
            checks.get(name) is not True
            for name in (
                "actor_unchanged_during_critic_warmup",
                "critic_explained_variance_gate_passed",
                "return_rms_continuous_without_reset",
                "observation_rms_contract_preserved",
                "per_iteration_telemetry_complete",
                "development_hard_violations_zero",
                "development_conservation_residuals_zero",
                "development_only_no_final_split_used",
                "training_complete",
                "all_registered_checkpoints_persisted",
            )
        )
    ):
        raise CombinedStudyError(f"combined receipt contract drifted: seed {seed}")
    actor_sha256 = initialization.get("actor_sha256")
    rms_sha256 = initialization.get("observation_rms_sha256")
    fresh_critic = behavior.get("fresh_critic_state_sha256")
    if (
        behavior.get("actor_warm_start_applied") is not True
        or behavior.get("method") != "approved_new_large_single_pass_oracle_bc"
        or behavior.get("teacher") != "privileged_same_budget_cem_oracle"
        or behavior.get("training_split_only") is not True
        or behavior.get("iterations") != 1
        or behavior.get("dagger_iterations") != 0
        or behavior.get("interactive_relabelling") is not False
        or behavior.get("legacy_hand_rule_demonstrations_recollected") is not False
        or behavior.get("legacy_bc_or_dagger_dataset_collected_by_ppo_worker")
        is not False
        or behavior.get("initialization_placeholder_observation_count") != 1
        or behavior.get("observation_count") != FIT_ROW_COUNT
        or behavior.get("epochs_per_iteration") != BC_EPOCHS
        or behavior.get("critic_imported_from_bc_checkpoint") is not False
        or behavior.get("fresh_critic_policy_seed") != seed
        or fresh_critic
        != contract["registered_fresh_critic_state_sha256_by_seed"][str(seed)]
        or actor_sha256 != fit_checkpoint["actor_state_sha256"]
        or rms_sha256 != fit_checkpoint["observation_rms_sha256"]
        or behavior.get("actor_state_sha256") != actor_sha256
        or behavior.get("policy_state_sha256")
        != initialization.get("policy_sha256")
        or behavior.get("observation_rms_sha256") != rms_sha256
        or behavior.get("source_large_bc_fit_success_sha256")
        != contract["fit_reference"]["fit_success"]["sha256"]
        or behavior.get("source_checkpoint_manifest_sha256")
        != fit_checkpoint["manifest_sha256"]
        or behavior.get("source_checkpoint_model_sha256")
        != fit_checkpoint["model_sha256"]
        or behavior.get("actor_byte_identical_to_large_bc_checkpoint") is not True
        or behavior.get("critic_fresh_for_registered_seed") is not True
        or normalization.get("observation_rms_frozen") is not True
        or normalization.get("observation_rms_sha256") != rms_sha256
        or warmup.get("actor_sha256_before") != actor_sha256
        or warmup.get("actor_sha256_after") != actor_sha256
        or warmup.get("actor_parameters_byte_identical") is not True
        or warmup.get("minimum_transitions")
        != FIXED_CRITIC_WARMUP_TRANSITIONS
        or warmup.get("maximum_transitions")
        != FIXED_CRITIC_WARMUP_TRANSITIONS
    ):
        raise CombinedStudyError(
            f"combined initialization binding drifted: seed {seed}"
        )
    if receipt.get("training_roster_and_tapes") != (
        train_policy.training_roster_and_tapes_contract()
    ):
        raise CombinedStudyError("canonical training roster binding drifted")

    curve = receipt.get("development_curve")
    expected_curve = {
        "bc_initialization": (0, 0),
        "post_critic_warmup": (0, FIXED_CRITIC_WARMUP_TRANSITIONS),
        **{
            f"ppo_{milestone}_transitions": (
                milestone,
                FIXED_CRITIC_WARMUP_TRANSITIONS + milestone,
            )
            for milestone in TRAINER_MILESTONES
        },
    }
    if not isinstance(curve, dict) or set(curve) != set(expected_curve):
        raise CombinedStudyError(f"development curve drifted: seed {seed}")
    validated: dict[int, dict[str, Any]] = {}
    all_results: dict[str, dict[str, Any]] = {}
    for key, (active, total) in expected_curve.items():
        result = _development_result_with_family_counts(
            curve.get(key),
            f"combined seed {seed} {key}",
            expected_active_transitions=active,
            expected_total_transitions=total,
        )
        all_results[key] = result
        if active in SELECTION_MILESTONES:
            validated[active] = result
    endpoint = _development_result_with_family_counts(
        receipt.get("development"),
        f"combined seed {seed} endpoint",
        expected_active_transitions=ACTIVE_TRANSITIONS,
        expected_total_transitions=(
            FIXED_CRITIC_WARMUP_TRANSITIONS + ACTIVE_TRANSITIONS
        ),
    )
    if endpoint != all_results[f"ppo_{ACTIVE_TRANSITIONS}_transitions"]:
        raise CombinedStudyError(f"2M endpoint differs from curve: seed {seed}")

    candidates: list[dict[str, Any]] = []
    for milestone in TRAINER_MILESTONES:
        candidate = distilled._bundle_candidate(
            path,
            receipt,
            config,
            seed,
            milestone,
            milestone_states[str(milestone)],
            FIXED_CRITIC_WARMUP_TRANSITIONS + milestone,
            all_results[f"ppo_{milestone}_transitions"],
            bundle_verifier=bundle_verifier,
        )
        if milestone in SELECTION_MILESTONES:
            candidates.append(candidate)
    return receipt, candidates


def rank_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) != len(POLICY_SEEDS) * len(SELECTION_MILESTONES):
        raise CombinedStudyError("combined candidate roster is incomplete")
    return sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -int(candidate["development"]["solved_count"]),
            int(candidate["active_actor_critic_transitions"]),
            int(candidate["policy_seed"]),
        ),
    )


def endpoint_summary(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    endpoints = sorted(
        (
            row
            for row in candidates
            if row["active_actor_critic_transitions"] == ACTIVE_TRANSITIONS
        ),
        key=lambda row: int(row["policy_seed"]),
    )
    if [row["policy_seed"] for row in endpoints] != list(POLICY_SEEDS):
        raise CombinedStudyError("combined endpoint roster drifted")
    solved = [int(row["development"]["solved_count"]) for row in endpoints]
    family_ids = list(
        endpoints[0]["development"]["per_family_solved_count"]
    )
    per_family: dict[str, Any] = {}
    for family_id in family_ids:
        counts = {
            str(row["policy_seed"]): row["development"][
                "per_family_solved_count"
            ][family_id]
            for row in endpoints
        }
        per_family[family_id] = {
            "solved_counts_by_seed": counts,
            "mean_solved_count": fmean(counts.values()),
        }
    paired = []
    for index, seed in enumerate(POLICY_SEEDS):
        paired.append(
            {
                "seed": seed,
                "combined": solved[index],
                "incumbent": INCUMBENT_ENDPOINTS[index],
                "delta_vs_incumbent": solved[index] - INCUMBENT_ENDPOINTS[index],
                "large_only": LARGE_ONLY_ENDPOINTS[index],
                "delta_vs_large_only": solved[index] - LARGE_ONLY_ENDPOINTS[index],
            }
        )
    return {
        "solved_counts_by_seed": {
            str(seed): solved[index] for index, seed in enumerate(POLICY_SEEDS)
        },
        "mean_solved_count": fmean(solved),
        "population_std_solved_count": pstdev(solved),
        "sample_std_solved_count": stdev(solved),
        "seed_count_at_or_above_172": sum(
            value >= PROMOTION_ENDPOINT_SOLVES for value in solved
        ),
        "per_family": per_family,
        "paired_same_seed": paired,
        "mean_delta_vs_incumbent_endpoints": fmean(
            row["delta_vs_incumbent"] for row in paired
        ),
        "mean_delta_vs_large_only_endpoints": fmean(
            row["delta_vs_large_only"] for row in paired
        ),
        "incumbent_five_seed_mean": INCUMBENT_FIVE_SEED_MEAN,
        "mean_delta_vs_incumbent_five_seed_mean": (
            fmean(solved) - INCUMBENT_FIVE_SEED_MEAN
        ),
    }


def promotion_decision(
    best_solved_count: int, endpoints: Mapping[str, Any]
) -> dict[str, Any]:
    best_passed = best_solved_count >= PROMOTION_BEST_SOLVES
    mean_passed = (
        float(endpoints["mean_solved_count"]) > INCUMBENT_FIVE_SEED_MEAN
    )
    consistency_passed = (
        int(endpoints["seed_count_at_or_above_172"])
        >= PROMOTION_ENDPOINT_SEED_COUNT
    )
    passed = best_passed and mean_passed and consistency_passed
    return {
        "all_conditions_required": True,
        "conditions": {
            "best_checkpoint_at_least_183_of_200_dev": {
                "observed": best_solved_count,
                "threshold": PROMOTION_BEST_SOLVES,
                "passed": best_passed,
            },
            "three_seed_2m_mean_strictly_above_171_4": {
                "observed": endpoints["mean_solved_count"],
                "threshold_exclusive": INCUMBENT_FIVE_SEED_MEAN,
                "passed": mean_passed,
            },
            "at_least_two_endpoints_at_or_above_172": {
                "observed": endpoints["seed_count_at_or_above_172"],
                "threshold": PROMOTION_ENDPOINT_SEED_COUNT,
                "passed": consistency_passed,
            },
        },
        "passed": passed,
        "decision": (
            "promotion_candidate_requires_owner_review"
            if passed
            else "complete_not_promoted"
        ),
        "final_evaluation_run_or_authorized": False,
    }


def build_summary(
    output_root: Path,
    contract: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    actor_hashes: set[str] = set()
    rms_hashes: set[str] = set()
    critic_hashes: set[str] = set()
    for seed in POLICY_SEEDS:
        path = seed_directory(output_root, seed) / "training-receipt.json"
        receipt, seed_candidates = _validate_training_receipt(
            path, seed, contract, bundle_verifier=bundle_verifier
        )
        actor_hashes.add(receipt["initialization"]["actor_sha256"])
        rms_hashes.add(receipt["initialization"]["observation_rms_sha256"])
        critic_hashes.add(
            receipt["behavior_cloning"]["fresh_critic_state_sha256"]
        )
        receipts.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": file_sha256(path),
                "training_config_sha256": canonical_hash(receipt["config"]),
            }
        )
        curves.append(
            {
                "seed": seed,
                "rows": [
                    {
                        "active_actor_critic_transitions": row[
                            "active_actor_critic_transitions"
                        ],
                        **row["development"],
                    }
                    for row in seed_candidates
                ],
            }
        )
        candidates.extend(seed_candidates)
    fit_checkpoint = contract["fit_reference"]["checkpoint"]
    if (
        actor_hashes != {fit_checkpoint["actor_state_sha256"]}
        or rms_hashes != {fit_checkpoint["observation_rms_sha256"]}
        or len(critic_hashes) != len(POLICY_SEEDS)
    ):
        raise CombinedStudyError("cross-seed initialization invariant drifted")
    ranked = rank_candidates(candidates)
    best = ranked[0]
    endpoints = endpoint_summary(ranked)
    promotion = promotion_decision(
        int(best["development"]["solved_count"]), endpoints
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": (
            "complete_promotion_candidate_requires_owner_review"
            if promotion["passed"]
            else "complete_not_promoted"
        ),
        "created_at_utc": _utc_now(),
        "ppo_contract_sha256": canonical_hash(contract),
        "split": "dev",
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "final_split_imported_or_used": False,
        "base_contract_sha256": contract["base_contract_sha256"],
        "upstream_evidence": contract["upstream_evidence"],
        "large_oracle_bc_fit": contract["fit_reference"],
        "architecture": architecture_config(),
        "receipts": receipts,
        "cross_seed_initialization": {
            "large_distilled_actor_identical": True,
            "frozen_observation_rms_identical": True,
            "fresh_critic_unique_per_seed": True,
            "actor_state_sha256": next(iter(actor_hashes)),
            "observation_rms_sha256": next(iter(rms_hashes)),
        },
        "development_curves": curves,
        "candidate_count": len(ranked),
        "ranking": {
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "candidates": ranked,
        },
        "best_checkpoint": best,
        "endpoint_summary": endpoints,
        "comparison": {
            "best_vs_shipped_best_of_about_20": {
                "combined": best["development"]["solved_count"],
                "incumbent": SHIPPED_DEVELOPMENT_SOLVED_COUNT,
                "delta": best["development"]["solved_count"]
                - SHIPPED_DEVELOPMENT_SOLVED_COUNT,
            },
            "paired_same_seed_vs_incumbent_and_large_only": endpoints[
                "paired_same_seed"
            ],
            "three_seed_mean_vs_incumbent_five_seed_mean": {
                "combined": endpoints["mean_solved_count"],
                "incumbent": INCUMBENT_FIVE_SEED_MEAN,
                "delta": endpoints[
                    "mean_delta_vs_incumbent_five_seed_mean"
                ],
            },
            "comparison_confounds": contract["upstream_evidence"][
                "comparison_confounds"
            ],
        },
        "promotion": promotion,
        "null_scope": (
            "A non-promotion result applies only to a newly fit large "
            "single-pass oracle-BC actor, large [768,512,256] PPO at 3e-5, "
            "these three seeds, and 2M active transitions. The comparison "
            "against historical large-only is nonfactorial: initialization "
            "teacher/DAgger, normalization, and seed-57017 critic warm-up also "
            "differ. It therefore does not isolate an incremental causal "
            "effect of distillation, does not show that capacity and "
            "initialization can never combine, and does not resolve offline-"
            "policy distribution shift."
        ),
    }


def _create_protocol(output_root: Path, contract: dict[str, Any]) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise CombinedStudyError("--execute requires a new output root")
    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_create_json(
        output_root / "protocol.json",
        {
            "created_at_utc": _utc_now(),
            "contract_sha256": canonical_hash(contract),
            "contract": contract,
        },
    )


def _validate_protocol(output_root: Path, contract: dict[str, Any]) -> None:
    protocol = _load_json(output_root / "protocol.json", "combined protocol")
    if (
        protocol.get("contract_sha256") != canonical_hash(contract)
        or protocol.get("contract") != contract
    ):
        raise CombinedStudyError("combined base protocol drifted")


def _ensure_ppo_protocol(
    output_root: Path, base: dict[str, Any], fit_reference: dict[str, Any]
) -> dict[str, Any]:
    contract = ppo_contract(base, fit_reference)
    path = output_root / "ppo-protocol.json"
    if not path.exists():
        _atomic_create_json(
            path,
            {
                "created_at_utc": _utc_now(),
                "contract_sha256": canonical_hash(contract),
                "contract": contract,
                "ppo_started": False,
            },
        )
    else:
        value = _load_json(path, "combined PPO protocol")
        if (
            value.get("contract_sha256") != canonical_hash(contract)
            or value.get("contract") != contract
            or value.get("ppo_started") is not False
        ):
            raise CombinedStudyError("combined PPO protocol drifted")
    return contract


def _run_worker(output_root: Path, seed: int) -> int:
    if seed not in POLICY_SEEDS:
        raise CombinedStudyError(f"unregistered policy seed: {seed}")
    ppo_protocol = _load_json(
        output_root / "ppo-protocol.json", "combined PPO protocol"
    )
    contract = ppo_protocol.get("contract")
    if (
        not isinstance(contract, dict)
        or ppo_protocol.get("contract_sha256") != canonical_hash(contract)
        or contract.get("tool") != TOOL_ID
        or contract.get("git_commit") != _git_commit()
        or contract.get("policy_seeds") != list(POLICY_SEEDS)
    ):
        raise CombinedStudyError("worker PPO protocol drifted")
    base_protocol = _load_json(
        output_root / "protocol.json", "combined base protocol"
    )
    base = base_protocol.get("contract")
    if (
        not isinstance(base, dict)
        or canonical_hash(base) != contract.get("base_contract_sha256")
        or base.get("source_identity") != _source_identity()
    ):
        raise CombinedStudyError("worker base/source contract drifted")
    fit_reference = load_large_fit_reference(output_root, base)
    if fit_reference != contract.get("fit_reference"):
        raise CombinedStudyError("worker large BC reference drifted")
    upstream = load_portable_upstream_evidence()
    if (
        upstream != base.get("upstream_evidence")
        or upstream != contract.get("upstream_evidence")
    ):
        raise CombinedStudyError("worker upstream evidence drifted")
    expected_config = expected_training_config(seed, fit_reference, upstream)
    if expected_config != contract["registered_training_configs"][str(seed)]:
        raise CombinedStudyError("worker registered config drifted")
    with capacity._inject_large_architecture(ARM):
        with inject_combined_initialization(fit_reference, upstream):
            return distilled._run_train_policy_main(
                trainer_arguments(output_root, seed)
            )


def _run_one_seed(
    output_root: Path, seed: int, contract: dict[str, Any]
) -> None:
    directory = seed_directory(output_root, seed)
    receipt_path = directory / "training-receipt.json"
    if receipt_path.exists():
        _validate_training_receipt(receipt_path, seed, contract)
        print(f"[combined] verified seed {seed}", flush=True)
        return
    if directory.exists() or directory.is_symlink():
        raise CombinedStudyError(
            "partial seed run cannot be retried in place; preserve this root "
            f"and choose a new one: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "trainer.log"
    print(f"[combined] starting seed {seed}", flush=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            worker_command(output_root, seed),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise CombinedStudyError(
            f"trainer failed for seed {seed}; see {log_path}"
        )
    _, candidates = _validate_training_receipt(receipt_path, seed, contract)
    print(
        f"[combined] finished seed {seed}: "
        f"{candidates[-1]['development']['solved_count']}/200",
        flush=True,
    )


def _publish_summary_idempotent(
    path: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        _atomic_create_json(path, summary)
        return summary
    existing = _load_json(path, "combined study summary")
    left = {key: value for key, value in existing.items() if key != "created_at_utc"}
    right = {key: value for key, value in summary.items() if key != "created_at_utc"}
    if left != right:
        raise CombinedStudyError("existing combined summary differs")
    return existing


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--source-student-root",
        type=Path,
        default=DEFAULT_SOURCE_STUDENT_ROOT,
    )
    modes = parser.add_mutually_exclusive_group(required=False)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    parser.add_argument("--_worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    public_modes = sum(
        bool(value)
        for value in (args.preflight, args.execute, args.resume, args.summarize)
    )
    if args._worker_seed is not None:
        if public_modes:
            parser.error("worker mode cannot be combined with a public mode")
    elif public_modes != 1:
        parser.error(
            "one of --preflight, --execute, --resume, or --summarize is required"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = _require_external_root(args.output_root, "--output-root")
    if args._worker_seed is not None:
        return _run_worker(output_root, args._worker_seed)
    dataset_root = _require_external_root(args.dataset_root, "--dataset-root")
    student_root = _require_external_root(
        args.source_student_root, "--source-student-root"
    )
    if (
        output_root == dataset_root
        or output_root in dataset_root.parents
        or dataset_root in output_root.parents
        or output_root == student_root
        or output_root in student_root.parents
        or student_root in output_root.parents
    ):
        raise CombinedStudyError(
            "output, dataset, and source-student roots must be separate"
        )
    upstream = load_portable_upstream_evidence()
    source_student = _load_source_student_reference(student_root)
    dataset = oracle_bc.load_oracle_dataset(dataset_root)
    contract = base_contract(
        dataset,
        upstream,
        source_student,
        dataset_root=dataset_root,
        source_student_root=student_root,
    )
    clean = _worktree_is_clean()
    if args.preflight:
        print(
            json.dumps(
                {
                    "tool": TOOL_ID,
                    "status": "ready" if clean else "blocked_dirty_worktree",
                    "filesystem_written": False,
                    "large_bc_fit_started": False,
                    "ppo_started": False,
                    "architecture": list(HIDDEN_LAYERS),
                    "parameter_counts": capacity.EXPECTED_PARAMETER_COUNTS,
                    "fit_rows": FIT_ROW_COUNT,
                    "heldout_rows": HOLDOUT_ROW_COUNT,
                    "source_observation_rms_sha256": source_student[
                        "checkpoint"
                    ]["observation_rms_sha256"],
                    "seeds": list(POLICY_SEEDS),
                    "learning_rate": LEARNING_RATE,
                    "critic_warmup_transitions": (
                        FIXED_CRITIC_WARMUP_TRANSITIONS
                    ),
                    "base_contract_sha256": canonical_hash(contract),
                    "final_split_imported_or_used": False,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if clean else 3
    if not clean:
        raise CombinedStudyError("refusing experiment from a dirty worktree")
    if args.execute:
        _create_protocol(output_root, contract)
        fit_reference = _persist_large_fit(
            output_root, contract, dataset, source_student
        )
        if fit_reference is None:
            gate = load_large_fit_gate(output_root, contract)
            print(
                json.dumps(
                    {
                        "status": "complete_large_oracle_bc_fit_aborted_before_ppo",
                        "catastrophic_fit_gate": gate,
                        "ppo_protocol_created": False,
                        "development_evaluated": False,
                        "final_split_imported_or_used": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return oracle_bc.COMPLETED_NEGATIVE_GATE_EXIT_CODE
    else:
        if not output_root.is_dir():
            raise CombinedStudyError(
                "--resume/--summarize requires an existing output root"
        )
        _validate_protocol(output_root, contract)
        gate = load_large_fit_gate(output_root, contract)
        if gate.get("passed") is not True:
            print(
                json.dumps(
                    {
                        "status": "complete_large_oracle_bc_fit_aborted_before_ppo",
                        "catastrophic_fit_gate": gate,
                        "ppo_protocol_created": False,
                        "development_evaluated": False,
                        "final_split_imported_or_used": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return oracle_bc.COMPLETED_NEGATIVE_GATE_EXIT_CODE
        fit_reference = load_large_fit_reference(output_root, contract)
    ppo = _ensure_ppo_protocol(output_root, contract, fit_reference)
    if args.execute or args.resume:
        for seed in POLICY_SEEDS:
            _run_one_seed(output_root, seed, ppo)
    summary = build_summary(output_root, ppo)
    summary_path = output_root / "combined-study-summary.json"
    summary = _publish_summary_idempotent(summary_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "best_checkpoint": summary["best_checkpoint"],
                "endpoint_summary": summary["endpoint_summary"],
                "promotion": summary["promotion"],
                "final_split_imported_or_used": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        CombinedStudyError,
        TrainingArtifactError,
        oracle_bc.OracleBCError,
        distilled.DistilledPPOStudyError,
        capacity.ArchitectureStudyError,
    ) as error:
        print(f"combined study failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
