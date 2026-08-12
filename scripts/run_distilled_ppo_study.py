#!/usr/bin/env python3
"""Run the registered development-only PPO continuation of oracle BC.

The approved input is an external, immutable single-pass oracle-distillation
receipt and its BC-only checkpoint.  Each registered PPO seed receives the
same actor bytes and frozen observation moments from that checkpoint, while
the critic is freshly initialized from the registered seed.  The canonical
trainer then owns critic warm-up, PPO, development evaluation, diagnostics,
and durable checkpoint publication.  This module never imports or evaluates
the final split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
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

from backend.app.city.environment import ACTION_ORDER, OBSERVATION_ORDER  # noqa: E402
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
)
from scripts import train_policy  # noqa: E402
from scripts.training_artifacts import (  # noqa: E402
    LoadedCheckpointBundle,
    TrainingArtifactError,
    load_checkpoint_bundle,
    verify_checkpoint_bundle,
)

TOOL_ID = "run_distilled_ppo_study.py"
SCHEMA_VERSION = 1
APPROVED_STUDENT_ROOT = Path(
    r"E:\city-recovery-oracle-bc-student-v4-attempt-01"
)
APPROVED_STUDENT_RECEIPT_SHA256 = (
    "76025a6376db6905b1d96d08122a14bccc7639040921768a79e4c83debabec84"
)
STUDENT_RECEIPT_NAME = "student-receipt.json"
STUDENT_CHECKPOINT_NAME = "bc-checkpoint"
POLICY_SEEDS = (37_017, 47_017, 57_017)
ACTIVE_TRANSITIONS = 2_000_000
REGISTERED_SELECTION_MILESTONES = (500_000, 1_000_000, 2_000_000)
EXPECTED_TRAINER_MILESTONES = (200_000, *REGISTERED_SELECTION_MILESTONES)
DEVELOPMENT_CASE_COUNT = 200
SHIPPED_DEVELOPMENT_SOLVED_COUNT = 178
INCUMBENT_ENDPOINT_MEAN = 171.4
INCUMBENT_ENDPOINT_POPULATION_STD = 1.624807680927192
INCUMBENT_ENDPOINT_SAMPLE_STD = 1.816590212458495
PROMOTION_BEST_SOLVES = 183
PROMOTION_ENDPOINT_SOLVES = 172
PROMOTION_ENDPOINT_SEED_COUNT = 2
CHECKPOINT_REFERENCE_FIELDS = {
    "checkpoint_id",
    "manifest_path",
    "manifest_sha256",
    "model_path",
    "model_sha256",
    "normalization_path",
    "normalization_sha256",
    "policy_state_sha256",
    "actor_state_sha256",
    "optimizer_state_sha256",
    "obs_rms_sha256",
    "ret_rms_sha256",
    "num_timesteps",
    "active_actor_critic_transitions",
}
MILESTONE_STATE_FIELDS = {
    "policy_sha256",
    "actor_sha256",
    "observation_rms_sha256",
    "return_rms_sha256",
    "return_rms_count",
}

DEFAULT_BASELINE_SUMMARY = (
    ROOT / "internal/developmental_runs/v4/training-study-200-summary.json"
)
DEFAULT_SELECTION_RECEIPT = (
    ROOT / "internal/developmental_runs/v4/checkpoint-selection-200.json"
)


class DistilledPPOStudyError(RuntimeError):
    """Raised when the fixed distilled-PPO study contract cannot be honored."""


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
    return load_json_object(path, label, error_type=DistilledPPOStudyError)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DistilledPPOStudyError(f"refusing to overwrite evidence: {path}")
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
        raise DistilledPPOStudyError(
            f"refusing to overwrite evidence: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_external_root(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise DistilledPPOStudyError(f"{label} must be absolute")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise DistilledPPOStudyError(f"{label} must be outside the repository")
    if resolved == Path(resolved.anchor):
        raise DistilledPPOStudyError(f"{label} cannot be a filesystem root")
    return resolved


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DistilledPPOStudyError(f"{label} is not a lowercase SHA-256")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DistilledPPOStudyError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DistilledPPOStudyError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise DistilledPPOStudyError(f"{label} must be finite")
    return result


def _checkpoint_reference_from_student_receipt(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    fit = receipt.get("fit")
    if not isinstance(fit, dict):
        raise DistilledPPOStudyError("student fit receipt is missing")
    checkpoint = fit.get("checkpoint_bundle")
    if not isinstance(checkpoint, dict):
        raise DistilledPPOStudyError("student checkpoint reference is missing")
    return checkpoint


def load_student_reference(
    student_root: Path,
    *,
    expected_receipt_sha256: str = APPROVED_STUDENT_RECEIPT_SHA256,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> dict[str, Any]:
    """Verify the approved BC gate, fit evidence, and checkpoint bundle."""

    root = _require_external_root(student_root, "student root")
    if not root.is_dir() or root.is_symlink():
        raise DistilledPPOStudyError("student root must be a real directory")
    receipt_path = root / STUDENT_RECEIPT_NAME
    actual_receipt_sha256 = file_sha256(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise DistilledPPOStudyError("approved student receipt SHA-256 drifted")
    receipt = _load_json(receipt_path, "approved student receipt")
    gate = receipt.get("catastrophic_gate")
    invariants = receipt.get("invariants")
    fit = receipt.get("fit")
    development = receipt.get("development")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("tool") != "train_oracle_bc_student.py"
        or receipt.get("status")
        != "complete_eligible_for_separately_authorized_3_seed_ppo"
        or receipt.get("completed") is not True
        or receipt.get("ppo_started") is not False
        or receipt.get("development_split_used") is not True
        or receipt.get("final_split_imported_or_used") is not False
        or receipt.get("development_evaluation_count") != 1
        or not isinstance(gate, dict)
        or gate.get("kind") != "catastrophic_only"
        or gate.get("passed") is not True
        or gate.get("decision")
        != "eligible_for_separately_authorized_3_seed_ppo"
        or not isinstance(invariants, dict)
        or any(value is not True for value in invariants.values())
        or not isinstance(fit, dict)
        or not isinstance(development, dict)
        or development.get("case_count") != DEVELOPMENT_CASE_COUNT
        or development.get("solved_count") != 157
        or development.get("hard_violation_count") != 0
        or development.get("maximum_conservation_residual") != 0.0
    ):
        raise DistilledPPOStudyError("approved student receipt contract drifted")
    offline = fit.get("offline_distillation_disclosure")
    fit_detail = fit.get("fit")
    if (
        fit.get("status") != "complete_matched_bc_only_fits"
        or fit.get("final_split_used") is not False
        or not isinstance(offline, dict)
        or offline.get("dagger") is not False
        or offline.get("interactive_relabelling") is not False
        or offline.get("distribution_shift_resolved") is not False
        or offline.get("distribution_shift_is_a_separate_confound") is not True
        or not isinstance(fit_detail, dict)
        or fit_detail.get("method") != "matched_behavior_cloning_only"
        or fit_detail.get("dagger_iterations") != 0
        or fit_detail.get("ppo_updates") != 0
        or fit_detail.get("normalization_frozen") is not True
    ):
        raise DistilledPPOStudyError("student offline-fit disclosure drifted")
    oracle_student = fit_detail.get("oracle_label_student")
    hand_rule_student = fit_detail.get("matched_hand_rule_control")
    if not isinstance(oracle_student, dict) or not isinstance(
        hand_rule_student, dict
    ):
        raise DistilledPPOStudyError("student fit comparison is missing")
    oracle_heldout = oracle_student.get("heldout")
    hand_rule_heldout = hand_rule_student.get("heldout")
    if not isinstance(oracle_heldout, dict) or not isinstance(
        hand_rule_heldout, dict
    ):
        raise DistilledPPOStudyError("student held-out fit evidence is missing")
    oracle_mse = _finite_float(
        oracle_heldout.get("trained", {}).get("mse"), "oracle held-out MSE"
    )
    hand_rule_mse = _finite_float(
        hand_rule_heldout.get("trained", {}).get("mse"),
        "hand-rule held-out MSE",
    )
    if oracle_mse <= 0.0 or hand_rule_mse <= 0.0:
        raise DistilledPPOStudyError("student held-out fit MSE is invalid")

    checkpoint_root = root / STUDENT_CHECKPOINT_NAME
    try:
        verified = bundle_verifier(checkpoint_root)
    except (TrainingArtifactError, OSError) as exc:
        raise DistilledPPOStudyError(
            "approved student checkpoint failed verification"
        ) from exc
    manifest = verified.manifest
    checkpoint = manifest.get("checkpoint", {})
    normalization = manifest.get("normalization", {})
    training = manifest.get("training", {})
    checkpoint_reference = _checkpoint_reference_from_student_receipt(receipt)
    manifest_sha256 = file_sha256(verified.manifest_path)
    if (
        verified.root != checkpoint_root.resolve()
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or checkpoint.get("id") != "oracle-bc-heldout-seed-67017"
        or checkpoint.get("active_actor_critic_transitions") != 0
        or checkpoint.get("counters", {}).get("num_timesteps") != 0
        or training.get("milestone") != "oracle-bc-only"
        or training.get("seed") != 67_017
        or training.get("config", {}).get("critic_trained") is not False
        or training.get("config", {}).get("ppo_updates") != 0
        or training.get("config", {}).get("actor_architecture")
        != [384, 256, 128]
        or normalization.get("norm_obs") is not True
        or normalization.get("norm_reward") is not False
        or normalization.get("training") is not False
        or normalization.get("observation_shape") != [len(OBSERVATION_ORDER)]
        or checkpoint_reference.get("manifest_sha256") != manifest_sha256
        or checkpoint_reference.get("model_sha256")
        != checkpoint.get("file", {}).get("sha256")
        or checkpoint_reference.get("normalization_sha256")
        != normalization.get("file", {}).get("sha256")
        or checkpoint_reference.get("actor_state_sha256")
        != checkpoint.get("actor_state_sha256")
        or checkpoint_reference.get("obs_rms_sha256")
        != normalization.get("observation_rms_sha256")
        or Path(str(checkpoint_reference.get("manifest_path", ""))).resolve()
        != verified.manifest_path
        or Path(str(checkpoint_reference.get("model_path", ""))).resolve()
        != verified.model_path
        or Path(
            str(checkpoint_reference.get("normalization_path", ""))
        ).resolve()
        != verified.normalization_path
    ):
        raise DistilledPPOStudyError("student checkpoint binding drifted")

    actor_sha256 = _sha256_value(
        checkpoint.get("actor_state_sha256"), "student actor state"
    )
    obs_rms_sha256 = _sha256_value(
        normalization.get("observation_rms_sha256"),
        "student observation RMS",
    )
    dataset_receipt_sha256 = _sha256_value(
        fit.get("dataset_receipt_sha256"), "oracle dataset receipt"
    )
    trajectory_split = fit_detail.get("trajectory_split", {})
    fit_split = trajectory_split.get("fit", {})
    dataset_components = {
        "observations_sha256": _sha256_value(
            fit_split.get("observations_sha256"), "fit observations"
        ),
        "oracle_targets_sha256": _sha256_value(
            fit_split.get("oracle_targets_sha256"), "fit oracle targets"
        ),
        "step_row_ids_sha256": _sha256_value(
            fit_split.get("step_row_ids_sha256"), "fit row ids"
        ),
    }
    reference = {
        "root": str(root),
        "student_receipt": {
            "path": str(receipt_path),
            "sha256": actual_receipt_sha256,
        },
        "student_contract_sha256": _sha256_value(
            receipt.get("contract_sha256"), "student contract"
        ),
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "dataset_components": dataset_components,
        "method": "single_pass_offline_oracle_behavior_cloning_no_dagger",
        "distribution_shift_disclosure": {
            "interactive_relabelling": False,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
        },
        "heldout_fit": {
            "oracle_action_mse": oracle_mse,
            "oracle_action_mean_absolute_error": _finite_float(
                oracle_heldout.get("trained", {}).get("mean_absolute_error"),
                "oracle held-out MAE",
            ),
            "hand_rule_action_mse": hand_rule_mse,
            "hand_rule_action_mean_absolute_error": _finite_float(
                hand_rule_heldout.get("trained", {}).get(
                    "mean_absolute_error"
                ),
                "hand-rule held-out MAE",
            ),
        },
        "bc_development_solved_count": 157,
        "catastrophic_gate_passed": True,
        "checkpoint": {
            "root": str(verified.root),
            "manifest_path": str(verified.manifest_path),
            "manifest_sha256": manifest_sha256,
            "model_path": str(verified.model_path),
            "model_sha256": checkpoint["file"]["sha256"],
            "normalization_path": str(verified.normalization_path),
            "normalization_sha256": normalization["file"]["sha256"],
            "policy_state_sha256": checkpoint["policy_state_sha256"],
            "actor_state_sha256": actor_sha256,
            "observation_rms_sha256": obs_rms_sha256,
            "return_rms_sha256": normalization["return_rms_sha256"],
        },
        "final_split_imported_or_used": False,
    }
    return {**reference, "reference_sha256": canonical_hash(reference)}


def _actor_entries(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a policy state dict into the canonical actor-only digest names."""

    actor: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name == "log_std":
            actor[name] = value.detach().cpu().clone()
        elif name.startswith("mlp_extractor.policy_net."):
            actor[name.removeprefix("mlp_extractor.")] = (
                value.detach().cpu().clone()
            )
        elif name.startswith("action_net."):
            actor[name] = value.detach().cpu().clone()
    if not actor or "log_std" not in actor:
        raise DistilledPPOStudyError("policy state has no canonical actor")
    return actor


def _critic_entries(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    critic = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if name.startswith("mlp_extractor.value_net.")
        or name.startswith("value_net.")
    }
    if not critic:
        raise DistilledPPOStudyError("policy state has no canonical critic")
    return critic


def merge_distilled_actor(
    fresh_policy_state: Mapping[str, torch.Tensor],
    distilled_policy_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a fresh critic plus a byte-identical distilled actor."""

    fresh = {
        name: value.detach().cpu().clone()
        for name, value in fresh_policy_state.items()
    }
    distilled_actor_names = {
        name
        for name in distilled_policy_state
        if name == "log_std"
        or name.startswith("mlp_extractor.policy_net.")
        or name.startswith("action_net.")
    }
    fresh_actor_names = {
        name
        for name in fresh
        if name == "log_std"
        or name.startswith("mlp_extractor.policy_net.")
        or name.startswith("action_net.")
    }
    if distilled_actor_names != fresh_actor_names:
        raise DistilledPPOStudyError("distilled and fresh actor schemas differ")
    critic_before = train_policy.state_digest(_critic_entries(fresh))
    for name in sorted(distilled_actor_names):
        source = distilled_policy_state[name].detach().cpu()
        if source.shape != fresh[name].shape or source.dtype != fresh[name].dtype:
            raise DistilledPPOStudyError(
                f"distilled actor tensor contract drifted: {name}"
            )
        fresh[name] = source.clone()
    if train_policy.state_digest(_critic_entries(fresh)) != critic_before:
        raise DistilledPPOStudyError("actor transplant changed the fresh critic")
    expected_actor = train_policy.state_digest(
        _actor_entries(distilled_policy_state)
    )
    if train_policy.state_digest(_actor_entries(fresh)) != expected_actor:
        raise DistilledPPOStudyError("distilled actor transplant was not exact")
    return fresh


def _normalization_state_for_trainer(
    loaded: LoadedCheckpointBundle,
) -> dict[str, Any]:
    normalization = loaded.normalization
    return {
        "mean": np.asarray(normalization.obs_mean, dtype=np.float64).copy(),
        "var": np.asarray(normalization.obs_var, dtype=np.float64).copy(),
        "count": float(normalization.obs_count),
    }


def distillation_config(student_reference: dict[str, Any]) -> dict[str, Any]:
    checkpoint = student_reference["checkpoint"]
    return {
        "study_tool": TOOL_ID,
        "initialization_method": "approved_external_oracle_bc_actor",
        "single_pass_behavior_cloning": True,
        "dagger_iterations": 0,
        "interactive_relabelling": False,
        "legacy_hand_rule_demonstrations_recollected": False,
        "legacy_bc_or_dagger_dataset_collected_by_ppo_worker": False,
        "ppo_worker_initialization_input": "one_row_deterministic_placeholder",
        "distribution_shift_is_a_separate_confound": True,
        "distilled_actor_byte_identical_across_seeds": True,
        "fresh_critic_initialized_from_each_policy_seed": True,
        "policy_rng_reset_before_ppo_environment_and_model_construction": True,
        "observation_rms_imported_and_frozen": True,
        "source_student_receipt_sha256": student_reference[
            "student_receipt"
        ]["sha256"],
        "source_student_reference_sha256": student_reference[
            "reference_sha256"
        ],
        "source_checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "source_checkpoint_model_sha256": checkpoint["model_sha256"],
        "source_actor_state_sha256": checkpoint["actor_state_sha256"],
        "source_observation_rms_sha256": checkpoint[
            "observation_rms_sha256"
        ],
        "source_dataset_receipt_sha256": student_reference[
            "dataset_receipt_sha256"
        ],
        "source_evidence_hash_independent_of_training_config_hash": True,
    }


def fresh_critic_state_sha256(seed: int) -> str:
    """Rebuild and hash the canonical random critic for one PPO seed."""

    if seed not in POLICY_SEEDS:
        raise DistilledPPOStudyError(f"unregistered policy seed: {seed}")
    fresh_state = train_policy.untrained_policy_state(
        seed=seed,
        n_steps=250,
        batch_size=500,
        learning_rate=7.5e-5,
        target_kl=0.02,
        ent_coef=0.003,
    )
    return train_policy.state_digest(_critic_entries(fresh_state))


@contextmanager
def inject_distilled_initialization(
    student_reference: dict[str, Any],
    *,
    checkpoint_loader: Callable[..., LoadedCheckpointBundle] = (
        load_checkpoint_bundle
    ),
) -> Iterator[None]:
    """Route canonical BC initialization to the approved external student."""

    checkpoint_root = Path(student_reference["checkpoint"]["root"])
    try:
        loaded = checkpoint_loader(
            checkpoint_root,
            algorithm_class=train_policy.InstrumentedPPO,
            device="cpu",
        )
    except (TrainingArtifactError, OSError) as exc:
        raise DistilledPPOStudyError(
            "approved student checkpoint could not be strongly loaded"
        ) from exc
    distilled_state = {
        name: value.detach().cpu().clone()
        for name, value in loaded.model.policy.state_dict().items()
    }
    distilled_actor_sha256 = train_policy.state_digest(
        _actor_entries(distilled_state)
    )
    expected_actor_sha256 = student_reference["checkpoint"][
        "actor_state_sha256"
    ]
    if distilled_actor_sha256 != expected_actor_sha256:
        raise DistilledPPOStudyError("strong-loaded distilled actor hash drifted")
    observation_rms = _normalization_state_for_trainer(loaded)
    if (
        train_policy.rms_digest(observation_rms)
        != student_reference["checkpoint"]["observation_rms_sha256"]
    ):
        raise DistilledPPOStudyError("strong-loaded observation RMS hash drifted")

    original_behavior_clone = train_policy.behavior_clone_policy
    original_behavior_dataset = train_policy.behavior_cloning_dataset
    original_resolved_config = train_policy.resolved_training_config

    def injected_behavior_dataset() -> tuple[np.ndarray, np.ndarray]:
        """Satisfy the trainer call site without recollecting hand-rule data."""

        return (
            np.zeros((1, len(OBSERVATION_ORDER)), dtype=np.float32),
            np.zeros((1, len(ACTION_ORDER)), dtype=np.float32),
        )

    def injected_behavior_clone(
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
        if (
            seed not in POLICY_SEEDS
            or epochs != 15
            or not normalize_observation
            or n_steps != 250
            or batch_size != 500
            or learning_rate != 7.5e-5
            or target_kl != 0.02
            or ent_coef != 0.003
            or observations.shape != (1, len(OBSERVATION_ORDER))
            or targets.shape != (1, len(ACTION_ORDER))
            or observations.dtype != np.float32
            or targets.dtype != np.float32
            or np.any(observations)
            or np.any(targets)
        ):
            raise DistilledPPOStudyError(
                "distilled initialization called outside registered config"
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
            _critic_entries(fresh_state)
        )
        independently_rebuilt_critic_sha256 = fresh_critic_state_sha256(seed)
        if fresh_critic_sha256 != independently_rebuilt_critic_sha256:
            raise DistilledPPOStudyError(
                "fresh critic does not match the registered seed"
            )
        merged = merge_distilled_actor(fresh_state, distilled_state)
        merged_policy_sha256 = train_policy.state_digest(merged)
        dataset_components = student_reference["dataset_components"]
        return merged, {
            "mean": observation_rms["mean"].copy(),
            "var": observation_rms["var"].copy(),
            "count": observation_rms["count"],
        }, {
            "teacher": "privileged_same_budget_cem_oracle",
            "training_split_only": True,
            "method": "approved_external_single_pass_behavior_cloning",
            "dagger_beta_schedule": [],
            "iterations": 1,
            "dagger_iterations": 0,
            "interactive_relabelling": False,
            "legacy_hand_rule_demonstrations_recollected": False,
            "legacy_bc_or_dagger_dataset_collected_by_ppo_worker": False,
            "initialization_placeholder_observation_count": 1,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
            "epochs_per_iteration": 15,
            "observation_count": 5_040,
            "dataset_sha256": canonical_hash(dataset_components),
            "dataset_component_hashes": dataset_components,
            "source_dataset_receipt_sha256": student_reference[
                "dataset_receipt_sha256"
            ],
            "source_student_receipt_sha256": student_reference[
                "student_receipt"
            ]["sha256"],
            "source_checkpoint_manifest_sha256": student_reference[
                "checkpoint"
            ]["manifest_sha256"],
            "source_checkpoint_model_sha256": student_reference["checkpoint"][
                "model_sha256"
            ],
            "actor_state_sha256": distilled_actor_sha256,
            "policy_state_sha256": merged_policy_sha256,
            "fresh_critic_state_sha256": fresh_critic_sha256,
            "fresh_critic_policy_seed": seed,
            "critic_imported_from_bc_checkpoint": False,
            "observation_normalization": True,
            "observation_rms_sha256": train_policy.rms_digest(
                observation_rms
            ),
            "observation_rms_count": observation_rms["count"],
            "actor_byte_identical_to_approved_bc_checkpoint": True,
            "critic_fresh_for_registered_seed": True,
        }

    def injected_config(
        args: argparse.Namespace,
        *,
        rollout_size: int,
        preparedness_alignment_coefficient: float,
    ) -> dict[str, Any]:
        value = original_resolved_config(
            args,
            rollout_size=rollout_size,
            preparedness_alignment_coefficient=(
                preparedness_alignment_coefficient
            ),
        )
        return {
            **value,
            "distillation_experiment": distillation_config(
                student_reference
            ),
        }

    train_policy.behavior_cloning_dataset = injected_behavior_dataset
    train_policy.behavior_clone_policy = injected_behavior_clone
    train_policy.resolved_training_config = injected_config
    try:
        yield
    finally:
        train_policy.behavior_cloning_dataset = original_behavior_dataset
        train_policy.behavior_clone_policy = original_behavior_clone
        train_policy.resolved_training_config = original_resolved_config


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
        "15",
        "--learning-rate",
        "7.5e-5",
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
        "50000",
        "--critic-warmup-max-transitions",
        "100000",
        "--freeze-observation-rms",
        "--checkpoint-dir",
        str(directory / "checkpoints"),
        "--json-output",
        str(directory / "training-receipt.json"),
    ]


def worker_command(output_root: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--_worker-seed",
        str(seed),
    ]


def expected_training_config(
    seed: int, student_reference: dict[str, Any]
) -> dict[str, Any]:
    args = train_policy.parse_args(trainer_arguments(Path("X:/unused"), seed))
    value = train_policy.resolved_training_config(
        args,
        rollout_size=20 * 250,
        preparedness_alignment_coefficient=10.0,
    )
    return {
        **value,
        "distillation_experiment": distillation_config(student_reference),
    }


def _expected_development_identity() -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape(scenario, tape_seed)
            identities.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "tape_sha256": canonical_hash(
                        [asdict(shock) for shock in schedule]
                    ),
                }
            )
    if (
        len(DEVELOPMENT_FAMILIES) != 5
        or DEVELOPMENT_SEEDS != tuple(range(820000, 820040))
        or len(identities) != DEVELOPMENT_CASE_COUNT
        or len({row["row_id"] for row in identities})
        != DEVELOPMENT_CASE_COUNT
    ):
        raise DistilledPPOStudyError(
            "canonical development roster must remain ordered 5 x 40"
        )
    return identities


def _development_result(
    value: Any,
    label: str,
    *,
    expected_active_transitions: int,
    expected_total_transitions: int | None = None,
) -> dict[str, Any]:
    """Recompute one result over the exact ordered canonical DEV roster."""

    if not isinstance(value, dict):
        raise DistilledPPOStudyError(f"{label} is not an object")
    solved_count = value.get("solved_count")
    rows = value.get("rows")
    solve_rate = _finite_float(value.get("solve_rate"), f"{label} solve rate")
    maximum_residual = _finite_float(
        value.get("maximum_conservation_residual"),
        f"{label} conservation residual",
    )
    mean_resilience_auc = _finite_float(
        value.get("mean_resilience_auc"), f"{label} resilience AUC"
    )
    mean_tail_margin = _finite_float(
        value.get("mean_minimum_tail_margin"), f"{label} tail margin"
    )
    total_transitions = value.get("total_environment_transitions")
    if (
        value.get("case_count") != DEVELOPMENT_CASE_COUNT
        or value.get("active_actor_critic_transitions")
        != expected_active_transitions
        or not isinstance(total_transitions, int)
        or isinstance(total_transitions, bool)
        or total_transitions < expected_active_transitions
        or (
            expected_total_transitions is not None
            and total_transitions != expected_total_transitions
        )
        or not isinstance(solved_count, int)
        or isinstance(solved_count, bool)
        or not 0 <= solved_count <= DEVELOPMENT_CASE_COUNT
        or not math.isclose(
            solve_rate,
            solved_count / DEVELOPMENT_CASE_COUNT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or value.get("hard_violation_count") != 0
        or maximum_residual != 0.0
        or not isinstance(rows, list)
        or len(rows) != DEVELOPMENT_CASE_COUNT
    ):
        raise DistilledPPOStudyError(f"{label} aggregate drifted")

    expected_identity = _expected_development_identity()
    recomputed_solved = 0
    recomputed_hard_violations = 0
    recomputed_maximum_residual = 0.0
    resilience_values: list[float] = []
    tail_values: list[float] = []
    failure_reasons: Counter[str] = Counter()
    for index, (row, expected) in enumerate(
        zip(rows, expected_identity, strict=True)
    ):
        if not isinstance(row, dict):
            raise DistilledPPOStudyError(f"{label} row {index} drifted")
        row_residual = _finite_float(
            row.get("max_conservation_residual"),
            f"{label} row {index} conservation residual",
        )
        row_resilience = _finite_float(
            row.get("resilience_auc"),
            f"{label} row {index} resilience AUC",
        )
        row_tail = _finite_float(
            row.get("minimum_tail_margin"),
            f"{label} row {index} tail margin",
        )
        hard_violations = row.get("hard_violation_count")
        reason_codes = row.get("reason_codes")
        if (
            any(row.get(key) != expected_value for key, expected_value in expected.items())
            or not isinstance(row.get("solved"), bool)
            or not isinstance(hard_violations, int)
            or isinstance(hard_violations, bool)
            or hard_violations < 0
            or row_residual < 0.0
            or not isinstance(reason_codes, list)
            or any(not isinstance(reason, str) or not reason for reason in reason_codes)
        ):
            raise DistilledPPOStudyError(f"{label} row {index} drifted")
        recomputed_solved += int(row["solved"])
        recomputed_hard_violations += hard_violations
        recomputed_maximum_residual = max(
            recomputed_maximum_residual, row_residual
        )
        resilience_values.append(row_resilience)
        tail_values.append(row_tail)
        if not row["solved"]:
            failure_reasons.update(reason_codes)
    if (
        recomputed_solved != solved_count
        or recomputed_hard_violations != value.get("hard_violation_count")
        or recomputed_maximum_residual != maximum_residual
        or round(fmean(resilience_values), 10) != mean_resilience_auc
        or round(fmean(tail_values), 10) != mean_tail_margin
        or dict(sorted(failure_reasons.items()))
        != value.get("failure_reason_code_histogram")
        or recomputed_hard_violations != 0
        or recomputed_maximum_residual != 0.0
    ):
        raise DistilledPPOStudyError(f"{label} rows disagree with aggregate")
    return {
        "active_actor_critic_transitions": expected_active_transitions,
        "total_environment_transitions": total_transitions,
        "case_count": DEVELOPMENT_CASE_COUNT,
        "solved_count": solved_count,
        "solve_rate": solve_rate,
        "mean_resilience_auc": mean_resilience_auc,
        "mean_minimum_tail_margin": mean_tail_margin,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "failure_reason_code_histogram": dict(sorted(failure_reasons.items())),
        "ordered_case_identity_sha256": canonical_hash(expected_identity),
        "rows_sha256": canonical_hash(rows),
    }


def load_incumbent_reference(
    summary_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    """Bind the recorded 178 best-of-20 and 171.4 five-seed mean."""

    summary_path = summary_path.resolve()
    selection_path = selection_path.resolve()
    summary = _load_json(summary_path, "incumbent training summary")
    selection = _load_json(selection_path, "incumbent selection receipt")
    scope = summary.get("scope")
    baseline = summary.get("baseline")
    endpoints = baseline.get("endpoints") if isinstance(baseline, dict) else None
    if (
        summary.get("kind") != "city-recovery-training-study-200-summary"
        or not isinstance(scope, dict)
        or scope.get("split") != "dev"
        or scope.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or scope.get("final_split_used") is not False
        or not isinstance(baseline, dict)
        or baseline.get("name") != "adopted_v3_equivalent_2m"
        or not isinstance(endpoints, list)
        or len(endpoints) != 5
        or selection.get("split") != "dev"
        or selection.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or selection.get("final_split_used") is not False
        or selection.get("candidate_count") != 20
        or selection.get("winner", {}).get("solved_count")
        != SHIPPED_DEVELOPMENT_SOLVED_COUNT
        or selection.get("ranking", {}).get(
            "resilience_auc_used_for_selection"
        )
        is not False
    ):
        raise DistilledPPOStudyError("incumbent evidence contract drifted")
    seeds = [int(row.get("seed", -1)) for row in endpoints]
    solved = [int(row.get("solved_count", -1)) for row in endpoints]
    if (
        seeds != [37_017, 47_017, 57_017, 67_017, 77_017]
        or not math.isclose(
            fmean(solved), INCUMBENT_ENDPOINT_MEAN, abs_tol=1e-12
        )
        or not math.isclose(
            pstdev(solved), INCUMBENT_ENDPOINT_POPULATION_STD, abs_tol=1e-12
        )
        or not math.isclose(
            stdev(solved), INCUMBENT_ENDPOINT_SAMPLE_STD, abs_tol=1e-12
        )
        or not math.isclose(
            float(baseline.get("aggregate", {}).get("mean_solved_count", -1)),
            INCUMBENT_ENDPOINT_MEAN,
            abs_tol=1e-12,
        )
    ):
        raise DistilledPPOStudyError("incumbent endpoint statistics drifted")
    return {
        "training_summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "selection_receipt": {
            "path": str(selection_path),
            "sha256": file_sha256(selection_path),
        },
        "best_of_20_development_solved_count": 178,
        "five_seed_2m_endpoints": {
            "solved_counts": solved,
            "mean": INCUMBENT_ENDPOINT_MEAN,
            "population_std": INCUMBENT_ENDPOINT_POPULATION_STD,
            "sample_std": INCUMBENT_ENDPOINT_SAMPLE_STD,
        },
        "comparison_framings": {
            "promotion_decisive": "conjunctive_best_margin_and_seed_consistency",
            "fair_seed_level": "challenger_3_seed_mean_vs_incumbent_5_seed_mean",
        },
        "final_split_imported_or_used": False,
    }


def source_identity() -> dict[str, str]:
    paths = (
        "scripts/run_distilled_ppo_study.py",
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


def study_contract(
    student_reference: dict[str, Any],
    incumbent_reference: dict[str, Any],
) -> dict[str, Any]:
    sources = source_identity()
    configs = {
        str(seed): expected_training_config(seed, student_reference)
        for seed in POLICY_SEEDS
    }
    config_hashes = {
        seed: canonical_hash(config) for seed, config in configs.items()
    }
    fresh_critic_hashes = {
        str(seed): fresh_critic_state_sha256(seed) for seed in POLICY_SEEDS
    }
    if len(set(fresh_critic_hashes.values())) != len(POLICY_SEEDS):
        raise DistilledPPOStudyError(
            "registered policy seeds did not produce distinct fresh critics"
        )
    if len(OBSERVATION_ORDER) != 73 or len(ACTION_ORDER) != 22:
        raise DistilledPPOStudyError("public 73-in/22-out interface drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "source_identity": sources,
        "source_identity_sha256": canonical_hash(sources),
        "registered_training_configs": configs,
        "registered_training_config_sha256_by_seed": config_hashes,
        "registered_fresh_critic_state_sha256_by_seed": fresh_critic_hashes,
        "source_and_config_hashes_are_independent": True,
        "approved_student_reference": student_reference,
        "incumbent_reference": incumbent_reference,
        "scope": {
            "training_split_used": True,
            "development_split_used": True,
            "development_case_count": DEVELOPMENT_CASE_COUNT,
            "final_split_imported_or_used": False,
            "public_policy_interface": {
                "input_name": "observation",
                "input_shape": ["batch", 73],
                "input_dtype": "float32",
                "output_name": "action",
                "output_shape": ["batch", 22],
                "output_dtype": "float32",
                "onnx_export_path_unchanged": True,
            },
        },
        "registered_policy_seeds": list(POLICY_SEEDS),
        "training": {
            "active_actor_critic_transitions": ACTIVE_TRANSITIONS,
            "learning_curve_milestones": list(
                REGISTERED_SELECTION_MILESTONES
            ),
            "trainer_diagnostic_milestones": list(EXPECTED_TRAINER_MILESTONES),
            "lanes": 20,
            "n_steps_per_lane": 250,
            "batch_size": 500,
            "learning_rate": 7.5e-5,
            "target_kl": 0.02,
            "ent_coef": 0.003,
            "reward_profile": "v3_equivalent",
            "preparedness_alignment_coefficient": 10.0,
            "critic_warmup_min_transitions": 50_000,
            "critic_warmup_max_transitions": 100_000,
            "same_distilled_actor_across_seeds": True,
            "fresh_seeded_critic_per_seed": True,
            "observation_rms_frozen": True,
        },
        "selection": {
            "split": "dev",
            "primary_metric": "solved_count",
            "resilience_auc_used_for_selection": False,
            "candidate_milestones": list(REGISTERED_SELECTION_MILESTONES),
            "tie_breakers": ["earlier_transition_count", "lower_policy_seed"],
        },
        "promotion_rule": {
            "all_conditions_required": True,
            "best_development_solved_count_at_least": PROMOTION_BEST_SOLVES,
            "three_seed_endpoint_mean_strictly_above": (
                INCUMBENT_ENDPOINT_MEAN
            ),
            "endpoint_seed_count_at_or_above_172_at_least": (
                PROMOTION_ENDPOINT_SEED_COUNT
            ),
            "final_evaluation_authorized": False,
        },
        "null_scope": (
            "A null result applies to this single-pass oracle-BC actor, the "
            "adopted optimizer, and 2M active transitions. It does not resolve "
            "offline-policy distribution shift, and it does not show that "
            "privileged future information is the only possible advantage."
        ),
    }


def _bundle_candidate(
    receipt_path: Path,
    receipt: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    milestone: int,
    milestone_state: dict[str, Any],
    expected_num_timesteps: int,
    development: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any],
) -> dict[str, Any]:
    reference = receipt.get("checkpoint_bundles", {}).get(str(milestone))
    if not isinstance(reference, dict):
        raise DistilledPPOStudyError(
            f"checkpoint reference is missing: seed {seed}/{milestone}"
        )
    if set(reference) != CHECKPOINT_REFERENCE_FIELDS:
        raise DistilledPPOStudyError(
            f"checkpoint reference fields drifted: seed {seed}/{milestone}"
        )
    if not isinstance(milestone_state, dict) or set(
        milestone_state
    ) != MILESTONE_STATE_FIELDS:
        raise DistilledPPOStudyError(
            f"milestone state fields drifted: seed {seed}/{milestone}"
        )
    policy_sha256 = _sha256_value(
        milestone_state.get("policy_sha256"),
        f"seed {seed}/{milestone} milestone policy",
    )
    actor_sha256 = _sha256_value(
        milestone_state.get("actor_sha256"),
        f"seed {seed}/{milestone} milestone actor",
    )
    observation_rms_sha256 = _sha256_value(
        milestone_state.get("observation_rms_sha256"),
        f"seed {seed}/{milestone} milestone observation RMS",
    )
    return_rms_sha256 = _sha256_value(
        milestone_state.get("return_rms_sha256"),
        f"seed {seed}/{milestone} milestone return RMS",
    )
    return_rms_count = _finite_float(
        milestone_state.get("return_rms_count"),
        f"seed {seed}/{milestone} milestone return RMS count",
    )
    if not math.isclose(
        return_rms_count,
        expected_num_timesteps + 0.0001,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise DistilledPPOStudyError(
            f"milestone return RMS count drifted: seed {seed}/{milestone}"
        )
    expected_root = (
        seed_directory(receipt_path.parents[1], seed)
        / "checkpoints"
        / f"ppo-{milestone}"
    ).resolve()
    manifest_path = Path(str(reference.get("manifest_path", ""))).resolve()
    if manifest_path.parent != expected_root:
        raise DistilledPPOStudyError("checkpoint path escaped registered root")
    try:
        verified = bundle_verifier(expected_root)
    except (TrainingArtifactError, OSError) as exc:
        raise DistilledPPOStudyError("checkpoint bundle verification failed") from exc
    manifest = verified.manifest
    manifest_training = manifest.get("training", {})
    checkpoint = manifest.get("checkpoint", {})
    normalization = manifest.get("normalization", {})
    counters = checkpoint.get("counters", {})
    manifest_sha256 = file_sha256(verified.manifest_path)
    optimizer_sha256 = _sha256_value(
        reference.get("optimizer_state_sha256"),
        f"seed {seed}/{milestone} optimizer state",
    )
    if (
        verified.root != expected_root
        or verified.manifest_path != expected_root / "manifest.json"
        or manifest_path != verified.manifest_path
        or verified.model_path != expected_root / "model.zip"
        or verified.normalization_path != expected_root / "normalization.npz"
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or manifest_training.get("config") != config
        or manifest_training.get("config_sha256") != canonical_hash(config)
        or manifest_training.get("seed") != seed
        or manifest_training.get("milestone") != milestone
        or checkpoint.get("id") != reference.get("checkpoint_id")
        or checkpoint.get("policy_state_sha256") != policy_sha256
        or checkpoint.get("actor_state_sha256") != actor_sha256
        or checkpoint.get("optimizer_state_sha256") != optimizer_sha256
        or counters.get("num_timesteps") != expected_num_timesteps
        or checkpoint.get("active_actor_critic_transitions") != milestone
        or reference.get("policy_state_sha256") != policy_sha256
        or reference.get("actor_state_sha256") != actor_sha256
        or reference.get("num_timesteps") != expected_num_timesteps
        or reference.get("active_actor_critic_transitions") != milestone
        or reference.get("manifest_sha256") != manifest_sha256
        or Path(str(reference.get("model_path", ""))).resolve()
        != verified.model_path
        or Path(str(reference.get("normalization_path", ""))).resolve()
        != verified.normalization_path
        or reference.get("model_sha256")
        != checkpoint.get("file", {}).get("sha256")
        or reference.get("normalization_sha256")
        != normalization.get("file", {}).get("sha256")
        or reference.get("obs_rms_sha256")
        != observation_rms_sha256
        or normalization.get("observation_rms_sha256")
        != observation_rms_sha256
        or reference.get("ret_rms_sha256") != return_rms_sha256
        or normalization.get("return_rms_sha256") != return_rms_sha256
        or observation_rms_sha256
        != receipt.get("initialization", {}).get("observation_rms_sha256")
    ):
        raise DistilledPPOStudyError(
            f"checkpoint bundle binding drifted: seed {seed}/{milestone}"
        )
    return {
        "id": reference["checkpoint_id"],
        "policy_seed": seed,
        "active_actor_critic_transitions": milestone,
        "num_timesteps": expected_num_timesteps,
        "policy_state_sha256": policy_sha256,
        "actor_state_sha256": actor_sha256,
        "optimizer_state_sha256": optimizer_sha256,
        "development": development,
        "training_receipt_path": str(receipt_path),
        "training_receipt_sha256": file_sha256(receipt_path),
        "training_config_sha256": canonical_hash(config),
        "bundle_path": str(verified.root),
        "bundle_manifest_path": str(verified.manifest_path),
        "bundle_manifest_sha256": manifest_sha256,
        "checkpoint_path": str(verified.model_path),
        "checkpoint_sha256": checkpoint["file"]["sha256"],
        "normalization_path": str(verified.normalization_path),
        "normalization_file_sha256": normalization["file"]["sha256"],
        "observation_rms_sha256": observation_rms_sha256,
        "return_rms_sha256": return_rms_sha256,
        "return_rms_count": return_rms_count,
    }


def validate_training_receipt(
    path: Path,
    seed: int,
    student_reference: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one complete canonical run and its ordered DEV bundles."""

    if seed not in POLICY_SEEDS:
        raise DistilledPPOStudyError(f"unregistered policy seed: {seed}")
    receipt = _load_json(path, f"distilled PPO seed {seed} receipt")
    config = receipt.get("config")
    expected_config = expected_training_config(seed, student_reference)
    checks = receipt.get("checks")
    transition_counts = receipt.get("transition_counts")
    behavior = receipt.get("behavior_cloning")
    initialization = receipt.get("initialization")
    normalization = receipt.get("normalization")
    critic_warmup = receipt.get("critic_warmup")
    milestone_states = receipt.get("milestone_states")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("tool") != train_policy.TOOL_ID
        or receipt.get("status") != "complete"
        or receipt.get("training_split") != "train"
        or receipt.get("evaluation_split") != "dev"
        or receipt.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or receipt.get("final_split_used") is not False
        or config != expected_config
        or not isinstance(checks, dict)
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
        or not isinstance(transition_counts, dict)
        or transition_counts.get("active_actor_critic") != ACTIVE_TRANSITIONS
        or transition_counts.get("critic_warmup") not in (50_000, 55_000, 60_000, 65_000, 70_000, 75_000, 80_000, 85_000, 90_000, 95_000, 100_000)
        or transition_counts.get("total_environment")
        != transition_counts.get("critic_warmup") + ACTIVE_TRANSITIONS
        or not isinstance(behavior, dict)
        or not isinstance(initialization, dict)
        or not isinstance(normalization, dict)
        or not isinstance(critic_warmup, dict)
        or not isinstance(milestone_states, dict)
        or set(milestone_states)
        != {str(milestone) for milestone in EXPECTED_TRAINER_MILESTONES}
    ):
        raise DistilledPPOStudyError(
            f"distilled PPO receipt contract drifted: seed {seed}"
        )
    source_checkpoint = student_reference["checkpoint"]
    actor_sha256 = _sha256_value(
        initialization.get("actor_sha256"), "initial actor"
    )
    policy_sha256 = _sha256_value(
        initialization.get("policy_sha256"), "initial policy"
    )
    observation_rms_sha256 = _sha256_value(
        initialization.get("observation_rms_sha256"), "initial observation RMS"
    )
    fresh_critic_sha256 = _sha256_value(
        behavior.get("fresh_critic_state_sha256"), "fresh critic"
    )
    if (
        behavior.get("actor_warm_start_applied") is not True
        or behavior.get("teacher") != "privileged_same_budget_cem_oracle"
        or behavior.get("training_split_only") is not True
        or behavior.get("method")
        != "approved_external_single_pass_behavior_cloning"
        or behavior.get("dagger_iterations") != 0
        or behavior.get("interactive_relabelling") is not False
        or behavior.get("legacy_hand_rule_demonstrations_recollected")
        is not False
        or behavior.get("legacy_bc_or_dagger_dataset_collected_by_ppo_worker")
        is not False
        or behavior.get("initialization_placeholder_observation_count") != 1
        or behavior.get("distribution_shift_resolved") is not False
        or behavior.get("distribution_shift_is_a_separate_confound") is not True
        or behavior.get("critic_imported_from_bc_checkpoint") is not False
        or behavior.get("fresh_critic_policy_seed") != seed
        or fresh_critic_sha256 != fresh_critic_state_sha256(seed)
        or behavior.get("actor_state_sha256") != actor_sha256
        or behavior.get("policy_state_sha256") != policy_sha256
        or behavior.get("observation_rms_sha256") != observation_rms_sha256
        or behavior.get("source_student_receipt_sha256")
        != student_reference["student_receipt"]["sha256"]
        or behavior.get("source_checkpoint_manifest_sha256")
        != source_checkpoint["manifest_sha256"]
        or behavior.get("source_checkpoint_model_sha256")
        != source_checkpoint["model_sha256"]
        or actor_sha256 != source_checkpoint["actor_state_sha256"]
        or observation_rms_sha256
        != source_checkpoint["observation_rms_sha256"]
        or normalization.get("observation_rms_frozen") is not True
        or normalization.get("observation_rms_sha256")
        != observation_rms_sha256
        or critic_warmup.get("actor_sha256_before") != actor_sha256
        or critic_warmup.get("actor_sha256_after") != actor_sha256
        or critic_warmup.get("actor_parameters_byte_identical") is not True
        or critic_warmup.get("minimum_transitions") != 50_000
        or critic_warmup.get("maximum_transitions") != 100_000
    ):
        raise DistilledPPOStudyError(
            f"distilled initialization binding drifted: seed {seed}"
        )

    training_roster = receipt.get("training_roster_and_tapes")
    if training_roster != train_policy.training_roster_and_tapes_contract():
        raise DistilledPPOStudyError("canonical training roster binding drifted")

    curve = receipt.get("development_curve")
    if not isinstance(curve, dict):
        raise DistilledPPOStudyError("development curve is missing")
    warmup = int(transition_counts["critic_warmup"])
    expected_curve = {
        "bc_initialization": (0, 0),
        "post_critic_warmup": (0, warmup),
        **{
            f"ppo_{milestone}_transitions": (milestone, warmup + milestone)
            for milestone in EXPECTED_TRAINER_MILESTONES
        },
    }
    if set(curve) != set(expected_curve):
        raise DistilledPPOStudyError("development curve milestones drifted")
    validated_curve: dict[int, dict[str, Any]] = {}
    all_validated: dict[str, dict[str, Any]] = {}
    for key, (active, total) in expected_curve.items():
        result = _development_result(
            curve.get(key),
            f"seed {seed} {key}",
            expected_active_transitions=active,
            expected_total_transitions=total,
        )
        all_validated[key] = result
        if active in REGISTERED_SELECTION_MILESTONES:
            validated_curve[active] = result
    if _development_result(
        receipt.get("development"),
        f"seed {seed} endpoint",
        expected_active_transitions=ACTIVE_TRANSITIONS,
        expected_total_transitions=warmup + ACTIVE_TRANSITIONS,
    ) != all_validated[f"ppo_{ACTIVE_TRANSITIONS}_transitions"]:
        raise DistilledPPOStudyError("endpoint differs from 2M curve")

    candidates: list[dict[str, Any]] = []
    for milestone in EXPECTED_TRAINER_MILESTONES:
        development = all_validated[f"ppo_{milestone}_transitions"]
        candidate = _bundle_candidate(
            path,
            receipt,
            config,
            seed,
            milestone,
            milestone_states[str(milestone)],
            warmup + milestone,
            development,
            bundle_verifier=bundle_verifier,
        )
        if milestone in REGISTERED_SELECTION_MILESTONES:
            candidates.append(candidate)
    return receipt, candidates


def rank_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = len(POLICY_SEEDS) * len(REGISTERED_SELECTION_MILESTONES)
    if len(candidates) != expected:
        raise DistilledPPOStudyError(f"expected {expected} candidates")
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
        raise DistilledPPOStudyError("endpoint seed roster drifted")
    solved = [int(row["development"]["solved_count"]) for row in endpoints]
    return {
        "rows": [
            {
                "seed": row["policy_seed"],
                "solved_count": row["development"]["solved_count"],
                "solve_rate": row["development"]["solve_rate"],
                "mean_resilience_auc": row["development"][
                    "mean_resilience_auc"
                ],
                "training_receipt_sha256": row[
                    "training_receipt_sha256"
                ],
                "training_config_sha256": row["training_config_sha256"],
            }
            for row in endpoints
        ],
        "mean_solved_count": fmean(solved),
        "population_std_solved_count": pstdev(solved),
        "sample_std_solved_count": stdev(solved),
        "seed_count_at_or_above_172": sum(
            count >= PROMOTION_ENDPOINT_SOLVES for count in solved
        ),
        "incumbent_five_seed_mean_solved_count": INCUMBENT_ENDPOINT_MEAN,
        "mean_delta_vs_incumbent": fmean(solved) - INCUMBENT_ENDPOINT_MEAN,
    }


def promotion_decision(
    selected: dict[str, Any], endpoints: dict[str, Any]
) -> dict[str, Any]:
    best_condition = (
        int(selected["development"]["solved_count"])
        >= PROMOTION_BEST_SOLVES
    )
    mean_condition = (
        float(endpoints["mean_solved_count"]) > INCUMBENT_ENDPOINT_MEAN
    )
    consistency_condition = (
        int(endpoints["seed_count_at_or_above_172"])
        >= PROMOTION_ENDPOINT_SEED_COUNT
    )
    passed = best_condition and mean_condition and consistency_condition
    return {
        "all_conditions_required": True,
        "conditions": {
            "best_checkpoint_at_least_183_of_200_dev": {
                "observed": selected["development"]["solved_count"],
                "threshold": PROMOTION_BEST_SOLVES,
                "passed": best_condition,
            },
            "three_seed_2m_mean_above_incumbent_171_4": {
                "observed": endpoints["mean_solved_count"],
                "threshold_exclusive": INCUMBENT_ENDPOINT_MEAN,
                "passed": mean_condition,
            },
            "at_least_two_of_three_2m_endpoints_at_or_above_172": {
                "observed": endpoints["seed_count_at_or_above_172"],
                "threshold": PROMOTION_ENDPOINT_SEED_COUNT,
                "passed": consistency_condition,
            },
        },
        "passed": passed,
        "decision": "promotion_candidate_requires_owner_review" if passed else "complete_not_promoted",
        "final_evaluation_run_or_authorized": False,
        "resilience_auc_used": False,
    }


def build_summary(
    output_root: Path,
    contract: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> dict[str, Any]:
    student_reference = contract["approved_student_reference"]
    candidates: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    actor_hashes: set[str] = set()
    rms_hashes: set[str] = set()
    critic_hashes: set[str] = set()
    curves: list[dict[str, Any]] = []
    for seed in POLICY_SEEDS:
        path = seed_directory(output_root, seed) / "training-receipt.json"
        receipt, seed_candidates = validate_training_receipt(
            path,
            seed,
            student_reference,
            bundle_verifier=bundle_verifier,
        )
        actor_hashes.add(receipt["initialization"]["actor_sha256"])
        rms_hashes.add(receipt["initialization"]["observation_rms_sha256"])
        critic_hashes.add(
            receipt["behavior_cloning"]["fresh_critic_state_sha256"]
        )
        if (
            receipt["behavior_cloning"]["fresh_critic_state_sha256"]
            != contract["registered_fresh_critic_state_sha256_by_seed"][
                str(seed)
            ]
        ):
            raise DistilledPPOStudyError(
                "fresh critic differs from registered seed identity"
            )
        config_sha256 = canonical_hash(receipt["config"])
        if (
            config_sha256
            != contract["registered_training_config_sha256_by_seed"][str(seed)]
        ):
            raise DistilledPPOStudyError("registered config hash drifted")
        receipt_rows.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": file_sha256(path),
                "config_sha256": config_sha256,
                "actor_state_sha256": receipt["initialization"][
                    "actor_sha256"
                ],
                "fresh_critic_state_sha256": receipt["behavior_cloning"][
                    "fresh_critic_state_sha256"
                ],
                "observation_rms_sha256": receipt["initialization"][
                    "observation_rms_sha256"
                ],
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
    if (
        actor_hashes != {student_reference["checkpoint"]["actor_state_sha256"]}
        or rms_hashes
        != {student_reference["checkpoint"]["observation_rms_sha256"]}
        or len(critic_hashes) != len(POLICY_SEEDS)
    ):
        raise DistilledPPOStudyError(
            "cross-seed actor/RMS/fresh-critic invariant drifted"
        )
    ranked = rank_candidates(candidates)
    selected = ranked[0]
    endpoints = endpoint_summary(ranked)
    promotion = promotion_decision(selected, endpoints)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": (
            "complete_promotion_candidate_requires_owner_review"
            if promotion["passed"]
            else "complete_not_promoted"
        ),
        "created_at_utc": _utc_now(),
        "contract_sha256": canonical_hash(contract),
        "source_identity_sha256": contract["source_identity_sha256"],
        "registered_training_config_sha256_by_seed": contract[
            "registered_training_config_sha256_by_seed"
        ],
        "source_and_config_hashes_are_independent": True,
        "split": "dev",
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "final_split_imported_or_used": False,
        "approved_student_reference": student_reference,
        "incumbent_reference": contract["incumbent_reference"],
        "receipts": receipt_rows,
        "cross_seed_initialization_checks": {
            "distilled_actor_byte_identical_across_all_seeds": True,
            "observation_rms_byte_identical_and_frozen_across_all_seeds": True,
            "fresh_critic_hash_unique_for_every_policy_seed": True,
            "actor_state_sha256": next(iter(actor_hashes)),
            "observation_rms_sha256": next(iter(rms_hashes)),
            "fresh_critic_state_sha256_by_seed": {
                str(row["seed"]): row["fresh_critic_state_sha256"]
                for row in receipt_rows
            },
        },
        "learning_curves": curves,
        "candidate_count": len(ranked),
        "ranking": {
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "candidates": ranked,
        },
        "best_checkpoint": selected,
        "endpoint_summary": endpoints,
        "comparison": {
            "best_of_registered_challenger_vs_incumbent_best_of_20": {
                "challenger": selected["development"]["solved_count"],
                "incumbent": SHIPPED_DEVELOPMENT_SOLVED_COUNT,
                "delta": selected["development"]["solved_count"]
                - SHIPPED_DEVELOPMENT_SOLVED_COUNT,
            },
            "challenger_three_seed_mean_vs_incumbent_five_seed_mean": {
                "challenger": endpoints["mean_solved_count"],
                "incumbent": INCUMBENT_ENDPOINT_MEAN,
                "delta": endpoints["mean_delta_vs_incumbent"],
                "fairer_seed_level_comparison": True,
            },
            "decisive_framing": "preregistered_conjunctive_promotion_rule",
        },
        "promotion": promotion,
        "null_scope": contract["null_scope"],
    }


def _validate_protocol(output_root: Path, contract: dict[str, Any]) -> None:
    protocol = _load_json(output_root / "protocol.json", "study protocol")
    if (
        protocol.get("contract_sha256") != canonical_hash(contract)
        or protocol.get("contract") != contract
    ):
        raise DistilledPPOStudyError(
            "existing study protocol differs from the current contract"
        )


def _create_study_protocol(
    output_root: Path, contract: dict[str, Any]
) -> Path:
    if output_root.exists():
        raise DistilledPPOStudyError("study protocol requires a new output root")
    path = output_root / "protocol.json"
    _atomic_create_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "phase": "preregistered_protocol",
            "created_at_utc": _utc_now(),
            "contract_sha256": canonical_hash(contract),
            "contract": contract,
        },
    )
    return path


def _publish_summary_idempotent(
    path: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        _atomic_create_json(path, summary)
        return summary
    existing = _load_json(path, "distilled PPO study summary")
    existing_comparable = {
        key: value for key, value in existing.items() if key != "created_at_utc"
    }
    summary_comparable = {
        key: value for key, value in summary.items() if key != "created_at_utc"
    }
    if existing_comparable != summary_comparable:
        raise DistilledPPOStudyError(
            "existing summary differs from recomputation"
        )
    return existing


def _run_one_seed(
    output_root: Path,
    seed: int,
    student_reference: dict[str, Any],
) -> None:
    directory = seed_directory(output_root, seed)
    receipt_path = directory / "training-receipt.json"
    if receipt_path.exists():
        validate_training_receipt(receipt_path, seed, student_reference)
        print(f"[distilled-ppo] verified seed {seed}", flush=True)
        return
    if directory.exists():
        raise DistilledPPOStudyError(
            "partial run cannot be retried in place; preserve it and choose a "
            f"new study root: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "trainer.log"
    print(f"[distilled-ppo] starting seed {seed}", flush=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            worker_command(output_root, seed),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise DistilledPPOStudyError(
            f"trainer failed for seed {seed}; see {log_path}"
        )
    _, candidates = validate_training_receipt(
        receipt_path, seed, student_reference
    )
    print(
        f"[distilled-ppo] finished seed {seed}: "
        f"{candidates[-1]['development']['solved_count']}/200",
        flush=True,
    )


def _run_worker(output_root: Path, seed: int) -> int:
    if seed not in POLICY_SEEDS:
        raise DistilledPPOStudyError(f"unregistered policy seed: {seed}")
    protocol = _load_json(output_root / "protocol.json", "study protocol")
    contract = protocol.get("contract")
    if (
        not isinstance(contract, dict)
        or protocol.get("contract_sha256") != canonical_hash(contract)
        or contract.get("tool") != TOOL_ID
        or contract.get("git_commit") != _git_commit()
        or contract.get("source_identity") != source_identity()
        or contract.get("source_identity_sha256")
        != canonical_hash(source_identity())
        or contract.get("registered_policy_seeds") != list(POLICY_SEEDS)
        or contract.get("registered_fresh_critic_state_sha256_by_seed", {}).get(
            str(seed)
        )
        != fresh_critic_state_sha256(seed)
    ):
        raise DistilledPPOStudyError("worker protocol or source identity drifted")
    student_reference = load_student_reference(APPROVED_STUDENT_ROOT)
    if student_reference != contract.get("approved_student_reference"):
        raise DistilledPPOStudyError("worker student evidence binding drifted")
    expected_config = expected_training_config(seed, student_reference)
    if (
        contract.get("registered_training_configs", {}).get(str(seed))
        != expected_config
        or contract.get("registered_training_config_sha256_by_seed", {}).get(
            str(seed)
        )
        != canonical_hash(expected_config)
    ):
        raise DistilledPPOStudyError("worker registered config binding drifted")
    train_policy.reset_policy_seed(seed)
    with inject_distilled_initialization(student_reference):
        return train_policy.main(trainer_arguments(output_root, seed))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--student-root",
        type=Path,
        default=APPROVED_STUDENT_ROOT,
        help="must resolve to the explicitly approved external BC root",
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    parser.add_argument(
        "--selection-receipt", type=Path, default=DEFAULT_SELECTION_RECEIPT
    )
    modes = parser.add_mutually_exclusive_group()
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
    output_root = _require_external_root(args.output_root, "output root")
    approved_root = APPROVED_STUDENT_ROOT.resolve()
    supplied_student_root = _require_external_root(
        args.student_root, "student root"
    )
    if supplied_student_root != approved_root:
        raise DistilledPPOStudyError(
            "--student-root must be the explicitly approved external root"
        )
    if args._worker_seed is not None:
        return _run_worker(output_root, args._worker_seed)

    student_reference = load_student_reference(supplied_student_root)
    incumbent_reference = load_incumbent_reference(
        args.baseline_summary, args.selection_receipt
    )
    contract = study_contract(student_reference, incumbent_reference)
    clean = _worktree_is_clean()
    if args.preflight:
        print(
            json.dumps(
                {
                    "tool": TOOL_ID,
                    "status": "ready" if clean else "blocked_dirty_worktree",
                    "filesystem_written": False,
                    "training_started": False,
                    "student_reference": student_reference,
                    "seeds": list(POLICY_SEEDS),
                    "active_actor_critic_transitions": ACTIVE_TRANSITIONS,
                    "registered_curve_milestones": list(
                        REGISTERED_SELECTION_MILESTONES
                    ),
                    "registered_run_count": len(POLICY_SEEDS),
                    "registered_checkpoint_candidate_count": len(POLICY_SEEDS)
                    * len(REGISTERED_SELECTION_MILESTONES),
                    "incumbent_reference": incumbent_reference,
                    "promotion_rule": contract["promotion_rule"],
                    "source_identity_sha256": contract[
                        "source_identity_sha256"
                    ],
                    "registered_training_config_sha256_by_seed": contract[
                        "registered_training_config_sha256_by_seed"
                    ],
                    "registered_fresh_critic_state_sha256_by_seed": contract[
                        "registered_fresh_critic_state_sha256_by_seed"
                    ],
                    "contract_sha256": canonical_hash(contract),
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
        raise DistilledPPOStudyError(
            "refusing training or summary publication from a dirty worktree"
        )

    if args.execute:
        _create_study_protocol(output_root, contract)
    else:
        if not output_root.is_dir():
            raise DistilledPPOStudyError(
                "--resume/--summarize requires an existing output root"
            )
        _validate_protocol(output_root, contract)
    if args.execute or args.resume:
        for seed in POLICY_SEEDS:
            _run_one_seed(output_root, seed, student_reference)

    summary = build_summary(output_root, contract)
    summary_path = output_root / "distilled-ppo-study-summary.json"
    summary = _publish_summary_idempotent(summary_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "learning_curves": summary["learning_curves"],
                "best_checkpoint": summary["best_checkpoint"],
                "endpoint_summary": summary["endpoint_summary"],
                "comparison": summary["comparison"],
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
    except DistilledPPOStudyError as exc:
        print(f"distilled PPO study failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
