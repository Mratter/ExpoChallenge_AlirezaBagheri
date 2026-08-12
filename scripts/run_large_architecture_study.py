#!/usr/bin/env python3
"""Run the preregistered development-only large-network PPO study.

This orchestration layer changes only the actor/critic hidden widths supplied
to the canonical trainer.  It preserves that trainer's BC/DAgger, frozen-actor
critic warm-up, PPO, normalization, fixed development evaluations, and durable
checkpoint primitives.  Two learning rates are registered over three seeds;
selection and promotion use solved counts only.  This module never imports or
accesses the final split.
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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Callable

# Match the canonical trainer's import-time native-library caps.  The worker is
# a fresh process, so these take effect before Torch and SB3 are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import torch  # noqa: E402
from stable_baselines3.common.vec_env import VecEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import ACTION_SIZE, OBSERVATION_SIZE  # noqa: E402
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
from scripts.training_artifacts import verify_checkpoint_bundle  # noqa: E402

TOOL_ID = "run_large_architecture_study.py"
SCHEMA_VERSION = 1
DEVELOPMENT_CASE_COUNT = 200
POLICY_SEEDS = (37_017, 47_017, 57_017)
HIDDEN_LAYERS = (768, 512, 256)
ACTIVE_TRANSITIONS = 2_000_000
REGISTERED_SELECTION_MILESTONES = (500_000, 1_000_000, 2_000_000)
EXPECTED_TRAINER_MILESTONES = (200_000, *REGISTERED_SELECTION_MILESTONES)
SHIPPED_DEVELOPMENT_SOLVED_COUNT = 178
BASELINE_ENDPOINT_MEAN = 171.4
BASELINE_ENDPOINT_POPULATION_STD = 1.624807680927192
BASELINE_ENDPOINT_SAMPLE_STD = 1.816590212458495
PROMOTION_SELECTED_SOLVES = 183
PROMOTION_ENDPOINT_SOLVES = 172
PROMOTION_ENDPOINT_SEED_COUNT = 2

DEFAULT_BASELINE_SUMMARY = (
    ROOT / "internal/developmental_runs/v4/training-study-200-summary.json"
)
DEFAULT_SELECTION_RECEIPT = (
    ROOT / "internal/developmental_runs/v4/checkpoint-selection-200.json"
)


class ArchitectureStudyError(RuntimeError):
    """Raised when the fixed architecture study contract cannot be honored."""


@dataclass(frozen=True)
class LearningRateArm:
    """One registered optimizer treatment for the shared large network."""

    id: str
    learning_rate: float


REGISTERED_ARMS = (
    LearningRateArm("large_lr_7_5e_5", 7.5e-5),
    LearningRateArm("large_lr_3e_5", 3.0e-5),
)


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
    return load_json_object(
        path, label, error_type=ArchitectureStudyError
    )


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArchitectureStudyError(f"refusing to overwrite evidence: {path}")
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
        raise ArchitectureStudyError(
            f"refusing to overwrite evidence: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_external_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ArchitectureStudyError("--output-root must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ArchitectureStudyError("--output-root must be outside the repository")
    if resolved == Path(resolved.anchor):
        raise ArchitectureStudyError("--output-root cannot be a filesystem root")
    return resolved


def parameter_counts(
    hidden_layers: Sequence[int] = HIDDEN_LAYERS,
) -> dict[str, int]:
    """Return exact SB3 MLP actor, critic, and policy parameter counts."""

    layers = tuple(int(width) for width in hidden_layers)
    if not layers or any(width <= 0 for width in layers):
        raise ArchitectureStudyError("hidden layers must be positive")

    def dense_stack(input_size: int) -> int:
        total = 0
        previous = input_size
        for width in layers:
            total += previous * width + width
            previous = width
        return total

    trunk = dense_stack(OBSERVATION_SIZE)
    actor = trunk + layers[-1] * ACTION_SIZE + ACTION_SIZE + ACTION_SIZE
    critic = trunk + layers[-1] + 1
    return {
        "actor": actor,
        "critic": critic,
        "total_policy": actor + critic,
    }


EXPECTED_PARAMETER_COUNTS = parameter_counts()


def model_parameter_counts(model: train_policy.InstrumentedPPO) -> dict[str, int]:
    """Count the constructed SB3 policy using the experiment ownership split."""

    actor_parameters = [
        *model.policy.mlp_extractor.policy_net.parameters(),
        *model.policy.action_net.parameters(),
        model.policy.log_std,
    ]
    critic_parameters = [
        *model.policy.mlp_extractor.value_net.parameters(),
        *model.policy.value_net.parameters(),
    ]
    return {
        "actor": sum(parameter.numel() for parameter in actor_parameters),
        "critic": sum(parameter.numel() for parameter in critic_parameters),
        "total_policy": sum(
            parameter.numel() for parameter in model.policy.parameters()
        ),
    }


def build_configurable_model(
    environment: VecEnv,
    *,
    hidden_layers: Sequence[int],
    seed: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float = train_policy.DEFAULT_LEARNING_RATE,
    target_kl: float = train_policy.DEFAULT_TARGET_KL,
    ent_coef: float = train_policy.DEFAULT_ENT_COEF,
) -> train_policy.InstrumentedPPO:
    """Construct canonical PPO while parameterizing only actor/critic widths."""

    layers = tuple(int(width) for width in hidden_layers)
    expected_counts = parameter_counts(layers)

    model = train_policy.InstrumentedPPO(
        "MlpPolicy",
        environment,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=target_kl,
        use_sde=False,
        policy_kwargs={
            "activation_fn": torch.nn.SiLU,
            "net_arch": {
                "pi": list(layers),
                "vf": list(layers),
            },
            "ortho_init": True,
            "log_std_init": -1.5,
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )
    actual = model_parameter_counts(model)
    if actual != expected_counts:
        raise ArchitectureStudyError(
            f"constructed parameter count drifted: {actual}"
        )
    return model


def build_large_model(
    environment: VecEnv,
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float = train_policy.DEFAULT_LEARNING_RATE,
    target_kl: float = train_policy.DEFAULT_TARGET_KL,
    ent_coef: float = train_policy.DEFAULT_ENT_COEF,
) -> train_policy.InstrumentedPPO:
    """Construct the preregistered large instance of the configurable model."""

    return build_configurable_model(
        environment,
        hidden_layers=HIDDEN_LAYERS,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        target_kl=target_kl,
        ent_coef=ent_coef,
    )


def architecture_config(arm: LearningRateArm) -> dict[str, Any]:
    return {
        "study_tool": TOOL_ID,
        "arm_id": arm.id,
        "actor_hidden_layers": list(HIDDEN_LAYERS),
        "critic_hidden_layers": list(HIDDEN_LAYERS),
        "activation": "SiLU",
        "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "registered_selection_milestones": list(
            REGISTERED_SELECTION_MILESTONES
        ),
        "selection_primary_metric": "development_solved_count",
        "resilience_auc_used_for_selection": False,
    }


@contextmanager
def _inject_large_architecture(arm: LearningRateArm) -> Iterator[None]:
    """Temporarily route every trainer model build through the large net."""

    original_build = train_policy.build_model
    original_config = train_policy.resolved_training_config

    def injected_build(
        environment: VecEnv,
        *,
        seed: int,
        n_steps: int,
        batch_size: int,
        learning_rate: float = train_policy.DEFAULT_LEARNING_RATE,
        target_kl: float = train_policy.DEFAULT_TARGET_KL,
        ent_coef: float = train_policy.DEFAULT_ENT_COEF,
    ) -> train_policy.InstrumentedPPO:
        return build_large_model(
            environment,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            target_kl=target_kl,
            ent_coef=ent_coef,
        )

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
        return {**value, "architecture_experiment": architecture_config(arm)}

    train_policy.build_model = injected_build
    train_policy.resolved_training_config = injected_config
    try:
        yield
    finally:
        train_policy.build_model = original_build
        train_policy.resolved_training_config = original_config


def arm_directory(output_root: Path, arm: LearningRateArm, seed: int) -> Path:
    return output_root / arm.id / f"seed-{seed}"


def trainer_arguments(
    output_root: Path, arm: LearningRateArm, seed: int
) -> list[str]:
    """Build the exact adopted trainer arguments for one challenger run."""

    directory = arm_directory(output_root, arm, seed)
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
        format(arm.learning_rate, ".12g"),
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


def worker_command(
    output_root: Path, arm: LearningRateArm, seed: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--_worker-arm",
        arm.id,
        "--_worker-seed",
        str(seed),
    ]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ArchitectureStudyError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArchitectureStudyError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ArchitectureStudyError(f"{label} must be a finite number")
    return result


def _expected_development_identity() -> list[dict[str, Any]]:
    """Recompute the exact ordered 5x40 development roster and tape hashes."""

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
        raise ArchitectureStudyError(
            "canonical development roster must remain ordered 5 x 40"
        )
    return identities


def _development_result(
    value: Any,
    label: str,
    *,
    expected_active_transitions: int,
) -> dict[str, Any]:
    """Recompute one result over the exact ordered canonical DEV roster."""

    if not isinstance(value, dict):
        raise ArchitectureStudyError(f"{label} is not an object")
    solved_count = value.get("solved_count")
    solve_rate = value.get("solve_rate")
    rows = value.get("rows")
    numeric_solve_rate = _finite_float(solve_rate, f"{label} solve rate")
    maximum_residual = _finite_float(
        value.get("maximum_conservation_residual"),
        f"{label} maximum conservation residual",
    )
    resilience_auc = _finite_float(
        value.get("mean_resilience_auc"), f"{label} resilience AUC"
    )
    tail_margin = _finite_float(
        value.get("mean_minimum_tail_margin"), f"{label} tail margin"
    )
    if (
        value.get("case_count") != DEVELOPMENT_CASE_COUNT
        or value.get("active_actor_critic_transitions")
        != expected_active_transitions
        or not isinstance(value.get("total_environment_transitions"), int)
        or isinstance(value.get("total_environment_transitions"), bool)
        or value.get("total_environment_transitions")
        < expected_active_transitions
        or not isinstance(solved_count, int)
        or isinstance(solved_count, bool)
        or not 0 <= solved_count <= DEVELOPMENT_CASE_COUNT
        or not math.isclose(
            numeric_solve_rate,
            solved_count / DEVELOPMENT_CASE_COUNT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or value.get("hard_violation_count") != 0
        or maximum_residual != 0.0
        or not isinstance(rows, list)
        or len(rows) != DEVELOPMENT_CASE_COUNT
    ):
        raise ArchitectureStudyError(f"{label} aggregate drifted")
    expected_identity = _expected_development_identity()
    recomputed_solved = 0
    recomputed_hard_violations = 0
    recomputed_maximum_residual = 0.0
    resilience_values: list[float] = []
    tail_margin_values: list[float] = []
    failure_reasons: Counter[str] = Counter()
    for index, (row, expected) in enumerate(
        zip(rows, expected_identity, strict=True)
    ):
        row_residual = (
            _finite_float(
                row.get("max_conservation_residual"),
                f"{label} row {index} conservation residual",
            )
            if isinstance(row, dict)
            else -1.0
        )
        row_resilience = (
            _finite_float(
                row.get("resilience_auc"),
                f"{label} row {index} resilience AUC",
            )
            if isinstance(row, dict)
            else float("nan")
        )
        row_tail_margin = (
            _finite_float(
                row.get("minimum_tail_margin"),
                f"{label} row {index} minimum tail margin",
            )
            if isinstance(row, dict)
            else float("nan")
        )
        hard_violations = row.get("hard_violation_count") if isinstance(row, dict) else None
        reason_codes = row.get("reason_codes") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or any(
                row.get(key) != expected_value
                for key, expected_value in expected.items()
            )
            or not isinstance(row.get("solved"), bool)
            or not isinstance(hard_violations, int)
            or isinstance(hard_violations, bool)
            or hard_violations < 0
            or row_residual < 0.0
            or not isinstance(reason_codes, list)
            or any(not isinstance(reason, str) or not reason for reason in reason_codes)
        ):
            raise ArchitectureStudyError(f"{label} row {index} drifted")
        recomputed_solved += int(row["solved"])
        recomputed_hard_violations += hard_violations
        recomputed_maximum_residual = max(
            recomputed_maximum_residual, row_residual
        )
        resilience_values.append(row_resilience)
        tail_margin_values.append(row_tail_margin)
        if not row["solved"]:
            failure_reasons.update(reason_codes)
    recomputed_resilience_auc = round(fmean(resilience_values), 10)
    recomputed_tail_margin = round(fmean(tail_margin_values), 10)
    recomputed_failure_histogram = dict(sorted(failure_reasons.items()))
    if (
        recomputed_solved != solved_count
        or value.get("hard_violation_count") != recomputed_hard_violations
        or maximum_residual != recomputed_maximum_residual
        or resilience_auc != recomputed_resilience_auc
        or tail_margin != recomputed_tail_margin
        or value.get("failure_reason_code_histogram")
        != recomputed_failure_histogram
        or recomputed_hard_violations != 0
        or recomputed_maximum_residual != 0.0
    ):
        raise ArchitectureStudyError(f"{label} rows disagree with aggregate")
    return {
        "case_count": DEVELOPMENT_CASE_COUNT,
        "solved_count": solved_count,
        "solve_rate": numeric_solve_rate,
        "mean_resilience_auc": resilience_auc,
        "mean_minimum_tail_margin": tail_margin,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "failure_reason_code_histogram": recomputed_failure_histogram,
        "ordered_case_identity_sha256": canonical_hash(expected_identity),
        "rows_sha256": canonical_hash(rows),
    }


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArchitectureStudyError(f"{label} is not a lowercase SHA-256")
    return value


def _bc_initialization_identity(
    receipt: dict[str, Any], label: str
) -> dict[str, str]:
    """Validate and return the state that must match across paired LR arms."""

    initialization = receipt.get("initialization")
    behavior_cloning = receipt.get("behavior_cloning")
    normalization = receipt.get("normalization")
    if (
        not isinstance(initialization, dict)
        or not isinstance(behavior_cloning, dict)
        or not isinstance(normalization, dict)
        or behavior_cloning.get("actor_warm_start_applied") is not True
        or behavior_cloning.get("teacher") != "preparedness_teacher_action"
        or behavior_cloning.get("training_split_only") is not True
        or behavior_cloning.get("observation_normalization") is not True
        or normalization.get("observation_rms_frozen") is not True
    ):
        raise ArchitectureStudyError(f"{label} BC initialization drifted")
    actor_sha256 = _sha256_value(
        initialization.get("actor_sha256"), f"{label} actor initialization"
    )
    policy_sha256 = _sha256_value(
        initialization.get("policy_sha256"), f"{label} policy initialization"
    )
    bc_policy_sha256 = _sha256_value(
        behavior_cloning.get("policy_state_sha256"),
        f"{label} BC policy state",
    )
    observation_rms_sha256 = _sha256_value(
        initialization.get("observation_rms_sha256"),
        f"{label} initialization observation RMS",
    )
    bc_observation_rms_sha256 = _sha256_value(
        behavior_cloning.get("observation_rms_sha256"),
        f"{label} BC observation RMS",
    )
    final_observation_rms_sha256 = _sha256_value(
        normalization.get("observation_rms_sha256"),
        f"{label} final observation RMS",
    )
    dataset_sha256 = _sha256_value(
        behavior_cloning.get("dataset_sha256"), f"{label} BC dataset"
    )
    if (
        policy_sha256 != bc_policy_sha256
        or len(
            {
                observation_rms_sha256,
                bc_observation_rms_sha256,
                final_observation_rms_sha256,
            }
        )
        != 1
    ):
        raise ArchitectureStudyError(
            f"{label} BC receipt hashes disagree internally"
        )
    return {
        "actor_sha256": actor_sha256,
        "policy_sha256": policy_sha256,
        "observation_rms_sha256": observation_rms_sha256,
        "dataset_sha256": dataset_sha256,
    }


def _resolve_baseline_root(
    summary: dict[str, Any], override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    source = summary.get("source_evidence", {}).get("external_study_root")
    if not isinstance(source, str) or not source:
        raise ArchitectureStudyError("baseline external study root is missing")
    return Path(source).resolve()


def load_baseline_reference(
    summary_path: Path,
    selection_path: Path,
    *,
    study_root_override: Path | None = None,
) -> dict[str, Any]:
    """Bind the existing five-seed curves and shipped 178 DEV checkpoint."""

    summary_path = summary_path.resolve()
    selection_path = selection_path.resolve()
    summary = _load_json(summary_path, "canonical baseline summary")
    selection = _load_json(selection_path, "shipped checkpoint selection")
    baseline = summary.get("baseline")
    scope = summary.get("scope")
    if (
        summary.get("kind") != "city-recovery-training-study-200-summary"
        or not isinstance(scope, dict)
        or scope.get("split") != "dev"
        or scope.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or scope.get("final_split_used") is not False
        or not isinstance(baseline, dict)
        or baseline.get("name") != "adopted_v3_equivalent_2m"
        or selection.get("split") != "dev"
        or selection.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or selection.get("final_split_used") is not False
        or selection.get("winner", {}).get("solved_count")
        != SHIPPED_DEVELOPMENT_SOLVED_COUNT
        or selection.get("candidate_count") != 20
        or selection.get("selected_checkpoint", {}).get("policy_seed")
        != 67_017
        or selection.get("selected_checkpoint", {}).get(
            "active_actor_critic_transitions"
        )
        != 1_000_000
        or selection.get("ranking", {}).get(
            "resilience_auc_used_for_selection"
        )
        is not False
    ):
        raise ArchitectureStudyError("baseline or shipped selection contract drifted")
    endpoints = baseline.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 5:
        raise ArchitectureStudyError("baseline endpoint roster drifted")
    solved = [int(row["solved_count"]) for row in endpoints]
    aggregate = baseline.get("aggregate", {})
    if (
        [int(row["seed"]) for row in endpoints]
        != [37_017, 47_017, 57_017, 67_017, 77_017]
        or not math.isclose(fmean(solved), BASELINE_ENDPOINT_MEAN, abs_tol=1e-12)
        or not math.isclose(
            pstdev(solved), BASELINE_ENDPOINT_POPULATION_STD, abs_tol=1e-12
        )
        or not math.isclose(
            stdev(solved), BASELINE_ENDPOINT_SAMPLE_STD, abs_tol=1e-12
        )
        or not math.isclose(
            float(aggregate.get("mean_solved_count", -1.0)),
            BASELINE_ENDPOINT_MEAN,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(aggregate.get("sample_std_solved_count", -1.0)),
            BASELINE_ENDPOINT_SAMPLE_STD,
            abs_tol=1e-12,
        )
    ):
        raise ArchitectureStudyError("baseline five-seed statistics drifted")

    baseline_root = _resolve_baseline_root(summary, study_root_override)
    curves: list[dict[str, Any]] = []
    endpoint_by_seed = {int(row["seed"]): row for row in endpoints}
    for seed in endpoint_by_seed:
        receipt_path = (
            baseline_root
            / "adopted_v3_equivalent_2m"
            / f"seed-{seed}"
            / "training-receipt.json"
        )
        receipt_hash = file_sha256(receipt_path)
        if receipt_hash != endpoint_by_seed[seed].get("receipt_sha256"):
            raise ArchitectureStudyError(
                f"baseline seed {seed} receipt hash drifted"
            )
        receipt = _load_json(receipt_path, f"baseline seed {seed} receipt")
        config = receipt.get("config", {})
        if (
            receipt.get("status") != "complete"
            or receipt.get("final_split_used") is not False
            or config.get("policy_seed") != seed
            or config.get("active_actor_critic_transitions")
            != ACTIVE_TRANSITIONS
            or config.get("lanes") != 20
            or config.get("n_steps_per_lane") != 250
            or config.get("batch_size") != 500
            or config.get("bc_epochs") != 15
            or config.get("bc_warm_start") is not True
            or config.get("vec_normalize") is not True
            or config.get("freeze_observation_rms") is not True
            or config.get("critic_warmup_min_transitions") != 50_000
            or config.get("critic_warmup_max_transitions") != 100_000
            or config.get("learning_rate") != 7.5e-5
            or config.get("target_kl") != 0.02
            or config.get("ent_coef") != 0.003
            or config.get("reward_profile") != "v3_equivalent"
            or config.get("preparedness_alignment_coefficient") != 10.0
            or config.get("evaluation_milestones")
            != list(EXPECTED_TRAINER_MILESTONES)
        ):
            raise ArchitectureStudyError(
                f"baseline seed {seed} training config drifted"
            )
        curve = []
        for milestone in REGISTERED_SELECTION_MILESTONES:
            development = _development_result(
                receipt.get("development_curve", {}).get(
                    f"ppo_{milestone}_transitions"
                ),
                f"baseline seed {seed} milestone {milestone}",
                expected_active_transitions=milestone,
            )
            curve.append(
                {
                    "active_actor_critic_transitions": milestone,
                    **development,
                }
            )
        if curve[-1]["solved_count"] != endpoint_by_seed[seed]["solved_count"]:
            raise ArchitectureStudyError(
                f"baseline seed {seed} endpoint disagrees with summary"
            )
        curves.append(
            {
                "seed": seed,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_hash,
                "config_sha256": canonical_hash(config),
                "bc_initialization_identity": _bc_initialization_identity(
                    receipt, f"baseline seed {seed}"
                ),
                "curve": curve,
            }
        )
    return {
        "canonical_summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "external_study_root": str(baseline_root),
        "curves": curves,
        "five_seed_2m_endpoints": {
            "solved_counts": solved,
            "mean": BASELINE_ENDPOINT_MEAN,
            "population_std": BASELINE_ENDPOINT_POPULATION_STD,
            "sample_std": BASELINE_ENDPOINT_SAMPLE_STD,
        },
        "shipped_selected_checkpoint": {
            "selection_receipt_path": str(selection_path),
            "selection_receipt_sha256": file_sha256(selection_path),
            "candidate_count": selection.get("candidate_count"),
            "solved_count": SHIPPED_DEVELOPMENT_SOLVED_COUNT,
            "checkpoint": selection.get("selected_checkpoint"),
        },
        "final_split_used": False,
    }


def _source_identity() -> dict[str, str]:
    paths = (
        "scripts/run_large_architecture_study.py",
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


def study_contract(baseline_reference: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed contract shared by preflight, workers, and summary."""

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "source_identity": _source_identity(),
        "scope": {
            "training_split_used": True,
            "development_split_used": True,
            "development_case_count": DEVELOPMENT_CASE_COUNT,
            "final_split_imported_or_used": False,
        },
        "architecture": {
            "actor_hidden_layers": list(HIDDEN_LAYERS),
            "critic_hidden_layers": list(HIDDEN_LAYERS),
            "parameter_counts": EXPECTED_PARAMETER_COUNTS,
        },
        "registered_arms": [asdict(arm) for arm in REGISTERED_ARMS],
        "registered_policy_seeds": list(POLICY_SEEDS),
        "paired_learning_rate_contract": {
            "same_policy_seed": True,
            "same_bc_actor_initialization_sha256": True,
            "same_bc_policy_initialization_sha256": True,
            "same_frozen_observation_rms_sha256": True,
            "same_bc_dataset_sha256": True,
            "only_registered_config_difference": "learning_rate",
        },
        "training": {
            "active_actor_critic_transitions": ACTIVE_TRANSITIONS,
            "lanes": 20,
            "n_steps_per_lane": 250,
            "batch_size": 500,
            "bc_epochs": 15,
            "target_kl": 0.02,
            "ent_coef": 0.003,
            "critic_warmup_min_transitions": 50_000,
            "critic_warmup_max_transitions": 100_000,
            "freeze_observation_rms": True,
            "bc_warm_start": True,
            "vec_normalize": True,
            "reward_profile": "v3_equivalent",
            "preparedness_alignment_coefficient": 10.0,
        },
        "development_selection": {
            "registered_milestones": list(REGISTERED_SELECTION_MILESTONES),
            "candidate_count": (
                len(REGISTERED_ARMS)
                * len(POLICY_SEEDS)
                * len(REGISTERED_SELECTION_MILESTONES)
            ),
            "primary_metric": "solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_break_order": [
                "earlier_active_actor_critic_transitions",
                "registered_arm_order",
                "lower_policy_seed",
            ],
        },
        "promotion_rule": {
            "all_conditions_required": True,
            "selected_checkpoint_solved_count": {
                "operator": ">=",
                "threshold": PROMOTION_SELECTED_SOLVES,
            },
            "selected_arm_three_seed_2m_mean": {
                "operator": ">",
                "threshold": BASELINE_ENDPOINT_MEAN,
            },
            "selected_arm_seed_endpoints_at_or_above_172": {
                "operator": ">=",
                "threshold": PROMOTION_ENDPOINT_SEED_COUNT,
                "per_seed_solved_threshold": PROMOTION_ENDPOINT_SOLVES,
            },
            "on_fail": "complete_not_promoted",
            "resilience_auc_used": False,
        },
        "baseline_reference": baseline_reference,
        "null_scope": (
            "A non-promotion result applies only to [768,512,256] with "
            "learning rates 7.5e-5 and 3e-5, these three registered seeds, "
            "and this 2M-transition budget."
        ),
    }


def _registered_arm(arm_id: str) -> LearningRateArm:
    for arm in REGISTERED_ARMS:
        if arm.id == arm_id:
            return arm
    raise ArchitectureStudyError(f"unregistered arm: {arm_id}")


def validate_training_receipt(
    path: Path,
    arm: LearningRateArm,
    seed: int,
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one complete challenger run and its three registered bundles."""

    path = path.resolve()
    receipt = _load_json(path, f"{arm.id} seed {seed} receipt")
    config = receipt.get("config")
    checks = receipt.get("checks")
    expected_architecture = architecture_config(arm)
    if (
        receipt.get("status") != "complete"
        or receipt.get("training_split") != "train"
        or receipt.get("evaluation_split") != "dev"
        or receipt.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or receipt.get("final_split_used") is not False
        or not isinstance(config, dict)
        or config.get("architecture_experiment") != expected_architecture
        or config.get("policy_seed") != seed
        or config.get("active_actor_critic_transitions") != ACTIVE_TRANSITIONS
        or config.get("lanes") != 20
        or config.get("n_steps_per_lane") != 250
        or config.get("batch_size") != 500
        or config.get("bc_epochs") != 15
        or config.get("critic_warmup_min_transitions") != 50_000
        or config.get("critic_warmup_max_transitions") != 100_000
        or config.get("learning_rate") != arm.learning_rate
        or config.get("target_kl") != 0.02
        or config.get("ent_coef") != 0.003
        or config.get("reward_profile") != "v3_equivalent"
        or config.get("preparedness_alignment_coefficient") != 10.0
        or config.get("bc_warm_start") is not True
        or config.get("vec_normalize") is not True
        or config.get("freeze_observation_rms") is not True
        or config.get("evaluation_milestones")
        != list(EXPECTED_TRAINER_MILESTONES)
        or not isinstance(checks, dict)
        or checks.get("training_complete") is not True
        or checks.get("development_only_no_final_split_used") is not True
        or checks.get("development_hard_violations_zero") is not True
        or checks.get("development_conservation_residuals_zero") is not True
        or checks.get("all_registered_checkpoints_persisted") is not True
    ):
        raise ArchitectureStudyError(
            f"challenger receipt contract drifted: {path}"
        )
    _bc_initialization_identity(receipt, f"{arm.id} seed {seed}")
    candidates: list[dict[str, Any]] = []
    for milestone in REGISTERED_SELECTION_MILESTONES:
        development = _development_result(
            receipt.get("development_curve", {}).get(
                f"ppo_{milestone}_transitions"
            ),
            f"{arm.id} seed {seed} milestone {milestone}",
            expected_active_transitions=milestone,
        )
        reference = receipt.get("checkpoint_bundles", {}).get(str(milestone))
        if not isinstance(reference, dict):
            raise ArchitectureStudyError(
                f"challenger checkpoint reference missing: {arm.id}/{seed}/{milestone}"
            )
        manifest_path = Path(str(reference.get("manifest_path", "")))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        verified = bundle_verifier(manifest_path.parent)
        manifest = verified.manifest
        manifest_training = manifest.get("training")
        manifest_checkpoint = manifest.get("checkpoint")
        if not isinstance(manifest_training, dict) or not isinstance(
            manifest_checkpoint, dict
        ):
            raise ArchitectureStudyError(
                f"challenger bundle structure drifted: {arm.id}/{seed}/{milestone}"
            )
        model_record = manifest_checkpoint["file"]
        normalization = manifest["normalization"]
        verified_manifest_sha256 = file_sha256(verified.manifest_path)
        if (
            manifest_training.get("config") != config
            or manifest_training.get("config_sha256") != canonical_hash(config)
            or manifest_training.get("seed") != seed
            or manifest_training.get("milestone") != milestone
            or reference.get("checkpoint_id")
            != manifest_checkpoint["id"]
            or reference.get("active_actor_critic_transitions") != milestone
            or manifest_checkpoint.get("active_actor_critic_transitions")
            != milestone
            or reference.get("manifest_sha256") != verified_manifest_sha256
            or Path(str(reference.get("model_path", ""))).resolve()
            != verified.model_path
            or Path(str(reference.get("normalization_path", ""))).resolve()
            != verified.normalization_path
            or reference.get("model_sha256") != model_record["sha256"]
            or reference.get("normalization_sha256")
            != normalization["file"]["sha256"]
            or reference.get("obs_rms_sha256")
            != normalization["observation_rms_sha256"]
        ):
            raise ArchitectureStudyError(
                f"challenger bundle binding drifted: {arm.id}/{seed}/{milestone}"
            )
        candidates.append(
            {
                "id": reference["checkpoint_id"],
                "arm_id": arm.id,
                "learning_rate": arm.learning_rate,
                "policy_seed": seed,
                "active_actor_critic_transitions": milestone,
                "development": development,
                "training_receipt_path": str(path),
                "training_receipt_sha256": file_sha256(path),
                "bundle_path": str(verified.root),
                "bundle_manifest_path": str(verified.manifest_path),
                "bundle_manifest_sha256": verified_manifest_sha256,
                "checkpoint_path": str(verified.model_path),
                "checkpoint_sha256": model_record["sha256"],
                "normalization_path": str(verified.normalization_path),
                "normalization_file_sha256": normalization["file"]["sha256"],
                "observation_rms_sha256": normalization[
                    "observation_rms_sha256"
                ],
            }
        )
    endpoint = candidates[-1]["development"]
    if endpoint != _development_result(
        receipt.get("development"),
        f"{arm.id} seed {seed} endpoint",
        expected_active_transitions=ACTIVE_TRANSITIONS,
    ):
        raise ArchitectureStudyError(
            f"challenger endpoint differs from 2M curve: {arm.id}/{seed}"
        )
    return receipt, candidates


def rank_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank challenger checkpoints by DEV solves and neutral tie-breakers."""

    expected_count = (
        len(REGISTERED_ARMS)
        * len(POLICY_SEEDS)
        * len(REGISTERED_SELECTION_MILESTONES)
    )
    if len(candidates) != expected_count:
        raise ArchitectureStudyError(
            f"expected {expected_count} challenger checkpoints"
        )
    arm_order = {arm.id: index for index, arm in enumerate(REGISTERED_ARMS)}
    if any(candidate.get("arm_id") not in arm_order for candidate in candidates):
        raise ArchitectureStudyError("candidate contains an unregistered arm")
    return sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -int(candidate["development"]["solved_count"]),
            int(candidate["active_actor_critic_transitions"]),
            arm_order[str(candidate["arm_id"])],
            int(candidate["policy_seed"]),
        ),
    )


def arm_endpoint_summary(
    arm: LearningRateArm, candidates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    endpoints = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["arm_id"] == arm.id
            and candidate["active_actor_critic_transitions"]
            == ACTIVE_TRANSITIONS
        ),
        key=lambda candidate: int(candidate["policy_seed"]),
    )
    if [row["policy_seed"] for row in endpoints] != list(POLICY_SEEDS):
        raise ArchitectureStudyError(f"{arm.id} endpoint roster drifted")
    solved = [int(row["development"]["solved_count"]) for row in endpoints]
    return {
        "arm": asdict(arm),
        "rows": [
            {
                "seed": row["policy_seed"],
                "solved_count": row["development"]["solved_count"],
                "solve_rate": row["development"]["solve_rate"],
                "mean_resilience_auc": row["development"][
                    "mean_resilience_auc"
                ],
                "receipt_sha256": row["training_receipt_sha256"],
            }
            for row in endpoints
        ],
        "mean_solved_count": fmean(solved),
        "population_std_solved_count": pstdev(solved),
        "sample_std_solved_count": stdev(solved),
        "seed_count_at_or_above_172": sum(
            count >= PROMOTION_ENDPOINT_SOLVES for count in solved
        ),
    }


def promotion_decision(
    selected: dict[str, Any], arm_summary: dict[str, Any]
) -> dict[str, Any]:
    """Apply all three preregistered promotion conditions."""

    selected_condition = (
        int(selected["development"]["solved_count"])
        >= PROMOTION_SELECTED_SOLVES
    )
    mean_condition = (
        float(arm_summary["mean_solved_count"]) > BASELINE_ENDPOINT_MEAN
    )
    seed_condition = (
        int(arm_summary["seed_count_at_or_above_172"])
        >= PROMOTION_ENDPOINT_SEED_COUNT
    )
    passed = selected_condition and mean_condition and seed_condition
    return {
        "all_conditions_required": True,
        "conditions": {
            "selected_checkpoint_at_least_183": {
                "observed": selected["development"]["solved_count"],
                "passed": selected_condition,
            },
            "selected_arm_three_seed_2m_mean_above_171_4": {
                "observed": arm_summary["mean_solved_count"],
                "passed": mean_condition,
            },
            "selected_arm_at_least_two_seed_endpoints_at_or_above_172": {
                "observed": arm_summary["seed_count_at_or_above_172"],
                "passed": seed_condition,
            },
        },
        "passed": passed,
        "decision": "promote" if passed else "complete_not_promoted",
        "resilience_auc_used": False,
    }


def _paired_config_without_learning_rate(
    config: dict[str, Any], arm: LearningRateArm
) -> dict[str, Any]:
    """Remove only the registered LR treatment labels before pair comparison."""

    comparable = json.loads(json.dumps(config, allow_nan=False))
    comparable.pop("learning_rate", None)
    architecture = comparable.get("architecture_experiment")
    if not isinstance(architecture, dict):
        raise ArchitectureStudyError(f"{arm.id} architecture config is missing")
    architecture.pop("arm_id", None)
    return comparable


def _validate_paired_learning_rate_receipts(
    receipt_payloads: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prove each LR pair shares BC state and differs only by treatment."""

    results: list[dict[str, Any]] = []
    for seed in POLICY_SEEDS:
        left_arm, right_arm = REGISTERED_ARMS
        try:
            left = receipt_payloads[(left_arm.id, seed)]
            right = receipt_payloads[(right_arm.id, seed)]
        except KeyError as exc:
            raise ArchitectureStudyError(
                f"paired LR receipt is missing for seed {seed}"
            ) from exc
        left_identity = _bc_initialization_identity(
            left, f"{left_arm.id} seed {seed}"
        )
        right_identity = _bc_initialization_identity(
            right, f"{right_arm.id} seed {seed}"
        )
        if left_identity != right_identity:
            raise ArchitectureStudyError(
                f"paired LR arms seed {seed} have different BC initialization"
            )
        if _paired_config_without_learning_rate(
            left["config"], left_arm
        ) != _paired_config_without_learning_rate(right["config"], right_arm):
            raise ArchitectureStudyError(
                f"paired LR arms seed {seed} differ beyond learning rate"
            )
        results.append(
            {
                "seed": seed,
                "arm_ids": [left_arm.id, right_arm.id],
                "learning_rates": [
                    left["config"]["learning_rate"],
                    right["config"]["learning_rate"],
                ],
                **left_identity,
                "identical_bc_initialization_and_observation_rms": True,
                "only_registered_config_difference_is_learning_rate": True,
            }
        )
    return results


def build_summary(
    output_root: Path,
    contract: dict[str, Any],
    *,
    bundle_verifier: Callable[[Path], Any] = verify_checkpoint_bundle,
) -> dict[str, Any]:
    """Validate six runs, rank 18 checkpoints, and apply promotion."""

    candidates: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    receipt_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in REGISTERED_ARMS:
        for seed in POLICY_SEEDS:
            receipt_path = (
                arm_directory(output_root, arm, seed) / "training-receipt.json"
            )
            receipt, arm_candidates = validate_training_receipt(
                receipt_path,
                arm,
                seed,
                bundle_verifier=bundle_verifier,
            )
            receipts.append(
                {
                    "arm_id": arm.id,
                    "seed": seed,
                    "path": str(receipt_path),
                    "sha256": file_sha256(receipt_path),
                    "status": receipt["status"],
                    "config_sha256": canonical_hash(receipt["config"]),
                    "bc_initialization_identity": (
                        _bc_initialization_identity(
                            receipt, f"{arm.id} seed {seed}"
                        )
                    ),
                }
            )
            receipt_payloads[(arm.id, seed)] = receipt
            candidates.extend(arm_candidates)
    paired_initialization_checks = _validate_paired_learning_rate_receipts(
        receipt_payloads
    )
    ranked = rank_candidates(candidates)
    selected = ranked[0]
    arm_summaries = [
        arm_endpoint_summary(arm, ranked) for arm in REGISTERED_ARMS
    ]
    selected_arm_summary = next(
        row for row in arm_summaries if row["arm"]["id"] == selected["arm_id"]
    )
    promotion = promotion_decision(selected, selected_arm_summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": (
            "complete_promoted"
            if promotion["passed"]
            else "complete_not_promoted"
        ),
        "created_at_utc": _utc_now(),
        "contract_sha256": canonical_hash(contract),
        "split": "dev",
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "final_split_imported_or_used": False,
        "receipts": receipts,
        "paired_learning_rate_checks": paired_initialization_checks,
        "candidate_count": len(ranked),
        "ranking": {
            "primary_metric": "solved_count",
            "resilience_auc_used_for_selection": False,
            "candidates": ranked,
        },
        "selected_checkpoint": selected,
        "arm_endpoint_summaries": arm_summaries,
        "baseline_reference": contract["baseline_reference"],
        "promotion": promotion,
        "null_scope": contract["null_scope"],
    }


def _validate_protocol(output_root: Path, contract: dict[str, Any]) -> None:
    protocol = _load_json(output_root / "protocol.json", "study protocol")
    if (
        protocol.get("contract_sha256") != canonical_hash(contract)
        or protocol.get("contract") != contract
    ):
        raise ArchitectureStudyError(
            "existing study protocol differs from the current contract"
        )


def _create_study_protocol(
    output_root: Path, contract: dict[str, Any]
) -> Path:
    """Create one new study root and immutable protocol."""

    if output_root.exists():
        raise ArchitectureStudyError("study protocol requires a new output root")
    output_root.mkdir(parents=True, exist_ok=False)
    protocol_path = output_root / "protocol.json"
    _atomic_create_json(
        protocol_path,
        {
            "created_at_utc": _utc_now(),
            "contract_sha256": canonical_hash(contract),
            "contract": contract,
        },
    )
    return protocol_path


def _publish_summary_idempotent(
    path: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    """Create a summary once or verify an equivalent prior publication."""

    if not path.exists():
        _atomic_create_json(path, summary)
        return summary
    existing = _load_json(path, "architecture study summary")
    existing_comparable = {
        key: value for key, value in existing.items() if key != "created_at_utc"
    }
    summary_comparable = {
        key: value for key, value in summary.items() if key != "created_at_utc"
    }
    if existing_comparable != summary_comparable:
        raise ArchitectureStudyError("existing summary differs from recomputation")
    return existing


def _run_one_arm(output_root: Path, arm: LearningRateArm, seed: int) -> None:
    directory = arm_directory(output_root, arm, seed)
    receipt_path = directory / "training-receipt.json"
    if receipt_path.exists():
        validate_training_receipt(receipt_path, arm, seed)
        print(f"[architecture] verified {arm.id} seed {seed}", flush=True)
        return
    if directory.exists():
        raise ArchitectureStudyError(
            "partial run cannot be retried in place; preserve it and choose a "
            f"new study root: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "trainer.log"
    print(f"[architecture] starting {arm.id} seed {seed}", flush=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            worker_command(output_root, arm, seed),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise ArchitectureStudyError(
            f"trainer failed for {arm.id} seed {seed}; see {log_path}"
        )
    _, candidates = validate_training_receipt(receipt_path, arm, seed)
    endpoint = candidates[-1]["development"]["solved_count"]
    print(
        f"[architecture] finished {arm.id} seed {seed}: "
        f"{endpoint}/{DEVELOPMENT_CASE_COUNT}",
        flush=True,
    )


def _run_worker(output_root: Path, arm_id: str, seed: int) -> int:
    arm = _registered_arm(arm_id)
    if seed not in POLICY_SEEDS:
        raise ArchitectureStudyError(f"unregistered policy seed: {seed}")
    protocol = _load_json(output_root / "protocol.json", "study protocol")
    contract = protocol.get("contract")
    if (
        not isinstance(contract, dict)
        or protocol.get("contract_sha256") != canonical_hash(contract)
        or contract.get("tool") != TOOL_ID
        or contract.get("git_commit") != _git_commit()
        or contract.get("source_identity") != _source_identity()
    ):
        raise ArchitectureStudyError("worker protocol or source identity drifted")
    if asdict(arm) not in contract.get("registered_arms", []):
        raise ArchitectureStudyError("worker arm is absent from protocol")
    with _inject_large_architecture(arm):
        return train_policy.main(trainer_arguments(output_root, arm, seed))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
    )
    parser.add_argument(
        "--baseline-study-root",
        type=Path,
        help="optional relocated copy of the external adopted-baseline study",
    )
    parser.add_argument(
        "--selection-receipt",
        type=Path,
        default=DEFAULT_SELECTION_RECEIPT,
    )
    modes = parser.add_mutually_exclusive_group(required=False)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    parser.add_argument("--_worker-arm", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    public_modes = sum(
        bool(value)
        for value in (args.preflight, args.execute, args.resume, args.summarize)
    )
    worker_mode = args._worker_arm is not None or args._worker_seed is not None
    if worker_mode:
        if public_modes or args._worker_arm is None or args._worker_seed is None:
            parser.error("internal worker arguments are incomplete")
    elif public_modes != 1:
        parser.error(
            "one of --preflight, --execute, --resume, or --summarize is required"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = _require_external_root(args.output_root)
    if args._worker_arm is not None:
        return _run_worker(output_root, args._worker_arm, args._worker_seed)

    baseline_reference = load_baseline_reference(
        args.baseline_summary,
        args.selection_receipt,
        study_root_override=args.baseline_study_root,
    )
    contract = study_contract(baseline_reference)
    contract_sha256 = canonical_hash(contract)
    clean = _worktree_is_clean()
    if args.preflight:
        print(
            json.dumps(
                {
                    "tool": TOOL_ID,
                    "status": "ready" if clean else "blocked_dirty_worktree",
                    "filesystem_written": False,
                    "training_started": False,
                    "registered_run_count": len(REGISTERED_ARMS)
                    * len(POLICY_SEEDS),
                    "registered_checkpoint_candidate_count": (
                        len(REGISTERED_ARMS)
                        * len(POLICY_SEEDS)
                        * len(REGISTERED_SELECTION_MILESTONES)
                    ),
                    "architecture": list(HIDDEN_LAYERS),
                    "parameter_counts": EXPECTED_PARAMETER_COUNTS,
                    "arms": [asdict(arm) for arm in REGISTERED_ARMS],
                    "seeds": list(POLICY_SEEDS),
                    "baseline_reference": baseline_reference,
                    "promotion_rule": contract["promotion_rule"],
                    "contract_sha256": contract_sha256,
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
        raise ArchitectureStudyError(
            "refusing training or summary publication from a dirty worktree"
        )

    if args.execute:
        _create_study_protocol(output_root, contract)
    else:
        if not output_root.is_dir():
            raise ArchitectureStudyError(
                "--resume/--summarize requires an existing output root"
            )
        _validate_protocol(output_root, contract)

    if args.execute or args.resume:
        for arm in REGISTERED_ARMS:
            for seed in POLICY_SEEDS:
                _run_one_arm(output_root, arm, seed)

    summary = build_summary(output_root, contract)
    summary_path = output_root / "architecture-study-summary.json"
    summary = _publish_summary_idempotent(summary_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "selected_checkpoint": {
                    key: summary["selected_checkpoint"][key]
                    for key in (
                        "id",
                        "arm_id",
                        "policy_seed",
                        "active_actor_critic_transitions",
                        "development",
                    )
                },
                "arm_endpoint_summaries": summary["arm_endpoint_summaries"],
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
    except ArchitectureStudyError as error:
        print(f"large architecture study failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
