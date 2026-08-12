"""Training-only primitives for the moderate family-reweighting experiment.

This module is deliberately separate from the canonical trainer and city
contract.  It measures family difficulty with the shipped v4 artifact on the
authored 6 x 32 training roster, then constructs a deterministic episode cycle
in which the two hardest training families appear twice and the other four
appear once.  The tuned rule is computed only as a contextual ranking and is
reported only if that ranking differs.  No development result participates in
choosing the weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Callable, Sequence

# Apply the same native thread caps as the canonical trainer before NumPy,
# ONNX Runtime, Torch, or SB3 can initialize a native worker pool.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    CityRecoveryEnv,
    CyclingScenarioEnv,
)
from backend.app.city.outcome import summarize_trajectory  # noqa: E402
from backend.app.city.planners import tuned_rule_action  # noqa: E402
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
)
from model.policy import load_policy  # noqa: E402
from scripts.training_artifacts import verify_checkpoint_bundle  # noqa: E402

TOOL_ID = "moderate_family_training.py"
DIFFICULTY_SCHEMA = "city-recovery-training-family-difficulty-v1"
STUDY_SCHEMA = "city-recovery-moderate-family-study-v1"
TRAINING_CASE_COUNT = 192
DEVELOPMENT_CASE_COUNT = 200
TRAINING_LANES = 20
TRAINING_EPISODE_DAYS = 30
HARD_FAMILY_COUNT = 2
HARD_FAMILY_WEIGHT = 2
OTHER_FAMILY_WEIGHT = 1
WEIGHTED_CYCLE_CASE_COUNT = 256
WEIGHTED_FAMILY_SLOT_COUNT = 8
CRITIC_WARMUP_PREFIX_TRANSITIONS = (50_000, 100_000)
POLICY_SEEDS = (37_017, 47_017, 57_017)
ACTIVE_TRANSITIONS = 2_000_000
CURVE_MILESTONES = (500_000, 1_000_000, 2_000_000)
CURRENT_SELECTED_SOLVED_COUNT = 178
INCUMBENT_SEED_MEAN_SOLVED_COUNT = 171.4
INCUMBENT_SEED_POPULATION_STD_SOLVED_COUNT = 1.62
PROMOTION_SELECTED_MINIMUM = 183
PROMOTION_ENDPOINT_MINIMUM = 172
PROMOTION_ENDPOINT_MINIMUM_COUNT = 2
EVALUATION_MILESTONES = (200_000, 500_000, 1_000_000, 2_000_000)
SHIPPED_ARTIFACT_PATH = ROOT / "artifacts" / "city_recovery_ppo.v4.onnx"
SHIPPED_ARTIFACT_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)

SOURCE_PATHS = (
    "scripts/moderate_family_training.py",
    "scripts/train_policy.py",
    "scripts/training_artifacts.py",
    "backend/app/shared_evidence.py",
    "backend/app/city/environment.py",
    "backend/app/city/scenarios.py",
    "backend/app/city/planners.py",
    "backend/app/city/outcome.py",
    "backend/app/city/physics.py",
    "model/policy.py",
)


class ModerateStudyError(RuntimeError):
    """Raised when the preregistered reweighting contract is not satisfied."""


@dataclass(frozen=True)
class TrainingCase:
    """One exact member of the authored training roster."""

    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    scenario: Any
    schedule: tuple[Any, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "family_id": self.family_id,
            "case_seed": self.case_seed,
            "tape_seed": self.tape_seed,
            "scenario_sha256": canonical_hash(
                self.scenario.model_dump(mode="json")
            ),
            "tape_sha256": canonical_hash(
                [asdict(shock) for shock in self.schedule]
            ),
        }


def source_identity() -> dict[str, str]:
    """Hash the complete implementation surface used by the measurement."""

    return {relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS}


def build_training_cases() -> list[TrainingCase]:
    """Recompute the exact ordered 6-family by 32-seed training roster."""

    cases: list[TrainingCase] = []
    for family in TRAINING_FAMILIES:
        for case_seed in TRAINING_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            cases.append(
                TrainingCase(
                    row_id=f"{family.id}:{case_seed}",
                    family_id=family.id,
                    case_seed=case_seed,
                    tape_seed=tape_seed,
                    scenario=scenario,
                    schedule=tuple(generate_disaster_tape(scenario, tape_seed)),
                )
            )
    if (
        len(TRAINING_FAMILIES) != 6
        or TRAINING_SEEDS != tuple(range(810000, 810032))
        or len(cases) != TRAINING_CASE_COUNT
        or len({case.row_id for case in cases}) != TRAINING_CASE_COUNT
        or {family.id for family in TRAINING_FAMILIES}
        & {family.id for family in DEVELOPMENT_FAMILIES}
    ):
        raise ModerateStudyError("authored training roster drifted from 6 x 32")
    return cases


def _rollout_policy(
    case: TrainingCase,
    *,
    label: str,
    action_function: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]],
    attach_evidence: bool,
) -> dict[str, Any]:
    """Roll out one fixed policy without changing the city contract."""

    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        case.schedule,
        collect_evidence=True,
    )
    observation, reset_evidence = environment.reset(seed=case.tape_seed)
    if reset_evidence.get("shock_schedule_sha256") != case.identity()["tape_sha256"]:
        raise ModerateStudyError(f"training tape identity drifted: {case.row_id}")
    terminated = False
    while not terminated:
        action, evidence = action_function(observation)
        if attach_evidence:
            observation, _, terminated, truncated, _ = (
                environment.step_with_evidence(action, evidence)
            )
        else:
            observation, _, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise ModerateStudyError(f"unexpected truncation: {case.row_id}")
    summary = summarize_trajectory(
        label,
        environment.trajectory,
        case.scenario,
    )
    outcome = summary["absolute_outcome"]
    tail = np.asarray(outcome["tail_minimum_services"], dtype=np.float64)
    targets = np.asarray(outcome["recovery_targets"], dtype=np.float64)
    row = {
        **case.identity(),
        "solved": bool(outcome["solved"]),
        "reason_codes": list(outcome["reason_codes"]),
        "resilience_auc": float(summary["rauc"]),
        "minimum_tail_margin": float(np.min(tail - targets)),
        "hard_violation_count": int(summary["hard_violation_count"]),
        "max_conservation_residual": float(
            summary["max_logistics_conservation_residual"]
        ),
        "trajectory_sha256": str(summary["trajectory_sha256"]),
    }
    if row["hard_violation_count"] != 0 or row["max_conservation_residual"] != 0.0:
        raise ModerateStudyError(f"planner violated invariants: {case.row_id}")
    return row


def rollout_tuned_rule(case: TrainingCase) -> dict[str, Any]:
    """Evaluate the fixed tuned rule as a non-selecting contextual comparator."""

    return _rollout_policy(
        case,
        label="tuned_rule_training_difficulty_context",
        action_function=tuned_rule_action,
        attach_evidence=True,
    )


def shipped_policy_evaluator() -> Callable[[TrainingCase], dict[str, Any]]:
    """Load the exact shipped artifact and return its TRAIN-only evaluator."""

    actual_sha256 = file_sha256(SHIPPED_ARTIFACT_PATH)
    if actual_sha256 != SHIPPED_ARTIFACT_SHA256:
        raise ModerateStudyError("shipped v4 artifact hash drifted")
    loaded = load_policy(SHIPPED_ARTIFACT_PATH)

    def evaluate(case: TrainingCase) -> dict[str, Any]:
        def action(observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            return loaded.predict(observation), {
                "runtime": "onnxruntime-cpu",
                "artifact_sha256": actual_sha256,
            }

        return _rollout_policy(
            case,
            label="shipped_v4_training_difficulty",
            action_function=action,
            attach_evidence=False,
        )

    return evaluate


def aggregate_family_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one complete 32-case training-family block."""

    if len(rows) != len(TRAINING_SEEDS):
        raise ModerateStudyError("training family must contain exactly 32 rows")
    reasons = Counter(
        reason
        for row in rows
        if not bool(row["solved"])
        for reason in row["reason_codes"]
    )
    solved = sum(bool(row["solved"]) for row in rows)
    return {
        "family_id": rows[0]["family_id"],
        "case_count": len(rows),
        "solved_count": solved,
        "solve_rate": solved / len(rows),
        "mean_resilience_auc": round(
            fmean(float(row["resilience_auc"]) for row in rows), 10
        ),
        "mean_minimum_tail_margin": round(
            fmean(float(row["minimum_tail_margin"]) for row in rows), 10
        ),
        "hard_violation_count": sum(
            int(row["hard_violation_count"]) for row in rows
        ),
        "maximum_conservation_residual": max(
            float(row["max_conservation_residual"]) for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
    }


def rank_training_families(
    per_family: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank hardest first using training-only, preregistered tie-breakers."""

    expected_ids = [family.id for family in TRAINING_FAMILIES]
    if (
        len(per_family) != len(expected_ids)
        or [row.get("family_id") for row in per_family] != expected_ids
        or any(
            isinstance(row.get("solved_count"), bool)
            or not isinstance(row.get("solved_count"), int)
            or not 0 <= row["solved_count"] <= len(TRAINING_SEEDS)
            for row in per_family
        )
    ):
        raise ModerateStudyError("family difficulty rows are not canonical")
    return sorted(
        (dict(row) for row in per_family),
        key=lambda row: (
            int(row["solved_count"]),
            str(row["family_id"]),
        ),
    )


def family_weights(hardest_family_ids: Sequence[str]) -> dict[str, int]:
    """Return the fixed two-versus-one family episode weights."""

    hardest = tuple(hardest_family_ids)
    known = [family.id for family in TRAINING_FAMILIES]
    if len(hardest) != HARD_FAMILY_COUNT or len(set(hardest)) != HARD_FAMILY_COUNT:
        raise ModerateStudyError("exactly two distinct hard families are required")
    if any(family_id not in known for family_id in hardest):
        raise ModerateStudyError("hard-family selection is outside TRAIN")
    return {
        family_id: (
            HARD_FAMILY_WEIGHT if family_id in hardest else OTHER_FAMILY_WEIGHT
        )
        for family_id in known
    }


def build_difficulty_receipt(
    *,
    shipped_evaluator: Callable[[TrainingCase], dict[str, Any]] | None = None,
    tuned_evaluator: Callable[[TrainingCase], dict[str, Any]] = rollout_tuned_rule,
) -> dict[str, Any]:
    """Measure shipped-policy TRAIN difficulty and derive the moderate sampler."""

    cases = build_training_cases()
    if shipped_evaluator is None:
        shipped_evaluator = shipped_policy_evaluator()
    rows = [shipped_evaluator(case) for case in cases]
    tuned_rows = [tuned_evaluator(case) for case in cases]
    expected_identities = [case.identity() for case in cases]
    for label, policy_rows in (("shipped", rows), ("tuned", tuned_rows)):
        for index, (row, expected) in enumerate(
            zip(policy_rows, expected_identities, strict=True)
        ):
            if any(row.get(key) != value for key, value in expected.items()):
                raise ModerateStudyError(
                    f"{label} training row identity drifted at {index}"
                )
    per_family = [
        aggregate_family_rows(
            [row for row in rows if row["family_id"] == family.id]
        )
        for family in TRAINING_FAMILIES
    ]
    ranked = rank_training_families(per_family)
    tuned_per_family = [
        aggregate_family_rows(
            [row for row in tuned_rows if row["family_id"] == family.id]
        )
        for family in TRAINING_FAMILIES
    ]
    tuned_ranked = rank_training_families(tuned_per_family)
    shipped_order = [row["family_id"] for row in ranked]
    tuned_order = [row["family_id"] for row in tuned_ranked]
    ordering_differs = shipped_order != tuned_order
    hardest = [row["family_id"] for row in ranked[:HARD_FAMILY_COUNT]]
    weights = family_weights(hardest)
    weighted_cases = weighted_training_cases(hardest)
    family_slot_pattern = [
        case.family_id for case in weighted_cases[:WEIGHTED_FAMILY_SLOT_COUNT]
    ]
    prefix_balance = warmup_prefix_balance(hardest)
    if (
        sum(int(row["hard_violation_count"]) for row in rows) != 0
        or max(float(row["max_conservation_residual"]) for row in rows) != 0.0
        or sum(int(row["hard_violation_count"]) for row in tuned_rows) != 0
        or max(float(row["max_conservation_residual"]) for row in tuned_rows)
        != 0.0
        or sum(weights.values()) != 8
        or sum(weights[case.family_id] for case in cases)
        != WEIGHTED_CYCLE_CASE_COUNT
    ):
        raise ModerateStudyError("difficulty measurement invariants failed")
    return {
        "schema_version": DIFFICULTY_SCHEMA,
        "tool": TOOL_ID,
        "phase": "training_family_difficulty",
        "split": "train",
        "selection_policy": {
            "id": "shipped_v4_onnx",
            "artifact_path": str(SHIPPED_ARTIFACT_PATH),
            "artifact_sha256": SHIPPED_ARTIFACT_SHA256,
            "runtime": "onnxruntime_CPUExecutionProvider",
        },
        "case_count": TRAINING_CASE_COUNT,
        "family_count": len(TRAINING_FAMILIES),
        "seeds_per_family": len(TRAINING_SEEDS),
        "ordered_case_contract_sha256": canonical_hash(expected_identities),
        "source_identity": source_identity(),
        "access_contract": {
            "training_split_used": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": True,
            "learned_policy_role": "training_family_weight_selection_only",
        },
        "ranking": {
            "purpose": "choose_two_hardest_of_six_training_families",
            "order": [
                "lower_shipped_policy_solved_count",
                "lexicographically_lower_family_id",
            ],
            "development_evidence_used": False,
            "ranked_family_ids": shipped_order,
        },
        "per_family": per_family,
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "contextual_tuned_rule": (
            {
                "reported": True,
                "reason": "family_ordering_differs_from_shipped_policy",
                "ranked_family_ids": tuned_order,
                "per_family": tuned_per_family,
                "rows": tuned_rows,
                "rows_sha256": canonical_hash(tuned_rows),
            }
            if ordering_differs
            else {
                "reported": False,
                "reason": "family_ordering_matches_shipped_policy",
            }
        ),
        "sampler": {
            "kind": "deterministic_interleaved_weighted_episode_cycle",
            "cycle_construction": (
                "32_seed_rounds_of_eight_interleaved_family_slots"
            ),
            "hardest_third_family_count": HARD_FAMILY_COUNT,
            "hardest_family_ids": hardest,
            "family_weights": weights,
            "family_slot_pattern": family_slot_pattern,
            "hard_family_weight": HARD_FAMILY_WEIGHT,
            "other_family_weight": OTHER_FAMILY_WEIGHT,
            "canonical_case_count": TRAINING_CASE_COUNT,
            "weighted_cycle_case_count": WEIGHTED_CYCLE_CASE_COUNT,
            "critic_warmup_prefix_balance": prefix_balance,
            "applies_to": [
                "behavior_cloning",
                "dagger_rollouts",
                "critic_warmup",
                "ppo_actor_critic_training",
            ],
        },
        "invariants": {
            "exact_training_roster_6x32": True,
            "all_hard_violation_counts_zero": True,
            "all_conservation_residuals_exactly_zero": True,
            "hardest_families_have_exactly_2x_weight": True,
            "finite_critic_warmup_prefixes_balanced_within_one_episode": True,
            "development_evidence_used_for_weight_selection": False,
            "shipped_policy_alone_selected_family_weights": True,
            "tuned_rule_did_not_select_family_weights": True,
            "final_split_used": False,
        },
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish canonical JSON with create-new semantics."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ModerateStudyError(f"refusing to overwrite evidence: {path}")
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_parent(path)
    except FileExistsError as exc:
        raise ModerateStudyError(f"refusing to overwrite evidence: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModerateStudyError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ModerateStudyError(f"{label} must be a JSON object")
    return value


def validate_difficulty_receipt(path: Path) -> dict[str, Any]:
    """Recompute and validate the training-only weight-selection evidence."""

    receipt = _load_object(path, "training-family difficulty receipt")
    cases = build_training_cases()
    expected_identities = [case.identity() for case in cases]
    rows = receipt.get("rows")
    per_family = receipt.get("per_family")
    sampler = receipt.get("sampler")
    if (
        receipt.get("schema_version") != DIFFICULTY_SCHEMA
        or receipt.get("tool") != TOOL_ID
        or receipt.get("phase") != "training_family_difficulty"
        or receipt.get("split") != "train"
        or receipt.get("selection_policy")
        != {
            "id": "shipped_v4_onnx",
            "artifact_path": str(SHIPPED_ARTIFACT_PATH),
            "artifact_sha256": SHIPPED_ARTIFACT_SHA256,
            "runtime": "onnxruntime_CPUExecutionProvider",
        }
        or receipt.get("case_count") != TRAINING_CASE_COUNT
        or receipt.get("ordered_case_contract_sha256")
        != canonical_hash(expected_identities)
        or receipt.get("access_contract")
        != {
            "training_split_used": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": True,
            "learned_policy_role": "training_family_weight_selection_only",
        }
        or not isinstance(rows, list)
        or len(rows) != TRAINING_CASE_COUNT
        or canonical_hash(rows) != receipt.get("rows_sha256")
        or not isinstance(per_family, list)
        or not isinstance(sampler, dict)
    ):
        raise ModerateStudyError("training-family difficulty contract drifted")
    if file_sha256(SHIPPED_ARTIFACT_PATH) != SHIPPED_ARTIFACT_SHA256:
        raise ModerateStudyError("shipped v4 artifact hash drifted")
    for relative, expected_hash in receipt.get("source_identity", {}).items():
        if relative not in SOURCE_PATHS or file_sha256(ROOT / relative) != expected_hash:
            raise ModerateStudyError(f"difficulty source identity drifted: {relative}")
    if set(receipt.get("source_identity", {})) != set(SOURCE_PATHS):
        raise ModerateStudyError("difficulty source identity is incomplete")
    for index, (row, identity) in enumerate(
        zip(rows, expected_identities, strict=True)
    ):
        if any(row.get(key) != value for key, value in identity.items()):
            raise ModerateStudyError(f"difficulty row identity drifted at {index}")
        if (
            not isinstance(row.get("solved"), bool)
            or int(row.get("hard_violation_count", -1)) != 0
            or float(row.get("max_conservation_residual", -1.0)) != 0.0
        ):
            raise ModerateStudyError(f"difficulty row outcome drifted at {index}")
    recomputed_per_family = [
        aggregate_family_rows(
            [row for row in rows if row["family_id"] == family.id]
        )
        for family in TRAINING_FAMILIES
    ]
    ranked = rank_training_families(recomputed_per_family)
    hardest = [row["family_id"] for row in ranked[:HARD_FAMILY_COUNT]]
    weights = family_weights(hardest)
    weighted_cases = weighted_training_cases(hardest)
    family_slot_pattern = [
        case.family_id for case in weighted_cases[:WEIGHTED_FAMILY_SLOT_COUNT]
    ]
    prefix_balance = warmup_prefix_balance(hardest)
    if (
        per_family != recomputed_per_family
        or receipt.get("ranking", {}).get("ranked_family_ids")
        != [row["family_id"] for row in ranked]
        or sampler.get("hardest_family_ids") != hardest
        or sampler.get("family_weights") != weights
        or sampler.get("kind")
        != "deterministic_interleaved_weighted_episode_cycle"
        or sampler.get("cycle_construction")
        != "32_seed_rounds_of_eight_interleaved_family_slots"
        or sampler.get("family_slot_pattern") != family_slot_pattern
        or sampler.get("weighted_cycle_case_count") != WEIGHTED_CYCLE_CASE_COUNT
        or sampler.get("critic_warmup_prefix_balance") != prefix_balance
        or sampler.get("applies_to")
        != [
            "behavior_cloning",
            "dagger_rollouts",
            "critic_warmup",
            "ppo_actor_critic_training",
        ]
    ):
        raise ModerateStudyError("derived family weighting drifted")
    tuned = receipt.get("contextual_tuned_rule")
    if not isinstance(tuned, dict) or not isinstance(tuned.get("reported"), bool):
        raise ModerateStudyError("tuned-rule context disclosure is invalid")
    if tuned["reported"]:
        tuned_rows = tuned.get("rows")
        if (
            tuned.get("reason") != "family_ordering_differs_from_shipped_policy"
            or not isinstance(tuned_rows, list)
            or len(tuned_rows) != TRAINING_CASE_COUNT
            or canonical_hash(tuned_rows) != tuned.get("rows_sha256")
        ):
            raise ModerateStudyError("reported tuned-rule context drifted")
        for index, (row, identity) in enumerate(
            zip(tuned_rows, expected_identities, strict=True)
        ):
            if any(row.get(key) != value for key, value in identity.items()):
                raise ModerateStudyError(
                    f"tuned-rule context identity drifted at {index}"
                )
        tuned_per_family = [
            aggregate_family_rows(
                [row for row in tuned_rows if row["family_id"] == family.id]
            )
            for family in TRAINING_FAMILIES
        ]
        tuned_order = [
            row["family_id"] for row in rank_training_families(tuned_per_family)
        ]
        if (
            tuned.get("per_family") != tuned_per_family
            or tuned.get("ranked_family_ids") != tuned_order
            or tuned_order == [row["family_id"] for row in ranked]
        ):
            raise ModerateStudyError("tuned-rule contextual ranking drifted")
    elif tuned != {
        "reported": False,
        "reason": "family_ordering_matches_shipped_policy",
    }:
        raise ModerateStudyError("omitted tuned-rule context disclosure drifted")
    return receipt


def weighted_training_cases(
    hardest_family_ids: Sequence[str],
) -> list[TrainingCase]:
    """Build an interleaved 256-episode cycle with 2x hard weights."""

    weights = family_weights(hardest_family_ids)
    canonical = build_training_cases()
    known_ids = [family.id for family in TRAINING_FAMILIES]
    hard = [family_id for family_id in known_ids if weights[family_id] == 2]
    other = [family_id for family_id in known_ids if weights[family_id] == 1]
    if len(hard) != 2 or len(other) != 4:
        raise ModerateStudyError("weighted family partition drifted")
    # Every consecutive four slots contain one occurrence of each hard family.
    # Repeating this eight-slot pattern therefore realizes the registered 2:1
    # treatment during finite warm-up prefixes, rather than deferring the hard
    # duplicates until the end of the 256-case cycle.
    family_slots = (
        hard[0],
        other[0],
        hard[1],
        other[1],
        hard[0],
        other[2],
        hard[1],
        other[3],
    )
    if Counter(family_slots) != Counter(weights):
        raise ModerateStudyError("weighted family slot pattern drifted")
    by_family = {
        family_id: [case for case in canonical if case.family_id == family_id]
        for family_id in known_ids
    }
    weighted = [
        by_family[family_id][seed_index]
        for seed_index in range(len(TRAINING_SEEDS))
        for family_id in family_slots
    ]
    counts = Counter(case.family_id for case in weighted)
    if (
        len(weighted) != WEIGHTED_CYCLE_CASE_COUNT
        or any(counts[family_id] != 32 * weight for family_id, weight in weights.items())
        or any(
            tuple(
                case.family_id
                for case in weighted[
                    start : start + WEIGHTED_FAMILY_SLOT_COUNT
                ]
            )
            != family_slots
            for start in range(0, len(weighted), WEIGHTED_FAMILY_SLOT_COUNT)
        )
    ):
        raise ModerateStudyError("weighted training cycle is not exactly 2:1")
    return weighted


def warmup_prefix_balance(
    hardest_family_ids: Sequence[str],
) -> dict[str, Any]:
    """Prove finite-prefix balance for the adopted 20-lane critic warm-up."""

    weights = family_weights(hardest_family_ids)
    cases = weighted_training_cases(hardest_family_ids)
    family_cycle = [case.family_id for case in cases]
    if {case.scenario.horizon_days for case in cases} != {TRAINING_EPISODE_DAYS}:
        raise ModerateStudyError("weighted TRAIN episodes no longer share a 30-day horizon")
    prefixes: list[dict[str, Any]] = []
    for transitions in CRITIC_WARMUP_PREFIX_TRANSITIONS:
        if transitions % TRAINING_LANES:
            raise ModerateStudyError("warm-up prefix does not divide across lanes")
        steps_per_lane = transitions // TRAINING_LANES
        episode_starts_per_lane = 1 + steps_per_lane // TRAINING_EPISODE_DAYS
        lane_rows: list[dict[str, Any]] = []
        maximum_lane_deviation = 0.0
        global_counts: Counter[str] = Counter()
        for lane in range(TRAINING_LANES):
            lane_counts = Counter(
                family_cycle[(lane + index) % len(family_cycle)]
                for index in range(episode_starts_per_lane)
            )
            deviations = {
                family_id: abs(
                    lane_counts[family_id]
                    - episode_starts_per_lane
                    * weight
                    / WEIGHTED_FAMILY_SLOT_COUNT
                )
                for family_id, weight in weights.items()
            }
            maximum_lane_deviation = max(
                maximum_lane_deviation,
                max(deviations.values()),
            )
            global_counts.update(lane_counts)
            lane_rows.append(
                {
                    "lane": lane,
                    "cycle_offset": lane,
                    "family_episode_starts": dict(lane_counts),
                    "maximum_absolute_deviation_from_weighted_target": max(
                        deviations.values()
                    ),
                }
            )
        if maximum_lane_deviation > 1.0:
            raise ModerateStudyError(
                "weighted warm-up prefix exceeds one episode of its target"
            )
        prefixes.append(
            {
                "global_transitions": transitions,
                "steps_per_lane": steps_per_lane,
                "episode_starts_per_lane": episode_starts_per_lane,
                "global_episode_starts": (
                    TRAINING_LANES * episode_starts_per_lane
                ),
                "global_family_episode_starts": dict(global_counts),
                "maximum_per_lane_absolute_deviation_from_weighted_target": (
                    maximum_lane_deviation
                ),
                "lanes": lane_rows,
            }
        )
    return {
        "lanes": TRAINING_LANES,
        "episode_horizon_days": TRAINING_EPISODE_DAYS,
        "lane_cycle_offset": "lane_index_modulo_weighted_cycle",
        "maximum_allowed_per_lane_episode_deviation": 1.0,
        "prefixes": prefixes,
    }


def weighted_training_scenarios(
    hardest_family_ids: Sequence[str],
) -> list[tuple[Any, int]]:
    """Return the scenario/tape pairs consumed by canonical trainer primitives."""

    return [
        (case.scenario, case.tape_seed)
        for case in weighted_training_cases(hardest_family_ids)
    ]


def sampler_contract(
    difficulty_path: Path,
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    """Bind the derived sampler to its training-only evidence."""

    hardest = difficulty["sampler"]["hardest_family_ids"]
    weighted_cases = weighted_training_cases(hardest)
    occurrences = [case.identity() for case in weighted_cases]
    return {
        "treatment": "training_family_reweighting_only",
        "selection_evidence_path": str(difficulty_path.resolve()),
        "selection_evidence_sha256": file_sha256(difficulty_path),
        "selection_split": "train",
        "selection_policy": "shipped_v4_onnx",
        "selection_policy_sha256": SHIPPED_ARTIFACT_SHA256,
        "development_evidence_used_for_weights": False,
        "hardest_family_ids": list(hardest),
        "family_weights": family_weights(hardest),
        "cycle_construction": (
            "32_seed_rounds_of_eight_interleaved_family_slots"
        ),
        "family_slot_pattern": [
            case.family_id
            for case in weighted_cases[:WEIGHTED_FAMILY_SLOT_COUNT]
        ],
        "canonical_case_count": TRAINING_CASE_COUNT,
        "weighted_cycle_case_count": WEIGHTED_CYCLE_CASE_COUNT,
        "weighted_cycle_sha256": canonical_hash(occurrences),
        "critic_warmup_prefix_balance": warmup_prefix_balance(hardest),
        "application_scope": [
            "behavior_cloning",
            "dagger_rollouts",
            "critic_warmup",
            "ppo_actor_critic_training",
        ],
    }


@dataclass(frozen=True)
class ModerateTrainingLaneFactory:
    """Spawn-safe deterministic lane over the weighted episode cycle."""

    lane: int
    hardest_family_ids: tuple[str, str]
    reward_profile: str = "v3_equivalent"
    preparedness_alignment_coefficient: float = 10.0

    def __call__(self) -> CyclingScenarioEnv:
        import torch

        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if os.environ.get(variable) != "1":
                raise ModerateStudyError(
                    f"spawned worker thread cap drifted: {variable}"
                )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        scenarios = weighted_training_scenarios(self.hardest_family_ids)
        offset = self.lane % len(scenarios)
        rotated = scenarios[offset:] + scenarios[:offset]
        return CyclingScenarioEnv(
            rotated,
            collect_evidence=False,
            reward_profile=self.reward_profile,
            preparedness_alignment_coefficient=(
                self.preparedness_alignment_coefficient
            ),
        )


def spawn_weighted_environment(
    lanes: int,
    seed: int,
    *,
    hardest_family_ids: Sequence[str],
    reward_profile: str = "v3_equivalent",
    preparedness_alignment_coefficient: float = 10.0,
) -> Any:
    """Create adopted SubprocVecEnv lanes over the weighted TRAIN cycle."""

    from stable_baselines3.common.vec_env import SubprocVecEnv

    hardest = tuple(hardest_family_ids)
    if len(hardest) != 2:
        raise ModerateStudyError("weighted environment requires two hard families")
    factories = [
        ModerateTrainingLaneFactory(
            lane=lane,
            hardest_family_ids=(hardest[0], hardest[1]),
            reward_profile=reward_profile,
            preparedness_alignment_coefficient=preparedness_alignment_coefficient,
        )
        for lane in range(lanes)
    ]
    environment = SubprocVecEnv(factories, start_method="spawn")
    environment.seed(seed)
    return environment


def weighted_roster_contract(
    difficulty_path: Path,
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    """Describe the exact weighted examples consumed by one trainer run."""

    contract = sampler_contract(difficulty_path, difficulty)
    return {
        "case_count": WEIGHTED_CYCLE_CASE_COUNT,
        "contract_sha256": contract["weighted_cycle_sha256"],
        "canonical_unique_case_count": TRAINING_CASE_COUNT,
        "sampling": contract,
    }


def run_one_weighted_training(
    *,
    difficulty_path: Path,
    output_root: Path,
    policy_seed: int,
) -> int:
    """Run the canonical trainer with only its TRAIN episode cycle substituted."""

    if policy_seed not in POLICY_SEEDS:
        raise ModerateStudyError("policy seed is outside the registered three-seed set")
    difficulty_path = difficulty_path.resolve()
    difficulty = validate_difficulty_receipt(difficulty_path)
    hardest = tuple(difficulty["sampler"]["hardest_family_ids"])
    if len(hardest) != 2:
        raise ModerateStudyError("difficulty receipt did not select two families")
    contract = sampler_contract(difficulty_path, difficulty)

    import scripts.train_policy as trainer

    original_training_scenarios = trainer.training_scenarios
    original_spawn_environment = trainer.spawn_environment
    original_roster_contract = trainer.training_roster_and_tapes_contract
    original_resolved_config = trainer.resolved_training_config

    def patched_training_scenarios() -> list[tuple[Any, int]]:
        return weighted_training_scenarios(hardest)

    def patched_spawn_environment(
        lanes: int,
        seed: int,
        *,
        reward_profile: str = "v3_equivalent",
        preparedness_alignment_coefficient: float = 10.0,
    ) -> Any:
        return spawn_weighted_environment(
            lanes,
            seed,
            hardest_family_ids=hardest,
            reward_profile=reward_profile,
            preparedness_alignment_coefficient=preparedness_alignment_coefficient,
        )

    def patched_roster_contract() -> dict[str, Any]:
        return weighted_roster_contract(difficulty_path, difficulty)

    def patched_resolved_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_resolved_config(*args, **kwargs)
        return {**value, "training_family_sampler": contract}

    trainer.training_scenarios = patched_training_scenarios
    trainer.spawn_environment = patched_spawn_environment
    trainer.training_roster_and_tapes_contract = patched_roster_contract
    trainer.resolved_training_config = patched_resolved_config
    output_root = output_root.resolve()
    receipt = output_root / "training-receipt.json"
    checkpoints = output_root / "checkpoints"
    argv = adopted_training_arguments(
        policy_seed=policy_seed,
        receipt_path=receipt,
        checkpoint_directory=checkpoints,
    )
    try:
        return int(trainer.main(argv))
    finally:
        trainer.training_scenarios = original_training_scenarios
        trainer.spawn_environment = original_spawn_environment
        trainer.training_roster_and_tapes_contract = original_roster_contract
        trainer.resolved_training_config = original_resolved_config


def adopted_training_arguments(
    *,
    policy_seed: int,
    receipt_path: Path,
    checkpoint_directory: Path,
) -> list[str]:
    """Return the exact adopted 2M optimizer invocation for one challenger seed."""

    if policy_seed not in POLICY_SEEDS:
        raise ModerateStudyError("policy seed is outside the registered set")
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
        str(policy_seed),
        "--bc-epochs",
        "15",
        "--learning-rate",
        "0.000075",
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
        "--freeze-observation-rms",
        "--critic-warmup-min-transitions",
        "50000",
        "--critic-warmup-max-transitions",
        "100000",
        "--checkpoint-dir",
        str(checkpoint_directory.resolve()),
        "--json-output",
        str(receipt_path.resolve()),
    ]


def development_family_aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and aggregate one exact 200-case DEV evaluation by family."""

    expected = [
        (family.id, case_seed, family.tape_seed(case_seed))
        for family in DEVELOPMENT_FAMILIES
        for case_seed in DEVELOPMENT_SEEDS
    ]
    if len(rows) != DEVELOPMENT_CASE_COUNT or len(expected) != DEVELOPMENT_CASE_COUNT:
        raise ModerateStudyError("development roster must contain exactly 200 rows")
    for index, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        family_id, case_seed, tape_seed = identity
        if (
            row.get("row_id") != f"{family_id}:{case_seed}"
            or row.get("case_seed") != case_seed
            or row.get("tape_seed") != tape_seed
            or not isinstance(row.get("solved"), bool)
            or int(row.get("hard_violation_count", -1)) != 0
            or float(row.get("max_conservation_residual", -1.0)) != 0.0
        ):
            raise ModerateStudyError(f"development row drifted at {index}")
    output: list[dict[str, Any]] = []
    for family in DEVELOPMENT_FAMILIES:
        family_rows = [row for row in rows if row["row_id"].startswith(f"{family.id}:")]
        solved = sum(bool(row["solved"]) for row in family_rows)
        output.append(
            {
                "family_id": family.id,
                "case_count": len(family_rows),
                "solved_count": solved,
                "solve_rate": solved / len(family_rows),
                "hard_violation_count": 0,
                "maximum_conservation_residual": 0.0,
            }
        )
    return output


def promotion_gate(
    selected_solved_count: int,
    endpoint_solved_counts: Sequence[int],
) -> dict[str, Any]:
    """Apply all three preregistered DEV-only promotion conditions."""

    if (
        isinstance(selected_solved_count, bool)
        or not isinstance(selected_solved_count, int)
        or not 0 <= selected_solved_count <= DEVELOPMENT_CASE_COUNT
        or len(endpoint_solved_counts) != len(POLICY_SEEDS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= DEVELOPMENT_CASE_COUNT
            for value in endpoint_solved_counts
        )
    ):
        raise ModerateStudyError("promotion evidence is outside its contract")
    endpoint_mean = fmean(endpoint_solved_counts)
    endpoints_at_or_above = sum(
        value >= PROMOTION_ENDPOINT_MINIMUM for value in endpoint_solved_counts
    )
    selected_passed = selected_solved_count >= PROMOTION_SELECTED_MINIMUM
    mean_passed = endpoint_mean > INCUMBENT_SEED_MEAN_SOLVED_COUNT
    reproducibility_passed = (
        endpoints_at_or_above >= PROMOTION_ENDPOINT_MINIMUM_COUNT
    )
    passed = selected_passed and mean_passed and reproducibility_passed
    return {
        "kind": "conjunctive_development_only_challenger_promotion",
        "conditions": {
            "selected_checkpoint_solved_count": {
                "operator": ">=",
                "threshold": PROMOTION_SELECTED_MINIMUM,
                "observed": selected_solved_count,
                "passed": selected_passed,
            },
            "three_seed_2m_endpoint_mean": {
                "operator": ">",
                "threshold": INCUMBENT_SEED_MEAN_SOLVED_COUNT,
                "observed": endpoint_mean,
                "passed": mean_passed,
            },
            "seed_endpoints_at_or_above_172": {
                "operator": ">=",
                "threshold_count": PROMOTION_ENDPOINT_MINIMUM_COUNT,
                "endpoint_threshold": PROMOTION_ENDPOINT_MINIMUM,
                "observed_count": endpoints_at_or_above,
                "observed_endpoints": list(endpoint_solved_counts),
                "passed": reproducibility_passed,
            },
        },
        "passed": passed,
        "decision": (
            "eligible_for_owner_review_no_final_evaluation"
            if passed
            else "retain_shipped_policy"
        ),
        "final_evaluation_authorized": False,
    }


def promotion_contract() -> dict[str, Any]:
    """Return thresholds without contaminating the protocol with observations."""

    return {
        "kind": "conjunctive_development_only_challenger_promotion",
        "all_conditions_required": True,
        "selected_checkpoint_solved_count": {
            "operator": ">=",
            "threshold": PROMOTION_SELECTED_MINIMUM,
        },
        "three_seed_2m_endpoint_mean": {
            "operator": ">",
            "threshold": INCUMBENT_SEED_MEAN_SOLVED_COUNT,
        },
        "seed_endpoint_reproducibility": {
            "operator": ">=",
            "required_seed_count": PROMOTION_ENDPOINT_MINIMUM_COUNT,
            "endpoint_solved_count_threshold": PROMOTION_ENDPOINT_MINIMUM,
            "registered_seed_count": len(POLICY_SEEDS),
        },
        "on_pass": "eligible_for_owner_review_no_final_evaluation",
        "on_fail": "retain_shipped_policy",
        "final_evaluation_authorized": False,
    }


def summarize_seed_endpoints(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exactly three 2M DEV endpoints and their family counts."""

    if len(rows) != len(POLICY_SEEDS) or [row.get("policy_seed") for row in rows] != list(
        POLICY_SEEDS
    ):
        raise ModerateStudyError("endpoint rows must match the registered seeds")
    solved = [int(row["solved_count"]) for row in rows]
    per_family = []
    for family in DEVELOPMENT_FAMILIES:
        counts = [
            next(
                int(item["solved_count"])
                for item in row["per_family"]
                if item["family_id"] == family.id
            )
            for row in rows
        ]
        per_family.append(
            {
                "family_id": family.id,
                "case_count_per_seed": len(DEVELOPMENT_SEEDS),
                "mean_solved_count": fmean(counts),
                "sample_std_solved_count": stdev(counts),
                "minimum_solved_count": min(counts),
                "maximum_solved_count": max(counts),
            }
        )
    return {
        "seed_count": len(rows),
        "mean_solved_count": fmean(solved),
        "population_std_solved_count": pstdev(solved),
        "sample_std_solved_count": stdev(solved),
        "minimum_solved_count": min(solved),
        "maximum_solved_count": max(solved),
        "mean_solve_rate": fmean(value / DEVELOPMENT_CASE_COUNT for value in solved),
        "incumbent_comparison": {
            "incumbent_mean_solved_count": INCUMBENT_SEED_MEAN_SOLVED_COUNT,
            "incumbent_population_std_solved_count": (
                INCUMBENT_SEED_POPULATION_STD_SOLVED_COUNT
            ),
            "challenger_minus_incumbent_mean_solved_count": (
                fmean(solved) - INCUMBENT_SEED_MEAN_SOLVED_COUNT
            ),
        },
        "per_family": per_family,
    }


def _require_external(path: Path, label: str) -> Path:
    """Resolve an absolute path outside the repository and filesystem root."""

    if not path.is_absolute():
        raise ModerateStudyError(f"{label} must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ModerateStudyError(f"{label} must be outside the repository")
    if resolved == Path(resolved.anchor):
        raise ModerateStudyError(f"{label} cannot be a filesystem root")
    return resolved


def _worktree_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _checkpoint_reference(
    reference: Any,
    *,
    seed: int,
    milestone: int,
) -> dict[str, Any]:
    """Verify and summarize one durable checkpoint bundle."""

    if not isinstance(reference, dict):
        raise ModerateStudyError("checkpoint reference is missing")
    manifest_path = Path(str(reference.get("manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    verified = verify_checkpoint_bundle(manifest_path.parent)
    expected_id = f"seed-{seed}-ppo-{milestone}"
    checkpoint = verified.manifest["checkpoint"]
    normalization = verified.manifest["normalization"]
    if (
        reference.get("checkpoint_id") != expected_id
        or reference.get("model_sha256") != checkpoint["file"]["sha256"]
        or reference.get("normalization_sha256")
        != normalization["file"]["sha256"]
        or reference.get("obs_rms_sha256")
        != normalization["observation_rms_sha256"]
    ):
        raise ModerateStudyError("checkpoint reference hash binding drifted")
    return {
        "checkpoint_id": expected_id,
        "bundle_path": str(verified.root),
        "manifest_path": str(verified.manifest_path),
        "manifest_sha256": file_sha256(verified.manifest_path),
        "checkpoint_sha256": checkpoint["file"]["sha256"],
        "normalization_sha256": normalization["file"]["sha256"],
        "observation_rms_sha256": normalization["observation_rms_sha256"],
    }


def _validated_curve(
    value: Any,
    *,
    seed: int,
    milestone: int,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModerateStudyError("development curve row is missing")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ModerateStudyError("development curve has no per-case rows")
    per_family = development_family_aggregate(rows)
    solved_count = sum(item["solved_count"] for item in per_family)
    if (
        value.get("active_actor_critic_transitions") != milestone
        or value.get("case_count") != DEVELOPMENT_CASE_COUNT
        or value.get("solved_count") != solved_count
        or float(value.get("solve_rate", -1.0))
        != solved_count / DEVELOPMENT_CASE_COUNT
        or int(value.get("hard_violation_count", -1)) != 0
        or float(value.get("maximum_conservation_residual", -1.0)) != 0.0
    ):
        raise ModerateStudyError("development curve aggregates drifted")
    return {
        "policy_seed": seed,
        "active_actor_critic_transitions": milestone,
        "case_count": DEVELOPMENT_CASE_COUNT,
        "solved_count": solved_count,
        "solve_rate": solved_count / DEVELOPMENT_CASE_COUNT,
        "mean_resilience_auc": value["mean_resilience_auc"],
        "mean_minimum_tail_margin": value["mean_minimum_tail_margin"],
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "per_family": per_family,
        "development_rows_sha256": canonical_hash(rows),
        "checkpoint": checkpoint,
    }


def validate_training_receipt(
    path: Path,
    *,
    difficulty_path: Path,
    difficulty: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Validate one adopted-config 2M run and return its registered curves."""

    receipt = _load_object(path, "moderate-family training receipt")
    config = receipt.get("config")
    checks = receipt.get("checks")
    expected_sampler = sampler_contract(difficulty_path, difficulty)
    expected_config = {
        "active_actor_critic_transitions": ACTIVE_TRANSITIONS,
        "critic_warmup_min_transitions": 50_000,
        "critic_warmup_max_transitions": 100_000,
        "lanes": 20,
        "n_steps_per_lane": 250,
        "rollout_size": 5_000,
        "batch_size": 500,
        "n_epochs": 5,
        "learning_rate": 7.5e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.003,
        "reward_profile": "v3_equivalent",
        "preparedness_alignment_coefficient": 10.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": -1.5,
        "target_kl": 0.02,
        "use_sde": False,
        "critic_explained_variance_threshold": 0.5,
        "vec_normalize": True,
        "freeze_observation_rms": True,
        "policy_seed": seed,
        "bc_epochs": 15,
        "bc_warm_start": True,
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "evaluation_milestones": list(EVALUATION_MILESTONES),
        "training_family_sampler": expected_sampler,
    }
    if (
        receipt.get("tool") != "train_policy.py"
        or receipt.get("status") != "complete"
        or receipt.get("training_split") != "train"
        or receipt.get("evaluation_split") != "dev"
        or receipt.get("final_split_used") is not False
        or not isinstance(config, dict)
        or any(config.get(key) != value for key, value in expected_config.items())
        or not isinstance(checks, dict)
        or checks.get("training_complete") is not True
        or checks.get("development_hard_violations_zero") is not True
        or checks.get("development_conservation_residuals_zero") is not True
        or receipt.get("behavior_cloning", {}).get("observation_count") != 30_720
        or abs(
            float(receipt.get("normalization", {}).get("observation_rms_count", -1.0))
            - 30_720.0001
        )
        > 1e-6
        or receipt.get("training_roster_and_tapes")
        != weighted_roster_contract(difficulty_path, difficulty)
    ):
        raise ModerateStudyError("moderate-family training receipt drifted")
    curves: list[dict[str, Any]] = []
    for milestone in CURVE_MILESTONES:
        checkpoint = _checkpoint_reference(
            receipt.get("checkpoint_bundles", {}).get(str(milestone)),
            seed=seed,
            milestone=milestone,
        )
        curves.append(
            _validated_curve(
                receipt.get("development_curve", {}).get(
                    f"ppo_{milestone}_transitions"
                ),
                seed=seed,
                milestone=milestone,
                checkpoint=checkpoint,
            )
        )
    if receipt.get("development") != receipt.get("development_curve", {}).get(
        "ppo_2000000_transitions"
    ):
        raise ModerateStudyError("2M endpoint is not the terminal development row")
    return {
        "policy_seed": seed,
        "training_receipt_path": str(path.resolve()),
        "training_receipt_sha256": file_sha256(path),
        "curves": curves,
        "endpoint": curves[-1],
    }


def select_development_checkpoint(
    seed_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select across nine registered checkpoints by solves and neutral ties."""

    candidates = [curve for row in seed_rows for curve in row["curves"]]
    if len(candidates) != len(POLICY_SEEDS) * len(CURVE_MILESTONES):
        raise ModerateStudyError("checkpoint candidate set must contain 9 rows")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["solved_count"]),
            int(row["active_actor_critic_transitions"]),
            int(row["policy_seed"]),
        ),
    )
    return ranked[0], ranked


def build_study_summary(
    *,
    difficulty_path: Path,
    study_root: Path,
) -> dict[str, Any]:
    """Build aggregate, per-family, selection, and promotion evidence."""

    difficulty = validate_difficulty_receipt(difficulty_path)
    seed_rows = [
        validate_training_receipt(
            study_root / f"seed-{seed}" / "training-receipt.json",
            difficulty_path=difficulty_path,
            difficulty=difficulty,
            seed=seed,
        )
        for seed in POLICY_SEEDS
    ]
    selected, ranked = select_development_checkpoint(seed_rows)
    endpoint_rows = [
        {
            "policy_seed": row["policy_seed"],
            **{
                key: value
                for key, value in row["endpoint"].items()
                if key != "policy_seed"
            },
        }
        for row in seed_rows
    ]
    endpoint_aggregate = summarize_seed_endpoints(endpoint_rows)
    endpoint_counts = [int(row["solved_count"]) for row in endpoint_rows]
    gate = promotion_gate(int(selected["solved_count"]), endpoint_counts)
    return {
        "schema_version": STUDY_SCHEMA,
        "tool": TOOL_ID,
        "phase": "three_seed_moderate_family_reweighting",
        "training_split": "weighted_train",
        "evaluation_split": "dev",
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "final_split_imported_or_used": False,
        "difficulty_evidence": {
            "path": str(difficulty_path.resolve()),
            "sha256": file_sha256(difficulty_path),
            "selection_policy": "shipped_v4_onnx",
            "selection_policy_sha256": SHIPPED_ARTIFACT_SHA256,
            "hardest_family_ids": difficulty["sampler"]["hardest_family_ids"],
            "family_weights": difficulty["sampler"]["family_weights"],
        },
        "registered_policy_seeds": list(POLICY_SEEDS),
        "registered_curve_milestones": list(CURVE_MILESTONES),
        "seed_rows": seed_rows,
        "two_million_endpoint_aggregate": endpoint_aggregate,
        "selection": {
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "candidate_count": len(ranked),
            "tie_break_order": [
                "earlier_active_actor_critic_transitions",
                "lower_policy_seed",
            ],
            "selected_checkpoint": selected,
            "current_shipped_checkpoint_solved_count": (
                CURRENT_SELECTED_SOLVED_COUNT
            ),
            "selected_minus_current_shipped_solved_count": (
                selected["solved_count"] - CURRENT_SELECTED_SOLVED_COUNT
            ),
            "ranked_candidates": ranked,
        },
        "promotion_gate": gate,
        "decision": gate["decision"],
        "final_evaluation_authorized": False,
    }


def _run_seed_subprocess(
    *,
    difficulty_path: Path,
    study_root: Path,
    seed: int,
) -> None:
    seed_root = study_root / f"seed-{seed}"
    receipt_path = seed_root / "training-receipt.json"
    if receipt_path.exists():
        difficulty = validate_difficulty_receipt(difficulty_path)
        validate_training_receipt(
            receipt_path,
            difficulty_path=difficulty_path,
            difficulty=difficulty,
            seed=seed,
        )
        return
    seed_root.mkdir(parents=True, exist_ok=True)
    log_path = seed_root / "trainer.log"
    if log_path.exists() or (seed_root / "checkpoints").exists():
        raise ModerateStudyError(
            f"partial seed directory requires inspection: {seed_root}"
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train-one",
        "--difficulty-receipt",
        str(difficulty_path),
        "--study-root",
        str(study_root),
        "--seed",
        str(seed),
    ]
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise ModerateStudyError(f"seed {seed} failed; see {log_path}")


def run_registered_study(difficulty_path: Path, study_root: Path) -> dict[str, Any]:
    """Run three sequential 20-lane seeds, then create the DEV-only summary."""

    if not _worktree_is_clean():
        raise ModerateStudyError("refusing to train from a dirty worktree")
    validate_difficulty_receipt(difficulty_path)
    protocol = {
        "schema_version": STUDY_SCHEMA,
        "tool": TOOL_ID,
        "phase": "preregistered_protocol",
        "difficulty_receipt_path": str(difficulty_path),
        "difficulty_receipt_sha256": file_sha256(difficulty_path),
        "source_identity": source_identity(),
        "policy_seeds": list(POLICY_SEEDS),
        "active_transitions_per_seed": ACTIVE_TRANSITIONS,
        "curve_milestones": list(CURVE_MILESTONES),
        "selection_metric": "development_solved_count",
        "promotion_rule": promotion_contract(),
        "final_evaluation_authorized": False,
    }
    protocol_path = study_root / "protocol.json"
    if study_root.exists():
        if not study_root.is_dir() or _load_object(
            protocol_path, "moderate-family protocol"
        ) != protocol:
            raise ModerateStudyError("existing study root has a different protocol")
    else:
        study_root.mkdir(parents=True, exist_ok=False)
        write_new_json(protocol_path, protocol)
    for seed in POLICY_SEEDS:
        _run_seed_subprocess(
            difficulty_path=difficulty_path,
            study_root=study_root,
            seed=seed,
        )
    summary = build_study_summary(
        difficulty_path=difficulty_path,
        study_root=study_root,
    )
    summary_path = study_root / "study-summary.json"
    if summary_path.exists():
        if _load_object(summary_path, "moderate-family study summary") != summary:
            raise ModerateStudyError("existing study summary drifted")
    else:
        write_new_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure-difficulty", action="store_true")
    mode.add_argument("--run-study", action="store_true")
    mode.add_argument("--train-one", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--difficulty-receipt", type=Path)
    parser.add_argument("--difficulty-output", type=Path)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.measure_difficulty:
        if args.difficulty_output is None:
            raise ModerateStudyError("--measure-difficulty needs --difficulty-output")
        if not _worktree_is_clean():
            raise ModerateStudyError(
                "refusing difficulty measurement from a dirty worktree"
            )
        output = _require_external(args.difficulty_output, "--difficulty-output")
        receipt = build_difficulty_receipt()
        write_new_json(output, receipt)
        print(
            json.dumps(
                {
                    "receipt": str(output),
                    "hardest_family_ids": receipt["sampler"][
                        "hardest_family_ids"
                    ],
                    "family_weights": receipt["sampler"]["family_weights"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.difficulty_receipt is None or args.study_root is None:
        raise ModerateStudyError(
            "study modes need --difficulty-receipt and --study-root"
        )
    difficulty_path = _require_external(
        args.difficulty_receipt, "--difficulty-receipt"
    )
    study_root = _require_external(args.study_root, "--study-root")
    if args.train_one:
        if args.seed is None:
            raise ModerateStudyError("--train-one needs --seed")
        if not _worktree_is_clean():
            raise ModerateStudyError("refusing training from a dirty worktree")
        seed_root = study_root / f"seed-{args.seed}"
        return run_one_weighted_training(
            difficulty_path=difficulty_path,
            output_root=seed_root,
            policy_seed=args.seed,
        )
    if args.run_study:
        summary = run_registered_study(difficulty_path, study_root)
    else:
        summary = build_study_summary(
            difficulty_path=difficulty_path,
            study_root=study_root,
        )
        write_new_json(study_root / "study-summary.json", summary)
    print(
        json.dumps(
            {
                "selected_checkpoint": summary["selection"][
                    "selected_checkpoint"
                ]["checkpoint"],
                "selected_solved_count": summary["selection"][
                    "selected_checkpoint"
                ]["solved_count"],
                "endpoint_aggregate": summary[
                    "two_million_endpoint_aggregate"
                ],
                "promotion_gate": summary["promotion_gate"],
                "final_evaluation_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except ModerateStudyError as error:
        raise SystemExit(f"moderate-family study failed: {error}") from error
