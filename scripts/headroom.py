#!/usr/bin/env python3
"""Measure achievable policy headroom on the development scenarios.

This tool does not select or export a policy and cannot authorize training or
final-split evaluation. It can compare imported policy outcomes with a
causal receding-horizon planner and an explicitly clairvoyant anytime CEM
search.  The latter establishes only an achieved lower bound, never a proof of
the true optimum or of physical infeasibility.
"""

from __future__ import annotations

import os
import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Callable, Sequence

# Spawned Windows workers inherit these values before importing NumPy/OR-Tools.
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

from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    load_json_object,
)
from backend.app.city.environment import (  # noqa: E402
    ACTION_SIZE,
    ENGINE_ID,
    OBSERVATION_SIZE,
    CityRecoveryEnv,
)
from backend.app.city.optimizer import BASELINE_ID, ortools_proposal  # noqa: E402
from backend.app.city.outcome import (  # noqa: E402
    CRITICAL_SERVICE_FLOOR,
    summarize_trajectory,
)
from backend.app.city.planners import (  # noqa: E402
    tuned_rule_action,
    weights_to_logits,
)
from backend.app.city.physics import (  # noqa: E402
    CONSTRAINT_TOLERANCE,
    SHOCK_BUDGET_FACTORS,
    SHOCK_IMPACTS,
    SHOCKS,
)
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    Shock,
    generate_disaster_tape,
)

TOOL_ID = "headroom.py"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    ROOT / "internal" / "developmental_runs" / "headroom-dev.json"
)
MPC_HORIZONS = (1, 3, 5)
MPC_FIXED_SEVERITY_RANGE = (0.10, 0.35)
MPC_FORECAST_ID = "constant-announced-risk-crn-v1"
WORKER_AFFINITY_ENV = "HEADROOM_WORKER_AFFINITY_MASK"
WORKER_PRIORITY_ENV = "HEADROOM_WORKER_PRIORITY"


class HeadroomError(RuntimeError):
    """Raised when the nonauthorizing headroom contract cannot be proved."""


@dataclass(frozen=True)
class HeadroomCase:
    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    scenario: Any
    schedule: tuple[Shock, ...]


@dataclass(frozen=True)
class PlannerResult:
    solved: bool
    minimum_tail_margin: float
    resilience_auc: float
    reason_codes: tuple[str, ...]
    hard_violation_count: int
    maximum_conservation_residual: float
    action_sequence_sha256: str | None
    trajectory_sha256: str | None = None

    def as_receipt(self) -> dict[str, Any]:
        return {
            "solved": self.solved,
            "minimum_tail_margin": round(self.minimum_tail_margin, 10),
            "resilience_auc": round(self.resilience_auc, 10),
            "reason_codes": list(self.reason_codes),
            "hard_violation_count": self.hard_violation_count,
            "maximum_conservation_residual": round(
                self.maximum_conservation_residual, 10
            ),
            "action_sequence_sha256": self.action_sequence_sha256,
            "trajectory_sha256": self.trajectory_sha256,
        }


@dataclass(frozen=True)
class PublicSnapshot:
    """Current causal simulator state with the tape and tape seed omitted."""

    scenario: Any
    trajectory: tuple[dict[str, Any], ...]
    state: dict[str, Any]
    context: Any
    day_index: int


@dataclass(frozen=True)
class OracleConfig:
    population: int
    elite_fraction: float
    min_iterations: int
    max_iterations: int
    patience: int
    initial_std: float
    std_floor: float
    smoothing: float


@dataclass(frozen=True)
class MPCConfig:
    population: int
    elite_count: int
    iterations: int
    fantasies: int
    initial_std: float
    std_floor: float
    smoothing: float


def configure_worker_runtime() -> dict[str, Any]:
    """Apply explicitly requested, receipt-bound Windows scheduling."""

    affinity_text = os.environ.get(WORKER_AFFINITY_ENV)
    priority = os.environ.get(WORKER_PRIORITY_ENV, "normal")
    evidence: dict[str, Any] = {
        "pid": os.getpid(),
        "affinity_mask": affinity_text,
        "priority": priority,
    }
    if sys.platform != "win32":
        if affinity_text is not None or priority != "normal":
            raise HeadroomError("worker affinity/priority controls require Windows")
        return evidence
    if affinity_text is None and priority == "normal":
        return evidence
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if affinity_text is not None:
        affinity = int(affinity_text, 0)
        if affinity <= 0 or not kernel32.SetProcessAffinityMask(
            handle, ctypes.c_size_t(affinity)
        ):
            raise HeadroomError(
                f"failed to set worker affinity: {ctypes.get_last_error()}"
            )
        evidence["affinity_mask_decimal"] = affinity
    priority_classes = {
        "normal": 0x00000020,
        "above_normal": 0x00008000,
        "high": 0x00000080,
    }
    if priority not in priority_classes:
        raise HeadroomError(f"unsupported worker priority: {priority}")
    if priority != "normal" and not kernel32.SetPriorityClass(
        handle, priority_classes[priority]
    ):
        raise HeadroomError(f"failed to set worker priority: {ctypes.get_last_error()}")
    return evidence


def action_sequence_sha256(actions: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_development_cases() -> list[HeadroomCase]:
    cases: list[HeadroomCase] = []
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            cases.append(
                HeadroomCase(
                    row_id=f"{family.id}:{case_seed}",
                    family_id=family.id,
                    case_seed=case_seed,
                    tape_seed=tape_seed,
                    scenario=scenario,
                    schedule=tuple(generate_disaster_tape(scenario, tape_seed)),
                )
            )
    if len(cases) != 40 or len({case.row_id for case in cases}) != 40:
        raise HeadroomError("development contract must contain 40 unique cases")
    if tuple(DEVELOPMENT_SEEDS) != tuple(range(820000, 820008)):
        raise HeadroomError("development seed contract drifted")
    return cases


def _result_from_trajectory(
    trajectory: Sequence[dict[str, Any]], actions: np.ndarray
) -> PlannerResult:
    if not trajectory or "absolute_outcome" not in trajectory[-1]:
        raise HeadroomError("rollout did not reach the terminal outcome")
    outcome = trajectory[-1]["absolute_outcome"]
    margin = float(
        np.min(
            np.asarray(outcome["tail_minimum_services"], dtype=np.float64)
            - np.asarray(outcome["recovery_targets"], dtype=np.float64)
        )
    )
    return PlannerResult(
        solved=bool(outcome["solved"]),
        minimum_tail_margin=margin,
        resilience_auc=float(outcome["resilience_auc"]),
        reason_codes=tuple(str(value) for value in outcome["reason_codes"]),
        hard_violation_count=int(outcome["hard_violation_count"]),
        maximum_conservation_residual=float(outcome["max_conservation_residual"]),
        action_sequence_sha256=action_sequence_sha256(actions),
    )


def rollout_actions(
    case: HeadroomCase,
    actions: np.ndarray,
    *,
    collect_evidence: bool,
) -> PlannerResult:
    sequence = np.asarray(actions, dtype=np.float64)
    if sequence.shape != (case.scenario.horizon_days, ACTION_SIZE):
        raise HeadroomError(f"{case.row_id} action sequence has shape {sequence.shape}")
    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=collect_evidence,
    )
    environment.reset(seed=case.tape_seed)
    terminated = False
    for action in sequence:
        _, _, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise HeadroomError(f"unexpected truncation for {case.row_id}")
    if not terminated:
        raise HeadroomError(f"incomplete action sequence for {case.row_id}")
    compact = _result_from_trajectory(environment.trajectory, sequence)
    if not collect_evidence:
        return compact
    summary = summarize_trajectory(TOOL_ID, environment.trajectory, case.scenario)
    evidence = PlannerResult(
        solved=bool(summary["absolute_outcome"]["solved"]),
        minimum_tail_margin=compact.minimum_tail_margin,
        resilience_auc=float(summary["rauc"]),
        reason_codes=tuple(summary["absolute_outcome"]["reason_codes"]),
        hard_violation_count=int(summary["hard_violation_count"]),
        maximum_conservation_residual=float(
            summary["max_logistics_conservation_residual"]
        ),
        action_sequence_sha256=compact.action_sequence_sha256,
        trajectory_sha256=str(summary["trajectory_sha256"]),
    )
    _assert_result_equivalent(compact, evidence, f"{case.row_id} evidence replay")
    return evidence


def tuned_rollout(case: HeadroomCase) -> tuple[PlannerResult, np.ndarray]:
    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=False,
    )
    observation, _ = environment.reset(seed=case.tape_seed)
    actions: list[np.ndarray] = []
    terminated = False
    while not terminated:
        action, _ = tuned_rule_action(observation)
        actions.append(action)
        observation, _, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise HeadroomError(f"unexpected tuned truncation for {case.row_id}")
    sequence = np.asarray(actions, dtype=np.float64)
    return _result_from_trajectory(environment.trajectory, sequence), sequence


def _assert_result_equivalent(
    left: PlannerResult, right: PlannerResult, label: str
) -> None:
    if (
        left.solved != right.solved
        or left.reason_codes != right.reason_codes
        or abs(left.minimum_tail_margin - right.minimum_tail_margin) > 1e-10
        or abs(left.resilience_auc - right.resilience_auc) > 1e-10
        or left.hard_violation_count != right.hard_violation_count
        or abs(left.maximum_conservation_residual - right.maximum_conservation_residual)
        > 1e-10
    ):
        raise HeadroomError(f"compact/full replay mismatch: {label}")


def lexicographic_key(result: PlannerResult) -> tuple[int, float, float]:
    return (
        int(result.solved),
        float(result.minimum_tail_margin),
        float(result.resilience_auc),
    )


def _prior_result(row: dict[str, Any]) -> PlannerResult:
    targets = np.asarray(row["recovery_targets"], dtype=np.float64)
    tail = np.asarray(row["tail_minimum_services"], dtype=np.float64)
    return PlannerResult(
        solved=bool(row["solved"]),
        minimum_tail_margin=float(np.min(tail - targets)),
        resilience_auc=float(row["resilience_auc"]),
        reason_codes=tuple(str(value) for value in row["reason_codes"]),
        hard_violation_count=int(row["hard_violation_count"]),
        maximum_conservation_residual=float(row["max_conservation_residual"]),
        action_sequence_sha256=None,
    )


def load_prior_evidence(
    receipt_path: Path,
    expected_row_ids: Sequence[str],
    expected_policy_seed: int | None = None,
) -> dict[str, Any]:
    """Load one nonauthorizing dev receipt for optional BC/PPO context.

    Headroom search is meaningful without this input.  When a receipt is
    supplied, its ordered per-case rows let the report distinguish policy
    failures from cases already solved by another causal policy.  The loader
    deliberately does not select among attempts or validate historical source
    trees; experiment selection belongs outside this diagnostic.
    """

    receipt = load_json_object(
        receipt_path,
        "prior policy receipt",
        error_type=HeadroomError,
    )
    if (
        receipt.get("authorizing") is not False
        or receipt.get("split") != "dev"
        or receipt.get(
            "final_split_used", receipt.get("uses_final_split")
        )
        is not False
        or receipt.get("selects_or_exports_policy") is not False
    ):
        raise HeadroomError("prior receipt is not nonauthorizing dev evidence")
    configured_seed = receipt.get("config", {}).get("policy_seed")
    if configured_seed is None:
        configured_seed = receipt.get("policy_seed")
    if expected_policy_seed is not None:
        if configured_seed is None or int(configured_seed) != expected_policy_seed:
            raise HeadroomError("prior receipt policy seed does not match")

    profiles = receipt.get("profiles")
    if not isinstance(profiles, dict) or len(profiles) != 1:
        raise HeadroomError("prior receipt must contain exactly one treatment profile")
    profile_id, profile = next(iter(profiles.items()))
    if not isinstance(profile, dict):
        raise HeadroomError("prior receipt treatment profile must be an object")
    curve = profile.get("development_curve")
    if not isinstance(curve, dict):
        raise HeadroomError("prior receipt has no development curve")
    bc_stage = curve.get("bc_initialization")
    if not isinstance(bc_stage, dict) or not isinstance(bc_stage.get("rows"), list):
        raise HeadroomError("prior receipt has no BC initialization rows")
    actor_stages = [
        (int(stage.get("active_actor_critic_transitions", 0)), name, stage)
        for name, stage in curve.items()
        if isinstance(stage, dict)
        and isinstance(stage.get("rows"), list)
        and int(stage.get("active_actor_critic_transitions", 0)) > 0
    ]
    if not actor_stages:
        raise HeadroomError("prior receipt has no actor-critic evaluation rows")
    actor_transitions, actor_stage_name, actor_stage = max(actor_stages)
    bc_rows = bc_stage["rows"]
    ppo_rows = actor_stage["rows"]
    expected = list(expected_row_ids)
    if [row["row_id"] for row in bc_rows] != expected:
        raise HeadroomError("BC prior rows do not match the ordered dev cases")
    if [row["row_id"] for row in ppo_rows] != expected:
        raise HeadroomError("PPO prior rows do not match the ordered dev cases")
    try:
        display_path = str(receipt_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        display_path = str(receipt_path)
    return {
        "receipt_path": display_path,
        "receipt_sha256": file_sha256(receipt_path),
        "policy_seed": None if configured_seed is None else int(configured_seed),
        "treatment_profile": str(profile_id),
        "actor_critic_stage": str(actor_stage_name),
        "active_actor_critic_transitions": actor_transitions,
        "selected_or_exported_policy": False,
        "limitation": (
            "BC and PPO rows are imported outcomes, not executions in this "
            "process; this diagnostic does not select or export their policy"
        ),
        "bc": {row["row_id"]: _prior_result(row) for row in bc_rows},
        "ppo": {row["row_id"]: _prior_result(row) for row in ppo_rows},
    }


def _glop_seed(
    context: Any, observation: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    priorities = np.asarray(observation[5:10], dtype=np.float64) * 2.0
    adapter = SimpleNamespace(
        shocked=np.asarray(context.shocked, dtype=np.float64),
        support=np.asarray(context.support, dtype=np.float64),
        available_budget=float(context.available_budget),
        lower=np.asarray(context.material_lower, dtype=np.float64),
        upper=np.asarray(context.material_upper, dtype=np.float64),
        stock_ready=np.asarray(context.stock_ready, dtype=np.float64),
        throughput=np.asarray(context.throughput, dtype=np.float64),
    )
    proposal, evidence = ortools_proposal(adapter, priorities)
    action, _ = tuned_rule_action(observation)
    action[:5] = weights_to_logits(proposal)
    action[5] = 1.0
    return action, {
        "baseline_id": BASELINE_ID,
        "solver": evidence["solver"],
        "status": evidence["status"],
        "allocation_solution": evidence["allocation_solution"],
        "scope": (
            "GLOP supplies only the current-day material seed; bounded raw "
            "logits still pass through exact projection and short CEM optimizes "
            "the complete nonlinear action window"
        ),
    }


def capture_public_snapshot(environment: CityRecoveryEnv) -> PublicSnapshot:
    context = environment.current_context()
    # The copied scenario is stripped of every authored future-shock channel.
    scenario = environment.scenario.model_copy(
        update={
            "name": "Causal MPC planning state",
            "forced_shock": None,
            "forced_shocks": [],
            "shock_probability": 0.0,
            "severity_min": MPC_FIXED_SEVERITY_RANGE[0],
            "severity_max": MPC_FIXED_SEVERITY_RANGE[1],
        }
    )
    state_names = (
        "_q",
        "_stocks",
        "_pending",
        "_damage_peak",
        "_damage_duration",
        "_damage_remaining",
        "_priorities",
        "_normalized_priorities",
        "_targets",
        "_critical_streak",
        "_preparedness",
        "_days_since_last_shock",
        "_previous_resilience",
        "_terminated",
    )
    return PublicSnapshot(
        scenario=scenario,
        trajectory=tuple(copy.deepcopy(environment.trajectory)),
        state={name: copy.deepcopy(getattr(environment, name)) for name in state_names},
        context=copy.deepcopy(context),
        day_index=int(environment._day_index),
    )


def _dummy_shock(day: int, *, assessment_tail: bool) -> Shock:
    return Shock(
        day=day,
        type=None,
        severity=0.0,
        impact=[0.0] * 5,
        budget_factor=0.0,
        forced=False,
        occurrence_probability=0.0,
        occurrence_draw=1.0,
        public_risk_before=[0.0] * 5,
        public_risk_next=[0.0] * 5,
        assessment_tail=assessment_tail,
    )


def _mpc_rng_seed(observation: np.ndarray, day_index: int, horizon: int) -> int:
    public = np.ascontiguousarray(np.asarray(observation, dtype=np.float32))
    digest = hashlib.sha256(
        b"headroom-mpc-v1"
        + int(day_index).to_bytes(2, "little", signed=False)
        + int(horizon).to_bytes(1, "little", signed=False)
        + public.tobytes()
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _fantasy_schedule(
    snapshot: PublicSnapshot,
    observation: np.ndarray,
    horizon: int,
    branch: int,
) -> list[Shock]:
    total_days = int(snapshot.scenario.horizon_days)
    tail_days = int(snapshot.scenario.assessment_tail_days)
    tail_start = total_days - tail_days + 1
    schedule = [
        _dummy_shock(day, assessment_tail=day >= tail_start)
        for day in range(1, total_days + 1)
    ]
    risk = np.clip(np.asarray(observation[68:73], dtype=np.float64), 0.0, 1.0)
    probability = min(1.0, float(np.sum(risk)))
    seed = _mpc_rng_seed(observation, snapshot.day_index, horizon)
    rng = np.random.Generator(np.random.PCG64(seed ^ ((branch + 1) * 0x9E3779B1)))
    current_day = snapshot.day_index + 1
    for day in range(current_day + 1, min(total_days, current_day + horizon - 1) + 1):
        assessment_tail = day >= tail_start
        draw = float(rng.random())
        if assessment_tail or probability <= 0.0 or draw >= probability:
            shock_type = None
            severity = 0.0
            impact = np.zeros(5, dtype=np.float64)
            budget_factor = 0.0
        else:
            normalized = risk / probability
            shock_index = min(
                int(np.searchsorted(np.cumsum(normalized), rng.random(), side="right")),
                4,
            )
            shock_type = SHOCKS[shock_index]
            low, high = MPC_FIXED_SEVERITY_RANGE
            severity = low + (high - low) * float(rng.beta(2.0, 4.5))
            impact = np.asarray(SHOCK_IMPACTS[shock_index], dtype=np.float64)
            budget_factor = float(SHOCK_BUDGET_FACTORS[shock_index])
        next_risk = np.zeros(5, dtype=np.float64) if day + 1 >= tail_start else risk
        schedule[day - 1] = Shock(
            day=day,
            type=shock_type,
            severity=float(round(severity, 8)),
            impact=[float(round(value, 8)) for value in impact],
            budget_factor=budget_factor,
            forced=False,
            occurrence_probability=0.0 if assessment_tail else probability,
            occurrence_draw=float(round(draw, 8)),
            public_risk_before=[float(value) for value in risk],
            public_risk_next=[float(value) for value in next_risk],
            assessment_tail=assessment_tail,
        )
    return schedule


def _template_from_snapshot(
    snapshot: PublicSnapshot,
    schedule: Sequence[Shock],
    *,
    include_history: bool,
) -> CityRecoveryEnv:
    environment = CityRecoveryEnv(
        snapshot.scenario,
        0,
        list(schedule),
        collect_evidence=False,
    )
    environment.reset(seed=0)
    # The transition state is fully represented by the explicit fields below.
    # Past records are needed only when this window reaches termination and
    # the terminal outcome must score the full 30-day trajectory. Avoiding the
    # nested history copy on preterminal candidate rollouts is a large, exact
    # performance win and does not change dynamics.
    environment.trajectory = (
        list(copy.deepcopy(snapshot.trajectory)) if include_history else []
    )
    for name, value in snapshot.state.items():
        setattr(environment, name, copy.deepcopy(value))
    environment._day_index = snapshot.day_index
    environment._context = copy.deepcopy(snapshot.context)
    environment.schedule = list(schedule)
    environment._provided_schedule = tuple(schedule)
    environment._headroom_snapshot = snapshot
    environment._headroom_include_history = include_history
    return environment


def _restore_planning_template(environment: CityRecoveryEnv) -> None:
    snapshot: PublicSnapshot = environment._headroom_snapshot
    include_history: bool = environment._headroom_include_history
    environment.trajectory = (
        list(copy.deepcopy(snapshot.trajectory)) if include_history else []
    )
    for name, value in snapshot.state.items():
        if isinstance(value, np.ndarray):
            setattr(environment, name, value.copy())
        else:
            setattr(environment, name, copy.deepcopy(value))
    environment._day_index = snapshot.day_index
    environment._context = copy.deepcopy(snapshot.context)


def _tuned_window(template: CityRecoveryEnv, horizon: int) -> np.ndarray:
    environment = template
    _restore_planning_template(environment)
    actions: list[np.ndarray] = []
    observation = environment._observation()
    for _ in range(horizon):
        action, _ = tuned_rule_action(observation)
        actions.append(action)
        observation, _, terminated, _, _ = environment.step(action)
        if terminated:
            break
    return np.asarray(actions, dtype=np.float64)


def _mpc_candidate_score(
    templates: Sequence[CityRecoveryEnv], actions: np.ndarray
) -> tuple[float, ...]:
    terminal = False
    solved: list[float] = []
    passed_checks: list[float] = []
    margins: list[float] = []
    aucs: list[float] = []
    critical_values: list[float] = []
    pending_values: list[float] = []
    endpoint_hits: list[float] = []
    for template in templates:
        environment = template
        _restore_planning_template(environment)
        start = len(environment.trajectory)
        for action in actions:
            _, _, terminated, _, _ = environment.step(action)
            if terminated:
                break
        new_records = environment.trajectory[start:]
        for record in new_records:
            if int(record["hard_violation_count"]) != 0:
                raise HeadroomError("MPC search candidate produced a hard violation")
            if any(
                float(value) != 0.0
                for value in record["logistics"]["conservation_residual"]
            ):
                raise HeadroomError(
                    "MPC search candidate produced a conservation residual"
                )
        if terminated:
            terminal = True
            outcome = environment.trajectory[-1]["absolute_outcome"]
            solved.append(float(outcome["solved"]))
            passed_checks.append(float(sum(outcome["checks"].values())))
            margins.append(
                float(
                    np.min(
                        np.asarray(outcome["tail_minimum_services"])
                        - np.asarray(outcome["recovery_targets"])
                    )
                )
            )
            aucs.append(float(outcome["resilience_auc"]))
            critical_values.append(
                float(
                    outcome["critical_service_day_cap"]
                    - outcome["critical_service_days"]
                )
            )
            pending_values.append(
                float(
                    np.min(
                        np.asarray(outcome["terminal_pending_capacity"])
                        - np.asarray(outcome["terminal_pending_arrivals"])
                    )
                )
            )
        else:
            targets = np.asarray(environment._targets, dtype=np.float64)
            endpoint_margin = float(np.min(environment._q - targets))
            margins.append(endpoint_margin)
            endpoint_hits.append(
                float(np.all(environment._q >= targets - CONSTRAINT_TOLERANCE))
            )
            aucs.append(fmean(float(row["resilience"]) for row in new_records))
            critical_values.append(
                -float(
                    sum(
                        np.count_nonzero(
                            np.asarray(row["services_end"]) < CRITICAL_SERVICE_FLOOR
                        )
                        for row in new_records
                    )
                )
            )
            pending_values.append(-float(np.sum(environment._pending)))
    q10_margin = float(np.quantile(np.asarray(margins), 0.10))
    if terminal:
        return (
            fmean(solved),
            fmean(passed_checks),
            q10_margin,
            fmean(margins),
            fmean(aucs),
            fmean(critical_values),
            fmean(pending_values),
        )
    return (
        fmean(endpoint_hits),
        q10_margin,
        fmean(margins),
        fmean(critical_values),
        fmean(aucs),
        fmean(pending_values),
    )


def _antithetic_samples(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    population: int,
) -> np.ndarray:
    half = math.ceil(population / 2)
    noise = rng.standard_normal((half,) + mean.shape)
    paired = np.concatenate((noise, -noise), axis=0)[:population]
    return np.clip(mean + std * paired, -1.0, 1.0)


def plan_mpc_action(
    snapshot: PublicSnapshot,
    observation: np.ndarray,
    *,
    horizon: int,
    previous_plan: np.ndarray | None,
    config: MPCConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    remaining = int(snapshot.scenario.horizon_days) - snapshot.day_index
    effective_horizon = min(horizon, remaining)
    branch_count = 1 if effective_horizon == 1 else config.fantasies
    schedules = [
        _fantasy_schedule(snapshot, observation, effective_horizon, branch)
        for branch in range(branch_count)
    ]
    include_history = effective_horizon == remaining
    templates = [
        _template_from_snapshot(snapshot, schedule, include_history=include_history)
        for schedule in schedules
    ]
    tuned = _tuned_window(templates[0], effective_horizon)
    glop_action, glop_evidence = _glop_seed(snapshot.context, observation)
    glop_sequence = tuned.copy()
    glop_sequence[0] = glop_action
    if previous_plan is not None and len(previous_plan) > 1:
        shifted = np.vstack((previous_plan[1:], tuned[-1]))[:effective_horizon]
        if len(shifted) < effective_horizon:
            shifted = np.vstack((shifted, tuned[len(shifted) :]))
        mean = shifted.copy()
    else:
        shifted = None
        mean = glop_sequence.copy()
    std = np.full_like(mean, config.initial_std)
    incumbent = mean.copy()
    incumbent_score = _mpc_candidate_score(templates, incumbent)
    rng = np.random.Generator(
        np.random.PCG64(_mpc_rng_seed(observation, snapshot.day_index, horizon))
    )
    score_trace: list[dict[str, Any]] = []
    candidate_evaluations = 0
    for iteration in range(config.iterations):
        candidates = _antithetic_samples(rng, mean, std, config.population)
        injections = [mean, tuned, glop_sequence, incumbent]
        if shifted is not None:
            injections.append(shifted)
        for index, injected in enumerate(injections[: config.population]):
            candidates[index] = injected
        scores = [_mpc_candidate_score(templates, item) for item in candidates]
        candidate_evaluations += len(candidates)
        order = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
        winner = order[0]
        if scores[winner] > incumbent_score:
            incumbent = candidates[winner].copy()
            incumbent_score = scores[winner]
        elite = candidates[order[: config.elite_count]]
        elite_mean = np.mean(elite, axis=0)
        elite_std = np.std(elite, axis=0)
        mean = (1.0 - config.smoothing) * mean + config.smoothing * elite_mean
        std = np.maximum(
            config.std_floor,
            (1.0 - config.smoothing) * std + config.smoothing * elite_std,
        )
        score_trace.append(
            {
                "iteration": iteration + 1,
                "incumbent_score": [
                    round(float(value), 10) for value in incumbent_score
                ],
            }
        )
    evidence = {
        "planner_id": "hybrid-glop-seeded-cem-mpc",
        "method": "hybrid-glop-seeded-cem",
        "horizon": horizon,
        "effective_horizon": effective_horizon,
        "future_tape_visible": False,
        "forecast_id": MPC_FORECAST_ID,
        "forecast_source": "exact observation[68:73] held constant within each solve",
        "forecast_rng_uses_actual_case_or_tape_seed": False,
        "fantasy_branch_count": branch_count,
        "candidate_evaluations": candidate_evaluations,
        "simulated_transitions": candidate_evaluations
        * branch_count
        * effective_horizon,
        "glop": glop_evidence,
        "score_trace": score_trace,
    }
    return incumbent[0].copy(), incumbent, evidence


def run_mpc_case(job: tuple[HeadroomCase, MPCConfig]) -> dict[str, Any]:
    worker_runtime = configure_worker_runtime()
    case, config = job
    horizons: dict[str, Any] = {}
    for horizon in MPC_HORIZONS:
        environment = CityRecoveryEnv(
            case.scenario,
            case.tape_seed,
            list(case.schedule),
            collect_evidence=False,
        )
        observation, _ = environment.reset(seed=case.tape_seed)
        previous_plan: np.ndarray | None = None
        actions: list[np.ndarray] = []
        day_evidence: list[dict[str, Any]] = []
        terminated = False
        while not terminated:
            snapshot = capture_public_snapshot(environment)
            action, previous_plan, evidence = plan_mpc_action(
                snapshot,
                observation,
                horizon=horizon,
                previous_plan=previous_plan,
                config=config,
            )
            actions.append(action)
            day_evidence.append(evidence)
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise HeadroomError(f"unexpected MPC truncation for {case.row_id}")
        sequence = np.asarray(actions, dtype=np.float64)
        result = _result_from_trajectory(environment.trajectory, sequence)
        horizons[str(horizon)] = {
            "result": result.as_receipt(),
            "actions": sequence.tolist(),
            "budget": {
                "candidate_evaluations": sum(
                    day["candidate_evaluations"] for day in day_evidence
                ),
                "simulated_transitions": sum(
                    day["simulated_transitions"] for day in day_evidence
                ),
                "glop_solves": len(day_evidence),
                "replans": len(day_evidence),
            },
            "days": day_evidence,
        }
    return {
        "row_id": case.row_id,
        "horizons": horizons,
        "worker_runtime": worker_runtime,
    }


def _result_from_receipt(value: dict[str, Any]) -> PlannerResult:
    return PlannerResult(
        solved=bool(value["solved"]),
        minimum_tail_margin=float(value["minimum_tail_margin"]),
        resilience_auc=float(value["resilience_auc"]),
        reason_codes=tuple(str(item) for item in value["reason_codes"]),
        hard_violation_count=int(value["hard_violation_count"]),
        maximum_conservation_residual=float(value["maximum_conservation_residual"]),
        action_sequence_sha256=value.get("action_sequence_sha256"),
        trajectory_sha256=value.get("trajectory_sha256"),
    )


def _evaluate_oracle_candidate(
    environment: CityRecoveryEnv, actions: np.ndarray, tape_seed: int
) -> PlannerResult:
    environment.reset(seed=tape_seed)
    terminated = False
    for action in actions:
        _, _, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise HeadroomError("oracle candidate unexpectedly truncated")
    if not terminated:
        raise HeadroomError("oracle candidate did not reach terminal state")
    return _result_from_trajectory(environment.trajectory, actions)


def _meaningfully_better(left: PlannerResult, right: PlannerResult) -> bool:
    if left.solved != right.solved:
        return left.solved
    margin_delta = left.minimum_tail_margin - right.minimum_tail_margin
    if margin_delta > 1e-6:
        return True
    if abs(margin_delta) <= 1e-6:
        return left.resilience_auc > right.resilience_auc + 1e-7
    return False


def run_oracle_case(
    job: tuple[HeadroomCase, np.ndarray, np.ndarray, OracleConfig],
) -> dict[str, Any]:
    worker_runtime = configure_worker_runtime()
    case, tuned_actions, mpc_actions, config = job
    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=False,
    )
    mean = np.asarray(tuned_actions, dtype=np.float64).copy()
    mpc_sequence = np.asarray(mpc_actions, dtype=np.float64)
    std = np.full_like(mean, config.initial_std)
    tuned_result = _evaluate_oracle_candidate(environment, mean, case.tape_seed)
    mpc_result = _evaluate_oracle_candidate(environment, mpc_sequence, case.tape_seed)
    if lexicographic_key(mpc_result) > lexicographic_key(tuned_result):
        incumbent = mpc_sequence.copy()
        incumbent_result = mpc_result
    else:
        incumbent = mean.copy()
        incumbent_result = tuned_result
    initial_key = lexicographic_key(incumbent_result)
    seed = int.from_bytes(
        hashlib.sha256(
            f"headroom-oracle-v1:{case.row_id}:{case.tape_seed}".encode("ascii")
        ).digest()[:8],
        "little",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    elite_count = max(1, math.ceil(config.population * config.elite_fraction))
    trace: list[dict[str, Any]] = []
    candidate_evaluations = 2
    simulated_transitions = 2 * case.scenario.horizon_days
    search_max_hard = max(
        tuned_result.hard_violation_count, mpc_result.hard_violation_count
    )
    search_max_residual = max(
        tuned_result.maximum_conservation_residual,
        mpc_result.maximum_conservation_residual,
    )
    stale_iterations = 0
    for iteration in range(config.max_iterations):
        candidates = _antithetic_samples(rng, mean, std, config.population)
        injections = (mean, tuned_actions, mpc_sequence, incumbent)
        for index, injected in enumerate(injections[: config.population]):
            candidates[index] = injected
        results = [
            _evaluate_oracle_candidate(environment, item, case.tape_seed)
            for item in candidates
        ]
        candidate_evaluations += len(candidates)
        simulated_transitions += len(candidates) * case.scenario.horizon_days
        search_max_hard = max(
            search_max_hard,
            max(result.hard_violation_count for result in results),
        )
        search_max_residual = max(
            search_max_residual,
            max(result.maximum_conservation_residual for result in results),
        )
        order = sorted(
            range(len(results)),
            key=lambda index: lexicographic_key(results[index]),
            reverse=True,
        )
        winner = order[0]
        prior_result = incumbent_result
        if lexicographic_key(results[winner]) > lexicographic_key(incumbent_result):
            incumbent = candidates[winner].copy()
            incumbent_result = results[winner]
        if _meaningfully_better(incumbent_result, prior_result):
            stale_iterations = 0
        else:
            stale_iterations += 1
        elite = candidates[order[:elite_count]]
        elite_mean = np.mean(elite, axis=0)
        elite_std = np.std(elite, axis=0)
        mean = (1.0 - config.smoothing) * mean + config.smoothing * elite_mean
        std = np.maximum(
            config.std_floor,
            (1.0 - config.smoothing) * std + config.smoothing * elite_std,
        )
        trace.append(
            {
                "iteration": iteration + 1,
                "solved": incumbent_result.solved,
                "minimum_tail_margin": round(incumbent_result.minimum_tail_margin, 10),
                "resilience_auc": round(incumbent_result.resilience_auc, 10),
                "stale_iterations": stale_iterations,
            }
        )
        if (
            iteration + 1 >= config.min_iterations
            and stale_iterations >= config.patience
        ):
            break
    replay = rollout_actions(case, incumbent, collect_evidence=True)
    _assert_result_equivalent(incumbent_result, replay, f"{case.row_id} oracle")
    if lexicographic_key(replay) < initial_key:
        raise HeadroomError(f"oracle regressed below its warm start: {case.row_id}")
    return {
        "row_id": case.row_id,
        "result": replay.as_receipt(),
        "actions": incumbent.tolist(),
        "budget": {
            "population": config.population,
            "elite_fraction": config.elite_fraction,
            "elite_count": elite_count,
            "iterations": len(trace),
            "candidate_evaluations": candidate_evaluations,
            "simulated_transitions": simulated_transitions,
            "initial_std": config.initial_std,
            "std_floor": config.std_floor,
            "smoothing": config.smoothing,
            "minimum_iterations": config.min_iterations,
            "maximum_iterations": config.max_iterations,
            "patience": config.patience,
        },
        "warm_start": {
            "mean": "tuned-rule mult=10.0 cap=0.50 full 30x22 sequence",
            "incumbents_injected": ["tuned_rule", "selected_global_mpc"],
            "initial_lexicographic_key": list(initial_key),
            "winner_not_worse_than_tuned_or_mpc": True,
        },
        "trace": trace,
        "search_wide_invariants": {
            "maximum_hard_violation_count": search_max_hard,
            "maximum_conservation_residual": search_max_residual,
        },
        "worker_runtime": worker_runtime,
    }


def aggregate_results(rows: Sequence[PlannerResult]) -> dict[str, Any]:
    if not rows:
        raise HeadroomError("cannot aggregate an empty planner result")
    reasons = Counter(
        reason for row in rows if not row.solved for reason in row.reason_codes
    )
    return {
        "case_count": len(rows),
        "solved_count": sum(row.solved for row in rows),
        "solve_rate": sum(row.solved for row in rows) / len(rows),
        "minimum_tail_margin": round(min(row.minimum_tail_margin for row in rows), 10),
        "mean_minimum_tail_margin": round(
            fmean(row.minimum_tail_margin for row in rows), 10
        ),
        "mean_resilience_auc": round(fmean(row.resilience_auc for row in rows), 10),
        "hard_violation_count": sum(row.hard_violation_count for row in rows),
        "maximum_conservation_residual": max(
            row.maximum_conservation_residual for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
    }


def select_best_mpc_k(
    rows_by_horizon: dict[int, dict[str, PlannerResult]],
) -> tuple[int, dict[str, Any]]:
    if set(rows_by_horizon) != set(MPC_HORIZONS):
        raise HeadroomError("MPC sweep must contain horizons 1, 3, and 5")
    aggregates = {
        horizon: aggregate_results(list(rows.values()))
        for horizon, rows in rows_by_horizon.items()
    }
    selected = max(
        MPC_HORIZONS,
        key=lambda horizon: (
            int(aggregates[horizon]["solved_count"]),
            float(aggregates[horizon]["mean_minimum_tail_margin"]),
            float(aggregates[horizon]["mean_resilience_auc"]),
            -horizon,
        ),
    )
    return selected, {
        "rule": (
            "one global horizon by solved count, then mean minimum tail "
            "margin, then mean resilience AUC, then shorter horizon"
        ),
        "selected_horizon": selected,
        "aggregates": {str(key): value for key, value in aggregates.items()},
    }


def classify_case(planners: dict[str, PlannerResult]) -> dict[str, str]:
    required = ("tuned_rule", "mpc", "oracle")
    if any(name not in planners for name in required):
        raise HeadroomError("case classification is missing a required planner")
    has_bc = "bc_initialization" in planners
    has_ppo = "best_ppo" in planners
    if has_bc != has_ppo:
        raise HeadroomError("BC and PPO prior evidence must be supplied together")
    every_solved = all(result.solved for result in planners.values())
    oracle_solved = planners["oracle"].solved
    reference_solved = (
        planners["best_ppo"].solved
        if has_ppo
        else planners["tuned_rule"].solved or planners["mpc"].solved
    )
    known_achievable = any(result.solved for result in planners.values())
    if every_solved:
        literal = "saturated"
    elif oracle_solved and not reference_solved:
        literal = "contested"
    elif not known_achievable:
        literal = "oracle_search_unsolved"
    else:
        literal = "achieved_nonunanimous"
    if has_ppo and reference_solved:
        decision_partition = "ppo_solved"
    elif not has_ppo and reference_solved:
        decision_partition = "causal_planner_solved"
    elif oracle_solved:
        decision_partition = "contested"
    elif known_achievable:
        decision_partition = "known_achievable_oracle_search_miss"
    else:
        decision_partition = "oracle_search_unsolved"
    return {
        "literal_taxonomy": literal,
        "decision_partition": decision_partition,
    }


def write_new_receipt(path: Path, receipt: dict[str, Any]) -> None:
    allowed = (ROOT / "internal" / "developmental_runs").resolve()
    target = path.resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise HeadroomError("receipt must stay under internal/developmental_runs") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise HeadroomError(f"refusing to overwrite existing receipt: {target}")
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise HeadroomError(
            f"refusing to overwrite existing receipt: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _run_parallel(
    jobs: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    *,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(worker, job): job[0].row_id for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            row_id = futures[future]
            results[row_id] = future.result()
            print(f"{label}: {completed}/{len(jobs)} {row_id}", flush=True)
    return [results[job[0].row_id] for job in jobs]


def _validate_result_invariants(planner: str, rows: dict[str, PlannerResult]) -> None:
    if len(rows) != 40:
        raise HeadroomError(f"{planner} did not produce exactly 40 cases")
    if any(row.hard_violation_count != 0 for row in rows.values()):
        raise HeadroomError(f"{planner} produced a hard violation")
    if any(row.maximum_conservation_residual != 0.0 for row in rows.values()):
        raise HeadroomError(f"{planner} produced a conservation residual")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--developmental-nonauthorizing",
        action="store_true",
        help="required acknowledgement that this is privileged dev-only analysis",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prior-evidence",
        type=Path,
        help="optional single nonauthorizing dev receipt containing BC/PPO rows",
    )
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument(
        "--worker-affinity-mask",
        help="optional Windows integer mask, for example 0xffff for P-core IDs 0-15",
    )
    parser.add_argument(
        "--worker-priority",
        choices=("normal", "above_normal", "high"),
        default="normal",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        help="optional policy-seed check for --prior-evidence",
    )
    parser.add_argument("--oracle-population", type=int, default=512)
    parser.add_argument("--oracle-elite-fraction", type=float, default=0.10)
    parser.add_argument("--oracle-min-iterations", type=int, default=20)
    parser.add_argument("--oracle-max-iterations", type=int, default=40)
    parser.add_argument("--oracle-patience", type=int, default=6)
    parser.add_argument("--oracle-initial-std", type=float, default=0.25)
    parser.add_argument("--oracle-std-floor", type=float, default=0.03)
    parser.add_argument("--oracle-smoothing", type=float, default=0.75)
    parser.add_argument("--mpc-population", type=int, default=48)
    parser.add_argument("--mpc-elite-count", type=int, default=6)
    parser.add_argument("--mpc-iterations", type=int, default=5)
    parser.add_argument("--mpc-fantasies", type=int, default=4)
    parser.add_argument("--mpc-initial-std", type=float, default=0.35)
    parser.add_argument("--mpc-std-floor", type=float, default=0.05)
    parser.add_argument("--mpc-smoothing", type=float, default=0.80)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.developmental_nonauthorizing:
        raise HeadroomError("--developmental-nonauthorizing is required")
    if not 1 <= args.workers <= 40:
        raise HeadroomError("workers must be in [1, 40]")
    if not 512 <= args.oracle_population <= 1024:
        raise HeadroomError("oracle population must be in the registered [512, 1024]")
    if not 20 <= args.oracle_min_iterations <= args.oracle_max_iterations <= 40:
        raise HeadroomError("oracle iterations must satisfy 20 <= min <= max <= 40")
    if not 0.0 < args.oracle_elite_fraction <= 0.2:
        raise HeadroomError("oracle elite fraction must be in (0, .2]")
    if args.mpc_elite_count <= 0 or args.mpc_elite_count > args.mpc_population:
        raise HeadroomError("invalid MPC elite count")
    if args.mpc_fantasies <= 0 or args.mpc_iterations <= 0:
        raise HeadroomError("MPC fantasies and iterations must be positive")
    if args.worker_affinity_mask is not None:
        try:
            affinity = int(args.worker_affinity_mask, 0)
        except ValueError as exc:
            raise HeadroomError("worker affinity mask must be an integer") from exc
        if affinity <= 0:
            raise HeadroomError("worker affinity mask must be positive")
        if sys.platform != "win32":
            raise HeadroomError("worker affinity mask requires Windows")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    if args.worker_affinity_mask is None:
        os.environ.pop(WORKER_AFFINITY_ENV, None)
    else:
        os.environ[WORKER_AFFINITY_ENV] = args.worker_affinity_mask
    os.environ[WORKER_PRIORITY_ENV] = args.worker_priority
    started = time.perf_counter()
    cases = build_development_cases()
    row_ids = [case.row_id for case in cases]
    prior: dict[str, Any] | None = None
    if args.prior_evidence is not None:
        prior_path = args.prior_evidence
        if not prior_path.is_absolute():
            prior_path = ROOT / prior_path
        prior = load_prior_evidence(
            prior_path.resolve(), row_ids, expected_policy_seed=args.policy_seed
        )

    print("tuned rule: replaying 40 development cases", flush=True)
    tuned_rows: dict[str, PlannerResult] = {}
    tuned_actions: dict[str, np.ndarray] = {}
    for case in cases:
        compact, sequence = tuned_rollout(case)
        replay = rollout_actions(case, sequence, collect_evidence=True)
        _assert_result_equivalent(compact, replay, f"{case.row_id} tuned")
        tuned_rows[case.row_id] = replay
        tuned_actions[case.row_id] = sequence

    mpc_config = MPCConfig(
        population=args.mpc_population,
        elite_count=args.mpc_elite_count,
        iterations=args.mpc_iterations,
        fantasies=args.mpc_fantasies,
        initial_std=args.mpc_initial_std,
        std_floor=args.mpc_std_floor,
        smoothing=args.mpc_smoothing,
    )
    print(
        f"MPC: sweeping k={MPC_HORIZONS} on {len(cases)} cases "
        f"with {args.workers} spawned workers",
        flush=True,
    )
    mpc_outputs = _run_parallel(
        [(case, mpc_config) for case in cases],
        run_mpc_case,
        workers=args.workers,
        label="MPC",
    )
    mpc_by_row = {item["row_id"]: item for item in mpc_outputs}
    mpc_rows_by_horizon: dict[int, dict[str, PlannerResult]] = {
        horizon: {
            row_id: _result_from_receipt(
                mpc_by_row[row_id]["horizons"][str(horizon)]["result"]
            )
            for row_id in row_ids
        }
        for horizon in MPC_HORIZONS
    }
    selected_horizon, mpc_selection = select_best_mpc_k(mpc_rows_by_horizon)
    selected_mpc_rows: dict[str, PlannerResult] = {}
    selected_mpc_actions: dict[str, np.ndarray] = {}
    for case in cases:
        payload = mpc_by_row[case.row_id]["horizons"][str(selected_horizon)]
        sequence = np.asarray(payload["actions"], dtype=np.float64)
        replay = rollout_actions(case, sequence, collect_evidence=True)
        _assert_result_equivalent(
            mpc_rows_by_horizon[selected_horizon][case.row_id],
            replay,
            f"{case.row_id} selected MPC",
        )
        selected_mpc_rows[case.row_id] = replay
        selected_mpc_actions[case.row_id] = sequence

    oracle_config = OracleConfig(
        population=args.oracle_population,
        elite_fraction=args.oracle_elite_fraction,
        min_iterations=args.oracle_min_iterations,
        max_iterations=args.oracle_max_iterations,
        patience=args.oracle_patience,
        initial_std=args.oracle_initial_std,
        std_floor=args.oracle_std_floor,
        smoothing=args.oracle_smoothing,
    )
    print(
        f"oracle: full 30x22 CEM on {len(cases)} true tapes with "
        f"population={oracle_config.population}",
        flush=True,
    )
    oracle_outputs = _run_parallel(
        [
            (
                case,
                tuned_actions[case.row_id],
                selected_mpc_actions[case.row_id],
                oracle_config,
            )
            for case in cases
        ],
        run_oracle_case,
        workers=args.workers,
        label="oracle",
    )
    oracle_by_row = {item["row_id"]: item for item in oracle_outputs}
    oracle_rows = {
        row_id: _result_from_receipt(oracle_by_row[row_id]["result"])
        for row_id in row_ids
    }
    if any(
        int(item["search_wide_invariants"]["maximum_hard_violation_count"]) != 0
        or float(item["search_wide_invariants"]["maximum_conservation_residual"]) != 0.0
        for item in oracle_outputs
    ):
        raise HeadroomError("oracle search-wide physics invariant failed")
    planner_rows = {
        "tuned_rule": tuned_rows,
        f"mpc_k{selected_horizon}": selected_mpc_rows,
        "clairvoyant_oracle_cem": oracle_rows,
    }
    if prior is not None:
        planner_rows["bc_initialization"] = prior["bc"]
        planner_rows["best_ppo_diagnostic"] = prior["ppo"]
    for planner, rows in planner_rows.items():
        _validate_result_invariants(planner, rows)

    table: list[dict[str, Any]] = []
    literal_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    achievable_count = 0
    contested_count = 0
    for case in cases:
        row_id = case.row_id
        planners = {
            "tuned_rule": tuned_rows[row_id],
            "mpc": selected_mpc_rows[row_id],
            "oracle": oracle_rows[row_id],
        }
        if prior is not None:
            planners["bc_initialization"] = prior["bc"][row_id]
            planners["best_ppo"] = prior["ppo"][row_id]
        classification = classify_case(planners)
        literal_counts[classification["literal_taxonomy"]] += 1
        partition_counts[classification["decision_partition"]] += 1
        known_achievable = any(result.solved for result in planners.values())
        achievable_count += int(known_achievable)
        reference_solved = (
            prior["ppo"][row_id].solved
            if prior is not None
            else tuned_rows[row_id].solved or selected_mpc_rows[row_id].solved
        )
        contested_count += int(oracle_rows[row_id].solved and not reference_solved)
        table.append(
            {
                "row_id": row_id,
                "family_id": case.family_id,
                "case_seed": case.case_seed,
                "tape_seed": case.tape_seed,
                "tape_sha256": canonical_hash(
                    [asdict(shock) for shock in case.schedule]
                ),
                "planners": {
                    name: result.as_receipt() for name, result in planners.items()
                },
                "known_achievable_lower_bound": known_achievable,
                "classification": classification,
                "diagnostic_action_sequences": {
                    "tuned_rule": tuned_actions[row_id].tolist(),
                    f"mpc_k{selected_horizon}": selected_mpc_actions[row_id].tolist(),
                    "clairvoyant_oracle_cem": oracle_by_row[row_id]["actions"],
                },
            }
        )

    totals = {
        planner: aggregate_results([rows[row_id] for row_id in row_ids])
        for planner, rows in planner_rows.items()
    }
    tuned_count = totals["tuned_rule"]["solved_count"]
    ppo_count = (
        totals["best_ppo_diagnostic"]["solved_count"] if prior is not None else None
    )
    mpc_count = totals[f"mpc_k{selected_horizon}"]["solved_count"]
    oracle_count = totals["clairvoyant_oracle_cem"]["solved_count"]
    reference_count = max(
        [tuned_count] + ([] if ppo_count is None else [int(ppo_count)])
    )
    mpc_clear_margin = mpc_count - reference_count
    mpc_clearly_better = mpc_clear_margin >= 2
    if mpc_clearly_better:
        decision_row = "mpc_distillation_pivot_indicated"
        recommendation = (
            "Planning clearly beats every available causal reference; recommend an "
            "MPC-distillation pivot, but do not start it without direction."
        )
    elif achievable_count >= 37 and contested_count >= 4:
        decision_row = "headroom"
        recommendation = (
            "Measured achievable lower bound and contested count show real "
            "headroom; a matched reward comparison remains meaningful before "
            "any full campaign."
        )
    elif achievable_count <= 34 and contested_count <= 1:
        decision_row = "empirical_saturation"
        recommendation = (
            "The exhausted diagnostic search matches the requested low-ceiling "
            "row; do not run the 8M campaign. This is empirical saturation, "
            "not a proof that unsolved cases are physically infeasible."
        )
    else:
        decision_row = "inconclusive_between_registered_rows"
        recommendation = (
            "The result matches no preregistered numeric row; stop for direction "
            "before a long campaign or a distillation pivot."
        )

    if any(
        result.hard_violation_count != 0 or result.maximum_conservation_residual != 0.0
        for rows in planner_rows.values()
        for result in rows.values()
    ):
        raise HeadroomError("global physics invariant failed")

    mpc_case_budgets = {
        row_id: {
            str(horizon): {
                "result": mpc_by_row[row_id]["horizons"][str(horizon)]["result"],
                "budget": mpc_by_row[row_id]["horizons"][str(horizon)]["budget"],
            }
            for horizon in MPC_HORIZONS
        }
        for row_id in row_ids
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "privileged_development_headroom_analysis_nonauthorizing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorizing": False,
        "authorizes_training": False,
        "selects_or_exports_policy": False,
        "privileged_diagnostic": True,
        "split": "dev",
        "final_split_used": False,
        "uses_final_split": False,
        "case_count": 40,
        "policy_seed": (
            prior["policy_seed"] if prior is not None else args.policy_seed
        ),
        "environment": {
            "id": ENGINE_ID,
            "outcome": "absolute_outcome",
            "action_dimension": ACTION_SIZE,
            "observation_dimension": OBSERVATION_SIZE,
            "executed_action_path": (
                "CityRecoveryEnv.step -> decode_action -> exact feasibility projection"
            ),
            "decoder_contract_note": (
                "all 22 raw action values pass through the production decoder "
                "and exact allocation projection"
            ),
        },
        "prior_policy_evidence": (
            None
            if prior is None
            else {
                key: value
                for key, value in prior.items()
                if key not in {"bc", "ppo"}
            }
        ),
        "planner_totals": totals,
        "mpc": {
            "method": "hybrid-glop-seeded-cem",
            "baseline_id": BASELINE_ID,
            "future_tape_visible": False,
            "information": (
                "current causal simulator state/history and exact public 73-value "
                "observation; forecasts use only observation[68:73]"
            ),
            "feed_forward_observation_only": False,
            "causal_state_history_access": True,
            "information_caveat": (
                "MPC receives no future tape, future draws, forced-shock schedule, "
                "case seed, or tape seed. It does receive exact causal simulator "
                "state accessed directly; that state is determined by the realized "
                "history in principle, but direct access is a state-estimation "
                "advantage over the deployed feed-forward PPO network."
            ),
            "forecast": {
                "id": MPC_FORECAST_ID,
                "announced_risk_held_constant_within_open_loop_solve": True,
                "fixed_severity_range": list(MPC_FIXED_SEVERITY_RANGE),
                "actual_case_or_tape_seed_used_for_planning_rng": False,
                "authored_forced_shocks_visible": False,
                "fantasy_branches": args.mpc_fantasies,
            },
            "config": asdict(mpc_config),
            "selection": mpc_selection,
            "case_budgets": mpc_case_budgets,
        },
        "oracle": {
            "method": "full-sequence-diagonal-cem",
            "future_tape_visible": True,
            "objective": ["solved", "minimum_tail_margin", "resilience_auc"],
            "interpretation": (
                "anytime achieved lower bound; never an exact optimum or an "
                "infeasibility certificate"
            ),
            "config": asdict(oracle_config),
            "solved_count": oracle_count,
            "ceiling_statement": f"ceiling >= {achievable_count}/40",
            "cases": {
                row_id: {
                    key: value
                    for key, value in oracle_by_row[row_id].items()
                    if key != "actions"
                }
                for row_id in row_ids
            },
        },
        "feasibility": {
            "literal_taxonomy_counts": dict(sorted(literal_counts.items())),
            "decision_partition_counts": dict(sorted(partition_counts.items())),
            "requested_three_counts_with_honesty_label": {
                "saturated_every_planner_solved": literal_counts["saturated"],
                "contested_oracle_solved_reference_failed": literal_counts[
                    "contested"
                ],
                "infeasible_not_proven_oracle_search_unsolved": literal_counts[
                    "oracle_search_unsolved"
                ],
            },
            "residual_nonunanimous_count": literal_counts["achieved_nonunanimous"],
            "taxonomy_note": (
                "The requested three literal classes are not exhaustive when "
                "the reference and oracle solve but a weaker planner fails. The "
                "residual category makes all 40 rows explicit. Oracle search "
                "failure is not proof of physical infeasibility."
            ),
            "achievable_lower_bound_solved_count": achievable_count,
            "oracle_cem_solved_count": oracle_count,
            "contested_count": contested_count,
        },
        "decision": {
            "row": decision_row,
            "recommendation": recommendation,
            "mpc_clearly_better_definition": (
                "selected global MPC horizon solves at least two more cases "
                "than every available causal policy reference"
            ),
            "mpc_solve_margin_over_stronger_reference": mpc_clear_margin,
            "matched_reward_comparison_started": False,
            "campaign_8m_started": False,
            "distillation_pivot_started": False,
            "true_ceiling_upper_bound_established": False,
            "decision_rule_applied_as_empirical_proxy": True,
            "stop_for_direction": True,
        },
        "rows": table,
        "invariants": {
            "development_case_count_exactly_40": len(table) == 40,
            "development_row_ids_unique": len({row["row_id"] for row in table}) == 40,
            "development_seed_interval_exact": list(DEVELOPMENT_SEEDS)
            == list(range(820000, 820008)),
            "same_true_tape_for_newly_executed_tuned_mpc_oracle": True,
            "prior_bc_ppo_rows_hash_validated_and_order_aligned": prior is not None,
            "prior_evidence_supplied": prior is not None,
            "prior_evidence_limitation": (
                None
                if prior is None
                else (
                    "Imported policy rows are receipt evidence rather than "
                    "executions in this process"
                )
            ),
            "mpc_future_tape_visible": False,
            "oracle_future_tape_visible": True,
            "all_actions_used_decoder_and_projection": True,
            "all_hard_violation_counts_zero": True,
            "all_maximum_conservation_residuals_exactly_zero": True,
        },
        "runtime": {
            "workers": args.workers,
            "multiprocessing_start_method": "spawn",
            "requested_worker_affinity_mask": args.worker_affinity_mask,
            "requested_worker_priority": args.worker_priority,
            "mpc_worker_runtime_records": sorted(
                {
                    canonical_hash(item["worker_runtime"]): item["worker_runtime"]
                    for item in mpc_outputs
                }.values(),
                key=lambda value: int(value["pid"]),
            ),
            "oracle_worker_runtime_records": sorted(
                {
                    canonical_hash(item["worker_runtime"]): item["worker_runtime"]
                    for item in oracle_outputs
                }.values(),
                key=lambda value: int(value["pid"]),
            ),
            "native_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    write_new_receipt(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "tuned": tuned_count,
                "bc": (
                    totals["bc_initialization"]["solved_count"]
                    if prior is not None
                    else None
                ),
                "best_ppo_diagnostic": ppo_count,
                "mpc_best_k": selected_horizon,
                "mpc": mpc_count,
                "oracle_cem": oracle_count,
                "ceiling_statement": f"ceiling >= {achievable_count}/40",
                "contested": contested_count,
                "decision": decision_row,
                "receipt": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except HeadroomError as error:
        print(f"headroom analysis failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
