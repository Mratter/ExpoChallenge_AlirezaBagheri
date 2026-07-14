from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from backend.app.models import Scenario

SERVICES = ("transport", "housing", "food", "healthcare", "public_services")
SHOCKS = ("aftershock", "supply", "epidemic", "utility", "weather")
SHOCK_TYPE_PROBABILITIES = np.array([0.24, 0.22, 0.18, 0.20, 0.16], dtype=np.float64)
SHOCK_IMPACTS = np.array(
    [
        [0.65, 1.00, 0.20, 0.35, 0.45],
        [0.35, 0.05, 1.00, 0.55, 0.10],
        [0.10, 0.20, 0.25, 1.00, 0.35],
        [0.30, 0.35, 0.45, 0.70, 1.00],
        [0.75, 0.55, 0.50, 0.40, 0.60],
    ],
    dtype=np.float64,
)
SHOCK_BUDGET_FACTORS = np.array([0.15, 0.25, 0.10, 0.30, 0.25], dtype=np.float64)
DEPENDENCIES = np.array(
    [
        [0.00, 0.10, 0.10, 0.20, 0.60],
        [0.30, 0.00, 0.15, 0.10, 0.45],
        [0.45, 0.10, 0.00, 0.15, 0.30],
        [0.30, 0.10, 0.20, 0.00, 0.40],
        [0.35, 0.20, 0.15, 0.30, 0.00],
    ],
    dtype=np.float64,
)
ETA = np.array([0.18, 0.16, 0.20, 0.22, 0.17], dtype=np.float64)
DELTA = np.array([0.010, 0.012, 0.015, 0.018, 0.010], dtype=np.float64)


@dataclass(frozen=True)
class Shock:
    day: int
    type: str | None
    severity: float
    impact: list[float]
    budget_factor: float
    forced: bool


def _round_vector(values: np.ndarray) -> list[float]:
    return [float(round(value, 8)) for value in values.tolist()]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_shock_schedule(scenario: Scenario, seed: int) -> list[Shock]:
    """Generate the entire PCG64 tape once, then apply a deterministic forced override."""
    rng = np.random.Generator(np.random.PCG64(seed))
    schedule: list[Shock] = []
    for day in range(1, scenario.horizon_days + 1):
        occurs = bool(rng.random() < scenario.shock_probability)
        shock_index = int(rng.choice(len(SHOCKS), p=SHOCK_TYPE_PROBABILITIES))
        severity_draw = scenario.severity_min + (
            scenario.severity_max - scenario.severity_min
        ) * float(rng.beta(2.0, 5.0))
        if occurs:
            schedule.append(
                Shock(
                    day=day,
                    type=SHOCKS[shock_index],
                    severity=float(round(severity_draw, 8)),
                    impact=_round_vector(SHOCK_IMPACTS[shock_index]),
                    budget_factor=float(SHOCK_BUDGET_FACTORS[shock_index]),
                    forced=False,
                )
            )
        else:
            schedule.append(Shock(day, None, 0.0, [0.0] * 5, 0.0, False))

    forced = scenario.forced_shock
    if forced is not None:
        shock_index = SHOCKS.index(forced.type)
        schedule[forced.day - 1] = Shock(
            day=forced.day,
            type=forced.type,
            severity=float(forced.severity),
            impact=_round_vector(SHOCK_IMPACTS[shock_index]),
            budget_factor=float(SHOCK_BUDGET_FACTORS[shock_index]),
            forced=True,
        )
    return schedule


def project_capped_simplex(
    proposal: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Euclidean projection x=clip(y-lambda, lower, upper), using 64 bisections."""
    if float(lower.sum()) > total + 1e-9 or float(upper.sum()) < total - 1e-9:
        raise ValueError("allocation constraints are infeasible")
    lo = float(np.min(proposal - upper))
    hi = float(np.max(proposal - lower))
    for _ in range(64):
        midpoint = (lo + hi) / 2.0
        candidate = np.clip(proposal - midpoint, lower, upper)
        if float(candidate.sum()) > total:
            lo = midpoint
        else:
            hi = midpoint
    projected = np.clip(proposal - ((lo + hi) / 2.0), lower, upper)
    rounded = np.round(projected, 8)
    residual = round(float(total - rounded.sum()), 8)
    if residual:
        order = (
            np.argsort(-(upper - rounded))
            if residual > 0
            else np.argsort(-(rounded - lower))
        )
        for index in order:
            capacity = (
                upper[index] - rounded[index]
                if residual > 0
                else rounded[index] - lower[index]
            )
            adjustment = np.sign(residual) * min(abs(residual), float(capacity))
            rounded[index] = round(float(rounded[index] + adjustment), 8)
            residual = round(float(residual - adjustment), 8)
            if residual == 0:
                break
    bindings = [
        {
            "service": SERVICES[index],
            "lower": bool(abs(rounded[index] - lower[index]) <= 1e-7),
            "upper": bool(abs(rounded[index] - upper[index]) <= 1e-7),
        }
        for index in range(5)
    ]
    return rounded, {
        "distance": round(float(np.linalg.norm(rounded - proposal)), 8),
        "bindings": bindings,
        "sum": round(float(rounded.sum()), 8),
    }


def _normalize_proposal(scores: np.ndarray, budget: float) -> np.ndarray:
    score_sum = float(scores.sum())
    if score_sum <= 0:
        raise ValueError("planner produced a non-positive proposal")
    return budget * scores / score_sum


def urgency_proposal(q: np.ndarray, priorities: np.ndarray, budget: float) -> np.ndarray:
    threshold = np.where(q < 0.30, 2.5, 1.0)
    scores = priorities * (1.0 - q) * threshold
    return _normalize_proposal(scores, budget)


def policy_proposal(
    q: np.ndarray,
    priorities: np.ndarray,
    budget: float,
    support: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    deficit = priorities * (1.0 - q)
    criticality = deficit * np.where(q < 0.30, 2.5, 1.0)
    marginal = priorities * ETA * support * (1.0 - q)
    centrality = priorities * (1.0 - q) * DEPENDENCIES.sum(axis=0)
    features = {
        "priority_deficit": deficit / max(float(deficit.max()), 1e-12),
        "criticality": criticality / max(float(criticality.max()), 1e-12),
        "marginal_gain": marginal / max(float(marginal.max()), 1e-12),
        "network_centrality": centrality / max(float(centrality.max()), 1e-12),
    }
    scores = sum(weights[name] * feature for name, feature in features.items())
    return _normalize_proposal(scores, budget)


def _run_planner(
    planner: Literal["urgency_baseline", "frozen_policy"],
    scenario: Scenario,
    schedule: list[Shock],
    policy_weights: dict[str, float],
) -> dict[str, Any]:
    q = np.asarray(scenario.initial_services, dtype=np.float64)
    priorities = np.asarray(scenario.priorities, dtype=np.float64)
    normalized_priorities = priorities / priorities.sum()
    trajectory: list[dict[str, Any]] = []
    resilience_values: list[float] = []
    total_projection_distance = 0.0
    constraint_violations = 0

    for shock in schedule:
        before = q.copy()
        impact = np.asarray(shock.impact, dtype=np.float64)
        shocked = np.clip(q * (1.0 - shock.severity * impact), 0.0, 1.0)
        available_budget = scenario.daily_budget * (1.0 - shock.severity * shock.budget_factor)
        support = 0.55 + 0.45 * (DEPENDENCIES @ shocked)
        if planner == "urgency_baseline":
            raw = urgency_proposal(shocked, priorities, available_budget)
        else:
            raw = policy_proposal(
                shocked, priorities, available_budget, support, policy_weights
            )
        lower = np.where(shocked < 0.30, 0.04 * available_budget, 0.0)
        upper = np.full(5, 0.50 * available_budget, dtype=np.float64)
        allocation, projection = project_capped_simplex(raw, available_budget, lower, upper)
        daily_violations = int(abs(float(allocation.sum()) - available_budget) > 1e-7)
        daily_violations += int(np.count_nonzero(allocation < lower - 1e-7))
        daily_violations += int(np.count_nonzero(allocation > upper + 1e-7))
        projection["constraint_violations"] = daily_violations
        constraint_violations += daily_violations
        total_projection_distance += projection["distance"]
        gain = ETA * np.sqrt(allocation / 200.0) * support * (1.0 - shocked)
        strain = DELTA * np.maximum(0.0, 0.35 - shocked) * (1.0 - allocation / available_budget)
        q = np.clip(shocked + gain - strain, 0.0, 1.0)
        resilience = float(normalized_priorities @ q)
        resilience_values.append(resilience)
        trajectory.append(
            {
                "day": shock.day,
                "shock": asdict(shock),
                "available_budget": round(float(available_budget), 8),
                "services_before": _round_vector(before),
                "services_after_shock": _round_vector(shocked),
                "raw_proposal": _round_vector(raw),
                "allocation": _round_vector(allocation),
                "projection": projection,
                "support": _round_vector(support),
                "gain": _round_vector(gain),
                "strain": _round_vector(strain),
                "services_end": _round_vector(q),
                "resilience": round(resilience, 8),
            }
        )
    rauc = float(np.mean(resilience_values))
    return {
        "planner": planner,
        "rauc": round(rauc, 8),
        "final_resilience": round(resilience_values[-1], 8),
        "minimum_resilience": round(min(resilience_values), 8),
        "total_projection_distance": round(total_projection_distance, 8),
        "constraint_violations": constraint_violations,
        "trajectory": trajectory,
    }


def compare(
    scenario: Scenario, seed: int, policy: dict[str, Any], policy_sha: str
) -> dict[str, Any]:
    schedule = generate_shock_schedule(scenario, seed)
    schedule_payload = [asdict(shock) for shock in schedule]
    weights = policy["feature_weights"]
    baseline = _run_planner("urgency_baseline", scenario, schedule, weights)
    candidate = _run_planner("frozen_policy", scenario, schedule, weights)
    delta = candidate["rauc"] - baseline["rauc"]
    if delta > 1e-8:
        outcome = "candidate_higher_rauc"
    elif delta < -1e-8:
        outcome = "baseline_higher_rauc"
    else:
        outcome = "rauc_tie"
    return {
        "schema_version": "1.0.0",
        "seed": seed,
        "generator": "numpy.PCG64",
        "scenario": scenario.model_dump(mode="json"),
        "services": list(SERVICES),
        "shock_schedule": schedule_payload,
        "shock_schedule_sha256": canonical_hash(schedule_payload),
        "policy": {
            "id": policy["id"],
            "artifact_type": policy["artifact_type"],
            "sha256": policy_sha,
            "disclosure": policy["disclosure"],
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "primary_metric": "weighted_daily_resilience_auc",
            "candidate_minus_baseline": round(delta, 8),
            "outcome": outcome,
        },
        "limitations": [
            "All dynamics and policy calibration inputs are synthetic and non-empirical.",
            "This frozen deterministic policy candidate is not PPO and is not deployment guidance.",
        ],
    }
