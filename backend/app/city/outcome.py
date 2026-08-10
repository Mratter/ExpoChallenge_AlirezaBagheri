"""Define and summarize the canonical absolute city-recovery outcome."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from backend.app.city.physics import (
    CONSTRAINT_TOLERANCE,
    DEPOT_CAPACITY,
    round_vector,
)
from backend.app.models import Scenario as ScenarioModel
from backend.app.shared_evidence import canonical_hash

SOLVED_RAUC_FLOOR = 0.44
CRITICAL_SERVICE_FLOOR = 0.30
CRITICAL_SERVICE_RATE_CAP = 0.08
CONSERVATION_TOLERANCE = 1e-6
TERMINAL_PENDING_CAPACITY_MULTIPLIER = 1.0

SOLVED_DEFINITION: dict[str, Any] = {
    "id": "city-recovery-solved-v3",
    "version": "1.0.0",
    "assessment_tail_days": 3,
    "all_services_meet_public_targets_for_entire_tail": True,
    "resilience_auc_floor": SOLVED_RAUC_FLOOR,
    "critical_service_floor": CRITICAL_SERVICE_FLOOR,
    "critical_service_day_rate_cap": CRITICAL_SERVICE_RATE_CAP,
    "hard_violation_count": 0,
    "max_conservation_residual": CONSERVATION_TOLERANCE,
    "terminal_pending_at_most_capacity_multiplier": (
        TERMINAL_PENDING_CAPACITY_MULTIPLIER
    ),
}
SOLVED_DEFINITION_SHA256 = canonical_hash(SOLVED_DEFINITION)


def absolute_outcome(
    trajectory: Sequence[dict[str, Any]],
    recovery_targets: Sequence[float],
    assessment_tail_days: int,
) -> dict[str, Any]:
    """Apply the six-check absolute success definition to one trajectory."""

    if not trajectory:
        raise ValueError("v3 outcome requires a nonempty trajectory")
    targets = np.asarray(recovery_targets, dtype=np.float64)
    services = np.asarray(
        [day["services_end"] for day in trajectory], dtype=np.float64
    )
    resilience = np.asarray(
        [day["resilience"] for day in trajectory], dtype=np.float64
    )
    if assessment_tail_days != 3 or len(trajectory) < assessment_tail_days:
        raise ValueError("v3 outcome requires the frozen three-day assessment tail")
    if targets.shape != (5,) or services.shape != (len(trajectory), 5):
        raise ValueError("v3 outcome service vectors must contain exactly five values")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(services)):
        raise ValueError("v3 outcome inputs must be finite")
    hard_violations = int(sum(day["hard_violation_count"] for day in trajectory))
    max_residual = max(
        abs(value)
        for day in trajectory
        for value in day["logistics"]["conservation_residual"]
    )
    critical_service_days = int(
        np.count_nonzero(services < CRITICAL_SERVICE_FLOOR)
    )
    critical_cap = int(np.floor(CRITICAL_SERVICE_RATE_CAP * services.size))
    tail = services[-assessment_tail_days:]
    terminal_pending = np.asarray(
        trajectory[-1]["logistics"]["pending_next_day"], dtype=np.float64
    )
    if terminal_pending.shape != (5,) or not np.all(np.isfinite(terminal_pending)):
        raise ValueError("v3 terminal pending vector must contain five finite values")
    target_met_by_service = np.all(
        tail >= targets - CONSTRAINT_TOLERANCE, axis=0
    )
    tail_targets_met = bool(np.all(target_met_by_service))
    rauc = float(np.mean(resilience))
    checks = {
        "zero_hard_violations": hard_violations == 0,
        "conservation_verified": max_residual <= CONSERVATION_TOLERANCE,
        "assessment_tail_targets_met": tail_targets_met,
        "resilience_auc_met": rauc >= SOLVED_RAUC_FLOOR,
        "critical_service_day_cap_met": critical_service_days <= critical_cap,
        "terminal_pending_within_capacity": bool(
            np.all(
                terminal_pending
                <= DEPOT_CAPACITY * TERMINAL_PENDING_CAPACITY_MULTIPLIER
                + CONSTRAINT_TOLERANCE
            )
        ),
    }
    solved = all(checks.values())
    return {
        "definition_id": SOLVED_DEFINITION["id"],
        "definition_sha256": SOLVED_DEFINITION_SHA256,
        "solved": solved,
        "status": "solved" if solved else "failed",
        "checks": checks,
        "recovery_targets": round_vector(targets),
        "target_met_by_service": [
            bool(value) for value in target_met_by_service.tolist()
        ],
        "assessment_tail_days": assessment_tail_days,
        "tail_minimum_services": round_vector(np.min(tail, axis=0)),
        "resilience_auc": round(rauc, 8),
        "resilience_auc_floor": SOLVED_RAUC_FLOOR,
        "critical_service_days": critical_service_days,
        "critical_service_day_cap": critical_cap,
        "hard_violation_count": hard_violations,
        "max_conservation_residual": round(float(max_residual), 10),
        "terminal_pending_arrivals": round_vector(terminal_pending),
        "terminal_pending_capacity": round_vector(
            DEPOT_CAPACITY * TERMINAL_PENDING_CAPACITY_MULTIPLIER
        ),
        "reason_codes": [name for name, passed in checks.items() if not passed],
    }


def summarize_trajectory(
    planner: str,
    trajectory: list[dict[str, Any]],
    scenario: ScenarioModel,
) -> dict[str, Any]:
    """Return the canonical planner summary and bind it to its trajectory."""

    resilience = np.asarray(
        [day["resilience"] for day in trajectory], dtype=np.float64
    )
    normalized_priorities = np.asarray(scenario.priorities, dtype=np.float64)
    normalized_priorities /= normalized_priorities.sum()
    before_resilience = np.asarray(
        [
            normalized_priorities @ np.asarray(day["services_before"])
            for day in trajectory
        ]
    )
    shocked_resilience = np.asarray(
        [
            normalized_priorities @ np.asarray(day["services_after_shock"])
            for day in trajectory
        ]
    )
    largest_loss_index = int(np.argmax(before_resilience - shocked_resilience))
    recovery_target = float(before_resilience[largest_loss_index])
    recovery_day = len(trajectory) + 1
    for index in range(largest_loss_index, len(trajectory)):
        if resilience[index] >= recovery_target - CONSTRAINT_TOLERANCE:
            recovery_day = index - largest_loss_index
            break
    outcome = absolute_outcome(
        trajectory, scenario.recovery_targets, scenario.assessment_tail_days
    )
    max_residual = max(
        abs(value)
        for day in trajectory
        for value in day["logistics"]["conservation_residual"]
    )
    return {
        "planner": planner,
        "rauc": round(float(np.mean(resilience)), 8),
        "final_resilience": round(float(resilience[-1]), 8),
        "minimum_resilience": round(float(np.min(resilience)), 8),
        "post_shock_recovery_shortfall_auc": round(
            float(
                np.mean(
                    np.maximum(
                        0.0, recovery_target - resilience[largest_loss_index:]
                    )
                )
            ),
            8,
        ),
        "days_to_pre_shock_recovery_after_largest_loss": recovery_day,
        "largest_shock_loss_day": largest_loss_index + 1,
        "critical_service_days": outcome["critical_service_days"],
        "hard_violation_count": outcome["hard_violation_count"],
        "constraint_violations": outcome["hard_violation_count"],
        "max_logistics_conservation_residual": round(float(max_residual), 10),
        "final_depot_stock": trajectory[-1]["logistics"]["depot_stock_end"],
        "final_pending_arrivals": trajectory[-1]["logistics"][
            "pending_next_day"
        ],
        "absolute_outcome": outcome,
        "trajectory_sha256": canonical_hash(trajectory),
        "trajectory": trajectory,
    }


__all__ = (
    "CONSERVATION_TOLERANCE",
    "CRITICAL_SERVICE_FLOOR",
    "CRITICAL_SERVICE_RATE_CAP",
    "SOLVED_RAUC_FLOOR",
    "SOLVED_DEFINITION",
    "SOLVED_DEFINITION_SHA256",
    "TERMINAL_PENDING_CAPACITY_MULTIPLIER",
    "absolute_outcome",
    "summarize_trajectory",
)
