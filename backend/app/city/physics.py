"""Deterministic allocation and logistics physics for city recovery."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol

import numpy as np

SERVICES = ("transport", "housing", "food", "healthcare", "public_services")
SHOCKS = ("aftershock", "supply", "epidemic", "utility", "weather")
BASE_OBSERVATION_ORDER = (
    *(f"service_{name}" for name in SERVICES),
    *(f"priority_{name}" for name in SERVICES),
    *(f"support_{name}" for name in SERVICES),
    *(f"shock_impact_{name}" for name in SERVICES),
    "available_budget_fraction",
    "horizon_remaining_fraction",
    "shock_severity",
)
SHOCK_TYPE_PROBABILITIES = np.array([0.24, 0.22, 0.18, 0.20, 0.16])
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
SHOCK_BUDGET_FACTORS = np.array([0.15, 0.25, 0.10, 0.30, 0.25])
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
ETA = np.array([0.18, 0.16, 0.20, 0.22, 0.17])
DELTA = np.array([0.010, 0.012, 0.015, 0.018, 0.010])
CONSTRAINT_TOLERANCE = 1e-7

DEPOT_CAPACITY = np.full(5, 400.0, dtype=np.float64)
IMMEDIATE_DELIVERY_FRACTION = 0.65
FOOD_SPOILAGE_RATE = 0.006
RESERVE_DRAW_FRACTION = 0.04
ROAD_CAPACITY_FLOOR = 0.40
DEPOT_THROUGHPUT_FLOOR = 0.30
DEPOT_DAMAGE_SCALE = 1.50
DEPOT_DAMAGE_PENALTY_CAP = 0.72
TRANSFER_STARVED_FRACTION = 0.15
TRANSFER_SURPLUS_FRACTION = 0.42
TRANSFER_DONOR_RESERVE_FRACTION = 0.35
TRANSFER_RECEIVER_TARGET_FRACTION = 0.30
TRANSFER_DAILY_CAP_FRACTION = 0.06
TRANSFER_MIN_THROUGHPUT = 0.55


class ShockLike(Protocol):
    """Structural input required by the depot-damage calculation."""

    type: str | None
    severity: float
    impact: list[float]


@dataclass(frozen=True)
class Transfer:
    """One deterministic mutual-aid movement between service depots."""

    from_service: str
    to_service: str
    units: float
    donor_stock_fraction_before: float
    receiver_stock_fraction_before: float


def round_vector(values: np.ndarray) -> list[float]:
    """Round a numeric vector using the simulator's stable evidence contract."""

    return [float(round(value, 8)) for value in values.tolist()]


def project_capped_simplex(
    proposal: np.ndarray,
    total: float,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    collect_evidence: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project exactly onto a five-value bounded budget simplex.

    KKT gives ``x_i = clip(proposal_i - multiplier, lower_i, upper_i)``.
    Sorting the ten upper/lower breakpoints identifies the affine interval that
    contains the multiplier, which is then solved in closed form.
    """

    proposal = np.asarray(proposal, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if proposal.shape != (5,) or not np.all(np.isfinite(proposal)):
        raise ValueError("planner proposal must contain five finite allocations")
    if float(lower.sum()) > total + 1e-9 or float(upper.sum()) < total - 1e-9:
        raise ValueError("allocation constraints are infeasible")

    upper_breakpoints = proposal - upper
    lower_breakpoints = proposal - lower
    breakpoints = np.concatenate((upper_breakpoints, lower_breakpoints))
    event_order = np.argsort(breakpoints, kind="stable")
    upper_sum = float(upper.sum())
    lower_sum = float(lower.sum())

    if total >= upper_sum:
        multiplier = float(np.min(upper_breakpoints))
    elif total <= lower_sum:
        multiplier = float(np.max(lower_breakpoints))
    else:
        affine_intercept = upper_sum
        free_count = 0
        cursor = 0
        multiplier = float(breakpoints[event_order[-1]])
        while cursor < event_order.size:
            breakpoint = float(breakpoints[event_order[cursor]])
            group_end = cursor
            while (
                group_end < event_order.size
                and float(breakpoints[event_order[group_end]]) == breakpoint
            ):
                event_index = int(event_order[group_end])
                if event_index < proposal.size:
                    affine_intercept += breakpoint
                    free_count += 1
                else:
                    affine_intercept -= breakpoint
                    free_count -= 1
                group_end += 1

            if group_end == event_order.size:
                break
            next_breakpoint = float(breakpoints[event_order[group_end]])
            if free_count:
                candidate = (affine_intercept - total) / free_count
                if candidate <= next_breakpoint:
                    multiplier = max(breakpoint, candidate)
                    break
            cursor = group_end

    projected = np.clip(proposal - multiplier, lower, upper)
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
    distance = round(float(np.linalg.norm(rounded - proposal)), 8)
    if not collect_evidence:
        return rounded, {"distance": distance}
    bindings = [
        {
            "service": SERVICES[index],
            "lower": bool(
                abs(rounded[index] - lower[index]) <= CONSTRAINT_TOLERANCE
            ),
            "upper": bool(
                abs(rounded[index] - upper[index]) <= CONSTRAINT_TOLERANCE
            ),
        }
        for index in range(5)
    ]
    return rounded, {
        "distance": distance,
        "bindings": bindings,
        "sum": round(float(rounded.sum()), 8),
    }


def action_to_proposal(action: np.ndarray, budget: float) -> np.ndarray:
    """Convert five bounded policy logits into a positive budget proposal."""

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (5,) or not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain five finite values")
    clipped = np.clip(action, -1.0, 1.0)
    exponentials = np.exp(clipped - float(np.max(clipped)))
    return budget * exponentials / float(exponentials.sum())


def measure_constraints(
    allocation: np.ndarray,
    total: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, int]:
    """Count hard allocation violations against total and bound constraints."""

    allocation_sum = float(allocation.sum())
    measurements = {
        "sum_violations": int(abs(allocation_sum - total) > CONSTRAINT_TOLERANCE),
        "budget_violations": int(allocation_sum > total + CONSTRAINT_TOLERANCE),
        "lower_violations": int(
            np.count_nonzero(allocation < lower - CONSTRAINT_TOLERANCE)
        ),
        "upper_violations": int(
            np.count_nonzero(allocation > upper + CONSTRAINT_TOLERANCE)
        ),
    }
    measurements["total"] = sum(measurements.values())
    return measurements


def apply_depot_damage(
    shock: ShockLike,
    peak_penalty: np.ndarray,
    duration_days: np.ndarray,
    remaining_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply typed depot damage while preserving the stronger/longer condition."""

    peaks = np.asarray(peak_penalty, dtype=np.float64).copy()
    durations = np.asarray(duration_days, dtype=np.int64).copy()
    remaining = np.asarray(remaining_days, dtype=np.int64).copy()
    current = np.divide(
        peaks * remaining,
        durations,
        out=np.zeros(5, dtype=np.float64),
        where=durations > 0,
    )
    if shock.type is not None and shock.severity > 0.0:
        impact = np.asarray(shock.impact, dtype=np.float64)
        candidate_penalty = np.clip(
            DEPOT_DAMAGE_SCALE * shock.severity * impact,
            0.0,
            DEPOT_DAMAGE_PENALTY_CAP,
        )
        candidate_duration = np.array(
            [ceil(2.0 + 8.0 * shock.severity * value) for value in impact],
            dtype=np.int64,
        )
        for index in range(5):
            if candidate_penalty[index] <= 0.0:
                continue
            combined_penalty = max(
                float(current[index]), float(candidate_penalty[index])
            )
            combined_remaining = max(
                int(remaining[index]), int(candidate_duration[index])
            )
            peaks[index] = combined_penalty
            durations[index] = combined_remaining
            remaining[index] = combined_remaining
            current[index] = combined_penalty
    return peaks, durations, remaining, current


def throughput_factors(
    shocked_services: np.ndarray,
    damage_penalty: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Calculate depot, road, and effective service throughput factors."""

    depot_factor = np.clip(
        1.0 - np.asarray(damage_penalty, dtype=np.float64),
        DEPOT_THROUGHPUT_FLOOR,
        1.0,
    )
    shocked = np.asarray(shocked_services, dtype=np.float64)
    road_capacity = float(
        np.clip(
            ROAD_CAPACITY_FLOOR
            + (1.0 - ROAD_CAPACITY_FLOOR) * shocked[0],
            0.0,
            1.0,
        )
    )
    throughput = depot_factor.copy()
    throughput[1:] *= road_capacity
    return depot_factor, road_capacity, np.clip(throughput, 0.0, 1.0)


def deterministic_transfer(
    stock: np.ndarray,
    throughput: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[Transfer, ...]]:
    """Move at most one bounded generic-supply load between eligible depots."""

    adjusted = np.asarray(stock, dtype=np.float64).copy()
    factors = np.asarray(throughput, dtype=np.float64)
    fractions = adjusted / DEPOT_CAPACITY
    receivers = [
        index
        for index in range(5)
        if fractions[index] < TRANSFER_STARVED_FRACTION
        and factors[index] >= TRANSFER_MIN_THROUGHPUT
    ]
    if not receivers:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    receiver = min(receivers, key=lambda index: (fractions[index], index))
    donors = [
        index
        for index in range(5)
        if index != receiver
        and fractions[index] > TRANSFER_SURPLUS_FRACTION
        and factors[index] >= TRANSFER_MIN_THROUGHPUT
    ]
    if not donors:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    donor = max(donors, key=lambda index: (fractions[index], -index))
    amount = min(
        TRANSFER_DAILY_CAP_FRACTION * DEPOT_CAPACITY[receiver],
        adjusted[donor]
        - TRANSFER_DONOR_RESERVE_FRACTION * DEPOT_CAPACITY[donor],
        TRANSFER_RECEIVER_TARGET_FRACTION * DEPOT_CAPACITY[receiver]
        - adjusted[receiver],
    )
    amount = float(max(0.0, amount))
    if amount <= CONSTRAINT_TOLERANCE:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    net = np.zeros(5, dtype=np.float64)
    adjusted[donor] -= amount
    adjusted[receiver] += amount
    net[donor] = -amount
    net[receiver] = amount
    event = Transfer(
        from_service=SERVICES[donor],
        to_service=SERVICES[receiver],
        units=float(round(amount, 8)),
        donor_stock_fraction_before=float(round(fractions[donor], 8)),
        receiver_stock_fraction_before=float(round(fractions[receiver], 8)),
    )
    return adjusted, net, (event,)


def land_capped(
    stock: np.ndarray,
    arrivals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Land arrivals up to depot capacity and keep overflow pending."""

    supplied_stock = np.asarray(stock, dtype=np.float64)
    supplied_arrivals = np.asarray(arrivals, dtype=np.float64)
    room = np.maximum(0.0, DEPOT_CAPACITY - supplied_stock)
    landed = np.minimum(supplied_arrivals, room)
    held = supplied_arrivals - landed
    return supplied_stock + landed, landed, held
