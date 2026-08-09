"""Shared deterministic math used by the active CityRecoveryEnv-v2 runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

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


def _round_vector(values: np.ndarray) -> list[float]:
    return [float(round(value, 8)) for value in values.tolist()]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def project_capped_simplex(
    proposal: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project onto the common bounded budget simplex with deterministic rounding."""
    proposal = np.asarray(proposal, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if proposal.shape != (5,) or not np.all(np.isfinite(proposal)):
        raise ValueError("planner proposal must contain five finite allocations")
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
            "lower": bool(abs(rounded[index] - lower[index]) <= CONSTRAINT_TOLERANCE),
            "upper": bool(abs(rounded[index] - upper[index]) <= CONSTRAINT_TOLERANCE),
        }
        for index in range(5)
    ]
    return rounded, {
        "distance": round(float(np.linalg.norm(rounded - proposal)), 8),
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
    allocation: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> dict[str, int]:
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
