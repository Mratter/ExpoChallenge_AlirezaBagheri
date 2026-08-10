"""Optimized allocation math for the parallel v4 simulator.

The v3 core is provenance-pinned by the shipped model and intentionally remains
byte-identical. V4 reuses its immutable constants and helpers while replacing
only capped-simplex multiplier search with an exact breakpoint solve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.simulator_core import (
    CONSTRAINT_TOLERANCE,
    SERVICES,
    _round_vector,
    action_to_proposal,
    measure_constraints,
)

__all__ = (
    "_round_vector",
    "action_to_proposal",
    "measure_constraints",
    "project_capped_simplex",
)


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
    # Preserve v3's deterministic rounding and residual-repair contract.
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
            "lower": bool(abs(rounded[index] - lower[index]) <= CONSTRAINT_TOLERANCE),
            "upper": bool(abs(rounded[index] - upper[index]) <= CONSTRAINT_TOLERANCE),
        }
        for index in range(5)
    ]
    return rounded, {
        "distance": distance,
        "bindings": bindings,
        "sum": round(float(rounded.sum()), 8),
    }
