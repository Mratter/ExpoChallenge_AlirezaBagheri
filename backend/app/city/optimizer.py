"""Build the causal OR-Tools allocation proposal from public city state."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from backend.app.city.physics import (
    DELTA,
    DEPENDENCIES,
    DEPOT_CAPACITY,
    ETA,
    IMMEDIATE_DELIVERY_FRACTION,
    RESERVE_DRAW_FRACTION,
    SERVICES,
    round_vector,
)

BASELINE_ID = "ortools-glop-visible-v2"
BASELINE_VERSION = "2.1.0"
UTILITY_SEGMENTS = 2
DIMINISHING_RETURN_BLEND = 0.15


class AllocationContext(Protocol):
    """Public state required by the one-day allocation program."""

    shocked: np.ndarray
    support: np.ndarray
    available_budget: float
    lower: np.ndarray
    upper: np.ndarray
    stock_ready: np.ndarray
    throughput: np.ndarray


def ortools_proposal(
    context: AllocationContext, priorities: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a causal one-day allocation with diminishing repair returns.

    The environment's public repair law is proportional to ``sqrt(dispatch)``.
    GLOP is linear, so the upgraded plan uses two decreasing-slope dispatch
    segments per service, then conservatively blends 15% of that plan into the
    established single-slope allocation. It remains myopic and never sees
    future shocks or policy actions.

    OR-Tools is imported only when this optional comparison planner is invoked,
    keeping the core simulation and policy-serving path free of solver startup
    cost and dependency side effects.
    """

    try:
        import ortools
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise RuntimeError("OR-Tools GLOP solver is unavailable") from exc

    supplied_priorities = np.asarray(priorities, dtype=np.float64)
    centrality = DEPENDENCIES.sum(axis=0)
    reserve = RESERVE_DRAW_FRACTION * DEPOT_CAPACITY * (1.0 - context.shocked)
    single_slope_coefficients = (
        supplied_priorities
        * (1.0 - context.shocked)
        * (ETA * context.support + 0.04 * centrality)
        * context.throughput
    )
    recovery_scale = (
        supplied_priorities
        * (1.0 - context.shocked)
        * ETA
        * context.support
        * context.throughput
        / np.sqrt(200.0)
    )
    strain_relief_slope = (
        supplied_priorities
        * DELTA
        * np.maximum(0.0, 0.35 - context.shocked)
        * context.throughput
        / context.available_budget
    )
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver is unavailable")
    solver.SetNumThreads(1)
    allocations = [
        solver.NumVar(
            float(context.lower[index]),
            float(context.upper[index]),
            f"x_{SERVICES[index]}",
        )
        for index in range(5)
    ]
    dispatches = [
        solver.NumVar(0.0, solver.infinity(), f"y_{SERVICES[index]}")
        for index in range(5)
    ]
    utility_segments: list[list[Any]] = []
    solver.Add(sum(allocations) == context.available_budget)
    for index in range(5):
        solver.Add(
            dispatches[index]
            <= float(context.stock_ready[index])
            + IMMEDIATE_DELIVERY_FRACTION * allocations[index]
        )
        solver.Add(dispatches[index] <= allocations[index] + float(reserve[index]))
        max_dispatch = min(
            float(context.stock_ready[index])
            + IMMEDIATE_DELIVERY_FRACTION * float(context.upper[index]),
            float(context.upper[index]) + float(reserve[index]),
        )
        width = max_dispatch / UTILITY_SEGMENTS
        segments = [
            solver.NumVar(0.0, width, f"u_{SERVICES[index]}_{segment}")
            for segment in range(UTILITY_SEGMENTS)
        ]
        solver.Add(sum(segments) == dispatches[index])
        utility_segments.append(segments)

    objective_terms = []
    marginal_slopes: list[list[float]] = []
    for index, segments in enumerate(utility_segments):
        max_dispatch = sum(float(segment.ub()) for segment in segments)
        breakpoints = np.linspace(0.0, max_dispatch, UTILITY_SEGMENTS + 1)
        service_slopes: list[float] = []
        for segment_index, segment in enumerate(segments):
            left = float(breakpoints[segment_index])
            right = float(breakpoints[segment_index + 1])
            sqrt_slope = (np.sqrt(right) - np.sqrt(left)) / (right - left)
            slope = float(recovery_scale[index] * sqrt_slope + strain_relief_slope[index])
            service_slopes.append(slope)
            objective_terms.append(slope * segment)
        marginal_slopes.append(service_slopes)
    solver.Maximize(sum(objective_terms))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools baseline failed with status {status}")
    diminishing_return_proposal = np.array(
        [variable.solution_value() for variable in allocations]
    )

    # Keep most of the established baseline and apply only a conservative public-
    # state correction toward the diminishing-return solution. A convex mixture
    # of two feasible full-budget allocations is itself feasible.
    single_solver = pywraplp.Solver.CreateSolver("GLOP")
    if single_solver is None:
        raise RuntimeError("OR-Tools GLOP solver is unavailable")
    single_solver.SetNumThreads(1)
    single_allocations = [
        single_solver.NumVar(
            float(context.lower[index]),
            float(context.upper[index]),
            f"single_x_{SERVICES[index]}",
        )
        for index in range(5)
    ]
    single_dispatches = [
        single_solver.NumVar(
            0.0,
            single_solver.infinity(),
            f"single_y_{SERVICES[index]}",
        )
        for index in range(5)
    ]
    single_solver.Add(sum(single_allocations) == context.available_budget)
    for index in range(5):
        single_solver.Add(
            single_dispatches[index]
            <= float(context.stock_ready[index])
            + IMMEDIATE_DELIVERY_FRACTION * single_allocations[index]
        )
        single_solver.Add(
            single_dispatches[index]
            <= single_allocations[index] + float(reserve[index])
        )
    single_solver.Maximize(
        sum(
            float(single_slope_coefficients[index]) * single_dispatches[index]
            for index in range(5)
        )
    )
    single_status = single_solver.Solve()
    if single_status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(
            f"OR-Tools single-slope baseline failed with status {single_status}"
        )
    single_slope_proposal = np.array(
        [variable.solution_value() for variable in single_allocations]
    )
    proposal = (
        (1.0 - DIMINISHING_RETURN_BLEND) * single_slope_proposal
        + DIMINISHING_RETURN_BLEND * diminishing_return_proposal
    )
    dispatch_solution = np.minimum(
        context.stock_ready + IMMEDIATE_DELIVERY_FRACTION * proposal,
        proposal + reserve,
    )
    return proposal, {
        "baseline_id": BASELINE_ID,
        "baseline_version": BASELINE_VERSION,
        "library": "OR-Tools",
        "library_version": ortools.__version__,
        "solver": "GLOP",
        "status": "OPTIMAL",
        "objective": (
            "dispatch_y objective with a 15% public-state correction toward a "
            "two-segment diminishing-return repair approximation"
        ),
        "decision_variables": {
            "x": "five allocation units; sum(x) equals the full available daily budget",
            "y": (
                "five raw repair dispatch units; y <= stock_ready + 0.65*x "
                "and y <= x + 0.04*capacity*deficit"
            ),
        },
        "utility_segments_per_service": UTILITY_SEGMENTS,
        "diminishing_return_blend": DIMINISHING_RETURN_BLEND,
        "single_slope_objective_coefficients": round_vector(
            single_slope_coefficients
        ),
        "marginal_utility_slopes": [
            round_vector(np.asarray(slopes, dtype=np.float64))
            for slopes in marginal_slopes
        ],
        "single_slope_allocation_solution": round_vector(single_slope_proposal),
        "diminishing_return_allocation_solution": round_vector(
            diminishing_return_proposal
        ),
        "allocation_solution": round_vector(proposal),
        "dispatch_solution": round_vector(dispatch_solution),
        "future_shocks_visible": False,
    }


__all__ = (
    "AllocationContext",
    "BASELINE_ID",
    "BASELINE_VERSION",
    "DIMINISHING_RETURN_BLEND",
    "UTILITY_SEGMENTS",
    "ortools_proposal",
)
