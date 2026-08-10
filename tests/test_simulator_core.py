from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.city.physics import (
    CONSTRAINT_TOLERANCE,
    SERVICES,
    project_capped_simplex,
)


def _project_capped_simplex_bisection_reference(
    proposal: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Snapshot of the pre-breakpoint implementation for equivalence tests."""
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
        "distance": round(float(np.linalg.norm(rounded - proposal)), 8),
        "bindings": bindings,
        "sum": round(float(rounded.sum()), 8),
    }


def _assert_matches_bisection_reference(
    proposal: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    expected, expected_evidence = _project_capped_simplex_bisection_reference(
        proposal, total, lower, upper
    )
    actual, actual_evidence = project_capped_simplex(proposal, total, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)
    assert actual_evidence == expected_evidence
    return actual


def test_breakpoint_projection_matches_bisection_over_random_feasible_inputs() -> None:
    rng = np.random.Generator(np.random.PCG64(20260810))
    for case_index in range(5_000):
        # The runtime contract caps resource totals at 500. The wider range
        # here also exercises very small totals and generous test headroom.
        scale = 10.0 ** rng.uniform(-3.0, 3.0)
        lower = rng.uniform(0.0, 2.0, size=5) * scale
        widths = rng.uniform(0.0, 5.0, size=5) * scale
        if case_index % 5 == 0:
            widths[case_index % 5] = 0.0
        upper = lower + widths
        total = float(lower.sum() + rng.beta(0.7, 0.7) * widths.sum())
        proposal = rng.normal(1.0, 6.0, size=5) * scale

        try:
            projected = _assert_matches_bisection_reference(
                proposal, total, lower, upper
            )
            assert np.all(projected >= lower - CONSTRAINT_TOLERANCE)
            assert np.all(projected <= upper + CONSTRAINT_TOLERANCE)
            assert abs(float(projected.sum()) - total) <= CONSTRAINT_TOLERANCE
        except AssertionError as error:
            raise AssertionError(f"random projection case {case_index} failed") from error


def test_breakpoint_projection_matches_bisection_on_edge_cases() -> None:
    cases = [
        # The two ends of the feasible interval.
        (
            np.full(5, 100.0),
            15.0,
            np.arange(1.0, 6.0),
            np.arange(6.0, 11.0),
        ),
        (
            np.full(5, -100.0),
            40.0,
            np.arange(1.0, 6.0),
            np.arange(6.0, 11.0),
        ),
        # Coincident entry/exit breakpoints and several fixed coordinates.
        (
            np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
            20.5,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([5.0, 6.0, 2.0, 3.0, 9.0]),
        ),
        # An already-feasible proposal is its own Euclidean projection.
        (
            np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
            12.5,
            np.zeros(5),
            np.full(5, 10.0),
        ),
        # Only one coordinate can absorb the remaining budget.
        (
            np.array([-50.0, 100.0, -100.0, 25.0, 8.0]),
            7.25,
            np.array([1.0, 2.0, 1.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 1.0, 4.0, 2.0]),
        ),
        # Zero budget and fully fixed bounds.
        (np.arange(5.0), 0.0, np.zeros(5), np.zeros(5)),
        # Highly skewed proposals against runtime-scale allocation bounds.
        (
            np.array([-1e9, 1e9, 1e-9, 3e8, -2e8]),
            250.0,
            np.zeros(5),
            np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        ),
    ]

    for proposal, total, lower, upper in cases:
        _assert_matches_bisection_reference(proposal, total, lower, upper)


def test_breakpoint_projection_preserves_deterministic_residual_repair() -> None:
    lower = np.zeros(5)
    upper = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    proposal = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    positive_residual = _assert_matches_bisection_reference(
        proposal, 1.0, lower, upper
    )
    negative_residual = _assert_matches_bisection_reference(
        proposal, 2.0, lower, upper
    )

    np.testing.assert_array_equal(
        positive_residual, np.array([0.33333334, 0.33333333, 0.33333333, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        negative_residual, np.array([0.66666666, 0.66666667, 0.66666667, 0.0, 0.0])
    )
    assert abs(float(positive_residual.sum()) - 1.0) <= 1e-12
    assert abs(float(negative_residual.sum()) - 2.0) <= 1e-12
