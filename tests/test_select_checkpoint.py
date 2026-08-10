from __future__ import annotations

import pytest

from scripts.select_checkpoint import (
    CANONICAL_DEVELOPMENT_CASE_COUNT,
    DEVELOPMENT_CASE_COUNT,
    LEGACY_DEVELOPMENT_CASE_COUNT,
    SelectionError,
    rank_candidates,
    receipt_development_case_count,
)


def _candidate(seed: int, transitions: int, solves: int, auc: float) -> dict:
    return {
        "policy_seed": seed,
        "active_actor_critic_transitions": transitions,
        "development": {
            "solved_count": solves,
            "solve_rate": solves / DEVELOPMENT_CASE_COUNT,
            "mean_resilience_auc": auc,
            "mean_minimum_tail_margin": 0.0,
        },
    }


def test_selection_uses_solves_never_auc_and_then_neutral_ties() -> None:
    candidates = [
        _candidate(57_017, 2_000_000, 175, 0.99),
        _candidate(47_017, 500_000, 175, 0.10),
        _candidate(37_017, 500_000, 175, 0.05),
        _candidate(77_017, 2_000_000, 170, 1.00),
    ]
    ranked = rank_candidates(candidates)
    assert [row["policy_seed"] for row in ranked] == [
        37_017,
        47_017,
        57_017,
        77_017,
    ]
    assert ranked[0]["development"]["mean_resilience_auc"] == 0.05


def test_selection_requires_multiple_complete_candidates() -> None:
    with pytest.raises(SelectionError):
        rank_candidates([])
    with pytest.raises(SelectionError):
        rank_candidates([_candidate(37_017, 500_000, 175, 0.5)])


def test_current_and_historical_development_counts_are_distinct() -> None:
    assert DEVELOPMENT_CASE_COUNT == CANONICAL_DEVELOPMENT_CASE_COUNT == 200
    assert receipt_development_case_count(
        {"development": {"case_count": LEGACY_DEVELOPMENT_CASE_COUNT}}
    ) == 40
