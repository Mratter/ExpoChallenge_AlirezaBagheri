from __future__ import annotations

import pytest

from scripts.select_checkpoint import SelectionError, rank_candidates


def _candidate(seed: int, transitions: int, solves: int, auc: float) -> dict:
    return {
        "policy_seed": seed,
        "active_actor_critic_transitions": transitions,
        "development": {
            "solved_count": solves,
            "solve_rate": solves / 40.0,
            "mean_resilience_auc": auc,
            "mean_minimum_tail_margin": 0.0,
        },
    }


def test_selection_uses_solves_never_auc_and_then_neutral_ties() -> None:
    candidates = [
        _candidate(57_017, 2_000_000, 35, 0.99),
        _candidate(47_017, 500_000, 35, 0.10),
        _candidate(37_017, 500_000, 35, 0.05),
        _candidate(77_017, 2_000_000, 34, 1.00),
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
        rank_candidates([_candidate(37_017, 500_000, 35, 0.5)])
