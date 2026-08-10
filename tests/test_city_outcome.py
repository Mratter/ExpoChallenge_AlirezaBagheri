from __future__ import annotations

from backend.app.city.outcome import (
    SOLVED_DEFINITION,
    SOLVED_DEFINITION_SHA256,
    absolute_outcome,
    summarize_trajectory,
)
from backend.app.simulator_v3 import (
    SOLVED_DEFINITION_V3,
    SOLVED_DEFINITION_V3_SHA256,
    _summarize_v3,
    absolute_outcome_v3,
)


def test_solved_definition_hash_remains_stable() -> None:
    assert SOLVED_DEFINITION_SHA256 == (
        "d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82"
    )


def test_legacy_outcome_exports_are_identity_aliases() -> None:
    assert SOLVED_DEFINITION_V3 is SOLVED_DEFINITION
    assert SOLVED_DEFINITION_V3_SHA256 == SOLVED_DEFINITION_SHA256
    assert absolute_outcome_v3 is absolute_outcome
    assert _summarize_v3 is summarize_trajectory
