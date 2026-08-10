from __future__ import annotations

from backend.app.city.outcome import (
    SOLVED_DEFINITION_SHA256,
)


def test_solved_definition_hash_remains_stable() -> None:
    assert SOLVED_DEFINITION_SHA256 == (
        "d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82"
    )
