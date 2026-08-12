"""Behavior locks for causal public-state city planners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from backend.app.city.environment import CityRecoveryEnv
from backend.app.city.planners import (
    preparedness_teacher_action,
    reactive_heuristic_action,
    tuned_rule_action,
)
from backend.app.city.scenarios import TRAINING_FAMILIES, TRAINING_SEEDS
from backend.app.shared_evidence import canonical_hash

Planner = Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]


def _training_observation() -> np.ndarray:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    environment = CityRecoveryEnv(scenario, seed)
    observation, _ = environment.reset(seed=seed)
    assert canonical_hash(observation.tolist()) == (
        "8cb560551a9ccb608aecdaac60bb157551afded868a44b1248325cdca7cb6030"
    )
    return observation


@pytest.mark.parametrize(
    ("planner", "action_sha256", "evidence_sha256"),
    (
        (
            reactive_heuristic_action,
            "4ec28b0c78e9d1a97d2616d898d8677f3a794eeb0a2b9d7c5c5e17e48595c3d2",
            "7f45dcd9168143daaf3ba131bfd862bf7929d05838e96f6c3750ace3b06dd8c7",
        ),
        (
            preparedness_teacher_action,
            "90144ae8579a0df68f2ebb5a0c325effe550d010857ccf8d1f597c561912171a",
            "06c3163603e1ef36a5c507a96e2b919859eeeba1da505d96ed2724db4f1e10d5",
        ),
        (
            tuned_rule_action,
            "51b47d8a2de60e823d1eb81a9d5afaa76492a52ed4c8fcf77721e82329376088",
            "0ac8012466e295df746e14122d9e178dfc89a17f0ec16902e2aa125fd5afefb9",
        ),
    ),
)
def test_planner_matches_frozen_public_action_and_evidence(
    planner: Planner,
    action_sha256: str,
    evidence_sha256: str,
) -> None:
    action, evidence = planner(_training_observation())

    assert action.shape == (22,)
    assert action.dtype == np.float64
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # NumPy's fractional powers can differ by a few final bits across libm
    # implementations. Lock the planner's meaningful eight-decimal action
    # contract while the downstream trajectory regressions guard behavior.
    rounded_action = [round(float(value), 8) for value in action]
    assert canonical_hash(rounded_action) == action_sha256
    assert canonical_hash(evidence) == evidence_sha256


@pytest.mark.parametrize("observation", (np.zeros(72), np.full(73, np.nan)))
def test_planners_reject_invalid_public_observations(observation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="73 finite public inputs"):
        reactive_heuristic_action(observation)
