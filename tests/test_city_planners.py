"""Behavior locks for causal public-state city planners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from backend.app.city.planners import (
    preparedness_teacher_action,
    reactive_heuristic_action,
    tuned_rule_action,
)
from backend.app.city.scenarios import TRAINING_FAMILIES, TRAINING_SEEDS
from backend.app.shared_evidence import canonical_hash
from backend.app.simulator_v3 import CityRecoveryEnvV3

Planner = Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]


def _training_observation() -> np.ndarray:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    environment = CityRecoveryEnvV3(scenario, seed)
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
            "0e72fd33654acfeb9067bdd9ced21b1f1eb0d956db53f1d8df332438bc13c063",
            "7f45dcd9168143daaf3ba131bfd862bf7929d05838e96f6c3750ace3b06dd8c7",
        ),
        (
            preparedness_teacher_action,
            "97522b127c69433254512dd9d6244aaeedf1ca32fc58cd7c86fabc741aa31910",
            "06c3163603e1ef36a5c507a96e2b919859eeeba1da505d96ed2724db4f1e10d5",
        ),
        (
            tuned_rule_action,
            "4d7b39e34e84748ab624a252f80ef2e66f2d8468ef739ba3860e70d0683910e6",
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
    assert canonical_hash(action.tolist()) == action_sha256
    assert canonical_hash(evidence) == evidence_sha256


@pytest.mark.parametrize("observation", (np.zeros(72), np.full(73, np.nan)))
def test_planners_reject_invalid_public_observations(observation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="73 finite public inputs"):
        reactive_heuristic_action(observation)
