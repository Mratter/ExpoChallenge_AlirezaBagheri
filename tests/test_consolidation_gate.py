"""Behavior anchors for safely flattening the simulator lineage."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from backend.app.scenarios_v3 import TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3
from backend.app.simulator_core import canonical_hash
from backend.app.simulator_v3 import (
    ACTION_SIZE_V3,
    ENGINE_V3_SPEC_SHA256,
    SOLVED_DEFINITION_V3_SHA256,
    _summarize_v3,
    generate_disaster_tape_v3,
)
from backend.app.simulator_v4 import CityRecoveryEnvV4


EXPECTED_SOLVED_DEFINITION_SHA256 = (
    "d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82"
)
EXPECTED_ENGINE_SPEC_SHA256 = (
    "34168cdf6a761dfd3be4ab7af8a4ef895561d4af1b3976c232c28502386b731e"
)


def test_scientific_contract_value_hashes_are_stable() -> None:
    assert SOLVED_DEFINITION_V3_SHA256 == EXPECTED_SOLVED_DEFINITION_SHA256
    assert ENGINE_V3_SPEC_SHA256 == EXPECTED_ENGINE_SPEC_SHA256


def test_full_evidence_training_trajectory_is_byte_stable() -> None:
    seed = TRAINING_SEEDS_V3[0]
    scenario = TRAINING_FAMILIES_V3[0].build(seed)
    schedule = generate_disaster_tape_v3(scenario, seed)
    assert canonical_hash([asdict(item) for item in schedule]) == (
        "cdade263357aeebff3d9c9274e04b14306c941efca579b3c79b6c73ba79511ae"
    )

    environment = CityRecoveryEnvV4(
        scenario,
        seed,
        schedule,
        collect_evidence=True,
        reward_profile="v3_equivalent",
    )
    environment.reset(seed=seed)
    random = np.random.Generator(np.random.PCG64(0xA17_2026))
    terminated = False
    while not terminated:
        action = random.uniform(-1.0, 1.0, size=ACTION_SIZE_V3)
        _, _, terminated, truncated, _ = environment.step(action)
        assert not truncated

    summary = _summarize_v3("golden", environment.trajectory, scenario)
    assert canonical_hash(environment.trajectory) == (
        "4be368fb957b480b1273989f17a1b80c2fa8520e86911628bfdd0e69692cf8d6"
    )
    assert canonical_hash(environment.trajectory[-1]["absolute_outcome"]) == (
        "57ebd50a36b3a4e88661ea17367a97c27765a09535cc307958b3252e7ab821e8"
    )
    assert summary["rauc"] == 0.36994078
    assert summary["absolute_outcome"]["solved"] is False
    assert summary["hard_violation_count"] == 0
    assert summary["max_logistics_conservation_residual"] == 0.0
    assert environment.trajectory[-1]["services_end"] == [
        0.35845802,
        0.33317664,
        0.42381176,
        0.38715337,
        0.34079291,
    ]
