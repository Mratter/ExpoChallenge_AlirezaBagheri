from __future__ import annotations

import numpy as np

from backend.app.scenarios_v3 import TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3
from backend.app.simulator_core import canonical_hash
from backend.app.simulator_v3 import (
    ACTION_SIZE_V3,
    CityRecoveryEnvV3,
    generate_disaster_tape_v3,
)
from backend.app.simulator_v4 import CityRecoveryEnvV4


def _fixture() -> tuple[object, int, list[object]]:
    seed = TRAINING_SEEDS_V3[0]
    scenario = TRAINING_FAMILIES_V3[0].build(seed)
    return scenario, seed, generate_disaster_tape_v3(scenario, seed)


def test_v4_optimization_preserves_the_frozen_v3_transition() -> None:
    scenario, seed, schedule = _fixture()
    frozen = CityRecoveryEnvV3(scenario, seed, schedule)
    optimized = CityRecoveryEnvV4(scenario, seed, schedule, collect_evidence=True)
    frozen_observation, frozen_info = frozen.reset(seed=seed)
    optimized_observation, optimized_info = optimized.reset(seed=seed)
    np.testing.assert_array_equal(frozen_observation, optimized_observation)
    assert frozen_info == optimized_info

    generator = np.random.Generator(np.random.PCG64(0xA17_2026))
    terminated = False
    while not terminated:
        action = generator.uniform(-1.0, 1.0, size=ACTION_SIZE_V3)
        frozen_step = frozen.step(action)
        optimized_step = optimized.step(action)
        np.testing.assert_array_equal(frozen_step[0], optimized_step[0])
        assert frozen_step[1:4] == optimized_step[1:4]
        assert frozen_step[4] == optimized_step[4]
        terminated = frozen_step[2]

    assert canonical_hash(frozen.trajectory) == canonical_hash(optimized.trajectory)


def test_v4_evidence_collection_does_not_change_numeric_trajectory() -> None:
    scenario, seed, schedule = _fixture()
    evidence_env = CityRecoveryEnvV4(
        scenario, seed, schedule, collect_evidence=True
    )
    training_env = CityRecoveryEnvV4(
        scenario, seed, schedule, collect_evidence=False
    )
    evidence_observation, evidence_reset = evidence_env.reset(seed=seed)
    training_observation, training_reset = training_env.reset(seed=seed)
    np.testing.assert_array_equal(evidence_observation, training_observation)
    assert evidence_reset["shock_seed"] == seed
    assert training_reset == {}

    generator = np.random.Generator(np.random.PCG64(0xE71D_EACE))
    terminated = False
    while not terminated:
        action = generator.uniform(-1.0, 1.0, size=ACTION_SIZE_V3)
        evidence_step = evidence_env.step(action)
        training_step = training_env.step(action)
        np.testing.assert_array_equal(evidence_step[0], training_step[0])
        assert evidence_step[1:4] == training_step[1:4]
        for attribute in ("_q", "_stocks", "_pending", "_preparedness"):
            np.testing.assert_array_equal(
                getattr(evidence_env, attribute), getattr(training_env, attribute)
            )
        evidence_day = evidence_step[4]["day"]
        training_day = training_step[4]["day"]
        assert {
            "services_end": evidence_day["services_end"],
            "resilience": evidence_day["resilience"],
            "reward": evidence_day["reward"],
            "hard_violation_count": evidence_day["hard_violation_count"],
            "logistics": {
                "pending_next_day": evidence_day["logistics"]["pending_next_day"],
                "conservation_residual": evidence_day["logistics"][
                    "conservation_residual"
                ],
            },
        } == {
            key: training_day[key]
            for key in (
                "services_end",
                "resilience",
                "reward",
                "hard_violation_count",
                "logistics",
            )
        }
        assert len(training_day) <= (8 if training_step[2] else 6)
        assert len(evidence_day) > 30
        terminated = evidence_step[2]

    assert (
        evidence_env.trajectory[-1]["absolute_outcome"]
        == training_env.trajectory[-1]["absolute_outcome"]
    )
