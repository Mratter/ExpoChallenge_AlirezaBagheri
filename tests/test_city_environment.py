"""Golden scientific-contract test for the canonical city environment."""

from __future__ import annotations

from dataclasses import asdict
from inspect import signature

import numpy as np

from backend.app.city.environment import (
    ACTION_SIZE,
    ENGINE_SPEC_SHA256,
    CityRecoveryEnv,
    CyclingScenarioEnv,
)
from backend.app.city.outcome import (
    SOLVED_DEFINITION_SHA256,
    summarize_trajectory,
)
from backend.app.city.scenarios import (
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import canonical_hash

EXPECTED_ENGINE_SPEC_SHA256 = (
    "34168cdf6a761dfd3be4ab7af8a4ef895561d4af1b3976c232c28502386b731e"
)
EXPECTED_SOLVED_DEFINITION_SHA256 = (
    "d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82"
)


def _actions(seed: int, count: int) -> list[np.ndarray]:
    random = np.random.Generator(np.random.PCG64(seed))
    return [random.uniform(-1.0, 1.0, size=ACTION_SIZE) for _ in range(count)]


def test_environment_preserves_golden_contract_and_trajectory() -> None:
    assert signature(CityRecoveryEnv).parameters["reward_profile"].default == (
        "v3_equivalent"
    )
    assert signature(CyclingScenarioEnv).parameters[
        "reward_profile"
    ].default == "v3_equivalent"
    assert ENGINE_SPEC_SHA256 == EXPECTED_ENGINE_SPEC_SHA256
    assert SOLVED_DEFINITION_SHA256 == EXPECTED_SOLVED_DEFINITION_SHA256

    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    schedule = generate_disaster_tape(scenario, seed)
    assert canonical_hash([asdict(shock) for shock in schedule]) == (
        "cdade263357aeebff3d9c9274e04b14306c941efca579b3c79b6c73ba79511ae"
    )
    environment = CityRecoveryEnv(
        scenario,
        seed,
        schedule,
        collect_evidence=True,
    )
    environment.reset(seed=seed)
    for action in _actions(0xA17_2026, scenario.horizon_days):
        _, _, _, truncated, _ = environment.step(action)
        assert not truncated

    summary = summarize_trajectory("golden", environment.trajectory, scenario)
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
