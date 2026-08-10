"""Differential and golden tests for the flattened city environment."""

from __future__ import annotations

from dataclasses import asdict
from inspect import signature

import numpy as np
import pytest

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
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    ScenarioFamily,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import canonical_hash
from backend.app.simulator_v4 import CityRecoveryEnvV4, CyclingScenarioEnvV4

EXPECTED_ENGINE_SPEC_SHA256 = (
    "34168cdf6a761dfd3be4ab7af8a4ef895561d4af1b3976c232c28502386b731e"
)
EXPECTED_SOLVED_DEFINITION_SHA256 = (
    "d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82"
)


def _actions(seed: int, count: int) -> list[np.ndarray]:
    random = np.random.Generator(np.random.PCG64(seed))
    return [random.uniform(-1.0, 1.0, size=ACTION_SIZE) for _ in range(count)]


def _representative_actions(seed: int, count: int) -> list[np.ndarray]:
    actions = _actions(seed, count)
    edge_cases = (
        np.zeros(ACTION_SIZE, dtype=np.float64),
        np.ones(ACTION_SIZE, dtype=np.float64),
        -np.ones(ACTION_SIZE, dtype=np.float64),
        np.resize(np.asarray([-1.0, 1.0]), ACTION_SIZE),
    )
    actions[: min(len(edge_cases), count)] = edge_cases[:count]
    return actions


def _assert_step_equal(
    current: tuple[np.ndarray, float, bool, bool, dict[str, object]],
    legacy: tuple[np.ndarray, float, bool, bool, dict[str, object]],
) -> None:
    np.testing.assert_array_equal(current[0], legacy[0])
    assert current[1:] == legacy[1:]


def _assert_rollout_equal(
    family: ScenarioFamily,
    case_seed: int,
    *,
    collect_evidence: bool,
    action_seed: int,
) -> None:
    scenario = family.build(case_seed)
    tape_seed = family.tape_seed(case_seed)
    schedule = generate_disaster_tape(scenario, tape_seed)
    current = CityRecoveryEnv(
        scenario,
        tape_seed,
        schedule,
        collect_evidence=collect_evidence,
    )
    legacy = CityRecoveryEnvV4(
        scenario,
        tape_seed,
        schedule,
        collect_evidence=collect_evidence,
        reward_profile="v3_equivalent",
    )
    current_observation, current_info = current.reset(seed=tape_seed)
    legacy_observation, legacy_info = legacy.reset(seed=tape_seed)
    np.testing.assert_array_equal(current_observation, legacy_observation)
    assert current_info == legacy_info

    for index, action in enumerate(
        _representative_actions(action_seed, scenario.horizon_days)
    ):
        if index == 0:
            evidence = {"source": "differential-test"}
            _assert_step_equal(
                current.step_with_evidence(action, evidence),
                legacy.step_with_evidence(action, evidence),
            )
        else:
            _assert_step_equal(current.step(action), legacy.step(action))

    assert current.trajectory == legacy.trajectory
    assert canonical_hash(current.trajectory) == canonical_hash(legacy.trajectory)


@pytest.mark.parametrize("collect_evidence", [True, False])
@pytest.mark.parametrize(
    ("family", "case_seed", "action_seed"),
    [
        (TRAINING_FAMILIES[0], TRAINING_SEEDS[0], 101),
        (DEVELOPMENT_FAMILIES[-1], DEVELOPMENT_SEEDS[-1], 202),
    ],
)
def test_flattened_environment_matches_legacy_v4(
    family: ScenarioFamily,
    case_seed: int,
    action_seed: int,
    collect_evidence: bool,
) -> None:
    _assert_rollout_equal(
        family,
        case_seed,
        collect_evidence=collect_evidence,
        action_seed=action_seed,
    )


def test_set_scenario_matches_legacy_v4() -> None:
    initial = TRAINING_FAMILIES[0].build(TRAINING_SEEDS[0])
    replacement_family = DEVELOPMENT_FAMILIES[0]
    replacement_seed = DEVELOPMENT_SEEDS[0]
    replacement = replacement_family.build(replacement_seed)
    tape_seed = replacement_family.tape_seed(replacement_seed)
    current = CityRecoveryEnv(initial, collect_evidence=False)
    legacy = CityRecoveryEnvV4(
        initial,
        collect_evidence=False,
        reward_profile="v3_equivalent",
    )

    current.set_scenario(replacement, tape_seed)
    legacy.set_scenario(replacement, tape_seed)
    current_observation, current_info = current.reset(seed=tape_seed)
    legacy_observation, legacy_info = legacy.reset(seed=tape_seed)
    np.testing.assert_array_equal(current_observation, legacy_observation)
    assert current_info == legacy_info == {}
    for action in _actions(303, replacement.horizon_days):
        _assert_step_equal(current.step(action), legacy.step(action))
    assert current.trajectory == legacy.trajectory


def test_cycling_environment_matches_legacy_v4() -> None:
    cases = [
        (
            TRAINING_FAMILIES[index].build(TRAINING_SEEDS[index]),
            TRAINING_FAMILIES[index].tape_seed(TRAINING_SEEDS[index]),
        )
        for index in range(2)
    ]
    current = CyclingScenarioEnv(cases, collect_evidence=False)
    legacy = CyclingScenarioEnvV4(
        cases,
        collect_evidence=False,
        reward_profile="v3_equivalent",
    )
    actions = _actions(404, 3)

    for _ in cases:
        current_observation, current_info = current.reset()
        legacy_observation, legacy_info = legacy.reset()
        np.testing.assert_array_equal(current_observation, legacy_observation)
        assert current_info == legacy_info == {}
        for action in actions:
            _assert_step_equal(current.step(action), legacy.step(action))
        assert current.render() == legacy.render()


def test_environment_preserves_golden_contract_and_trajectory() -> None:
    assert "reward_profile" not in signature(CityRecoveryEnv).parameters
    assert "reward_profile" not in signature(CyclingScenarioEnv).parameters
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
