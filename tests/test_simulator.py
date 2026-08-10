"""Behavior tests for the canonical city-recovery simulator."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.city.environment import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    CityRecoveryEnv,
    CyclingScenarioEnv,
)
from backend.app.city.outcome import absolute_outcome
from backend.app.city.scenarios import (
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    Shock,
    generate_disaster_tape,
)
from backend.app.models import Scenario
from backend.app.shared_evidence import canonical_hash


def _fixture() -> tuple[Scenario, int, list[Shock]]:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    return scenario, seed, generate_disaster_tape(scenario, seed)


def test_observation_and_action_contract_rejects_invalid_actions() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    observation, _ = environment.reset(seed=seed)

    assert OBSERVATION_SIZE == 73
    assert ACTION_SIZE == 22
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all((0.0 <= observation) & (observation <= 1.0))
    assert environment.action_space.shape == (ACTION_SIZE,)

    with pytest.raises(ValueError):
        environment.step(np.zeros(ACTION_SIZE - 1, dtype=np.float32))
    bad = np.zeros(ACTION_SIZE, dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        environment.step(bad)


def test_same_actions_and_tape_replay_exactly() -> None:
    scenario, seed, schedule = _fixture()
    actions = [np.linspace(-1.0, 1.0, ACTION_SIZE) for _ in range(30)]
    hashes: list[str] = []

    for _ in range(2):
        environment = CityRecoveryEnv(scenario, seed, schedule)
        environment.reset(seed=seed)
        terminated = False
        action_index = 0
        while not terminated:
            _, _, terminated, _, _ = environment.step(actions[action_index])
            action_index += 1
        hashes.append(canonical_hash(environment.trajectory))

    assert hashes[0] == hashes[1]


def test_decoder_conserves_material_crew_and_inventory() -> None:
    scenario, seed, schedule = _fixture()
    random = np.random.Generator(np.random.PCG64(918273))
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    terminated = False

    while not terminated:
        action = random.uniform(-1.0, 1.0, size=ACTION_SIZE)
        _, _, terminated, _, info = environment.step(action)
        day = info["day"]
        assert sum(day["material_allocation"]) == pytest.approx(
            day["material_used"], abs=1e-6
        )
        assert day["material_used"] + day["material_unspent"] == pytest.approx(
            day["available_budget"], abs=1e-6
        )
        assert sum(day["crew_allocation"]) == pytest.approx(
            day["crew_used"], abs=1e-6
        )
        assert day["crew_used"] + day["crew_idle"] == pytest.approx(
            day["available_crew"], abs=1e-6
        )
        assert day["hard_violation_count"] == 0
        assert max(
            abs(value) for value in day["logistics"]["conservation_residual"]
        ) <= 1e-6


def test_utilization_gates_can_leave_resources_idle() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    action = np.zeros(ACTION_SIZE, dtype=np.float64)
    action[5] = -1.0
    action[11] = -1.0

    _, _, _, _, info = environment.step(action)
    day = info["day"]
    assert day["material_unspent"] > 0.0
    assert day["crew_idle"] > 0.0
    assert day["hard_violation_count"] == 0


def test_overordering_without_release_cannot_count_as_solved() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    action = np.zeros(ACTION_SIZE, dtype=np.float64)
    action[5] = 1.0
    action[11] = 1.0
    action[12:17] = -1.0
    action[17:22] = -1.0
    terminated = False

    while not terminated:
        _, _, terminated, _, _ = environment.step(action)

    outcome = environment.trajectory[-1]["absolute_outcome"]
    assert not outcome["solved"]
    assert not outcome["checks"]["terminal_pending_within_capacity"]


def test_preparedness_consumes_real_resources_and_builds_state() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    action = np.zeros(ACTION_SIZE, dtype=np.float64)
    action[17:22] = 1.0

    _, _, _, _, info = environment.step(action)
    day = info["day"]
    assert all(value > 0.0 for value in day["preparedness_gain"])
    assert all(
        value > 0.0
        for value in day["logistics"]["preparedness_material_consumed"]
    )
    assert all(
        committed + 1e-7 >= lower
        for committed, lower in zip(
            day["logistics"]["repair_material_committed"],
            day["lower_bounds"],
            strict=True,
        )
    )
    assert all(
        assigned + 1e-7 >= lower
        for assigned, lower in zip(
            day["logistics"]["repair_crew_assigned"],
            day["crew_lower_bounds"],
            strict=True,
        )
    )
    assert all(
        realized <= requested + 1e-8
        for realized, requested in zip(
            day["preparedness_gain"],
            day["preparedness_gain_requested"],
            strict=True,
        )
    )
    assert day["hard_violation_count"] == 0


def test_zero_stock_release_blocks_all_inventory_dispatch() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    action = np.zeros(ACTION_SIZE, dtype=np.float64)
    action[12:17] = -1.0
    action[17:22] = 1.0

    _, _, _, _, info = environment.step(action)
    day = info["day"]
    assert all(
        abs(value) <= 1e-9
        for value in day["logistics"]["preparedness_material_consumed"]
    )
    assert all(
        abs(value) <= 1e-9 for value in day["logistics"]["repair_dispatch"]
    )
    assert all(abs(value) <= 1e-9 for value in day["preparedness_gain"])
    assert day["hard_violation_count"] == 0


def test_absolute_outcome_recomputes_from_trajectory() -> None:
    scenario, seed, schedule = _fixture()
    environment = CityRecoveryEnv(scenario, seed, schedule)
    environment.reset(seed=seed)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = environment.step(np.zeros(ACTION_SIZE))

    outcome = absolute_outcome(
        environment.trajectory,
        scenario.recovery_targets,
        scenario.assessment_tail_days,
    )
    assert outcome == environment.trajectory[-1]["absolute_outcome"]


def test_evidence_collection_does_not_change_numeric_trajectory() -> None:
    scenario, seed, schedule = _fixture()
    evidence_environment = CityRecoveryEnv(
        scenario, seed, schedule, collect_evidence=True
    )
    training_environment = CityRecoveryEnv(
        scenario, seed, schedule, collect_evidence=False
    )
    evidence_observation, evidence_reset = evidence_environment.reset(seed=seed)
    training_observation, training_reset = training_environment.reset(seed=seed)
    np.testing.assert_array_equal(evidence_observation, training_observation)
    assert evidence_reset["shock_seed"] == seed
    assert training_reset == {}

    random = np.random.Generator(np.random.PCG64(0xE71D_EACE))
    terminated = False
    while not terminated:
        action = random.uniform(-1.0, 1.0, size=ACTION_SIZE)
        evidence_step = evidence_environment.step(action)
        training_step = training_environment.step(action)
        np.testing.assert_array_equal(evidence_step[0], training_step[0])
        assert evidence_step[1:4] == training_step[1:4]
        for attribute in ("_q", "_stocks", "_pending", "_preparedness"):
            np.testing.assert_array_equal(
                getattr(evidence_environment, attribute),
                getattr(training_environment, attribute),
            )

        evidence_day = evidence_step[4]["day"]
        training_day = training_step[4]["day"]
        for field in (
            "services_end",
            "resilience",
            "reward",
            "hard_violation_count",
        ):
            assert evidence_day[field] == training_day[field]
        assert evidence_day["logistics"]["pending_next_day"] == training_day[
            "logistics"
        ]["pending_next_day"]
        assert evidence_day["logistics"]["conservation_residual"] == training_day[
            "logistics"
        ]["conservation_residual"]
        assert len(training_day) <= (10 if training_step[2] else 6)
        assert len(evidence_day) > 30
        terminated = evidence_step[2]

    assert (
        evidence_environment.trajectory[-1]["absolute_outcome"]
        == training_environment.trajectory[-1]["absolute_outcome"]
    )


def test_set_scenario_and_cycling_environment_select_registered_cases() -> None:
    initial, _, _ = _fixture()
    replacement_family = DEVELOPMENT_FAMILIES[0]
    replacement_seed = DEVELOPMENT_SEEDS[0]
    replacement = replacement_family.build(replacement_seed)
    tape_seed = replacement_family.tape_seed(replacement_seed)

    environment = CityRecoveryEnv(initial, collect_evidence=False)
    environment.set_scenario(replacement, tape_seed)
    observation, info = environment.reset(seed=tape_seed)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert info == {}
    assert environment.scenario == replacement
    assert environment.shock_seed == tape_seed

    cases = [
        (
            TRAINING_FAMILIES[index].build(TRAINING_SEEDS[index]),
            TRAINING_FAMILIES[index].tape_seed(TRAINING_SEEDS[index]),
        )
        for index in range(2)
    ]
    cycling = CyclingScenarioEnv(cases, collect_evidence=False)
    first, first_info = cycling.reset()
    second, second_info = cycling.reset()
    assert first.shape == second.shape == (OBSERVATION_SIZE,)
    assert first_info == second_info == {}
    assert not np.array_equal(first, second)
