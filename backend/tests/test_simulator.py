import numpy as np

from backend.app.models import Scenario
from backend.app.scenarios import (
    HELD_OUT_FAMILIES,
    HELD_OUT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
)
from backend.app.simulator import (
    OBSERVATION_SIZE,
    CityRecoveryEnv,
    action_to_proposal,
    generate_shock_schedule,
    project_capped_simplex,
    rollout_baseline,
)


def test_projector_obeys_caps_and_total() -> None:
    proposal = np.array([100.0, 20.0, 15.0, 5.0, 40.0])
    lower = np.array([0.0, 7.2, 0.0, 0.0, 7.2])
    upper = np.full(5, 90.0)
    projected, evidence = project_capped_simplex(proposal, 180.0, lower, upper)

    assert projected.sum() == 180.0
    assert np.all(projected >= lower)
    assert np.all(projected <= upper)
    assert evidence["sum"] == 180.0


def test_action_mapping_is_positive_and_budget_normalized() -> None:
    proposal = action_to_proposal(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), 147.0)
    assert np.all(proposal > 0)
    assert np.isclose(proposal.sum(), 147.0)
    assert proposal.tolist() == sorted(proposal.tolist())


def test_shock_tape_is_repeatable_and_forced_override_does_not_shift_tail() -> None:
    scenario = Scenario()
    first = generate_shock_schedule(scenario, 424242)
    second = generate_shock_schedule(scenario, 424242)

    assert first == second
    assert first[4].forced is True
    assert first[4].type == "utility"
    assert first[4].severity == 0.26


def test_gym_environment_replays_complete_inspectable_trajectory() -> None:
    scenario = Scenario(horizon_days=7)
    trajectories = []
    for _ in range(2):
        env = CityRecoveryEnv(scenario, 9001)
        observation, info = env.reset(seed=9001)
        assert observation.shape == (OBSERVATION_SIZE,)
        assert info["shock_schedule_sha256"]
        terminated = False
        while not terminated:
            observation, _, terminated, truncated, day_info = env.step(
                np.zeros(5, dtype=np.float32)
            )
            assert truncated is False
            assert day_info["day"]["projection"]["constraint_violations"] == 0
        assert observation.shape == (OBSERVATION_SIZE,)
        trajectories.append(env.render())
    assert trajectories[0] == trajectories[1]
    assert len(trajectories[0]) == 7


def test_ortools_baseline_uses_bounds_and_has_zero_violations() -> None:
    result = rollout_baseline(Scenario(), 424242)
    assert result["planner"] == "ortools_glop_baseline"
    assert result["constraint_violations"] == 0
    assert result["violation_breakdown"] == {
        "budget_violations": 0,
        "lower_violations": 0,
        "sum_violations": 0,
        "upper_violations": 0,
    }
    assert all(day["planner_evidence"]["library"] == "OR-Tools" for day in result["trajectory"])


def test_authored_scenario_families_are_bounded_repeatable_and_disjoint() -> None:
    assert not set(TRAINING_SEEDS).intersection(HELD_OUT_SEEDS)
    assert not {family.id for family in TRAINING_FAMILIES}.intersection(
        family.id for family in HELD_OUT_FAMILIES
    )
    for family in (*TRAINING_FAMILIES, *HELD_OUT_FAMILIES):
        first = family.build(271700)
        second = family.build(271700)
        assert first == second
        assert 7 <= first.horizon_days <= 30
        assert 50 <= first.daily_budget <= 500
        assert all(0.05 <= value <= 0.95 for value in first.initial_services)
        assert all(0.5 <= value <= 2.0 for value in first.priorities)
