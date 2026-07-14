import numpy as np

from backend.app.models import Scenario
from backend.app.simulator import generate_shock_schedule, project_capped_simplex


def test_projector_obeys_caps_and_total() -> None:
    proposal = np.array([100.0, 20.0, 15.0, 5.0, 40.0])
    lower = np.array([0.0, 7.2, 0.0, 0.0, 7.2])
    upper = np.full(5, 90.0)
    projected, evidence = project_capped_simplex(proposal, 180.0, lower, upper)

    assert projected.sum() == 180.0
    assert np.all(projected >= lower)
    assert np.all(projected <= upper)
    assert evidence["sum"] == 180.0


def test_shock_tape_is_repeatable_and_forced_override_does_not_shift_tail() -> None:
    scenario = Scenario()
    first = generate_shock_schedule(scenario, 424242)
    second = generate_shock_schedule(scenario, 424242)

    assert first == second
    assert first[4].forced is True
    assert first[4].type == "utility"
    assert first[4].severity == 0.26

