from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_training_study import (
    ABLATIONS,
    ABLATION_SEEDS,
    BASELINE,
    CANONICAL_DEVELOPMENT_CASE_COUNT,
    DEVELOPMENT_CASE_COUNT,
    LEGACY_DEVELOPMENT_CASE_COUNT,
    POLICY_SEEDS,
    StudyError,
    summarize_rows,
    receipt_development_case_count,
    training_command,
)


def test_registered_study_has_five_sweep_seeds_and_five_matched_ablations(
    tmp_path: Path,
) -> None:
    assert POLICY_SEEDS == (37_017, 47_017, 57_017, 67_017, 77_017)
    assert DEVELOPMENT_CASE_COUNT == CANONICAL_DEVELOPMENT_CASE_COUNT == 200
    assert ABLATION_SEEDS == POLICY_SEEDS[:3]
    assert BASELINE.transitions == 2_000_000
    assert BASELINE.reward_profile == "v3_equivalent"
    assert BASELINE.preparedness_alignment_coefficient == 10.0
    assert [arm.name for arm in ABLATIONS] == [
        "no_bc_warm_start",
        "risk_averse_reward",
        "no_vec_normalize",
        "preparedness_alignment_2",
        "budget_645k",
    ]
    for arm in (BASELINE, *ABLATIONS):
        command = training_command(tmp_path, arm, POLICY_SEEDS[0])
        assert "--lanes" in command and command[command.index("--lanes") + 1] == "20"
        assert "--learning-rate" in command
        assert command[command.index("--learning-rate") + 1] == "0.000075"
        assert "--target-kl" in command
        assert command[command.index("--target-kl") + 1] == "0.02"
        assert "--freeze-observation-rms" in command
        assert "final" not in " ".join(command).lower()


def test_summary_reports_sample_standard_deviation() -> None:
    rows = [
        {"solved_count": 170, "solve_rate": 0.85},
        {"solved_count": 175, "solve_rate": 0.875},
        {"solved_count": 180, "solve_rate": 0.9},
    ]
    summary = summarize_rows(rows)
    assert summary["mean_solved_count"] == 175.0
    assert summary["sample_std_solved_count"] == 5.0
    assert summary["mean_solve_rate"] == pytest.approx(0.875)
    assert summary["sample_std_solve_rate"] == pytest.approx(0.025)
    with pytest.raises(StudyError):
        summarize_rows(rows[:1])


def test_historical_40_case_receipt_count_remains_readable() -> None:
    payload = {"development": {"case_count": LEGACY_DEVELOPMENT_CASE_COUNT}}

    assert receipt_development_case_count(payload) == 40
