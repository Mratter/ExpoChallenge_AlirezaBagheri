from __future__ import annotations

from scripts.evaluate import ProbeRow, exact_mcnemar_p, paired_contingency


def _row(row_id: str, solved: bool) -> ProbeRow:
    return ProbeRow(
        row_id=row_id,
        family_id="family",
        case_seed=1,
        tape_seed=2,
        solved=solved,
        status="solved" if solved else "failed",
        reason_codes=() if solved else ("assessment_tail_targets_met",),
        resilience_auc=0.5,
        minimum_tail_margin=0.01,
        critical_service_days=0,
        hard_violation_count=0,
        max_conservation_residual=0.0,
        trajectory_sha256="0" * 64,
    )


def test_exact_mcnemar_matches_known_extremes() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 5) == 0.0625
    assert exact_mcnemar_p(2, 3) == 1.0


def test_paired_contingency_reports_all_cells() -> None:
    left = [_row("both", True), _row("left", True), _row("right", False), _row("neither", False)]
    right = [_row("both", True), _row("left", False), _row("right", True), _row("neither", False)]

    result = paired_contingency(left, right)

    assert result == {
        "both_solved": 1,
        "left_only": 1,
        "right_only": 1,
        "neither": 1,
        "table": [[1, 1], [1, 1]],
        "discordant_count": 2,
        "exact_mcnemar_p_two_sided": 1.0,
    }
