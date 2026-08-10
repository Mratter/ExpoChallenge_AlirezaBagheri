from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.evaluate import (
    DEFAULT_ONNX_PATH,
    DEFAULT_POLICIES,
    ProbeRow,
    exact_mcnemar_p,
    paired_contingency,
    resolve_policy,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_REPORT = ROOT / "docs" / "evidence" / "legacy-final-40.json"


def test_default_model_is_the_explicit_legacy_fixture() -> None:
    assert DEFAULT_ONNX_PATH.parts[-3:] == (
        "tests",
        "fixtures",
        "legacy_policy.onnx",
    )
    assert DEFAULT_POLICIES[-1] == "onnx:tests/fixtures/legacy_policy.onnx"
    policy = resolve_policy(DEFAULT_POLICIES[-1])
    assert policy.label == DEFAULT_POLICIES[-1]
    assert policy.kind == "onnx"


def test_legacy_report_is_retained_byte_for_byte() -> None:
    assert hashlib.sha256(LEGACY_REPORT.read_bytes()).hexdigest() == (
        "f6d3b654ca6b2831af5bec07530b81ecf0e72b2aae44029a805d98325bfe5fb3"
    )


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
