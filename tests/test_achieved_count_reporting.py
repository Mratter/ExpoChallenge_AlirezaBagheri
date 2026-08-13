from __future__ import annotations

import hashlib
import re

import pytest

from backend.app.shared_evidence import file_sha256
from scripts import build_development_baselines, publish_final_evaluation_v4
from scripts.publish_oracle_study import _render_markdown as render_oracle_markdown
from scripts.render_achieved_count_reports import (
    BEGIN,
    DEV_BASELINES,
    DEV_BASELINES_SHA256,
    DEV_ORACLE,
    DEV_ORACLE_SHA256,
    FINAL_ORACLE,
    FINAL_ORACLE_SHA256,
    FINAL_RECEIPT,
    FINAL_RECEIPT_SHA256,
    REPORTS,
    TRAINING_SUMMARY,
    TRAINING_SUMMARY_SHA256,
    apply_overlay,
    load_evidence,
    overlays,
    restore_frozen_report_wording,
    strip_overlay,
)
from scripts.reporting_denominators import (
    ReportingDenominatorError,
    development_achieved_count_reference,
    final_achieved_count_reference,
    oracle_failure_diagnostics,
)

TRAINING_REPORT_BASE_SHA256 = (
    "98e252340b66d09c0a52e7a6edc6c5acabab260a1f5ec13c1e2c4cd84e7dea54"
)
FROZEN_REPORT_SOURCE_SHA256 = {
    "scripts/publish_final_evaluation_v4.py": (
        "017c6ea61a1bb7ffe69c867b6efb4e1aff8c1ffc60584230704d3c2171f509a4"
    ),
    "scripts/publish_oracle_study.py": (
        "5f467d7f4ccd96398f32604ff0c71899558261db8c3dc547b4f428f9d540becd"
    ),
    "scripts/build_development_baselines.py": (
        "1886ff01b487a87285167ab79d43d35ff762ec3ce72bcd90e7e12fdc848a2178"
    ),
}


def test_overlay_inputs_remain_bound_to_retained_receipts() -> None:
    expected = {
        FINAL_RECEIPT: FINAL_RECEIPT_SHA256,
        FINAL_ORACLE: FINAL_ORACLE_SHA256,
        DEV_ORACLE: DEV_ORACLE_SHA256,
        DEV_BASELINES: DEV_BASELINES_SHA256,
        TRAINING_SUMMARY: TRAINING_SUMMARY_SHA256,
    }
    assert {path: file_sha256(path) for path in expected} == expected
    root = FINAL_RECEIPT.parents[3]
    assert {
        path: file_sha256(root / path) for path in FROZEN_REPORT_SOURCE_SHA256
    } == FROZEN_REPORT_SOURCE_SHA256


def test_final_reference_math_and_failure_diagnostics() -> None:
    evidence = load_evidence()
    result = final_achieved_count_reference(evidence["final"])

    assert result["reference_count"] == 182
    assert result["policy"]["achieved_count_ratio"] == pytest.approx(163 / 182)
    assert result["policy"]["wilson_95"] == [0.8427110768, 0.9321429354]
    assert result["pairing"] == {
        "both": 162,
        "policy_only": 1,
        "oracle_only": 20,
        "neither": 17,
    }
    assert result["casewise_policy_coverage"] == pytest.approx(162 / 182)
    assert result["demonstrably_solvable_union_count"] == 183

    diagnostics = oracle_failure_diagnostics(evidence["final_oracle"]["rows"])
    assert diagnostics == {
        "failure_count": 18,
        "failure_reason_code_histogram": {
            "assessment_tail_targets_met": 3,
            "critical_service_day_cap_met": 9,
            "resilience_auc_met": 14,
        },
        "nonnegative_tail_margin_count": 15,
        "mean_minimum_tail_margin": pytest.approx(0.03391064888888889),
        "minimum_tail_margin": -0.01072877,
        "maximum_tail_margin": 0.11473788,
    }


def test_development_reference_math() -> None:
    evidence = load_evidence()
    counts = {
        {"heuristic": "heuristic", "teacher": "teacher", "tuned": "tuned"}.get(
            row["id"], "legacy"
        ): row["solved_count"]
        for row in evidence["dev_baselines"]["policies"]
    }
    result = development_achieved_count_reference(
        oracle_receipt=evidence["dev_oracle"],
        baseline_solved_counts=counts,
        seed_mean_solved_count=171.4,
    )

    assert result["reference_count"] == 187
    assert result["rows"]["selected_v4"]["achieved_count_ratio"] == pytest.approx(
        178 / 187
    )
    assert result["rows"]["selected_v4"]["wilson_95"] == [
        0.9110759827,
        0.9744758311,
    ]
    assert result["rows"]["selected_mpc"]["achieved_count_ratio"] == pytest.approx(
        153 / 187
    )
    assert result["rows"]["five_seed_endpoint_mean"][
        "achieved_count_ratio"
    ] == pytest.approx(171.4 / 187)
    assert result["rows"]["tuned"]["achieved_count_ratio"] == pytest.approx(
        160 / 187
    )
    assert result["pairing"] == {
        "both": 177,
        "policy_only": 1,
        "oracle_only": 10,
        "neither": 12,
    }
    assert result["casewise_policy_coverage"] == pytest.approx(177 / 187)
    assert result["demonstrably_solvable_union_count"] == 188


def test_overlay_is_current_and_preserves_frozen_renderer_outputs() -> None:
    evidence = load_evidence()
    expected_overlays = overlays()
    for name, path in REPORTS.items():
        current = path.read_text(encoding="utf-8")
        assert current.count(BEGIN) == 1
        assert apply_overlay(current, expected_overlays[name]) == current

    final_base = publish_final_evaluation_v4.render_markdown(
        evidence["final"], FINAL_RECEIPT_SHA256
    )
    assert restore_frozen_report_wording(
        strip_overlay(REPORTS["final"].read_text(encoding="utf-8"))
    ) == final_base

    development_base = build_development_baselines.render_markdown(
        evidence["dev_baselines"]
    )
    assert (
        strip_overlay(REPORTS["development"].read_text(encoding="utf-8"))
        == development_base
    )

    raw = evidence["dev_oracle"]["publication"]["raw_evidence"]
    oracle_base = render_oracle_markdown(
        portable={
            "dev": evidence["dev_oracle"],
            "final": evidence["final_oracle"],
        },
        raw_receipt_hashes={
            "dev": raw["receipt_sha256"],
            "final": evidence["final_oracle"]["publication"]["raw_evidence"][
                "receipt_sha256"
            ],
        },
        raw_summary_sha256=raw["summary_sha256"],
        raw_protocol_sha256=raw["protocol_sha256"],
    )
    assert restore_frozen_report_wording(
        strip_overlay(REPORTS["oracle"].read_text(encoding="utf-8"))
    ) == oracle_base

    training_base = strip_overlay(REPORTS["training"].read_text(encoding="utf-8"))
    assert hashlib.sha256(training_base.encode("utf-8")).hexdigest() == (
        TRAINING_REPORT_BASE_SHA256
    )


def test_overlays_define_reference_and_disclose_non_nesting() -> None:
    expected = overlays()
    for overlay in expected.values():
        assert re.search(r"\b(?:possible|feasible)\b", overlay, re.IGNORECASE) is None
        assert "Known-feasible union" not in overlay
    assert (
        "Demonstrated-achievable reference denominator = the 182 of 200 final cases"
        in expected["final"]
    )
    assert "163/182 = 89.6%" in expected["final"]
    assert "162/182 = 89.0%" in expected["final"]
    assert "183/200" in expected["final"]
    for name in ("development", "oracle", "training"):
        assert (
            "Demonstrated-achievable reference denominator = the 187 of 200 development cases"
            in expected[name]
        )
    assert "178/187 = 95.2%" in expected["development"]
    assert "171.4/187 = 91.7%" in expected["training"]
    assert "160/187 = 85.6%" in expected["development"]


def test_reference_validation_fails_closed_on_nonnesting_drift() -> None:
    evidence = load_evidence()
    mutated = dict(evidence["final"])
    comparison = dict(mutated["oracle_comparison"])
    comparison["pairing"] = {**comparison["pairing"], "policy_only": 0}
    mutated["oracle_comparison"] = comparison
    with pytest.raises(ReportingDenominatorError, match="pairing count drifted"):
        final_achieved_count_reference(mutated)


def test_manual_docs_keep_the_achieved_count_reporting_contract() -> None:
    root = FINAL_RECEIPT.parents[3]
    readme = (root / "README.md").read_text(encoding="utf-8")
    code_tour = (root / "docs/CODE_TOUR.md").read_text(encoding="utf-8")
    evidence = (root / "docs/EVIDENCE.md").read_text(encoding="utf-8")
    plan = (root / "docs/TRAINING_DEPLOYMENT_PLAN.md").read_text(
        encoding="utf-8"
    )

    assert "163 / 200 (81.5%)" in readme
    assert "oracle-solved reference" in readme
    assert "not a proof that the other 18 cases are infeasible" in readme
    assert "162 both solved, 1 policy-only, 20 oracle-only, and 17 neither" in readme
    assert "183 / 200" in readme
    for comparator in (
        "Privileged future-aware CEM",
        "**Shipped v4 PPO**",
        "Tuned constant rule",
        "Preparedness teacher",
        "Causal MPC, `k=5`",
        "Legacy ONNX fixture",
        "Reactive heuristic",
    ):
        assert comparator in readme
    for solved_count in (182, 163, 147, 139, 135, 125, 72):
        assert f"**{solved_count} / 200**" in readme

    detailed_docs = "\n".join((code_tour, evidence, plan))
    for value in (
        "163 / 182 = 89.6%",
        "163 / 200",
        "162 / 182 = 89.0%",
        "183 / 200",
        "171.4 / 187 = 91.7%",
        "171.4 / 200",
        "178 / 187 = 95.2%",
        "178 / 200",
    ):
        assert value in detailed_docs
    assert "Development oracle-solved reference" in plan
    assert "13 search failures are not proofs of infeasibility" in plan
    assert "Final oracle-solved reference" in plan
    assert "18 search failures are not proofs of infeasibility" in plan
    assert "[0.8427, 0.9321]" in evidence
