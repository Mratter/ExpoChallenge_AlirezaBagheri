#!/usr/bin/env python3
"""Maintain additive achieved-count overlays on frozen receipt-rendered reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import load_json_object  # noqa: E402
from scripts.reporting_denominators import (  # noqa: E402
    achieved_count_row,
    development_achieved_count_reference,
    final_achieved_count_reference,
    oracle_failure_diagnostics,
)

BEGIN = "<!-- BEGIN ACHIEVED-COUNT REPORTING OVERLAY -->"
END = "<!-- END ACHIEVED-COUNT REPORTING OVERLAY -->"
LEGACY_UNION_LABEL = "Known-feasible union"
PRESENTATION_UNION_LABEL = "Demonstrated union"
FINAL_RECEIPT_SHA256 = "6c21f3be7dc1af8c7bbc00e671210e315e42d6211bd276eccd45adc74421f373"
FINAL_ORACLE_SHA256 = "baf5aa6ec8e419a50f87e744eac7779f30a53b6aab60018ff1a7043126b0b5ec"
DEV_ORACLE_SHA256 = "8a83eac8b9de2c439c5441e4be97631bf0d294bae4b5a36eb82a12aa11fc4287"
DEV_BASELINES_SHA256 = "aea0b85da3c46338d44727e5854be64655fb89c124917063edbef5ffd4f46c27"
TRAINING_SUMMARY_SHA256 = "7c39de680d74e22a4429a940f08473f9572dd50e9757ec94e006124e05a2925d"
FINAL_RECEIPT = ROOT / "internal/evaluation_runs/v4/final-evaluation-200.success.json"
FINAL_ORACLE = ROOT / "internal/developmental_runs/v4/clairvoyant-oracle-200-final.json"
DEV_ORACLE = ROOT / "internal/developmental_runs/v4/clairvoyant-oracle-200-dev.json"
DEV_BASELINES = ROOT / "internal/developmental_runs/v4/development-baselines-200.json"
TRAINING_SUMMARY = ROOT / "internal/developmental_runs/v4/training-study-200-summary.json"
REPORTS = {
    "final": ROOT / "benchmarks/v4/final-results-200.md",
    "oracle": ROOT / "benchmarks/v4/clairvoyant-oracle-200.md",
    "development": ROOT / "benchmarks/v4/development-baselines-200.md",
    "training": ROOT / "benchmarks/v4/training-study-200.md",
}


def _load(path: Path, digest: str, label: str) -> dict[str, Any]:
    return load_json_object(path, label, expected_sha256=digest)


def load_evidence() -> dict[str, dict[str, Any]]:
    """Load every overlay input under its retained immutable digest."""

    return {
        "final": _load(FINAL_RECEIPT, FINAL_RECEIPT_SHA256, "final receipt"),
        "final_oracle": _load(FINAL_ORACLE, FINAL_ORACLE_SHA256, "final oracle"),
        "dev_oracle": _load(DEV_ORACLE, DEV_ORACLE_SHA256, "dev oracle"),
        "dev_baselines": _load(DEV_BASELINES, DEV_BASELINES_SHA256, "dev baselines"),
        "training": _load(TRAINING_SUMMARY, TRAINING_SUMMARY_SHA256, "training"),
    }


def strip_overlay(markdown: str) -> str:
    """Remove exactly one marker-bounded overlay and preserve all other bytes."""

    if BEGIN not in markdown and END not in markdown:
        return markdown
    if markdown.count(BEGIN) != 1 or markdown.count(END) != 1:
        raise ValueError("report must contain exactly one complete overlay")
    start = markdown.index(BEGIN)
    stop = markdown.index(END, start) + len(END)
    if markdown[start - 2 : start] != "\n\n":
        raise ValueError("overlay must follow one blank line")
    if markdown[stop : stop + 2] != "\n\n":
        raise ValueError("overlay must precede one blank line")
    return markdown[: start - 2] + markdown[stop:]


def restore_frozen_report_wording(markdown: str) -> str:
    """Reverse the one exact legacy label modernization for renderer checks."""

    restored = markdown.replace(PRESENTATION_UNION_LABEL, LEGACY_UNION_LABEL)
    if restored.count(LEGACY_UNION_LABEL) != markdown.count(
        PRESENTATION_UNION_LABEL
    ):
        raise ValueError("unexpected demonstrated-union wording")
    return restored


def apply_overlay(markdown: str, overlay: str) -> str:
    """Place an overlay after the first Markdown heading."""

    base = strip_overlay(markdown).replace(
        LEGACY_UNION_LABEL, PRESENTATION_UNION_LABEL
    )
    first_newline = base.find("\n")
    if first_newline < 0 or not base.startswith("# "):
        raise ValueError("report lacks a top-level heading")
    block = f"\n\n{BEGIN}\n{overlay.rstrip()}\n{END}"
    return base[:first_newline] + block + base[first_newline:]


def _development_reference(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    id_map = {
        "heuristic": "heuristic",
        "teacher": "teacher",
        "tuned": "tuned",
        "onnx:tests/fixtures/legacy_policy.onnx": "legacy",
    }
    counts = {
        id_map[row["id"]]: row["solved_count"]
        for row in evidence["dev_baselines"]["policies"]
    }
    mean = evidence["training"]["baseline"]["aggregate"]["mean_solved_count"]
    return development_achieved_count_reference(
        oracle_receipt=evidence["dev_oracle"],
        baseline_solved_counts=counts,
        seed_mean_solved_count=mean,
    )


def _final_overlay(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    reference = final_achieved_count_reference(evidence["final"])
    failures = oracle_failure_diagnostics(evidence["final_oracle"]["rows"])
    oracle_aggregates = evidence["final_oracle"]["planner_aggregates"]
    regression = evidence["final"]["bound_evidence"]["regression_gate"]
    if regression != {
        "legacy_onnx_fixture_solved_count": 125,
        "path": "tests/test_consolidation_gate.py",
        "preparedness_teacher_solved_count": 139,
        "reactive_heuristic_solved_count": 72,
        "sha256": "97bdeb13556a2fdb9b291c62e699da739441e593ad57f6a5adc014e7ece38638",
    }:
        raise ValueError("final regression-gate evidence drifted")
    results = (
        (
            "Privileged CEM",
            oracle_aggregates["clairvoyant_oracle_cem"]["solved_count"],
        ),
        ("Shipped v4 PPO", evidence["final"]["aggregate"]["solved_count"]),
        ("Tuned rule", oracle_aggregates["tuned_rule"]["solved_count"]),
        (
            "Preparedness teacher",
            regression["preparedness_teacher_solved_count"],
        ),
        ("Selected MPC", oracle_aggregates["selected_mpc_k5"]["solved_count"]),
        ("Legacy fixture", regression["legacy_onnx_fixture_solved_count"]),
        ("Reactive heuristic", regression["reactive_heuristic_solved_count"]),
    )
    lines = [
        "## Demonstrated-achievable reference",
        "",
        (
            "The shipped v4 policy solved **163/182 = 89.6%** relative to the "
            "privileged CEM achieved-count reference (Wilson 95% **[84.3%, "
            "93.2%]**), alongside its raw **163/200** held-out result."
        ),
        "",
        (
            "**Demonstrated-achievable reference denominator = the 182 of 200 "
            "final cases solved by the privileged future-aware CEM run; its 18 "
            "search failures are not proofs of infeasibility.**"
        ),
        "",
        "| Method | Raw solved / 200 | Achieved-count ratio (/182 reference) | Wilson 95% CI on /182 |",
        "|---|---:|---:|---:|",
    ]
    for label, solved in results:
        row = achieved_count_row(solved_count=solved, reference_count=182)
        lines.append(
            f"| {label} | {solved}/200 | {solved}/182 = "
            f"{row['achieved_count_ratio']:.1%} | "
            f"[{row['wilson_95'][0]:.4f}, {row['wilson_95'][1]:.4f}] |"
        )
    pairing = reference["pairing"]
    lines.extend(
        [
            "",
            (
                "The headline ratio compares aggregate solved counts, not a "
                "contained case set: casewise policy coverage is **162/182 = "
                "89.0%** because one case is policy-only. The two recorded methods "
                "jointly demonstrate solutions on **183/200** cases; 20 oracle-only "
                "cases demonstrate remaining headroom."
            ),
            "",
            (
                f"Receipt audit of the oracle's 18 failed searches found "
                f"**{failures['nonnegative_tail_margin_count']}/18** with nonnegative "
                f"minimum tail margin (mean {failures['mean_minimum_tail_margin']:+.8f}; "
                f"range {failures['minimum_tail_margin']:+.8f} to "
                f"{failures['maximum_tail_margin']:+.8f}). Failed-check occurrences "
                "were 14 resilience-AUC, 9 critical-day-cap, and 3 assessment-tail "
                "checks, with overlaps. The portable receipt does not retain numeric "
                "day-cap excess, and these search failures do not prove infeasibility."
            ),
            "",
            (
                "The /182 intervals are descriptive post-hoc Wilson intervals. They "
                "and the raw /200 interval treat cases as Bernoulli units and do not "
                "model clustering within five fixed scenario families."
            ),
        ]
    )
    if sum(pairing.values()) != 200:
        raise ValueError("final pairing drifted")
    return "\n".join(lines)


def _development_overlay(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    reference = _development_reference(evidence)
    rows = reference["rows"]
    values = (
        ("Privileged CEM", 187, "privileged_cem"),
        ("Selected shipped v4", 178, "selected_v4"),
        ("Five-seed 2M endpoint mean", 171.4, "five_seed_endpoint_mean"),
        ("Tuned rule", 160, "tuned"),
        ("Preparedness teacher", 151, "teacher"),
        ("Selected MPC", 153, "selected_mpc"),
        ("Legacy fixture", 141, "legacy"),
        ("Reactive heuristic", 91, "heuristic"),
    )
    lines = [
        "## Demonstrated-achievable reference",
        "",
        (
            "**Demonstrated-achievable reference denominator = the 187 of 200 "
            "development cases solved by the privileged future-aware CEM run; its "
            "13 search failures are not proofs of infeasibility.**"
        ),
        "",
        "| Development result | Raw solved / 200 | Achieved-count ratio (/187 reference) | Wilson 95% CI on /187 |",
        "|---|---:|---:|---:|",
    ]
    for label, solved, key in values:
        row = rows[key]
        interval = (
            "not reported: optimizer-seed mean"
            if row["wilson_95"] is None
            else f"[{row['wilson_95'][0]:.4f}, {row['wilson_95'][1]:.4f}]"
        )
        lines.append(
            f"| {label} | {solved}/200 | {solved}/187 = "
            f"{row['achieved_count_ratio']:.1%} | {interval} |"
        )
    lines.extend(
        [
            "",
            (
                "The headline **178/187 = 95.2%** is an aggregate achieved-count "
                "ratio; casewise policy coverage is **177/187 = 94.7%** because one "
                "case is policy-only. The two methods jointly demonstrate solutions "
                "on **188/200** cases, and 10 oracle-only cases demonstrate remaining "
                "headroom. Ratios and intervals are descriptive and post-hoc."
            ),
        ]
    )
    return "\n".join(lines)


def _oracle_overlay(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    development = _development_overlay(evidence)
    return development + "\n\n" + (
        "On final, the same fixed CEM protocol solved **182/200** and the separately "
        "authorized frozen policy solved **163/200**. The **163/182 = 89.6%** "
        "headline is an aggregate achieved-count ratio; casewise coverage is "
        "**162/182 = 89.0%** because one case is policy-only, and the two methods "
        "jointly demonstrate solutions on **183/200** cases."
    )


def _training_overlay(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    reference = _development_reference(evidence)
    ratio = reference["rows"]["five_seed_endpoint_mean"]["achieved_count_ratio"]
    return "\n".join(
        [
            "## Demonstrated-achievable reference",
            "",
            (
                "**Demonstrated-achievable reference denominator = the 187 of 200 "
                "development cases solved by the privileged future-aware CEM run; "
                "its 13 search failures are not proofs of infeasibility.**"
            ),
            "",
            (
                f"The five registered 2M endpoints average **171.4/200 raw**, or "
                f"**171.4/187 = {ratio:.1%}** of that achieved-count reference; "
                "selection chose **178/200 raw**, or **178/187 = 95.2%**. The mean "
                "answers reproducibility across optimizer seeds, while 178 answers "
                "development checkpoint selection. No Wilson interval is reported "
                "for the optimizer-seed mean; the achieved-count ratios are "
                "descriptive and post-hoc."
            ),
        ]
    )


def overlays() -> dict[str, str]:
    evidence = load_evidence()
    return {
        "final": _final_overlay(evidence),
        "oracle": _oracle_overlay(evidence),
        "development": _development_overlay(evidence),
        "training": _training_overlay(evidence),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = overlays()
    stale = []
    for name, path in REPORTS.items():
        current = path.read_text(encoding="utf-8")
        rendered = apply_overlay(current, expected[name])
        if args.check:
            if current != rendered:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_text(rendered, encoding="utf-8", newline="\n")
    if stale:
        print("stale achieved-count overlays: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
