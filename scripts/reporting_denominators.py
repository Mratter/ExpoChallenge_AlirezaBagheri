"""Reporting-only achieved-count reference calculations from retained evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import floor
from typing import Any

from backend.app.shared_evidence import wilson_interval


class ReportingDenominatorError(RuntimeError):
    """Raised when retained evidence cannot support a reporting denominator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportingDenominatorError(message)


def achieved_count_row(
    *,
    solved_count: float,
    reference_count: int,
    digits: int = 10,
) -> dict[str, Any]:
    """Return one achieved-count ratio row without implying set containment."""

    _require(reference_count > 0, "reference count must be positive")
    _require(0 <= solved_count <= reference_count, "solved count exceeds reference")
    _require(
        solved_count == floor(solved_count),
        "Wilson intervals require an integer solved count",
    )
    return {
        "solved_count": solved_count,
        "reference_count": reference_count,
        "achieved_count_ratio": solved_count / reference_count,
        "wilson_95": wilson_interval(
            int(solved_count), reference_count, digits=digits
        ),
    }


def final_achieved_count_reference(
    final_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the final CEM achieved-count reference."""

    aggregate = final_receipt.get("aggregate")
    comparison = final_receipt.get("oracle_comparison")
    _require(isinstance(aggregate, Mapping), "final aggregate missing")
    _require(isinstance(comparison, Mapping), "final oracle comparison missing")
    pairing = comparison.get("pairing")
    _require(isinstance(pairing, Mapping), "final pairing missing")
    counts = {key: int(pairing.get(key, -1)) for key in (
        "both",
        "policy_only",
        "oracle_only",
        "neither",
    )}
    _require(sum(counts.values()) == 200, "final pairing count drifted")
    policy_count = counts["both"] + counts["policy_only"]
    oracle_count = counts["both"] + counts["oracle_only"]
    union_count = policy_count + counts["oracle_only"]
    _require(policy_count == aggregate.get("solved_count") == 163, "final policy count drifted")
    _require(oracle_count == 182, "final oracle count drifted")
    _require(
        union_count == comparison.get("known_feasible_union_count") == 183,
        "final demonstrated union drifted",
    )
    return {
        "reference_count": oracle_count,
        "reference_definition": (
            "the 182 of 200 final cases solved by the privileged future-aware "
            "CEM run; its 18 search failures are not proofs of infeasibility"
        ),
        "policy": achieved_count_row(
            solved_count=policy_count, reference_count=oracle_count
        ),
        "pairing": counts,
        "casewise_policy_coverage": counts["both"] / oracle_count,
        "demonstrably_solvable_union_count": union_count,
    }


def development_achieved_count_reference(
    *,
    oracle_receipt: Mapping[str, Any],
    baseline_solved_counts: Mapping[str, int],
    seed_mean_solved_count: float,
) -> dict[str, Any]:
    """Validate and summarize the development CEM achieved-count reference."""

    oracle = oracle_receipt.get("planner_aggregates", {}).get(
        "clairvoyant_oracle_cem"
    )
    comparison = oracle_receipt.get("development_shipped_policy_comparison")
    _require(isinstance(oracle, Mapping), "development oracle aggregate missing")
    _require(isinstance(comparison, Mapping), "development comparison missing")
    pairing = comparison.get("pairing")
    _require(isinstance(pairing, Mapping), "development pairing missing")
    counts = {key: int(pairing.get(key, -1)) for key in (
        "both",
        "policy_only",
        "oracle_only",
        "neither",
    )}
    _require(sum(counts.values()) == 200, "development pairing count drifted")
    oracle_count = int(oracle.get("solved_count", -1))
    mpc_keys = [
        key
        for key in oracle_receipt.get("planner_aggregates", {})
        if key.startswith("selected_mpc_k")
    ]
    _require(len(mpc_keys) == 1, "development selected MPC aggregate missing")
    mpc_count = int(
        oracle_receipt["planner_aggregates"][mpc_keys[0]].get("solved_count", -1)
    )
    policy_count = counts["both"] + counts["policy_only"]
    union_count = policy_count + counts["oracle_only"]
    _require(oracle_count == counts["both"] + counts["oracle_only"] == 187, "development oracle count drifted")
    _require(policy_count == comparison.get("shipped_policy_solved_count") == 178, "development policy count drifted")
    _require(union_count == comparison.get("known_feasible_union_count") == 188, "development demonstrated union drifted")
    expected = {"heuristic": 91, "teacher": 151, "tuned": 160, "legacy": 141}
    _require(dict(baseline_solved_counts) == expected, "development baselines drifted")
    _require(seed_mean_solved_count == 171.4, "development seed mean drifted")
    rows = {
        "privileged_cem": achieved_count_row(
            solved_count=oracle_count, reference_count=oracle_count
        ),
        "selected_v4": achieved_count_row(
            solved_count=policy_count, reference_count=oracle_count
        ),
        "selected_mpc": achieved_count_row(
            solved_count=mpc_count, reference_count=oracle_count
        ),
        "five_seed_endpoint_mean": {
            "mean_solved_count": seed_mean_solved_count,
            "reference_count": oracle_count,
            "achieved_count_ratio": seed_mean_solved_count / oracle_count,
            "wilson_95": None,
            "interval_not_reported": "the numerator is a mean across optimizer seeds",
        },
    }
    _require(mpc_count == 153, "development selected MPC count drifted")
    rows.update(
        {
            name: achieved_count_row(
                solved_count=solved, reference_count=oracle_count
            )
            for name, solved in expected.items()
        }
    )
    return {
        "reference_count": oracle_count,
        "reference_definition": (
            "the 187 of 200 development cases solved by the privileged "
            "future-aware CEM run; its 13 search failures are not proofs of "
            "infeasibility"
        ),
        "rows": rows,
        "pairing": counts,
        "casewise_policy_coverage": counts["both"] / oracle_count,
        "demonstrably_solvable_union_count": union_count,
    }


def oracle_failure_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute final CEM failure diagnostics from portable per-case rows."""

    oracle_rows = [row.get("clairvoyant_oracle_cem") for row in rows]
    _require(
        len(oracle_rows) == 200 and all(isinstance(row, Mapping) for row in oracle_rows),
        "oracle rows missing",
    )
    failures = [row for row in oracle_rows if not row.get("solved")]
    _require(len(failures) == 18, "oracle failure count drifted")
    histogram: dict[str, int] = {}
    for row in failures:
        for reason in row.get("reason_codes", []):
            histogram[reason] = histogram.get(reason, 0) + 1
    margins = [float(row["minimum_tail_margin"]) for row in failures]
    return {
        "failure_count": len(failures),
        "failure_reason_code_histogram": dict(sorted(histogram.items())),
        "nonnegative_tail_margin_count": sum(margin >= 0.0 for margin in margins),
        "mean_minimum_tail_margin": sum(margins) / len(margins),
        "minimum_tail_margin": min(margins),
        "maximum_tail_margin": max(margins),
    }
