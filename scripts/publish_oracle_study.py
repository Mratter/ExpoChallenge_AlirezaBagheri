#!/usr/bin/env python3
"""Validate and publish the matched 200-case privileged-oracle study.

The expensive runner deliberately writes resumable raw evidence outside the
repository.  This tool validates that evidence against the current canonical
split rosters, fixed historical CEM budget, shipped-policy parity receipt, and
source identities before creating small, portable tracked receipts.  It never
runs a planner or a learned policy and refuses to overwrite prior evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
    wilson_interval,
)
from scripts.headroom import (  # noqa: E402
    MPC_HORIZONS,
    PlannerResult,
    _result_from_receipt,
    aggregate_results,
)
from scripts.run_oracle_study import (  # noqa: E402
    CANONICAL_CASE_COUNT,
    HISTORICAL_MPC_CONFIG,
    HISTORICAL_ORACLE_CONFIG,
    HISTORICAL_RECEIPT,
    HISTORICAL_RECEIPT_SHA256,
    SCHEMA_VERSION as RAW_SCHEMA_VERSION,
    SHIPPED_ONNX_SHA256,
    SHIPPED_PARITY_ROWS_SHA256,
    SHIPPED_PARITY_SHA256,
    TOOL_ID as RAW_TOOL_ID,
    _case_contract,
    _source_identity,
    _study_contract,
    build_cases,
    build_reference_comparison,
)

TOOL_ID = "publish_oracle_study.py"
SCHEMA_VERSION = 1
PORTABLE_KIND = "city-recovery-clairvoyant-oracle-200-portable-receipt"
TRACKED_RECEIPTS = {
    "dev": ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "clairvoyant-oracle-200-dev.json",
    "final": ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "clairvoyant-oracle-200-final.json",
}
TRACKED_MARKDOWN = ROOT / "benchmarks" / "v4" / "clairvoyant-oracle-200.md"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ABSOLUTE_PATH_FRAGMENT = re.compile(
    r"(?:^|[\s(\"'])(?:[A-Za-z]:[\\/]|\\\\|/[A-Za-z0-9_.-])"
)
_FIXED_HISTORICAL_RECEIPT_SHA256 = (
    "f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9"
)
_FIXED_SHIPPED_PARITY_SHA256 = (
    "e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c"
)
_FIXED_SHIPPED_PARITY_ROWS_SHA256 = (
    "ca9320566b86dfb7a02d2cb9232c7a28c80f08dbbd700dffc6d2af9af1c22d6b"
)
_FIXED_SHIPPED_ONNX_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)
_FIXED_MPC_CONFIG = {
    "population": 48,
    "elite_count": 6,
    "iterations": 5,
    "fantasies": 4,
    "initial_std": 0.35,
    "std_floor": 0.05,
    "smoothing": 0.80,
}
_FIXED_ORACLE_CONFIG = {
    "population": 512,
    "elite_fraction": 0.10,
    "min_iterations": 20,
    "max_iterations": 40,
    "patience": 6,
    "initial_std": 0.25,
    "std_floor": 0.03,
    "smoothing": 0.75,
}


class OraclePublicationError(RuntimeError):
    """Raised when raw oracle evidence is unsafe or inconsistent to publish."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=OraclePublicationError)


def _relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError as exc:
        raise OraclePublicationError(
            f"tracked output must be inside the repository: {path}"
        ) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OraclePublicationError(message)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _validate_fixed_dependencies() -> None:
    fixed_hashes = {
        "historical receipt": (
            HISTORICAL_RECEIPT_SHA256,
            _FIXED_HISTORICAL_RECEIPT_SHA256,
        ),
        "shipped parity receipt": (
            SHIPPED_PARITY_SHA256,
            _FIXED_SHIPPED_PARITY_SHA256,
        ),
        "shipped parity rows": (
            SHIPPED_PARITY_ROWS_SHA256,
            _FIXED_SHIPPED_PARITY_ROWS_SHA256,
        ),
        "shipped ONNX": (SHIPPED_ONNX_SHA256, _FIXED_SHIPPED_ONNX_SHA256),
    }
    for label, (runner_value, fixed_value) in fixed_hashes.items():
        _require(runner_value == fixed_value, f"fixed {label} hash changed")
    _require(
        asdict(HISTORICAL_MPC_CONFIG) == _FIXED_MPC_CONFIG,
        "fixed historical MPC budget changed",
    )
    _require(
        asdict(HISTORICAL_ORACLE_CONFIG) == _FIXED_ORACLE_CONFIG,
        "fixed historical oracle budget changed",
    )


def _result(value: Any, label: str) -> PlannerResult:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(isinstance(value.get("solved"), bool), f"{label}.solved must be boolean")
    hard = value.get("hard_violation_count")
    residual = value.get("maximum_conservation_residual")
    _require(
        isinstance(hard, int) and not isinstance(hard, bool) and hard == 0,
        f"{label} hard violations must be zero",
    )
    _require(
        isinstance(residual, (int, float))
        and not isinstance(residual, bool)
        and math.isfinite(float(residual))
        and float(residual) == 0.0,
        f"{label} conservation residual must be exactly 0.0",
    )
    for field in ("minimum_tail_margin", "resilience_auc"):
        number = value.get(field)
        _require(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number)),
            f"{label}.{field} must be finite",
        )
    reasons = value.get("reason_codes")
    _require(
        isinstance(reasons, list) and all(isinstance(item, str) for item in reasons),
        f"{label}.reason_codes must be a string list",
    )
    for field in ("action_sequence_sha256", "trajectory_sha256"):
        _require_sha256(value.get(field), f"{label}.{field}")
    return _result_from_receipt(value)


def _validate_contract(protocol: dict[str, Any]) -> tuple[dict[str, Any], str]:
    _validate_fixed_dependencies()
    contract = protocol.get("contract")
    _require(isinstance(contract, dict), "protocol contract must be an object")
    contract_sha256 = _require_sha256(
        protocol.get("contract_sha256"), "protocol contract_sha256"
    )
    _require(
        canonical_hash(contract) == contract_sha256,
        "protocol contract canonical hash mismatch",
    )
    commit = contract.get("git_commit")
    _require(
        isinstance(commit, str) and _HEX_GIT_COMMIT.fullmatch(commit) is not None,
        "protocol git_commit must be a lowercase 40-character digest",
    )
    expected = _study_contract(("dev", "final"))
    # Source bytes, not the later publication commit, determine reproducibility.
    # Accept the raw runner's recorded commit while requiring every other field.
    expected["git_commit"] = commit
    _require(contract == expected, "raw study contract differs from the fixed protocol")
    _require(
        contract.get("source_identity") == _source_identity(),
        "raw study source identity differs from current source bytes",
    )
    _require(
        file_sha256(HISTORICAL_RECEIPT) == HISTORICAL_RECEIPT_SHA256,
        "historical 37/40 receipt changed",
    )
    return contract, contract_sha256


def _validate_mpc_selection(selection: Any, split: str) -> int:
    _require(isinstance(selection, dict), f"{split} MPC selection must be an object")
    aggregates = selection.get("aggregates")
    _require(
        isinstance(aggregates, dict)
        and set(aggregates) == {str(value) for value in MPC_HORIZONS},
        f"{split} MPC selection must retain k=1/3/5 aggregates",
    )
    for horizon, aggregate in aggregates.items():
        _require(isinstance(aggregate, dict), f"{split} MPC k={horizon} aggregate invalid")
        _require(
            aggregate.get("case_count") == CANONICAL_CASE_COUNT,
            f"{split} MPC k={horizon} case count drifted",
        )
        _require(
            aggregate.get("hard_violation_count") == 0,
            f"{split} MPC k={horizon} hard violations must be zero",
        )
        _require(
            aggregate.get("maximum_conservation_residual") == 0.0,
            f"{split} MPC k={horizon} conservation residual must be 0.0",
        )
    expected = max(
        MPC_HORIZONS,
        key=lambda horizon: (
            int(aggregates[str(horizon)]["solved_count"]),
            float(aggregates[str(horizon)]["mean_minimum_tail_margin"]),
            float(aggregates[str(horizon)]["mean_resilience_auc"]),
            -horizon,
        ),
    )
    _require(
        selection.get("selected_horizon") == expected,
        f"{split} selected MPC horizon violates its registered rule",
    )
    return expected


def _validate_budget(
    value: Any,
    *,
    split: str,
    row_id: str,
    horizon_days: int,
) -> tuple[int, int]:
    label = f"{split} {row_id} oracle budget"
    _require(isinstance(value, dict), f"{label} must be an object")
    config = HISTORICAL_ORACLE_CONFIG
    expected_fixed = {
        "population": config.population,
        "elite_fraction": config.elite_fraction,
        "elite_count": math.ceil(config.population * config.elite_fraction),
        "initial_std": config.initial_std,
        "std_floor": config.std_floor,
        "smoothing": config.smoothing,
        "minimum_iterations": config.min_iterations,
        "maximum_iterations": config.max_iterations,
        "patience": config.patience,
    }
    for key, expected in expected_fixed.items():
        _require(value.get(key) == expected, f"{label}.{key} changed")
    iterations = value.get("iterations")
    _require(
        isinstance(iterations, int)
        and not isinstance(iterations, bool)
        and config.min_iterations <= iterations <= config.max_iterations,
        f"{label}.iterations is outside the registered range",
    )
    candidate_evaluations = 2 + config.population * iterations
    simulated_transitions = candidate_evaluations * horizon_days
    _require(
        value.get("candidate_evaluations") == candidate_evaluations,
        f"{label}.candidate_evaluations is inconsistent",
    )
    _require(
        value.get("simulated_transitions") == simulated_transitions,
        f"{label}.simulated_transitions is inconsistent",
    )
    return candidate_evaluations, simulated_transitions


def _validate_aggregate(
    actual: Any,
    rows: Sequence[PlannerResult],
    label: str,
    *,
    include_wilson: bool = False,
) -> dict[str, Any]:
    expected = aggregate_results(rows)
    _require(isinstance(actual, dict), f"{label} aggregate must be an object")
    if include_wilson:
        expected["wilson_95"] = wilson_interval(
            int(expected["solved_count"]), CANONICAL_CASE_COUNT, digits=10
        )
    exact_fields = (
        "case_count",
        "solved_count",
        "solve_rate",
        "hard_violation_count",
        "maximum_conservation_residual",
        "failure_reason_code_histogram",
    )
    for field in exact_fields:
        _require(
            actual.get(field) == expected[field],
            f"{label} aggregate field does not recompute: {field}",
        )
    # Raw rows contain 10-decimal receipt values while the runner aggregates
    # replay values before serializing each row. Permit only that rounding gap.
    for field in (
        "minimum_tail_margin",
        "mean_minimum_tail_margin",
        "mean_resilience_auc",
    ):
        _require(
            isinstance(actual.get(field), (int, float))
            and abs(float(actual[field]) - float(expected[field])) <= 1e-9,
            f"{label} aggregate field does not recompute: {field}",
        )
    if include_wilson:
        _require(
            actual.get("wilson_95") == expected["wilson_95"],
            f"{label} Wilson interval does not recompute",
        )
    return actual


def _validate_split_receipt(
    *,
    split: str,
    receipt: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    external_root: Path,
) -> dict[str, Any]:
    prefix = f"raw {split} receipt"
    expected_flags = {
        "schema_version": RAW_SCHEMA_VERSION,
        "tool": RAW_TOOL_ID,
        "status": "complete_privileged_clairvoyant_oracle_diagnostic",
        "authorizing": False,
        "model_selection_used": False,
        "submission_baseline": False,
        "privileged_diagnostic": True,
        "split": split,
        "case_count": CANONICAL_CASE_COUNT,
        "final_split_used": split == "final",
        "learned_v4_policy_evaluated": False,
        "learned_v4_final_evaluated": False,
        "full_future_shock_tape_visible": True,
        "study_contract_sha256": contract_sha256,
    }
    for key, expected in expected_flags.items():
        _require(receipt.get(key) == expected, f"{prefix}.{key} changed")
    interpretation = receipt.get("interpretation")
    _require(isinstance(interpretation, dict), f"{prefix} interpretation missing")
    for key, expected in {
        "anytime_achieved_lower_bound": True,
        "mathematical_optimum_claimed": False,
        "infeasibility_certificate_claimed": False,
    }.items():
        _require(
            interpretation.get(key) == expected,
            f"{prefix} interpretation.{key} changed",
        )
    _require(
        receipt.get("split_contract") == contract["split_contracts"][split],
        f"{prefix} split contract mismatch",
    )
    _require(
        receipt.get("source_identity") == contract["source_identity"],
        f"{prefix} source identity mismatch",
    )
    _require(
        receipt.get("historical_reference") == contract["historical_reference"],
        f"{prefix} historical reference mismatch",
    )
    mpc = receipt.get("mpc_warm_start")
    _require(isinstance(mpc, dict), f"{prefix} MPC warm start missing")
    _require(
        mpc.get("config") == asdict(HISTORICAL_MPC_CONFIG),
        f"{prefix} MPC config changed",
    )
    selected_horizon = _validate_mpc_selection(mpc.get("selection"), split)
    oracle_section = receipt.get("oracle")
    _require(isinstance(oracle_section, dict), f"{prefix} oracle section missing")
    _require(
        oracle_section.get("config") == asdict(HISTORICAL_ORACLE_CONFIG),
        f"{prefix} oracle config changed",
    )
    _require(
        oracle_section.get("objective") == contract["objective"],
        f"{prefix} oracle objective changed",
    )
    _require(
        oracle_section.get("rng_namespace") == contract["rng_namespace"],
        f"{prefix} oracle RNG namespace changed",
    )

    cases = build_cases(split)
    rows = receipt.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == CANONICAL_CASE_COUNT,
        f"{prefix} must contain exactly 200 rows",
    )
    _require(
        len({row.get("row_id") for row in rows if isinstance(row, dict)})
        == CANONICAL_CASE_COUNT,
        f"{prefix} row IDs must be unique",
    )
    tuned_rows: list[PlannerResult] = []
    mpc_rows: list[PlannerResult] = []
    oracle_rows: list[PlannerResult] = []
    candidate_total = 0
    transition_total = 0
    mpc_key = f"selected_mpc_k{selected_horizon}"
    for index, (case, row) in enumerate(zip(cases, rows, strict=True)):
        _require(isinstance(row, dict), f"{prefix} row {index} must be an object")
        expected_case = _case_contract(case)
        for key, expected in expected_case.items():
            _require(
                row.get(key) == expected,
                f"{prefix} row {index} ordered case/tape mismatch: {key}",
            )
        tuned = _result(row.get("tuned_rule"), f"{split} {case.row_id} tuned rule")
        selected_mpc = _result(row.get(mpc_key), f"{split} {case.row_id} {mpc_key}")
        oracle = _result(
            row.get("clairvoyant_oracle_cem"),
            f"{split} {case.row_id} clairvoyant oracle",
        )
        _require(
            int(oracle.solved) >= max(int(tuned.solved), int(selected_mpc.solved)),
            f"{split} {case.row_id} oracle lost a warm-start solve",
        )
        search = row.get("oracle_search_wide_invariants")
        _require(isinstance(search, dict), f"{split} {case.row_id} search invariants missing")
        _require(
            search.get("maximum_hard_violation_count") == 0,
            f"{split} {case.row_id} oracle search hard violations must be zero",
        )
        _require(
            search.get("maximum_conservation_residual") == 0.0,
            f"{split} {case.row_id} oracle search residual must be 0.0",
        )
        warm_start = row.get("oracle_warm_start")
        _require(
            isinstance(warm_start, dict)
            and warm_start.get("winner_not_worse_than_tuned_or_mpc") is True,
            f"{split} {case.row_id} oracle warm-start invariant missing",
        )
        candidates, transitions = _validate_budget(
            row.get("oracle_budget"),
            split=split,
            row_id=case.row_id,
            horizon_days=int(case.scenario.horizon_days),
        )
        candidate_total += candidates
        transition_total += transitions
        tuned_rows.append(tuned)
        mpc_rows.append(selected_mpc)
        oracle_rows.append(oracle)

    _require(
        receipt.get("rows_sha256") == canonical_hash(rows),
        f"{prefix} rows_sha256 mismatch",
    )
    planner_aggregates = receipt.get("planner_aggregates")
    _require(isinstance(planner_aggregates, dict), f"{prefix} planner aggregates missing")
    _validate_aggregate(
        planner_aggregates.get("tuned_rule"), tuned_rows, f"{split} tuned rule"
    )
    _validate_aggregate(
        planner_aggregates.get(mpc_key), mpc_rows, f"{split} selected MPC"
    )
    oracle_aggregate = _validate_aggregate(
        planner_aggregates.get("clairvoyant_oracle_cem"),
        oracle_rows,
        f"{split} clairvoyant oracle",
        include_wilson=True,
    )
    _require(
        oracle_section.get("aggregate") == oracle_aggregate,
        f"{prefix} oracle aggregate mismatch",
    )
    _require(
        oracle_section.get("total_candidate_evaluations") == candidate_total,
        f"{prefix} oracle candidate total mismatch",
    )
    _require(
        oracle_section.get("total_simulated_transitions") == transition_total,
        f"{prefix} oracle transition total mismatch",
    )
    invariants = receipt.get("invariants")
    _require(isinstance(invariants, dict), f"{prefix} invariants missing")
    required_invariants = (
        "case_count_exactly_200",
        "row_ids_unique",
        "all_planner_hard_violation_counts_zero",
        "all_planner_conservation_residuals_exactly_zero",
        "all_oracle_search_hard_violation_counts_zero",
        "all_oracle_search_conservation_residuals_exactly_zero",
        "production_decoder_and_exact_projection_used",
    )
    _require(
        all(invariants.get(key) is True for key in required_invariants)
        and invariants.get("learned_policy_loaded_or_run") is False,
        f"{prefix} receipt invariants are incomplete",
    )
    runtime = receipt.get("runtime")
    _require(isinstance(runtime, dict), f"{prefix} runtime missing")
    raw_output_root = runtime.get("output_root")
    _require(
        isinstance(raw_output_root, str)
        and Path(raw_output_root).resolve() == external_root,
        f"{prefix} runtime output root does not match the supplied study root",
    )
    if split == "dev":
        expected_comparison = build_reference_comparison(
            cases=cases,
            oracle_rows=oracle_rows,
            reference_path=(
                ROOT
                / "internal"
                / "developmental_runs"
                / "v4"
                / "city_recovery_ppo.v4.parity.json"
            ),
        )
        _require(
            receipt.get("development_shipped_policy_comparison")
            == expected_comparison,
            "development shipped-policy pairing does not recompute",
        )
        reference = expected_comparison["reference"]
        _require(
            reference
            == {
                "policy": "shipped city_recovery_ppo.v4.onnx",
                "policy_evaluated_in_this_run": False,
                "parity_receipt_sha256": SHIPPED_PARITY_SHA256,
                "parity_rows_sha256": SHIPPED_PARITY_ROWS_SHA256,
                "onnx_sha256": SHIPPED_ONNX_SHA256,
            },
            "development shipped-policy fixed evidence hashes changed",
        )
    else:
        _require(
            receipt.get("development_shipped_policy_comparison") is None,
            "final receipt must not contain a learned-policy comparison",
        )
    return oracle_aggregate


def _validate_summary(
    *,
    summary: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    receipt_hashes: dict[str, str],
    contract_sha256: str,
) -> None:
    _require(summary.get("schema_version") == RAW_SCHEMA_VERSION, "summary schema changed")
    _require(summary.get("tool") == RAW_TOOL_ID, "summary producer changed")
    _require(summary.get("status") == "complete", "raw study is not complete")
    _require(
        summary.get("contract_sha256") == contract_sha256,
        "summary contract hash mismatch",
    )
    _require(
        summary.get("learned_v4_final_evaluated") is False,
        "summary indicates that the learned v4 final evaluation ran",
    )
    _require(
        summary.get("historical_37_of_40_preserved") is True,
        "summary does not preserve historical 37/40 evidence",
    )
    split_summaries = summary.get("splits")
    _require(
        isinstance(split_summaries, dict) and set(split_summaries) == {"dev", "final"},
        "summary must contain exactly dev and final",
    )
    for split in ("dev", "final"):
        split_summary = split_summaries[split]
        aggregate = receipts[split]["oracle"]["aggregate"]
        expected = {
            "receipt": f"{split}/receipt.json",
            "receipt_sha256": receipt_hashes[split],
            "solved_count": aggregate["solved_count"],
            "solve_rate": aggregate["solve_rate"],
            "wilson_95": aggregate["wilson_95"],
            "learned_v4_policy_evaluated": False,
        }
        _require(split_summary == expected, f"summary {split} evidence mismatch")


def _portable_receipt(
    *,
    split: str,
    raw_receipt: dict[str, Any],
    raw_receipt_sha256: str,
    raw_summary_sha256: str,
    raw_protocol_sha256: str,
    tracked_path: Path,
    published_at_utc: str,
) -> dict[str, Any]:
    portable = copy.deepcopy(raw_receipt)
    runtime = portable.get("runtime")
    _require(isinstance(runtime, dict), f"{split} runtime missing during publication")
    runtime["output_root"] = "distribution:not_in_repository"
    runtime["external_output_root_omitted"] = True
    portable["kind"] = PORTABLE_KIND
    portable["publication"] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "created_at_utc": published_at_utc,
        "tracked_path": _relative_repo_path(tracked_path),
        "raw_evidence": {
            "receipt": f"{split}/receipt.json",
            "receipt_sha256": raw_receipt_sha256,
            "summary": "summary.json",
            "summary_sha256": raw_summary_sha256,
            "protocol": "protocol.json",
            "protocol_sha256": raw_protocol_sha256,
            "external_output_root": "distribution:not_in_repository",
        },
        "portable": True,
        "raw_per_case_rows_retained": True,
        "raw_receipt_byte_hash_bound": True,
    }
    _reject_absolute_paths(portable)
    return portable


def _reject_absolute_paths(value: Any, path: str = "receipt") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(
            _ABSOLUTE_PATH_FRAGMENT.search(value) is None,
            f"portable evidence contains an absolute path at {path}",
        )


def _format_ci(interval: Sequence[float]) -> str:
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def _render_markdown(
    *,
    portable: dict[str, dict[str, Any]],
    raw_receipt_hashes: dict[str, str],
    raw_summary_sha256: str,
    raw_protocol_sha256: str,
) -> str:
    dev = portable["dev"]
    final = portable["final"]
    dev_aggregate = dev["oracle"]["aggregate"]
    final_aggregate = final["oracle"]["aggregate"]
    comparison = dev["development_shipped_policy_comparison"]
    pairing = comparison["pairing"]
    dev_mpc_key = next(
        key for key in dev["planner_aggregates"] if key.startswith("selected_mpc_k")
    )
    final_mpc_key = next(
        key for key in final["planner_aggregates"] if key.startswith("selected_mpc_k")
    )
    lines = [
        "# Matched 200-case privileged clairvoyant-oracle study",
        "",
        (
            "The CEM oracle sees each case's complete future shock tape. It is a "
            "privileged anytime achieved lower bound, **not** a submission baseline, "
            "a proven mathematical optimum, or an infeasibility certificate."
        ),
        "",
        "## Matched split results",
        "",
        "| Split | Oracle solved | Wilson 95% CI | Tuned rule | Selected MPC | Learned v4 policy |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Development | **{dev_aggregate['solved_count']}/200** | "
            f"{_format_ci(dev_aggregate['wilson_95'])} | "
            f"{dev['planner_aggregates']['tuned_rule']['solved_count']}/200 | "
            f"{dev['planner_aggregates'][dev_mpc_key]['solved_count']}/200 | "
            f"178/200 (accepted shipped-policy receipt) |"
        ),
        (
            f"| Final | **{final_aggregate['solved_count']}/200** | "
            f"{_format_ci(final_aggregate['wilson_95'])} | "
            f"{final['planner_aggregates']['tuned_rule']['solved_count']}/200 | "
            f"{final['planner_aggregates'][final_mpc_key]['solved_count']}/200 | "
            "**not evaluated** |"
        ),
        "",
        (
            "The learned v4 model was not run on the final split. The final row is "
            "oracle/planner diagnostic evidence only."
        ),
        "",
        "## Development casewise comparison",
        "",
        (
            "The comparison joins the oracle rows to the already accepted shipped-ONNX "
            "development parity receipt; it does not rerun the model."
        ),
        "",
        "| Both solve | Policy only | Oracle only | Neither | Known-feasible union |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {pairing['both']} | {pairing['policy_only']} | "
            f"**{pairing['oracle_only']}** | {pairing['neither']} | "
            f"{comparison['known_feasible_union_count']}/200 |"
        ),
        "",
        (
            f"Remaining directly demonstrated headroom is **{pairing['oracle_only']} "
            "cases**: the oracle solves those identical tapes while the shipped policy "
            "does not. Policy-only and oracle-only cases are reported separately, so "
            "an aggregate ratio is not presented as casewise ceiling coverage."
        ),
        "",
        "## Evidence and safety",
        "",
        (
            "Every tuned-rule, selected-MPC, and oracle rollout has zero hard "
            "violations and exactly `0.0` maximum conservation residual. The same "
            "holds across all evaluated oracle candidates."
        ),
        "",
        (
            "The historical **37/40** result remains intact as the original 40-case "
            "development-subset diagnostic. This study neither overwrites nor "
            "reinterprets that receipt."
        ),
        "",
        "| Evidence | SHA-256 |",
        "|---|---|",
        (
            "| [Portable development receipt](../../internal/developmental_runs/v4/"
            "clairvoyant-oracle-200-dev.json) raw receipt | "
            f"`{raw_receipt_hashes['dev']}` |"
        ),
        (
            "| [Portable final receipt](../../internal/developmental_runs/v4/"
            "clairvoyant-oracle-200-final.json) raw receipt | "
            f"`{raw_receipt_hashes['final']}` |"
        ),
        f"| Raw study summary | `{raw_summary_sha256}` |",
        f"| Raw study protocol | `{raw_protocol_sha256}` |",
        f"| Historical 37/40 receipt | `{HISTORICAL_RECEIPT_SHA256}` |",
        f"| Shipped policy parity receipt | `{SHIPPED_PARITY_SHA256}` |",
        f"| Shipped policy parity rows | `{SHIPPED_PARITY_ROWS_SHA256}` |",
        f"| Shipped v4 ONNX | `{SHIPPED_ONNX_SHA256}` |",
        "",
    ]
    markdown = "\n".join(lines)
    _reject_absolute_paths(markdown, "markdown")
    return markdown


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise OraclePublicationError(f"refusing to overwrite existing evidence: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_parent(path)
    except FileExistsError as exc:
        raise OraclePublicationError(
            f"refusing to overwrite existing evidence: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def publish_study(
    external_root: Path,
    *,
    dev_output: Path = TRACKED_RECEIPTS["dev"],
    final_output: Path = TRACKED_RECEIPTS["final"],
    markdown_output: Path = TRACKED_MARKDOWN,
) -> dict[str, Any]:
    """Validate a completed raw study and create portable evidence outputs."""

    _require(external_root.is_absolute(), "external study root must be absolute")
    root = external_root.resolve()
    try:
        root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise OraclePublicationError("external study root must be outside the repository")
    _require(root.is_dir(), "external study root is missing")
    destinations = (dev_output, final_output, markdown_output)
    _require(len({path.resolve() for path in destinations}) == 3, "output paths must differ")
    for path in destinations:
        _require(
            not path.exists(),
            f"refusing to overwrite existing evidence: {path}",
        )

    protocol_path = root / "protocol.json"
    summary_path = root / "summary.json"
    receipt_paths = {
        split: root / split / "receipt.json" for split in ("dev", "final")
    }
    protocol = _load_json(protocol_path, "raw oracle protocol")
    summary = _load_json(summary_path, "raw oracle summary")
    receipts = {
        split: _load_json(path, f"raw {split} oracle receipt")
        for split, path in receipt_paths.items()
    }
    contract, contract_sha256 = _validate_contract(protocol)
    receipt_hashes = {
        split: file_sha256(path) for split, path in receipt_paths.items()
    }
    summary_sha256 = file_sha256(summary_path)
    protocol_sha256 = file_sha256(protocol_path)
    for split in ("dev", "final"):
        _validate_split_receipt(
            split=split,
            receipt=receipts[split],
            contract=contract,
            contract_sha256=contract_sha256,
            external_root=root,
        )
    _validate_summary(
        summary=summary,
        receipts=receipts,
        receipt_hashes=receipt_hashes,
        contract_sha256=contract_sha256,
    )

    published_at = _utc_now()
    portable = {
        "dev": _portable_receipt(
            split="dev",
            raw_receipt=receipts["dev"],
            raw_receipt_sha256=receipt_hashes["dev"],
            raw_summary_sha256=summary_sha256,
            raw_protocol_sha256=protocol_sha256,
            tracked_path=TRACKED_RECEIPTS["dev"],
            published_at_utc=published_at,
        ),
        "final": _portable_receipt(
            split="final",
            raw_receipt=receipts["final"],
            raw_receipt_sha256=receipt_hashes["final"],
            raw_summary_sha256=summary_sha256,
            raw_protocol_sha256=protocol_sha256,
            tracked_path=TRACKED_RECEIPTS["final"],
            published_at_utc=published_at,
        ),
    }
    markdown = _render_markdown(
        portable=portable,
        raw_receipt_hashes=receipt_hashes,
        raw_summary_sha256=summary_sha256,
        raw_protocol_sha256=protocol_sha256,
    )
    _atomic_create(dev_output, _json_bytes(portable["dev"]))
    _atomic_create(final_output, _json_bytes(portable["final"]))
    _atomic_create(markdown_output, markdown.encode("utf-8"))
    return {
        "status": "published",
        "outputs": {
            "dev": {
                "path": _relative_repo_path(TRACKED_RECEIPTS["dev"]),
                "raw_receipt_sha256": receipt_hashes["dev"],
                "solved_count": portable["dev"]["oracle"]["aggregate"][
                    "solved_count"
                ],
            },
            "final": {
                "path": _relative_repo_path(TRACKED_RECEIPTS["final"]),
                "raw_receipt_sha256": receipt_hashes["final"],
                "solved_count": portable["final"]["oracle"]["aggregate"][
                    "solved_count"
                ],
            },
            "markdown": _relative_repo_path(TRACKED_MARKDOWN),
        },
        "learned_v4_final_evaluated": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = publish_study(args.external_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OraclePublicationError as error:
        print(f"oracle publication failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
