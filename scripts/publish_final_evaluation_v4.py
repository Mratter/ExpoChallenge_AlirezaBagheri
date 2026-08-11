#!/usr/bin/env python3
"""Claim, run, and publish the one authorized v4 ONNX final evaluation.

The repeatable preflight never imports or builds the reserved split. Execution
atomically consumes a fixed claim before lazily importing the existing evaluator.
After that point, success or failure is terminal and a retry is impossible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

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

TOOL_ID = "publish_final_evaluation_v4.py"
SCHEMA_VERSION = 1
FINAL_SPLIT_ID = "final"
EXPECTED_CASE_COUNT = 200
EXPECTED_SOLVED_COUNT = 163
EXPECTED_SOLVE_RATE = EXPECTED_SOLVED_COUNT / EXPECTED_CASE_COUNT
EXPECTED_WILSON_95 = [0.7554293724, 0.862698072]
EXPECTED_ARTIFACT_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)
ARTIFACT = ROOT / "artifacts" / "city_recovery_ppo.v4.onnx"
ARTIFACT_MANIFEST = ROOT / "artifacts" / "city_recovery_ppo.v4.manifest.json"
ARTIFACT_MANIFEST_SHA256 = (
    "7ecc9948789163febf9cc9a455e20c0d5e5fb75c70919598169f21614e1a5a06"
)
DEV_PARITY_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "city_recovery_ppo.v4.parity.json"
)
DEV_PARITY_RECEIPT_SHA256 = (
    "e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c"
)

AUTHORIZATION = {
    "id": "owner-v4-final-200-2026-08-12",
    "authorized_by": "repository owner",
    "authorized_on": "2026-08-12",
    "one_time": True,
    "scope": (
        "exactly one shipped v4 ONNX policy on the canonical 200-case final "
        "roster, with exactly one rollout per case"
    ),
}
EXPECTED_FINAL_SPLIT_CONTRACT = {
    "id": "final",
    "family_count": 5,
    "family_ids": [
        "v3_final_coastal_isolation",
        "v3_final_grid_cascade",
        "v3_final_food_access",
        "v3_final_aftershock_corridor",
        "v3_final_public_health",
    ],
    "seed_interval": {"first": 830000, "last": 830039, "count": 40},
    "cartesian_case_count": 200,
    "iteration_order": "family_order_then_ascending_seed",
}
CORE_SOURCE_SHA256 = {
    "backend/app/city/environment.py": (
        "52ddc80249d3d5ad1853a0f20b60e05258d5ba692ee1d9edaa2f75edc499ab1e"
    ),
    "backend/app/city/scenarios.py": (
        "a28c4e998f7387b02df568605dea558fcbf62aa7a9d93f30ef86707acc1f8677"
    ),
    "backend/app/city/outcome.py": (
        "4ef91abcab7f12363b0eac403aa188ce7c759d88a07f5ece968a026b48a9e032"
    ),
    "model/policy.py": (
        "81eced960627e68e14eff9f1213ad8196b6abfecac2094486bbae2b4b4a4cad5"
    ),
    "scripts/evaluate.py": (
        "60e2ecbd315c3956b0ac1a098e53fd53f675415569e9e30f879bd76c08cfe3da"
    ),
}

EVALUATION_ROOT = ROOT / "internal" / "evaluation_runs" / "v4"
CLAIM_PATH = EVALUATION_ROOT / "final-evaluation-200.claim.json"
SUCCESS_PATH = EVALUATION_ROOT / "final-evaluation-200.success.json"
FAILURE_PATH = EVALUATION_ROOT / "final-evaluation-200.failure.json"
MARKDOWN_PATH = ROOT / "benchmarks" / "v4" / "final-results-200.md"

ORACLE_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "clairvoyant-oracle-200-final.json"
)
ORACLE_RECEIPT_SHA256 = (
    "baf5aa6ec8e419a50f87e744eac7779f30a53b6aab60018ff1a7043126b0b5ec"
)
REGRESSION_GATE = ROOT / "tests" / "test_consolidation_gate.py"
REGRESSION_GATE_SHA256 = (
    "97bdeb13556a2fdb9b291c62e699da739441e593ad57f6a5adc014e7ece38638"
)
LEGACY_FIXTURE = ROOT / "tests" / "fixtures" / "legacy_policy.onnx"
LEGACY_FIXTURE_SHA256 = (
    "6a08ae284fb93cff1155ce37dcec4fac1121697add0fabd9d367486be344bf0b"
)

REFERENCE_RESULTS = (
    ("Privileged clairvoyant CEM", 182, "privileged anytime achieved lower bound"),
    ("Shipped v4 PPO", EXPECTED_SOLVED_COUNT, "owner-authorized learned policy"),
    ("Tuned constant rule", 147, "deterministic oracle warm start"),
    ("Preparedness teacher", 139, "public deterministic regression"),
    ("Selected causal MPC, k=5", 135, "causal receding-horizon diagnostic"),
    ("Legacy ONNX regression fixture", 125, "retired-policy regression fixture"),
    ("Reactive heuristic", 72, "public deterministic regression"),
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROW_FIELDS = {
    "row_id",
    "family_id",
    "case_seed",
    "tape_seed",
    "tape_sha256",
    "solved",
    "status",
    "reason_codes",
    "resilience_auc",
    "minimum_tail_margin",
    "critical_service_days",
    "hard_violation_count",
    "max_conservation_residual",
    "trajectory_sha256",
}
_AGGREGATE_FIELDS = {
    "case_count",
    "solved_count",
    "solve_rate",
    "mean_resilience_auc",
    "mean_minimum_tail_margin",
    "hard_violation_count",
    "maximum_conservation_residual",
    "failure_reason_code_histogram",
}


@dataclass(frozen=True)
class PublicationPaths:
    claim: Path
    success: Path
    failure: Path
    markdown: Path


DEFAULT_PATHS = PublicationPaths(CLAIM_PATH, SUCCESS_PATH, FAILURE_PATH, MARKDOWN_PATH)

GitProbe = Callable[[], Mapping[str, Any]]
EvidenceProbe = Callable[[], Mapping[str, Any]]
ReferenceLoader = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
EvaluationRunner = Callable[[Path], Mapping[str, Any]]


class FinalEvaluationError(RuntimeError):
    """Raised when the irreversible single-use contract cannot be satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalEvaluationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise FinalEvaluationError(f"path must be inside the repository: {path}") from exc


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _policy_label() -> str:
    return f"onnx:{_repo_path(ARTIFACT)}"


def _run_git(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _probe_git() -> Mapping[str, Any]:
    commit = _run_git(("rev-parse", "HEAD"))
    status = _run_git(("status", "--porcelain=v1", "--untracked-files=all"))
    _require(_GIT_COMMIT.fullmatch(commit) is not None, "HEAD is not a commit")
    _require(status == "", "worktree must be completely clean before the claim")
    return {"commit": commit, "clean": True}


def _probe_evidence_hashes() -> Mapping[str, Any]:
    values = {
        "artifact": file_sha256(ARTIFACT),
        "artifact_manifest": file_sha256(ARTIFACT_MANIFEST),
        "dev_parity_receipt": file_sha256(DEV_PARITY_RECEIPT),
        "oracle_receipt": file_sha256(ORACLE_RECEIPT),
        "regression_gate": file_sha256(REGRESSION_GATE),
        "legacy_fixture": file_sha256(LEGACY_FIXTURE),
        "core_sources": {
            path: file_sha256(ROOT / path) for path in CORE_SOURCE_SHA256
        },
    }
    _require(values["artifact"] == EXPECTED_ARTIFACT_SHA256, "artifact hash changed")
    _require(
        values["artifact_manifest"] == ARTIFACT_MANIFEST_SHA256,
        "artifact manifest hash changed",
    )
    _require(
        values["dev_parity_receipt"] == DEV_PARITY_RECEIPT_SHA256,
        "development parity receipt hash changed",
    )
    _require(
        values["oracle_receipt"] == ORACLE_RECEIPT_SHA256,
        "oracle receipt hash changed",
    )
    _require(
        values["regression_gate"] == REGRESSION_GATE_SHA256,
        "regression gate hash changed",
    )
    _require(
        values["legacy_fixture"] == LEGACY_FIXTURE_SHA256,
        "legacy fixture hash changed",
    )
    _require(values["core_sources"] == CORE_SOURCE_SHA256, "core source hash changed")
    return values


def preflight(
    *,
    paths: PublicationPaths = DEFAULT_PATHS,
    git_probe: GitProbe = _probe_git,
    evidence_probe: EvidenceProbe = _probe_evidence_hashes,
) -> dict[str, Any]:
    """Repeatable no-final-access gate; it performs no imports or writes."""

    for label, path in (
        ("claim", paths.claim),
        ("success", paths.success),
        ("failure", paths.failure),
        ("Markdown", paths.markdown),
    ):
        _require(
            not path.exists() and not path.is_symlink(),
            f"{label} output already exists or is a symlink: {path}",
        )
    git = dict(git_probe())
    _require(git.get("clean") is True, "git probe did not certify a clean worktree")
    _require(
        isinstance(git.get("commit"), str)
        and _GIT_COMMIT.fullmatch(git["commit"]) is not None,
        "git probe did not return a committed HEAD",
    )
    evidence = dict(evidence_probe())
    return {
        "status": "ready",
        "git": git,
        "evidence_sha256": evidence,
        "claim_path": _display_path(paths.claim),
        "success_path": _display_path(paths.success),
        "failure_path": _display_path(paths.failure),
        "markdown_path": _display_path(paths.markdown),
        "reserved_split_imported_or_built": False,
        "filesystem_written": False,
    }


def _load_reference_evidence() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    oracle = load_json_object(
        ORACLE_RECEIPT,
        "privileged-oracle final receipt",
        expected_sha256=ORACLE_RECEIPT_SHA256,
        error_type=FinalEvaluationError,
    )
    aggregates = oracle.get("planner_aggregates")
    invariants = oracle.get("invariants")
    _require(
        oracle.get("split") == FINAL_SPLIT_ID
        and oracle.get("case_count") == EXPECTED_CASE_COUNT
        and oracle.get("split_contract") == EXPECTED_FINAL_SPLIT_CONTRACT
        and oracle.get("final_split_used") is True
        and oracle.get("learned_v4_policy_evaluated") is False
        and isinstance(aggregates, Mapping)
        and isinstance(invariants, Mapping),
        "oracle receipt scope changed",
    )
    _require(
        oracle.get("oracle", {}).get("aggregate", {}).get("solved_count") == 182,
        "oracle solved count changed",
    )
    _require(
        aggregates.get("tuned_rule", {}).get("solved_count") == 147
        and aggregates.get("selected_mpc_k5", {}).get("solved_count") == 135,
        "oracle warm-start aggregates changed",
    )
    _require(
        invariants.get("all_planner_hard_violation_counts_zero") is True
        and invariants.get("all_planner_conservation_residuals_exactly_zero") is True
        and invariants.get("all_oracle_search_hard_violation_counts_zero") is True
        and invariants.get("all_oracle_search_conservation_residuals_exactly_zero")
        is True,
        "oracle safety invariants changed",
    )
    binding = {
        "artifact_manifest": {
            "path": _repo_path(ARTIFACT_MANIFEST),
            "sha256": ARTIFACT_MANIFEST_SHA256,
        },
        "development_parity_receipt": {
            "path": _repo_path(DEV_PARITY_RECEIPT),
            "sha256": DEV_PARITY_RECEIPT_SHA256,
        },
        "oracle_receipt": {
            "path": _repo_path(ORACLE_RECEIPT),
            "sha256": ORACLE_RECEIPT_SHA256,
        },
        "regression_gate": {
            "path": _repo_path(REGRESSION_GATE),
            "sha256": REGRESSION_GATE_SHA256,
            "reactive_heuristic_solved_count": 72,
            "preparedness_teacher_solved_count": 139,
            "legacy_onnx_fixture_solved_count": 125,
        },
        "legacy_fixture": {
            "path": _repo_path(LEGACY_FIXTURE),
            "sha256": LEGACY_FIXTURE_SHA256,
        },
        "core_sources": dict(CORE_SOURCE_SHA256),
    }
    return oracle, binding


def _production_runner(artifact: Path) -> Mapping[str, Any]:
    """Build once and perform exactly one rollout on each of 200 cases."""

    _require(
        artifact.resolve() == ARTIFACT.resolve(),
        "production runner received a noncanonical artifact path",
    )
    _require(
        file_sha256(artifact) == EXPECTED_ARTIFACT_SHA256,
        "artifact hash changed after the irreversible claim",
    )
    evaluation = importlib.import_module("scripts.evaluate")
    policy = evaluation.resolve_policy(_policy_label())
    _require(getattr(policy, "kind", None) == "onnx", "resolved policy is not ONNX")
    cases = evaluation.build_cases(FINAL_SPLIT_ID)
    _require(len(cases) == EXPECTED_CASE_COUNT, "evaluator did not build 200 cases")
    probe_rows: list[Any] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        probe_row = evaluation.rollout(case, policy)
        probe_rows.append(probe_row)
        row = asdict(probe_row)
        row["tape_sha256"] = canonical_hash([asdict(shock) for shock in case.schedule])
        rows.append(row)
    return {
        "schema_version": 1,
        "tool": "evaluate",
        "authorizing": False,
        "split": FINAL_SPLIT_ID,
        "case_count": len(cases),
        "same_tapes": True,
        "policies": {_policy_label(): evaluation.aggregate(probe_rows)},
        "paired_comparisons": {},
        "rows": {_policy_label(): rows},
        "rollout_count": len(probe_rows),
        "artifact_path": str(artifact),
    }


def _json_row(value: Any, index: int) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    _require(isinstance(value, Mapping), f"row {index} must be an object")
    row = dict(value)
    _require(set(row) == _ROW_FIELDS, f"row {index} fields changed")
    for field in ("row_id", "family_id", "status"):
        _require(isinstance(row[field], str) and row[field], f"row {index} {field} invalid")
    for field in ("tape_sha256", "trajectory_sha256"):
        _require(
            isinstance(row[field], str) and _SHA256.fullmatch(row[field]) is not None,
            f"row {index} {field} invalid",
        )
    for field in ("case_seed", "tape_seed", "critical_service_days", "hard_violation_count"):
        _require(
            isinstance(row[field], int) and not isinstance(row[field], bool),
            f"row {index} {field} must be an integer",
        )
    _require(row["critical_service_days"] >= 0, f"row {index} critical days invalid")
    _require(isinstance(row["solved"], bool), f"row {index} solved invalid")
    reasons = row["reason_codes"]
    _require(
        isinstance(reasons, (list, tuple))
        and all(isinstance(reason, str) and reason for reason in reasons),
        f"row {index} reasons invalid",
    )
    row["reason_codes"] = list(reasons)
    for field in ("resilience_auc", "minimum_tail_margin", "max_conservation_residual"):
        _require(
            isinstance(row[field], (int, float))
            and not isinstance(row[field], bool)
            and math.isfinite(float(row[field])),
            f"row {index} {field} must be finite",
        )
        row[field] = float(row[field])
    _require(row["hard_violation_count"] == 0, f"row {index} has a hard violation")
    _require(
        row["max_conservation_residual"] == 0.0,
        f"row {index} has a conservation residual",
    )
    return row


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        reason
        for row in rows
        if not row["solved"]
        for reason in row["reason_codes"]
    )
    solved = sum(row["solved"] for row in rows)
    return {
        "case_count": len(rows),
        "solved_count": solved,
        "solve_rate": solved / len(rows),
        "mean_resilience_auc": round(fmean(row["resilience_auc"] for row in rows), 10),
        "mean_minimum_tail_margin": round(
            fmean(row["minimum_tail_margin"] for row in rows), 10
        ),
        "hard_violation_count": sum(row["hard_violation_count"] for row in rows),
        "maximum_conservation_residual": max(
            row["max_conservation_residual"] for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
    }


def _split_and_family_evidence(
    rows: Sequence[dict[str, Any]], split_contract: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        dict(split_contract) == EXPECTED_FINAL_SPLIT_CONTRACT,
        "final split contract changed",
    )
    family_ids = split_contract["family_ids"]
    seeds = split_contract["seed_interval"]
    per_family: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for family_index, family_id in enumerate(family_ids):
        start = family_index * seeds["count"]
        family_rows = rows[start : start + seeds["count"]]
        _require(len(family_rows) == seeds["count"], f"{family_id} row count changed")
        for seed_offset, row in enumerate(family_rows):
            expected_seed = seeds["first"] + seed_offset
            _require(row["family_id"] == family_id, f"{family_id} ordering changed")
            _require(row["case_seed"] == expected_seed, f"{family_id} seed ordering changed")
            identities.append(
                {
                    "row_id": row["row_id"],
                    "family_id": row["family_id"],
                    "case_seed": row["case_seed"],
                    "tape_seed": row["tape_seed"],
                    "tape_sha256": row["tape_sha256"],
                }
            )
        solved = sum(row["solved"] for row in family_rows)
        per_family.append(
            {
                "family_id": family_id,
                "case_count": len(family_rows),
                "solved_count": solved,
                "solve_rate": solved / len(family_rows),
                "wilson_95": wilson_interval(solved, len(family_rows), digits=10),
                "hard_violation_count": sum(
                    row["hard_violation_count"] for row in family_rows
                ),
                "maximum_conservation_residual": max(
                    row["max_conservation_residual"] for row in family_rows
                ),
            }
        )
    return {
        "split_contract": dict(split_contract),
        "split_contract_sha256": canonical_hash(split_contract),
        "ordered_split_identity": identities,
        "ordered_split_identity_sha256": canonical_hash(identities),
        "per_family": per_family,
    }


def _join_oracle(
    model_rows: Sequence[dict[str, Any]], oracle_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    oracle_rows = oracle_receipt.get("rows")
    _require(
        isinstance(oracle_rows, list) and len(oracle_rows) == EXPECTED_CASE_COUNT,
        "oracle receipt must contain 200 rows",
    )
    _require(
        oracle_receipt.get("rows_sha256") == canonical_hash(oracle_rows),
        "oracle receipt rows hash mismatch",
    )
    counts = {"both": 0, "policy_only": 0, "oracle_only": 0, "neither": 0}
    joined: list[dict[str, Any]] = []
    for index, (model, oracle) in enumerate(zip(model_rows, oracle_rows, strict=True)):
        _require(isinstance(oracle, Mapping), f"oracle row {index} invalid")
        for field in ("row_id", "tape_seed", "tape_sha256"):
            _require(
                model[field] == oracle.get(field),
                f"model/oracle ordered identity mismatch at row {index}: {field}",
            )
        oracle_result = oracle.get("clairvoyant_oracle_cem")
        search = oracle.get("oracle_search_wide_invariants")
        _require(isinstance(oracle_result, Mapping), f"oracle row {index} result missing")
        _require(
            oracle_result.get("hard_violation_count") == 0
            and oracle_result.get("maximum_conservation_residual") == 0.0
            and isinstance(search, Mapping)
            and search.get("maximum_hard_violation_count") == 0
            and search.get("maximum_conservation_residual") == 0.0,
            f"oracle row {index} safety invariant failed",
        )
        policy_solved = model["solved"]
        oracle_solved = bool(oracle_result.get("solved"))
        if policy_solved and oracle_solved:
            classification = "both"
        elif policy_solved:
            classification = "policy_only"
        elif oracle_solved:
            classification = "oracle_only"
        else:
            classification = "neither"
        counts[classification] += 1
        joined.append(
            {
                "row_id": model["row_id"],
                "tape_seed": model["tape_seed"],
                "tape_sha256": model["tape_sha256"],
                "policy_solved": policy_solved,
                "oracle_solved": oracle_solved,
                "classification": classification,
            }
        )
    _require(sum(counts.values()) == EXPECTED_CASE_COUNT, "pairing count mismatch")
    _require(counts["both"] + counts["policy_only"] == EXPECTED_SOLVED_COUNT, "policy pairing drift")
    _require(counts["both"] + counts["oracle_only"] == 182, "oracle pairing drift")
    union = counts["both"] + counts["policy_only"] + counts["oracle_only"]
    return {
        "pairing": counts,
        "known_feasible_union_count": union,
        "oracle_only_provable_headroom_count": counts["oracle_only"],
        "aggregate_count_ratio_policy_to_oracle_achieved": round(
            EXPECTED_SOLVED_COUNT / 182, 10
        ),
        "casewise_policy_coverage_of_oracle_achieved": round(counts["both"] / 182, 10),
        "interpretation": (
            "Aggregate count ratio and casewise coverage are distinct when the "
            "finite anytime oracle has any policy-only cases. The oracle result "
            "is an achieved lower bound, not a proven mathematical ceiling."
        ),
        "rows": joined,
        "rows_sha256": canonical_hash(joined),
    }


def build_success_receipt(
    *,
    result: Mapping[str, Any],
    oracle_receipt: Mapping[str, Any],
    bound_evidence: Mapping[str, Any],
    claim_sha256: str,
    claim: Mapping[str, Any],
    created_at_utc: str,
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    _require(result.get("schema_version") == 1 and result.get("tool") == "evaluate", "evaluator contract changed")
    _require(result.get("authorizing") is False, "evaluator became authorizing")
    _require(result.get("split") == FINAL_SPLIT_ID, "evaluator scope changed")
    _require(result.get("case_count") == EXPECTED_CASE_COUNT, "expected 200 cases")
    _require(result.get("same_tapes") is True, "same-tapes invariant failed")
    _require(result.get("rollout_count") == EXPECTED_CASE_COUNT, "expected one rollout per case")
    _require(result.get("paired_comparisons") == {}, "more than one policy ran")
    label = _policy_label()
    policies = result.get("policies")
    rows_by_policy = result.get("rows")
    _require(isinstance(policies, Mapping) and set(policies) == {label}, "expected one fixed ONNX policy")
    _require(
        isinstance(rows_by_policy, Mapping) and set(rows_by_policy) == {label},
        "expected one fixed ONNX row set",
    )
    raw_rows = rows_by_policy[label]
    _require(isinstance(raw_rows, Sequence) and len(raw_rows) == EXPECTED_CASE_COUNT, "expected 200 rows")
    rows = [_json_row(row, index) for index, row in enumerate(raw_rows)]
    _require(len({row["row_id"] for row in rows}) == EXPECTED_CASE_COUNT, "row IDs not unique")
    aggregate = _aggregate_rows(rows)
    metrics = policies[label]
    _require(isinstance(metrics, Mapping) and set(metrics) == _AGGREGATE_FIELDS, "aggregate fields changed")
    _require(dict(metrics) == aggregate, "aggregate does not recompute from rows")
    _require(aggregate["solved_count"] == EXPECTED_SOLVED_COUNT, "expected exactly 163 solves")
    _require(aggregate["solve_rate"] == EXPECTED_SOLVE_RATE, "expected solve rate 0.815")
    _require(aggregate["hard_violation_count"] == 0, "hard violations must be zero")
    _require(aggregate["maximum_conservation_residual"] == 0.0, "residual must be 0.0")
    interval = wilson_interval(EXPECTED_SOLVED_COUNT, EXPECTED_CASE_COUNT, digits=10)
    _require(interval == EXPECTED_WILSON_95, "registered Wilson interval changed")
    aggregate["wilson_95"] = interval
    comparison = _join_oracle(rows, oracle_receipt)
    split_evidence = _split_and_family_evidence(rows, oracle_receipt["split_contract"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "city-recovery-v4-single-use-final-evaluation-success",
        "tool": TOOL_ID,
        "status": "complete_owner_authorized_final_evaluation",
        "created_at_utc": created_at_utc,
        "authorization": dict(AUTHORIZATION),
        "timing": {
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "elapsed_seconds": elapsed_seconds,
        },
        "authorizing": False,
        "owner_authorized_execution": True,
        "model_selection_used": False,
        "training_performed": False,
        "split": FINAL_SPLIT_ID,
        "final_split_used": True,
        "case_count": EXPECTED_CASE_COUNT,
        "policy_count": 1,
        "rollout_count": EXPECTED_CASE_COUNT,
        "evaluation_pipeline": "scripts.evaluate.build_cases/rollout/aggregate",
        "claim": {
            "path": claim["claim_path"],
            "sha256": claim_sha256,
            "git_commit": claim["git"]["commit"],
            "worktree_clean_before_claim": True,
        },
        "artifact": {
            "path": _repo_path(ARTIFACT),
            "sha256": EXPECTED_ARTIFACT_SHA256,
            "policy_label": label,
            "runtime": "onnxruntime-cpu",
        },
        "aggregate": aggregate,
        "split_contract": split_evidence["split_contract"],
        "split_contract_sha256": split_evidence["split_contract_sha256"],
        "ordered_split_identity": split_evidence["ordered_split_identity"],
        "ordered_split_identity_sha256": split_evidence[
            "ordered_split_identity_sha256"
        ],
        "per_family": split_evidence["per_family"],
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "oracle_comparison": comparison,
        "invariants": {
            "case_count_exactly_200": True,
            "row_ids_unique": True,
            "exactly_one_onnx_policy_evaluated": True,
            "exactly_one_rollout_per_case": True,
            "solved_count_exactly_163": True,
            "all_hard_violation_counts_zero": True,
            "all_conservation_residuals_exactly_zero": True,
            "artifact_sha256_matched_before_claim": True,
            "clean_committed_head_recorded_before_claim": True,
        },
        "bound_evidence": dict(bound_evidence),
        "source_identity": {
            "scripts/publish_final_evaluation_v4.py": file_sha256(Path(__file__)),
            "backend/app/shared_evidence.py": file_sha256(
                ROOT / "backend" / "app" / "shared_evidence.py"
            ),
            **{
                path: file_sha256(ROOT / path)
                for path in CORE_SOURCE_SHA256
            },
        },
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FinalEvaluationError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise FinalEvaluationError(f"refusing to overwrite existing evidence: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _observed_result_evidence(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"runner_returned": False}
    try:
        payload = _jsonable(result)
        canonical_result_sha256 = canonical_hash(payload)
        policies = payload.get("policies")
        rows_by_policy = payload.get("rows")
        summary: dict[str, Any] = {
            "runner_returned": True,
            "canonical_result_sha256": canonical_result_sha256,
            "result": payload,
        }
        if isinstance(policies, Mapping) and len(policies) == 1:
            label, aggregate = next(iter(policies.items()))
            summary["policy_label"] = label
            summary["observed_aggregate"] = aggregate
            if isinstance(aggregate, Mapping):
                summary["observed_solved_count"] = aggregate.get("solved_count")
        if isinstance(rows_by_policy, Mapping) and len(rows_by_policy) == 1:
            rows = next(iter(rows_by_policy.values()))
            if isinstance(rows, list):
                summary["observed_row_count"] = len(rows)
                summary["observed_rows_sha256"] = canonical_hash(rows)
        return summary
    except BaseException as exc:
        return {
            "runner_returned": True,
            "evidence_conversion_error": f"{type(exc).__name__}: {exc}",
        }


def execute_once(
    *,
    paths: PublicationPaths = DEFAULT_PATHS,
    git_probe: GitProbe = _probe_git,
    evidence_probe: EvidenceProbe = _probe_evidence_hashes,
    reference_loader: ReferenceLoader = _load_reference_evidence,
    runner: EvaluationRunner = _production_runner,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Consume the claim before any reserved-split import, then terminate once."""

    started_monotonic = time.perf_counter()
    timestamp = created_at_utc or _utc_now()
    ready = preflight(paths=paths, git_probe=git_probe, evidence_probe=evidence_probe)
    oracle_receipt, bound_evidence = reference_loader()
    claim = {
        "schema_version": SCHEMA_VERSION,
        "kind": "city-recovery-v4-single-use-final-evaluation-claim",
        "tool": TOOL_ID,
        "status": "claimed_before_reserved_split_import",
        "created_at_utc": timestamp,
        "authorization": dict(AUTHORIZATION),
        "irreversible": True,
        "git": ready["git"],
        "timing": {"started_at_utc": timestamp},
        "split_contract": dict(oracle_receipt["split_contract"]),
        "split_contract_sha256": canonical_hash(oracle_receipt["split_contract"]),
        "preflight_evidence_sha256": ready["evidence_sha256"],
        "source_identity": {
            "scripts/publish_final_evaluation_v4.py": file_sha256(Path(__file__)),
            **{
                path: file_sha256(ROOT / path)
                for path in CORE_SOURCE_SHA256
            },
        },
        "artifact": {
            "path": _repo_path(ARTIFACT),
            "sha256": EXPECTED_ARTIFACT_SHA256,
        },
        "registered_result_gate": {
            "case_count": EXPECTED_CASE_COUNT,
            "solved_count": EXPECTED_SOLVED_COUNT,
            "wilson_95": EXPECTED_WILSON_95,
            "hard_violation_count": 0,
            "maximum_conservation_residual": 0.0,
        },
        "terminal_outputs": {
            "success": _display_path(paths.success),
            "failure": _display_path(paths.failure),
        },
        "claim_path": _display_path(paths.claim),
    }
    claim_bytes = _json_bytes(claim)
    claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
    _atomic_create(paths.claim, claim_bytes)
    stage = "lazy_import_and_exactly_200_rollouts"
    result: Mapping[str, Any] | None = None
    try:
        result = runner(ARTIFACT)
        stage = "result_validation_and_oracle_join"
        completed_at_utc = _utc_now()
        elapsed_seconds = round(time.perf_counter() - started_monotonic, 6)
        success = build_success_receipt(
            result=result,
            oracle_receipt=oracle_receipt,
            bound_evidence=bound_evidence,
            claim_sha256=claim_sha256,
            claim=claim,
            created_at_utc=timestamp,
            started_at_utc=timestamp,
            completed_at_utc=completed_at_utc,
            elapsed_seconds=elapsed_seconds,
        )
        success["timing"] = {
            "started_at_utc": timestamp,
            "evaluation_completed_at_utc": completed_at_utc,
            "evaluation_elapsed_seconds": elapsed_seconds,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started_monotonic, 6),
        }
        stage = "success_receipt_create_new"
        success_bytes = _json_bytes(success)
        summary = {
            "status": "success",
            "claim": _display_path(paths.claim),
            "claim_sha256": claim_sha256,
            "receipt": _display_path(paths.success),
            "receipt_sha256": hashlib.sha256(success_bytes).hexdigest(),
            "solved_count": EXPECTED_SOLVED_COUNT,
            "case_count": EXPECTED_CASE_COUNT,
            "wilson_95": EXPECTED_WILSON_95,
            "markdown_pending": _display_path(paths.markdown),
        }
        _atomic_create(paths.success, success_bytes)
        return summary
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "kind": "city-recovery-v4-single-use-final-evaluation-failure",
            "tool": TOOL_ID,
            "status": "terminal_failure_no_retry",
            "created_at_utc": _utc_now(),
            "authorization": dict(AUTHORIZATION),
            "claim": {"path": _display_path(paths.claim), "sha256": claim_sha256},
            "git_commit": claim["git"]["commit"],
            "timing": {
                "started_at_utc": timestamp,
                "failed_at_utc": _utc_now(),
                "elapsed_seconds": round(time.perf_counter() - started_monotonic, 6),
            },
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "observed_result_evidence": _observed_result_evidence(result),
            "retry_permitted": False,
        }
        try:
            _atomic_create(paths.failure, _json_bytes(failure))
        except BaseException as failure_error:
            raise FinalEvaluationError(
                "claim consumed; evaluation failed and failure receipt could not be written: "
                f"{failure_error}"
            ) from exc
        raise FinalEvaluationError(
            f"claim consumed; terminal failure recorded at {paths.failure}: {exc}"
        ) from exc


def _format_ci(interval: Sequence[float]) -> str:
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def render_markdown(receipt: Mapping[str, Any], receipt_sha256: str) -> str:
    """Render the fixed final table solely from a validated success receipt."""

    _require(
        receipt.get("kind") == "city-recovery-v4-single-use-final-evaluation-success"
        and receipt.get("aggregate", {}).get("solved_count") == EXPECTED_SOLVED_COUNT,
        "Markdown requires the validated success receipt",
    )
    _require(_SHA256.fullmatch(receipt_sha256) is not None, "receipt SHA invalid")
    comparison = receipt.get("oracle_comparison")
    _require(isinstance(comparison, Mapping), "oracle comparison missing")
    pairing = comparison.get("pairing")
    _require(isinstance(pairing, Mapping), "oracle pairing missing")
    per_family = receipt.get("per_family")
    _require(
        isinstance(per_family, list) and len(per_family) == 5,
        "per-family results missing",
    )
    lines = [
        "# Final 200-case results",
        "",
        (
            "The shipped v4 row is the single owner-authorized learned-policy final "
            "evaluation. The final result was not used for model selection or training."
        ),
        "",
        "| Method | Solved | Rate | Wilson 95% CI | Scope |",
        "|---|---:|---:|---:|---|",
    ]
    for label, solved, scope in REFERENCE_RESULTS:
        interval = wilson_interval(solved, EXPECTED_CASE_COUNT, digits=10)
        emphasis = "**" if label == "Shipped v4 PPO" else ""
        lines.append(
            f"| {label} | {emphasis}{solved}/200{emphasis} | "
            f"{emphasis}{solved / EXPECTED_CASE_COUNT:.3f}{emphasis} | "
            f"{emphasis}{_format_ci(interval)}{emphasis} | {scope} |"
        )
    lines.extend(
        [
            "",
            "## Shipped v4 results by scenario family",
            "",
            "| Family | Solved | Wilson 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for family in per_family:
        lines.append(
            f"| `{family['family_id']}` | {family['solved_count']}/"
            f"{family['case_count']} | {_format_ci(family['wilson_95'])} |"
        )
    lines.extend(
        [
            "",
            (
                "The 200 cases are clustered within five fixed scenario families. "
                "The overall Wilson interval treats case outcomes as Bernoulli "
                "observations and does not model within-family dependence, so its "
                "precision is slightly overstated; the 40-case family rows expose "
                "that heterogeneity directly."
            ),
            "",
            (
                "The clairvoyant CEM sees the complete future shock tape. It is a "
                "privileged anytime achieved lower bound, not a causal baseline or "
                "a proven mathematical ceiling."
            ),
            "",
            "## Matched shipped-policy / oracle cases",
            "",
            "| Both | Policy only | Oracle only | Neither | Known-feasible union |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {pairing['both']} | {pairing['policy_only']} | "
                f"{pairing['oracle_only']} | {pairing['neither']} | "
                f"{comparison['known_feasible_union_count']}/200 |"
            ),
            "",
            (
                f"The aggregate count ratio is 163/182 = "
                f"{comparison['aggregate_count_ratio_policy_to_oracle_achieved']:.1%}; "
                f"casewise policy coverage of oracle-achieved cases is "
                f"{comparison['casewise_policy_coverage_of_oracle_achieved']:.1%}. "
                "They are reported separately because finite CEM solved sets need not nest."
            ),
            (
                "The shipped policy is 16 solved cases ahead of the strongest "
                "hand-coded planner, the tuned constant rule at 147/200."
            ),
            "",
            "Every bound result has zero hard violations and exactly `0.0` conservation residual.",
            "",
            "## Evidence",
            "",
            f"- Success receipt SHA-256: `{receipt_sha256}`",
            f"- Shipped-policy ordered rows SHA-256: `{receipt['rows_sha256']}`",
            f"- Oracle pairing rows SHA-256: `{comparison['rows_sha256']}`",
            f"- Shipped v4 ONNX SHA-256: `{EXPECTED_ARTIFACT_SHA256}`",
            f"- Shipped artifact manifest SHA-256: `{ARTIFACT_MANIFEST_SHA256}`",
            f"- Development parity receipt SHA-256: `{DEV_PARITY_RECEIPT_SHA256}`",
            f"- Privileged oracle receipt SHA-256: `{ORACLE_RECEIPT_SHA256}`",
            f"- Public/legacy regression gate SHA-256: `{REGRESSION_GATE_SHA256}`",
            "",
        ]
    )
    return "\n".join(lines)


def publish_markdown_from_success(
    *, paths: PublicationPaths = DEFAULT_PATHS
) -> dict[str, Any]:
    _require(paths.claim.is_file(), "claim receipt is missing")
    _require(paths.success.is_file(), "success receipt is missing")
    _require(not paths.failure.exists(), "failure receipt exists")
    _require(not paths.markdown.exists(), "final Markdown already exists")
    receipt = load_json_object(paths.success, "final success receipt", error_type=FinalEvaluationError)
    _require(
        receipt.get("kind") == "city-recovery-v4-single-use-final-evaluation-success"
        and receipt.get("status") == "complete_owner_authorized_final_evaluation",
        "success receipt status changed",
    )
    _require(
        receipt.get("claim", {}).get("sha256") == file_sha256(paths.claim),
        "success receipt does not bind the claim",
    )
    _require(
        receipt.get("artifact", {}).get("sha256") == EXPECTED_ARTIFACT_SHA256,
        "success receipt artifact changed",
    )
    aggregate = receipt.get("aggregate")
    _require(
        isinstance(aggregate, Mapping)
        and aggregate.get("case_count") == EXPECTED_CASE_COUNT
        and aggregate.get("solved_count") == EXPECTED_SOLVED_COUNT
        and aggregate.get("wilson_95") == EXPECTED_WILSON_95
        and aggregate.get("hard_violation_count") == 0
        and aggregate.get("maximum_conservation_residual") == 0.0,
        "success aggregate changed",
    )
    _require(
        receipt.get("rows_sha256") == canonical_hash(receipt.get("rows")),
        "success rows hash mismatch",
    )
    _require(
        receipt.get("split_contract") == EXPECTED_FINAL_SPLIT_CONTRACT
        and receipt.get("split_contract_sha256")
        == canonical_hash(EXPECTED_FINAL_SPLIT_CONTRACT)
        and receipt.get("ordered_split_identity_sha256")
        == canonical_hash(receipt.get("ordered_split_identity"))
        and isinstance(receipt.get("per_family"), list)
        and len(receipt["per_family"]) == 5,
        "success split or per-family evidence changed",
    )
    comparison = receipt.get("oracle_comparison")
    _require(
        isinstance(comparison, Mapping)
        and comparison.get("rows_sha256") == canonical_hash(comparison.get("rows")),
        "oracle comparison rows hash mismatch",
    )
    pairing = comparison.get("pairing")
    _require(
        isinstance(pairing, Mapping)
        and sum(int(pairing.get(key, -1)) for key in ("both", "policy_only", "oracle_only", "neither"))
        == EXPECTED_CASE_COUNT
        and pairing.get("both", -1) + pairing.get("policy_only", -1)
        == EXPECTED_SOLVED_COUNT
        and pairing.get("both", -1) + pairing.get("oracle_only", -1) == 182,
        "oracle pairing changed",
    )
    bound = receipt.get("bound_evidence")
    _require(
        isinstance(bound, Mapping)
        and bound.get("artifact_manifest", {}).get("sha256")
        == ARTIFACT_MANIFEST_SHA256
        and bound.get("development_parity_receipt", {}).get("sha256")
        == DEV_PARITY_RECEIPT_SHA256
        and bound.get("oracle_receipt", {}).get("sha256") == ORACLE_RECEIPT_SHA256
        and bound.get("regression_gate", {}).get("sha256") == REGRESSION_GATE_SHA256
        and bound.get("legacy_fixture", {}).get("sha256") == LEGACY_FIXTURE_SHA256,
        "bound evidence changed",
    )
    _require(
        bound.get("core_sources") == CORE_SOURCE_SHA256,
        "bound core source evidence changed",
    )
    receipt_sha256 = file_sha256(paths.success)
    markdown = render_markdown(receipt, receipt_sha256).encode("utf-8")
    _atomic_create(paths.markdown, markdown)
    return {
        "status": "published",
        "markdown": _display_path(paths.markdown),
        "receipt_sha256": receipt_sha256,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute-owner-authorized-single-use", action="store_true")
    modes.add_argument("--publish-markdown-from-success", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.preflight:
            result = preflight()
        elif args.execute_owner_authorized_single_use:
            result = execute_once()
        else:
            result = publish_markdown_from_success()
    except (FinalEvaluationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"final evaluation publication failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
