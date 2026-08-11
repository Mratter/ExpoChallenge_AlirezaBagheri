from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.shared_evidence import file_sha256
from scripts.run_oracle_study import (
    CANONICAL_CASE_COUNT,
    DEFAULT_REFERENCE_EVIDENCE,
    HISTORICAL_MPC_CONFIG,
    HISTORICAL_ORACLE_CONFIG,
    HISTORICAL_RECEIPT,
    HISTORICAL_RECEIPT_SHA256,
    OracleStudyError,
    _atomic_create_json,
    _case_contract,
    _load_record,
    _wrap_record,
    build_cases,
    build_reference_comparison,
)
from scripts.headroom import PlannerResult


def _planner_result(*, solved: bool) -> PlannerResult:
    return PlannerResult(
        solved=solved,
        minimum_tail_margin=0.01 if solved else -0.01,
        resilience_auc=0.45,
        reason_codes=() if solved else ("assessment_tail_targets_met",),
        hard_violation_count=0,
        maximum_conservation_residual=0.0,
        action_sequence_sha256="a" * 64,
        trajectory_sha256="b" * 64,
    )


def test_build_cases_uses_canonical_200_case_rosters() -> None:
    development = build_cases("dev")
    final = build_cases("final")

    assert len(development) == len(final) == CANONICAL_CASE_COUNT
    assert len({case.row_id for case in development}) == CANONICAL_CASE_COUNT
    assert len({case.row_id for case in final}) == CANONICAL_CASE_COUNT
    assert [case.case_seed for case in development[:40]] == list(range(820000, 820040))
    assert [case.case_seed for case in final[:40]] == list(range(830000, 830040))
    assert not set(case.row_id for case in development) & set(
        case.row_id for case in final
    )


def test_registered_budget_matches_historical_37_of_40_receipt() -> None:
    historical = json.loads(HISTORICAL_RECEIPT.read_text(encoding="utf-8"))

    assert file_sha256(HISTORICAL_RECEIPT) == HISTORICAL_RECEIPT_SHA256
    assert historical["oracle"]["config"] == {
        **HISTORICAL_ORACLE_CONFIG.__dict__
    }
    assert historical["mpc"]["config"] == {**HISTORICAL_MPC_CONFIG.__dict__}
    assert historical["oracle"]["solved_count"] == 37
    assert historical["case_count"] == 40


def test_external_records_are_create_new_and_contract_bound(tmp_path: Path) -> None:
    case = build_cases("dev")[0]
    path = tmp_path / "dev" / "preparation" / "000.json"
    record = _wrap_record(
        contract_sha256="c" * 64,
        split="dev",
        phase="preparation",
        index=0,
        case=case,
        payload={"row_id": case.row_id},
    )
    _atomic_create_json(path, record)

    assert _load_record(
        path,
        contract_sha256="c" * 64,
        split="dev",
        phase="preparation",
        index=0,
        case=case,
    ) == {"row_id": case.row_id}
    with pytest.raises(OracleStudyError, match="overwrite"):
        _atomic_create_json(path, record)
    with pytest.raises(OracleStudyError, match="mismatched"):
        _load_record(
            path,
            contract_sha256="d" * 64,
            split="dev",
            phase="preparation",
            index=0,
            case=case,
        )


def test_dev_comparison_uses_accepted_rows_without_policy_inference() -> None:
    cases = build_cases("dev")
    receipt = json.loads(DEFAULT_REFERENCE_EVIDENCE.read_text(encoding="utf-8"))
    policy_rows = receipt["parity"]["rows"]
    oracle_rows = [
        _planner_result(solved=bool(row["onnx"]["solved"])) for row in policy_rows
    ]
    first_policy_failure = next(
        index for index, row in enumerate(policy_rows) if not row["onnx"]["solved"]
    )
    oracle_rows[first_policy_failure] = _planner_result(solved=True)

    comparison = build_reference_comparison(
        cases=cases,
        oracle_rows=oracle_rows,
        reference_path=DEFAULT_REFERENCE_EVIDENCE,
    )

    assert comparison["shipped_policy_solved_count"] == 178
    assert comparison["oracle_solved_count"] == 179
    assert comparison["pairing"] == {
        "both": 178,
        "policy_only": 0,
        "oracle_only": 1,
        "neither": 21,
    }
    assert comparison["oracle_only_contested_row_ids"] == [
        cases[first_policy_failure].row_id
    ]
    assert comparison["known_feasible_union_count"] == 179


def test_final_cases_are_not_joined_to_or_imported_from_a_policy() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "run_oracle_study.py").read_text(
        encoding="utf-8"
    )

    assert "model.policy" not in source
    assert "scripts.evaluate" not in source
    assert 'if split == "dev"\n        else None' in source
    assert '"learned_v4_final_evaluated": False' in source
    assert _case_contract(build_cases("final")[0])["row_id"].startswith("v3_final_")
