from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256, wilson_interval
from scripts.headroom import PlannerResult, aggregate_results
from scripts.publish_oracle_study import (
    PORTABLE_KIND,
    OraclePublicationError,
    publish_study,
)
from scripts.run_oracle_study import (
    CANONICAL_CASE_COUNT,
    DEFAULT_REFERENCE_EVIDENCE,
    HISTORICAL_MPC_CONFIG,
    HISTORICAL_ORACLE_CONFIG,
    HISTORICAL_RECEIPT,
    HISTORICAL_RECEIPT_SHA256,
    _case_contract,
    _study_contract,
    build_cases,
    build_reference_comparison,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _planner_result(*, solved: bool, index: int) -> PlannerResult:
    digest = f"{index:064x}"[-64:]
    trajectory = f"{index + 1000:064x}"[-64:]
    return PlannerResult(
        solved=solved,
        minimum_tail_margin=0.02 if solved else -0.02,
        resilience_auc=0.50 if solved else 0.40,
        reason_codes=() if solved else ("assessment_tail_targets_met",),
        hard_violation_count=0,
        maximum_conservation_residual=0.0,
        action_sequence_sha256=digest,
        trajectory_sha256=trajectory,
    )


def _build_receipt(
    *,
    split: str,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    cases = build_cases(split)
    if split == "dev":
        parity = json.loads(DEFAULT_REFERENCE_EVIDENCE.read_text(encoding="utf-8"))
        solved_values = [bool(row["onnx"]["solved"]) for row in parity["parity"]["rows"]]
    else:
        solved_values = [index < 180 for index in range(CANONICAL_CASE_COUNT)]
    results = [
        _planner_result(solved=solved, index=index + (0 if split == "dev" else 500))
        for index, solved in enumerate(solved_values)
    ]
    base_aggregate = aggregate_results(results)
    oracle_aggregate = dict(base_aggregate)
    oracle_aggregate["wilson_95"] = wilson_interval(
        int(oracle_aggregate["solved_count"]), CANONICAL_CASE_COUNT, digits=10
    )
    mpc_aggregates = {str(horizon): dict(base_aggregate) for horizon in (1, 3, 5)}
    iterations = HISTORICAL_ORACLE_CONFIG.min_iterations
    candidate_evaluations = 2 + HISTORICAL_ORACLE_CONFIG.population * iterations
    rows: list[dict[str, Any]] = []
    for case, result in zip(cases, results, strict=True):
        receipt = result.as_receipt()
        rows.append(
            {
                **_case_contract(case),
                "tuned_rule": receipt,
                "selected_mpc_k1": receipt,
                "clairvoyant_oracle_cem": receipt,
                "oracle_budget": {
                    "population": HISTORICAL_ORACLE_CONFIG.population,
                    "elite_fraction": HISTORICAL_ORACLE_CONFIG.elite_fraction,
                    "elite_count": 52,
                    "iterations": iterations,
                    "candidate_evaluations": candidate_evaluations,
                    "simulated_transitions": candidate_evaluations
                    * int(case.scenario.horizon_days),
                    "initial_std": HISTORICAL_ORACLE_CONFIG.initial_std,
                    "std_floor": HISTORICAL_ORACLE_CONFIG.std_floor,
                    "smoothing": HISTORICAL_ORACLE_CONFIG.smoothing,
                    "minimum_iterations": HISTORICAL_ORACLE_CONFIG.min_iterations,
                    "maximum_iterations": HISTORICAL_ORACLE_CONFIG.max_iterations,
                    "patience": HISTORICAL_ORACLE_CONFIG.patience,
                },
                "oracle_warm_start": {
                    "winner_not_worse_than_tuned_or_mpc": True,
                },
                "oracle_search_wide_invariants": {
                    "maximum_hard_violation_count": 0,
                    "maximum_conservation_residual": 0.0,
                },
                "oracle_worker_runtime": {
                    "pid": 1234,
                    "affinity_mask": None,
                    "priority": "normal",
                },
            }
        )
    comparison = (
        build_reference_comparison(
            cases=cases,
            oracle_rows=results,
            reference_path=DEFAULT_REFERENCE_EVIDENCE,
        )
        if split == "dev"
        else None
    )
    return {
        "schema_version": 1,
        "tool": "run_oracle_study.py",
        "status": "complete_privileged_clairvoyant_oracle_diagnostic",
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "authorizing": False,
        "model_selection_used": False,
        "submission_baseline": False,
        "privileged_diagnostic": True,
        "split": split,
        "split_contract": contract["split_contracts"][split],
        "case_count": CANONICAL_CASE_COUNT,
        "final_split_used": split == "final",
        "learned_v4_policy_evaluated": False,
        "learned_v4_final_evaluated": False,
        "full_future_shock_tape_visible": True,
        "interpretation": {
            "anytime_achieved_lower_bound": True,
            "mathematical_optimum_claimed": False,
            "infeasibility_certificate_claimed": False,
            "wording": "synthetic fixed-protocol test evidence",
        },
        "study_contract_sha256": contract_sha256,
        "source_identity": contract["source_identity"],
        "historical_reference": contract["historical_reference"],
        "mpc_warm_start": {
            "config": asdict(HISTORICAL_MPC_CONFIG),
            "selection": {
                "rule": "registered global-horizon rule",
                "selected_horizon": 1,
                "aggregates": mpc_aggregates,
            },
        },
        "oracle": {
            "config": asdict(HISTORICAL_ORACLE_CONFIG),
            "objective": contract["objective"],
            "rng_namespace": contract["rng_namespace"],
            "aggregate": oracle_aggregate,
            "total_candidate_evaluations": candidate_evaluations
            * CANONICAL_CASE_COUNT,
            "total_simulated_transitions": candidate_evaluations
            * sum(int(case.scenario.horizon_days) for case in cases),
        },
        "planner_aggregates": {
            "tuned_rule": base_aggregate,
            "selected_mpc_k1": base_aggregate,
            "clairvoyant_oracle_cem": oracle_aggregate,
        },
        "development_shipped_policy_comparison": comparison,
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "invariants": {
            "case_count_exactly_200": True,
            "row_ids_unique": True,
            "all_planner_hard_violation_counts_zero": True,
            "all_planner_conservation_residuals_exactly_zero": True,
            "all_oracle_search_hard_violation_counts_zero": True,
            "all_oracle_search_conservation_residuals_exactly_zero": True,
            "production_decoder_and_exact_projection_used": True,
            "learned_policy_loaded_or_run": False,
        },
        "runtime": {
            "output_root": str(root.resolve()),
            "requested_workers": 8,
            "multiprocessing_start_method": "spawn",
            "events": [],
            "elapsed_seconds": 1.0,
        },
    }


def _raw_study(root: Path) -> dict[str, Path]:
    contract = _study_contract(("dev", "final"))
    contract_sha256 = canonical_hash(contract)
    protocol = {
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "contract_sha256": contract_sha256,
        "contract": contract,
    }
    receipts = {
        split: _build_receipt(
            split=split,
            root=root,
            contract=contract,
            contract_sha256=contract_sha256,
        )
        for split in ("dev", "final")
    }
    protocol_path = root / "protocol.json"
    _write_json(protocol_path, protocol)
    receipt_paths = {
        split: root / split / "receipt.json" for split in ("dev", "final")
    }
    for split, receipt in receipts.items():
        _write_json(receipt_paths[split], receipt)
    summary = {
        "schema_version": 1,
        "tool": "run_oracle_study.py",
        "status": "complete",
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "contract_sha256": contract_sha256,
        "splits": {
            split: {
                "receipt": f"{split}/receipt.json",
                "receipt_sha256": file_sha256(receipt_paths[split]),
                "solved_count": receipts[split]["oracle"]["aggregate"]["solved_count"],
                "solve_rate": receipts[split]["oracle"]["aggregate"]["solve_rate"],
                "wilson_95": receipts[split]["oracle"]["aggregate"]["wilson_95"],
                "learned_v4_policy_evaluated": False,
            }
            for split in ("dev", "final")
        },
        "learned_v4_final_evaluated": False,
        "historical_37_of_40_preserved": True,
    }
    summary_path = root / "summary.json"
    _write_json(summary_path, summary)
    return {
        "protocol": protocol_path,
        "summary": summary_path,
        "dev": receipt_paths["dev"],
        "final": receipt_paths["final"],
    }


def _outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "tracked" / "dev.json",
        tmp_path / "tracked" / "final.json",
        tmp_path / "tracked" / "study.md",
    )


def test_publish_study_creates_portable_hash_bound_evidence(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    raw = _raw_study(external_root)
    dev_output, final_output, markdown_output = _outputs(tmp_path)
    historical_before = file_sha256(HISTORICAL_RECEIPT)

    result = publish_study(
        external_root.resolve(),
        dev_output=dev_output,
        final_output=final_output,
        markdown_output=markdown_output,
    )

    dev = json.loads(dev_output.read_text(encoding="utf-8"))
    final = json.loads(final_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert result["learned_v4_final_evaluated"] is False
    assert dev["kind"] == final["kind"] == PORTABLE_KIND
    assert len(dev["rows"]) == len(final["rows"]) == CANONICAL_CASE_COUNT
    assert dev["rows_sha256"] == canonical_hash(dev["rows"])
    assert final["rows_sha256"] == canonical_hash(final["rows"])
    assert dev["runtime"]["output_root"] == "distribution:not_in_repository"
    assert final["runtime"]["output_root"] == "distribution:not_in_repository"
    assert str(external_root.resolve()) not in dev_output.read_text(encoding="utf-8")
    assert str(external_root.resolve()) not in final_output.read_text(encoding="utf-8")
    assert dev["publication"]["raw_evidence"]["receipt_sha256"] == file_sha256(
        raw["dev"]
    )
    assert final["publication"]["raw_evidence"]["receipt_sha256"] == file_sha256(
        raw["final"]
    )
    assert "privileged anytime achieved lower bound" in markdown
    assert "Policy only" in markdown and "Oracle only" in markdown
    assert "**not evaluated**" in markdown
    assert file_sha256(HISTORICAL_RECEIPT) == historical_before == HISTORICAL_RECEIPT_SHA256
    with pytest.raises(OraclePublicationError, match="overwrite"):
        publish_study(
            external_root.resolve(),
            dev_output=dev_output,
            final_output=final_output,
            markdown_output=markdown_output,
        )


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    (
        (
            "final",
            lambda receipt: receipt.__setitem__("learned_v4_policy_evaluated", True),
            "learned_v4_policy_evaluated",
        ),
        (
            "dev",
            lambda receipt: receipt["rows"][0].__setitem__("tape_sha256", "0" * 64),
            "ordered case/tape mismatch",
        ),
        (
            "dev",
            lambda receipt: receipt["rows"][0]["oracle_search_wide_invariants"].__setitem__(
                "maximum_conservation_residual", 1e-9
            ),
            "search residual",
        ),
        (
            "dev",
            lambda receipt: receipt["development_shipped_policy_comparison"][
                "pairing"
            ].__setitem__("oracle_only", 99),
            "pairing does not recompute",
        ),
    ),
)
def test_publish_study_rejects_scientific_contract_drift(
    tmp_path: Path,
    target: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    external_root = tmp_path / "external"
    raw = _raw_study(external_root)
    receipt = json.loads(raw[target].read_text(encoding="utf-8"))
    mutate(receipt)
    _write_json(raw[target], receipt)
    dev_output, final_output, markdown_output = _outputs(tmp_path)

    with pytest.raises(OraclePublicationError, match=message):
        publish_study(
            external_root.resolve(),
            dev_output=dev_output,
            final_output=final_output,
            markdown_output=markdown_output,
        )


def test_publish_study_requires_summary_to_bind_raw_receipt_bytes(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    raw = _raw_study(external_root)
    dev_receipt = json.loads(raw["dev"].read_text(encoding="utf-8"))
    dev_receipt["created_at_utc"] = "2026-08-11T00:00:01+00:00"
    _write_json(raw["dev"], dev_receipt)
    dev_output, final_output, markdown_output = _outputs(tmp_path)

    with pytest.raises(OraclePublicationError, match="summary dev evidence mismatch"):
        publish_study(
            external_root.resolve(),
            dev_output=dev_output,
            final_output=final_output,
            markdown_output=markdown_output,
        )
