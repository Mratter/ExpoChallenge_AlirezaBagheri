"""Receipt-only integrity checks for the development seed-ensemble note."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.shared_evidence import canonical_hash, file_sha256


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "action-mean-ensemble-5x2m-dev-200.json"
)
PARITY_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "city_recovery_ppo.v4.parity.json"
)
NOTE = ROOT / "benchmarks" / "v4" / "seed-ensemble-200.md"

EXPECTED_RMS_BY_SEED = {
    37017: "456c8fab41d53a8d1ecc23fdf461cc9df5642726cff0f84f5bb2f94643876835",
    47017: "77156039dd87a2873fb4f1098385d2f163346f4a153a30fe728e7673a9f342ea",
    57017: "a75c02959bde3cc909e9409c980ee3685ae726a5f94bf2cb5096e2ab19252c97",
    67017: "6823fd134e915a0d22d149895479a003c51711d3f2b0649c37205674365cd022",
    77017: "6cca61ae612700d33cbbcdae8e46d9e4997cd8916859f7255b16dbb9cb344b4f",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_seed_ensemble_receipt_is_dev_only_complete_and_source_bound() -> None:
    receipt = _load(RECEIPT)
    deterministic_bytes = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    assert RECEIPT.read_bytes() == deterministic_bytes
    assert receipt["schema_version"] == (
        "city-recovery-development-action-mean-ensemble-v1"
    )
    assert receipt["status"] == "complete_development_experiment"
    assert receipt["split"] == receipt["split_contract"]["id"] == "dev"
    assert receipt["final_split_used"] is False
    assert receipt["split_contract"]["cartesian_case_count"] == 200
    assert receipt["decision"] == {
        "candidate_count": 1,
        "candidate_kept": True,
        "candidate_promoted": False,
        "configuration_search_performed": False,
        "reason": receipt["decision"]["reason"],
    }

    evaluation = receipt["evaluation"]
    rows = evaluation["rows"]
    assert evaluation["case_count"] == len(rows) == 200
    assert evaluation["solved_count"] == 179
    assert evaluation["solve_rate"] == 0.895
    assert evaluation["hard_violation_count"] == 0
    assert evaluation["maximum_conservation_residual"] == 0.0
    assert len({row["row_id"] for row in rows}) == 200
    assert canonical_hash(rows) == evaluation["rows_sha256"]

    members = receipt["ensemble"]["members"]
    assert receipt["ensemble"]["member_count"] == len(members) == 5
    assert [member["weight"] for member in members] == [0.2] * 5
    assert canonical_hash(members) == receipt["ensemble"]["members_identity_sha256"]
    assert {
        member["seed"]: member["checkpoint_bundle"]["obs_rms_sha256"]
        for member in members
    } == EXPECTED_RMS_BY_SEED
    assert len(set(EXPECTED_RMS_BY_SEED.values())) == 5
    assert [member["checkpoint_bundle"]["checkpoint_id"] for member in members] == [
        f"seed-{seed}-ppo-2000000" for seed in EXPECTED_RMS_BY_SEED
    ]

    for relative_path, expected_sha256 in receipt["source_files"].items():
        assert file_sha256(ROOT / relative_path) == expected_sha256


def test_seed_ensemble_matched_contingency_recomputes_from_pinned_dev_rows() -> None:
    receipt = _load(RECEIPT)
    parity = _load(PARITY_RECEIPT)
    comparison = receipt["comparison_to_selected_single_checkpoint"]

    assert file_sha256(PARITY_RECEIPT) == comparison["parity_receipt_sha256"]
    selected_rows = parity["parity"]["rows"]
    ensemble_rows = receipt["evaluation"]["rows"]
    assert canonical_hash(selected_rows) == comparison["parity_rows_sha256"]
    assert [row["row_id"] for row in ensemble_rows] == [
        row["row_id"] for row in selected_rows
    ]
    assert [row["tape_sha256"] for row in ensemble_rows] == [
        row["tape_sha256"] for row in selected_rows
    ]

    both = []
    ensemble_only = []
    selected_only = []
    neither = []
    for ensemble_row, selected_row in zip(ensemble_rows, selected_rows, strict=True):
        ensemble_solved = ensemble_row["solved"]
        selected_solved = selected_row["onnx"]["solved"]
        if ensemble_solved and selected_solved:
            target = both
        elif ensemble_solved:
            target = ensemble_only
        elif selected_solved:
            target = selected_only
        else:
            target = neither
        target.append(ensemble_row["row_id"])

    assert comparison["solved_count"] == len(both) + len(selected_only) == 178
    assert (
        receipt["evaluation"]["solved_count"] == len(both) + len(ensemble_only) == 179
    )
    assert comparison["ensemble_minus_selected_solved_cases"] == 1
    assert comparison["matched_contingency"] == {
        "both_solved": len(both),
        "ensemble_only": len(ensemble_only),
        "selected_only": len(selected_only),
        "neither_solved": len(neither),
        "ensemble_only_row_ids": ensemble_only,
        "selected_only_row_ids": selected_only,
    }
    assert (len(both), len(ensemble_only), len(selected_only), len(neither)) == (
        176,
        3,
        2,
        19,
    )
    assert sum(row["onnx"]["hard_violation_count"] for row in selected_rows) == 0
    assert (
        max(row["onnx"]["maximum_conservation_residual"] for row in selected_rows)
        == 0.0
    )


def test_seed_ensemble_note_is_receipt_bound_and_keeps_deployment_distinct() -> None:
    receipt = _load(RECEIPT)
    note = NOTE.read_text(encoding="utf-8")

    assert f"SHA-256: `{file_sha256(RECEIPT)}`" in note
    assert f"Ensemble rows SHA-256: `{receipt['evaluation']['rows_sha256']}`" in note
    assert (
        "Ensemble-member identity SHA-256: "
        f"`{receipt['ensemble']['members_identity_sha256']}`"
    ) in note
    assert (
        "Matched selected-policy parity rows SHA-256: "
        f"`{receipt['comparison_to_selected_single_checkpoint']['parity_rows_sha256']}`"
    ) in note
    assert "**development-only exploratory candidate**" in note
    assert "**not promoted**" in note
    assert "not a final-split result" in note
    assert "| Five-seed action mean | **179/200** | **0.895** | 0 | 0.0 |" in note
    assert "| Selected single checkpoint | 178/200 | 0.890 | 0 | 0.0 |" in note
    assert "| 176 | 3 | 2 | 19 |" in note
    assert "does not strictly dominate" in note

    for member in receipt["ensemble"]["members"]:
        checkpoint = member["checkpoint_bundle"]
        assert (
            f"| {member['seed']} | `{checkpoint['checkpoint_id']}` | "
            f"`{checkpoint['obs_rms_sha256']}` |"
        ) in note

    for requirement in (
        "self-contained five-actor export",
        "all five frozen observation transforms",
        "SB3-to-ONNX action parity",
        "Exact full-development outcome parity",
        "new lightweight manifest",
        "Explicit application wiring",
    ):
        assert requirement in note
    assert (
        "shipped single-actor ONNX artifact and application wiring remain unchanged"
        in note
    )
    assert "No final case was constructed or evaluated" in note
