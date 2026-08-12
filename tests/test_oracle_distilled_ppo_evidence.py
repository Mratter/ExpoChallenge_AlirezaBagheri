"""Receipt-only integrity checks for the oracle-distilled PPO study."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any

import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import publish_oracle_distilled_ppo_evidence as publisher


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "oracle-distilled-ppo-study-200.json"
)
REPORT = ROOT / "benchmarks" / "v4" / "oracle-distilled-ppo-study-200.md"
FAMILY_IDS = (
    "v3_dev_river_flood",
    "v3_dev_industrial_outage",
    "v3_dev_logistics_strike",
    "v3_dev_seismic_cluster",
    "v3_dev_health_compound",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_portable_receipt_is_dev_only_complete_and_deterministic() -> None:
    receipt = _load(RECEIPT)
    expected_bytes = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert RECEIPT.read_bytes() == expected_bytes
    assert receipt["schema_version"] == (
        "city-recovery-oracle-distilled-ppo-dev-evidence-v1"
    )
    assert receipt["status"] == "complete_not_promoted"
    assert receipt["split"] == "dev"
    assert receipt["development_case_count"] == 200
    assert receipt["final_split_imported_or_used"] is False
    assert receipt["registered_policy_seeds"] == [37017, 47017, 57017]
    assert receipt["candidate_count"] == 9
    assert all(receipt["invariants"].values())
    assert receipt["source_contract"]["source_identity_sha256"] == canonical_hash(
        receipt["source_contract"]["source_files"]
    )
    for relative_path, expected_sha256 in receipt["source_contract"][
        "source_files"
    ].items():
        assert file_sha256(ROOT / relative_path) == expected_sha256
    incumbent = receipt["incumbent"]["five_seed_2m_endpoints"]
    assert incumbent == {
        "mean": 171.4,
        "population_std": 1.624807680927192,
        "sample_std": 1.816590212458495,
        "solved_counts": [172, 171, 171, 174, 169],
    }
    assert receipt["comparison"] == {
        "best_of_registered_challenger_vs_incumbent_best_of_20": {
            "challenger": 178,
            "delta": 0,
            "incumbent": 178,
        },
        "challenger_three_seed_mean_vs_incumbent_five_seed_mean": {
            "challenger": 174.0,
            "delta": 2.5999999999999943,
            "fairer_seed_level_comparison": True,
            "incumbent": 171.4,
        },
        "decisive_framing": "preregistered_conjunctive_promotion_rule",
    }


def test_all_nine_candidates_recompute_from_portable_rows() -> None:
    receipt = _load(RECEIPT)
    candidates = receipt["ranking"]["candidates"]
    assert len(candidates) == 9
    expected_identity: list[tuple[str, int, int, str]] | None = None
    for candidate in candidates:
        development = candidate["development"]
        rows = development["rows"]
        identity = [
            (
                row["row_id"],
                row["case_seed"],
                row["tape_seed"],
                row["tape_sha256"],
            )
            for row in rows
        ]
        if expected_identity is None:
            expected_identity = identity
        assert identity == expected_identity
        assert len(rows) == len(set(identity)) == 200
        assert len(development["source_rows_sha256"]) == 64
        assert canonical_hash(rows) == development["portable_rows_sha256"]
        # source_rows_sha256 binds the fuller external rows; this portable
        # projection independently reproduces every published claim.
        solved = sum(row["solved"] for row in rows)
        family = Counter(row["family_id"] for row in rows if row["solved"])
        reasons: Counter[str] = Counter()
        for row in rows:
            if not row["solved"]:
                reasons.update(row["reason_codes"])
        assert solved == development["solved_count"]
        assert development["solve_rate"] == solved / 200
        assert development["per_family_solved_count"] == {
            family_id: family[family_id] for family_id in FAMILY_IDS
        }
        assert development["failure_reason_code_histogram"] == dict(
            sorted(reasons.items())
        )
        assert sum(row["hard_violation_count"] for row in rows) == 0
        assert max(row["maximum_conservation_residual"] for row in rows) == 0.0
        assert round(fmean(row["resilience_auc"] for row in rows), 10) == (
            development["mean_resilience_auc"]
        )
        assert round(fmean(row["minimum_tail_margin"] for row in rows), 10) == (
            development["mean_minimum_tail_margin"]
        )

    recomputed_rank = sorted(
        candidates,
        key=lambda row: (
            -row["development"]["solved_count"],
            row["active_actor_critic_transitions"],
            row["policy_seed"],
        ),
    )
    assert [row["id"] for row in candidates] == [
        row["id"] for row in recomputed_rank
    ]
    assert candidates[0]["id"] == "seed-37017-ppo-2000000"
    assert candidates[0]["development"]["solved_count"] == 178


def test_endpoint_statistics_family_counts_and_promotion_recompute() -> None:
    receipt = _load(RECEIPT)
    endpoints = sorted(
        (
            row
            for row in receipt["ranking"]["candidates"]
            if row["active_actor_critic_transitions"] == 2_000_000
        ),
        key=lambda row: row["policy_seed"],
    )
    solved = [row["development"]["solved_count"] for row in endpoints]
    assert solved == [178, 174, 170]
    endpoint = receipt["endpoint_summary"]
    assert endpoint["mean_solved_count"] == fmean(solved) == 174.0
    assert math.isclose(
        endpoint["population_std_solved_count"], pstdev(solved), abs_tol=1e-12
    )
    assert endpoint["sample_std_solved_count"] == stdev(solved) == 4.0
    assert endpoint["mean_delta_vs_incumbent"] == fmean(solved) - 171.4
    assert endpoint["seed_count_at_or_above_172"] == 2
    for family_id in FAMILY_IDS:
        per_seed = {
            str(row["policy_seed"]): row["development"][
                "per_family_solved_count"
            ][family_id]
            for row in endpoints
        }
        assert endpoint["per_family"][family_id] == {
            "mean_solved_count": fmean(per_seed.values()),
            "solved_counts_by_seed": per_seed,
        }

    promotion = receipt["promotion"]
    assert promotion["passed"] is False
    assert promotion["decision"] == "complete_not_promoted"
    assert promotion["final_evaluation_run_or_authorized"] is False
    assert promotion["resilience_auc_used"] is False
    assert promotion["conditions"] == {
        "at_least_two_of_three_2m_endpoints_at_or_above_172": {
            "observed": 2,
            "passed": True,
            "threshold": 2,
        },
        "best_checkpoint_at_least_183_of_200_dev": {
            "observed": 178,
            "passed": False,
            "threshold": 183,
        },
        "three_seed_2m_mean_above_incumbent_171_4": {
            "observed": 174.0,
            "passed": True,
            "threshold_exclusive": 171.4,
        },
    }


def test_upstream_and_checkpoint_identities_are_pinned_portably() -> None:
    receipt = _load(RECEIPT)
    upstream = receipt["upstream_evidence"]
    assert upstream["oracle_training_dataset"] == {
        "action_count": 22,
        "case_count": 192,
        "dataset_index_sha256": (
            "cc4a2c3885cef77a3c161b681f729d2843b83a28e1ef21319208984d936dde14"
        ),
        "demonstration_row_count": 5760,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "observation_count": 73,
        "receipt_sha256": (
            "e7777e53f20b886bbb82b167e0303b20ee0de32dcf9b87f50d175a0b71c5dc89"
        ),
        "rows_sha256": (
            "a57bb67d74ebf0c78402ec3b45f5a59c1c2915b4c24ff586846f439a14adffea"
        ),
        "solved_count": 187,
        "split": "train",
    }
    student = upstream["oracle_bc_student"]
    assert student["development_solved_count"] == 157
    assert student["receipt_sha256"] == (
        "76025a6376db6905b1d96d08122a14bccc7639040921768a79e4c83debabec84"
    )
    assert student["actor_state_sha256"] == (
        "73b7a9097386f0ae772981056aa331216ac97c5877c1a57e322f06cc95e43601"
    )
    assert student["observation_rms_sha256"] == (
        "cb7b9a46369a0c225c3a6254433f6ef37e52b822ef44598fa4311b64e63a4ba4"
    )
    assert {row["training_receipt"]["path_within_external_study"] for row in receipt["seed_runs"]} == {
        "seed-37017/training-receipt.json",
        "seed-47017/training-receipt.json",
        "seed-57017/training-receipt.json",
    }
    assert len(
        {
            row["fresh_critic_state_sha256"]
            for row in receipt["seed_runs"]
        }
    ) == 3
    assert all(
        candidate["bundle"]["selection_evaluation_export_supported"] is True
        for candidate in receipt["ranking"]["candidates"]
    )


def test_report_is_receipt_bound_and_states_the_narrow_null() -> None:
    receipt = _load(RECEIPT)
    report = REPORT.read_text(encoding="utf-8")
    assert f"SHA-256: `{file_sha256(RECEIPT)}`" in report
    assert "**development-only post-release evidence**" in report
    assert "**complete—not promoted**" in report
    assert "**178, 174, and 170**" in report
    assert "population standard deviation **3.266**" in report
    assert "sample standard deviation **4.0**" in report
    assert "**+2.6 cases**" in report
    assert "zero DAgger iterations" in report
    assert "24 complete trajectories—720 action-labeled observations" in report
    assert "**0.0426693 / 0.148059**" in report
    assert "**0.0247503 / 0.0949414**" in report
    assert "nonoperative metadata" in report
    assert "does not resolve offline-policy distribution shift" in report
    assert "No final case was constructed or evaluated" in report
    assert receipt["method_disclosure"]["distribution_shift_resolved"] is False
    assert receipt["method_disclosure"][
        "legacy_generic_trainer_flow_label_is_nonoperative"
    ] is True
    assert receipt["method_disclosure"][
        "operative_zero_dagger_fields_are_authoritative"
    ] is True


def test_publisher_rejects_mutated_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = Path(r"E:\city-recovery-distilled-ppo-v4-attempt-02")
    if not external.is_dir():
        pytest.skip("external attempt-02 evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "protocol.json":
            value["contract"]["registered_policy_seeds"] = [37017, 47017]
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="contract"):
        publisher.build_portable_receipt(external)


def test_publisher_rejects_mutated_development_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = Path(r"E:\city-recovery-distilled-ppo-v4-attempt-02")
    if not external.is_dir():
        pytest.skip("external attempt-02 evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "training-receipt.json" and "seed-37017" in path.parts:
            value["development_curve"]["ppo_500000_transitions"]["rows"][0][
                "solved"
            ] = True
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="aggregate"):
        publisher.build_portable_receipt(external)


def test_publisher_rejects_mutated_bundle_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = Path(r"E:\city-recovery-distilled-ppo-v4-attempt-02")
    if not external.is_dir():
        pytest.skip("external attempt-02 evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if (
            path.name == "manifest.json"
            and "seed-47017" in path.parts
            and "ppo-1000000" in path.parts
        ):
            value["checkpoint"]["optimizer_state_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="bundle"):
        publisher.build_portable_receipt(external)
