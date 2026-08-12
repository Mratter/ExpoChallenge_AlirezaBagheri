"""Receipt-only integrity checks for the network-capacity study."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any

import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import publish_network_capacity_evidence as publisher

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "network-capacity-study-200.json"
)
REPORT = ROOT / "benchmarks" / "v4" / "network-capacity-study-200.md"
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


def test_receipt_is_canonical_complete_dev_only_and_source_bound() -> None:
    receipt = _load(RECEIPT)
    expected = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert RECEIPT.read_bytes() == expected
    assert receipt["schema_version"] == (
        "city-recovery-network-capacity-dev-evidence-v1"
    )
    assert receipt["status"] == "complete_not_promoted"
    assert receipt["split"] == "dev"
    assert receipt["development_case_count"] == 200
    assert receipt["candidate_count"] == 18
    assert receipt["registered_policy_seeds"] == [37017, 47017, 57017]
    assert receipt["final_split_imported_or_used"] is False
    assert all(receipt["invariants"].values())
    source = receipt["source_contract"]
    assert source["source_identity_sha256"] == canonical_hash(
        source["source_files"]
    )
    for path, expected_sha in source["source_files"].items():
        assert file_sha256(ROOT / path) == expected_sha
    publication = receipt["publication_source_contract"]
    assert publication["source_identity_sha256"] == canonical_hash(
        publication["source_files"]
    )
    for path, expected_sha in publication["source_files"].items():
        assert file_sha256(ROOT / path) == expected_sha


def test_architecture_parameter_semantics_and_incumbent_curves() -> None:
    receipt = _load(RECEIPT)
    architecture = receipt["architecture"]
    assert architecture["actor_hidden_layers"] == [768, 512, 256]
    assert architecture["critic_hidden_layers"] == [768, 512, 256]
    assert architecture["activation"] == "SiLU"
    assert architecture["public_interface"] == {
        "action_count": 22,
        "observation_count": 73,
    }
    assert architecture["parameter_counts"] == {
        "actor": 587564,
        "critic": 582145,
        "total_policy": 1169709,
    }
    assert architecture["actor_parameter_semantics"] == {
        "deterministic_mean_path": 587542,
        "learned_log_standard_deviation": 22,
        "total_actor": 587564,
    }
    assert architecture["incumbent_parameter_counts"] == {
        "actor": 162732,
        "actor_deterministic_mean_path": 162710,
        "critic": 160001,
        "learned_log_standard_deviation": 22,
        "total_policy": 322733,
    }
    assert architecture["smaller_optional_arm_run"] is False

    incumbent = receipt["incumbent"]
    assert incumbent["five_seed_2m_endpoints"] == {
        "mean": 171.4,
        "population_std": 1.624807680927192,
        "sample_std": 1.816590212458495,
        "solved_counts": [172, 171, 171, 174, 169],
    }
    curves = incumbent["five_seed_curves"]
    assert len(curves) == 5
    means = {
        str(milestone): fmean(
            next(
                row["solved_count"]
                for row in curve["curve"]
                if row["active_actor_critic_transitions"] == milestone
            )
            for curve in curves
        )
        for milestone in (500_000, 1_000_000, 2_000_000)
    }
    assert means == incumbent["five_seed_mean_solved_count_by_milestone"] == {
        "500000": 168.2,
        "1000000": 171.4,
        "2000000": 171.4,
    }


def test_all_18_candidates_recompute_from_exact_portable_rows() -> None:
    receipt = _load(RECEIPT)
    candidates = receipt["ranking"]["candidates"]
    assert len(candidates) == 18
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
        assert canonical_hash(rows) == development["portable_rows_sha256"]
        assert len(development["source_rows_sha256"]) == 64
        solved = sum(row["solved"] for row in rows)
        families = Counter(row["family_id"] for row in rows if row["solved"])
        reasons: Counter[str] = Counter()
        for row in rows:
            if not row["solved"]:
                reasons.update(row["reason_codes"])
        assert solved == development["solved_count"]
        assert development["solve_rate"] == solved / 200
        assert development["per_family_solved_count"] == {
            family_id: families[family_id] for family_id in FAMILY_IDS
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
        bundle = candidate["bundle"]
        assert bundle["selection_evaluation_export_supported"] is True
        assert all(
            len(bundle[name]) == 64
            for name in (
                "actor_state_sha256",
                "manifest_sha256",
                "model_sha256",
                "normalization_sha256",
                "observation_rms_sha256",
                "optimizer_state_sha256",
                "policy_state_sha256",
                "return_rms_sha256",
            )
        )

    arm_order = {"large_lr_7_5e_5": 0, "large_lr_3e_5": 1}
    assert candidates == sorted(
        candidates,
        key=lambda row: (
            -row["development"]["solved_count"],
            row["active_actor_critic_transitions"],
            arm_order[row["arm_id"]],
            row["policy_seed"],
        ),
    )
    assert candidates[0]["arm_id"] == "large_lr_3e_5"
    assert candidates[0]["policy_seed"] == 37017
    assert candidates[0]["development"]["solved_count"] == 178


def test_six_curves_endpoint_statistics_and_family_aggregates_recompute() -> None:
    receipt = _load(RECEIPT)
    assert len(receipt["study_runs"]) == 6
    expected_curves = {
        ("large_lr_7_5e_5", 37017): [152, 152, 160, 166, 175, 171],
        ("large_lr_7_5e_5", 47017): [153, 153, 161, 167, 171, 169],
        ("large_lr_7_5e_5", 57017): [155, 155, 161, 170, 171, 172],
        ("large_lr_3e_5", 37017): [152, 152, 154, 164, 173, 178],
        ("large_lr_3e_5", 47017): [153, 153, 157, 164, 173, 176],
        ("large_lr_3e_5", 57017): [155, 155, 157, 170, 174, 175],
    }
    for run in receipt["study_runs"]:
        key = (run["arm_id"], run["policy_seed"])
        assert [row["solved_count"] for row in run["development_curve"]] == (
            expected_curves[key]
        )
        assert run["training_receipt"]["path_within_external_study"] == (
            f"{run['arm_id']}/seed-{run['policy_seed']}/training-receipt.json"
        )

    expected_endpoints = {
        "large_lr_7_5e_5": [171, 169, 172],
        "large_lr_3e_5": [178, 176, 175],
    }
    for arm_id, counts in expected_endpoints.items():
        endpoint = receipt["arm_endpoint_summaries"][arm_id]
        assert list(endpoint["solved_counts_by_seed"].values()) == counts
        assert endpoint["mean_solved_count"] == fmean(counts)
        assert math.isclose(
            endpoint["population_std_solved_count"], pstdev(counts), abs_tol=1e-12
        )
        assert math.isclose(
            endpoint["sample_std_solved_count"], stdev(counts), abs_tol=1e-12
        )
        candidates = [
            row
            for row in receipt["ranking"]["candidates"]
            if row["arm_id"] == arm_id
            and row["active_actor_critic_transitions"] == 2_000_000
        ]
        for family_id in FAMILY_IDS:
            per_seed = {
                str(row["policy_seed"]): row["development"][
                    "per_family_solved_count"
                ][family_id]
                for row in candidates
            }
            assert endpoint["per_family"][family_id] == {
                "mean_solved_count": fmean(per_seed.values()),
                "solved_counts_by_seed": per_seed,
            }


def test_curve_analysis_comparisons_and_promotion_recompute() -> None:
    receipt = _load(RECEIPT)
    assert receipt["curve_analysis"] == {
        "large_lr_3e_5": {
            "all_three_seeds_increased_from_1m_to_2m": True,
            "mean_1m_solved_count": 173.33333333333334,
            "mean_2m_solved_count": 176.33333333333334,
            "mean_delta_1m_to_2m": 3.0,
            "per_seed_delta_1m_to_2m": {"37017": 5, "47017": 3, "57017": 1},
        },
        "large_lr_7_5e_5": {
            "all_three_seeds_increased_from_1m_to_2m": False,
            "mean_1m_solved_count": 172.33333333333334,
            "mean_2m_solved_count": 170.66666666666666,
            "mean_delta_1m_to_2m": -1.6666666666666856,
            "per_seed_delta_1m_to_2m": {"37017": -4, "47017": -2, "57017": 1},
        },
    }
    assert receipt["comparison"] == {
        "best_of_registered_challenger_vs_incumbent_best_of_20": {
            "challenger": 178,
            "delta": 0,
            "incumbent": 178,
        },
        "decisive_framing": "preregistered_conjunctive_promotion_rule",
        "large_lr_3e_5_vs_incumbent_same_seed_2m": {
            "challenger_solved_counts": [178, 176, 175],
            "deltas": [6, 5, 4],
            "incumbent_solved_counts": [172, 171, 171],
            "policy_seeds": [37017, 47017, 57017],
        },
        "selected_arm_three_seed_mean_vs_incumbent_five_seed_mean": {
            "challenger": 176.33333333333334,
            "delta": 4.933333333333337,
            "fairer_seed_level_comparison": True,
            "incumbent": 171.4,
        },
    }
    promotion = receipt["promotion"]
    assert promotion["passed"] is False
    assert promotion["decision"] == "complete_not_promoted"
    assert promotion["final_evaluation_run_or_authorized"] is False
    assert promotion["resilience_auc_used"] is False
    assert promotion["passed"] == all(
        row["passed"] for row in promotion["conditions"].values()
    )
    assert promotion["conditions"] == {
        "selected_arm_at_least_two_seed_endpoints_at_or_above_172": {
            "observed": 3,
            "passed": True,
            "threshold": 2,
        },
        "selected_arm_three_seed_2m_mean_above_171_4": {
            "observed": 176.33333333333334,
            "passed": True,
            "threshold_exclusive": 171.4,
        },
        "selected_checkpoint_at_least_183": {
            "observed": 178,
            "passed": False,
            "threshold": 183,
        },
    }


def test_pairing_and_external_hash_rosters_are_complete() -> None:
    receipt = _load(RECEIPT)
    assert len(receipt["paired_learning_rate_checks"]) == 3
    assert [row["policy_seed"] for row in receipt["paired_learning_rate_checks"]] == [
        37017,
        47017,
        57017,
    ]
    assert all(
        row["only_registered_config_difference_is_learning_rate"] is True
        for row in receipt["paired_learning_rate_checks"]
    )
    hashes = receipt["external_evidence"]["training_receipts_sha256"]
    assert hashes == {
        "large_lr_3e_5/seed-37017": (
            "cf0428b3dd4f50873549b51c18324014fb70273860dd8109cf7fea95a0176848"
        ),
        "large_lr_3e_5/seed-47017": (
            "b537c9f483bdb02f448054e04ede7843f433578159c2458af92af6cf3d90f0df"
        ),
        "large_lr_3e_5/seed-57017": (
            "d973b21a5c24c142f4cc026808ef311be86a2d4ade3b61c6ad7fda19b6963427"
        ),
        "large_lr_7_5e_5/seed-37017": (
            "3437ecfca07fc0d5db468b27858fccd62e3fcc9a6c360919f122e0f397e7004b"
        ),
        "large_lr_7_5e_5/seed-47017": (
            "48ea18a1dfa000a784451eeb685b782b287f102ba0d289948a5faddb6d1593cf"
        ),
        "large_lr_7_5e_5/seed-57017": (
            "6a76fa8aad678f79faf63dabfa97fc409389ef01e23aad56fdd487b37bfefcaa"
        ),
    }


def test_report_is_bound_and_scientifically_narrow() -> None:
    receipt = _load(RECEIPT)
    report = REPORT.read_text(encoding="utf-8")
    assert f"SHA-256: `{file_sha256(RECEIPT)}`" in report
    assert "**development-only post-release evidence**" in report
    assert "**complete—not promoted**" in report
    assert "**176.333 / 200**" in report
    assert "**+4.933 cases**" in report
    assert "**+5, +3, and +1**" in report
    assert "exactly **+6, +5, and +4**" in report
    assert "optional smaller control arm was not run" in report
    assert "does not isolate capacity alone" in report
    assert "not a generic claim that capacity is irrelevant" in report
    assert "longer-budget behavior explicitly unresolved" in report
    assert "No final case was constructed or evaluated" in report
    assert "scope is exactly `[768, 512, 256]`, these two learning rates" in report
    assert receipt["null_scope"].startswith(
        "A non-promotion result applies only to [768,512,256]"
    )


def test_publisher_rejects_mutated_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    external = Path(r"E:\city-recovery-large-architecture-v4-attempt-01")
    if not external.is_dir():
        pytest.skip("external capacity evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "protocol.json":
            value["contract"]["registered_policy_seeds"] = [37017, 47017]
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="protocol|contract"):
        publisher.build_portable_receipt(external)


def test_publisher_rejects_mutated_development_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = Path(r"E:\city-recovery-large-architecture-v4-attempt-01")
    if not external.is_dir():
        pytest.skip("external capacity evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if (
            path.name == "training-receipt.json"
            and "large_lr_3e_5" in path.parts
            and "seed-37017" in path.parts
        ):
            value["development_curve"]["ppo_500000_transitions"]["rows"][0][
                "solved"
            ] = True
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError):
        publisher.build_portable_receipt(external)


def test_publisher_rejects_mutated_bundle_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = Path(r"E:\city-recovery-large-architecture-v4-attempt-01")
    if not external.is_dir():
        pytest.skip("external capacity evidence is unavailable")
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if (
            path.name == "manifest.json"
            and "large_lr_3e_5" in path.parts
            and "seed-47017" in path.parts
            and "ppo-1000000" in path.parts
        ):
            value["checkpoint"]["optimizer_state_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="bundle"):
        publisher.build_portable_receipt(external)
