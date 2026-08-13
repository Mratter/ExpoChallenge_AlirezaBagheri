"""Receipt-only integrity checks for moderate family reweighting."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any

import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import publish_moderate_family_evidence as publisher

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "moderate-family-study-200.json"
)
REPORT = ROOT / "benchmarks" / "v4" / "moderate-family-study-200.md"
FAMILIES = (
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


def _external_evidence() -> tuple[Path, Path]:
    if os.environ.get("INNOVERSE_TEST_EXTERNAL_EVIDENCE") != "1":
        pytest.skip(
            "set INNOVERSE_TEST_EXTERNAL_EVIDENCE=1 to validate machine-local study artifacts"
        )
    study_root = Path(r"E:\city-recovery-moderate-family-v4-attempt-01")
    difficulty = Path(r"E:\city-recovery-moderate-family-v4-difficulty-attempt-01.json")
    if not study_root.is_dir() or not difficulty.is_file():
        pytest.skip("external moderate-family evidence is unavailable")
    return study_root, difficulty


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
        "city-recovery-moderate-family-dev-evidence-v1"
    )
    assert receipt["status"] == "complete_not_promoted"
    assert receipt["split"] == "dev"
    assert receipt["development_case_count"] == 200
    assert receipt["candidate_count"] == 9
    assert receipt["registered_policy_seeds"] == [37017, 47017, 57017]
    assert receipt["final_split_imported_or_used"] is False
    assert all(receipt["invariants"].values())
    for contract_name in ("source_contract", "publication_source_contract"):
        contract = receipt[contract_name]
        assert contract["source_identity_sha256"] == canonical_hash(
            contract["source_files"]
        )
        for path, expected_sha in contract["source_files"].items():
            assert file_sha256(ROOT / path) == expected_sha


def test_receipt_contains_no_machine_local_absolute_paths() -> None:
    receipt = _load(RECEIPT)

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield from strings(key)
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    leaked = [
        value
        for value in strings(receipt)
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\")
    ]
    assert leaked == []


def test_training_difficulty_ranking_weights_and_confound_are_bound() -> None:
    receipt = _load(RECEIPT)
    difficulty = receipt["difficulty_evidence"]
    assert difficulty["split"] == "train"
    assert difficulty["case_count"] == 192
    assert difficulty["shipped_policy_solved_count"] == 186
    assert difficulty["contextual_tuned_rule"]["solved_count"] == 180
    assert canonical_hash(difficulty["rows"]) == difficulty["rows_sha256"]
    assert canonical_hash(difficulty["contextual_tuned_rule"]["rows"]) == (
        difficulty["contextual_tuned_rule"]["rows_sha256"]
    )
    assert difficulty["ranking"]["ranked_family_ids"][:2] == [
        "v3_train_grid_failure",
        "v3_train_displacement",
    ]
    assert difficulty["contextual_tuned_rule"]["ranked_family_ids"][:2] == [
        "v3_train_grid_failure",
        "v3_train_weather_isolation",
    ]
    assert difficulty["sampler"]["family_weights"] == {
        "v3_train_displacement": 2,
        "v3_train_grid_failure": 2,
        "v3_train_health_surge": 1,
        "v3_train_supply_chain": 1,
        "v3_train_transit_nexus": 1,
        "v3_train_weather_isolation": 1,
    }
    assert difficulty["sampler"]["weighted_cycle_case_count"] == 256
    assert difficulty["sampler"]["canonical_case_count"] == 192
    assert difficulty["access_contract"]["development_split_used"] is False
    assert difficulty["access_contract"]["final_split_used"] is False
    scope = receipt["interpretation_scope"]
    assert scope["pure_fixed_volume_importance_reweighting"] is False
    assert scope["imitation_observation_count_treatment"] == 30720
    assert scope["imitation_observation_count_incumbent"] == 23040
    assert scope["imitation_observation_exposure_multiplier"] == 4 / 3
    assert scope["observation_rms_count_treatment"] == 30720.0001
    assert scope["observation_rms_count_incumbent"] == 23040.0001


def test_nine_selectable_candidates_recompute_from_portable_rows() -> None:
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
        assert canonical_hash(rows) == development["portable_rows_sha256"]
        assert len(development["source_rows_sha256"]) == 64
        solved = sum(row["solved"] for row in rows)
        family = Counter(row["family_id"] for row in rows if row["solved"])
        reasons: Counter[str] = Counter()
        for row in rows:
            if not row["solved"]:
                reasons.update(row["reason_codes"])
        assert solved == development["solved_count"]
        assert development["solve_rate"] == solved / 200
        assert development["per_family_solved_count"] == {
            family_id: family[family_id] for family_id in FAMILIES
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
    assert candidates == sorted(
        candidates,
        key=lambda row: (
            -row["development"]["solved_count"],
            row["active_actor_critic_transitions"],
            row["policy_seed"],
        ),
    )
    assert [row["development"]["solved_count"] for row in candidates] == [
        176,
        175,
        172,
        171,
        170,
        168,
        166,
        166,
        165,
    ]


def test_all_six_stage_curves_and_12_bundles_are_retained() -> None:
    receipt = _load(RECEIPT)
    expected = {
        37017: [156, 156, 161, 171, 176, 175],
        47017: [151, 151, 162, 165, 168, 170],
        57017: [153, 153, 158, 166, 166, 172],
    }
    bundle_count = 0
    for run in receipt["study_runs"]:
        assert [row["solved_count"] for row in run["development_curve"]] == (
            expected[run["policy_seed"]]
        )
        assert [row["phase"] for row in run["development_curve"]] == [
            "bc_initialization",
            "post_critic_warmup",
            "ppo_200000",
            "ppo_500000",
            "ppo_1000000",
            "ppo_2000000",
        ]
        assert run["critic_warmup_transitions"] == 50000
        assert run["training_receipt"]["path_within_external_study"] == (
            f"seed-{run['policy_seed']}/training-receipt.json"
        )
        assert canonical_hash(run["training_config"]) == run[
            "portable_training_config_sha256"
        ]
        sampler = run["training_config"]["training_family_sampler"]
        assert "selection_evidence_path" not in sampler
        assert sampler["selection_evidence"] == {
            "portable_receipt_section": "difficulty_evidence",
            "source_receipt_sha256": publisher.EXPECTED_DIFFICULTY_SHA256,
        }
        for row in run["development_curve"]:
            if "bundle" in row:
                bundle_count += 1
                assert row["bundle"]["selection_evaluation_export_supported"] is True
                assert row["bundle"]["training_config_sha256"] == run[
                    "training_config_sha256"
                ]
                assert all(
                    len(row["bundle"][field]) == 64
                    for field in (
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
    assert bundle_count == 12
    incumbent_by_seed = {
        row["policy_seed"]: row for row in receipt["incumbent"]["matched_seed_curves"]
    }
    for run in receipt["study_runs"]:
        comparable = {
            key: value
            for key, value in run["training_config"].items()
            if key != "training_family_sampler"
        }
        assert comparable == incumbent_by_seed[run["policy_seed"]][
            "training_config"
        ]
        assert [
            row["solved_count"]
            for row in incumbent_by_seed[run["policy_seed"]]["development_curve"]
        ] in ([167, 170, 172], [166, 170, 171], [167, 170, 171])


def test_endpoint_statistics_gate_and_comparisons_recompute() -> None:
    receipt = _load(RECEIPT)
    endpoint = receipt["endpoint_summary"]
    counts = [175, 170, 172]
    assert list(endpoint["solved_counts_by_seed"].values()) == counts
    assert endpoint["mean_solved_count"] == fmean(counts)
    assert math.isclose(
        endpoint["population_std_solved_count"], pstdev(counts), abs_tol=1e-12
    )
    assert math.isclose(
        endpoint["sample_std_solved_count"], stdev(counts), abs_tol=1e-12
    )
    assert endpoint["seed_count_at_or_above_172"] == 2
    promotion = receipt["promotion"]
    assert promotion["passed"] is False
    assert promotion["decision"] == "retain_shipped_policy"
    assert promotion["final_evaluation_authorized"] is False
    assert promotion["passed"] == all(
        condition["passed"] for condition in promotion["conditions"].values()
    )
    assert promotion["conditions"]["selected_checkpoint_solved_count"] == {
        "observed": 176,
        "operator": ">=",
        "passed": False,
        "threshold": 183,
    }
    comparison = receipt["comparison"]
    assert comparison["best_registered_challenger_vs_incumbent_best_of_20"] == {
        "challenger": 176,
        "delta": -2,
        "incumbent": 178,
        "selection_asymmetric": True,
        "selection_candidate_counts": {"challenger": 9, "incumbent": 20},
    }
    assert comparison["matched_same_seed_2m"]["deltas"] == [3, -1, 1]
    assert comparison["matched_same_seed_2m"]["mean_delta"] == 1.0
    assert comparison["matched_same_seed_2m"][
        "fairer_matched_seed_comparison"
    ] is True
    assert comparison["matched_same_seed_2m"]["pooled_case_pairing"] == {
        "both_solved": 504,
        "challenger_only": 13,
        "incumbent_only": 10,
        "neither_solved": 73,
    }
    selected_pairing = comparison[
        "best_registered_challenger_vs_shipped_selected_case_pairing"
    ]
    assert {
        key: selected_pairing[key]
        for key in (
            "both_solved",
            "challenger_only",
            "neither_solved",
            "shipped_only",
        )
    } == {
        "both_solved": 175,
        "challenger_only": 1,
        "neither_solved": 21,
        "shipped_only": 3,
    }
    assert selected_pairing["family_solved_counts"] == {
        "v3_dev_health_compound": {"challenger": 36, "delta": 1, "shipped": 35},
        "v3_dev_industrial_outage": {"challenger": 38, "delta": 0, "shipped": 38},
        "v3_dev_logistics_strike": {"challenger": 40, "delta": 0, "shipped": 40},
        "v3_dev_river_flood": {"challenger": 33, "delta": -1, "shipped": 34},
        "v3_dev_seismic_cluster": {"challenger": 29, "delta": -2, "shipped": 31},
    }
    assert comparison["matched_same_seed_curve"] == {
        "500000": {
            "challenger_mean": 167.33333333333334,
            "challenger_solved_counts": [171, 165, 166],
            "incumbent_mean": 166.66666666666666,
            "incumbent_solved_counts": [167, 166, 167],
            "mean_delta": 0.6666666666666856,
        },
        "1000000": {
            "challenger_mean": 170.0,
            "challenger_solved_counts": [176, 168, 166],
            "incumbent_mean": 170.0,
            "incumbent_solved_counts": [170, 170, 170],
            "mean_delta": 0.0,
        },
        "2000000": {
            "challenger_mean": 172.33333333333334,
            "challenger_solved_counts": [175, 170, 172],
            "incumbent_mean": 171.33333333333334,
            "incumbent_solved_counts": [172, 171, 171],
            "mean_delta": 1.0,
        },
    }


def test_matched_family_deltas_and_case_discordance_are_exact() -> None:
    receipt = _load(RECEIPT)
    comparison = receipt["comparison"]["per_family_matched_same_seed_2m"]
    expected = {
        "v3_dev_river_flood": ([1, 0, 1], 4, 2),
        "v3_dev_industrial_outage": ([0, -1, 2], 3, 2),
        "v3_dev_logistics_strike": ([0, 0, 0], 0, 0),
        "v3_dev_seismic_cluster": ([3, 0, -1], 6, 4),
        "v3_dev_health_compound": ([-1, 0, -1], 0, 2),
    }
    for family_id, (deltas, challenger_only, incumbent_only) in expected.items():
        row = comparison[family_id]
        assert [
            challenger - incumbent
            for challenger, incumbent in zip(
                row["challenger_counts"],
                row["matched_incumbent_counts"],
                strict=True,
            )
        ] == deltas
        assert math.isclose(
            row["matched_mean_delta"], fmean(deltas), abs_tol=1e-12
        )
        assert row["pooled_case_pairing"]["challenger_only"] == challenger_only
        assert row["pooled_case_pairing"]["incumbent_only"] == incumbent_only


def test_external_hash_rosters_and_report_disclosures() -> None:
    receipt = _load(RECEIPT)
    assert receipt["external_evidence"] == {
        "attempt_id": "city-recovery-moderate-family-v4-attempt-01",
        "difficulty_receipt": {
            "sha256": (
                "27d4b675273ebdfabc7ec5f6546a2d4c75ec5774e024c9d0c57484f800e4e5d4"
            ),
            "size_bytes": 286168,
        },
        "protocol": {
            "sha256": (
                "4cc902fdee9e090df0be6042ccb5f2953eadde9693867e553f26b61ca8c65ad7"
            ),
            "size_bytes": 2297,
        },
        "summary": {
            "sha256": (
                "935a0069d3c1eb53885e4ff5843ec5545eef4277a4a73ff6376a9948ea64e8a0"
            ),
            "size_bytes": 61536,
        },
        "training_receipts_sha256_by_seed": {
            "37017": (
                "159d433a2b876d9009a1886b87360f7d3c1f91d0e00a59501587f9eafd44482b"
            ),
            "47017": (
                "692ead754e1471ec84fdc4bc2fe91baeb600e521d88307a9d563cfba1121bf0c"
            ),
            "57017": (
                "7091811f846273ff24719280f01f044fa22754b4a0a59bf2707d9e153ead228f"
            ),
        },
    }
    report = REPORT.read_text(encoding="utf-8")
    assert f"SHA-256: `{file_sha256(RECEIPT)}`" in report
    assert "**development-only post-release evidence**" in report
    assert "**complete—not promoted**" in report
    assert "**175, 170, and 172 / 200**" in report
    assert "**30,720**" in report and "**23,040**" in report
    assert "not pure fixed-volume importance weighting" in report
    assert "**504 were solved by both, 13 by" in report
    assert "no robust\ntargeted weak-family improvement" in report
    assert "No final case was constructed or evaluated" in report


def test_publisher_rejects_mutated_difficulty(monkeypatch: pytest.MonkeyPatch) -> None:
    study_root, difficulty = _external_evidence()
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "protocol.json":
            value["difficulty_receipt_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="difficulty|protocol"):
        publisher.build_portable_receipt(study_root, difficulty)


def test_publisher_rejects_mutated_development_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, difficulty = _external_evidence()
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "training-receipt.json" and "seed-37017" in path.parts:
            value["development_curve"]["ppo_500000_transitions"]["rows"][0][
                "solved"
            ] = True
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError):
        publisher.build_portable_receipt(study_root, difficulty)


def test_publisher_rejects_mutated_bundle_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, difficulty = _external_evidence()
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if (
            path.name == "manifest.json"
            and "seed-47017" in path.parts
            and "ppo-200000" in path.parts
        ):
            value["checkpoint"]["optimizer_state_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="checkpoint"):
        publisher.build_portable_receipt(study_root, difficulty)


def test_publisher_rejects_bundle_config_not_bound_to_parent_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, difficulty = _external_evidence()
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if (
            path.name == "manifest.json"
            and "seed-57017" in path.parts
            and "ppo-1000000" in path.parts
        ):
            value["training"]["config"]["learning_rate"] = 3e-5
            value["training"]["config_sha256"] = canonical_hash(
                value["training"]["config"]
            )
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="checkpoint"):
        publisher.build_portable_receipt(study_root, difficulty)


def test_publisher_rejects_mutated_shipped_selected_receipt_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root, difficulty = _external_evidence()
    original = publisher._load

    def mutated(path: Path, label: str) -> dict[str, Any]:
        value = original(path, label)
        if path.name == "checkpoint-selection-200.json":
            value["selected_checkpoint"]["training_receipt_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(publisher, "_load", mutated)
    with pytest.raises(publisher.PublicationError, match="shipped selected receipt"):
        publisher.build_portable_receipt(study_root, difficulty)
