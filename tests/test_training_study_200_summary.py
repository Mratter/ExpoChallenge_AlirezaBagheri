from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "training-study-200-summary.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_portable_paths(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "path" and isinstance(child, str):
                assert not Path(child).is_absolute()
                assert not (
                    len(child) >= 3
                    and child[1] == ":"
                    and child[2] in ("/", "\\")
                )
            _assert_portable_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_portable_paths(child)


def _assert_aggregate(endpoints: list[dict], aggregate: dict) -> None:
    solved = [row["solved_count"] for row in endpoints]
    rates = [row["solve_rate"] for row in endpoints]
    assert aggregate["seed_count"] == len(endpoints)
    assert math.isclose(aggregate["mean_solved_count"], fmean(solved))
    assert math.isclose(aggregate["sample_std_solved_count"], stdev(solved))
    assert math.isclose(aggregate["mean_solve_rate"], fmean(rates))
    assert math.isclose(aggregate["sample_std_solve_rate"], stdev(rates))
    assert aggregate["minimum_solved_count"] == min(solved)
    assert aggregate["maximum_solved_count"] == max(solved)
    assert all(row["solve_rate"] == row["solved_count"] / 200 for row in endpoints)


def test_training_study_summary_is_dev_only_and_arithmetically_consistent() -> None:
    summary = _load(SUMMARY_PATH)
    assert summary["scope"] == {
        "split": "dev",
        "development_case_count": 200,
        "training_roster_case_count": 192,
        "final_split_used": False,
        "statement": (
            "Development-only training, ablation, selection, and SB3-to-ONNX "
            "parity evidence. This is not a final-split result."
        ),
    }

    baseline = summary["baseline"]
    assert [row["seed"] for row in baseline["endpoints"]] == [
        37017,
        47017,
        57017,
        67017,
        77017,
    ]
    assert [row["solved_count"] for row in baseline["endpoints"]] == [
        172,
        171,
        171,
        174,
        169,
    ]
    _assert_aggregate(baseline["endpoints"], baseline["aggregate"])

    controls = dict(
        zip(
            summary["matched_ablation_control"]["seeds"],
            summary["matched_ablation_control"]["solved_counts"],
            strict=True,
        )
    )
    expected = {
        "no_bc_warm_start": ([145, 156, 151], [-27, -15, -20]),
        "risk_averse_reward": ([173, 171, 177], [1, 0, 6]),
        "no_vec_normalize": ([140, 134, 144], [-32, -37, -27]),
        "preparedness_alignment_2": ([169, 170, 173], [-3, -1, 2]),
        "budget_645k": ([170, 169, 168], [-2, -2, -3]),
    }
    assert {row["name"] for row in summary["ablations"]} == set(expected)
    for ablation in summary["ablations"]:
        solved, deltas = expected[ablation["name"]]
        assert [row["solved_count"] for row in ablation["endpoints"]] == solved
        assert [row["delta"] for row in ablation["paired_solved_deltas"]] == deltas
        for endpoint, pair in zip(
            ablation["endpoints"],
            ablation["paired_solved_deltas"],
            strict=True,
        ):
            assert pair["seed"] == endpoint["seed"]
            assert pair["control"] == controls[endpoint["seed"]]
            assert pair["treatment"] == endpoint["solved_count"]
            assert pair["delta"] == pair["treatment"] - pair["control"]
        assert math.isclose(
            ablation["mean_treatment_minus_control_solved"],
            fmean(deltas),
        )
        _assert_aggregate(ablation["endpoints"], ablation["aggregate"])

    risk = next(
        row for row in summary["ablations"] if row["name"] == "risk_averse_reward"
    )
    assert risk["coefficient_tuning_performed"] is False
    assert summary["endpoint_invariants"] == {
        "training_receipt_count": 20,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
    }


def test_training_study_summary_binds_selection_and_onnx_evidence() -> None:
    summary = _load(SUMMARY_PATH)
    selection_record = summary["selection"]["receipt"]
    parity_record = summary["publication"]["parity_receipt"]
    raw_manifest_record = summary["publication"]["raw_export_manifest"]
    portable_manifest_record = summary["publication"]["manifest"]
    artifact_record = summary["publication"]["artifact"]

    for record in (
        selection_record,
        parity_record,
        raw_manifest_record,
        artifact_record,
    ):
        path = ROOT / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert _sha256(path) == record["sha256"]

    selection = _load(ROOT / selection_record["path"])
    selected = summary["selection"]["selected_checkpoint"]
    assert selection["split"] == "dev"
    assert selection["final_split_used"] is False
    assert selection["candidate_count"] == summary["selection"]["candidate_count"]
    assert selection["ranking"]["primary_metric"] == "solved_count"
    assert selection["ranking"]["resilience_auc_used_for_selection"] is False
    assert selection["selected_checkpoint"]["id"] == selected["id"]
    assert selection["selected_checkpoint"]["sha256"] == selected[
        "checkpoint_sha256"
    ]
    assert selection["selected_checkpoint"]["normalization_file_sha256"] == selected[
        "normalization_file_sha256"
    ]
    assert selection["selected_checkpoint"]["observation_rms_sha256"] == selected[
        "observation_rms_sha256"
    ]
    assert selection["winner"]["solved_count"] == selected["solved_count"] == 178
    assert selection["runner_up"]["solved_count"] == 174
    assert selection["candidates"][1]["id"] == summary["selection"]["runner_up"][
        "id"
    ]
    assert selection["margin"] == summary["selection"]["margin"]
    assert selection["tie_break"]["used"] is summary["selection"][
        "tie_break_used"
    ]
    candidate = next(
        row for row in selection["candidates"] if row["id"] == selected["id"]
    )
    assert candidate["bundle_manifest_sha256"] == selected[
        "bundle_manifest_sha256"
    ]
    assert candidate["development"]["solved_count"] == selected["solved_count"]
    assert selection["source_seed_sweep_summary"]["sha256"] == summary[
        "source_evidence"
    ]["seed_sweep_summary"]["sha256"]
    for path_key, hash_key in (
        ("path", "checkpoint_sha256"),
        ("normalization_path", "normalization_file_sha256"),
    ):
        external = Path(selection["selected_checkpoint"][path_key])
        if external.exists():
            assert _sha256(external) == selected[hash_key]
    for path_key, hash_key in (
        ("bundle_manifest_path", "bundle_manifest_sha256"),
        ("training_receipt_path", "training_receipt_sha256"),
    ):
        external = Path(candidate[path_key])
        if external.exists():
            assert _sha256(external) == selected[hash_key]

    parity = _load(ROOT / parity_record["path"])
    raw_manifest = _load(ROOT / raw_manifest_record["path"])
    portable_manifest = _load(ROOT / portable_manifest_record["path"])
    facts = summary["publication"]["development_parity"]
    assert parity["split"] == "dev"
    assert parity["final_split_used"] is False
    assert parity["parity"]["passed"] is True
    assert parity["source_checkpoint"]["id"] == selected["id"]
    assert parity["source_checkpoint"]["sha256"] == selected[
        "checkpoint_sha256"
    ]
    assert parity["normalization"]["file_sha256"] == selected[
        "normalization_file_sha256"
    ]
    assert parity["onnx_artifact"]["sha256"] == artifact_record["sha256"]
    for key, value in facts.items():
        assert parity["parity"][key] == value
    assert raw_manifest["publication_status"] == "development_parity_passed"
    assert raw_manifest["artifact"]["sha256"] == artifact_record["sha256"]
    assert raw_manifest["parity_receipt"]["sha256"] == parity_record["sha256"]

    assert portable_manifest["scope"]["final_split_used"] is False
    _assert_portable_paths(portable_manifest)
    assert portable_manifest["derived_from"]["raw_export_manifest"] == {
        key: raw_manifest_record[key] for key in ("path", "sha256", "size_bytes")
    }
    assert portable_manifest["derived_from"]["selection_receipt"] == {
        key: selection_record[key] for key in ("path", "sha256", "size_bytes")
    }
    assert portable_manifest["derived_from"]["parity_receipt"] == {
        key: parity_record[key] for key in ("path", "sha256", "size_bytes")
    }
    assert portable_manifest["derived_from"]["training_study_summary"] == {
        "path": SUMMARY_PATH.relative_to(ROOT).as_posix(),
        "sha256": _sha256(SUMMARY_PATH),
        "size_bytes": SUMMARY_PATH.stat().st_size,
    }
    assert portable_manifest["artifact"] == {
        "distribution": "included",
        **artifact_record,
    }
    assert portable_manifest["interface"]["raw_output"]["shape"] == ["batch", 22]
    assert portable_manifest["training"]["registered_policy_seeds"] == [
        37017,
        47017,
        57017,
        67017,
        77017,
    ]
    assert portable_manifest["training"]["config"] == raw_manifest["training"][
        "config"
    ]
    assert portable_manifest["runtime_versions"] == raw_manifest["runtime_versions"]
    assert portable_manifest["development_parity"] == facts
    for record in (
        portable_manifest["selected_checkpoint"],
        portable_manifest["selected_checkpoint"]["bundle_manifest"],
        portable_manifest["selected_checkpoint"]["normalization"],
        portable_manifest["selected_checkpoint"]["training_receipt"],
    ):
        assert record["distribution"] == "not_in_repository"
        assert record["path"] is None

    for source in summary["source_evidence"].values():
        if not isinstance(source, dict):
            continue
        external = Path(source["path"])
        if external.exists():
            assert external.stat().st_size == source["size_bytes"]
            assert _sha256(external) == source["sha256"]


def test_training_study_summary_matches_external_machine_receipts_when_present() -> None:
    summary = _load(SUMMARY_PATH)
    seed_path = Path(summary["source_evidence"]["seed_sweep_summary"]["path"])
    ablation_path = Path(summary["source_evidence"]["ablation_summary"]["path"])
    if not seed_path.exists():
        assert not ablation_path.exists()
        return
    assert ablation_path.exists()

    seed_source = _load(seed_path)
    assert seed_source["phase"] == "seed_sweep"
    assert seed_source["split"] == "dev"
    assert seed_source["development_case_count"] == 200
    assert seed_source["final_split_used"] is False
    assert seed_source["aggregate"] == summary["baseline"]["aggregate"]
    baseline_config = summary["baseline"]["config"]
    source_config = seed_source["baseline"]
    assert baseline_config == {
        "active_actor_critic_transitions": source_config["transitions"],
        "reward_profile": source_config["reward_profile"],
        "preparedness_alignment_coefficient": source_config[
            "preparedness_alignment_coefficient"
        ],
        "bc_warm_start": source_config["bc_warm_start"],
        "vec_normalize": source_config["vec_normalize"],
    }
    metric_keys = (
        "seed",
        "active_actor_critic_transitions",
        "solved_count",
        "solve_rate",
        "mean_resilience_auc",
        "mean_minimum_tail_margin",
    )
    for tracked, source in zip(
        summary["baseline"]["endpoints"],
        seed_source["rows"],
        strict=True,
    ):
        assert {key: tracked[key] for key in metric_keys} == {
            key: source[key] for key in metric_keys
        }
        assert source["hard_violation_count"] == 0
        assert source["maximum_conservation_residual"] == 0.0
        assert _sha256(Path(source["receipt"])) == tracked["receipt_sha256"]

    ablation_source = _load(ablation_path)
    assert ablation_source["phase"] == "ablations"
    assert ablation_source["split"] == "dev"
    assert ablation_source["development_case_count"] == 200
    assert ablation_source["final_split_used"] is False
    source_by_name = {
        row["treatment"]["name"]: row for row in ablation_source["comparisons"]
    }
    for tracked in summary["ablations"]:
        source = source_by_name[tracked["name"]]
        source_treatment = source["treatment"]
        assert tracked["treatment"] == {
            "active_actor_critic_transitions": source_treatment["transitions"],
            "reward_profile": source_treatment["reward_profile"],
            "preparedness_alignment_coefficient": source_treatment[
                "preparedness_alignment_coefficient"
            ],
            "bc_warm_start": source_treatment["bc_warm_start"],
            "vec_normalize": source_treatment["vec_normalize"],
        }
        assert tracked["aggregate"] == source["treatment_aggregate"]
        assert tracked["mean_treatment_minus_control_solved"] == source[
            "mean_treatment_minus_control_solved"
        ]
        for endpoint, source_endpoint in zip(
            tracked["endpoints"],
            source["treatment_rows"],
            strict=True,
        ):
            assert {key: endpoint[key] for key in metric_keys} == {
                key: source_endpoint[key] for key in metric_keys
            }
            assert source_endpoint["hard_violation_count"] == 0
            assert source_endpoint["maximum_conservation_residual"] == 0.0
            assert _sha256(Path(source_endpoint["receipt"])) == endpoint[
                "receipt_sha256"
            ]
        for pair, source_pair in zip(
            tracked["paired_solved_deltas"],
            source["paired_rows"],
            strict=True,
        ):
            assert pair == {
                "seed": source_pair["seed"],
                "control": source_pair["control_solved_count"],
                "treatment": source_pair["treatment_solved_count"],
                "delta": source_pair["treatment_minus_control_solved"],
            }
