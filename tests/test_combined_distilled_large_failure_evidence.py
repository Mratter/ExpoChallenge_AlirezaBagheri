"""Receipt-only checks for the stopped combined experiment publication."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import publish_combined_distilled_large_failure_evidence as publisher

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal/developmental_runs/v4/combined-distilled-large-study-200.incomplete.json"
)
REPORT = ROOT / "benchmarks/v4/combined-distilled-large-study-200.incomplete.md"


def _load() -> dict[str, object]:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_portable_receipt_is_canonical_incomplete_dev_only_and_source_bound() -> None:
    receipt = _load()
    assert RECEIPT.read_bytes() == publisher._canonical_json_bytes(receipt)
    assert receipt["schema_version"] == publisher.SCHEMA_VERSION
    assert receipt["status"] == (
        "incomplete_stopped_at_preregistered_critic_warmup_gate"
    )
    assert receipt["split"] == "dev"
    assert receipt["final_split_imported_or_used"] is False
    assert receipt["attempt"] == {
        "completed_policy_seeds": [37017, 47017],
        "git_commit": publisher.EXPECTED_GIT_COMMIT,
        "no_retry_or_resume_performed": True,
        "registered_policy_seeds": [37017, 47017, 57017],
        "stopped_policy_seed": 57017,
    }
    assert all(receipt["invariants"].values())
    publication = receipt["publication_source_contract"]
    assert publication["source_identity_sha256"] == canonical_hash(
        publication["source_files"]
    )
    for path, expected in publication["source_files"].items():
        assert file_sha256(ROOT / path) == expected
    assert receipt["protocol"]["source_identity_sha256"] == canonical_hash(
        receipt["protocol"]["source_files"]
    )
    for path, expected in receipt["protocol"]["source_files"].items():
        assert file_sha256(ROOT / path) == expected


def test_external_inventory_and_console_logs_are_fully_bound() -> None:
    receipt = _load()
    inventory = receipt["external_root_inventory"]
    assert inventory["file_count"] == 37
    assert len(inventory["files"]) == 37
    assert canonical_hash(inventory["files"]) == inventory["inventory_sha256"]
    assert inventory["inventory_sha256"] == (
        publisher.EXPECTED_ROOT_INVENTORY_SHA256
    )
    assert receipt["console_logs"] == publisher.EXPECTED_CONSOLE_LOGS
    encoded = RECEIPT.read_text(encoding="utf-8")
    assert "E:\\" not in encoded
    assert "E:/" not in encoded
    assert "Alireza" not in encoded


def test_fit_metrics_method_gate_and_checkpoint_are_exact() -> None:
    fit = _load()["fit"]
    assert fit["method"] == {
        "architecture": [768, 512, 256],
        "batch_size": 512,
        "dagger_iterations": 0,
        "epochs": 15,
        "fit_rows": 5040,
        "heldout_rows": 720,
        "learning_rate": 0.001,
        "observation_rms_sha256": (
            "cb7b9a46369a0c225c3a6254433f6ef37e52b822ef44598fa4311b64e63a4ba4"
        ),
        "parameter_counts": {
            "actor": 587564,
            "critic": 582145,
            "total_policy": 1169709,
        },
        "public_input_count": 73,
        "public_output_count": 22,
        "single_pass_offline": True,
    }
    oracle = fit["oracle_label_student"]
    control = fit["matched_hand_rule_control"]
    assert oracle["fit"]["mse"] == 0.038270253688097
    assert oracle["fit"]["mean_absolute_error"] == 0.138376384973526
    assert oracle["heldout"]["trained"]["mse"] == 0.04105028137564659
    assert oracle["heldout"]["trained"]["mean_absolute_error"] == (
        0.14487330615520477
    )
    assert oracle["heldout"]["relative_mse_improvement"] == 0.8704107290430282
    assert control["heldout"]["trained"]["mse"] == 0.02172856591641903
    assert control["heldout"]["trained"]["mean_absolute_error"] == (
        0.08579652011394501
    )
    assert fit["gate"]["passed"] is True
    assert fit["checkpoint"]["manifest_sha256"] == (
        "f031ce08a2715615e407402f95a729776e705a0ea8ef6a2ef0b9407c97ce5992"
    )


def test_two_complete_curves_and_stopped_seed_recompute_from_rows() -> None:
    receipt = _load()
    seeds = {
        row["policy_seed"]: row for row in receipt["development_evidence"]["seeds"]
    }
    expected = {
        37017: {
            "bc_initialization": 153,
            "post_critic_warmup": 153,
            "ppo_200000_transitions": 158,
            "ppo_500000_transitions": 162,
            "ppo_1000000_transitions": 173,
            "ppo_2000000_transitions": 170,
        },
        47017: {
            "bc_initialization": 153,
            "post_critic_warmup": 153,
            "ppo_200000_transitions": 160,
            "ppo_500000_transitions": 167,
            "ppo_1000000_transitions": 170,
            "ppo_2000000_transitions": 174,
        },
        57017: {
            "bc_initialization": 153,
            "post_critic_warmup": 153,
        },
    }
    for seed, row in seeds.items():
        curve = row["development_curve"]
        assert {
            key: value["solved_count"] for key, value in curve.items()
        } == expected[seed]
        for value in curve.values():
            assert len(value["rows"]) == 200
            assert sum(item["solved"] for item in value["rows"]) == value[
                "solved_count"
            ]
            assert value["rows_sha256"] == canonical_hash(value["rows"])
            assert len(value["source_rows_sha256"]) == 64
            assert sum(value["per_family_solved_count"].values()) == value[
                "solved_count"
            ]
            assert value["hard_violation_count"] == 0
            assert value["maximum_conservation_residual"] == 0.0
            reasons: Counter[str] = Counter()
            for item in value["rows"]:
                if not item["solved"]:
                    reasons.update(item["reason_codes"])
            assert value["failure_reason_code_histogram"] == dict(
                sorted(reasons.items())
            )
            assert value["mean_minimum_tail_margin"] == round(
                fmean(item["minimum_tail_margin"] for item in value["rows"]),
                10,
            )
            assert value["mean_resilience_auc"] == round(
                fmean(item["resilience_auc"] for item in value["rows"]), 10
            )
            assert value["split_identity_sha256"] == (
                publisher.EXPECTED_DEV_IDENTITY_SHA256
            )
    assert [len(seeds[seed]["bundles"]) for seed in (37017, 47017, 57017)] == [
        4,
        4,
        0,
    ]
    assert receipt["development_evidence"][
        "best_observed_completed_checkpoint"
    ] == {
        "active_actor_critic_transitions": 2000000,
        "checkpoint_id": "seed-47017-ppo-2000000",
        "policy_seed": 47017,
        "solved_count": 174,
    }
    assert receipt["development_evidence"][
        "retained_development_evaluation_count"
    ] == 14


def test_stop_gate_and_partial_comparison_are_not_reinterpreted() -> None:
    evidence = _load()["development_evidence"]
    stopped = next(
        row for row in evidence["seeds"] if row["policy_seed"] == 57017
    )
    assert stopped["status"] == "critic_warmup_incomplete"
    assert stopped["transition_counts"] == {
        "active_actor_critic": 0,
        "critic_warmup": 50000,
        "total_environment": 50000,
    }
    assert stopped["warmup"]["explained_variance_threshold"] == 0.5
    assert stopped["warmup"]["first_rollout_explained_variance"] == (
        -0.02818441390991211
    )
    assert stopped["warmup"]["last_warmup_rollout_explained_variance"] == (
        0.47894805669784546
    )
    assert stopped["warmup"]["gate_passed"] is False
    assert max(
        row["explained_variance"] for row in stopped["warmup"]["iterations"]
    ) > 0.5
    assert evidence["completed_seed_endpoint_pairs_only"] == [
        {
            "combined_endpoint": 170,
            "delta_vs_incumbent_endpoint": -2,
            "delta_vs_large_only_endpoint": -8,
            "incumbent_endpoint": 172,
            "large_only_endpoint": 178,
            "policy_seed": 37017,
        },
        {
            "combined_endpoint": 174,
            "delta_vs_incumbent_endpoint": 3,
            "delta_vs_large_only_endpoint": -2,
            "incumbent_endpoint": 171,
            "large_only_endpoint": 176,
            "policy_seed": 47017,
        },
    ]
    assert evidence["no_three_seed_mean_or_standard_deviation"] is True
    assert evidence["promotion_decision"] == (
        "not_evaluable_incomplete_registered_seed_roster"
    )
    assert "mean_solved_count" not in evidence
    assert _load()["stop_provenance"] == {
        "expected_trainer_return_code": 3,
        "orchestrator_action": "stop_on_nonzero_trainer_return_code",
        "strict_gate_operator": ">",
        "worker_status": "critic_warmup_incomplete",
    }


def test_upstream_evidence_binds_partial_comparison_inputs() -> None:
    upstream = _load()["upstream_evidence"]
    assert upstream["incumbent_same_seed_endpoints"] == [172, 171, 171]
    assert upstream["capacity"] == {
        "large_only_endpoints": [178, 176, 175],
        "portable_path": (
            "internal/developmental_runs/v4/network-capacity-study-200.json"
        ),
        "sha256": publisher.EXPECTED_CAPACITY_EVIDENCE_SHA256,
    }
    assert upstream["distillation"]["sha256"] == (
        publisher.EXPECTED_DISTILLATION_EVIDENCE_SHA256
    )
    assert upstream["distillation"]["source_student"] == {
        "checkpoint_manifest_sha256": (
            "4516d22c5ae20a8b460210d96c271f53abee531b1d7c2fcd977919a4eec5b02e"
        ),
        "checkpoint_model_sha256": (
            "f95f248a421480b1f516b0d523481b9d7c71dc025016036da60f786cc3156ee0"
        ),
        "checkpoint_normalization_sha256": (
            "5c3fda003eb32979a02284f5aed8b00c98be12b201aa662fef853b214ec6bbab"
        ),
        "receipt_sha256": (
            "76025a6376db6905b1d96d08122a14bccc7639040921768a79e4c83debabec84"
        ),
    }
    assert upstream["training_oracle_dataset"]["sha256"] == (
        publisher.EXPECTED_DATASET_RECEIPT_SHA256
    )


def test_report_and_publisher_check_are_exact() -> None:
    receipt = _load()
    expected = publisher.render_markdown(receipt, file_sha256(RECEIPT))
    assert REPORT.read_text(encoding="utf-8") == expected
    assert "not a completed three-seed study" in expected
    assert "not retried or resumed" in expected
    assert "no final case" in expected
    assert "does **not** calculate a two-seed substitute" in expected
    assert "33/40 | 35/40 | 40/40 | 27/40 | 35/40" in expected
    assert "all 14 retained development evaluations" in expected
    assert "returns code `3`" in expected


def test_portable_check_never_requires_author_local_external_evidence() -> None:
    """The CI publication gate must be reproducible from tracked portable bytes."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    publisher_call = "python scripts/publish_combined_distilled_large_failure_evidence.py --check"
    assert publisher_call not in workflow
    assert (
        "python -m pytest -q tests/test_combined_distilled_large_failure_evidence.py"
        in workflow
    )
    assert REPORT.read_text(encoding="utf-8") == publisher.render_markdown(
        _load(), file_sha256(RECEIPT)
    )


@pytest.mark.skipif(
    not publisher.DEFAULT_STUDY_ROOT.is_dir()
    or not publisher.DATASET_RECEIPT.is_file(),
    reason="external stopped-attempt evidence is unavailable",
)
def test_live_external_receipt_replay_is_byte_identical_when_available() -> None:
    assert publisher.main(["--check"]) == 0


def test_publisher_fails_closed_on_inventory_or_receipt_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    (study / "unexpected.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="file count drifted"):
        publisher.build_portable_evidence(study, tmp_path)

    receipt = _load()
    mutated = copy.deepcopy(receipt)
    mutated["development_evidence"]["promotion_decision"] = "promoted"
    assert publisher._canonical_json_bytes(mutated) != RECEIPT.read_bytes()
