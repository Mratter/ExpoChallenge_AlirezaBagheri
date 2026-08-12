from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.shared_evidence import canonical_hash
from scripts import run_combined_distilled_large_study as combined
from scripts import run_large_architecture_study as capacity
from scripts import train_oracle_bc_student as oracle_bc
from scripts import train_policy


def _fit_reference() -> dict[str, object]:
    reference = {
        "root": "E:/outside/large-oracle-bc/checkpoint",
        "fit_success": {
            "path": "E:/outside/large-oracle-bc/fit.success.json",
            "sha256": "1" * 64,
        },
        "checkpoint": {
            "root": "E:/outside/large-oracle-bc/checkpoint",
            "manifest_sha256": "2" * 64,
            "model_sha256": "3" * 64,
            "actor_state_sha256": "4" * 64,
            "observation_rms_sha256": "5" * 64,
        },
        "dataset_receipt_sha256": "6" * 64,
        "dataset_components": {
            "trajectory_split_sha256": "7" * 64,
            "dataset_index_sha256": "8" * 64,
        },
        "final_split_imported_or_used": False,
    }
    return {**reference, "reference_sha256": canonical_hash(reference)}


def _endpoint_candidate(seed: int, solved: int) -> dict[str, object]:
    return {
        "policy_seed": seed,
        "active_actor_critic_transitions": combined.ACTIVE_TRANSITIONS,
        "development": {
            "solved_count": solved,
            "per_family_solved_count": {
                "family_a": solved // 5,
                "family_b": solved - solved // 5,
            },
        },
    }


def _fit_contract_fixture() -> dict[str, object]:
    runtime = combined.distilled._torch_runtime_binding()
    return {
        **runtime,
        "dataset": {"trajectory_split": {"fit": 5_040, "heldout": 720}},
        "normalization": {"observation_rms_sha256": "5" * 64},
        "causal_input_contract": {
            "training_split_only": True,
            "student_input": "73_public_causal_observation_channels",
            "student_input_count": 73,
            "student_output": "22_continuous_action_targets",
            "student_output_count": 22,
            "student_input_future_tape_visible": False,
            "teacher_target_uses_full_future_tape": True,
            "future_tape_use": "teacher_label_generator_only",
        },
    }


def _fit_report_fixture() -> dict[str, object]:
    fit_contract = _fit_contract_fixture()
    before_actor = "1" * 64
    after_actor = "2" * 64
    critic = "3" * 64
    metrics = {
        "mse": 0.2,
        "mean_absolute_error": 0.1,
        "mean_absolute_error_by_dimension": [0.1] * 22,
    }
    trained = {
        "mse": 0.1,
        "mean_absolute_error": 0.05,
        "mean_absolute_error_by_dimension": [0.05] * 22,
    }
    return {
        "method": "matched_large_behavior_cloning_only",
        "architecture": [768, 512, 256],
        "parameter_counts": capacity.EXPECTED_PARAMETER_COUNTS,
        "policy_seed": 67017,
        "epochs": 15,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "training_row_count_per_student": 5_040,
        "heldout_row_count_per_student": 720,
        "trajectory_split": {"fit": 5_040, "heldout": 720},
        "matched_initialization": {
            "actor_state_sha256": before_actor,
            "critic_state_sha256": critic,
            "actor_hashes_equal_before_fit": True,
            "critic_hashes_equal_before_fit": True,
            "observation_rms_hashes_equal": True,
            "minibatch_permutations_identical": True,
        },
        "oracle_label_student": {
            "label_source": "privileged_same_budget_cem_oracle",
            "actor_state_sha256_before": before_actor,
            "actor_state_sha256_after": after_actor,
            "critic_state_sha256_before": critic,
            "critic_state_sha256_after": critic,
            "critic_unchanged": True,
            "heldout": {
                "untrained": metrics,
                "trained": trained,
                "relative_mse_improvement": 0.5,
            },
        },
        "matched_hand_rule_control": {
            "label_source": "preparedness_teacher_action_public_rule",
            "actor_state_sha256_before": before_actor,
            "actor_state_sha256_after": "4" * 64,
            "critic_state_sha256_before": critic,
            "critic_state_sha256_after": critic,
            "critic_unchanged": True,
        },
        "dagger_iterations": 0,
        "ppo_updates": 0,
        "active_actor_critic_transitions": 0,
        "observation_rms_sha256": "5" * 64,
        "observation_rms_imported_from_distillation_run": True,
        "normalization_frozen": True,
        "holdout_excluded_from_fit": True,
        "causal_input_contract": fit_contract["causal_input_contract"],
    }


def test_registered_combined_contract_and_trainer_arguments() -> None:
    assert combined.HIDDEN_LAYERS == (768, 512, 256)
    assert combined.LEARNING_RATE == 3e-5
    assert combined.POLICY_SEEDS == (37017, 47017, 57017)
    assert combined.FIT_POLICY_SEED == 67017
    assert combined.FIT_ROW_COUNT == 5040
    assert combined.HOLDOUT_ROW_COUNT == 720
    assert combined.BC_EPOCHS == 15
    assert combined.BC_BATCH_SIZE == 512
    assert combined.BC_LEARNING_RATE == 1e-3
    assert combined.FIXED_CRITIC_WARMUP_TRANSITIONS == 50_000
    assert capacity.EXPECTED_PARAMETER_COUNTS == {
        "actor": 587_564,
        "critic": 582_145,
        "total_policy": 1_169_709,
    }

    arguments = combined.trainer_arguments(Path("E:/combined"), 37017)
    parsed = train_policy.parse_args(arguments)
    assert parsed.transitions == 2_000_000
    assert parsed.policy_seed == 37017
    assert parsed.learning_rate == 3e-5
    assert parsed.critic_warmup_min_transitions == 50_000
    assert parsed.critic_warmup_max_transitions == 50_000
    assert parsed.freeze_observation_rms is True
    assert parsed.reward_profile == "v3_equivalent"
    assert train_policy.validate_runtime_config(parsed) == 5_000


def test_portable_upstream_receipts_are_hash_and_source_bound() -> None:
    upstream = combined.load_portable_upstream_evidence()
    assert upstream["distillation"]["sha256"] == (
        combined.DISTILLATION_EVIDENCE_SHA256
    )
    assert upstream["distillation"]["endpoint_solved_counts"] == [178, 174, 170]
    assert upstream["capacity"]["sha256"] == combined.CAPACITY_EVIDENCE_SHA256
    assert upstream["capacity"]["large_lr_3e_5_endpoints"] == [178, 176, 175]
    assert upstream["capacity"]["architecture"]["actor_hidden_layers"] == [
        768,
        512,
        256,
    ]
    assert [
        row["critic_warmup_transitions"]
        for row in upstream["capacity"]["large_lr_3e_5_run_bindings"]
    ] == [50_000, 50_000, 60_000]
    confounds = upstream["comparison_confounds"]
    assert confounds["comparison_is_nonfactorial"] is True
    assert confounds["causal_increment_of_distilled_initialization_isolated"] is False
    assert confounds["critic_warmup_transitions_by_seed"]["combined"]["57017"] == 50_000


def test_portable_upstream_rejects_receipt_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(combined, "DISTILLATION_EVIDENCE_SHA256", "0" * 64)
    with pytest.raises(combined.CombinedStudyError, match="evidence drifted"):
        combined.load_portable_upstream_evidence()


def test_expected_config_contains_both_treatments_and_no_legacy_collection() -> None:
    upstream = combined.load_portable_upstream_evidence()
    fit_reference = _fit_reference()
    config = combined.expected_training_config(37017, fit_reference, upstream)

    architecture = config["architecture_experiment"]
    treatment = config["combined_distillation_capacity_experiment"]
    assert architecture["actor_hidden_layers"] == [768, 512, 256]
    assert architecture["critic_hidden_layers"] == [768, 512, 256]
    assert architecture["arm_id"] == combined.ARM.id
    assert config["learning_rate"] == 3e-5
    assert config["critic_warmup_min_transitions"] == 50_000
    assert config["critic_warmup_max_transitions"] == 50_000
    assert treatment["initialization_method"] == (
        "new_large_oracle_distilled_bc_actor"
    )
    assert treatment["dagger_iterations"] == 0
    assert treatment["legacy_bc_or_dagger_dataset_collected_by_ppo_worker"] is False
    assert treatment["source_large_bc_fit_reference_sha256"] == (
        fit_reference["reference_sha256"]
    )
    assert treatment["source_distillation_evidence_sha256"] == (
        combined.DISTILLATION_EVIDENCE_SHA256
    )
    assert treatment["source_capacity_evidence_sha256"] == (
        combined.CAPACITY_EVIDENCE_SHA256
    )


def test_fresh_policy_state_uses_large_actor_and_critic_shapes() -> None:
    state = combined._fresh_large_policy_state(37017)
    assert tuple(state["mlp_extractor.policy_net.0.weight"].shape) == (768, 73)
    assert tuple(state["mlp_extractor.policy_net.2.weight"].shape) == (512, 768)
    assert tuple(state["mlp_extractor.policy_net.4.weight"].shape) == (256, 512)
    assert tuple(state["mlp_extractor.value_net.0.weight"].shape) == (768, 73)
    assert tuple(state["action_net.weight"].shape) == (22, 256)
    assert tuple(state["value_net.weight"].shape) == (1, 256)


def test_fit_contract_reuses_exact_split_and_source_rms() -> None:
    sha = "a" * 64
    dataset = oracle_bc.OracleDataset(
        observations=np.empty((0, 73), dtype=np.float32),
        oracle_targets=np.empty((0, 22), dtype=np.float32),
        hand_rule_targets=np.empty((0, 22), dtype=np.float32),
        row_ids=(),
        fit_indices=np.empty(0, dtype=np.int64),
        holdout_indices=np.empty(0, dtype=np.int64),
        split_contract={"fit": 5_040, "heldout": 720},
        receipt_path=Path("E:/dataset/training/receipt.json"),
        receipt_sha256=sha,
        contract_sha256="b" * 64,
        dataset_index_sha256="c" * 64,
    )
    upstream = {"distillation": {"sha256": "d" * 64}}
    source_student = {
        "checkpoint": {
            "manifest_sha256": "e" * 64,
            "observation_rms_sha256": "f" * 64,
        }
    }
    contract = combined._fit_contract(dataset, upstream, source_student)
    assert contract["architecture"] == [768, 512, 256]
    assert contract["fit_rows"] == 5_040
    assert contract["heldout_rows"] == 720
    assert contract["matched_hand_rule_control"] is True
    assert contract["dagger_iterations"] == 0
    assert contract["normalization"]["observation_rms_sha256"] == "f" * 64
    assert contract["normalization"]["frozen"] is True
    assert contract["torch_runtime_sha256"] == canonical_hash(
        contract["torch_runtime"]
    )
    causal = contract["causal_input_contract"]
    assert causal == {
        "training_split_only": True,
        "student_input": "73_public_causal_observation_channels",
        "student_input_count": 73,
        "student_output": "22_continuous_action_targets",
        "student_output_count": 22,
        "student_input_future_tape_visible": False,
        "teacher_target_uses_full_future_tape": True,
        "future_tape_use": "teacher_label_generator_only",
    }


def test_fit_runtime_rejects_mutation_even_with_self_consistent_hash() -> None:
    contract = _fit_contract_fixture()
    combined._validate_fit_runtime(contract)
    mutated = copy.deepcopy(contract)
    mutated["torch_runtime"]["torch_version"] = "mutated"
    mutated["torch_runtime_sha256"] = canonical_hash(mutated["torch_runtime"])
    with pytest.raises(combined.CombinedStudyError, match="Torch runtime"):
        combined._validate_fit_runtime(mutated)


def test_large_fit_report_binds_actor_critics_hyperparameters_split_and_rms() -> None:
    contract = _fit_contract_fixture()
    report = _fit_report_fixture()
    combined._validate_large_fit_report(
        report,
        contract,
        checkpoint_actor_sha256="2" * 64,
        normalization_rms_sha256="5" * 64,
    )

    mutations = []
    actor_mutation = copy.deepcopy(report)
    actor_mutation["oracle_label_student"]["actor_state_sha256_after"] = "9" * 64
    mutations.append(actor_mutation)
    critic_mutation = copy.deepcopy(report)
    critic_mutation["matched_hand_rule_control"]["critic_unchanged"] = False
    mutations.append(critic_mutation)
    hyperparameter_mutation = copy.deepcopy(report)
    hyperparameter_mutation["epochs"] = 14
    mutations.append(hyperparameter_mutation)
    split_mutation = copy.deepcopy(report)
    split_mutation["trajectory_split"] = {"fit": 5_039, "heldout": 721}
    mutations.append(split_mutation)
    rms_mutation = copy.deepcopy(report)
    rms_mutation["normalization_frozen"] = False
    mutations.append(rms_mutation)
    causal_mutation = copy.deepcopy(report)
    causal_mutation["causal_input_contract"]["student_input_future_tape_visible"] = True
    mutations.append(causal_mutation)
    for mutated in mutations:
        with pytest.raises(combined.CombinedStudyError, match="report contract"):
            combined._validate_large_fit_report(
                mutated,
                contract,
                checkpoint_actor_sha256="2" * 64,
                normalization_rms_sha256="5" * 64,
            )


def test_large_bc_fit_gate_blocks_nonlearning_or_nonfinite_actor() -> None:
    report = _fit_report_fixture()
    gate = combined.large_bc_fit_gate(report)
    assert gate["passed"] is True
    assert gate["decision"] == "eligible_for_ppo"
    assert gate["development_evaluated"] is False

    no_change = copy.deepcopy(report)
    no_change["oracle_label_student"]["actor_state_sha256_after"] = "1" * 64
    assert combined.large_bc_fit_gate(no_change)["passed"] is False
    at_floor = copy.deepcopy(report)
    at_floor["oracle_label_student"]["heldout"]["relative_mse_improvement"] = 0.01
    assert combined.large_bc_fit_gate(at_floor)["passed"] is False
    spoofed = copy.deepcopy(report)
    spoofed["oracle_label_student"]["heldout"]["relative_mse_improvement"] = 0.9
    assert combined.large_bc_fit_gate(spoofed)["passed"] is False
    nonfinite = copy.deepcopy(report)
    nonfinite["oracle_label_student"]["heldout"]["trained"]["mse"] = float("nan")
    assert combined.large_bc_fit_gate(nonfinite)["passed"] is False


def test_failed_large_bc_gate_returns_before_ppo_protocol_or_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "combined-output"
    minimal_contract = {
        "tool": combined.TOOL_ID,
        "large_oracle_bc_fit": {"causal_input_contract": {}},
    }
    failed_gate = {
        "passed": False,
        "decision": "abort_before_ppo",
    }
    monkeypatch.setattr(combined, "load_portable_upstream_evidence", lambda: {})
    monkeypatch.setattr(
        combined, "_load_source_student_reference", lambda _root: {}
    )
    monkeypatch.setattr(combined.oracle_bc, "load_oracle_dataset", lambda _root: object())
    monkeypatch.setattr(combined, "base_contract", lambda *_args, **_kwargs: minimal_contract)
    monkeypatch.setattr(combined, "_worktree_is_clean", lambda: True)
    monkeypatch.setattr(combined, "_persist_large_fit", lambda *_args: None)
    monkeypatch.setattr(combined, "load_large_fit_gate", lambda *_args: failed_gate)

    result = combined.main(
        [
            "--output-root",
            str(output_root),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--source-student-root",
            str(tmp_path / "student"),
            "--execute",
        ]
    )
    assert result == combined.oracle_bc.COMPLETED_NEGATIVE_GATE_EXIT_CODE
    assert (output_root / "protocol.json").is_file()
    assert not (output_root / "ppo-protocol.json").exists()
    assert not any(output_root.glob("seed-*"))


def test_endpoint_summary_reports_both_paired_comparisons_and_family_counts() -> None:
    candidates = [
        _endpoint_candidate(37017, 184),
        _endpoint_candidate(47017, 177),
        _endpoint_candidate(57017, 173),
    ]
    summary = combined.endpoint_summary(candidates)
    assert summary["solved_counts_by_seed"] == {
        "37017": 184,
        "47017": 177,
        "57017": 173,
    }
    assert summary["mean_solved_count"] == pytest.approx(178.0)
    assert [
        row["delta_vs_incumbent"] for row in summary["paired_same_seed"]
    ] == [12, 6, 2]
    assert [
        row["delta_vs_large_only"] for row in summary["paired_same_seed"]
    ] == [6, 1, -2]
    assert summary["per_family"]["family_a"]["solved_counts_by_seed"] == {
        "37017": 36,
        "47017": 35,
        "57017": 34,
    }


def test_real_200_row_shape_recomputes_and_binds_family_counts() -> None:
    portable = json.loads(
        combined.CAPACITY_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    development = portable["best_checkpoint"]["development"]
    raw_rows = [
        {
            "row_id": row["row_id"],
            "case_seed": row["case_seed"],
            "tape_seed": row["tape_seed"],
            "tape_sha256": row["tape_sha256"],
            "solved": row["solved"],
            "reason_codes": row["reason_codes"],
            "resilience_auc": row["resilience_auc"],
            "minimum_tail_margin": row["minimum_tail_margin"],
            "hard_violation_count": row["hard_violation_count"],
            "max_conservation_residual": row[
                "maximum_conservation_residual"
            ],
        }
        for row in development["rows"]
    ]
    raw = {
        "active_actor_critic_transitions": development[
            "active_actor_critic_transitions"
        ],
        "total_environment_transitions": development[
            "total_environment_transitions"
        ],
        "case_count": development["case_count"],
        "solved_count": development["solved_count"],
        "solve_rate": development["solve_rate"],
        "mean_resilience_auc": development["mean_resilience_auc"],
        "mean_minimum_tail_margin": development["mean_minimum_tail_margin"],
        "hard_violation_count": development["hard_violation_count"],
        "maximum_conservation_residual": development[
            "maximum_conservation_residual"
        ],
        "failure_reason_code_histogram": development[
            "failure_reason_code_histogram"
        ],
        "rows": raw_rows,
    }
    validated = combined._development_result_with_family_counts(
        raw,
        "tracked capacity endpoint",
        expected_active_transitions=development[
            "active_actor_critic_transitions"
        ],
        expected_total_transitions=development[
            "total_environment_transitions"
        ],
    )
    assert len(raw_rows) == 200
    assert validated["per_family_solved_count"] == development[
        "per_family_solved_count"
    ]
    assert set(validated["per_family_rows_sha256"]) == set(
        development["per_family_solved_count"]
    )
    assert sum(validated["per_family_solved_count"].values()) == development[
        "solved_count"
    ]


def test_promotion_is_conjunctive() -> None:
    passing = {
        "mean_solved_count": 175.0,
        "seed_count_at_or_above_172": 3,
    }
    assert combined.promotion_decision(183, passing)["passed"] is True
    assert combined.promotion_decision(182, passing)["passed"] is False
    assert (
        combined.promotion_decision(
            183,
            {
                "mean_solved_count": 171.4,
                "seed_count_at_or_above_172": 3,
            },
        )["passed"]
        is False
    )
    assert (
        combined.promotion_decision(
            183,
            {
                "mean_solved_count": 175.0,
                "seed_count_at_or_above_172": 1,
            },
        )["passed"]
        is False
    )


def test_rank_candidates_uses_solve_count_then_earlier_milestone_then_seed() -> None:
    candidates: list[dict[str, object]] = []
    for seed in combined.POLICY_SEEDS:
        for milestone in combined.SELECTION_MILESTONES:
            candidates.append(
                {
                    "policy_seed": seed,
                    "active_actor_critic_transitions": milestone,
                    "development": {"solved_count": 170},
                }
            )
    candidates[-1]["development"] = {"solved_count": 180}
    ranked = combined.rank_candidates(candidates)
    assert ranked[0]["policy_seed"] == 57017
    assert ranked[0]["active_actor_critic_transitions"] == 2_000_000
    assert ranked[1]["policy_seed"] == 37017
    assert ranked[1]["active_actor_critic_transitions"] == 500_000


def test_create_new_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    combined._atomic_create_json(path, {"status": "complete"})
    with pytest.raises(combined.CombinedStudyError, match="refusing to overwrite"):
        combined._atomic_create_json(path, {"status": "changed"})
    assert path.read_text(encoding="utf-8") == '{\n  "status": "complete"\n}\n'


def test_output_root_must_be_external_and_not_a_drive_root() -> None:
    with pytest.raises(combined.CombinedStudyError, match="outside"):
        combined._require_external_root(
            combined.ROOT / "would-be-output", "--output-root"
        )
    with pytest.raises(combined.CombinedStudyError, match="filesystem root"):
        combined._require_external_root(
            Path(combined.ROOT.anchor), "--output-root"
        )


def test_module_has_no_final_split_dependency() -> None:
    source = Path(combined.__file__).read_text(encoding="utf-8")
    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert "evaluate_final" not in source
