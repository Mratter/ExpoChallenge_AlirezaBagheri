from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts.train_oracle_bc_student import (
    ACTION_SIZE,
    ACTION_ORDER,
    BC_BATCH_SIZE,
    BC_EPOCHS,
    BC_LEARNING_RATE,
    COMPLETED_NEGATIVE_GATE_EXIT_CODE,
    DATASET_ROW_COUNT,
    DATASET_SOURCE_PATHS,
    DEVELOPMENT_CASE_COUNT,
    DEVELOPMENT_CATASTROPHIC_FLOOR,
    FIT_CASE_COUNT,
    FIT_ROW_COUNT,
    HOLDOUT_CASE_COUNT,
    HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR,
    HOLDOUT_ROW_COUNT,
    HORIZON_DAYS,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    OLD_BC_SOLVED_COUNT,
    POLICY_SEED,
    EXPECTED_ORACLE_CONFIG,
    TRAINING_CASE_COUNT,
    OracleBCError,
    OracleDataset,
    StudentFit,
    _atomic_create_json,
    _expected_development_identity,
    _expected_training_case_contracts,
    _model_environment,
    _trajectory_split_contract,
    _validate_development,
    catastrophic_gate,
    evaluate_once,
    fit_student,
    load_oracle_dataset,
    result_exit_code,
    student_contract,
    validate_old_bc_anchor,
)


def _demonstration(row_id: str) -> dict[str, Any]:
    observations = np.zeros(
        (HORIZON_DAYS, OBSERVATION_SIZE), dtype=np.float32
    ).tolist()
    targets = np.zeros((HORIZON_DAYS, ACTION_SIZE), dtype=np.float32).tolist()
    return {
        "row_id": row_id,
        "input_contract": "73_public_causal_observations",
        "student_input_future_tape_visible": False,
        "teacher_target_uses_full_future_tape": True,
        "observation_dtype": "float32",
        "target_dtype": "float32",
        "observation_shape": [HORIZON_DAYS, OBSERVATION_SIZE],
        "target_shape": [HORIZON_DAYS, ACTION_SIZE],
        "observations": observations,
        "targets": targets,
        "observations_sha256": canonical_hash(observations),
        "targets_sha256": canonical_hash(targets),
        "dataset_sha256": canonical_hash(
            {"observations": observations, "targets": targets}
        ),
    }


def _fake_dataset(
    root: Path,
    *,
    extra_input_field: bool = False,
    fabricated_identity: bool = False,
    mutated_tape_identity: bool = False,
    reordered_rows: bool = False,
) -> Path:
    expected_cases = _expected_training_case_contracts()
    contract = {
        "tool": "run_training_oracle_trajectories.py",
        "split": {
            "id": "train",
            "family_count": 6,
            "family_ids": list(dict.fromkeys(case["family_id"] for case in expected_cases)),
            "cartesian_case_count": 192,
            "seed_interval": {"first": 810000, "last": 810031, "count": 32},
        },
        "ordered_case_contract_sha256": canonical_hash(expected_cases),
        "access_contract": {
            "training_split_used": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": False,
        },
        "oracle_config": EXPECTED_ORACLE_CONFIG,
        "demonstration_contract": {
            "case_count": 192,
            "horizon_days": 30,
            "row_count": 5760,
            "observation_count": 73,
            "action_count": 22,
            "observation_order": list(OBSERVATION_ORDER),
            "action_order": list(ACTION_ORDER),
            "student_input_future_tape_visible": False,
            "teacher_target_uses_full_future_tape": True,
        },
        "source_identity": {
            relative_path: file_sha256(Path(__file__).parents[1] / relative_path)
            for relative_path in DATASET_SOURCE_PATHS
        },
    }
    contract_sha256 = canonical_hash(contract)
    _atomic_create_json(
        root / "protocol.json",
        {"contract_sha256": contract_sha256, "contract": contract},
    )
    rows: list[dict[str, Any]] = []
    dataset_index: list[dict[str, str]] = []
    for index, expected_case in enumerate(expected_cases):
        case_identity = dict(expected_case)
        if fabricated_identity and index == 0:
            case_identity["family_id"] = "v3_train_fabricated"
            case_identity["row_id"] = (
                f"v3_train_fabricated:{case_identity['case_seed']}"
            )
        if mutated_tape_identity and index == 0:
            case_identity["tape_sha256"] = "f" * 64
        row_id = case_identity["row_id"]
        demonstration = _demonstration(row_id)
        if extra_input_field and index == 0:
            demonstration["future_tape"] = ["forbidden"]
        shard = {
            "contract_sha256": contract_sha256,
            "split": "train",
            "phase": "oracle",
            "index": index,
            "case": case_identity,
            "payload": {
                "row_id": row_id,
                "demonstration": demonstration,
            },
        }
        relative = f"training/oracle/{index:03d}.json"
        shard_path = root / relative
        _atomic_create_json(shard_path, shard)
        dataset_sha256 = demonstration["dataset_sha256"]
        rows.append(
            {
                **case_identity,
                "row_id": row_id,
                "shard": relative,
                "shard_sha256": file_sha256(shard_path),
                "dataset_sha256": dataset_sha256,
            }
        )
        dataset_index.append(
            {"row_id": row_id, "dataset_sha256": dataset_sha256}
        )
    if reordered_rows:
        rows[0], rows[1] = rows[1], rows[0]
        dataset_index[0], dataset_index[1] = dataset_index[1], dataset_index[0]
    invariants = {
        "case_count_exactly_192": True,
        "demonstration_rows_exactly_5760": True,
        "row_ids_unique": True,
        "observation_dimension_exactly_73": True,
        "action_dimension_exactly_22": True,
        "all_hard_violation_counts_zero": True,
        "all_conservation_residuals_exactly_zero": True,
        "development_split_used": False,
        "final_split_used": False,
        "learned_policy_loaded_or_run": False,
    }
    _atomic_create_json(
        root / "training" / "receipt.json",
        {
            "tool": "run_training_oracle_trajectories.py",
            "status": "complete_training_oracle_demonstrations",
            "contract_sha256": contract_sha256,
            "case_count": TRAINING_CASE_COUNT,
            "demonstration_row_count": DATASET_ROW_COUNT,
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "student_trained": False,
            "rows": rows,
            "rows_sha256": canonical_hash(rows),
            "dataset_index_sha256": canonical_hash(dataset_index),
            "invariants": invariants,
        },
    )
    return root


def test_loads_exactly_5760_public_rows_from_fake_shards(tmp_path: Path) -> None:
    dataset = load_oracle_dataset(_fake_dataset(tmp_path / "dataset"))

    assert dataset.observations.shape == (5760, 73)
    assert dataset.oracle_targets.shape == (5760, 22)
    assert dataset.hand_rule_targets.shape == (5760, 22)
    assert dataset.observations.dtype == np.float32
    assert dataset.oracle_targets.dtype == np.float32
    assert dataset.hand_rule_targets.dtype == np.float32
    assert not dataset.observations.flags.writeable
    assert not dataset.oracle_targets.flags.writeable
    assert not dataset.hand_rule_targets.flags.writeable
    assert len(dataset.row_ids) == 192
    assert dataset.fit_indices.shape == (FIT_ROW_COUNT,) == (5040,)
    assert dataset.holdout_indices.shape == (HOLDOUT_ROW_COUNT,) == (720,)
    assert dataset.split_contract["fit"]["trajectory_count"] == FIT_CASE_COUNT == 168
    assert (
        dataset.split_contract["holdout"]["trajectory_count"]
        == HOLDOUT_CASE_COUNT
        == 24
    )
    assert all(
        int(row_id.rsplit(":", 1)[1]) <= 810027
        for row_id in dataset.split_contract["fit"]["trajectory_row_ids"]
    )
    assert all(
        810028 <= int(row_id.rsplit(":", 1)[1]) <= 810031
        for row_id in dataset.split_contract["holdout"][
            "trajectory_row_ids"
        ]
    )
    assert len(
        dataset.split_contract["holdout"]["step_row_ids_sha256"]
    ) == 64
    contract = student_contract(dataset, {"development_solved_count": 152})
    assert contract["dataset"]["trajectory_split"] == dataset.split_contract
    assert contract["normalization"]["fit_once_on_5040_fit_observations"] is True
    assert contract["normalization"][
        "holdout_observations_excluded_from_rms"
    ] is True
    assert contract["offline_distillation_disclosure"] == {
        "teacher_forced_off_policy_dataset_collection_passes": 1,
        "static_dataset_optimization_epochs": 15,
        "interactive_relabelling": False,
        "dagger": False,
        "oracle_relabelled_student_states": False,
        "distribution_shift_resolved": False,
        "distribution_shift_is_a_separate_confound": True,
        "why_no_oracle_dagger": (
            "Full CEM oracle relabelling at student-visited states is not "
            "computationally affordable or safely supported by this fixed "
            "trajectory dataset."
        ),
    }
    assert len(contract["matched_control_reference"]) == 64
    assert contract["catastrophic_gate"]["development_condition"] == {
        "metric": "solved_count",
        "operator": ">=",
        "threshold": 140,
    }


def test_rejects_any_unregistered_demonstration_field(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path / "dataset", extra_input_field=True)

    with pytest.raises(OracleBCError, match="demonstration fields drifted"):
        load_oracle_dataset(root)


@pytest.mark.parametrize(
    "fixture_option",
    ("fabricated_identity", "mutated_tape_identity", "reordered_rows"),
)
def test_rejects_noncanonical_training_case_identity(
    tmp_path: Path,
    fixture_option: str,
) -> None:
    root = _fake_dataset(
        tmp_path / "dataset",
        **{fixture_option: True},
    )

    with pytest.raises(OracleBCError, match="canonical case identity drifted"):
        load_oracle_dataset(root)


def test_validates_selected_seed_old_bc_anchor(tmp_path: Path) -> None:
    path = tmp_path / "old-bc.json"
    _atomic_create_json(
        path,
        {
            "tool": "train_policy.py",
            "training_split": "train",
            "final_split_used": False,
            "config": {"policy_seed": POLICY_SEED},
            "development_curve": {
                "bc_initialization": {
                    "solved_count": OLD_BC_SOLVED_COUNT,
                    "case_count": DEVELOPMENT_CASE_COUNT,
                }
            },
            "behavior_cloning": {
                "teacher": "preparedness_teacher_action",
                "training_split_only": True,
                "observation_normalization": True,
                "iterations": 4,
                "dagger_beta_schedule": [1.0, 0.0, 0.0, 0.0],
                "observation_count": 23040,
                "epochs_per_iteration": 15,
                "batch_size": 512,
            },
        },
    )

    anchor = validate_old_bc_anchor(path, expected_sha256=file_sha256(path))

    assert anchor["policy_seed"] == 67017
    assert anchor["development_solved_count"] == 152
    assert anchor["training_method"] == "behavior_cloning_with_dagger"
    assert anchor["dagger_iterations"] == 4
    assert anchor["training_observation_count"] == 23040
    with pytest.raises(OracleBCError, match="hash mismatch"):
        validate_old_bc_anchor(path, expected_sha256="0" * 64)


def test_matched_bc_fits_use_trajectory_holdout_and_preserve_critics() -> None:
    rng = np.random.default_rng(67017)
    observations = rng.normal(size=(DATASET_ROW_COUNT, OBSERVATION_SIZE)).astype(
        np.float32
    )
    oracle_targets = np.tanh(
        observations[:, :ACTION_SIZE] * np.float32(0.25)
    ).astype(np.float32)
    hand_rule_targets = np.tanh(
        observations[:, :ACTION_SIZE] * np.float32(-0.15)
    ).astype(np.float32)
    row_ids = tuple(
        case["row_id"] for case in _expected_training_case_contracts()
    )
    fit_indices, holdout_indices, split_contract = _trajectory_split_contract(
        row_ids,
        observations,
        oracle_targets,
        hand_rule_targets,
    )
    dataset = OracleDataset(
        observations=observations,
        oracle_targets=oracle_targets,
        hand_rule_targets=hand_rule_targets,
        row_ids=row_ids,
        fit_indices=fit_indices,
        holdout_indices=holdout_indices,
        split_contract=split_contract,
        receipt_path=Path("fake-receipt.json"),
        receipt_sha256="a" * 64,
        contract_sha256="b" * 64,
        dataset_index_sha256="c" * 64,
    )

    fit = fit_student(dataset)
    try:
        layers = [
            layer
            for layer in fit.model.policy.mlp_extractor.policy_net
            if isinstance(layer, torch.nn.Linear)
        ]
        activations = [
            layer
            for layer in fit.model.policy.mlp_extractor.policy_net
            if isinstance(layer, torch.nn.SiLU)
        ]
        assert [(layer.in_features, layer.out_features) for layer in layers] == [
            (73, 384),
            (384, 256),
            (256, 128),
        ]
        assert len(activations) == 3
        assert fit.report["policy_seed"] == POLICY_SEED
        assert fit.report["epochs"] == BC_EPOCHS == 15
        assert fit.report["batch_size"] == BC_BATCH_SIZE == 512
        assert fit.report["learning_rate"] == BC_LEARNING_RATE == 1e-3
        oracle = fit.report["oracle_label_student"]
        control = fit.report["matched_hand_rule_control"]
        assert oracle["critic_unchanged"] is True
        assert control["critic_unchanged"] is True
        assert fit.report["matched_initialization"] == {
            "actor_state_sha256": oracle["actor_state_sha256_before"],
            "critic_state_sha256": oracle["critic_state_sha256_before"],
            "actor_hashes_equal_before_fit": True,
            "critic_hashes_equal_before_fit": True,
            "observation_rms_hashes_equal": True,
            "minibatch_permutations_identical": True,
        }
        assert (
            oracle["actor_state_sha256_before"]
            == control["actor_state_sha256_before"]
        )
        assert (
            oracle["critic_state_sha256_before"]
            == control["critic_state_sha256_before"]
        )
        assert (
            oracle["actor_state_sha256_after"]
            != control["actor_state_sha256_after"]
        )
        assert fit.report["dagger_iterations"] == 0
        assert fit.report["ppo_updates"] == 0
        assert fit.model.num_timesteps == 0
        assert fit.report["training_row_count_per_student"] == 5040
        assert fit.report["heldout_row_count_per_student"] == 720
        assert fit.report["observation_rms_count"] == pytest.approx(5040.0001)
        for treatment in (oracle, control):
            assert treatment["fit"]["trained"]["mse"] < treatment["fit"][
                "untrained"
            ]["mse"]
            assert treatment["heldout"]["trained"]["mse"] < treatment[
                "heldout"
            ]["untrained"]["mse"]
            assert treatment["heldout"]["relative_mse_improvement"] > 0.01
            assert len(
                treatment["heldout"]["trained"][
                    "mean_absolute_error_by_dimension"
                ]
            ) == 22
            assert treatment["heldout"]["trained"][
                "mean_absolute_error"
            ] == pytest.approx(
                np.mean(
                    treatment["heldout"]["trained"][
                        "mean_absolute_error_by_dimension"
                    ]
                )
            )
    finally:
        fit.normalizer.close()


def test_catastrophic_gate_requires_140_and_strictly_more_than_one_percent() -> None:
    assert DEVELOPMENT_CATASTROPHIC_FLOOR == 140
    assert HOLDOUT_RELATIVE_MSE_IMPROVEMENT_FLOOR == 0.01
    assert catastrophic_gate(140, 0.0100001)["passed"] is True
    assert catastrophic_gate(140, 0.0100001)["decision"] == (
        "eligible_for_separately_authorized_3_seed_ppo"
    )
    below_development = catastrophic_gate(139, 0.5)
    assert below_development["passed"] is False
    assert below_development["failed_conditions"] == [
        "development_solved_count_below_140"
    ]
    at_improvement_floor = catastrophic_gate(200, 0.01)
    assert at_improvement_floor["passed"] is False
    assert at_improvement_floor["failed_conditions"] == [
        "oracle_holdout_mse_improvement_at_or_below_1pct"
    ]
    invalid = catastrophic_gate(200, float("nan"))
    assert invalid["passed"] is False
    assert invalid["conditions"][
        "oracle_holdout_relative_mse_improvement"
    ]["observed"] is None
    assert invalid["failed_conditions"] == [
        "oracle_holdout_mse_improvement_invalid"
    ]


def _fake_development(solved_count: int) -> dict[str, Any]:
    identities = _expected_development_identity()
    return {
        "case_count": DEVELOPMENT_CASE_COUNT,
        "solved_count": solved_count,
        "solve_rate": solved_count / DEVELOPMENT_CASE_COUNT,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "rows": [
            {
                **identity,
                "solved": index < solved_count,
                "hard_violation_count": 0,
                "max_conservation_residual": 0.0,
            }
            for index, identity in enumerate(identities)
        ],
    }


def test_development_gate_aggregates_are_derived_from_canonical_rows() -> None:
    valid = _fake_development(163)
    assert _validate_development(valid)["solved_count"] == 163

    wrong_count = copy.deepcopy(valid)
    wrong_count["solved_count"] = 164
    with pytest.raises(OracleBCError, match="aggregates do not match"):
        _validate_development(wrong_count)

    wrong_rate = copy.deepcopy(valid)
    wrong_rate["solve_rate"] = 0.82
    with pytest.raises(OracleBCError, match="aggregates do not match"):
        _validate_development(wrong_rate)

    hidden_violation = copy.deepcopy(valid)
    hidden_violation["rows"][0]["hard_violation_count"] = 1
    with pytest.raises(OracleBCError, match="aggregates do not match"):
        _validate_development(hidden_violation)

    hidden_residual = copy.deepcopy(valid)
    hidden_residual["rows"][0]["max_conservation_residual"] = 1e-8
    with pytest.raises(OracleBCError, match="aggregates do not match"):
        _validate_development(hidden_residual)


def _fake_fit(normalizer: Any, improvement: float = 0.02) -> StudentFit:
    return StudentFit(
        model=cast(PPO, object()),
        normalizer=normalizer,
        report={
            "dagger_iterations": 0,
            "ppo_updates": 0,
            "oracle_label_student": {
                "heldout": {
                    "untrained": {"mse": 1.0},
                    "trained": {"mse": 1.0 - improvement},
                    "relative_mse_improvement": improvement,
                }
            },
            "matched_hand_rule_control": {
                "heldout": {
                    "untrained": {"mse": 0.5},
                    "trained": {"mse": 0.25},
                    "relative_mse_improvement": 0.5,
                }
            },
        },
    )


def test_development_evaluation_is_claimed_once(tmp_path: Path) -> None:
    normalizer = _model_environment()
    normalizer.training = False
    fit = _fake_fit(normalizer)
    fit_success = {"checkpoint_bundle": {"checkpoint_id": "fake"}}
    _atomic_create_json(tmp_path / "fit.success.json", fit_success)
    calls = 0

    def evaluator(_: PPO, __: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_development(163)

    try:
        receipt = evaluate_once(
            output_root=tmp_path,
            contract_sha256="c" * 64,
            fit=fit,
            fit_success=fit_success,
            evaluator=evaluator,
        )
        assert calls == 1
        assert receipt["catastrophic_gate"]["passed"] is True
        assert result_exit_code(receipt) == 0
        comparison = receipt["comparison"]
        assert comparison["old_selected_seed_bc"] == {
            "development_solved_count": 152,
            "method": "behavior_cloning_with_4_dagger_rounds",
            "training_observation_count": 23040,
        }
        assert comparison["new_oracle_bc"] == {
            "development_solved_count": 163,
            "method": "single_fixed_offline_behavior_cloning_fit",
            "training_observation_count": 5040,
            "heldout_observation_count": 720,
        }
        assert comparison["matched_hand_rule_control"] == {
            "development_evaluated": False,
            "method": "single_fixed_offline_behavior_cloning_fit",
            "label_source": "preparedness_teacher_action",
            "training_observation_count": 5040,
            "heldout_observation_count": 720,
            "heldout": fit.report["matched_hand_rule_control"]["heldout"],
            "distinct_from_historical_old_bc": True,
        }
        assert comparison["delta"] == 11
        assert comparison["like_for_like_training_volume_claimed"] is False
        with pytest.raises(OracleBCError, match="terminal receipt"):
            evaluate_once(
                output_root=tmp_path,
                contract_sha256="c" * 64,
                fit=fit,
                fit_success=fit_success,
                evaluator=evaluator,
            )
        assert calls == 1
    finally:
        normalizer.close()


def test_negative_gate_is_completed_exit_four_not_retryable() -> None:
    receipt = {
        "catastrophic_gate": catastrophic_gate(139, 0.5),
        "completed": True,
        "retry_recommended": False,
    }

    assert result_exit_code(receipt) == COMPLETED_NEGATIVE_GATE_EXIT_CODE == 4
    assert receipt["catastrophic_gate"]["decision"] == "abort_before_ppo"
    assert receipt["catastrophic_gate"]["completed"] is True
    assert receipt["catastrophic_gate"]["retry_recommended"] is False


def test_failed_claim_is_terminal_and_observation_rms_mutation_is_detected(
    tmp_path: Path,
) -> None:
    normalizer = _model_environment()
    normalizer.training = False
    fit = _fake_fit(normalizer)
    fit_success = {"checkpoint_bundle": {"checkpoint_id": "fake"}}
    _atomic_create_json(tmp_path / "fit.success.json", fit_success)
    calls = 0

    def mutating_evaluator(_: PPO, __: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        normalizer.obs_rms.mean[0] += 1.0
        return _fake_development(163)

    try:
        with pytest.raises(OracleBCError, match="changed observation RMS"):
            evaluate_once(
                output_root=tmp_path,
                contract_sha256="d" * 64,
                fit=fit,
                fit_success=fit_success,
                evaluator=mutating_evaluator,
            )
        assert calls == 1
        failure = json.loads(
            (tmp_path / "development-evaluation.failure.json").read_text(
                encoding="utf-8"
            )
        )
        assert failure["retry_permitted"] is False
        assert failure["error_type"] == "OracleBCError"
        with pytest.raises(OracleBCError, match="terminal receipt"):
            evaluate_once(
                output_root=tmp_path,
                contract_sha256="d" * 64,
                fit=fit,
                fit_success=fit_success,
                evaluator=mutating_evaluator,
            )
        assert calls == 1
    finally:
        normalizer.close()


def test_source_has_no_final_roster_or_online_training_path() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "train_oracle_bc_student.py"
    ).read_text(encoding="utf-8")

    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert "policy_rollout_dataset" not in source
    assert "behavior_clone_policy" not in source
    assert ".learn(" not in source
    assert '"dagger_iterations": 0' in source
    assert '"ppo_updates": 0' in source
    assert "preparedness_teacher_action(observation)" in source
    assert "HOLDOUT_SEEDS = tuple(range(810028, 810032))" in source
    assert "FIT_ROW_COUNT = FIT_CASE_COUNT * HORIZON_DAYS" in source
    assert "holdout_observations_excluded_from_rms" in source
