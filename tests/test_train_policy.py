from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.city.environment import CityRecoveryEnv
from backend.app.city.scenarios import TRAINING_FAMILIES, TRAINING_SEEDS
from scripts import train_policy
from scripts.train_policy import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CRITIC_WARMUP_MIN_TRANSITIONS,
    DEFAULT_ENT_COEF,
    DEFAULT_LANES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_POLICY_SEED,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_TARGET_KL,
    DEFAULT_TRANSITIONS,
    CANONICAL_DEVELOPMENT_CASE_COUNT,
    DEVELOPMENT_CASE_COUNT,
    TrainingError,
    actor_state,
    build_model,
    development_curve_evaluation,
    diagnostic_rows_valid,
    early_stop_row_summary,
    freeze_actor_for_critic_warmup,
    learning_milestones,
    load_rms_state,
    parse_args,
    return_rms_continuity_valid,
    rms_digest,
    rms_state,
    state_digest,
    target_kl_diagnostics,
    unfreeze_policy,
    validate_runtime_config,
    write_receipt,
)


def test_canonical_defaults_match_the_adopted_optimizer_regime(
    tmp_path: Path,
) -> None:
    args = parse_args(["--json-output", str(tmp_path / "receipt.json")])

    assert args.transitions == DEFAULT_TRANSITIONS == 8_000_000
    assert args.lanes == DEFAULT_LANES == 20
    assert args.batch_size == DEFAULT_BATCH_SIZE == 500
    assert args.policy_seed == DEFAULT_POLICY_SEED == 37_017
    assert args.learning_rate == DEFAULT_LEARNING_RATE == 7.5e-5
    assert args.target_kl == DEFAULT_TARGET_KL == 0.02
    assert args.ent_coef == DEFAULT_ENT_COEF == 0.003
    assert args.reward_profile == DEFAULT_REWARD_PROFILE == "v3_equivalent"
    assert args.preparedness_alignment_coefficient is None
    assert args.bc_warm_start is True
    assert args.vec_normalize is True
    assert (
        args.critic_warmup_min_transitions
        == DEFAULT_CRITIC_WARMUP_MIN_TRANSITIONS
        == 50_000
    )
    assert args.freeze_observation_rms is True
    assert validate_runtime_config(args) == 5_000
    assert DEVELOPMENT_CASE_COUNT == CANONICAL_DEVELOPMENT_CASE_COUNT == 200


def test_learning_milestones_make_the_training_curve_visible() -> None:
    assert learning_milestones(100_000, 5_000) == [100_000]
    assert learning_milestones(200_000, 5_000) == [200_000]
    assert learning_milestones(1_000_000, 5_000) == [
        200_000,
        500_000,
        1_000_000,
    ]
    assert learning_milestones(8_000_000, 5_000) == [
        200_000,
        500_000,
        1_000_000,
        8_000_000,
    ]
    with pytest.raises(TrainingError, match="complete rollouts"):
        learning_milestones(200_001, 5_000)


def test_instrumented_critic_warmup_keeps_actor_byte_identical() -> None:
    case_seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(case_seed)
    tape_seed = TRAINING_FAMILIES[0].tape_seed(case_seed)
    environment = DummyVecEnv(
        [
            lambda: CityRecoveryEnv(
                scenario,
                tape_seed,
                collect_evidence=False,
            )
        ]
    )
    try:
        model = build_model(
            environment,
            seed=DEFAULT_POLICY_SEED,
            n_steps=2,
            batch_size=2,
        )
        actor_before = state_digest(actor_state(model))
        critic_before = {
            name: parameter.detach().clone()
            for name, parameter in model.policy.named_parameters()
            if name.startswith(("mlp_extractor.value_net", "value_net"))
        }

        trainable_count = freeze_actor_for_critic_warmup(model)
        assert trainable_count == sum(
            parameter.numel()
            for parameter in model.policy.parameters()
            if parameter.requires_grad
        )
        model.set_diagnostic_phase("critic_warmup")
        model.learn(total_timesteps=2, progress_bar=False)

        assert state_digest(actor_state(model)) == actor_before
        assert any(
            not torch.equal(parameter.detach(), critic_before[name])
            for name, parameter in model.policy.named_parameters()
            if name in critic_before
        )
        assert len(model.training_iterations) == 1
        metrics = model.training_iterations[0]
        assert metrics["phase"] == "critic_warmup"
        assert metrics["total_transitions"] == 2
        assert len(metrics["action_std_by_dimension"]) == 22
        assert {
            "explained_variance",
            "approx_kl",
            "clip_fraction",
            "entropy_loss",
            "value_loss",
            "policy_gradient_loss",
            "action_std_mean",
        } <= set(metrics)
        assert diagnostic_rows_valid([metrics], expected_count=1)

        unfreeze_policy(model)
        assert all(
            parameter.requires_grad for parameter in model.policy.parameters()
        )
    finally:
        environment.close()


def test_target_kl_is_an_early_stop_diagnostic_not_a_reported_maximum_gate() -> None:
    iterations = [
        {
            "approx_kl": 0.037,
            "early_stop_detected_before_final_epoch": True,
            "target_kl_guard_enabled": True,
        },
        {
            "approx_kl": 0.012,
            "early_stop_detected_before_final_epoch": False,
            "target_kl_guard_enabled": True,
        },
    ]

    summary = target_kl_diagnostics(iterations, target_kl=0.02)
    assert summary["reported_approx_kl_max"] == 0.037
    assert summary["reported_approx_kl_rows_above_target"] == 1
    assert summary["target_kl_guard_enabled_on_every_iteration"] is True
    assert summary["early_stop_detected_before_final_epoch_count"] == 1
    assert summary["additional_reported_kl_ceiling_applied"] is False
    assert "reported maximum above the target is expected" in summary["semantics"]
    assert early_stop_row_summary(iterations) == {
        "iteration_row_count": 2,
        "early_stop_row_count": 1,
        "full_epoch_row_count": 1,
    }


def test_return_rms_continues_without_reset_across_training_phases() -> None:
    result = {
        "critic_warmup": {
            "return_rms_before_sha256": "initial",
            "return_rms_before_count": 0.0001,
            "return_rms_after_count": 50_000.0001,
        },
        "normalization": {"return_rms_count": 1_050_000.0001},
    }
    assert return_rms_continuity_valid(
        result,
        "initial",
        warmup_transitions=50_000,
        active_transitions=1_000_000,
    ) is True

    result["normalization"]["return_rms_count"] = 1_000_000.0001
    assert not return_rms_continuity_valid(
        result,
        "initial",
        warmup_transitions=50_000,
        active_transitions=1_000_000,
    )


def test_rms_state_round_trip_preserves_normalization_identity() -> None:
    source = RunningMeanStd(shape=(3,))
    source.update(np.asarray([[1.0, 2.0, 3.0], [3.0, 5.0, 8.0]]))
    copied = rms_state(source)
    target = RunningMeanStd(shape=(3,))
    load_rms_state(target, copied)

    assert rms_digest(rms_state(target)) == rms_digest(copied)


def test_curve_points_copy_transition_accounting_without_mutating_results() -> None:
    evaluation = {"solved_count": 35}
    point = development_curve_evaluation(
        evaluation,
        active_actor_critic_transitions=500_000,
        total_environment_transitions=550_000,
    )
    assert point == {
        "active_actor_critic_transitions": 500_000,
        "total_environment_transitions": 550_000,
        "solved_count": 35,
    }
    assert evaluation == {"solved_count": 35}


def test_receipts_are_create_new_and_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "training-receipt.json"
    payload = {"tool": "train_policy.py", "final_split_used": False}

    write_receipt(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(TrainingError, match="refusing to overwrite"):
        write_receipt(path, {"replacement": True})


def test_trainer_uses_only_neutral_modules_and_never_accesses_final_split() -> None:
    source_path = Path(train_policy.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert source_path.name == "train_policy.py"
    assert train_policy.TOOL_ID == "train_policy.py"
    expected_app_imports = {
        "backend.app.city.environment",
        "backend.app.city.outcome",
        "backend.app.city.physics",
        "backend.app.city.planners",
        "backend.app.city.scenarios",
        "backend.app.shared_evidence",
    }
    assert {
        module for module in imported_modules if module.startswith("backend.app")
    } == expected_app_imports
    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert 'choices=("v3_equivalent", "risk_averse")' in source
    assert "--gate-mode" not in source
