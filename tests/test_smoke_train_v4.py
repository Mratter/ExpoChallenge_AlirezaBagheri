from __future__ import annotations

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.scenarios_v3 import TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3
from backend.app.simulator_v4 import CityRecoveryEnvV4
from scripts.smoke_train_v4 import (
    actor_state,
    build_model,
    development_curve_evaluation,
    early_stop_row_summary,
    freeze_actor_for_critic_warmup,
    reported_approx_kl_summary,
    state_digest,
    unfreeze_policy,
)


def test_receipt_helpers_make_transition_and_optimizer_semantics_explicit() -> None:
    evaluation = {"solved_count": 32}
    point = development_curve_evaluation(
        evaluation,
        active_actor_critic_transitions=50_000,
        total_environment_transitions=100_000,
    )
    assert point == {
        "active_actor_critic_transitions": 50_000,
        "total_environment_transitions": 100_000,
        "solved_count": 32,
    }
    assert evaluation == {"solved_count": 32}

    iterations = [
        {
            "approx_kl": 0.0075,
            "early_stop_detected_before_final_epoch": True,
        },
        {
            "approx_kl": 0.002,
            "early_stop_detected_before_final_epoch": False,
        },
    ]
    kl_summary = reported_approx_kl_summary(iterations, target_kl=0.005)
    assert kl_summary["reported_approx_kl_stability_limit"] == 0.0075
    assert kl_summary["reported_approx_kl_stability_multiplier"] == 1.5
    assert kl_summary["reported_approx_kl_stable"] is True
    assert "1.5 * configured target_kl" in kl_summary[
        "reported_approx_kl_stability_definition"
    ]
    assert early_stop_row_summary(iterations) == {
        "iteration_row_count": 2,
        "early_stop_row_count": 1,
        "full_epoch_row_count": 1,
    }

    iterations[0]["approx_kl"] = 0.0075001
    assert (
        reported_approx_kl_summary(iterations, target_kl=0.005)[
            "reported_approx_kl_stable"
        ]
        is False
    )


def test_instrumented_critic_warmup_keeps_actor_byte_identical() -> None:
    case_seed = TRAINING_SEEDS_V3[0]
    scenario = TRAINING_FAMILIES_V3[0].build(case_seed)
    tape_seed = TRAINING_FAMILIES_V3[0].tape_seed(case_seed)
    environment = DummyVecEnv(
        [
            lambda: CityRecoveryEnvV4(
                scenario,
                tape_seed,
                collect_evidence=False,
                reward_profile="v3_equivalent",
            )
        ]
    )
    try:
        model = build_model(
            environment,
            seed=37_017,
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
            not torch.equal(
                parameter.detach(), critic_before[name]
            )
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

        unfreeze_policy(model)
        assert all(
            parameter.requires_grad for parameter in model.policy.parameters()
        )
    finally:
        environment.close()
