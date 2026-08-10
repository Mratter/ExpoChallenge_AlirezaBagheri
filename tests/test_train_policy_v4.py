from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.scenarios_v3 import TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3
from backend.app.simulator_v4 import CityRecoveryEnvV4
from scripts.smoke_train_v4 import (
    ROOT,
    STEP3E_ACTIVE_MILESTONES,
    STEP3E_ACTIVE_TRANSITIONS,
    STEP3E_ATTEMPT06_200K_ROWS_SHA256,
    STEP3E_ATTEMPT06_BC_ROWS_SHA256,
    STEP3E_CONTESTED_ROW_IDS,
    STEP3E_HEADROOM_RECEIPT,
    STEP3E_OPTIMIZER_RECEIPT,
    PROTECTED_V3_EXTRA_FILES_SHA256,
    STEP3E_SUPERVISOR_GATE_CORRECTION,
    SmokeError,
    actor_state,
    build_model,
    contested_case_outcomes,
    development_curve_evaluation,
    early_stop_row_summary,
    freeze_actor_for_critic_warmup,
    historical_step3_rows_hash,
    learning_milestones,
    parse_args,
    protected_v3_snapshot,
    reported_approx_kl_summary,
    state_digest,
    step3e_carry_decision,
    step3e_return_rms_continuity_valid,
    step3e_target_kl_diagnostics,
    unfreeze_policy,
    validate_step3e_provenance,
    validate_step3e_runtime_config,
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


def _step3e_args() -> object:
    return parse_args(
        [
            "--gate-mode",
            "step3e",
            "--transitions",
            "1000000",
            "--lanes",
            "20",
            "--n-steps",
            "250",
            "--batch-size",
            "500",
            "--policy-seed",
            "37017",
            "--bc-epochs",
            "15",
            "--learning-rate",
            "7.5e-5",
            "--target-kl",
            "0.02",
            "--ent-coef",
            "0.003",
            "--critic-warmup-min-transitions",
            "50000",
            "--critic-warmup-max-transitions",
            "100000",
            "--critic-ev-threshold",
            "0.5",
            "--freeze-observation-rms",
            "--supervisor-step3e-authorization",
            "--json-output",
            "internal/developmental_runs/v4/test-step3e-unused.json",
        ]
    )


def _evaluation(
    solved_count: int,
    contested_solved: int,
    margin: float,
    auc: float,
) -> dict[str, object]:
    rows = [
        {
            "row_id": row_id,
            "solved": index < contested_solved,
            "minimum_tail_margin": margin,
            "resilience_auc": auc,
            "reason_codes": [] if index < contested_solved else ["tail"],
        }
        for index, row_id in enumerate(STEP3E_CONTESTED_ROW_IDS)
    ]
    return {
        "solved_count": solved_count,
        "mean_minimum_tail_margin": margin,
        "mean_resilience_auc": auc,
        "rows": rows,
    }


def test_step3e_mode_is_exact_supervisor_authorized_attempt06_regime() -> None:
    args = _step3e_args()
    validate_step3e_runtime_config(args, rollout_size=5_000)
    assert learning_milestones(
        STEP3E_ACTIVE_TRANSITIONS, 5_000, "step3e"
    ) == list(STEP3E_ACTIVE_MILESTONES)

    args.transitions = 200_000
    with pytest.raises(SmokeError, match="exact supervisor-adopted"):
        validate_step3e_runtime_config(args, rollout_size=5_000)

    args = _step3e_args()
    args.supervisor_step3e_authorization = False
    with pytest.raises(SmokeError, match="requires --supervisor"):
        validate_step3e_runtime_config(args, rollout_size=5_000)


def test_step3e_provenance_derives_and_validates_four_contested_cases() -> None:
    provenance = validate_step3e_provenance(
        STEP3E_HEADROOM_RECEIPT, STEP3E_OPTIMIZER_RECEIPT
    )
    assert provenance["original_compound_gate_passed"] is False
    assert provenance["corrected_step_3d_declared_passed"] is True
    assert tuple(
        provenance["headroom_receipt"]["contested_row_ids"]
    ) == STEP3E_CONTESTED_ROW_IDS
    assert provenance["optimizer_receipt"]["attempt"] == 6
    assert "+2 solves" in STEP3E_SUPERVISOR_GATE_CORRECTION
    assert "2.5%" in STEP3E_SUPERVISOR_GATE_CORRECTION

    protected = protected_v3_snapshot(
        provenance["protected_v3_expected_files_sha256"]
    )
    assert protected["file_count"] == len(
        {
            **provenance["protected_v3_expected_files_sha256"],
            **PROTECTED_V3_EXTRA_FILES_SHA256,
        }
    )


def test_step3e_historical_row_projection_matches_attempt06() -> None:
    receipt = json.loads(STEP3E_OPTIMIZER_RECEIPT.read_text(encoding="utf-8"))
    curve = receipt["profiles"]["v3_equivalent"]["development_curve"]
    bc_rows = curve["bc_initialization"]["rows"]
    rows_200k = curve["active_actor_critic_200000_transitions"]["rows"]
    assert historical_step3_rows_hash(bc_rows) == STEP3E_ATTEMPT06_BC_ROWS_SHA256
    assert (
        historical_step3_rows_hash(rows_200k)
        == STEP3E_ATTEMPT06_200K_ROWS_SHA256
    )

    enriched = [dict(row, minimum_tail_margin=0.123) for row in rows_200k]
    assert (
        historical_step3_rows_hash(enriched)
        == STEP3E_ATTEMPT06_200K_ROWS_SHA256
    )


def test_step3e_contested_conversion_and_preregistered_carry_rule() -> None:
    control = _evaluation(33, 1, 0.02, 0.48)
    risk_500k = _evaluation(34, 2, 0.03, 0.49)
    risk_final = _evaluation(35, 3, 0.04, 0.50)
    contested = contested_case_outcomes(risk_final, STEP3E_CONTESTED_ROW_IDS)
    assert contested["converted_from_step3_best_ppo_unsolved_count"] == 3
    assert contested["converted_row_ids"] == list(STEP3E_CONTESTED_ROW_IDS[:3])

    carry = step3e_carry_decision(
        control, risk_500k, risk_final, STEP3E_CONTESTED_ROW_IDS
    )
    assert carry["risk_averse_ahead_on_final_solved_count"] is True
    assert carry["risk_averse_still_rising_500k_to_1m"] is True
    assert carry["decision"] == "carry_risk_averse"

    tied = _evaluation(33, 3, 0.05, 0.51)
    carry = step3e_carry_decision(
        control, risk_500k, tied, STEP3E_CONTESTED_ROW_IDS
    )
    assert carry["decision"] == "carry_v3_equivalent"
    assert "indistinguishable" in carry["interpretation"]

    plateaued = _evaluation(34, 2, 0.03, 0.49)
    carry = step3e_carry_decision(
        control, risk_500k, plateaued, STEP3E_CONTESTED_ROW_IDS
    )
    assert carry["decision"] == "carry_v3_equivalent"
    assert "inconclusive" in carry["interpretation"]


def test_step3e_kl_is_logged_as_early_stop_diagnostic_not_ceiling() -> None:
    iterations = [
        {
            "approx_kl": 0.031,
            "early_stop_detected_before_final_epoch": True,
            "target_kl_guard_enabled": True,
        }
    ]
    diagnostic = step3e_target_kl_diagnostics(iterations, target_kl=0.02)
    assert diagnostic["reported_approx_kl_max_diagnostic"] == 0.031
    assert diagnostic["early_stop_detected_before_final_epoch_count"] == 1
    assert diagnostic["obsolete_max_kl_ceiling_applied"] is False
    assert "reported_approx_kl_stable" not in diagnostic


def test_step3e_return_rms_continuity_allows_treatment_hash_divergence() -> None:
    result = {
        "critic_warmup": {
            "return_rms_before_sha256": "initial",
            "return_rms_before_count": 0.0001,
            "return_rms_after_sha256": "warm",
            "return_rms_after_count": 50_000.0001,
        },
        "vecnormalize": {
            "return_rms_sha256": "reward-specific-final",
            "return_rms_count": 1_050_000.0001,
        },
    }
    assert step3e_return_rms_continuity_valid(
        result,
        "initial",
        warmup_transitions=50_000,
        active_transitions=1_000_000,
    )
    result["vecnormalize"]["return_rms_count"] = 1_000_000.0001
    assert not step3e_return_rms_continuity_valid(
        result,
        "initial",
        warmup_transitions=50_000,
        active_transitions=1_000_000,
    )


def test_step3e_runner_has_no_final_split_import_or_policy_access() -> None:
    source = (ROOT / "scripts" / "smoke_train_v4.py").read_text(encoding="utf-8")
    assert "FINAL_FAMILIES_V3" not in source
    assert "FINAL_SEEDS_V3" not in source
    assert "authorize-final" not in source
    assert "onnxruntime" not in source
    assert "load_policy" not in source
    assert Path(STEP3E_HEADROOM_RECEIPT).is_file()
