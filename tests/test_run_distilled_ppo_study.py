from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import run_distilled_ppo_study as study
from scripts import train_policy


def _policy_state(actor_value: float, critic_value: float) -> dict[str, torch.Tensor]:
    return {
        "log_std": torch.full((2,), actor_value),
        "mlp_extractor.policy_net.0.weight": torch.full(
            (3, 2), actor_value
        ),
        "mlp_extractor.policy_net.0.bias": torch.full((3,), actor_value),
        "action_net.weight": torch.full((2, 3), actor_value),
        "action_net.bias": torch.full((2,), actor_value),
        "mlp_extractor.value_net.0.weight": torch.full(
            (3, 2), critic_value
        ),
        "mlp_extractor.value_net.0.bias": torch.full((3,), critic_value),
        "value_net.weight": torch.full((1, 3), critic_value),
        "value_net.bias": torch.full((1,), critic_value),
    }


def _student_reference(
    actor_sha256: str = "a" * 64,
    rms_sha256: str = "b" * 64,
) -> dict[str, Any]:
    base = {
        "root": "E:\\synthetic-student",
        "student_receipt": {
            "path": "E:\\synthetic-student\\student-receipt.json",
            "sha256": "c" * 64,
        },
        "student_contract_sha256": "d" * 64,
        "dataset_receipt_sha256": "e" * 64,
        "dataset_components": {
            "observations_sha256": "1" * 64,
            "oracle_targets_sha256": "2" * 64,
            "step_row_ids_sha256": "3" * 64,
        },
        "method": "single_pass_offline_oracle_behavior_cloning_no_dagger",
        "distribution_shift_disclosure": {
            "interactive_relabelling": False,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
        },
        "heldout_fit": {
            "oracle_action_mse": 0.04,
            "oracle_action_mean_absolute_error": 0.14,
            "hand_rule_action_mse": 0.02,
            "hand_rule_action_mean_absolute_error": 0.09,
        },
        "bc_development_solved_count": 157,
        "catastrophic_gate_passed": True,
        "checkpoint": {
            "root": "E:\\synthetic-student\\bc-checkpoint",
            "manifest_path": "E:\\synthetic-student\\bc-checkpoint\\manifest.json",
            "manifest_sha256": "f" * 64,
            "model_path": "E:\\synthetic-student\\bc-checkpoint\\model.zip",
            "model_sha256": "4" * 64,
            "normalization_path": "E:\\synthetic-student\\bc-checkpoint\\normalization.npz",
            "normalization_sha256": "5" * 64,
            "policy_state_sha256": "6" * 64,
            "actor_state_sha256": actor_sha256,
            "observation_rms_sha256": rms_sha256,
            "return_rms_sha256": "7" * 64,
        },
        "final_split_imported_or_used": False,
    }
    return {**base, "reference_sha256": canonical_hash(base)}


def test_actor_transplant_keeps_distilled_actor_and_fresh_critic() -> None:
    fresh = _policy_state(actor_value=-1.0, critic_value=37.0)
    distilled = _policy_state(actor_value=9.0, critic_value=-99.0)
    merged = study.merge_distilled_actor(fresh, distilled)

    assert train_policy.state_digest(study._actor_entries(merged)) == (
        train_policy.state_digest(study._actor_entries(distilled))
    )
    assert train_policy.state_digest(study._critic_entries(merged)) == (
        train_policy.state_digest(study._critic_entries(fresh))
    )
    assert train_policy.state_digest(study._critic_entries(merged)) != (
        train_policy.state_digest(study._critic_entries(distilled))
    )


def test_coordinator_and_worker_bind_same_critic_across_runtime_contexts() -> None:
    seed = 37_017
    coordinator_sha256 = study.fresh_critic_state_sha256(seed)
    worker_script = """
import json
import torch
from scripts import run_distilled_ppo_study as study

before = {
    "intraop_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
}
critic_sha256 = study.fresh_critic_state_sha256(37_017)
after = {
    "intraop_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
}
print(json.dumps({"before": before, "after": after, "critic": critic_sha256}))
"""
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "2"
    completed = subprocess.run(
        [sys.executable, "-c", worker_script],
        cwd=study.ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    worker = json.loads(completed.stdout)

    assert worker["before"] != worker["after"]
    assert worker["after"] == {
        "intraop_threads": study.TORCH_INTRAOP_THREADS,
        "interop_threads": study.TORCH_INTEROP_THREADS,
        "deterministic_algorithms": True,
        "warn_only": False,
    }
    assert worker["critic"] == coordinator_sha256


@pytest.mark.parametrize("mode", ("wrong", "extra"))
def test_trainer_torch_proxy_rejects_wrong_or_extra_setup_and_restores(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    original_train_policy_torch = train_policy.torch

    def drifted_main(_argv: list[str]) -> int:
        if mode == "wrong":
            train_policy.torch.set_num_threads(
                study.TORCH_INTRAOP_THREADS + 1
            )
        else:
            train_policy.torch.set_num_threads(study.TORCH_INTRAOP_THREADS)
            train_policy.torch.set_num_interop_threads(
                study.TORCH_INTEROP_THREADS
            )
            train_policy.torch.use_deterministic_algorithms(True)
            train_policy.torch.set_num_interop_threads(
                study.TORCH_INTEROP_THREADS
            )
        return 0

    monkeypatch.setattr(train_policy, "main", drifted_main)
    with pytest.raises(
        study.DistilledPPOStudyError,
        match="Torch runtime setup drifted",
    ):
        study._run_train_policy_main([])
    assert train_policy.torch is original_train_policy_torch


def test_trainer_torch_proxy_restores_when_trainer_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrainerFailure(RuntimeError):
        pass

    original_train_policy_torch = train_policy.torch

    def failed_main(_argv: list[str]) -> int:
        train_policy.torch.set_num_threads(study.TORCH_INTRAOP_THREADS)
        train_policy.torch.set_num_interop_threads(
            study.TORCH_INTEROP_THREADS
        )
        train_policy.torch.use_deterministic_algorithms(True)
        raise TrainerFailure("synthetic trainer failure")

    monkeypatch.setattr(train_policy, "main", failed_main)
    with pytest.raises(TrainerFailure, match="synthetic trainer failure"):
        study._run_train_policy_main([])
    assert train_policy.torch is original_train_policy_torch


def test_actor_transplant_rejects_schema_or_shape_drift() -> None:
    fresh = _policy_state(actor_value=1.0, critic_value=2.0)
    missing = _policy_state(actor_value=3.0, critic_value=4.0)
    del missing["action_net.bias"]
    with pytest.raises(study.DistilledPPOStudyError, match="schemas differ"):
        study.merge_distilled_actor(fresh, missing)

    wrong_shape = _policy_state(actor_value=3.0, critic_value=4.0)
    wrong_shape["action_net.bias"] = torch.ones(3)
    with pytest.raises(study.DistilledPPOStudyError, match="tensor contract"):
        study.merge_distilled_actor(fresh, wrong_shape)


def test_injected_initialization_uses_same_actor_rms_and_seeded_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distilled = _policy_state(actor_value=8.0, critic_value=-8.0)
    actor_sha256 = train_policy.state_digest(study._actor_entries(distilled))
    obs_mean = np.arange(73, dtype=np.float64)
    obs_var = np.arange(73, dtype=np.float64) + 1.0
    rms_state = {"mean": obs_mean, "var": obs_var, "count": 5040.0001}
    rms_sha256 = train_policy.rms_digest(rms_state)
    reference = _student_reference(actor_sha256, rms_sha256)
    loaded = SimpleNamespace(
        model=SimpleNamespace(
            policy=SimpleNamespace(state_dict=lambda: distilled)
        ),
        normalization=SimpleNamespace(
            obs_mean=obs_mean,
            obs_var=obs_var,
            obs_count=5040.0001,
        ),
    )
    fresh_by_seed = {
        seed: _policy_state(actor_value=-1.0, critic_value=float(seed))
        for seed in study.POLICY_SEEDS
    }
    monkeypatch.setattr(
        train_policy,
        "untrained_policy_state",
        lambda *, seed, **_kwargs: fresh_by_seed[seed],
    )
    legacy_calls = {"behavior_cloning": 0, "dagger": 0}

    def legacy_behavior_cloning_bomb() -> tuple[np.ndarray, np.ndarray]:
        legacy_calls["behavior_cloning"] += 1
        raise AssertionError("legacy hand-rule demonstrations were recollected")

    def legacy_dagger_bomb(*_args: Any, **_kwargs: Any) -> Any:
        legacy_calls["dagger"] += 1
        raise AssertionError("legacy DAgger collection was invoked")

    monkeypatch.setattr(
        train_policy, "behavior_cloning_dataset", legacy_behavior_cloning_bomb
    )
    monkeypatch.setattr(train_policy, "policy_rollout_dataset", legacy_dagger_bomb)

    original_behavior_clone = train_policy.behavior_clone_policy
    original_behavior_dataset = train_policy.behavior_cloning_dataset
    receipts: list[dict[str, Any]] = []
    actors: list[str] = []
    with study.inject_distilled_initialization(
        reference,
        checkpoint_loader=lambda *_args, **_kwargs: loaded,
    ):
        for seed in study.POLICY_SEEDS:
            placeholder_observations, placeholder_targets = (
                train_policy.behavior_cloning_dataset()
            )
            state, restored_rms, receipt = train_policy.behavior_clone_policy(
                placeholder_observations,
                placeholder_targets,
                seed=seed,
                n_steps=250,
                batch_size=500,
                epochs=15,
                learning_rate=7.5e-5,
                target_kl=0.02,
                ent_coef=0.003,
                normalize_observation=True,
            )
            actors.append(train_policy.state_digest(study._actor_entries(state)))
            receipts.append(receipt)
            assert train_policy.rms_digest(restored_rms) == rms_sha256
            assert train_policy.state_digest(study._critic_entries(state)) == (
                train_policy.state_digest(
                    study._critic_entries(fresh_by_seed[seed])
                )
            )
            assert placeholder_observations.shape == (1, 73)
            assert placeholder_targets.shape == (1, 22)
            assert not np.any(placeholder_observations)
            assert not np.any(placeholder_targets)
    assert train_policy.behavior_clone_policy is original_behavior_clone
    assert train_policy.behavior_cloning_dataset is original_behavior_dataset
    assert legacy_calls == {"behavior_cloning": 0, "dagger": 0}
    assert actors == [actor_sha256] * 3
    assert len({row["fresh_critic_state_sha256"] for row in receipts}) == 3
    assert [row["fresh_critic_policy_seed"] for row in receipts] == list(
        study.POLICY_SEEDS
    )
    assert all(row["critic_imported_from_bc_checkpoint"] is False for row in receipts)
    assert all(
        row["legacy_hand_rule_demonstrations_recollected"] is False
        and row["legacy_bc_or_dagger_dataset_collected_by_ppo_worker"] is False
        and row["initialization_placeholder_observation_count"] == 1
        for row in receipts
    )


def test_worker_path_never_calls_legacy_bc_or_dagger_collectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 37_017
    distilled = _policy_state(actor_value=8.0, critic_value=-8.0)
    actor_sha256 = train_policy.state_digest(study._actor_entries(distilled))
    obs_mean = np.arange(73, dtype=np.float64)
    obs_var = np.arange(73, dtype=np.float64) + 1.0
    rms_state = {"mean": obs_mean, "var": obs_var, "count": 5040.0001}
    reference = _student_reference(
        actor_sha256,
        train_policy.rms_digest(rms_state),
    )
    loaded = SimpleNamespace(
        model=SimpleNamespace(
            policy=SimpleNamespace(state_dict=lambda: distilled)
        ),
        normalization=SimpleNamespace(
            obs_mean=obs_mean,
            obs_var=obs_var,
            obs_count=5040.0001,
        ),
    )
    fresh = _policy_state(actor_value=-1.0, critic_value=float(seed))
    monkeypatch.setattr(
        train_policy,
        "untrained_policy_state",
        lambda **_kwargs: fresh,
    )
    calls = {"legacy_bc": 0, "dagger": 0, "main": 0}

    def legacy_bc_bomb() -> tuple[np.ndarray, np.ndarray]:
        calls["legacy_bc"] += 1
        raise AssertionError("legacy hand-rule BC collection ran")

    def dagger_bomb(*_args: Any, **_kwargs: Any) -> Any:
        calls["dagger"] += 1
        raise AssertionError("legacy DAgger collection ran")

    monkeypatch.setattr(train_policy, "behavior_cloning_dataset", legacy_bc_bomb)
    monkeypatch.setattr(train_policy, "policy_rollout_dataset", dagger_bomb)

    def fake_main(_argv: list[str]) -> int:
        calls["main"] += 1
        assert torch.get_num_threads() == study.TORCH_INTRAOP_THREADS
        assert torch.get_num_interop_threads() == study.TORCH_INTEROP_THREADS
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        train_policy.torch.set_num_threads(min(12, os.cpu_count() or 1))
        train_policy.torch.set_num_interop_threads(1)
        train_policy.torch.use_deterministic_algorithms(True)
        observations, targets = train_policy.behavior_cloning_dataset()
        assert observations.shape == (1, 73)
        assert targets.shape == (1, 22)
        _, _, initialization = train_policy.behavior_clone_policy(
            observations,
            targets,
            seed=seed,
            n_steps=250,
            batch_size=500,
            epochs=15,
            learning_rate=7.5e-5,
            target_kl=0.02,
            ent_coef=0.003,
            normalize_observation=True,
        )
        assert initialization[
            "legacy_hand_rule_demonstrations_recollected"
        ] is False
        assert initialization[
            "legacy_bc_or_dagger_dataset_collected_by_ppo_worker"
        ] is False
        return 19

    original_inject = study.inject_distilled_initialization
    monkeypatch.setattr(
        study,
        "inject_distilled_initialization",
        lambda value: original_inject(
            value,
            checkpoint_loader=lambda *_args, **_kwargs: loaded,
        ),
    )
    monkeypatch.setattr(train_policy, "main", fake_main)
    monkeypatch.setattr(study, "load_student_reference", lambda _path: reference)
    monkeypatch.setattr(study, "_git_commit", lambda: "synthetic-commit")
    sources = {"runner": "1" * 64}
    monkeypatch.setattr(study, "source_identity", lambda: sources)
    config = study.expected_training_config(seed, reference)
    fresh_critic_sha256 = train_policy.state_digest(
        study._critic_entries(fresh)
    )
    contract = {
        "tool": study.TOOL_ID,
        **study._torch_runtime_binding(),
        "git_commit": "synthetic-commit",
        "source_identity": sources,
        "source_identity_sha256": canonical_hash(sources),
        "registered_policy_seeds": list(study.POLICY_SEEDS),
        "registered_fresh_critic_state_sha256_by_seed": {
            str(seed): fresh_critic_sha256
        },
        "approved_student_reference": reference,
        "registered_training_configs": {str(seed): config},
        "registered_training_config_sha256_by_seed": {
            str(seed): canonical_hash(config)
        },
    }
    protocol = {
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
    }
    monkeypatch.setattr(study, "_load_json", lambda *_args: protocol)

    assert study._run_worker(tmp_path.resolve(), seed) == 19
    assert calls == {"legacy_bc": 0, "dagger": 0, "main": 1}
    assert train_policy.behavior_cloning_dataset is legacy_bc_bomb
    assert train_policy.torch is torch


def test_registered_trainer_config_is_exact_and_hashes_are_per_seed() -> None:
    reference = _student_reference()
    configs = {
        seed: study.expected_training_config(seed, reference)
        for seed in study.POLICY_SEEDS
    }
    hashes = {canonical_hash(config) for config in configs.values()}
    assert len(hashes) == 3
    for seed, config in configs.items():
        assert config["policy_seed"] == seed
        assert config["active_actor_critic_transitions"] == 2_000_000
        assert config["evaluation_milestones"] == [
            200_000,
            500_000,
            1_000_000,
            2_000_000,
        ]
        assert config["learning_rate"] == 7.5e-5
        assert config["target_kl"] == 0.02
        assert config["ent_coef"] == 0.003
        assert config["reward_profile"] == "v3_equivalent"
        assert config["critic_warmup_min_transitions"] == 50_000
        assert config["critic_warmup_max_transitions"] == 100_000
        assert config["freeze_observation_rms"] is True
        distillation = config["distillation_experiment"]
        assert distillation["torch_runtime"] == study._resolved_torch_runtime()
        assert distillation["torch_runtime_sha256"] == canonical_hash(
            distillation["torch_runtime"]
        )
        assert distillation["torch_runtime"][
            "deterministic_algorithms_warn_only"
        ] is False
        assert distillation["source_actor_state_sha256"] == "a" * 64
        assert distillation["fresh_critic_initialized_from_each_policy_seed"] is True
        assert distillation["source_evidence_hash_independent_of_training_config_hash"] is True


def _synthetic_development(
    solved: int,
    active_transitions: int,
    total_transitions: int,
) -> dict[str, Any]:
    rows = [
        {
            **identity,
            "solved": index < solved,
            "reason_codes": [] if index < solved else ["not_solved"],
            "resilience_auc": 0.5,
            "minimum_tail_margin": 0.04,
            "hard_violation_count": 0,
            "max_conservation_residual": 0.0,
        }
        for index, identity in enumerate(study._expected_development_identity())
    ]
    return {
        "active_actor_critic_transitions": active_transitions,
        "total_environment_transitions": total_transitions,
        "case_count": 200,
        "solved_count": solved,
        "solve_rate": solved / 200,
        "mean_resilience_auc": 0.5,
        "mean_minimum_tail_margin": 0.04,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "failure_reason_code_histogram": (
            {} if solved == 200 else {"not_solved": 200 - solved}
        ),
        "rows": rows,
    }


@pytest.mark.parametrize("mutation", ("order", "tape", "aggregate", "physics"))
def test_development_validation_recomputes_canonical_rows(
    mutation: str,
) -> None:
    result = _synthetic_development(173, 500_000, 550_000)
    validated = study._development_result(
        result,
        "synthetic",
        expected_active_transitions=500_000,
        expected_total_transitions=550_000,
    )
    assert validated["solved_count"] == 173
    assert len(validated["rows_sha256"]) == 64

    if mutation == "order":
        result["rows"][0], result["rows"][1] = (
            result["rows"][1],
            result["rows"][0],
        )
    elif mutation == "tape":
        result["rows"][0]["tape_seed"] += 1
    elif mutation == "aggregate":
        result["mean_resilience_auc"] = 0.6
    else:
        result["rows"][0]["hard_violation_count"] = 1
    with pytest.raises(
        study.DistilledPPOStudyError,
        match="row 0 drifted|rows disagree with aggregate",
    ):
        study._development_result(
            result,
            "synthetic",
            expected_active_transitions=500_000,
            expected_total_transitions=550_000,
        )


def _fake_student_source(tmp_path: Path) -> tuple[Path, str, Any]:
    root = tmp_path / "student"
    checkpoint_root = root / "bc-checkpoint"
    checkpoint_root.mkdir(parents=True)
    manifest_path = checkpoint_root / "manifest.json"
    model_path = checkpoint_root / "model.zip"
    normalization_path = checkpoint_root / "normalization.npz"
    model_path.write_bytes(b"model")
    normalization_path.write_bytes(b"normalization")
    actor_sha = "a" * 64
    rms_sha = "b" * 64
    manifest = {
        "kind": "city-recovery-ppo-checkpoint",
        "checkpoint": {
            "id": "oracle-bc-heldout-seed-67017",
            "active_actor_critic_transitions": 0,
            "counters": {"num_timesteps": 0},
            "file": {"sha256": file_sha256(model_path)},
            "policy_state_sha256": "c" * 64,
            "actor_state_sha256": actor_sha,
        },
        "normalization": {
            "norm_obs": True,
            "norm_reward": False,
            "training": False,
            "observation_shape": [73],
            "file": {"sha256": file_sha256(normalization_path)},
            "observation_rms_sha256": rms_sha,
            "return_rms_sha256": "d" * 64,
        },
        "training": {
            "milestone": "oracle-bc-only",
            "seed": 67_017,
            "config": {
                "critic_trained": False,
                "ppo_updates": 0,
                "actor_architecture": [384, 256, 128],
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = file_sha256(manifest_path)
    receipt = {
        "schema_version": 1,
        "tool": "train_oracle_bc_student.py",
        "status": "complete_eligible_for_separately_authorized_3_seed_ppo",
        "completed": True,
        "ppo_started": False,
        "development_split_used": True,
        "final_split_imported_or_used": False,
        "development_evaluation_count": 1,
        "contract_sha256": "e" * 64,
        "catastrophic_gate": {
            "kind": "catastrophic_only",
            "passed": True,
            "decision": "eligible_for_separately_authorized_3_seed_ppo",
        },
        "invariants": {"one": True, "two": True},
        "development": {
            "case_count": 200,
            "solved_count": 157,
            "hard_violation_count": 0,
            "maximum_conservation_residual": 0.0,
        },
        "fit": {
            "status": "complete_matched_bc_only_fits",
            "final_split_used": False,
            "dataset_receipt_sha256": "f" * 64,
            "offline_distillation_disclosure": {
                "dagger": False,
                "interactive_relabelling": False,
                "distribution_shift_resolved": False,
                "distribution_shift_is_a_separate_confound": True,
            },
            "fit": {
                "method": "matched_behavior_cloning_only",
                "dagger_iterations": 0,
                "ppo_updates": 0,
                "normalization_frozen": True,
                "oracle_label_student": {
                    "heldout": {
                        "trained": {"mse": 0.04, "mean_absolute_error": 0.14}
                    }
                },
                "matched_hand_rule_control": {
                    "heldout": {
                        "trained": {"mse": 0.02, "mean_absolute_error": 0.09}
                    }
                },
                "trajectory_split": {
                    "fit": {
                        "observations_sha256": "1" * 64,
                        "oracle_targets_sha256": "2" * 64,
                        "step_row_ids_sha256": "3" * 64,
                    }
                },
            },
            "checkpoint_bundle": {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha,
                "model_path": str(model_path.resolve()),
                "model_sha256": file_sha256(model_path),
                "normalization_path": str(normalization_path.resolve()),
                "normalization_sha256": file_sha256(normalization_path),
                "actor_state_sha256": actor_sha,
                "obs_rms_sha256": rms_sha,
            },
        },
    }
    receipt_path = root / "student-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = SimpleNamespace(
        root=checkpoint_root.resolve(),
        manifest_path=manifest_path.resolve(),
        model_path=model_path.resolve(),
        normalization_path=normalization_path.resolve(),
        manifest=manifest,
    )
    return root.resolve(), file_sha256(receipt_path), verified


def test_student_reference_validates_gate_fit_and_bundle(tmp_path: Path) -> None:
    root, receipt_sha, verified = _fake_student_source(tmp_path)
    reference = study.load_student_reference(
        root,
        expected_receipt_sha256=receipt_sha,
        bundle_verifier=lambda _path: verified,
    )
    assert reference["bc_development_solved_count"] == 157
    assert reference["checkpoint"]["actor_state_sha256"] == "a" * 64
    assert reference["heldout_fit"]["oracle_action_mse"] == 0.04
    assert reference["distribution_shift_disclosure"][
        "distribution_shift_resolved"
    ] is False

    receipt_path = root / "student-receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["final_split_imported_or_used"] = True
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(study.DistilledPPOStudyError, match="contract drifted"):
        study.load_student_reference(
            root,
            expected_receipt_sha256=file_sha256(receipt_path),
            bundle_verifier=lambda _path: verified,
        )


def _fake_training_receipt(
    tmp_path: Path,
    seed: int = 37_017,
) -> tuple[Path, dict[Path, Any], dict[str, Any]]:
    output_root = tmp_path / "study"
    directory = output_root / f"seed-{seed}"
    directory.mkdir(parents=True)
    reference = _student_reference(
        actor_sha256="a" * 64,
        rms_sha256="b" * 64,
    )
    config = study.expected_training_config(seed, reference)
    warmup = 50_000
    curve = {
        "bc_initialization": _synthetic_development(157, 0, 0),
        "post_critic_warmup": _synthetic_development(157, 0, warmup),
        **{
            f"ppo_{milestone}_transitions": _synthetic_development(
                170 + index,
                milestone,
                warmup + milestone,
            )
            for index, milestone in enumerate(
                study.EXPECTED_TRAINER_MILESTONES
            )
        },
    }
    verified_by_root: dict[Path, Any] = {}
    bundle_references: dict[str, Any] = {}
    milestone_states: dict[str, Any] = {}
    for milestone in study.EXPECTED_TRAINER_MILESTONES:
        bundle_root = directory / "checkpoints" / f"ppo-{milestone}"
        bundle_root.mkdir(parents=True)
        manifest_path = bundle_root / "manifest.json"
        model_path = bundle_root / "model.zip"
        normalization_path = bundle_root / "normalization.npz"
        manifest_path.write_text("{}\n", encoding="utf-8")
        model_path.write_bytes(b"model")
        normalization_path.write_bytes(b"normalization")
        checkpoint_id = f"seed-{seed}-ppo-{milestone}"
        model_sha = f"{milestone:064x}"[-64:]
        normalization_sha = f"{milestone + 1:064x}"[-64:]
        policy_sha = f"{milestone + 2:064x}"[-64:]
        actor_sha = f"{milestone + 3:064x}"[-64:]
        optimizer_sha = f"{milestone + 4:064x}"[-64:]
        return_rms_sha = f"{milestone + 5:064x}"[-64:]
        num_timesteps = warmup + milestone
        manifest = {
            "schema_version": 1,
            "kind": "city-recovery-ppo-checkpoint",
            "training": {
                "config": config,
                "config_sha256": canonical_hash(config),
                "seed": seed,
                "milestone": milestone,
            },
            "checkpoint": {
                "id": checkpoint_id,
                "active_actor_critic_transitions": milestone,
                "file": {"sha256": model_sha},
                "policy_state_sha256": policy_sha,
                "actor_state_sha256": actor_sha,
                "optimizer_state_sha256": optimizer_sha,
                "counters": {"num_timesteps": num_timesteps},
            },
            "normalization": {
                "file": {"sha256": normalization_sha},
                "observation_rms_sha256": "b" * 64,
                "return_rms_sha256": return_rms_sha,
            },
        }
        verified_by_root[bundle_root.resolve()] = SimpleNamespace(
            root=bundle_root.resolve(),
            manifest_path=manifest_path.resolve(),
            model_path=model_path.resolve(),
            normalization_path=normalization_path.resolve(),
            manifest=manifest,
        )
        bundle_references[str(milestone)] = {
            "checkpoint_id": checkpoint_id,
            "active_actor_critic_transitions": milestone,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "model_path": str(model_path.resolve()),
            "model_sha256": model_sha,
            "normalization_path": str(normalization_path.resolve()),
            "normalization_sha256": normalization_sha,
            "policy_state_sha256": policy_sha,
            "actor_state_sha256": actor_sha,
            "optimizer_state_sha256": optimizer_sha,
            "obs_rms_sha256": "b" * 64,
            "ret_rms_sha256": return_rms_sha,
            "num_timesteps": num_timesteps,
        }
        milestone_states[str(milestone)] = {
            "policy_sha256": policy_sha,
            "actor_sha256": actor_sha,
            "observation_rms_sha256": "b" * 64,
            "return_rms_sha256": return_rms_sha,
            "return_rms_count": num_timesteps + 0.0001,
        }
    fresh_critic_sha = study.fresh_critic_state_sha256(seed)
    receipt = {
        "schema_version": 1,
        "tool": train_policy.TOOL_ID,
        "status": "complete",
        "training_split": "train",
        "evaluation_split": "dev",
        "development_case_count": 200,
        "final_split_used": False,
        "config": config,
        "checks": {
            "actor_unchanged_during_critic_warmup": True,
            "critic_explained_variance_gate_passed": True,
            "return_rms_continuous_without_reset": True,
            "observation_rms_contract_preserved": True,
            "per_iteration_telemetry_complete": True,
            "development_hard_violations_zero": True,
            "development_conservation_residuals_zero": True,
            "development_only_no_final_split_used": True,
            "training_complete": True,
            "all_registered_checkpoints_persisted": True,
        },
        "transition_counts": {
            "critic_warmup": warmup,
            "active_actor_critic": 2_000_000,
            "total_environment": warmup + 2_000_000,
        },
        "behavior_cloning": {
            "actor_warm_start_applied": True,
            "teacher": "privileged_same_budget_cem_oracle",
            "training_split_only": True,
            "method": "approved_external_single_pass_behavior_cloning",
            "dagger_iterations": 0,
            "interactive_relabelling": False,
            "legacy_hand_rule_demonstrations_recollected": False,
            "legacy_bc_or_dagger_dataset_collected_by_ppo_worker": False,
            "initialization_placeholder_observation_count": 1,
            "distribution_shift_resolved": False,
            "distribution_shift_is_a_separate_confound": True,
            "critic_imported_from_bc_checkpoint": False,
            "fresh_critic_policy_seed": seed,
            "fresh_critic_state_sha256": fresh_critic_sha,
            "actor_state_sha256": "a" * 64,
            "policy_state_sha256": "c" * 64,
            "observation_rms_sha256": "b" * 64,
            "source_student_receipt_sha256": "c" * 64,
            "source_checkpoint_manifest_sha256": "f" * 64,
            "source_checkpoint_model_sha256": "4" * 64,
        },
        "initialization": {
            "actor_sha256": "a" * 64,
            "policy_sha256": "c" * 64,
            "observation_rms_sha256": "b" * 64,
        },
        "normalization": {
            "observation_rms_frozen": True,
            "observation_rms_sha256": "b" * 64,
        },
        "critic_warmup": {
            "actor_sha256_before": "a" * 64,
            "actor_sha256_after": "a" * 64,
            "actor_parameters_byte_identical": True,
            "minimum_transitions": 50_000,
            "maximum_transitions": 100_000,
        },
        "training_roster_and_tapes": (
            train_policy.training_roster_and_tapes_contract()
        ),
        "development_curve": curve,
        "development": curve["ppo_2000000_transitions"],
        "milestone_states": milestone_states,
        "checkpoint_bundles": bundle_references,
    }
    receipt_path = directory / "training-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, verified_by_root, reference


def test_training_receipt_binds_all_curves_and_checkpoint_bundles(
    tmp_path: Path,
) -> None:
    receipt_path, verified_by_root, reference = _fake_training_receipt(tmp_path)

    def verifier(path: Path) -> Any:
        return verified_by_root[path.resolve()]

    _, candidates = study.validate_training_receipt(
        receipt_path,
        37_017,
        reference,
        bundle_verifier=verifier,
    )
    assert [
        row["active_actor_critic_transitions"] for row in candidates
    ] == [500_000, 1_000_000, 2_000_000]
    assert candidates[-1]["development"]["solved_count"] == 173
    assert candidates[-1]["num_timesteps"] == 2_050_000
    assert len(candidates[-1]["policy_state_sha256"]) == 64
    assert len(candidates[-1]["actor_state_sha256"]) == 64
    assert len(candidates[-1]["optimizer_state_sha256"]) == 64
    assert len(candidates[-1]["return_rms_sha256"]) == 64

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt["development_curve"]["ppo_500000_transitions"]["rows"]
    rows[0], rows[1] = rows[1], rows[0]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(study.DistilledPPOStudyError, match="row 0 drifted"):
        study.validate_training_receipt(
            receipt_path,
            37_017,
            reference,
            bundle_verifier=verifier,
        )


def test_training_receipt_rejects_bundle_config_or_milestone_drift(
    tmp_path: Path,
) -> None:
    receipt_path, verified_by_root, reference = _fake_training_receipt(tmp_path)
    first_root = (
        receipt_path.parent / "checkpoints" / "ppo-200000"
    ).resolve()
    verified_by_root[first_root].manifest["training"]["config"][
        "learning_rate"
    ] = 3e-5

    with pytest.raises(study.DistilledPPOStudyError, match="bundle binding drifted"):
        study.validate_training_receipt(
            receipt_path,
            37_017,
            reference,
            bundle_verifier=lambda path: verified_by_root[path.resolve()],
        )


def test_training_receipt_rejects_self_consistent_mutated_torch_runtime(
    tmp_path: Path,
) -> None:
    receipt_path, _, reference = _fake_training_receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    distillation = receipt["config"]["distillation_experiment"]
    distillation["torch_runtime"]["interop_threads"] += 1
    distillation["torch_runtime_sha256"] = canonical_hash(
        distillation["torch_runtime"]
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        study.DistilledPPOStudyError,
        match="training config Torch runtime binding drifted",
    ):
        study.validate_training_receipt(
            receipt_path,
            37_017,
            reference,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "reference_policy",
        "manifest_num_timesteps",
        "milestone_policy",
        "manifest_path",
    ),
)
def test_training_receipt_rejects_state_chain_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt_path, verified_by_root, reference = _fake_training_receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle_root = (
        receipt_path.parent / "checkpoints" / "ppo-500000"
    ).resolve()
    if mutation == "reference_policy":
        receipt["checkpoint_bundles"]["500000"][
            "policy_state_sha256"
        ] = "9" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif mutation == "manifest_num_timesteps":
        verified_by_root[bundle_root].manifest["checkpoint"]["counters"][
            "num_timesteps"
        ] += 5_000
    elif mutation == "milestone_policy":
        receipt["milestone_states"]["500000"]["policy_sha256"] = "8" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        receipt["checkpoint_bundles"]["500000"]["manifest_path"] = str(
            bundle_root / "different-manifest.json"
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        study.DistilledPPOStudyError,
        match="bundle binding drifted",
    ):
        study.validate_training_receipt(
            receipt_path,
            37_017,
            reference,
            bundle_verifier=lambda path: verified_by_root[path.resolve()],
        )


def test_promotion_is_conjunctive_and_uses_endpoint_consistency() -> None:
    selected = {"development": {"solved_count": 183}}
    endpoints = {
        "mean_solved_count": 173.0,
        "seed_count_at_or_above_172": 2,
    }
    assert study.promotion_decision(selected, endpoints)["passed"] is True

    for changed in (
        ({"development": {"solved_count": 182}}, endpoints),
        (selected, {**endpoints, "mean_solved_count": 171.4}),
        (selected, {**endpoints, "seed_count_at_or_above_172": 1}),
    ):
        assert study.promotion_decision(*changed)["passed"] is False


def test_endpoint_summary_reports_mean_population_and_sample_sd() -> None:
    solved = (170, 172, 175)
    candidates = [
        {
            "policy_seed": seed,
            "active_actor_critic_transitions": 2_000_000,
            "development": {
                "solved_count": count,
                "solve_rate": count / 200,
                "mean_resilience_auc": 0.5,
            },
            "training_receipt_sha256": f"{seed:064x}",
            "training_config_sha256": f"{seed + 1:064x}",
        }
        for seed, count in zip(study.POLICY_SEEDS, solved, strict=True)
    ]
    summary = study.endpoint_summary(candidates)
    assert summary["mean_solved_count"] == pytest.approx(sum(solved) / 3)
    assert summary["population_std_solved_count"] == pytest.approx(
        np.std(solved, ddof=0)
    )
    assert summary["sample_std_solved_count"] == pytest.approx(
        np.std(solved, ddof=1)
    )
    assert summary["seed_count_at_or_above_172"] == 2


def test_protocol_is_create_new_and_resume_fails_closed(tmp_path: Path) -> None:
    output_root = tmp_path / "study"
    contract = {
        "tool": study.TOOL_ID,
        **study._torch_runtime_binding(),
        "source_identity_sha256": "a" * 64,
        "registered_training_config_sha256_by_seed": {"37017": "b" * 64},
    }
    protocol_path = study._create_study_protocol(output_root, contract)
    assert protocol_path.is_file()
    study._validate_protocol(output_root, contract)
    with pytest.raises(study.DistilledPPOStudyError, match="new output root"):
        study._create_study_protocol(output_root, contract)

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["contract"]["tool"] = "drifted"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(study.DistilledPPOStudyError, match="differs"):
        study._validate_protocol(output_root, contract)


def test_protocol_rejects_self_consistent_mutated_torch_runtime(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "study"
    contract = {"tool": study.TOOL_ID, **study._torch_runtime_binding()}
    protocol_path = study._create_study_protocol(output_root, contract)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    mutated = protocol["contract"]
    mutated["torch_runtime"]["intraop_threads"] += 1
    mutated["torch_runtime_sha256"] = canonical_hash(
        mutated["torch_runtime"]
    )
    protocol["contract_sha256"] = canonical_hash(mutated)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(
        study.DistilledPPOStudyError,
        match="existing study protocol Torch runtime binding drifted",
    ):
        study._validate_protocol(output_root, mutated)


def test_summary_publication_is_idempotent_but_rejects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    first = {"status": "complete", "created_at_utc": "one", "value": 1}
    assert study._publish_summary_idempotent(path, first) == first
    second = {"status": "complete", "created_at_utc": "two", "value": 1}
    assert study._publish_summary_idempotent(path, second) == first
    with pytest.raises(study.DistilledPPOStudyError, match="differs"):
        study._publish_summary_idempotent(
            path,
            {"status": "complete", "created_at_utc": "three", "value": 2},
        )


def test_source_has_no_final_roster_direct_learning_or_protected_writes() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_distilled_ppo_study.py"
    ).read_text(encoding="utf-8")
    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert ".learn(" not in source
    assert "train_policy.main" in source
    assert "persist_checkpoint_bundle" not in source
    assert "backend/app/city" not in source.replace(
        '"backend/app/city/environment.py"', ""
    ).replace('"backend/app/city/scenarios.py"', "").replace(
        '"backend/app/city/outcome.py"', ""
    ).replace('"backend/app/city/physics.py"', "").replace(
        '"backend/app/city/planners.py"', ""
    ).replace('"backend/app/city/optimizer.py"', "")
