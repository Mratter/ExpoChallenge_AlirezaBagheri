from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import train_policy
from scripts.run_large_architecture_study import (
    ACTIVE_TRANSITIONS,
    BASELINE_ENDPOINT_MEAN,
    DEVELOPMENT_CASE_COUNT,
    EXPECTED_PARAMETER_COUNTS,
    EXPECTED_TRAINER_MILESTONES,
    HIDDEN_LAYERS,
    POLICY_SEEDS,
    REGISTERED_ARMS,
    REGISTERED_SELECTION_MILESTONES,
    ArchitectureStudyError,
    _create_study_protocol,
    _expected_development_identity,
    _inject_large_architecture,
    _publish_summary_idempotent,
    _run_one_arm,
    _validate_paired_learning_rate_receipts,
    _validate_protocol,
    architecture_config,
    arm_endpoint_summary,
    build_large_model,
    load_baseline_reference,
    model_parameter_counts,
    parameter_counts,
    promotion_decision,
    rank_candidates,
    trainer_arguments,
    validate_training_receipt,
    worker_command,
)


class _TinyEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(73,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(22,), dtype=np.float32
    )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(73, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(73, dtype=np.float32), 0.0, False, False, {}


def test_registry_is_exactly_two_large_lr_arms_over_three_seeds(
    tmp_path: Path,
) -> None:
    assert HIDDEN_LAYERS == (768, 512, 256)
    assert POLICY_SEEDS == (37_017, 47_017, 57_017)
    assert [(arm.id, arm.learning_rate) for arm in REGISTERED_ARMS] == [
        ("large_lr_7_5e_5", 7.5e-5),
        ("large_lr_3e_5", 3.0e-5),
    ]
    assert REGISTERED_SELECTION_MILESTONES == (500_000, 1_000_000, 2_000_000)
    assert EXPECTED_TRAINER_MILESTONES == (
        200_000,
        500_000,
        1_000_000,
        2_000_000,
    )
    assert tuple(
        train_policy.learning_milestones(ACTIVE_TRANSITIONS, 20 * 250)
    ) == EXPECTED_TRAINER_MILESTONES
    assert EXPECTED_PARAMETER_COUNTS == {
        "actor": 587_564,
        "critic": 582_145,
        "total_policy": 1_169_709,
    }
    assert parameter_counts() == EXPECTED_PARAMETER_COUNTS
    assert parameter_counts((256, 128, 64)) == {
        "actor": 61_548,
        "critic": 60_161,
        "total_policy": 121_709,
    }
    for arm in REGISTERED_ARMS:
        arguments = trainer_arguments(tmp_path, arm, POLICY_SEEDS[0])
        assert arguments[arguments.index("--transitions") + 1] == "2000000"
        assert arguments[arguments.index("--lanes") + 1] == "20"
        assert arguments[arguments.index("--n-steps") + 1] == "250"
        assert arguments[arguments.index("--batch-size") + 1] == "500"
        assert arguments[arguments.index("--learning-rate") + 1] == format(
            arm.learning_rate, ".12g"
        )
        assert "--bc-warm-start" in arguments
        assert "--freeze-observation-rms" in arguments
        assert "v3_equivalent" in arguments
        assert "final" not in " ".join(arguments).lower()
        command = worker_command(tmp_path, arm, POLICY_SEEDS[0])
        assert command[-3:] == [arm.id, "--_worker-seed", "37017"]


def test_constructed_large_policy_has_exact_parameter_counts() -> None:
    environment = DummyVecEnv([_TinyEnv])
    try:
        model = build_large_model(
            environment,
            seed=37_017,
            n_steps=2,
            batch_size=2,
        )
        assert model_parameter_counts(model) == EXPECTED_PARAMETER_COUNTS
        policy_layers = [
            layer.out_features
            for layer in model.policy.mlp_extractor.policy_net
            if isinstance(layer, __import__("torch").nn.Linear)
        ]
        value_layers = [
            layer.out_features
            for layer in model.policy.mlp_extractor.value_net
            if isinstance(layer, __import__("torch").nn.Linear)
        ]
        assert policy_layers == value_layers == list(HIDDEN_LAYERS)
    finally:
        environment.close()


def test_injection_changes_only_net_arch_and_records_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_ppo(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(train_policy, "InstrumentedPPO", fake_ppo)
    monkeypatch.setattr(
        "scripts.run_large_architecture_study.model_parameter_counts",
        lambda _: EXPECTED_PARAMETER_COUNTS,
    )
    environment = object()
    canonical = train_policy.build_model(
        environment,
        seed=37_017,
        n_steps=250,
        batch_size=500,
        learning_rate=7.5e-5,
        target_kl=0.02,
        ent_coef=0.003,
    )
    assert canonical is not None
    canonical_args, canonical_kwargs = calls.pop()
    with _inject_large_architecture(REGISTERED_ARMS[0]):
        challenger = train_policy.build_model(
            environment,
            seed=37_017,
            n_steps=250,
            batch_size=500,
            learning_rate=7.5e-5,
            target_kl=0.02,
            ent_coef=0.003,
        )
        assert challenger is not None
        challenger_args, challenger_kwargs = calls.pop()
        args = argparse.Namespace(
            transitions=ACTIVE_TRANSITIONS,
            critic_warmup_min_transitions=50_000,
            critic_warmup_max_transitions=100_000,
            critic_ev_threshold=0.5,
            lanes=20,
            n_steps=250,
            batch_size=500,
            learning_rate=7.5e-5,
            target_kl=0.02,
            ent_coef=0.003,
            reward_profile="v3_equivalent",
            bc_warm_start=True,
            vec_normalize=True,
            freeze_observation_rms=True,
            policy_seed=37_017,
            bc_epochs=15,
        )
        config = train_policy.resolved_training_config(
            args,
            rollout_size=5_000,
            preparedness_alignment_coefficient=10.0,
        )
        assert config["architecture_experiment"] == architecture_config(
            REGISTERED_ARMS[0]
        )
    assert canonical_args == challenger_args
    canonical_policy = canonical_kwargs.pop("policy_kwargs")
    challenger_policy = challenger_kwargs.pop("policy_kwargs")
    assert canonical_kwargs == challenger_kwargs
    assert {
        key: value for key, value in canonical_policy.items() if key != "net_arch"
    } == {
        key: value for key, value in challenger_policy.items() if key != "net_arch"
    }
    assert canonical_policy["net_arch"] == {
        "pi": [384, 256, 128],
        "vf": [384, 256, 128],
    }
    assert challenger_policy["net_arch"] == {
        "pi": [768, 512, 256],
        "vf": [768, 512, 256],
    }


def _candidate(
    arm_id: str,
    seed: int,
    milestone: int,
    solved: int,
    *,
    auc: float = 0.5,
) -> dict[str, Any]:
    return {
        "id": f"{arm_id}-{seed}-{milestone}",
        "arm_id": arm_id,
        "policy_seed": seed,
        "active_actor_critic_transitions": milestone,
        "development": {
            "solved_count": solved,
            "solve_rate": solved / DEVELOPMENT_CASE_COUNT,
            "mean_resilience_auc": auc,
        },
        "training_receipt_sha256": "a" * 64,
    }


def _registered_candidates() -> list[dict[str, Any]]:
    return [
        _candidate(arm.id, seed, milestone, 170)
        for arm in REGISTERED_ARMS
        for seed in POLICY_SEEDS
        for milestone in REGISTERED_SELECTION_MILESTONES
    ]


def test_selection_ignores_auc_and_promotion_requires_all_three_conditions() -> None:
    candidates = _registered_candidates()
    selected_arm = REGISTERED_ARMS[1]
    for candidate in candidates:
        if (
            candidate["arm_id"] == selected_arm.id
            and candidate["active_actor_critic_transitions"] == ACTIVE_TRANSITIONS
        ):
            candidate["development"]["solved_count"] = (
                172 if candidate["policy_seed"] != 57_017 else 171
            )
        if (
            candidate["arm_id"] == selected_arm.id
            and candidate["policy_seed"] == 47_017
            and candidate["active_actor_critic_transitions"] == 1_000_000
        ):
            candidate["development"]["solved_count"] = 183
            candidate["development"]["mean_resilience_auc"] = 0.1
        if candidate["id"] == f"{REGISTERED_ARMS[0].id}-37017-2000000":
            candidate["development"]["mean_resilience_auc"] = 1.0
    ranked = rank_candidates(candidates)
    assert ranked[0]["arm_id"] == selected_arm.id
    assert ranked[0]["policy_seed"] == 47_017
    assert ranked[0]["active_actor_critic_transitions"] == 1_000_000
    endpoint = arm_endpoint_summary(selected_arm, ranked)
    assert endpoint["mean_solved_count"] == pytest.approx(171.6666666667)
    assert endpoint["mean_solved_count"] > BASELINE_ENDPOINT_MEAN
    assert endpoint["seed_count_at_or_above_172"] == 2
    assert promotion_decision(ranked[0], endpoint)["passed"] is True

    ranked[0]["development"]["solved_count"] = 182
    failed = promotion_decision(ranked[0], endpoint)
    assert failed["passed"] is False
    assert failed["decision"] == "complete_not_promoted"


def _development(solved: int, milestone: int) -> dict[str, Any]:
    identities = _expected_development_identity()
    return {
        "case_count": 200,
        "active_actor_critic_transitions": milestone,
        "total_environment_transitions": milestone + 50_000,
        "solved_count": solved,
        "solve_rate": solved / 200,
        "mean_resilience_auc": 0.5,
        "mean_minimum_tail_margin": 0.04,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "failure_reason_code_histogram": (
            {"not_solved": 200 - solved} if solved < 200 else {}
        ),
        "rows": [
            {
                **identity,
                "solved": index < solved,
                "reason_codes": [] if index < solved else ["not_solved"],
                "resilience_auc": 0.5,
                "minimum_tail_margin": 0.04,
                "hard_violation_count": 0,
                "max_conservation_residual": 0.0,
            }
            for index, identity in enumerate(identities)
        ],
    }


def _fake_receipt(
    tmp_path: Path, arm_index: int = 0, seed: int = 37_017
) -> tuple[Path, Any]:
    arm = REGISTERED_ARMS[arm_index]
    config = {
        "architecture_experiment": architecture_config(arm),
        "policy_seed": seed,
        "active_actor_critic_transitions": 2_000_000,
        "lanes": 20,
        "n_steps_per_lane": 250,
        "batch_size": 500,
        "bc_epochs": 15,
        "critic_warmup_min_transitions": 50_000,
        "critic_warmup_max_transitions": 100_000,
        "learning_rate": arm.learning_rate,
        "target_kl": 0.02,
        "ent_coef": 0.003,
        "reward_profile": "v3_equivalent",
        "preparedness_alignment_coefficient": 10.0,
        "bc_warm_start": True,
        "vec_normalize": True,
        "freeze_observation_rms": True,
        "evaluation_milestones": list(EXPECTED_TRAINER_MILESTONES),
    }
    checkpoint_bundles: dict[str, Any] = {}
    verified_by_root: dict[Path, Any] = {}
    curve: dict[str, Any] = {}
    for milestone in REGISTERED_SELECTION_MILESTONES:
        bundle_root = tmp_path / f"ppo-{milestone}"
        bundle_root.mkdir()
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        model_path = bundle_root / "model.zip"
        normalization_path = bundle_root / "normalization.npz"
        checkpoint_id = f"seed-{seed}-ppo-{milestone}"
        model_sha = f"{milestone:064x}"[-64:]
        norm_sha = f"{milestone + 1:064x}"[-64:]
        obs_sha = f"{milestone + 2:064x}"[-64:]
        manifest = {
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
            },
            "normalization": {
                "file": {"sha256": norm_sha},
                "observation_rms_sha256": obs_sha,
            },
        }
        verified_by_root[bundle_root.resolve()] = SimpleNamespace(
            manifest=manifest,
            root=bundle_root.resolve(),
            manifest_path=manifest_path.resolve(),
            model_path=model_path.resolve(),
            normalization_path=normalization_path.resolve(),
        )
        checkpoint_bundles[str(milestone)] = {
            "checkpoint_id": checkpoint_id,
            "active_actor_critic_transitions": milestone,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "normalization_path": str(normalization_path),
            "normalization_sha256": norm_sha,
            "obs_rms_sha256": obs_sha,
        }
        curve[f"ppo_{milestone}_transitions"] = _development(170, milestone)
    initialization_actor = f"{seed:064x}"[-64:]
    initialization_policy = f"{seed + 1:064x}"[-64:]
    observation_rms = f"{seed + 2:064x}"[-64:]
    dataset_sha = "d" * 64
    receipt = {
        "status": "complete",
        "training_split": "train",
        "evaluation_split": "dev",
        "development_case_count": 200,
        "final_split_used": False,
        "config": config,
        "initialization": {
            "actor_sha256": initialization_actor,
            "policy_sha256": initialization_policy,
            "observation_rms_sha256": observation_rms,
        },
        "behavior_cloning": {
            "actor_warm_start_applied": True,
            "teacher": "preparedness_teacher_action",
            "training_split_only": True,
            "observation_normalization": True,
            "policy_state_sha256": initialization_policy,
            "observation_rms_sha256": observation_rms,
            "dataset_sha256": dataset_sha,
        },
        "normalization": {
            "observation_rms_frozen": True,
            "observation_rms_sha256": observation_rms,
        },
        "checks": {
            "training_complete": True,
            "development_only_no_final_split_used": True,
            "development_hard_violations_zero": True,
            "development_conservation_residuals_zero": True,
            "all_registered_checkpoints_persisted": True,
        },
        "development_curve": curve,
        "development": curve["ppo_2000000_transitions"],
        "checkpoint_bundles": checkpoint_bundles,
    }
    path = tmp_path / "training-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    def verifier(root: Path) -> Any:
        return verified_by_root[root.resolve()]

    return path, verifier


def test_receipt_requires_dev_only_registered_curves_and_architecture(
    tmp_path: Path,
) -> None:
    path, verifier = _fake_receipt(tmp_path)
    _, candidates = validate_training_receipt(
        path,
        REGISTERED_ARMS[0],
        37_017,
        bundle_verifier=verifier,
    )
    assert [
        row["active_actor_critic_transitions"] for row in candidates
    ] == list(REGISTERED_SELECTION_MILESTONES)

    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["final_split_used"] = True
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="receipt contract drifted"):
        validate_training_receipt(
            path,
            REGISTERED_ARMS[0],
            37_017,
            bundle_verifier=verifier,
        )


@pytest.mark.parametrize(
    "mutation",
    ("reorder", "tape_seed", "aggregate", "histogram"),
)
def test_receipt_recomputes_canonical_ordered_dev_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    path, verifier = _fake_receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    development = receipt["development_curve"]["ppo_500000_transitions"]
    if mutation == "reorder":
        development["rows"][0], development["rows"][1] = (
            development["rows"][1],
            development["rows"][0],
        )
    elif mutation == "tape_seed":
        development["rows"][0]["tape_seed"] += 1
    elif mutation == "aggregate":
        development["mean_resilience_auc"] = 0.6
    else:
        development["failure_reason_code_histogram"] = {}
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        ArchitectureStudyError, match="row 0 drifted|rows disagree with aggregate"
    ):
        validate_training_receipt(
            path,
            REGISTERED_ARMS[0],
            37_017,
            bundle_verifier=verifier,
        )


def test_bundle_binds_full_config_milestone_and_registered_warmup(
    tmp_path: Path,
) -> None:
    path, verifier = _fake_receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["config"]["unregistered_extra"] = "drift"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="bundle binding drifted"):
        validate_training_receipt(
            path,
            REGISTERED_ARMS[0],
            37_017,
            bundle_verifier=verifier,
        )

    other = tmp_path / "active-transition-drift"
    other.mkdir()
    path, verifier = _fake_receipt(other)
    manifest = verifier(other / "ppo-500000").manifest
    manifest["checkpoint"]["active_actor_critic_transitions"] = 499_999
    with pytest.raises(ArchitectureStudyError, match="bundle binding drifted"):
        validate_training_receipt(
            path,
            REGISTERED_ARMS[0],
            37_017,
            bundle_verifier=verifier,
        )

    other = tmp_path / "warmup-drift"
    other.mkdir()
    path, verifier = _fake_receipt(other)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["config"]["critic_warmup_min_transitions"] = 45_000
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="receipt contract drifted"):
        validate_training_receipt(
            path,
            REGISTERED_ARMS[0],
            37_017,
            bundle_verifier=verifier,
        )


def _paired_receipts(tmp_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for arm_index, arm in enumerate(REGISTERED_ARMS):
        for seed in POLICY_SEEDS:
            directory = tmp_path / arm.id / f"seed-{seed}"
            directory.mkdir(parents=True)
            path, _ = _fake_receipt(directory, arm_index=arm_index, seed=seed)
            payloads[(arm.id, seed)] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return payloads


def test_paired_lr_arms_require_identical_bc_initialization_and_rms(
    tmp_path: Path,
) -> None:
    payloads = _paired_receipts(tmp_path)
    checks = _validate_paired_learning_rate_receipts(payloads)
    assert [row["seed"] for row in checks] == list(POLICY_SEEDS)
    assert all(
        row["identical_bc_initialization_and_observation_rms"]
        and row["only_registered_config_difference_is_learning_rate"]
        for row in checks
    )

    drifted = payloads.copy()
    key = (REGISTERED_ARMS[1].id, POLICY_SEEDS[0])
    drifted[key] = json.loads(json.dumps(payloads[key]))
    drifted[key]["initialization"]["actor_sha256"] = "f" * 64
    with pytest.raises(ArchitectureStudyError, match="different BC initialization"):
        _validate_paired_learning_rate_receipts(drifted)

    drifted = {key: json.loads(json.dumps(value)) for key, value in payloads.items()}
    key = (REGISTERED_ARMS[1].id, POLICY_SEEDS[1])
    drifted[key]["config"]["target_kl"] = 0.03
    with pytest.raises(ArchitectureStudyError, match="differ beyond learning rate"):
        _validate_paired_learning_rate_receipts(drifted)


def test_protocol_is_create_new_and_rejects_resume_drift(tmp_path: Path) -> None:
    output_root = tmp_path / "study"
    contract = {"tool": "synthetic", "registered": [1, 2, 3]}
    protocol_path = _create_study_protocol(output_root, contract)
    assert protocol_path.is_file()
    _validate_protocol(output_root, contract)
    with pytest.raises(ArchitectureStudyError, match="new output root"):
        _create_study_protocol(output_root, contract)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["contract"]["registered"].append(4)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="differs"):
        _validate_protocol(output_root, contract)


def test_summary_publication_is_idempotent_and_rejects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    first = {"status": "complete", "created_at_utc": "first", "value": 1}
    assert _publish_summary_idempotent(path, first) == first
    recomputed = {
        "status": "complete",
        "created_at_utc": "second",
        "value": 1,
    }
    assert _publish_summary_idempotent(path, recomputed) == first
    with pytest.raises(ArchitectureStudyError, match="differs"):
        _publish_summary_idempotent(
            path,
            {"status": "complete", "created_at_utc": "third", "value": 2},
        )


def test_resume_skips_complete_and_rejects_partial_or_drifted_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = REGISTERED_ARMS[0]
    complete_root = tmp_path / "complete"
    directory = complete_root / arm.id / "seed-37017"
    directory.mkdir(parents=True)
    (directory / "training-receipt.json").write_text("{}", encoding="utf-8")
    calls: list[Path] = []

    def validate(path: Path, *_: Any, **__: Any) -> tuple[dict[str, Any], list[Any]]:
        calls.append(path)
        return {}, []

    monkeypatch.setattr(
        "scripts.run_large_architecture_study.validate_training_receipt", validate
    )
    monkeypatch.setattr(
        "scripts.run_large_architecture_study.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("complete run must not relaunch"),
    )
    _run_one_arm(complete_root, arm, 37_017)
    assert calls == [directory / "training-receipt.json"]

    partial_root = tmp_path / "partial"
    (partial_root / arm.id / "seed-37017").mkdir(parents=True)
    with pytest.raises(ArchitectureStudyError, match="partial run"):
        _run_one_arm(partial_root, arm, 37_017)

    def reject(*_: Any, **__: Any) -> Any:
        raise ArchitectureStudyError("synthetic receipt drift")

    monkeypatch.setattr(
        "scripts.run_large_architecture_study.validate_training_receipt", reject
    )
    with pytest.raises(ArchitectureStudyError, match="receipt drift"):
        _run_one_arm(complete_root, arm, 37_017)


def _fake_baseline_evidence(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path]]:
    study_root = tmp_path / "baseline-study"
    solved_counts = [172, 171, 171, 174, 169]
    seeds = [37_017, 47_017, 57_017, 67_017, 77_017]
    endpoints: list[dict[str, Any]] = []
    receipt_paths: list[Path] = []
    for seed, solved in zip(seeds, solved_counts, strict=True):
        path = (
            study_root
            / "adopted_v3_equivalent_2m"
            / f"seed-{seed}"
            / "training-receipt.json"
        )
        path.parent.mkdir(parents=True)
        observation_rms = f"{seed + 2:064x}"[-64:]
        policy_sha = f"{seed + 1:064x}"[-64:]
        config = {
            "policy_seed": seed,
            "active_actor_critic_transitions": 2_000_000,
            "lanes": 20,
            "n_steps_per_lane": 250,
            "batch_size": 500,
            "bc_epochs": 15,
            "bc_warm_start": True,
            "vec_normalize": True,
            "freeze_observation_rms": True,
            "critic_warmup_min_transitions": 50_000,
            "critic_warmup_max_transitions": 100_000,
            "learning_rate": 7.5e-5,
            "target_kl": 0.02,
            "ent_coef": 0.003,
            "reward_profile": "v3_equivalent",
            "preparedness_alignment_coefficient": 10.0,
            "evaluation_milestones": list(EXPECTED_TRAINER_MILESTONES),
        }
        curve = {
            f"ppo_{milestone}_transitions": _development(
                solved if milestone == ACTIVE_TRANSITIONS else solved - 2,
                milestone,
            )
            for milestone in REGISTERED_SELECTION_MILESTONES
        }
        receipt = {
            "status": "complete",
            "final_split_used": False,
            "config": config,
            "initialization": {
                "actor_sha256": f"{seed:064x}"[-64:],
                "policy_sha256": policy_sha,
                "observation_rms_sha256": observation_rms,
            },
            "behavior_cloning": {
                "actor_warm_start_applied": True,
                "teacher": "preparedness_teacher_action",
                "training_split_only": True,
                "observation_normalization": True,
                "policy_state_sha256": policy_sha,
                "observation_rms_sha256": observation_rms,
                "dataset_sha256": "d" * 64,
            },
            "normalization": {
                "observation_rms_frozen": True,
                "observation_rms_sha256": observation_rms,
            },
            "development_curve": curve,
        }
        path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_paths.append(path)
        endpoints.append(
            {
                "seed": seed,
                "solved_count": solved,
                "receipt_sha256": file_sha256(path),
            }
        )
    summary_path = tmp_path / "baseline-summary.json"
    summary = {
        "kind": "city-recovery-training-study-200-summary",
        "scope": {
            "split": "dev",
            "development_case_count": 200,
            "final_split_used": False,
        },
        "source_evidence": {"external_study_root": str(study_root)},
        "baseline": {
            "name": "adopted_v3_equivalent_2m",
            "endpoints": endpoints,
            "aggregate": {
                "mean_solved_count": 171.4,
                "sample_std_solved_count": 1.816590212458495,
            },
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "split": "dev",
                "development_case_count": 200,
                "final_split_used": False,
                "candidate_count": 20,
                "winner": {"solved_count": 178},
                "selected_checkpoint": {
                    "policy_seed": 67_017,
                    "active_actor_critic_transitions": 1_000_000,
                },
                "ranking": {"resilience_auc_used_for_selection": False},
            }
        ),
        encoding="utf-8",
    )
    return summary_path, selection_path, study_root, receipt_paths


def test_baseline_reference_binds_receipt_hash_and_adopted_config(
    tmp_path: Path,
) -> None:
    summary, selection, study_root, receipts = _fake_baseline_evidence(
        tmp_path / "valid"
    )
    reference = load_baseline_reference(summary, selection)
    assert reference["external_study_root"] == str(study_root.resolve())
    assert reference["five_seed_2m_endpoints"] == {
        "solved_counts": [172, 171, 171, 174, 169],
        "mean": 171.4,
        "population_std": 1.624807680927192,
        "sample_std": 1.816590212458495,
    }
    assert all(len(row["config_sha256"]) == 64 for row in reference["curves"])

    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    receipt["config"]["learning_rate"] = 3e-5
    receipts[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="receipt hash drifted"):
        load_baseline_reference(summary, selection)

    summary, selection, _, receipts = _fake_baseline_evidence(
        tmp_path / "config-drift"
    )
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    receipt["config"]["critic_warmup_max_transitions"] = 90_000
    receipts[0].write_text(json.dumps(receipt), encoding="utf-8")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["baseline"]["endpoints"][0]["receipt_sha256"] = file_sha256(
        receipts[0]
    )
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchitectureStudyError, match="training config drifted"):
        load_baseline_reference(summary, selection)


def test_source_has_no_final_roster_or_direct_training_loop() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_large_architecture_study.py"
    ).read_text(encoding="utf-8")
    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert ".learn(" not in source
    assert "train_policy.main" in source
    assert '"resilience_auc_used_for_selection": False' in source
    assert '"threshold": PROMOTION_SELECTED_SOLVES' in source
