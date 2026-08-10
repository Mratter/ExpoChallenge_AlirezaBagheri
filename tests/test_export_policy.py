from __future__ import annotations

import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts import export_policy
from scripts.export_policy import (
    ACTION_COUNT,
    ACTION_TOLERANCE,
    EXPECTED_DEVELOPMENT_CASES,
    EXPECTED_HORIZON_DAYS,
    INPUT_NAME,
    OBSERVATION_COUNT,
    ONNX_OPSET,
    OUTPUT_NAME,
    ExportError,
    action_parity,
    build_manifest,
    build_parity_receipt,
    development_cases,
    export_deterministic_actor,
    inspect_onnx_contract,
    load_observation_normalization,
    load_selection_provenance,
    load_sb3_checkpoint,
    load_training_provenance,
    write_new_json,
)


class TinyCityInterfaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal environment used only to construct a compatible SB3 actor."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            0.0,
            1.0,
            shape=(OBSERVATION_COUNT,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(ACTION_COUNT,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(OBSERVATION_COUNT, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(OBSERVATION_COUNT, dtype=np.float32), 0.0, True, False, {}


def _write_normalization(path: Path, *, norm_obs: bool = True) -> None:
    rng = np.random.default_rng(91)
    np.savez(
        path,
        obs_mean=rng.uniform(0.1, 0.7, size=OBSERVATION_COUNT).astype(np.float64),
        obs_var=rng.uniform(0.01, 0.3, size=OBSERVATION_COUNT).astype(np.float64),
        obs_count=np.asarray(48_000.0001, dtype=np.float64),
        ret_mean=np.asarray(0.0, dtype=np.float64),
        ret_var=np.asarray(1.0, dtype=np.float64),
        ret_count=np.asarray(48_000.0001, dtype=np.float64),
        clip_obs=np.asarray(10.0, dtype=np.float64),
        clip_reward=np.asarray(10.0, dtype=np.float64),
        epsilon=np.asarray(1e-8, dtype=np.float64),
        gamma=np.asarray(0.99, dtype=np.float64),
        norm_obs=np.asarray(norm_obs, dtype=np.bool_),
        norm_reward=np.asarray(True, dtype=np.bool_),
        training=np.asarray(False, dtype=np.bool_),
    )


def _write_checkpoint(path: Path) -> None:
    environment = DummyVecEnv([TinyCityInterfaceEnv])
    try:
        model = PPO(
            "MlpPolicy",
            environment,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            seed=17,
            device="cpu",
            policy_kwargs={
                "activation_fn": torch.nn.Tanh,
                "net_arch": {"pi": [16], "vf": [16]},
            },
            verbose=0,
        )
        model.save(path)
    finally:
        environment.close()


def test_normalization_loader_uses_checkpoint_npz_contract(tmp_path: Path) -> None:
    path = tmp_path / "normalization.npz"
    _write_normalization(path)

    normalization = load_observation_normalization(path)
    raw = np.vstack(
        (
            np.zeros(OBSERVATION_COUNT, dtype=np.float32),
            np.ones(OBSERVATION_COUNT, dtype=np.float32),
        )
    )
    expected = np.clip(
        (raw.astype(np.float64) - normalization.mean)
        / np.sqrt(normalization.var + normalization.epsilon),
        -normalization.clip_obs,
        normalization.clip_obs,
    ).astype(np.float32)

    assert normalization.normalize(raw).dtype == np.float32
    np.testing.assert_array_equal(normalization.normalize(raw), expected)
    assert normalization.state_sha256 == export_policy._array_digest(
        normalization.mean,
        normalization.var,
        np.asarray([normalization.count], dtype=np.float64),
    )


def test_normalization_loader_rejects_disabled_observation_transform(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normalization.npz"
    _write_normalization(path, norm_obs=False)

    with pytest.raises(ExportError, match="must enable norm_obs"):
        load_observation_normalization(path)


def test_exported_actor_bakes_normalization_clips_actions_and_matches_sb3(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "selected.zip"
    normalization_path = tmp_path / "normalization.npz"
    onnx_path = tmp_path / "selected.onnx"
    _write_checkpoint(checkpoint_path)
    _write_normalization(normalization_path)

    model = load_sb3_checkpoint(checkpoint_path)
    normalization = load_observation_normalization(normalization_path)
    interface = export_deterministic_actor(model, normalization, onnx_path)
    inspected = inspect_onnx_contract(onnx_path)

    assert interface == inspected
    assert inspected["opset"] == ONNX_OPSET == 17
    assert inspected["providers"] == ["CPUExecutionProvider"]
    assert inspected["input"] == {
        "name": INPUT_NAME,
        "type": "tensor(float)",
        "shape": ["batch", OBSERVATION_COUNT],
        "normalization": "embedded_vecnormalize_observation_rms",
    }
    assert inspected["output"]["name"] == OUTPUT_NAME
    assert inspected["output"]["shape"] == ["batch", ACTION_COUNT]
    assert inspected["output"]["action_clip"] == [-1.0, 1.0]

    session = export_policy._cpu_session(onnx_path.read_bytes())
    raw = np.random.default_rng(52).uniform(
        -0.5,
        1.5,
        size=(11, OBSERVATION_COUNT),
    ).astype(np.float32)
    comparison = action_parity(model, normalization, session, raw)
    assert comparison["sample_count"] == 11
    assert comparison["element_count"] == 11 * ACTION_COUNT
    assert comparison["maximum_absolute_error"] <= ACTION_TOLERANCE
    assert comparison["elements_over_tolerance"] == 0
    assert comparison["passed"] is True


def test_development_roster_is_exact_and_exporter_cannot_import_final() -> None:
    cases = development_cases()
    source = inspect.getsource(export_policy)

    assert len(cases) == EXPECTED_DEVELOPMENT_CASES == 40
    assert len({case.row_id for case in cases}) == 40
    assert all(case.scenario.horizon_days == EXPECTED_HORIZON_DAYS for case in cases)
    assert all(case.row_id.startswith("v3_dev_") for case in cases)
    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert 'parser.add_argument("--split"' not in source


def _passing_parity() -> dict[str, object]:
    return {
        "passed": True,
        "case_count": 40,
        "action_tolerance": 1e-5,
        "action_relative_tolerance": 0.0,
        "maximum_action_absolute_error": 2e-7,
        "resilience_auc_tolerance": 1e-6,
        "maximum_resilience_auc_absolute_error": 0.0,
        "sb3_solved_count": 35,
        "onnx_solved_count": 35,
        "onnx_hard_violation_count": 0,
        "onnx_maximum_conservation_residual": 0.0,
        "deterministic_replay_mismatch_count": 0,
        "rows_sha256": "a" * 64,
    }


def _runtime_versions() -> dict[str, str]:
    return export_policy.runtime_versions()


def _training_provenance() -> dict[str, object]:
    return {
        "path": "output/training.json",
        "sha256": "b" * 64,
        "schema_version": 1,
        "tool": "train_policy.py",
        "policy_seed": 37_017,
        "registered_active_actor_critic_transitions": 2_000_000,
        "config": {
            "policy_seed": 37_017,
            "active_actor_critic_transitions": 2_000_000,
        },
        "training_roster_and_tapes": {
            "case_count": 192,
            "contract_sha256": "c" * 64,
        },
        "runtime_versions": _runtime_versions(),
    }


def _selection_provenance(
    checkpoint_path: Path,
    normalization_path: Path,
    normalization_sha256: str,
) -> dict[str, object]:
    return {
        "path": "output/selection.json",
        "sha256": "d" * 64,
        "schema_version": 1,
        "tool": "select_policy.py",
        "split": "dev",
        "ranking": {
            "primary_metric": "solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_break_order": [
                "earlier_active_actor_critic_transitions",
                "lower_policy_seed",
            ],
        },
        "selected_checkpoint": {
            "id": "seed-37017-transition-2000000",
            "path": checkpoint_path.as_posix(),
            "sha256": file_sha256(checkpoint_path),
            "policy_seed": 37_017,
            "active_actor_critic_transitions": 2_000_000,
            "normalization_path": normalization_path.as_posix(),
            "normalization_file_sha256": file_sha256(normalization_path),
            "observation_rms_sha256": normalization_sha256,
        },
        "winner": {
            "solved_count": 35,
            "solve_rate": 0.875,
            "mean_resilience_auc": 0.49,
            "mean_minimum_tail_margin": 0.01,
        },
        "runner_up": {
            "solved_count": 34,
            "solve_rate": 0.85,
            "mean_resilience_auc": 0.50,
            "mean_minimum_tail_margin": 0.02,
        },
        "margin": {"solved_cases": 1, "percentage_points": 2.5},
        "tie_break": {"used": False, "level": None},
        "candidate_count": 5,
        "policy_seed_set": [37_017, 47_017, 57_017, 67_017, 77_017],
        "checkpoint_bundle": {
            "manifest_path": "output/checkpoint/manifest.json",
            "manifest_sha256": "e" * 64,
            "policy_state_sha256": "f" * 64,
            "actor_state_sha256": "1" * 64,
            "optimizer_state_sha256": "2" * 64,
            "num_timesteps": 2_050_000,
            "return_rms_sha256": "3" * 64,
        },
        "runtime_versions": _runtime_versions(),
    }


def test_receipt_and_manifest_bind_every_artifact_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "selected.zip"
    checkpoint_path.write_bytes(b"checkpoint")
    normalization_path = tmp_path / "normalization.npz"
    _write_normalization(normalization_path)
    onnx_path = tmp_path / "selected.onnx"
    onnx_path.write_bytes(b"onnx")
    receipt_path = tmp_path / "parity.json"
    manifest_path = tmp_path / "selected.manifest.json"
    normalization = load_observation_normalization(normalization_path)
    training = _training_provenance()
    selection = _selection_provenance(
        checkpoint_path,
        normalization_path,
        normalization.state_sha256,
    )

    receipt = build_parity_receipt(
        model_id="city-recovery-test",
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        normalization=normalization,
        onnx_path=onnx_path,
        interface={"opset": 17},
        parity=_passing_parity(),
        training=training,
        selection=selection,
    )
    write_new_json(receipt_path, receipt)
    manifest = build_manifest(receipt_path=receipt_path, receipt=receipt)
    write_new_json(manifest_path, manifest)

    assert manifest["runtime_enforcement"] == "descriptive_only"
    assert manifest["artifact"]["sha256"] == file_sha256(onnx_path)
    assert manifest["source_checkpoint"]["sha256"] == file_sha256(checkpoint_path)
    assert manifest["normalization"]["file_sha256"] == file_sha256(
        normalization_path
    )
    assert manifest["parity_receipt"]["sha256"] == file_sha256(receipt_path)
    assert manifest["development_parity"]["sb3_solved_count"] == 35
    assert manifest["development_parity"]["onnx_solved_count"] == 35
    assert manifest["source_checkpoint"]["id"] == (
        "seed-37017-transition-2000000"
    )
    assert manifest["training"]["config"] == training["config"]
    assert manifest["selection"]["margin"] == {
        "solved_cases": 1,
        "percentage_points": 2.5,
    }
    assert set(manifest["runtime_versions"]) == {
        "training",
        "selection",
        "export_and_parity",
    }

    with pytest.raises(ExportError, match="refusing to overwrite"):
        write_new_json(receipt_path, receipt)


def test_failed_parity_cannot_produce_publication_manifest(tmp_path: Path) -> None:
    receipt_path = tmp_path / "parity.json"
    receipt_path.write_text("{}", encoding="utf-8")
    receipt = {"parity": {"passed": False}}

    with pytest.raises(ExportError, match="failed parity"):
        build_manifest(receipt_path=receipt_path, receipt=receipt)


def test_provenance_validation_binds_training_selection_bundle_and_rms(
    tmp_path: Path,
) -> None:
    bundle_directory = tmp_path / "checkpoint-2000000"
    bundle_directory.mkdir()
    checkpoint_path = bundle_directory / "model.zip"
    checkpoint_path.write_bytes(b"selected checkpoint")
    normalization_path = bundle_directory / "normalization.npz"
    _write_normalization(normalization_path)
    normalization = load_observation_normalization(normalization_path)
    config = {
        "policy_seed": 37_017,
        "active_actor_critic_transitions": 2_000_000,
        "learning_rate": 7.5e-5,
    }
    checkpoint_id = "seed-37017-transition-2000000"
    checkpoint_manifest = {
        "schema_version": 1,
        "kind": "city_recovery_checkpoint_bundle",
        "checkpoint": {
            "id": checkpoint_id,
            "algorithm": "PPO",
            "policy_class": "ActorCriticPolicy",
            "file": {
                "path": "model.zip",
                "sha256": file_sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "policy_state_sha256": "1" * 64,
            "actor_state_sha256": "2" * 64,
            "optimizer_state_sha256": "3" * 64,
            "counters": {
                "num_timesteps": 2_050_000,
                "n_updates": 100,
                "episode_num": 1,
                "current_progress_remaining": 0.0,
            },
            "active_actor_critic_transitions": 2_000_000,
        },
        "normalization": {
            "file": {
                "path": "normalization.npz",
                "sha256": file_sha256(normalization_path),
                "size_bytes": normalization_path.stat().st_size,
            },
            "observation_shape": [OBSERVATION_COUNT],
            "return_shape": [],
            "observation_rms_sha256": normalization.state_sha256,
            "return_rms_sha256": "4" * 64,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "epsilon": 1e-8,
            "gamma": 0.99,
            "norm_obs": True,
            "norm_reward": True,
            "training": True,
        },
        "training": {
            "config": config,
            "config_sha256": canonical_hash(config),
            "seed": 37_017,
            "milestone": 2_000_000,
        },
        "resume": {
            "selection_evaluation_export_supported": True,
            "bit_exact": False,
            "disclosure": "selection-valid; not bit-exact resumable",
        },
        "publication": {
            "method": "atomic_directory_rename",
            "complete": True,
            "overwrite_permitted": False,
        },
    }
    checkpoint_manifest_path = bundle_directory / "manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    bundle_reference = {
        "checkpoint_id": checkpoint_id,
        "manifest_path": checkpoint_manifest_path.as_posix(),
        "manifest_sha256": file_sha256(checkpoint_manifest_path),
        "model_path": checkpoint_path.as_posix(),
        "model_sha256": file_sha256(checkpoint_path),
        "normalization_path": normalization_path.as_posix(),
        "normalization_sha256": file_sha256(normalization_path),
        "policy_state_sha256": "1" * 64,
        "actor_state_sha256": "2" * 64,
        "optimizer_state_sha256": "3" * 64,
        "obs_rms_sha256": normalization.state_sha256,
        "ret_rms_sha256": "4" * 64,
        "num_timesteps": 2_050_000,
        "active_actor_critic_transitions": 2_000_000,
    }
    training_receipt_path = tmp_path / "training.json"
    training_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "train_policy.py",
                "status": "complete",
                "training_split": "train",
                "evaluation_split": "dev",
                "final_split_used": False,
                "config": config,
                "training_roster_and_tapes": {
                    "case_count": 192,
                    "contract_sha256": "5" * 64,
                },
                "checkpoint_bundles": {"2000000": bundle_reference},
                "runtime_versions": _runtime_versions(),
            }
        ),
        encoding="utf-8",
    )
    selection_receipt_path = tmp_path / "selection.json"
    selection_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "select_policy.py",
                "split": "dev",
                "final_split_used": False,
                "ranking": {
                    "primary_metric": "solved_count",
                    "resilience_auc_used_for_selection": False,
                    "tie_break_order": [
                        "earlier_active_actor_critic_transitions",
                        "lower_policy_seed",
                    ],
                },
                "selected_checkpoint": {
                    "id": checkpoint_id,
                    "path": checkpoint_path.as_posix(),
                    "sha256": file_sha256(checkpoint_path),
                    "policy_seed": 37_017,
                    "active_actor_critic_transitions": 2_000_000,
                    "normalization_path": normalization_path.as_posix(),
                    "normalization_file_sha256": file_sha256(normalization_path),
                    "observation_rms_sha256": normalization.state_sha256,
                    "training_receipt_path": training_receipt_path.as_posix(),
                    "training_receipt_sha256": file_sha256(training_receipt_path),
                },
                "winner": {
                    "solved_count": 35,
                    "solve_rate": 0.875,
                    "mean_resilience_auc": 0.49,
                    "mean_minimum_tail_margin": 0.01,
                },
                "runner_up": {
                    "solved_count": 34,
                    "solve_rate": 0.85,
                    "mean_resilience_auc": 0.50,
                    "mean_minimum_tail_margin": 0.02,
                },
                "margin": {"solved_cases": 1, "percentage_points": 2.5},
                "tie_break": {"used": False, "level": None},
                "candidate_count": 5,
                "candidates": [
                    {
                        "id": checkpoint_id,
                        "policy_seed": 37_017,
                        "development": {
                            "solved_count": 35,
                            "solve_rate": 0.875,
                            "mean_resilience_auc": 0.49,
                            "mean_minimum_tail_margin": 0.01,
                        },
                    },
                    {
                        "id": "runner-up",
                        "policy_seed": 47_017,
                        "development": {
                            "solved_count": 34,
                            "solve_rate": 0.85,
                            "mean_resilience_auc": 0.50,
                            "mean_minimum_tail_margin": 0.02,
                        },
                    },
                    {"id": "third", "policy_seed": 57_017},
                    {"id": "fourth", "policy_seed": 67_017},
                    {"id": "fifth", "policy_seed": 77_017},
                ],
                "runtime_versions": _runtime_versions(),
            }
        ),
        encoding="utf-8",
    )

    training = load_training_provenance(training_receipt_path)
    selection = load_selection_provenance(
        selection_receipt_path,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        normalization=normalization,
        training=training,
    )

    assert selection["selected_checkpoint"]["id"] == checkpoint_id
    assert selection["selected_checkpoint"]["observation_rms_sha256"] == (
        normalization.state_sha256
    )
    assert selection["checkpoint_bundle"]["manifest_sha256"] == file_sha256(
        checkpoint_manifest_path
    )
    assert training["config"] == config
