from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backend.app.shared_evidence import canonical_bytes
from scripts.training_artifacts import (
    EXPECTED_BUNDLE_FILES,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    NON_BIT_EXACT_RESUME_DISCLOSURE,
    NORMALIZATION_FILENAME,
    TrainingArtifactError,
    apply_normalization_state,
    actor_state_sha256,
    capture_normalization_state,
    checkpoint_bundle_reference,
    load_checkpoint_bundle,
    optimizer_state_sha256,
    persist_checkpoint_bundle,
    policy_state_sha256,
    verify_checkpoint_bundle,
)


class ArtifactTestEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = spaces.Box(
        low=-10.0,
        high=10.0,
        shape=(73,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(22,),
        dtype=np.float32,
    )

    def __init__(self) -> None:
        super().__init__()
        self.day = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self.day = 0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.day += 1
        reward = -float(np.square(np.asarray(action, dtype=np.float64)).mean())
        return self._observation(), reward, self.day >= 4, False, {}

    def _observation(self) -> np.ndarray:
        return np.linspace(-0.8, 0.9, 73, dtype=np.float32) + self.day * 0.01


@pytest.fixture
def checkpoint_source() -> tuple[PPO, VecNormalize]:
    base = DummyVecEnv([ArtifactTestEnv])
    normalizer = VecNormalize(
        base,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=7.5,
        clip_reward=4.5,
        gamma=0.97,
        epsilon=2e-8,
    )
    observations = np.stack(
        [
            np.linspace(-0.8, 0.9, 73, dtype=np.float64),
            np.linspace(0.4, 1.4, 73, dtype=np.float64),
            np.linspace(-1.1, 0.2, 73, dtype=np.float64),
        ]
    )
    normalizer.obs_rms.update(observations)
    normalizer.ret_rms.update(np.asarray([0.25, -0.75, 1.5], dtype=np.float64))

    model = PPO(
        "MlpPolicy",
        normalizer,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=37017,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
    )
    optimizer = model.policy.optimizer
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().mean() for parameter in model.policy.parameters())
    loss.backward()
    optimizer.step()
    model.num_timesteps = 550_000
    model._n_updates = 7
    model._episode_num = 3
    model._current_progress_remaining = 0.625

    try:
        yield model, normalizer
    finally:
        normalizer.close()


def _publish(
    root: Path,
    source: tuple[PPO, VecNormalize],
) -> Path:
    model, normalizer = source
    destination = root / "checkpoint-500000"
    reference = persist_checkpoint_bundle(
        destination,
        model=model,
        normalizer=normalizer,
        training_config={
            "learning_rate": 7.5e-5,
            "target_kl": 0.02,
            "lanes": 20,
            "freeze_observation_rms": True,
        },
        seed=37017,
        milestone=500_000,
        checkpoint_id="active-500000",
        active_actor_critic_transitions=500_000,
    )
    assert reference["checkpoint_id"] == "active-500000"
    assert reference["active_actor_critic_transitions"] == 500_000
    json.dumps(reference, allow_nan=False)
    return destination


def test_bundle_strong_reload_round_trip_preserves_complete_state(
    tmp_path: Path,
    checkpoint_source: tuple[PPO, VecNormalize],
) -> None:
    model, normalizer = checkpoint_source
    source_normalization = capture_normalization_state(normalizer)
    fixed_raw_observation = np.linspace(-0.4, 0.7, 73, dtype=np.float64)
    fixed_normalized_observation = np.clip(
        (fixed_raw_observation - source_normalization.obs_mean)
        / np.sqrt(source_normalization.obs_var + source_normalization.epsilon),
        -source_normalization.clip_obs,
        source_normalization.clip_obs,
    ).astype(np.float32)
    action_before, _ = model.predict(
        fixed_normalized_observation,
        deterministic=True,
    )
    policy_sha256_before = policy_state_sha256(model)
    optimizer_sha256_before = optimizer_state_sha256(model)
    actor_sha256_before = actor_state_sha256(model)
    assert model.policy.optimizer.state

    destination = _publish(tmp_path, checkpoint_source)
    assert {path.name for path in destination.iterdir()} == EXPECTED_BUNDLE_FILES
    verified = verify_checkpoint_bundle(destination)
    assert verified.manifest_path.read_bytes() == canonical_bytes(verified.manifest)
    assert verified.manifest["checkpoint"]["file"]["path"] == MODEL_FILENAME
    assert verified.manifest["normalization"]["file"]["path"] == (
        NORMALIZATION_FILENAME
    )
    assert verified.manifest["checkpoint"]["policy_state_sha256"] == (
        policy_sha256_before
    )
    assert verified.manifest["checkpoint"]["optimizer_state_sha256"] == (
        optimizer_sha256_before
    )
    assert verified.manifest["checkpoint"]["actor_state_sha256"] == (
        actor_sha256_before
    )
    assert verified.manifest["checkpoint"]["counters"] == {
        "num_timesteps": 550_000,
        "n_updates": 7,
        "episode_num": 3,
        "current_progress_remaining": 0.625,
    }
    assert verified.manifest["training"]["seed"] == 37017
    assert verified.manifest["training"]["milestone"] == 500_000
    assert verified.manifest["checkpoint"]["id"] == "active-500000"
    assert verified.manifest["checkpoint"][
        "active_actor_critic_transitions"
    ] == 500_000
    assert verified.manifest["resume"] == {
        "selection_evaluation_export_supported": True,
        "bit_exact": False,
        "disclosure": NON_BIT_EXACT_RESUME_DISCLOSURE,
    }

    with torch.no_grad():
        next(model.policy.parameters()).add_(1.0)
    loaded = load_checkpoint_bundle(destination)
    action_after, _ = loaded.model.predict(
        fixed_normalized_observation,
        deterministic=True,
    )
    np.testing.assert_array_equal(action_after, action_before)
    assert policy_state_sha256(loaded.model) == policy_sha256_before
    assert optimizer_state_sha256(loaded.model) == optimizer_sha256_before
    assert actor_state_sha256(loaded.model) == actor_sha256_before
    assert loaded.model.num_timesteps == 550_000
    assert loaded.model._n_updates == 7
    assert loaded.model._episode_num == 3
    assert loaded.model._current_progress_remaining == 0.625
    np.testing.assert_array_equal(
        loaded.normalization.obs_mean,
        source_normalization.obs_mean,
    )
    np.testing.assert_array_equal(
        loaded.normalization.obs_var,
        source_normalization.obs_var,
    )
    np.testing.assert_array_equal(
        loaded.normalization.ret_mean,
        source_normalization.ret_mean,
    )
    np.testing.assert_array_equal(
        loaded.normalization.ret_var,
        source_normalization.ret_var,
    )
    assert loaded.normalization.observation_rms_sha256 == (
        source_normalization.observation_rms_sha256
    )
    assert loaded.normalization.return_rms_sha256 == (
        source_normalization.return_rms_sha256
    )

    reference = checkpoint_bundle_reference(loaded.bundle)
    assert reference == {
        "checkpoint_id": "active-500000",
        "manifest_path": str(verified.manifest_path),
        "manifest_sha256": reference["manifest_sha256"],
        "model_path": str(verified.model_path),
        "model_sha256": verified.manifest["checkpoint"]["file"]["sha256"],
        "normalization_path": str(verified.normalization_path),
        "normalization_sha256": verified.manifest["normalization"]["file"][
            "sha256"
        ],
        "policy_state_sha256": policy_sha256_before,
        "actor_state_sha256": actor_sha256_before,
        "optimizer_state_sha256": optimizer_sha256_before,
        "obs_rms_sha256": source_normalization.observation_rms_sha256,
        "ret_rms_sha256": source_normalization.return_rms_sha256,
        "num_timesteps": 550_000,
        "active_actor_critic_transitions": 500_000,
    }

    restored_base = DummyVecEnv([ArtifactTestEnv])
    restored_normalizer = VecNormalize(restored_base)
    try:
        apply_normalization_state(restored_normalizer, loaded.normalization)
        restored = capture_normalization_state(restored_normalizer)
        assert restored.observation_rms_sha256 == (
            source_normalization.observation_rms_sha256
        )
        assert restored.return_rms_sha256 == source_normalization.return_rms_sha256
        assert restored.clip_obs == 7.5
        assert restored.clip_reward == 4.5
        assert restored.epsilon == 2e-8
        assert restored.gamma == 0.97
        assert restored.norm_obs is True
        assert restored.norm_reward is True
        assert restored.training is True
    finally:
        restored_normalizer.close()


def test_bundle_refuses_overwrite_and_existing_partial_destination(
    tmp_path: Path,
    checkpoint_source: tuple[PPO, VecNormalize],
) -> None:
    model, normalizer = checkpoint_source
    destination = _publish(tmp_path, checkpoint_source)
    original_manifest = (destination / MANIFEST_FILENAME).read_bytes()

    with pytest.raises(TrainingArtifactError, match="refusing to overwrite"):
        persist_checkpoint_bundle(
            destination,
            model=model,
            normalizer=normalizer,
            training_config={"run": 2},
            seed=37017,
            milestone=1_000_000,
            checkpoint_id="active-1000000",
            active_actor_critic_transitions=500_000,
        )
    assert (destination / MANIFEST_FILENAME).read_bytes() == original_manifest

    partial = tmp_path / "partial-checkpoint"
    partial.mkdir()
    (partial / MODEL_FILENAME).write_bytes(b"partial")
    with pytest.raises(TrainingArtifactError, match="refusing to overwrite"):
        persist_checkpoint_bundle(
            partial,
            model=model,
            normalizer=normalizer,
            training_config={"run": 3},
            seed=37017,
            milestone="terminal",
            checkpoint_id="terminal",
            active_actor_critic_transitions=500_000,
        )
    with pytest.raises(TrainingArtifactError, match="missing or unexpected files"):
        verify_checkpoint_bundle(partial)


def test_bundle_verifies_and_runs_deterministically_in_fresh_interpreter(
    tmp_path: Path,
    checkpoint_source: tuple[PPO, VecNormalize],
) -> None:
    destination = _publish(tmp_path, checkpoint_source)
    loaded = load_checkpoint_bundle(destination)
    raw_observation = np.linspace(-0.4, 0.7, 73, dtype=np.float64)
    normalized_observation = np.clip(
        (raw_observation - loaded.normalization.obs_mean)
        / np.sqrt(loaded.normalization.obs_var + loaded.normalization.epsilon),
        -loaded.normalization.clip_obs,
        loaded.normalization.clip_obs,
    ).astype(np.float32)
    expected_action, _ = loaded.model.predict(
        normalized_observation,
        deterministic=True,
    )

    program = r"""
import json
import sys
import numpy as np
from scripts.training_artifacts import checkpoint_bundle_reference, load_checkpoint_bundle

loaded = load_checkpoint_bundle(sys.argv[1])
state = loaded.normalization
raw = np.linspace(-0.4, 0.7, 73, dtype=np.float64)
normalized = np.clip(
    (raw - state.obs_mean) / np.sqrt(state.obs_var + state.epsilon),
    -state.clip_obs,
    state.clip_obs,
).astype(np.float32)
action, _ = loaded.model.predict(normalized, deterministic=True)
print(json.dumps({
    "reference": checkpoint_bundle_reference(loaded.bundle),
    "action": np.asarray(action, dtype=np.float64).tolist(),
}, allow_nan=False, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(destination)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["reference"] == checkpoint_bundle_reference(loaded.bundle)
    np.testing.assert_array_equal(
        np.asarray(payload["action"], dtype=np.float64),
        np.asarray(expected_action, dtype=np.float64),
    )


def test_bundle_verification_detects_artifact_and_manifest_tampering(
    tmp_path: Path,
    checkpoint_source: tuple[PPO, VecNormalize],
) -> None:
    destination = _publish(tmp_path, checkpoint_source)
    normalization_path = destination / NORMALIZATION_FILENAME
    normalization_path.write_bytes(normalization_path.read_bytes() + b"tampered")
    with pytest.raises(TrainingArtifactError, match="file size mismatch"):
        verify_checkpoint_bundle(destination)

    second = _publish(tmp_path / "second", checkpoint_source)
    manifest_path = second / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume"]["bit_exact"] = True
    manifest_path.write_bytes(canonical_bytes(manifest))
    with pytest.raises(TrainingArtifactError, match="resume disclosure drifted"):
        verify_checkpoint_bundle(second)
