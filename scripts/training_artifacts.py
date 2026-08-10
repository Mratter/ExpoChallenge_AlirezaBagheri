"""Persist and verify complete, selection-ready PPO checkpoint bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from backend.app.shared_evidence import (
    canonical_bytes,
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "city-recovery-ppo-checkpoint"
MODEL_FILENAME = "model.zip"
NORMALIZATION_FILENAME = "normalization.npz"
MANIFEST_FILENAME = "manifest.json"
OBSERVATION_COUNT = 73

NORMALIZATION_KEYS = (
    "obs_mean",
    "obs_var",
    "obs_count",
    "ret_mean",
    "ret_var",
    "ret_count",
    "clip_obs",
    "clip_reward",
    "epsilon",
    "gamma",
    "norm_obs",
    "norm_reward",
    "training",
)
EXPECTED_BUNDLE_FILES = frozenset(
    {MODEL_FILENAME, NORMALIZATION_FILENAME, MANIFEST_FILENAME}
)
CHECKPOINT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
WINDOWS_DIRECTORY_RENAME_ATTEMPTS = 8
WINDOWS_DIRECTORY_RENAME_BASE_DELAY_SECONDS = 0.025
WINDOWS_TRANSIENT_DIRECTORY_RENAME_ERRORS = frozenset({5, 32, 33})
NON_BIT_EXACT_RESUME_DISCLOSURE = (
    "The bundle restores policy and optimizer state, training counters, and "
    "normalization statistics, but it is not bit-exact resumable because "
    "vector-environment lane state, lane RNG state, the current rollout buffer, "
    "and per-lane in-progress discounted returns are not captured."
)

PPOType = TypeVar("PPOType", bound=PPO)


class TrainingArtifactError(RuntimeError):
    """Raised when a checkpoint bundle cannot be published or verified."""


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingArtifactError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise TrainingArtifactError(f"{label} must be a finite number")
    return result


def _readonly_float64(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise TrainingArtifactError(f"{label} must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise TrainingArtifactError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    """Hash numerical state using the trainer's dtype-and-shape convention."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _update_structure_digest(digest: Any, value: Any) -> None:
    """Add one JSON-or-tensor value to a deterministic state digest."""

    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
        return
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        payload = str(int(value)).encode("ascii")
        digest.update(b"I" + len(payload).to_bytes(8, "big") + payload)
        return
    if isinstance(value, (float, np.floating)):
        number = _finite_float(value, "state value")
        digest.update(b"F" + np.asarray([number], dtype="<f8").tobytes())
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"S" + len(payload).to_bytes(8, "big") + payload)
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _update_structure_digest(digest, str(tensor.dtype))
        _update_structure_digest(digest, list(tensor.shape))
        try:
            payload = tensor.numpy().tobytes()
        except TypeError:
            payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big") + payload)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TrainingArtifactError("state arrays cannot use object dtype")
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _update_structure_digest(digest, array.dtype.str)
        _update_structure_digest(digest, list(array.shape))
        payload = array.tobytes()
        digest.update(len(payload).to_bytes(8, "big") + payload)
        return
    if isinstance(value, Mapping):
        digest.update(b"D" + len(value).to_bytes(8, "big"))
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__name__, repr(item[0])),
        )
        for key, item in items:
            _update_structure_digest(digest, key)
            _update_structure_digest(digest, item)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L" + len(value).to_bytes(8, "big"))
        for item in value:
            _update_structure_digest(digest, item)
        return
    raise TrainingArtifactError(
        f"state digest does not support {type(value).__name__}"
    )


def _structure_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_structure_digest(digest, value)
    return digest.hexdigest()


def policy_state_sha256(model: PPO) -> str:
    """Hash every named policy tensor deterministically."""

    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_state_sha256(model: PPO) -> str:
    """Hash the policy branch, action head, and action distribution state."""

    state = {
        f"policy_net.{name}": parameter.detach()
        for name, parameter in model.policy.mlp_extractor.policy_net.named_parameters()
    }
    state.update(
        {
            f"action_net.{name}": parameter.detach()
            for name, parameter in model.policy.action_net.named_parameters()
        }
    )
    state["log_std"] = model.policy.log_std.detach()
    digest = hashlib.sha256()
    for name, tensor_value in sorted(state.items()):
        tensor = tensor_value.cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(model: PPO) -> str:
    """Hash the optimizer state and parameter groups deterministically."""

    return _structure_digest(model.policy.optimizer.state_dict())


@dataclass(frozen=True, slots=True)
class NormalizationState:
    """Serializable VecNormalize moments and scalar settings."""

    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_count: float
    ret_mean: np.ndarray
    ret_var: np.ndarray
    ret_count: float
    clip_obs: float
    clip_reward: float
    epsilon: float
    gamma: float
    norm_obs: bool
    norm_reward: bool
    training: bool

    def __post_init__(self) -> None:
        obs_mean = _readonly_float64(self.obs_mean, "observation RMS mean")
        obs_var = _readonly_float64(self.obs_var, "observation RMS variance")
        ret_mean = _readonly_float64(self.ret_mean, "return RMS mean")
        ret_var = _readonly_float64(self.ret_var, "return RMS variance")
        if obs_mean.shape != (OBSERVATION_COUNT,) or obs_var.shape != obs_mean.shape:
            raise TrainingArtifactError(
                f"observation RMS arrays must have shape ({OBSERVATION_COUNT},)"
            )
        if ret_mean.shape != ret_var.shape:
            raise TrainingArtifactError("return RMS mean and variance shapes differ")
        if np.any(obs_var < 0.0) or np.any(ret_var < 0.0):
            raise TrainingArtifactError("RMS variances cannot be negative")

        obs_count = _finite_float(self.obs_count, "observation RMS count")
        ret_count = _finite_float(self.ret_count, "return RMS count")
        clip_obs = _finite_float(self.clip_obs, "observation clip")
        clip_reward = _finite_float(self.clip_reward, "reward clip")
        epsilon = _finite_float(self.epsilon, "normalization epsilon")
        gamma = _finite_float(self.gamma, "normalization gamma")
        if obs_count <= 0.0 or ret_count <= 0.0:
            raise TrainingArtifactError("RMS counts must be positive")
        if clip_obs <= 0.0 or clip_reward <= 0.0 or epsilon <= 0.0:
            raise TrainingArtifactError("normalization clip values and epsilon must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise TrainingArtifactError("normalization gamma must be in [0, 1]")
        for label, value in (
            ("norm_obs", self.norm_obs),
            ("norm_reward", self.norm_reward),
            ("training", self.training),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TrainingArtifactError(f"{label} must be boolean")

        object.__setattr__(self, "obs_mean", obs_mean)
        object.__setattr__(self, "obs_var", obs_var)
        object.__setattr__(self, "obs_count", obs_count)
        object.__setattr__(self, "ret_mean", ret_mean)
        object.__setattr__(self, "ret_var", ret_var)
        object.__setattr__(self, "ret_count", ret_count)
        object.__setattr__(self, "clip_obs", clip_obs)
        object.__setattr__(self, "clip_reward", clip_reward)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "norm_obs", bool(self.norm_obs))
        object.__setattr__(self, "norm_reward", bool(self.norm_reward))
        object.__setattr__(self, "training", bool(self.training))

    @property
    def observation_rms_sha256(self) -> str:
        return _array_digest(
            self.obs_mean,
            self.obs_var,
            np.asarray([self.obs_count], dtype=np.float64),
        )

    @property
    def return_rms_sha256(self) -> str:
        return _array_digest(
            self.ret_mean,
            self.ret_var,
            np.asarray([self.ret_count], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class VerifiedCheckpointBundle:
    """Paths and validated metadata for one complete bundle."""

    root: Path
    model_path: Path
    normalization_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    normalization: NormalizationState


@dataclass(frozen=True, slots=True)
class LoadedCheckpointBundle:
    """A verified bundle with its deserialized SB3 model."""

    bundle: VerifiedCheckpointBundle
    model: PPO

    @property
    def normalization(self) -> NormalizationState:
        return self.bundle.normalization

    @property
    def manifest(self) -> dict[str, Any]:
        return self.bundle.manifest


def capture_normalization_state(normalizer: VecNormalize) -> NormalizationState:
    """Copy all deployable and continuation-relevant VecNormalize state."""

    if not isinstance(normalizer, VecNormalize):
        raise TrainingArtifactError("normalizer must be VecNormalize")
    if isinstance(normalizer.obs_rms, dict):
        raise TrainingArtifactError("dictionary observation normalization is unsupported")
    return NormalizationState(
        obs_mean=normalizer.obs_rms.mean,
        obs_var=normalizer.obs_rms.var,
        obs_count=normalizer.obs_rms.count,
        ret_mean=normalizer.ret_rms.mean,
        ret_var=normalizer.ret_rms.var,
        ret_count=normalizer.ret_rms.count,
        clip_obs=normalizer.clip_obs,
        clip_reward=normalizer.clip_reward,
        epsilon=normalizer.epsilon,
        gamma=normalizer.gamma,
        norm_obs=normalizer.norm_obs,
        norm_reward=normalizer.norm_reward,
        training=normalizer.training,
    )


def apply_normalization_state(
    normalizer: VecNormalize,
    state: NormalizationState,
) -> None:
    """Restore a verified normalization snapshot onto a compatible wrapper."""

    if not isinstance(normalizer, VecNormalize):
        raise TrainingArtifactError("normalizer must be VecNormalize")
    if isinstance(normalizer.obs_rms, dict):
        raise TrainingArtifactError("dictionary observation normalization is unsupported")
    if normalizer.obs_rms.mean.shape != state.obs_mean.shape:
        raise TrainingArtifactError("normalizer observation shape does not match bundle")
    if normalizer.ret_rms.mean.shape != state.ret_mean.shape:
        raise TrainingArtifactError("normalizer return shape does not match bundle")

    normalizer.obs_rms.mean = state.obs_mean.copy()
    normalizer.obs_rms.var = state.obs_var.copy()
    normalizer.obs_rms.count = state.obs_count
    normalizer.ret_rms.mean = state.ret_mean.copy()
    normalizer.ret_rms.var = state.ret_var.copy()
    normalizer.ret_rms.count = state.ret_count
    normalizer.clip_obs = state.clip_obs
    normalizer.clip_reward = state.clip_reward
    normalizer.epsilon = state.epsilon
    normalizer.gamma = state.gamma
    normalizer.norm_obs = state.norm_obs
    normalizer.norm_reward = state.norm_reward
    normalizer.training = state.training


def _write_normalization(path: Path, state: NormalizationState) -> None:
    try:
        with path.open("xb") as handle:
            np.savez(
                handle,
                obs_mean=state.obs_mean,
                obs_var=state.obs_var,
                obs_count=np.asarray(state.obs_count, dtype=np.float64),
                ret_mean=state.ret_mean,
                ret_var=state.ret_var,
                ret_count=np.asarray(state.ret_count, dtype=np.float64),
                clip_obs=np.asarray(state.clip_obs, dtype=np.float64),
                clip_reward=np.asarray(state.clip_reward, dtype=np.float64),
                epsilon=np.asarray(state.epsilon, dtype=np.float64),
                gamma=np.asarray(state.gamma, dtype=np.float64),
                norm_obs=np.asarray(state.norm_obs, dtype=np.bool_),
                norm_reward=np.asarray(state.norm_reward, dtype=np.bool_),
                training=np.asarray(state.training, dtype=np.bool_),
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TrainingArtifactError("normalization state could not be written") from exc


def _scalar(array: np.ndarray, label: str) -> Any:
    if array.shape != ():
        raise TrainingArtifactError(f"{label} must be a scalar")
    return array.item()


def load_normalization_state(path: str | Path) -> NormalizationState:
    """Load and validate one canonical normalization NPZ."""

    candidate = Path(path)
    try:
        with np.load(candidate, allow_pickle=False) as archive:
            if set(archive.files) != set(NORMALIZATION_KEYS):
                raise TrainingArtifactError(
                    "normalization NPZ has missing or unexpected fields"
                )
            arrays = {name: np.asarray(archive[name]) for name in NORMALIZATION_KEYS}
    except TrainingArtifactError:
        raise
    except (OSError, ValueError) as exc:
        raise TrainingArtifactError(
            f"normalization NPZ is missing or invalid: {candidate}"
        ) from exc

    float_fields = NORMALIZATION_KEYS[:10]
    boolean_fields = NORMALIZATION_KEYS[10:]
    if any(arrays[name].dtype != np.dtype(np.float64) for name in float_fields):
        raise TrainingArtifactError("normalization numeric fields must use float64")
    if any(arrays[name].dtype != np.dtype(np.bool_) for name in boolean_fields):
        raise TrainingArtifactError("normalization flags must use boolean dtype")

    return NormalizationState(
        obs_mean=arrays["obs_mean"],
        obs_var=arrays["obs_var"],
        obs_count=_scalar(arrays["obs_count"], "obs_count"),
        ret_mean=arrays["ret_mean"],
        ret_var=arrays["ret_var"],
        ret_count=_scalar(arrays["ret_count"], "ret_count"),
        clip_obs=_scalar(arrays["clip_obs"], "clip_obs"),
        clip_reward=_scalar(arrays["clip_reward"], "clip_reward"),
        epsilon=_scalar(arrays["epsilon"], "epsilon"),
        gamma=_scalar(arrays["gamma"], "gamma"),
        norm_obs=_scalar(arrays["norm_obs"], "norm_obs"),
        norm_reward=_scalar(arrays["norm_reward"], "norm_reward"),
        training=_scalar(arrays["training"], "training"),
    )


def _checkpoint_counters(model: PPO) -> dict[str, Any]:
    progress = _finite_float(
        getattr(model, "_current_progress_remaining", 1.0),
        "current progress remaining",
    )
    return {
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(getattr(model, "_n_updates", 0)),
        "episode_num": int(getattr(model, "_episode_num", 0)),
        "current_progress_remaining": progress,
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": file_sha256(
            path,
            label=path.name,
            error_type=TrainingArtifactError,
        ),
        "size_bytes": path.stat().st_size,
    }


def _normalization_record(
    state: NormalizationState,
    file_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "file": file_record,
        "observation_shape": list(state.obs_mean.shape),
        "return_shape": list(state.ret_mean.shape),
        "observation_rms_sha256": state.observation_rms_sha256,
        "return_rms_sha256": state.return_rms_sha256,
        "clip_obs": state.clip_obs,
        "clip_reward": state.clip_reward,
        "epsilon": state.epsilon,
        "gamma": state.gamma,
        "norm_obs": state.norm_obs,
        "norm_reward": state.norm_reward,
        "training": state.training,
    }


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        payload = canonical_bytes(dict(value))
        result = json.loads(payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingArtifactError(f"{label} must be canonical JSON data") from exc
    if not isinstance(result, dict):
        raise TrainingArtifactError(f"{label} must be an object")
    return result


def _validated_milestone(value: str | int) -> str | int:
    if isinstance(value, bool):
        raise TrainingArtifactError("milestone must be a non-negative integer or name")
    if isinstance(value, int):
        if value < 0:
            raise TrainingArtifactError(
                "milestone must be a non-negative integer or name"
            )
        return value
    if isinstance(value, str) and value.strip() == value and value:
        return value
    raise TrainingArtifactError("milestone must be a non-negative integer or name")


def _validated_checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or CHECKPOINT_ID_PATTERN.fullmatch(value) is None:
        raise TrainingArtifactError(
            "checkpoint_id must contain only letters, numbers, '.', '_', or '-'"
        )
    return value


def _validated_active_transitions(value: int, num_timesteps: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > num_timesteps
    ):
        raise TrainingArtifactError(
            "active_actor_critic_transitions must be between zero and num_timesteps"
        )
    return value


def _manifest(
    *,
    model: PPO,
    checkpoint_id: str,
    active_actor_critic_transitions: int,
    model_record: dict[str, Any],
    normalization: NormalizationState,
    normalization_record: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    milestone: str | int,
) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "checkpoint": {
            "id": checkpoint_id,
            "algorithm": "PPO",
            "policy_class": (
                f"{type(model.policy).__module__}.{type(model.policy).__qualname__}"
            ),
            "file": model_record,
            "policy_state_sha256": policy_state_sha256(model),
            "actor_state_sha256": actor_state_sha256(model),
            "optimizer_state_sha256": optimizer_state_sha256(model),
            "counters": _checkpoint_counters(model),
            "active_actor_critic_transitions": active_actor_critic_transitions,
        },
        "normalization": normalization_record,
        "training": {
            "config": training_config,
            "config_sha256": canonical_hash(training_config),
            "seed": seed,
            "milestone": milestone,
        },
        "resume": {
            "selection_evaluation_export_supported": True,
            "bit_exact": False,
            "disclosure": NON_BIT_EXACT_RESUME_DISCLOSURE,
        },
        "publication": {
            "method": "same-filesystem-staging-directory-rename",
            "complete": True,
            "overwrite_permitted": False,
        },
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(canonical_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TrainingArtifactError("checkpoint manifest could not be written") from exc


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingArtifactError(f"{label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _verify_file_record(
    root: Path,
    value: Any,
    expected_name: str,
) -> dict[str, Any]:
    record = _require_object(value, f"{expected_name} file record")
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise TrainingArtifactError(f"{expected_name} file record fields drifted")
    if record.get("path") != expected_name:
        raise TrainingArtifactError(f"{expected_name} file path drifted")
    size = record.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise TrainingArtifactError(f"{expected_name} file size is invalid")
    expected_sha256 = _require_sha256(
        record.get("sha256"), f"{expected_name} file hash"
    )
    path = root / expected_name
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise TrainingArtifactError(f"{expected_name} is unreadable") from exc
    if actual_size != size:
        raise TrainingArtifactError(f"{expected_name} file size mismatch")
    actual_sha256 = file_sha256(
        path,
        label=expected_name,
        error_type=TrainingArtifactError,
    )
    if actual_sha256 != expected_sha256:
        raise TrainingArtifactError(f"{expected_name} file hash mismatch")
    return record


def _verify_checkpoint_record(value: Any, root: Path) -> None:
    checkpoint = _require_object(value, "checkpoint record")
    expected_fields = {
        "id",
        "algorithm",
        "policy_class",
        "file",
        "policy_state_sha256",
        "actor_state_sha256",
        "optimizer_state_sha256",
        "counters",
        "active_actor_critic_transitions",
    }
    if set(checkpoint) != expected_fields or checkpoint.get("algorithm") != "PPO":
        raise TrainingArtifactError("checkpoint record fields drifted")
    if not isinstance(checkpoint.get("policy_class"), str):
        raise TrainingArtifactError("checkpoint policy class is invalid")
    _validated_checkpoint_id(checkpoint.get("id"))
    _verify_file_record(root, checkpoint.get("file"), MODEL_FILENAME)
    _require_sha256(checkpoint.get("policy_state_sha256"), "policy state hash")
    _require_sha256(checkpoint.get("actor_state_sha256"), "actor state hash")
    _require_sha256(
        checkpoint.get("optimizer_state_sha256"), "optimizer state hash"
    )
    counters = _require_object(checkpoint.get("counters"), "checkpoint counters")
    if set(counters) != {
        "num_timesteps",
        "n_updates",
        "episode_num",
        "current_progress_remaining",
    }:
        raise TrainingArtifactError("checkpoint counter fields drifted")
    for name in ("num_timesteps", "n_updates", "episode_num"):
        count = counters.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TrainingArtifactError(f"checkpoint {name} is invalid")
    _validated_active_transitions(
        checkpoint.get("active_actor_critic_transitions"),
        counters["num_timesteps"],
    )
    _finite_float(
        counters.get("current_progress_remaining"),
        "checkpoint current_progress_remaining",
    )


def _verify_training_record(value: Any) -> None:
    training = _require_object(value, "training record")
    if set(training) != {"config", "config_sha256", "seed", "milestone"}:
        raise TrainingArtifactError("training record fields drifted")
    config = _require_object(training.get("config"), "training config")
    expected_config_sha256 = _require_sha256(
        training.get("config_sha256"), "training config hash"
    )
    if canonical_hash(config) != expected_config_sha256:
        raise TrainingArtifactError("training config hash mismatch")
    seed = training.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TrainingArtifactError("training seed is invalid")
    _validated_milestone(training.get("milestone"))


def verify_checkpoint_bundle(path: str | Path) -> VerifiedCheckpointBundle:
    """Fail closed unless a bundle is complete and internally consistent."""

    root = Path(path).expanduser()
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingArtifactError(f"checkpoint bundle is missing: {root}") from exc
    if root.is_symlink() or not resolved.is_dir():
        raise TrainingArtifactError("checkpoint bundle must be a real directory")
    try:
        entries = list(resolved.iterdir())
    except OSError as exc:
        raise TrainingArtifactError("checkpoint bundle is unreadable") from exc
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise TrainingArtifactError("checkpoint bundle contains an invalid entry")
    names = {entry.name for entry in entries}
    if names != EXPECTED_BUNDLE_FILES:
        raise TrainingArtifactError(
            "checkpoint bundle has missing or unexpected files"
        )

    manifest_path = resolved / MANIFEST_FILENAME
    manifest = load_json_object(
        manifest_path,
        "checkpoint manifest",
        error_type=TrainingArtifactError,
    )
    try:
        manifest_payload = manifest_path.read_bytes()
    except OSError as exc:
        raise TrainingArtifactError("checkpoint manifest is unreadable") from exc
    if canonical_bytes(manifest) != manifest_payload:
        raise TrainingArtifactError("checkpoint manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "kind",
        "checkpoint",
        "normalization",
        "training",
        "resume",
        "publication",
    }:
        raise TrainingArtifactError("checkpoint manifest fields drifted")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("kind") != BUNDLE_KIND
    ):
        raise TrainingArtifactError("checkpoint manifest identity drifted")

    _verify_checkpoint_record(manifest.get("checkpoint"), resolved)
    normalization_record = _require_object(
        manifest.get("normalization"), "normalization record"
    )
    file_record = _verify_file_record(
        resolved,
        normalization_record.get("file"),
        NORMALIZATION_FILENAME,
    )
    normalization_path = resolved / NORMALIZATION_FILENAME
    normalization = load_normalization_state(normalization_path)
    if normalization_record != _normalization_record(normalization, file_record):
        raise TrainingArtifactError("normalization manifest does not match NPZ")

    _verify_training_record(manifest.get("training"))
    if manifest.get("resume") != {
        "selection_evaluation_export_supported": True,
        "bit_exact": False,
        "disclosure": NON_BIT_EXACT_RESUME_DISCLOSURE,
    }:
        raise TrainingArtifactError("checkpoint resume disclosure drifted")
    if manifest.get("publication") != {
        "method": "same-filesystem-staging-directory-rename",
        "complete": True,
        "overwrite_permitted": False,
    }:
        raise TrainingArtifactError("checkpoint publication record drifted")

    return VerifiedCheckpointBundle(
        root=resolved,
        model_path=resolved / MODEL_FILENAME,
        normalization_path=normalization_path,
        manifest_path=manifest_path,
        manifest=manifest,
        normalization=normalization,
    )


def checkpoint_bundle_reference(
    bundle: VerifiedCheckpointBundle | str | Path,
) -> dict[str, Any]:
    """Return the compact JSON-safe reference stored in training receipts."""

    if not isinstance(bundle, VerifiedCheckpointBundle):
        bundle = verify_checkpoint_bundle(bundle)
    checkpoint = bundle.manifest["checkpoint"]
    normalization = bundle.manifest["normalization"]
    return {
        "checkpoint_id": checkpoint["id"],
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": file_sha256(
            bundle.manifest_path,
            label=MANIFEST_FILENAME,
            error_type=TrainingArtifactError,
        ),
        "model_path": str(bundle.model_path),
        "model_sha256": checkpoint["file"]["sha256"],
        "normalization_path": str(bundle.normalization_path),
        "normalization_sha256": normalization["file"]["sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "actor_state_sha256": checkpoint["actor_state_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "obs_rms_sha256": normalization["observation_rms_sha256"],
        "ret_rms_sha256": normalization["return_rms_sha256"],
        "num_timesteps": checkpoint["counters"]["num_timesteps"],
        "active_actor_critic_transitions": checkpoint[
            "active_actor_critic_transitions"
        ],
    }


def _rename_directory(source: Path, destination: Path) -> None:
    os.rename(source, destination)


def _publish_staging_directory(staging: Path, target: Path) -> None:
    """Publish one create-new bundle, retrying transient Windows share locks."""

    for attempt in range(WINDOWS_DIRECTORY_RENAME_ATTEMPTS):
        if target.exists() or target.is_symlink():
            raise TrainingArtifactError(
                f"refusing to overwrite checkpoint bundle: {target}"
            )
        try:
            _rename_directory(staging, target)
            return
        except OSError as exc:
            if target.exists() or target.is_symlink():
                raise TrainingArtifactError(
                    f"refusing to overwrite checkpoint bundle: {target}"
                ) from exc
            transient_windows_error = getattr(exc, "winerror", None) in (
                WINDOWS_TRANSIENT_DIRECTORY_RENAME_ERRORS
            )
            final_attempt = attempt + 1 == WINDOWS_DIRECTORY_RENAME_ATTEMPTS
            if not transient_windows_error or final_attempt:
                raise TrainingArtifactError(
                    "checkpoint bundle could not be atomically published"
                ) from exc
            time.sleep(
                WINDOWS_DIRECTORY_RENAME_BASE_DELAY_SECONDS * (2**attempt)
            )

    raise AssertionError("checkpoint publication retry loop did not terminate")


def persist_checkpoint_bundle(
    destination: str | Path,
    *,
    model: PPO,
    normalizer: VecNormalize,
    training_config: Mapping[str, Any],
    seed: int,
    milestone: str | int,
    checkpoint_id: str,
    active_actor_critic_transitions: int,
) -> dict[str, Any]:
    """Atomically publish a new complete bundle and refuse every overwrite."""

    if not isinstance(model, PPO):
        raise TrainingArtifactError("model must be a PPO instance")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise TrainingArtifactError("seed must be a non-negative integer")
    milestone = _validated_milestone(milestone)
    checkpoint_id = _validated_checkpoint_id(checkpoint_id)
    active_actor_critic_transitions = _validated_active_transitions(
        active_actor_critic_transitions,
        int(model.num_timesteps),
    )
    config = _json_object(training_config, "training config")
    normalization = capture_normalization_state(normalizer)

    target = Path(destination).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise TrainingArtifactError(
            f"refusing to overwrite checkpoint bundle: {target}"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainingArtifactError("checkpoint bundle parent is not writable") from exc
    if target.exists() or target.is_symlink():
        raise TrainingArtifactError(
            f"refusing to overwrite checkpoint bundle: {target}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    published = False
    try:
        model_path = staging / MODEL_FILENAME
        try:
            model.save(model_path)
        except Exception as exc:
            raise TrainingArtifactError("SB3 model ZIP could not be written") from exc
        if not model_path.is_file():
            raise TrainingArtifactError("SB3 model save did not create model.zip")
        try:
            with model_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TrainingArtifactError("SB3 model ZIP could not be synchronized") from exc

        normalization_path = staging / NORMALIZATION_FILENAME
        _write_normalization(normalization_path, normalization)
        model_record = _file_record(model_path)
        normalization_file_record = _file_record(normalization_path)
        manifest = _manifest(
            model=model,
            checkpoint_id=checkpoint_id,
            active_actor_critic_transitions=active_actor_critic_transitions,
            model_record=model_record,
            normalization=normalization,
            normalization_record=_normalization_record(
                normalization,
                normalization_file_record,
            ),
            training_config=config,
            seed=seed,
            milestone=milestone,
        )
        _write_manifest(staging / MANIFEST_FILENAME, manifest)
        fsync_parent(staging / MANIFEST_FILENAME)
        verify_checkpoint_bundle(staging)

        _publish_staging_directory(staging, target)
        published = True
        fsync_parent(target)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return checkpoint_bundle_reference(verify_checkpoint_bundle(target))


def load_checkpoint_bundle(
    path: str | Path,
    *,
    algorithm_class: type[PPOType] = PPO,
    env: VecEnv | None = None,
    device: str | torch.device = "cpu",
) -> LoadedCheckpointBundle:
    """Verify and strongly reload a PPO bundle for training or export."""

    if not isinstance(algorithm_class, type) or not issubclass(algorithm_class, PPO):
        raise TrainingArtifactError("algorithm_class must be PPO or a PPO subclass")
    bundle = verify_checkpoint_bundle(path)
    try:
        model = algorithm_class.load(
            bundle.model_path,
            env=env,
            device=device,
            force_reset=False,
        )
    except Exception as exc:
        raise TrainingArtifactError("SB3 model ZIP could not be loaded") from exc

    checkpoint = bundle.manifest["checkpoint"]
    if (
        f"{type(model.policy).__module__}.{type(model.policy).__qualname__}"
        != checkpoint["policy_class"]
    ):
        raise TrainingArtifactError("loaded policy class does not match manifest")
    if _checkpoint_counters(model) != checkpoint["counters"]:
        raise TrainingArtifactError("loaded checkpoint counters do not match manifest")
    if policy_state_sha256(model) != checkpoint["policy_state_sha256"]:
        raise TrainingArtifactError("loaded policy state does not match manifest")
    if actor_state_sha256(model) != checkpoint["actor_state_sha256"]:
        raise TrainingArtifactError("loaded actor state does not match manifest")
    if optimizer_state_sha256(model) != checkpoint["optimizer_state_sha256"]:
        raise TrainingArtifactError("loaded optimizer state does not match manifest")
    return LoadedCheckpointBundle(bundle=bundle, model=model)


__all__ = (
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA_VERSION",
    "LoadedCheckpointBundle",
    "MANIFEST_FILENAME",
    "MODEL_FILENAME",
    "NON_BIT_EXACT_RESUME_DISCLOSURE",
    "NORMALIZATION_FILENAME",
    "NORMALIZATION_KEYS",
    "NormalizationState",
    "TrainingArtifactError",
    "VerifiedCheckpointBundle",
    "apply_normalization_state",
    "actor_state_sha256",
    "capture_normalization_state",
    "checkpoint_bundle_reference",
    "load_checkpoint_bundle",
    "load_normalization_state",
    "optimizer_state_sha256",
    "persist_checkpoint_bundle",
    "policy_state_sha256",
    "verify_checkpoint_bundle",
)
