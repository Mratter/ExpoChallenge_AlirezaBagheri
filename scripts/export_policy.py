#!/usr/bin/env python3
"""Export one selected PPO actor and prove development-only ONNX parity.

The exported graph accepts raw city observations. It embeds the selected
VecNormalize observation moments, runs the deterministic SB3 actor, and clips
the 22 actions to the environment bounds. Publication evidence is deliberately
descriptive: it does not create a source seal or authorize final-split use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium
import numpy as np
import onnx
import onnxruntime as ort
import stable_baselines3
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    ACTION_ORDER,
    OBSERVATION_ORDER,
    CityRecoveryEnv,
)
from backend.app.city.outcome import summarize_trajectory  # noqa: E402
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
)

TOOL_ID = "export_policy.py"
OBSERVATION_COUNT = 73
ACTION_COUNT = 22
ONNX_OPSET = 17
INPUT_NAME = "observation"
OUTPUT_NAME = "action"
ACTION_TOLERANCE = 1e-5
AUC_TOLERANCE = 1e-6
RESIDUAL_TOLERANCE = 1e-6
EXPECTED_DEVELOPMENT_CASES = len(DEVELOPMENT_FAMILIES) * len(DEVELOPMENT_SEEDS)
CANONICAL_DEVELOPMENT_CASES = 200
LEGACY_DEVELOPMENT_CASES = 40
SUPPORTED_DEVELOPMENT_CASE_COUNTS = frozenset(
    {LEGACY_DEVELOPMENT_CASES, CANONICAL_DEVELOPMENT_CASES}
)
EXPECTED_HORIZON_DAYS = 30
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_SB3_RUNTIME_KEYS = (
    "python",
    "numpy",
    "torch",
    "stable-baselines3",
    "gymnasium",
    "operating_system",
)
_EXPORT_RUNTIME_KEYS = (*_SB3_RUNTIME_KEYS, "onnx", "onnxruntime")


class ExportError(RuntimeError):
    """Raised when export or development parity violates its contract."""


@dataclass(frozen=True, slots=True)
class ObservationNormalization:
    """The frozen observation portion of one VecNormalize state."""

    mean: np.ndarray
    var: np.ndarray
    count: float
    epsilon: float
    clip_obs: float

    def normalize(self, raw_observations: Any) -> np.ndarray:
        """Apply SB3's float64 normalization and return policy float32 input."""

        observations = _raw_observation_batch(raw_observations)
        normalized = (observations.astype(np.float64) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(
            normalized,
            -self.clip_obs,
            self.clip_obs,
        ).astype(np.float32)

    def receipt(self) -> dict[str, Any]:
        """Return the normalization fields that define exported behavior."""

        return {
            "observation_count": OBSERVATION_COUNT,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
            "epsilon": self.epsilon,
            "clip_obs": self.clip_obs,
        }

    @property
    def state_sha256(self) -> str:
        """Match the trainer's binary observation-RMS digest exactly."""

        return _array_digest(
            np.asarray(self.mean, dtype=np.float64),
            np.asarray(self.var, dtype=np.float64),
            np.asarray([self.count], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentCase:
    """One deterministic member of the current development roster."""

    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    scenario: Any
    schedule: tuple[Any, ...]
    tape_sha256: str


@dataclass(frozen=True, slots=True)
class RolloutEvidence:
    """Compact outcome plus the exact observation and action traces."""

    row: dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray


class DeterministicActor(torch.nn.Module):
    """Raw-observation wrapper around the deterministic SB3 actor."""

    def __init__(
        self,
        policy: torch.nn.Module,
        normalization: ObservationNormalization,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer(
            "observation_mean",
            torch.tensor(normalization.mean.copy(), dtype=torch.float64),
        )
        self.register_buffer(
            "observation_var",
            torch.tensor(normalization.var.copy(), dtype=torch.float64),
        )
        self.register_buffer(
            "normalization_epsilon",
            torch.tensor(normalization.epsilon, dtype=torch.float64),
        )
        self.register_buffer(
            "observation_clip",
            torch.tensor(normalization.clip_obs, dtype=torch.float64),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize raw observations, infer the mean action, and clip it."""

        normalized = (
            observation.to(dtype=torch.float64) - self.observation_mean
        ) / torch.sqrt(self.observation_var + self.normalization_epsilon)
        normalized = torch.clamp(
            normalized,
            -self.observation_clip,
            self.observation_clip,
        ).to(dtype=torch.float32)
        action = self.policy._predict(normalized, deterministic=True)
        return torch.clamp(action, -1.0, 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def runtime_versions() -> dict[str, str]:
    """Record the libraries and operating system used for export and parity."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable-baselines3": stable_baselines3.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "gymnasium": gymnasium.__version__,
        "operating_system": platform.platform(),
    }


def _validated_runtime_versions(
    value: Any,
    *,
    label: str,
    required_keys: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ExportError(f"{label} runtime_versions must be an object")
    result: dict[str, str] = {}
    for key, version in value.items():
        if not isinstance(key, str) or not isinstance(version, str) or not version:
            raise ExportError(f"{label} runtime_versions entries must be strings")
        result[key] = version
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise ExportError(
            f"{label} runtime_versions is missing required keys: {missing}"
        )
    return result


def _load_evidence_object(path: Path, *, label: str) -> dict[str, Any]:
    receipt = load_json_object(path, label, error_type=ExportError)
    try:
        canonical_hash(receipt)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{label} contains non-canonical JSON values") from exc
    return receipt


def _required_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExportError(f"{label} must be an integer >= {minimum}")
    return value


def _required_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ExportError(f"{label} must be finite")
    return result


def _required_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ExportError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _recorded_path_matches(
    recorded: Any,
    actual: Path,
    *,
    receipt_path: Path,
    label: str,
) -> None:
    if not isinstance(recorded, str) or not recorded.strip():
        raise ExportError(f"{label} must be a nonempty path")
    candidate = Path(recorded).expanduser()
    possible = (
        {candidate.resolve()}
        if candidate.is_absolute()
        else {
            (ROOT / candidate).resolve(),
            (receipt_path.parent / candidate).resolve(),
        }
    )
    if actual.resolve() not in possible:
        raise ExportError(f"{label} does not identify the supplied file")


def _resolve_recorded_path(
    recorded: Any,
    *,
    receipt_path: Path,
    label: str,
) -> Path:
    if not isinstance(recorded, str) or not recorded.strip():
        raise ExportError(f"{label} must be a nonempty path")
    candidate = Path(recorded).expanduser()
    possible = (
        [candidate.resolve()]
        if candidate.is_absolute()
        else [
            (ROOT / candidate).resolve(),
            (receipt_path.parent / candidate).resolve(),
        ]
    )
    existing = list(dict.fromkeys(path for path in possible if path.is_file()))
    if len(existing) != 1:
        raise ExportError(f"{label} must resolve to exactly one readable file")
    return existing[0]


def training_development_case_count(receipt: dict[str, Any]) -> int:
    """Read explicit current counts while accepting legacy 40-case receipts."""

    config = receipt.get("config")
    development = receipt.get("development")
    values = [receipt.get("development_case_count")]
    if isinstance(config, dict):
        values.append(config.get("development_case_count"))
    if isinstance(development, dict):
        values.append(development.get("case_count"))
    present = [value for value in values if value is not None]
    if not present:
        raise ExportError("training receipt development case count is missing")
    counts = {
        _required_int(value, label="training development case count")
        for value in present
    }
    if len(counts) != 1:
        raise ExportError("training receipt development case counts disagree")
    count = counts.pop()
    if count not in SUPPORTED_DEVELOPMENT_CASE_COUNTS:
        raise ExportError("training development case count is unsupported")
    return count


def load_training_provenance(path: Path) -> dict[str, Any]:
    """Validate and normalize the selected run's training receipt."""

    receipt = _load_evidence_object(path, label="training receipt")
    if (
        receipt.get("status") != "complete"
        or receipt.get("training_split") != "train"
        or receipt.get("evaluation_split") != "dev"
        or receipt.get("final_split_used") is not False
    ):
        raise ExportError("training receipt is not complete train/dev evidence")
    config = receipt.get("config")
    roster = receipt.get("training_roster_and_tapes")
    checkpoint_bundles = receipt.get("checkpoint_bundles")
    if not isinstance(config, dict) or not config:
        raise ExportError("training receipt config must be a nonempty object")
    if not isinstance(roster, dict) or not roster:
        raise ExportError(
            "training receipt training_roster_and_tapes must be a nonempty object"
        )
    if not isinstance(checkpoint_bundles, dict) or not checkpoint_bundles:
        raise ExportError("training receipt checkpoint_bundles must be nonempty")
    for milestone, reference in checkpoint_bundles.items():
        if (
            not isinstance(milestone, str)
            or not milestone.isdecimal()
            or not isinstance(reference, dict)
            or int(milestone)
            != _required_int(
                reference.get("active_actor_critic_transitions"),
                label=f"checkpoint bundle {milestone} active transitions",
            )
        ):
            raise ExportError("training checkpoint_bundles contract is invalid")
    policy_seed = _required_int(
        config.get("policy_seed"), label="training config policy_seed"
    )
    registered_transitions = _required_int(
        config.get("active_actor_critic_transitions"),
        label="training config active_actor_critic_transitions",
        minimum=1,
    )
    versions = _validated_runtime_versions(
        receipt.get("runtime_versions"),
        label="training receipt",
        required_keys=_SB3_RUNTIME_KEYS,
    )
    development_case_count = training_development_case_count(receipt)
    return {
        "path": _portable_path(path),
        "sha256": file_sha256(
            path,
            label="training receipt",
            error_type=ExportError,
        ),
        "schema_version": receipt.get("schema_version"),
        "tool": receipt.get("tool"),
        "policy_seed": policy_seed,
        "registered_active_actor_critic_transitions": registered_transitions,
        "development_case_count": development_case_count,
        "config": config,
        "training_roster_and_tapes": roster,
        "checkpoint_bundles": checkpoint_bundles,
        "runtime_versions": versions,
    }


def _selection_score(
    value: Any,
    *,
    label: str,
    case_count: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"selection {label} must be an object")
    solved_count = _required_int(
        value.get("solved_count"),
        label=f"selection {label} solved_count",
    )
    if case_count not in SUPPORTED_DEVELOPMENT_CASE_COUNTS:
        raise ExportError("selection development case count is unsupported")
    if solved_count > case_count:
        raise ExportError(
            f"selection {label} solved_count exceeds {case_count}"
        )
    solve_rate = _required_float(
        value.get("solve_rate"), label=f"selection {label} solve_rate"
    )
    expected_rate = solved_count / case_count
    if abs(solve_rate - expected_rate) > 1e-12:
        raise ExportError(f"selection {label} solve_rate disagrees with solved_count")
    return {
        "solved_count": solved_count,
        "solve_rate": solve_rate,
        "mean_resilience_auc": _required_float(
            value.get("mean_resilience_auc"),
            label=f"selection {label} mean_resilience_auc",
        ),
        "mean_minimum_tail_margin": _required_float(
            value.get("mean_minimum_tail_margin"),
            label=f"selection {label} mean_minimum_tail_margin",
        ),
    }


def selection_development_case_count(receipt: dict[str, Any]) -> int:
    """Read current counts explicitly and legacy v1 evidence as 40 cases."""

    value = receipt.get("development_case_count")
    if value is None and receipt.get("schema_version") in (
        1,
        "city-recovery-checkpoint-selection-v1",
    ):
        return LEGACY_DEVELOPMENT_CASES
    count = _required_int(value, label="selection development_case_count")
    if count not in SUPPORTED_DEVELOPMENT_CASE_COUNTS:
        raise ExportError("selection development case count is unsupported")
    return count


def load_selection_provenance(
    path: Path,
    *,
    checkpoint_path: Path,
    normalization_path: Path,
    normalization: ObservationNormalization,
    training: dict[str, Any],
) -> dict[str, Any]:
    """Validate the development-only selection and its selected artifact."""

    receipt = _load_evidence_object(path, label="selection receipt")
    if receipt.get("split") != "dev" or receipt.get("final_split_used") is not False:
        raise ExportError("selection receipt must describe only the development split")
    if receipt.get("schema_version") is None or not isinstance(receipt.get("tool"), str):
        raise ExportError("selection receipt schema_version and tool are required")
    development_case_count = selection_development_case_count(receipt)
    if development_case_count != training["development_case_count"]:
        raise ExportError(
            "selection and training development case counts disagree"
        )

    ranking = receipt.get("ranking")
    expected_tie_breaks = [
        "earlier_active_actor_critic_transitions",
        "lower_policy_seed",
    ]
    if (
        not isinstance(ranking, dict)
        or ranking.get("primary_metric") != "solved_count"
        or ranking.get("resilience_auc_used_for_selection") is not False
        or ranking.get("tie_break_order") != expected_tie_breaks
    ):
        raise ExportError("selection ranking contract is invalid")

    selected = receipt.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ExportError("selection selected_checkpoint must be an object")
    checkpoint_id = selected.get("id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ExportError("selected checkpoint id must be nonempty")
    _recorded_path_matches(
        selected.get("path"),
        checkpoint_path,
        receipt_path=path,
        label="selected checkpoint path",
    )
    _recorded_path_matches(
        selected.get("normalization_path"),
        normalization_path,
        receipt_path=path,
        label="selected normalization path",
    )
    checkpoint_sha256 = _required_sha256(
        selected.get("sha256"), label="selected checkpoint sha256"
    )
    actual_checkpoint_sha256 = file_sha256(
        checkpoint_path,
        label="SB3 checkpoint",
        error_type=ExportError,
    )
    if checkpoint_sha256 != actual_checkpoint_sha256:
        raise ExportError("selected checkpoint SHA-256 does not match supplied ZIP")
    normalization_file_sha256 = _required_sha256(
        selected.get("normalization_file_sha256"),
        label="selected normalization file sha256",
    )
    actual_normalization_sha256 = file_sha256(
        normalization_path,
        label="normalization NPZ",
        error_type=ExportError,
    )
    if normalization_file_sha256 != actual_normalization_sha256:
        raise ExportError("selected normalization SHA-256 does not match supplied NPZ")
    observation_rms_sha256 = _required_sha256(
        selected.get("observation_rms_sha256"),
        label="selected observation RMS sha256",
    )
    if observation_rms_sha256 != normalization.state_sha256:
        raise ExportError("selected observation RMS digest does not match NPZ state")

    policy_seed = _required_int(
        selected.get("policy_seed"), label="selected policy_seed"
    )
    if policy_seed != training["policy_seed"]:
        raise ExportError("selected policy_seed does not match training config")
    active_transitions = _required_int(
        selected.get("active_actor_critic_transitions"),
        label="selected active_actor_critic_transitions",
    )
    if active_transitions > training["registered_active_actor_critic_transitions"]:
        raise ExportError("selected transitions exceed the registered training budget")

    training_receipt_path = Path(training["path"])
    if not training_receipt_path.is_absolute():
        training_receipt_path = ROOT / training_receipt_path
    _recorded_path_matches(
        selected.get("training_receipt_path"),
        training_receipt_path,
        receipt_path=path,
        label="selected training receipt path",
    )
    selected_training_sha256 = _required_sha256(
        selected.get("training_receipt_sha256"),
        label="selected training receipt sha256",
    )
    if selected_training_sha256 != training["sha256"]:
        raise ExportError("selected training receipt SHA-256 disagrees")

    matching_bundles = [
        reference
        for reference in training["checkpoint_bundles"].values()
        if reference.get("checkpoint_id") == checkpoint_id
    ]
    if len(matching_bundles) != 1:
        raise ExportError(
            "selected checkpoint must match exactly one training checkpoint bundle"
        )
    bundle = matching_bundles[0]
    bundle_checks = {
        "model_sha256": checkpoint_sha256,
        "normalization_sha256": normalization_file_sha256,
        "obs_rms_sha256": observation_rms_sha256,
        "active_actor_critic_transitions": active_transitions,
    }
    for key, expected in bundle_checks.items():
        if bundle.get(key) != expected:
            raise ExportError(f"training checkpoint bundle {key} disagrees")
    _recorded_path_matches(
        bundle.get("model_path"),
        checkpoint_path,
        receipt_path=training_receipt_path,
        label="training checkpoint bundle model_path",
    )
    _recorded_path_matches(
        bundle.get("normalization_path"),
        normalization_path,
        receipt_path=training_receipt_path,
        label="training checkpoint bundle normalization_path",
    )
    manifest_path = _resolve_recorded_path(
        bundle.get("manifest_path"),
        receipt_path=training_receipt_path,
        label="training checkpoint bundle manifest_path",
    )
    manifest_sha256 = _required_sha256(
        bundle.get("manifest_sha256"),
        label="training checkpoint bundle manifest_sha256",
    )
    if file_sha256(
        manifest_path,
        label="checkpoint bundle manifest",
        error_type=ExportError,
    ) != manifest_sha256:
        raise ExportError("checkpoint bundle manifest SHA-256 disagrees")
    checkpoint_manifest = _load_evidence_object(
        manifest_path, label="checkpoint bundle manifest"
    )
    manifest_checkpoint = checkpoint_manifest.get("checkpoint")
    manifest_normalization = checkpoint_manifest.get("normalization")
    manifest_training = checkpoint_manifest.get("training")
    manifest_resume = checkpoint_manifest.get("resume")
    manifest_publication = checkpoint_manifest.get("publication")
    if (
        checkpoint_manifest.get("schema_version") is None
        or not isinstance(checkpoint_manifest.get("kind"), str)
        or not isinstance(manifest_checkpoint, dict)
        or not isinstance(manifest_normalization, dict)
        or not isinstance(manifest_training, dict)
        or not isinstance(manifest_resume, dict)
        or not isinstance(manifest_publication, dict)
    ):
        raise ExportError("checkpoint bundle manifest structure is invalid")
    if (
        manifest_checkpoint.get("id") != checkpoint_id
        or manifest_checkpoint.get("algorithm") != "PPO"
        or manifest_checkpoint.get("active_actor_critic_transitions")
        != active_transitions
    ):
        raise ExportError("checkpoint bundle manifest checkpoint identity disagrees")
    manifest_model_file = manifest_checkpoint.get("file")
    manifest_normalization_file = manifest_normalization.get("file")
    if not isinstance(manifest_model_file, dict) or not isinstance(
        manifest_normalization_file, dict
    ):
        raise ExportError("checkpoint bundle manifest file records are invalid")
    _recorded_path_matches(
        manifest_model_file.get("path"),
        checkpoint_path,
        receipt_path=manifest_path,
        label="checkpoint manifest model path",
    )
    _recorded_path_matches(
        manifest_normalization_file.get("path"),
        normalization_path,
        receipt_path=manifest_path,
        label="checkpoint manifest normalization path",
    )
    if (
        manifest_model_file.get("sha256") != checkpoint_sha256
        or manifest_model_file.get("size_bytes") != checkpoint_path.stat().st_size
        or manifest_normalization_file.get("sha256")
        != normalization_file_sha256
        or manifest_normalization_file.get("size_bytes")
        != normalization_path.stat().st_size
    ):
        raise ExportError("checkpoint bundle manifest file identity disagrees")
    if (
        manifest_normalization.get("observation_rms_sha256")
        != observation_rms_sha256
        or manifest_normalization.get("observation_shape")
        != [OBSERVATION_COUNT]
        or manifest_normalization.get("norm_obs") is not True
        or manifest_normalization.get("epsilon") != normalization.epsilon
        or manifest_normalization.get("clip_obs") != normalization.clip_obs
    ):
        raise ExportError("checkpoint manifest observation normalization disagrees")
    if (
        manifest_training.get("config") != training["config"]
        or manifest_training.get("config_sha256")
        != canonical_hash(training["config"])
        or manifest_training.get("seed") != policy_seed
    ):
        raise ExportError("checkpoint manifest training provenance disagrees")
    if (
        manifest_resume.get("selection_evaluation_export_supported") is not True
        or manifest_publication.get("complete") is not True
        or manifest_publication.get("overwrite_permitted") is not False
    ):
        raise ExportError("checkpoint bundle is incomplete or not exportable")

    winner = _selection_score(
        receipt.get("winner"),
        label="winner",
        case_count=development_case_count,
    )
    runner_up = _selection_score(
        receipt.get("runner_up"),
        label="runner_up",
        case_count=development_case_count,
    )
    if winner["solved_count"] < runner_up["solved_count"]:
        raise ExportError("selection winner has fewer solves than runner-up")
    margin = receipt.get("margin")
    if not isinstance(margin, dict):
        raise ExportError("selection margin must be an object")
    solved_margin = _required_int(
        margin.get("solved_cases"), label="selection solved-case margin"
    )
    percentage_margin = _required_float(
        margin.get("percentage_points"),
        label="selection percentage-point margin",
    )
    if solved_margin != winner["solved_count"] - runner_up["solved_count"]:
        raise ExportError("selection solved-case margin is inconsistent")
    expected_percentage_margin = 100.0 * (
        winner["solve_rate"] - runner_up["solve_rate"]
    )
    if abs(percentage_margin - expected_percentage_margin) > 1e-10:
        raise ExportError("selection percentage-point margin is inconsistent")

    tie_break = receipt.get("tie_break")
    if (
        not isinstance(tie_break, dict)
        or not isinstance(tie_break.get("used"), bool)
    ):
        raise ExportError("selection tie_break must contain used and level")
    if tie_break["used"] != (solved_margin == 0):
        raise ExportError("selection tie_break usage disagrees with solved margin")
    if tie_break["used"] and tie_break.get("level") not in expected_tie_breaks:
        raise ExportError("selection tie_break level is not preregistered")
    if not tie_break["used"] and tie_break.get("level") is not None:
        raise ExportError("unused selection tie_break level must be null")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ExportError("selection candidates must contain at least two rows")
    if receipt.get("candidate_count") != len(candidates):
        raise ExportError("selection candidate_count disagrees with candidates")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise ExportError("selection candidates must be objects")
    if candidates[0].get("id") != checkpoint_id:
        raise ExportError("selected checkpoint is not the first ranked candidate")
    if _selection_score(
        candidates[0].get("development"),
        label="first candidate",
        case_count=development_case_count,
    ) != winner:
        raise ExportError("selection winner disagrees with first candidate")
    if _selection_score(
        candidates[1].get("development"),
        label="second candidate",
        case_count=development_case_count,
    ) != runner_up:
        raise ExportError("selection runner-up disagrees with second candidate")
    policy_seed_set = sorted(
        {
            _required_int(
                candidate.get("policy_seed"),
                label="selection candidate policy_seed",
            )
            for candidate in candidates
        }
    )
    if len(policy_seed_set) != 5:
        raise ExportError("selection must cover exactly five policy seeds")
    versions = _validated_runtime_versions(
        receipt.get("runtime_versions"),
        label="selection receipt",
        required_keys=_SB3_RUNTIME_KEYS,
    )
    return {
        "path": _portable_path(path),
        "sha256": file_sha256(
            path,
            label="selection receipt",
            error_type=ExportError,
        ),
        "schema_version": receipt["schema_version"],
        "tool": receipt["tool"],
        "split": "dev",
        "development_case_count": development_case_count,
        "ranking": ranking,
        "selected_checkpoint": {
            "id": checkpoint_id,
            "path": _portable_path(checkpoint_path),
            "sha256": checkpoint_sha256,
            "policy_seed": policy_seed,
            "active_actor_critic_transitions": active_transitions,
            "normalization_path": _portable_path(normalization_path),
            "normalization_file_sha256": normalization_file_sha256,
            "observation_rms_sha256": observation_rms_sha256,
        },
        "winner": winner,
        "runner_up": runner_up,
        "margin": {
            "solved_cases": solved_margin,
            "percentage_points": percentage_margin,
        },
        "tie_break": tie_break,
        "candidate_count": len(candidates),
        "policy_seed_set": policy_seed_set,
        "checkpoint_bundle": {
            "manifest_path": _portable_path(manifest_path),
            "manifest_sha256": manifest_sha256,
            "policy_state_sha256": bundle.get("policy_state_sha256"),
            "actor_state_sha256": bundle.get("actor_state_sha256"),
            "optimizer_state_sha256": bundle.get("optimizer_state_sha256"),
            "num_timesteps": bundle.get("num_timesteps"),
            "return_rms_sha256": bundle.get("ret_rms_sha256"),
        },
        "runtime_versions": versions,
    }


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _raw_observation_batch(raw_observations: Any) -> np.ndarray:
    try:
        observations = np.asarray(raw_observations, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExportError("raw observations must be a numeric array") from exc
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_COUNT:
        raise ExportError(
            f"raw observations must have shape [batch, {OBSERVATION_COUNT}]"
        )
    if observations.shape[0] == 0 or not np.all(np.isfinite(observations)):
        raise ExportError("raw observations must be nonempty and finite")
    return observations


def _action_batch(actions: Any, expected_rows: int) -> np.ndarray:
    try:
        action_array = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExportError("policy actions must be numeric") from exc
    if action_array.ndim == 1 and expected_rows == 1:
        action_array = action_array.reshape(1, -1)
    if action_array.shape != (expected_rows, ACTION_COUNT):
        raise ExportError(
            f"policy actions must have shape [{expected_rows}, {ACTION_COUNT}]"
        )
    if not np.all(np.isfinite(action_array)):
        raise ExportError("policy actions must be finite")
    if np.any(np.abs(action_array) > 1.0):
        raise ExportError("policy actions must remain within [-1, 1]")
    return action_array


def _npz_value(
    archive: Any,
    aliases: tuple[str, ...],
    *,
    label: str,
) -> np.ndarray:
    present = [name for name in aliases if name in archive.files]
    if not present:
        raise ExportError(
            f"normalization NPZ is missing {label}; expected one of {aliases}"
        )
    values = [np.asarray(archive[name]) for name in present]
    if any(
        value.shape != values[0].shape
        or not np.array_equal(value, values[0], equal_nan=True)
        for value in values[1:]
    ):
        raise ExportError(f"normalization NPZ has conflicting {label} aliases")
    return values[0]


def _npz_scalar(
    archive: Any,
    aliases: tuple[str, ...],
    *,
    label: str,
) -> float:
    value = _npz_value(archive, aliases, label=label)
    if value.size != 1:
        raise ExportError(f"normalization {label} must be scalar")
    try:
        return float(value.reshape(()))
    except (TypeError, ValueError) as exc:
        raise ExportError(f"normalization {label} must be numeric") from exc


def load_observation_normalization(path: Path) -> ObservationNormalization:
    """Load and validate the observation transform from a checkpoint NPZ."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            mean = _npz_value(
                archive,
                ("obs_mean", "obs_rms_mean"),
                label="observation mean",
            ).astype(np.float64, copy=True)
            var = _npz_value(
                archive,
                ("obs_var", "obs_rms_var"),
                label="observation variance",
            ).astype(np.float64, copy=True)
            count = _npz_scalar(
                archive,
                ("obs_count", "obs_rms_count"),
                label="observation count",
            )
            epsilon = _npz_scalar(
                archive,
                ("epsilon", "vecnormalize_epsilon"),
                label="epsilon",
            )
            clip_obs = _npz_scalar(
                archive,
                ("clip_obs", "vecnormalize_clip_obs"),
                label="observation clip",
            )
            if "norm_obs" in archive.files:
                norm_obs = np.asarray(archive["norm_obs"])
                if norm_obs.size != 1 or not bool(norm_obs.reshape(())):
                    raise ExportError(
                        "selected normalization state must enable norm_obs"
                    )
    except ExportError:
        raise
    except (OSError, ValueError) as exc:
        raise ExportError(f"normalization NPZ is missing or invalid: {path}") from exc

    if mean.shape != (OBSERVATION_COUNT,) or var.shape != (OBSERVATION_COUNT,):
        raise ExportError(
            f"normalization mean and variance must contain {OBSERVATION_COUNT} values"
        )
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(var))
        or np.any(var < 0.0)
        or not np.isfinite(count)
        or count <= 0.0
        or not np.isfinite(epsilon)
        or epsilon <= 0.0
        or not np.isfinite(clip_obs)
        or clip_obs <= 0.0
    ):
        raise ExportError("normalization state contains invalid values")
    mean.setflags(write=False)
    var.setflags(write=False)
    return ObservationNormalization(mean, var, count, epsilon, clip_obs)


def load_sb3_checkpoint(path: Path) -> PPO:
    """Load one PPO ZIP on CPU and validate the city policy interface."""

    try:
        model = PPO.load(path, device="cpu", print_system_info=False)
    except Exception as exc:
        raise ExportError(f"SB3 checkpoint is missing or invalid: {path}") from exc
    if tuple(model.observation_space.shape or ()) != (OBSERVATION_COUNT,):
        raise ExportError("SB3 checkpoint observation shape is not (73,)")
    if tuple(model.action_space.shape or ()) != (ACTION_COUNT,):
        raise ExportError("SB3 checkpoint action shape is not (22,)")
    if not (
        np.array_equal(model.action_space.low, -np.ones(ACTION_COUNT))
        and np.array_equal(model.action_space.high, np.ones(ACTION_COUNT))
    ):
        raise ExportError("SB3 checkpoint action bounds are not [-1, 1]")
    model.policy.to("cpu")
    model.policy.set_training_mode(False)
    return model


def _cpu_session(payload: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ExportError("ONNX artifact is not loadable on CPU") from exc


def _serialized_onnx_shape(
    value_info: onnx.ValueInfoProto,
) -> list[str | int | None]:
    dimensions = value_info.type.tensor_type.shape.dim
    return [
        dimension.dim_value
        if dimension.HasField("dim_value")
        else dimension.dim_param
        if dimension.HasField("dim_param")
        else None
        for dimension in dimensions
    ]


def inspect_onnx_contract(path: Path) -> dict[str, Any]:
    """Inspect the exact runtime interface and CPU-only execution contract."""

    try:
        payload = path.read_bytes()
        graph = onnx.load_model_from_string(payload)
    except (OSError, ValueError) as exc:
        raise ExportError(f"ONNX artifact is missing or invalid: {path}") from exc
    default_opsets = [item.version for item in graph.opset_import if item.domain == ""]
    if default_opsets != [ONNX_OPSET]:
        raise ExportError(f"ONNX graph must use default-domain opset {ONNX_OPSET}")
    if any(initializer.external_data for initializer in graph.graph.initializer):
        raise ExportError("ONNX graph must be self-contained")

    input_shape = ["batch", OBSERVATION_COUNT]
    output_shape = ["batch", ACTION_COUNT]
    graph_inputs = graph.graph.input
    graph_outputs = graph.graph.output
    if (
        len(graph_inputs) != 1
        or graph_inputs[0].name != INPUT_NAME
        or graph_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _serialized_onnx_shape(graph_inputs[0]) != input_shape
    ):
        raise ExportError(
            "serialized ONNX input must be observation: "
            "tensor(float)[batch, 73]"
        )
    if (
        len(graph_outputs) != 1
        or graph_outputs[0].name != OUTPUT_NAME
        or graph_outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _serialized_onnx_shape(graph_outputs[0]) != output_shape
    ):
        raise ExportError(
            "serialized ONNX output must be action: tensor(float)[batch, 22]"
        )

    session = _cpu_session(payload)
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ExportError("ONNX session must use only CPUExecutionProvider")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != INPUT_NAME
        or inputs[0].type != "tensor(float)"
        or list(inputs[0].shape) != input_shape
    ):
        raise ExportError(
            "ONNX Runtime input must be observation: tensor(float)[batch, 73]"
        )
    if (
        len(outputs) != 1
        or outputs[0].name != OUTPUT_NAME
        or outputs[0].type != "tensor(float)"
        or list(outputs[0].shape) != output_shape
    ):
        raise ExportError(
            "ONNX Runtime output must be action: tensor(float)[batch, 22]"
        )

    smoke_shapes: list[list[int]] = []
    for batch_size in (1, 3):
        try:
            result = session.run(
                [OUTPUT_NAME],
                {
                    INPUT_NAME: np.zeros(
                        (batch_size, OBSERVATION_COUNT), dtype=np.float32
                    )
                },
            )[0]
        except Exception as exc:
            raise ExportError("ONNX dynamic-batch smoke inference failed") from exc
        _action_batch(result, batch_size)
        smoke_shapes.append(list(result.shape))
    return {
        "format": "onnx",
        "opset": ONNX_OPSET,
        "self_contained": True,
        "providers": session.get_providers(),
        "input": {
            "name": INPUT_NAME,
            "type": inputs[0].type,
            "shape": input_shape,
            "normalization": "embedded_vecnormalize_observation_rms",
        },
        "output": {
            "name": OUTPUT_NAME,
            "type": outputs[0].type,
            "shape": output_shape,
            "action_clip": [-1.0, 1.0],
        },
        "dynamic_batch_smoke_shapes": smoke_shapes,
    }


def load_onnx_cpu_session(path: Path) -> ort.InferenceSession:
    """Validate an exported artifact and return its CPU-only ORT session."""

    inspect_onnx_contract(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExportError(f"ONNX artifact is missing or unreadable: {path}") from exc
    return _cpu_session(payload)


def _infer_and_save_onnx_shapes(path: Path) -> None:
    """Persist inferred metadata without changing the exported computation."""

    try:
        graph = onnx.load(path, load_external_data=False)
        inferred = onnx.shape_inference.infer_shapes(
            graph,
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )
        onnx.checker.check_model(inferred)
        onnx.save_model(inferred, path, save_as_external_data=False)
    except Exception as exc:
        raise ExportError("ONNX shape inference failed") from exc


def export_deterministic_actor(
    model: PPO,
    normalization: ObservationNormalization,
    output_path: Path,
) -> dict[str, Any]:
    """Create a new self-contained opset-17 ONNX artifact atomically."""

    output_path = output_path.resolve()
    if output_path.exists():
        raise ExportError(f"refusing to overwrite ONNX artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    wrapper = DeterministicActor(model.policy, normalization)
    wrapper.eval()
    example = torch.zeros((2, OBSERVATION_COUNT), dtype=torch.float32)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                example,
                temporary,
                export_params=True,
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={
                    INPUT_NAME: {0: "batch"},
                    OUTPUT_NAME: {0: "batch"},
                },
                keep_initializers_as_inputs=False,
                dynamo=False,
            )
        _infer_and_save_onnx_shapes(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        contract = inspect_onnx_contract(temporary)
        if output_path.exists():
            raise ExportError(f"refusing to overwrite ONNX artifact: {output_path}")
        os.replace(temporary, output_path)
        fsync_parent(output_path)
    except ExportError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ExportError("deterministic actor ONNX export failed") from exc
    return contract


def sb3_actions(
    model: PPO,
    normalization: ObservationNormalization,
    raw_observations: Any,
) -> np.ndarray:
    """Return clipped deterministic SB3 actions for raw observations."""

    observations = _raw_observation_batch(raw_observations)
    try:
        actions, _ = model.predict(
            normalization.normalize(observations),
            deterministic=True,
        )
    except Exception as exc:
        raise ExportError("SB3 deterministic inference failed") from exc
    return _action_batch(np.clip(actions, -1.0, 1.0), observations.shape[0])


def onnx_actions(session: ort.InferenceSession, raw_observations: Any) -> np.ndarray:
    """Return actions from the raw-observation ONNX interface."""

    observations = _raw_observation_batch(raw_observations)
    try:
        actions = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: observations},
        )[0]
    except Exception as exc:
        raise ExportError("ONNX deterministic inference failed") from exc
    return _action_batch(actions, observations.shape[0])


def action_parity(
    model: PPO,
    normalization: ObservationNormalization,
    session: ort.InferenceSession,
    raw_observations: Any,
    *,
    tolerance: float = ACTION_TOLERANCE,
) -> dict[str, Any]:
    """Compare SB3 and ONNX on one identical batch of raw observations."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ExportError("action tolerance must be positive and finite")
    observations = _raw_observation_batch(raw_observations)
    reference = sb3_actions(model, normalization, observations)
    candidate = onnx_actions(session, observations)
    errors = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    flat_index = int(np.argmax(errors))
    sample_index, action_index = np.unravel_index(flat_index, errors.shape)
    maximum = float(errors[sample_index, action_index])
    return {
        "sample_count": int(observations.shape[0]),
        "element_count": int(errors.size),
        "tolerance": tolerance,
        "relative_tolerance": 0.0,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": float(np.mean(errors)),
        "elements_over_tolerance": int(np.count_nonzero(errors > tolerance)),
        "maximum_error_location": {
            "sample_index": int(sample_index),
            "action_index": int(action_index),
            "action_name": ACTION_ORDER[action_index],
        },
        "passed": bool(maximum <= tolerance),
    }


def development_cases() -> list[DevelopmentCase]:
    """Build only the canonical current development roster."""

    cases: list[DevelopmentCase] = []
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = tuple(generate_disaster_tape(scenario, tape_seed))
            tape_payload = [asdict(shock) for shock in schedule]
            cases.append(
                DevelopmentCase(
                    row_id=f"{family.id}:{case_seed}",
                    family_id=family.id,
                    case_seed=case_seed,
                    tape_seed=tape_seed,
                    scenario=scenario,
                    schedule=schedule,
                    tape_sha256=canonical_hash(tape_payload),
                )
            )
    if (
        EXPECTED_DEVELOPMENT_CASES != CANONICAL_DEVELOPMENT_CASES
        or len(cases) != EXPECTED_DEVELOPMENT_CASES
        or len({case.row_id for case in cases}) != EXPECTED_DEVELOPMENT_CASES
        or any(case.scenario.horizon_days != EXPECTED_HORIZON_DAYS for case in cases)
    ):
        raise ExportError("development roster contract drifted")
    return cases


def _rollout(
    case: DevelopmentCase,
    actor: Callable[[np.ndarray], np.ndarray],
    *,
    label: str,
) -> RolloutEvidence:
    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        case.schedule,
        collect_evidence=True,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    try:
        observation, _ = environment.reset(seed=case.tape_seed)
        terminated = False
        while not terminated:
            raw = np.asarray(observation, dtype=np.float32).copy()
            action = _action_batch(actor(raw), 1)[0]
            observations.append(raw)
            actions.append(action.copy())
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise ExportError(f"unexpected truncated episode: {case.row_id}")
        summary = summarize_trajectory(label, environment.trajectory, case.scenario)
    finally:
        environment.close()

    observation_trace = np.asarray(observations, dtype=np.float32)
    action_trace = np.asarray(actions, dtype=np.float32)
    if observation_trace.shape != (EXPECTED_HORIZON_DAYS, OBSERVATION_COUNT):
        raise ExportError(f"observation trace contract drifted: {case.row_id}")
    if action_trace.shape != (EXPECTED_HORIZON_DAYS, ACTION_COUNT):
        raise ExportError(f"action trace contract drifted: {case.row_id}")
    outcome = summary["absolute_outcome"]
    row = {
        "row_id": case.row_id,
        "family_id": case.family_id,
        "case_seed": case.case_seed,
        "tape_seed": case.tape_seed,
        "tape_sha256": case.tape_sha256,
        "solved": bool(outcome["solved"]),
        "reason_codes": list(outcome["reason_codes"]),
        "resilience_auc": float(summary["rauc"]),
        "hard_violation_count": int(summary["hard_violation_count"]),
        "maximum_conservation_residual": float(
            summary["max_logistics_conservation_residual"]
        ),
        "trajectory_sha256": str(summary["trajectory_sha256"]),
        "observation_trace_sha256": _array_digest(observation_trace),
        "action_trace_sha256": _array_digest(action_trace),
    }
    return RolloutEvidence(row, observation_trace, action_trace)


def _rollout_replay_matches(
    first: RolloutEvidence,
    second: RolloutEvidence,
) -> bool:
    return bool(
        canonical_hash(first.row) == canonical_hash(second.row)
        and np.array_equal(first.observations, second.observations)
        and np.array_equal(first.actions, second.actions)
    )


def development_parity(
    model: PPO,
    normalization: ObservationNormalization,
    session: ort.InferenceSession,
) -> dict[str, Any]:
    """Run full deterministic SB3-versus-ONNX development parity."""

    def sb3_actor(raw: np.ndarray) -> np.ndarray:
        return sb3_actions(model, normalization, raw)

    def onnx_actor(raw: np.ndarray) -> np.ndarray:
        return onnx_actions(session, raw)

    rows: list[dict[str, Any]] = []
    for case in development_cases():
        sb3_first = _rollout(case, sb3_actor, label="sb3_reference")
        onnx_first = _rollout(case, onnx_actor, label="onnx_candidate")
        comparisons = action_parity(
            model,
            normalization,
            session,
            sb3_first.observations,
        )
        sb3_replay = _rollout(case, sb3_actor, label="sb3_reference")
        onnx_replay = _rollout(case, onnx_actor, label="onnx_candidate")
        sb3_deterministic = _rollout_replay_matches(sb3_first, sb3_replay)
        onnx_deterministic = _rollout_replay_matches(onnx_first, onnx_replay)
        auc_error = abs(
            sb3_first.row["resilience_auc"] - onnx_first.row["resilience_auc"]
        )
        rows.append(
            {
                "row_id": case.row_id,
                "family_id": case.family_id,
                "case_seed": case.case_seed,
                "tape_seed": case.tape_seed,
                "tape_sha256": case.tape_sha256,
                "action_comparison": comparisons,
                "action_comparison_observation_source": (
                    "sb3_reference_raw_observation_trace"
                ),
                "sb3": sb3_first.row,
                "onnx": onnx_first.row,
                "outcome_match": (
                    sb3_first.row["solved"] == onnx_first.row["solved"]
                ),
                "resilience_auc_absolute_error": auc_error,
                "sb3_deterministic_replay": sb3_deterministic,
                "onnx_deterministic_replay": onnx_deterministic,
            }
        )

    action_sample_count = sum(
        row["action_comparison"]["sample_count"] for row in rows
    )
    action_element_count = sum(
        row["action_comparison"]["element_count"] for row in rows
    )
    if (
        action_sample_count
        != EXPECTED_DEVELOPMENT_CASES * EXPECTED_HORIZON_DAYS
        or action_element_count
        != EXPECTED_DEVELOPMENT_CASES * EXPECTED_HORIZON_DAYS * ACTION_COUNT
    ):
        raise ExportError("development action sample contract drifted")

    sb3_solves = sum(row["sb3"]["solved"] for row in rows)
    onnx_solves = sum(row["onnx"]["solved"] for row in rows)
    outcome_mismatches = sum(not row["outcome_match"] for row in rows)
    action_maximum = max(
        row["action_comparison"]["maximum_absolute_error"] for row in rows
    )
    action_elements_over = sum(
        row["action_comparison"]["elements_over_tolerance"] for row in rows
    )
    auc_maximum = max(row["resilience_auc_absolute_error"] for row in rows)
    replay_mismatches = sum(
        int(not row["sb3_deterministic_replay"])
        + int(not row["onnx_deterministic_replay"])
        for row in rows
    )
    onnx_hard_violations = sum(
        row["onnx"]["hard_violation_count"] for row in rows
    )
    onnx_maximum_residual = max(
        row["onnx"]["maximum_conservation_residual"] for row in rows
    )
    conditions = {
        "complete_development_roster": (
            len(rows) == EXPECTED_DEVELOPMENT_CASES
        ),
        "all_actions_within_tolerance": (
            action_maximum <= ACTION_TOLERANCE and action_elements_over == 0
        ),
        "all_actions_finite_and_bounded": True,
        "all_per_case_outcomes_equal": outcome_mismatches == 0,
        "aggregate_solve_counts_equal": sb3_solves == onnx_solves,
        "all_resilience_auc_values_within_tolerance": (
            auc_maximum <= AUC_TOLERANCE
        ),
        "onnx_has_zero_hard_violations": onnx_hard_violations == 0,
        "onnx_conservation_within_tolerance": (
            onnx_maximum_residual <= RESIDUAL_TOLERANCE
        ),
        "both_runtimes_replay_deterministically": replay_mismatches == 0,
    }
    return {
        "split": "dev",
        "final_split_used": False,
        "expected_case_count": EXPECTED_DEVELOPMENT_CASES,
        "case_count": len(rows),
        "action_sample_count": action_sample_count,
        "action_element_count": action_element_count,
        "action_tolerance": ACTION_TOLERANCE,
        "action_relative_tolerance": 0.0,
        "maximum_action_absolute_error": action_maximum,
        "action_elements_over_tolerance": action_elements_over,
        "resilience_auc_tolerance": AUC_TOLERANCE,
        "maximum_resilience_auc_absolute_error": auc_maximum,
        "sb3_solved_count": int(sb3_solves),
        "onnx_solved_count": int(onnx_solves),
        "per_case_outcome_mismatch_count": int(outcome_mismatches),
        "deterministic_replay_mismatch_count": int(replay_mismatches),
        "onnx_hard_violation_count": int(onnx_hard_violations),
        "conservation_residual_tolerance": RESIDUAL_TOLERANCE,
        "onnx_maximum_conservation_residual": float(onnx_maximum_residual),
        "conditions": conditions,
        "passed": all(conditions.values()),
        "rows_sha256": canonical_hash(rows),
        "rows": rows,
    }


def build_parity_receipt(
    *,
    model_id: str,
    checkpoint_path: Path,
    normalization_path: Path,
    normalization: ObservationNormalization,
    onnx_path: Path,
    interface: dict[str, Any],
    parity: dict[str, Any],
    training: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable payload written after one full parity run."""

    if not model_id.strip():
        raise ExportError("model id must be nonempty")
    case_counts = (
        EXPECTED_DEVELOPMENT_CASES,
        parity.get("expected_case_count"),
        parity.get("case_count"),
        training.get("development_case_count"),
        selection.get("development_case_count"),
    )
    if (
        EXPECTED_DEVELOPMENT_CASES != CANONICAL_DEVELOPMENT_CASES
        or any(count != EXPECTED_DEVELOPMENT_CASES for count in case_counts)
    ):
        raise ExportError(
            "parity publication requires one consistent 200-case development contract"
        )
    return {
        "schema_version": "city-recovery-onnx-parity-v1",
        "created_at": _utc_now(),
        "tool": TOOL_ID,
        "model_id": model_id,
        "split": "dev",
        "development_case_count": parity["case_count"],
        "final_split_used": False,
        "source_checkpoint": {
            "id": selection["selected_checkpoint"]["id"],
            "path": _portable_path(checkpoint_path),
            "sha256": file_sha256(
                checkpoint_path,
                label="SB3 checkpoint",
                error_type=ExportError,
            ),
            "policy_seed": selection["selected_checkpoint"]["policy_seed"],
            "active_actor_critic_transitions": selection[
                "selected_checkpoint"
            ]["active_actor_critic_transitions"],
            "bundle_manifest": selection["checkpoint_bundle"],
        },
        "normalization": {
            "path": _portable_path(normalization_path),
            "file_sha256": file_sha256(
                normalization_path,
                label="normalization NPZ",
                error_type=ExportError,
            ),
            "state_sha256": normalization.state_sha256,
            "observation_rms_sha256": normalization.state_sha256,
            "count": normalization.count,
            "epsilon": normalization.epsilon,
            "clip_obs": normalization.clip_obs,
            "normalization_baked_into_graph": True,
        },
        "onnx_artifact": {
            "path": _portable_path(onnx_path),
            "sha256": file_sha256(
                onnx_path,
                label="ONNX artifact",
                error_type=ExportError,
            ),
        },
        "training": training,
        "selection": selection,
        "runtime_versions": {
            "training": training["runtime_versions"],
            "selection": selection["runtime_versions"],
            "export_and_parity": _validated_runtime_versions(
                runtime_versions(),
                label="export and parity",
                required_keys=_EXPORT_RUNTIME_KEYS,
            ),
        },
        "interface": interface,
        "parity": parity,
    }


def build_manifest(
    *,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Build lightweight descriptive metadata for an accepted artifact."""

    parity = receipt["parity"]
    if not parity.get("passed"):
        raise ExportError("cannot publish a manifest for failed parity")
    if (
        receipt.get("development_case_count") != EXPECTED_DEVELOPMENT_CASES
        or receipt.get("training", {}).get("development_case_count")
        != EXPECTED_DEVELOPMENT_CASES
        or receipt.get("selection", {}).get("development_case_count")
        != EXPECTED_DEVELOPMENT_CASES
        or parity.get("expected_case_count") != EXPECTED_DEVELOPMENT_CASES
        or parity.get("case_count") != EXPECTED_DEVELOPMENT_CASES
    ):
        raise ExportError(
            "manifest publication requires one consistent 200-case development contract"
        )
    return {
        "schema_version": "city-recovery-onnx-manifest-v1",
        "model_id": receipt["model_id"],
        "publication_status": "development_parity_passed",
        "runtime_enforcement": "descriptive_only",
        "artifact": {
            **receipt["onnx_artifact"],
            "format": "onnx",
            "opset": ONNX_OPSET,
        },
        "source_checkpoint": receipt["source_checkpoint"],
        "normalization": receipt["normalization"],
        "interface": receipt["interface"],
        "observation_order_sha256": canonical_hash(list(OBSERVATION_ORDER)),
        "action_order_sha256": canonical_hash(list(ACTION_ORDER)),
        "training": {
            "receipt_path": receipt["training"]["path"],
            "receipt_sha256": receipt["training"]["sha256"],
            "config": receipt["training"]["config"],
            "policy_seed": receipt["training"]["policy_seed"],
            "seed_set": {
                "policy_seed": receipt["training"]["policy_seed"],
                "study_policy_seeds": receipt["selection"]["policy_seed_set"],
                "training_roster_and_tapes_sha256": canonical_hash(
                    receipt["training"]["training_roster_and_tapes"]
                ),
            },
            "registered_active_actor_critic_transitions": receipt["training"][
                "registered_active_actor_critic_transitions"
            ],
            "development_case_count": receipt["training"][
                "development_case_count"
            ],
            "training_roster_and_tapes": receipt["training"][
                "training_roster_and_tapes"
            ],
        },
        "selection": {
            "receipt_path": receipt["selection"]["path"],
            "receipt_sha256": receipt["selection"]["sha256"],
            "split": receipt["selection"]["split"],
            "development_case_count": receipt["selection"][
                "development_case_count"
            ],
            "ranking": receipt["selection"]["ranking"],
            "winner": receipt["selection"]["winner"],
            "runner_up": receipt["selection"]["runner_up"],
            "margin": receipt["selection"]["margin"],
            "tie_break": receipt["selection"]["tie_break"],
            "candidate_count": receipt["selection"]["candidate_count"],
            "policy_seed_set": receipt["selection"]["policy_seed_set"],
            "checkpoint_bundle": receipt["selection"]["checkpoint_bundle"],
        },
        "runtime_versions": receipt["runtime_versions"],
        "parity_receipt": {
            "path": _portable_path(receipt_path),
            "sha256": file_sha256(
                receipt_path,
                label="parity receipt",
                error_type=ExportError,
            ),
        },
        "development_parity": {
            "case_count": parity["case_count"],
            "action_tolerance": parity["action_tolerance"],
            "action_relative_tolerance": parity["action_relative_tolerance"],
            "maximum_action_absolute_error": parity[
                "maximum_action_absolute_error"
            ],
            "resilience_auc_tolerance": parity["resilience_auc_tolerance"],
            "maximum_resilience_auc_absolute_error": parity[
                "maximum_resilience_auc_absolute_error"
            ],
            "sb3_solved_count": parity["sb3_solved_count"],
            "onnx_solved_count": parity["onnx_solved_count"],
            "hard_violation_count": parity["onnx_hard_violation_count"],
            "maximum_conservation_residual": parity[
                "onnx_maximum_conservation_residual"
            ],
            "deterministic_replay_mismatch_count": parity[
                "deterministic_replay_mismatch_count"
            ],
            "rows_sha256": parity["rows_sha256"],
        },
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create one canonical pretty-printed JSON file."""

    path = path.resolve()
    if path.exists():
        raise ExportError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExportError("evidence payload is not canonical JSON data") from exc
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ExportError(f"refusing to overwrite evidence file: {path}")
        os.replace(temporary, path)
        fsync_parent(path)
    except ExportError:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ExportError(f"could not create evidence file: {path}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a selected PPO checkpoint and prove parity on the current "
            "development roster. This tool cannot access the final split."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--onnx-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--model-id", default="city-recovery-ppo")
    parser.add_argument("--training-receipt", required=True, type=Path)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    normalization_path = args.normalization.resolve()
    onnx_path = args.onnx_output.resolve()
    receipt_path = args.receipt_output.resolve()
    manifest_path = args.manifest_output.resolve()
    training_receipt_path = args.training_receipt.resolve()
    selection_receipt_path = args.selection_receipt.resolve()
    if len({onnx_path, receipt_path, manifest_path}) != 3:
        raise ExportError("ONNX, receipt, and manifest outputs must be distinct")

    normalization = load_observation_normalization(normalization_path)
    training = load_training_provenance(training_receipt_path)
    selection = load_selection_provenance(
        selection_receipt_path,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        normalization=normalization,
        training=training,
    )
    if (
        EXPECTED_DEVELOPMENT_CASES != CANONICAL_DEVELOPMENT_CASES
        or training["development_case_count"] != EXPECTED_DEVELOPMENT_CASES
        or selection["development_case_count"] != EXPECTED_DEVELOPMENT_CASES
    ):
        raise ExportError(
            "publication requires the canonical 200-case development contract"
        )
    model = load_sb3_checkpoint(checkpoint_path)
    interface = export_deterministic_actor(model, normalization, onnx_path)
    session = load_onnx_cpu_session(onnx_path)
    parity = development_parity(model, normalization, session)
    receipt = build_parity_receipt(
        model_id=args.model_id,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        normalization=normalization,
        onnx_path=onnx_path,
        interface=interface,
        parity=parity,
        training=training,
        selection=selection,
    )
    write_new_json(receipt_path, receipt)
    if not parity["passed"]:
        print(
            json.dumps(
                {
                    "status": "parity_failed",
                    "onnx_path": str(onnx_path),
                    "receipt_path": str(receipt_path),
                    "conditions": parity["conditions"],
                },
                sort_keys=True,
            )
        )
        return 2

    manifest = build_manifest(receipt_path=receipt_path, receipt=receipt)
    write_new_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "parity_passed",
                "onnx_path": str(onnx_path),
                "receipt_path": str(receipt_path),
                "manifest_path": str(manifest_path),
                "solved_count": parity["onnx_solved_count"],
                "maximum_action_absolute_error": parity[
                    "maximum_action_absolute_error"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as error:
        raise SystemExit(f"error: {error}") from error
