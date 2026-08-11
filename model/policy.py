"""Load and run the bundled or explicitly selected City Recovery ONNX policy."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

OBSERVATION_COUNT = 73
ACTION_COUNT = 22
ACTION_BOUND = 1.0
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "city_recovery_ppo.v4.onnx"
)

_INPUT_NAME = "observation"
_OUTPUT_NAME = "action"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class PolicyError(RuntimeError):
    """Raised when a policy artifact or inference result violates its contract."""


@dataclass(frozen=True, slots=True)
class Policy:
    """An immutable identity and CPU inference session for one ONNX artifact."""

    path: Path
    sha256: str
    session: ort.InferenceSession = field(repr=False, compare=False)

    def predict(self, observation: Any) -> np.ndarray:
        """Return one action for one raw, finite 73-value observation."""

        try:
            raw_observation = np.asarray(observation)
        except Exception as exc:
            raise PolicyError("observation must be a numeric array") from exc
        if raw_observation.shape != (OBSERVATION_COUNT,):
            raise PolicyError(
                f"observation must have shape ({OBSERVATION_COUNT},), "
                f"got {raw_observation.shape}"
            )
        try:
            if not np.all(np.isfinite(raw_observation)):
                raise PolicyError("observation must contain only finite values")
            with np.errstate(over="ignore", invalid="ignore"):
                model_observation = raw_observation.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise PolicyError("observation must be a numeric array") from exc
        if not np.all(np.isfinite(model_observation)):
            raise PolicyError("observation cannot be represented as finite float32 values")

        try:
            result = self.session.run(
                [_OUTPUT_NAME],
                {_INPUT_NAME: model_observation.reshape(1, OBSERVATION_COUNT)},
            )[0]
        except Exception as exc:
            raise PolicyError("ONNX policy inference failed") from exc
        action = _validated_action(result, context="policy inference")
        return action.reshape(ACTION_COUNT).astype(np.float64, copy=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_policy(path: str | Path) -> tuple[Path, bytes, str]:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("path is not a file")
        payload = resolved.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise PolicyError(f"policy artifact is missing or unreadable: {candidate}") from exc
    return resolved, payload, _sha256(payload)


def _validated_action(result: Any, *, context: str) -> np.ndarray:
    action = np.asarray(result)
    if action.shape != (1, ACTION_COUNT):
        raise PolicyError(
            f"{context} must return shape (1, {ACTION_COUNT}), got {action.shape}"
        )
    try:
        finite = bool(np.all(np.isfinite(action)))
        bounded = bool(np.all(np.abs(action) <= ACTION_BOUND))
    except TypeError as exc:
        raise PolicyError(f"{context} returned a non-numeric action") from exc
    if not finite:
        raise PolicyError(f"{context} returned a non-finite action")
    if not bounded:
        raise PolicyError(
            f"{context} returned an action outside [-{ACTION_BOUND}, {ACTION_BOUND}]"
        )
    return action


def _create_session(payload: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise PolicyError("policy artifact is not a loadable ONNX model") from exc

    if session.get_providers() != ["CPUExecutionProvider"]:
        raise PolicyError("policy session must use only CPUExecutionProvider")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != _INPUT_NAME
        or inputs[0].type != "tensor(float)"
        or list(inputs[0].shape) != ["batch", OBSERVATION_COUNT]
    ):
        raise PolicyError(
            "policy input must be observation: tensor(float)[batch, 73]"
        )
    if (
        len(outputs) != 1
        or outputs[0].name != _OUTPUT_NAME
        or outputs[0].type != "tensor(float)"
        or list(outputs[0].shape) != ["batch", ACTION_COUNT]
    ):
        raise PolicyError("policy output must be action: tensor(float)[batch, 22]")

    try:
        smoke_result = session.run(
            [_OUTPUT_NAME],
            {
                _INPUT_NAME: np.zeros(
                    (1, OBSERVATION_COUNT),
                    dtype=np.float32,
                )
            },
        )[0]
    except Exception as exc:
        raise PolicyError("policy failed its zero-observation smoke inference") from exc
    _validated_action(smoke_result, context="zero-observation smoke inference")
    return session


@lru_cache(maxsize=16)
def _load_cached(resolved_path: str, artifact_sha256: str) -> Policy:
    path = Path(resolved_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PolicyError(f"policy artifact is missing or unreadable: {path}") from exc
    if _sha256(payload) != artifact_sha256:
        raise PolicyError("policy artifact changed while it was being loaded")
    session = _create_session(payload)
    return Policy(path=path, sha256=artifact_sha256, session=session)


def load_policy(
    path: str | Path,
    expected_sha256: str | None = None,
) -> Policy:
    """Load an explicit ONNX path, optionally requiring an expected SHA-256."""

    resolved, _payload, artifact_sha256 = _read_policy(path)
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise PolicyError("expected_sha256 must be a 64-character SHA-256 digest")
        if artifact_sha256 != expected_sha256.lower():
            raise PolicyError("policy artifact SHA-256 does not match expected_sha256")
    return _load_cached(str(resolved), artifact_sha256)


__all__ = [
    "ACTION_COUNT",
    "DEFAULT_POLICY_PATH",
    "OBSERVATION_COUNT",
    "Policy",
    "PolicyError",
    "load_policy",
]
