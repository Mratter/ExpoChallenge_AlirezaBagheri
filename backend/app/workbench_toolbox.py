"""Real, CPU-only inference for the active synthetic showcase model.

The toolbox consumes only the 21 public features used during training.  It
does not import Torch, the simulator, hidden targets, scenario seeds, or any
training code.  Input normalization is read from the sealed ONNX graph so the
interactive path cannot silently drift from the exported model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Final

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.capi.onnxruntime_pybind11_state import (
    EngineError,
    EPFail,
    Fail,
    InvalidArgument,
    InvalidGraph,
    InvalidProtobuf,
    ModelLoadCanceled,
    ModelLoaded,
    ModelRequiresCompilation,
    NoModel,
    NoSuchFile,
    NotFound,
    NotImplemented,
    RuntimeException,
)
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from backend.app import workbench_service
from backend.app.workbench_service import WorkbenchEvidenceError, load_workbench_overview

TOOLBOX_SCHEMA_VERSION: Final = "model-toolbox-evaluation-v1"
EXPECTED_ONNX_SHA256: Final = "b3edf8007feb749ddc33fc3ebbb008a02ef98d561bd74cfde286dde030a4dae0"
MODEL_ID: Final = "adaptive-cascade-mlp-v2-300k"
MODEL_PARAMETER_COUNT: Final = 300_113
HEURISTIC_ID: Final = "static-visible-need-heuristic-v1"
SERVICE_IDS: Final = (
    "transport",
    "housing",
    "food",
    "healthcare",
    "public_services",
)
FEATURE_ORDER: Final = (
    *(f"public_forecast_signal_{service}" for service in SERVICE_IDS),
    *(f"visible_service_need_{service}" for service in SERVICE_IDS),
    *(f"public_regime_{index}" for index in range(4)),
    *(f"current_service_health_{service}" for service in SERVICE_IDS),
    "phase_sin",
    "phase_cos",
)
ORT_ERRORS: Final = (
    EPFail,
    EngineError,
    Fail,
    InvalidArgument,
    InvalidGraph,
    InvalidProtobuf,
    ModelLoadCanceled,
    ModelLoaded,
    ModelRequiresCompilation,
    NoModel,
    NoSuchFile,
    NotFound,
    NotImplemented,
    RuntimeException,
)

ForecastValue = Annotated[StrictFloat, Field(ge=-8.0, le=8.0, allow_inf_nan=False)]
NeedValue = Annotated[StrictFloat, Field(ge=0.0, le=1.5, allow_inf_nan=False)]
HealthValue = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ToolboxEvaluationRequest(BaseModel):
    """Structured public observation that expands to the model's 21 inputs."""

    model_config = ConfigDict(extra="forbid")

    public_forecast_signal: list[ForecastValue] = Field(min_length=5, max_length=5)
    visible_service_need: list[NeedValue] = Field(min_length=5, max_length=5)
    public_regime: StrictInt = Field(ge=0, le=3)
    current_service_health: list[HealthValue] = Field(min_length=5, max_length=5)
    phase_window: StrictInt = Field(ge=1, le=12)


@dataclass(frozen=True)
class ToolboxRuntime:
    session: ort.InferenceSession
    input_mean: np.ndarray
    input_scale: np.ndarray
    onnx_sha256: str


def _raw_observation(request: ToolboxEvaluationRequest) -> np.ndarray:
    phase = 2.0 * np.pi * (request.phase_window - 1) / 12.0
    regime = np.eye(4, dtype=np.float32)[request.public_regime]
    observation = np.concatenate(
        (
            np.asarray(request.public_forecast_signal, dtype=np.float32),
            np.asarray(request.visible_service_need, dtype=np.float32),
            regime,
            np.asarray(request.current_service_health, dtype=np.float32),
            np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32),
        )
    )
    if observation.shape != (21,) or not np.isfinite(observation).all():
        raise WorkbenchEvidenceError("toolbox observation expansion is invalid")
    return observation


def _initializer_vector(model: onnx.ModelProto, name: str) -> np.ndarray:
    matches = [initializer for initializer in model.graph.initializer if initializer.name == name]
    if len(matches) != 1:
        raise WorkbenchEvidenceError(f"showcase ONNX {name} initializer differs")
    value = np.asarray(numpy_helper.to_array(matches[0]), dtype=np.float32)
    if value.shape != (21,) or not np.isfinite(value).all():
        raise WorkbenchEvidenceError(f"showcase ONNX {name} initializer is invalid")
    return value


@lru_cache(maxsize=2)
def _load_runtime(onnx_sha256: str, payload: bytes) -> ToolboxRuntime:
    if hashlib.sha256(payload).hexdigest() != onnx_sha256:
        raise WorkbenchEvidenceError("showcase ONNX model digest differs before inference")
    try:
        model = onnx.load_model_from_string(payload)
        onnx.checker.check_model(model)
        input_mean = _initializer_vector(model, "input_mean")
        input_scale = _initializer_vector(model, "input_scale")
        if np.any(input_scale <= 0.0):
            raise WorkbenchEvidenceError("showcase ONNX input_scale must be positive")
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    except WorkbenchEvidenceError:
        raise
    except (OSError, ValueError, onnx.checker.ValidationError, *ORT_ERRORS) as exc:
        raise WorkbenchEvidenceError("showcase ONNX runtime could not be initialized") from exc

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "observation"
        or inputs[0].shape[-1] != 21
        or len(outputs) != 1
        or outputs[0].name != "action_logits"
        or outputs[0].shape[-1] != 5
        or session.get_providers() != ["CPUExecutionProvider"]
    ):
        raise WorkbenchEvidenceError("showcase ONNX inference contract differs")
    return ToolboxRuntime(
        session=session,
        input_mean=input_mean,
        input_scale=input_scale,
        onnx_sha256=onnx_sha256,
    )


def _verified_runtime() -> tuple[dict[str, Any], ToolboxRuntime]:
    overview = load_workbench_overview()
    onnx_path = workbench_service.ACTIVE_SHOWCASE_SPEC.onnx_path
    try:
        payload = onnx_path.read_bytes()
    except OSError as exc:
        raise WorkbenchEvidenceError("showcase ONNX model is unreadable") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_ONNX_SHA256:
        raise WorkbenchEvidenceError("showcase ONNX model digest differs")
    return overview, _load_runtime(digest, payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    exponentials = np.exp(shifted)
    probabilities = exponentials / float(np.sum(exponentials))
    if probabilities.shape != (5,) or not np.isfinite(probabilities).all():
        raise WorkbenchEvidenceError("showcase ONNX probabilities are invalid")
    return probabilities


def evaluate_toolbox(request: ToolboxEvaluationRequest) -> dict[str, Any]:
    """Run the actual sealed model and the fixed public heuristic once."""

    overview, runtime = _verified_runtime()
    observation = _raw_observation(request)
    try:
        output = runtime.session.run(
            ["action_logits"],
            {"observation": observation[None, :]},
        )[0]
    except ORT_ERRORS as exc:
        raise WorkbenchEvidenceError("showcase ONNX inference failed") from exc
    logits = np.asarray(output, dtype=np.float32)
    if logits.shape != (1, 5) or not np.isfinite(logits).all():
        raise WorkbenchEvidenceError("showcase ONNX inference output is invalid")

    logits_vector = logits[0]
    probabilities = _softmax(logits_vector)
    model_action = int(np.argmax(logits_vector))
    heuristic_scores = observation[5:10]
    heuristic_action = int(np.argmax(heuristic_scores))
    ordered_probabilities = np.sort(probabilities)
    benchmark = overview["benchmark"]
    objective = benchmark["objective"]
    head_to_head = benchmark["head_to_head"]

    return {
        "schema_version": TOOLBOX_SCHEMA_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "synthetic_disclosure": benchmark["synthetic_disclosure"],
        "input": {
            "structured": request.model_dump(mode="json"),
            "feature_order": list(FEATURE_ORDER),
            "vector": [float(value) for value in observation],
            "normalization": {
                "method": "embedded_z_score",
                "input_vector_is_raw": True,
                "normalization_executed_inside_model": True,
                "mean": [float(value) for value in runtime.input_mean],
                "scale": [float(value) for value in runtime.input_scale],
            },
        },
        "model": {
            "id": MODEL_ID,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "action_index": model_action,
            "action_label": SERVICE_IDS[model_action],
            "logits": [float(value) for value in logits_vector],
            "probabilities": [float(value) for value in probabilities],
            "confidence": float(probabilities[model_action]),
            "probability_margin": float(ordered_probabilities[-1] - ordered_probabilities[-2]),
        },
        "heuristic": {
            "id": HEURISTIC_ID,
            "rule": "argmax_visible_service_need",
            "action_index": heuristic_action,
            "action_label": SERVICE_IDS[heuristic_action],
            "scores": [float(value) for value in heuristic_scores],
        },
        "comparison": {"same_action": model_action == heuristic_action},
        "benchmark_summary": {
            "scenario_total": benchmark["scenario_total"],
            "objective_passes": {
                "model": objective["learned_policy"]["passes"],
                "heuristic": objective["static_heuristic"]["passes"],
            },
            "head_to_head": {
                "model_wins": head_to_head["learned_wins"],
                "heuristic_wins": head_to_head["heuristic_wins"],
                "ties": head_to_head["ties"],
            },
        },
        "runtime": {
            "engine": "onnxruntime",
            "execution_provider": "CPUExecutionProvider",
            "real_model_inference": True,
            "onnx_sha256": runtime.onnx_sha256,
        },
    }


__all__ = [
    "FEATURE_ORDER",
    "HEURISTIC_ID",
    "MODEL_ID",
    "MODEL_PARAMETER_COUNT",
    "SERVICE_IDS",
    "TOOLBOX_SCHEMA_VERSION",
    "ToolboxEvaluationRequest",
    "evaluate_toolbox",
]
