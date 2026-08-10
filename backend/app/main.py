from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app.models import CompareRequestV3
from backend.app.persistence import PersistenceError, RunStore
from backend.app.shared_evidence import canonical_bytes
from backend.app.simulator_core import SERVICES
from backend.app.simulator_v3 import (
    ACTION_GROUPS_V3,
    ACTION_ORDER_V3,
    ENGINE_V3_ID,
    ENGINE_V3_SPEC,
    ENGINE_V3_SPEC_SHA256,
    ENGINE_V3_VERSION,
    OBSERVATION_ORDER_V3,
    RESULT_SCHEMA_V3,
    SOLVED_DEFINITION_V3,
    SOLVED_DEFINITION_V3_SHA256,
    compare_v3,
)
from model.ppo_v3 import (
    ARTIFACT_LICENSE as V3_ARTIFACT_LICENSE,
    POLICY_SCHEMA_VERSION as V3_POLICY_SCHEMA_VERSION,
    ArtifactError as V3ArtifactError,
    PolicyBundle as V3PolicyBundle,
    load_policy_v3,
)

APP_VERSION = "1.1.0"
API_SCHEMA_VERSION = RESULT_SCHEMA_V3
DATASET_SCHEMA_VERSION = "4.0.0"
DATASET_VERSION = "3.0.0"
DEFAULT_SEED = 424242


class CanonicalJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return canonical_bytes(content)


app = FastAPI(
    title="City Recovery Planner API",
    version=APP_VERSION,
    default_response_class=CanonicalJSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def error_payload(
    code: str, message: str, details: Any | None = None
) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or []}}


def dependency_error(exc: Exception) -> CanonicalJSONResponse:
    return CanonicalJSONResponse(
        status_code=503,
        content=error_payload("DEPENDENCY_NOT_READY", str(exc)),
    )


def persistence_error(exc: PersistenceError) -> CanonicalJSONResponse:
    status = 404 if str(exc) == "persisted result was not found" else 500
    return CanonicalJSONResponse(
        status_code=status,
        content=error_payload("PERSISTENCE_FAILED", str(exc)),
    )


def _benchmark_payload(bundle: V3PolicyBundle) -> dict[str, Any]:
    report = bundle.benchmark
    return {
        "artifact_sha256": bundle.benchmark_sha256,
        "artifact_type": report["artifact_type"],
        "status": report["status"],
        "primary_metric": report["primary_metric"],
        "synthetic_case_count": report["synthetic_case_count"],
        "candidate": report["candidate"],
        "baseline": report["baseline"],
        "paired_absolute_outcomes": report["paired_absolute_outcomes"],
        "secondary_head_to_head_resilience_auc": report[
            "secondary_head_to_head_resilience_auc"
        ],
        "invariants": report["invariants"],
        "rows_sha256": report["rows_sha256"],
        "ledger_sha256": report["ledger_sha256"],
    }


def metadata_payload_v3(bundle: V3PolicyBundle) -> dict[str, Any]:
    metadata = bundle.metadata
    training = metadata["training"]
    architecture = metadata["architecture"]
    return {
        "app": "Autonomous City Recovery Planner",
        "version": APP_VERSION,
        "schema_version": API_SCHEMA_VERSION,
        "commit": os.environ.get("INNOVERSE_COMMIT", "development"),
        "profile": os.environ.get("INNOVERSE_PROFILE", "cpu"),
        "default_seed": DEFAULT_SEED,
        "services": list(SERVICES),
        "model": {
            "id": metadata["id"],
            "version": metadata["version"],
            "schema_version": V3_POLICY_SCHEMA_VERSION,
            "manifest_schema_version": bundle.manifest_schema_version,
            "artifact_type": metadata["artifact_type"],
            "algorithm": training["algorithm"],
            "training_library": training["library"],
            "trainable_parameters": architecture["trainable_parameters"],
            "requested_training_transitions": training["requested_transitions"],
            "completed_training_transitions": training["completed_transitions"],
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "observation_order": list(OBSERVATION_ORDER_V3),
            "action_order": list(ACTION_ORDER_V3),
            "action_groups": list(ACTION_GROUPS_V3),
            "license": V3_ARTIFACT_LICENSE,
            "source": "model/ppo_v3.py",
            "onnx_sha256": bundle.onnx_sha256,
            "selected_checkpoint_sha256": bundle.sb3_sha256,
            "manifest_sha256": bundle.manifest_sha256,
            "parity_sha256": bundle.parity_sha256,
            "scientific_source_sha256": bundle.scientific_source_sha256,
        },
        "environment": {
            "id": ENGINE_V3_ID,
            "version": ENGINE_V3_VERSION,
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "spec_sha256": ENGINE_V3_SPEC_SHA256,
            "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
            "policy_neutral_transition": ENGINE_V3_SPEC["policy_neutral_transition"],
            "future_tape_visible": ENGINE_V3_SPEC["future_tape_visible"],
        },
        "outcome_definition": SOLVED_DEFINITION_V3,
        "baseline": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "uses_same_observation_contract": True,
            "uses_same_action_contract": True,
            "uses_public_risk_signal": True,
            "future_tape_visible": False,
        },
        "benchmark": _benchmark_payload(bundle),
        "dataset": {
            "id": "synthetic-city-dynamics-v3",
            "version": DATASET_VERSION,
            "schema_version": DATASET_SCHEMA_VERSION,
            "license": "CC0-1.0",
            "source": "backend/app/simulator_v3.py and backend/app/scenarios_v3.py",
            "service_order": list(SERVICES),
            "empirical": False,
        },
        "persistence": {
            "format": "canonical-json-v1",
            "identity": (
                "sha256(schema, engine, engine-spec, outcome-definition, seed, "
                "scenario, policy, baseline-id, baseline-version)"
            ),
            "idempotent": True,
        },
        "determinism": (
            "NumPy PCG64 tape generated once; selected ONNX Runtime CPU session "
            "uses sequential single-thread inference"
        ),
    }


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"path": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
        for item in exc.errors()
    ]
    return CanonicalJSONResponse(
        status_code=422,
        content=error_payload(
            "INVALID_SCENARIO", "Scenario validation failed.", details
        ),
    )


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready", response_model=None)
def health_ready() -> Response | dict[str, Any]:
    try:
        bundle = load_policy_v3()
    except V3ArtifactError as exc:
        return dependency_error(exc)
    return {
        "status": "ready",
        "engine_id": ENGINE_V3_ID,
        "policy_id": bundle.metadata["id"],
        "policy_sha256": bundle.onnx_sha256,
        "policy_type": bundle.metadata["artifact_type"],
        "observation_count": len(OBSERVATION_ORDER_V3),
        "action_count": len(ACTION_ORDER_V3),
        "benchmark_sha256": bundle.benchmark_sha256,
    }


@app.get("/api/v1/meta", response_model=None)
def metadata_v3() -> Response | dict[str, Any]:
    try:
        return metadata_payload_v3(load_policy_v3())
    except V3ArtifactError as exc:
        return dependency_error(exc)


@app.get("/api/v1/simulations", response_model=None)
def list_simulations(engine_version: str | None = None) -> Response | dict[str, Any]:
    try:
        results = RunStore().list_summaries(engine_version=engine_version)
        return {
            "schema_version": "2.0.0",
            "engine_version_filter": engine_version,
            "count": len(results),
            "results": results,
        }
    except PersistenceError as exc:
        return persistence_error(exc)


@app.get("/api/v1/simulations/{result_id}", response_model=None)
def get_simulation(result_id: str) -> Response | dict[str, Any]:
    try:
        return RunStore().load(result_id)
    except PersistenceError as exc:
        return persistence_error(exc)


@app.post("/api/v1/simulations/compare", response_model=None)
def compare_simulations(payload: CompareRequestV3) -> Response | dict[str, Any]:
    try:
        bundle = load_policy_v3()
    except V3ArtifactError as exc:
        return dependency_error(exc)
    try:
        result = compare_v3(payload.scenario, payload.seed, bundle)
        return RunStore().save(result)
    except PersistenceError as exc:
        return persistence_error(exc)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return CanonicalJSONResponse(
            status_code=500,
            content=error_payload("COMPUTATION_FAILED", str(exc)),
        )


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
elif os.environ.get("INNOVERSE_RUNTIME") == "1":
    raise RuntimeError("frontend build is missing; run scripts/setup.ps1")
