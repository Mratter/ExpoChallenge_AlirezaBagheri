from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.app.artifact import (
    ARTIFACT_LICENSE,
    MANIFEST_SCHEMA_VERSION,
    POLICY_FEATURE_ORDER,
    POLICY_SCHEMA_VERSION,
    ArtifactError,
    PolicyBundle,
    load_policy,
)
from backend.app.models import CompareRequest
from backend.app.persistence import PersistenceError, RunStore
from backend.app.simulator import SERVICES, compare

APP_VERSION = "0.3.0"
API_SCHEMA_VERSION = "2.2.0"
DATASET_SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "2.0.0"
DEFAULT_SEED = 20260714


class CanonicalJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


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


def error_payload(code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or []}}


def dependency_error(exc: ArtifactError) -> CanonicalJSONResponse:
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


def metadata_payload(bundle: PolicyBundle) -> dict[str, Any]:
    metadata = bundle.metadata
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
            "schema_version": POLICY_SCHEMA_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": metadata["artifact_type"],
            "algorithm": metadata["training"]["algorithm"],
            "training_library": metadata["training"]["library"],
            "training_library_version": metadata["training"]["library_version"],
            "observation_order": list(POLICY_FEATURE_ORDER),
            "action_order": metadata["action_order"],
            "license": ARTIFACT_LICENSE,
            "source": "scripts/train_policy.py",
            "onnx_sha256": bundle.onnx_sha256,
            "sb3_checkpoint_sha256": bundle.sb3_sha256,
            "metadata_sha256": bundle.metadata_sha256,
            "parity_report_sha256": bundle.parity_sha256,
            "legacy_candidate": metadata["legacy_candidate"],
        },
        "baseline": {
            "id": "ortools-glop-visible-v1",
            "library": "OR-Tools",
            "solver": "GLOP",
            "future_shocks_visible": False,
        },
        "dataset": {
            "id": "synthetic-city-dynamics-v2",
            "version": DATASET_VERSION,
            "schema_version": DATASET_SCHEMA_VERSION,
            "license": "CC0-1.0",
            "source": "backend/app/simulator.py and backend/app/scenarios.py",
            "service_order": list(SERVICES),
            "empirical": False,
        },
        "persistence": {
            "format": "canonical-json-v1",
            "identity": "sha256(schema, seed, scenario, policy, baseline)",
            "idempotent": True,
        },
        "determinism": (
            "NumPy PCG64 shock tape generated once; ONNX Runtime CPU session uses "
            "sequential single-thread inference"
        ),
    }


@app.middleware("http")
async def require_policy_dependency(request: Request, call_next: Any) -> Response:
    if request.url.path == "/health/live":
        return await call_next(request)
    try:
        request.state.policy_bundle = load_policy()
    except ArtifactError as exc:
        return dependency_error(exc)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"path": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
        for item in exc.errors()
    ]
    return CanonicalJSONResponse(
        status_code=422,
        content=error_payload("INVALID_SCENARIO", "Scenario validation failed.", details),
    )


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready", response_model=None)
def health_ready(request: Request) -> dict[str, str]:
    bundle: PolicyBundle = request.state.policy_bundle
    return {
        "policy_sha256": bundle.onnx_sha256,
        "policy_type": bundle.metadata["artifact_type"],
        "status": "ready",
    }


@app.get("/api/v1/meta", response_model=None)
def metadata(request: Request) -> dict[str, Any]:
    return metadata_payload(request.state.policy_bundle)


@app.get("/api/v1/simulations", response_model=None)
def list_simulations() -> Response | dict[str, Any]:
    try:
        results = RunStore().list_summaries()
        return {"schema_version": "1.0.0", "count": len(results), "results": results}
    except PersistenceError as exc:
        return persistence_error(exc)


@app.get("/api/v1/simulations/{result_id}", response_model=None)
def get_simulation(result_id: str) -> Response | dict[str, Any]:
    try:
        return RunStore().load(result_id)
    except PersistenceError as exc:
        return persistence_error(exc)


@app.post("/api/v1/simulations/compare", response_model=None)
def compare_simulations(request: Request, payload: CompareRequest) -> Response | dict[str, Any]:
    try:
        result = compare(payload.scenario, payload.seed, request.state.policy_bundle)
        return RunStore().save(result)
    except PersistenceError as exc:
        return persistence_error(exc)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return CanonicalJSONResponse(
            status_code=500,
            content=error_payload("COMPUTATION_FAILED", str(exc)),
        )
