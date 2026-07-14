from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app.artifact import (
    ARTIFACT_LICENSE,
    ARTIFACT_SOURCE,
    MANIFEST_SCHEMA_VERSION,
    POLICY_FEATURE_ORDER,
    POLICY_SCHEMA_VERSION,
    ArtifactError,
    load_policy,
)
from backend.app.models import CompareRequest
from backend.app.simulator import SERVICES, compare

APP_VERSION = "0.2.0"
API_SCHEMA_VERSION = "1.0.0"
DATASET_SCHEMA_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"
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


def metadata_payload(policy: dict[str, Any], checksum: str) -> dict[str, Any]:
    return {
        "app": "Autonomous City Recovery Planner",
        "version": APP_VERSION,
        "schema_version": API_SCHEMA_VERSION,
        "commit": os.environ.get("INNOVERSE_COMMIT", "development"),
        "profile": os.environ.get("INNOVERSE_PROFILE", "cpu"),
        "default_seed": DEFAULT_SEED,
        "services": list(SERVICES),
        "model": {
            "id": policy["id"],
            "version": policy["version"],
            "schema_version": POLICY_SCHEMA_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": policy["artifact_type"],
            "feature_order": list(POLICY_FEATURE_ORDER),
            "license": ARTIFACT_LICENSE,
            "source": ARTIFACT_SOURCE,
            "sha256": checksum,
        },
        "dataset": {
            "id": "synthetic-city-dynamics-v1",
            "version": DATASET_VERSION,
            "schema_version": DATASET_SCHEMA_VERSION,
            "license": "CC0-1.0",
            "source": "backend/app/simulator.py",
            "service_order": list(SERVICES),
            "empirical": False,
        },
        "determinism": "numpy.PCG64 shock tape generated once per comparison",
    }


@app.middleware("http")
async def require_policy_dependency(request: Request, call_next: Any) -> Response:
    if request.url.path == "/health/live":
        return await call_next(request)
    try:
        load_policy()
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
def health_ready() -> Response | dict[str, str]:
    try:
        _, checksum = load_policy()
    except ArtifactError as exc:
        return dependency_error(exc)
    return {"policy_sha256": checksum, "status": "ready"}


@app.get("/api/v1/meta", response_model=None)
def metadata() -> Response | dict[str, Any]:
    try:
        policy, checksum = load_policy()
    except ArtifactError as exc:
        return dependency_error(exc)
    return metadata_payload(policy, checksum)


@app.post("/api/v1/simulations/compare", response_model=None)
def compare_simulations(request: CompareRequest) -> Response | dict[str, Any]:
    try:
        policy, checksum = load_policy()
        return compare(request.scenario, request.seed, policy, checksum)
    except ArtifactError as exc:
        return dependency_error(exc)
    except (KeyError, TypeError, ValueError) as exc:
        return CanonicalJSONResponse(
            status_code=500,
            content=error_payload("COMPUTATION_FAILED", str(exc)),
        )


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
elif os.environ.get("INNOVERSE_RUNTIME") == "1":
    raise RuntimeError("frontend build is missing; run scripts/setup.ps1")
