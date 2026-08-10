"""HTTP runtime for deterministic policy-versus-baseline comparisons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app.city.environment import (
    ACTION_GROUPS,
    ACTION_ORDER,
    ACTION_SIZE,
    ENGINE_ID,
    ENGINE_SPEC,
    ENGINE_SPEC_SHA256,
    ENGINE_VERSION,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    RAW_OBSERVATION_CONTRACT,
    RESULT_SCHEMA,
    compare,
    policy_identity,
)
from backend.app.city.outcome import SOLVED_DEFINITION, SOLVED_DEFINITION_SHA256
from backend.app.city.physics import SERVICES
from backend.app.models import CompareRequest
from backend.app.persistence import PersistenceError, RunStore
from backend.app.shared_evidence import canonical_bytes
from model.policy import Policy, PolicyError, load_policy

APP_VERSION = "2.0.0"
DEFAULT_SEED = 424242
POLICY_PATH_ENV = "INNOVERSE_POLICY_PATH"
POLICY_SHA256_ENV = "INNOVERSE_POLICY_SHA256"


class CanonicalJSONResponse(Response):
    """Serialize every API object with the evidence canonicalizer."""

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
    """Return the stable public error envelope."""

    return {"error": {"code": code, "message": message, "details": details or []}}


def dependency_error(exc: Exception) -> CanonicalJSONResponse:
    """Report a missing or invalid explicitly configured policy."""

    return CanonicalJSONResponse(
        status_code=503,
        content=error_payload("DEPENDENCY_NOT_READY", str(exc)),
    )


def persistence_error(exc: PersistenceError) -> CanonicalJSONResponse:
    """Map persistence failures to stable HTTP responses."""

    status = 404 if str(exc) == "persisted result was not found" else 500
    return CanonicalJSONResponse(
        status_code=status,
        content=error_payload("PERSISTENCE_FAILED", str(exc)),
    )


def configured_policy() -> Policy:
    """Load only the ONNX artifact explicitly selected by the operator."""

    path = os.environ.get(POLICY_PATH_ENV, "").strip()
    if not path:
        raise PolicyError(f"{POLICY_PATH_ENV} is required")
    expected_sha256 = os.environ.get(POLICY_SHA256_ENV)
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip() or None
    return load_policy(path, expected_sha256=expected_sha256)


def metadata_payload(policy: Policy) -> dict[str, Any]:
    """Describe the current runtime without legacy release-lineage fields."""

    model = policy_identity(policy)
    model.update(
        {
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "observation_order": list(OBSERVATION_ORDER),
            "action_order": list(ACTION_ORDER),
            "action_groups": list(ACTION_GROUPS),
        }
    )
    return {
        "app": "Autonomous City Recovery Planner",
        "version": APP_VERSION,
        "schema_version": RESULT_SCHEMA,
        "default_seed": DEFAULT_SEED,
        "services": list(SERVICES),
        "model": model,
        "environment": {
            "id": ENGINE_ID,
            "version": ENGINE_VERSION,
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "spec_sha256": ENGINE_SPEC_SHA256,
            "policy_neutral_transition": ENGINE_SPEC["policy_neutral_transition"],
            "future_tape_visible": ENGINE_SPEC["future_tape_visible"],
        },
        "outcome_definition": SOLVED_DEFINITION,
        "outcome_definition_sha256": SOLVED_DEFINITION_SHA256,
        "baseline": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "uses_same_observation_contract": True,
            "uses_same_action_contract": True,
            "uses_public_risk_signal": True,
            "future_tape_visible": False,
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
            "NumPy PCG64 tape generated once; ONNX Runtime CPU session uses "
            "sequential single-thread inference"
        ),
    }


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return field-local validation evidence before loading the policy."""

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
    """Confirm that the HTTP process is alive, independent of its policy."""

    return {"status": "live"}


@app.get("/health/ready", response_model=None)
def health_ready() -> Response | dict[str, Any]:
    """Confirm that the explicitly configured policy is ready for inference."""

    try:
        policy = configured_policy()
    except PolicyError as exc:
        return dependency_error(exc)
    identity = policy_identity(policy)
    return {
        "status": "ready",
        "engine_id": ENGINE_ID,
        "policy_id": identity["id"],
        "policy_path_stem": identity["path_stem"],
        "policy_sha256": identity["sha256"],
        "policy_type": identity["artifact_type"],
        "runtime": identity["runtime"],
        "observation_contract": dict(RAW_OBSERVATION_CONTRACT),
        "observation_count": OBSERVATION_SIZE,
        "action_count": ACTION_SIZE,
    }


@app.get("/api/v1/meta", response_model=None)
def metadata() -> Response | dict[str, Any]:
    """Return the lean current environment and policy contract."""

    try:
        return metadata_payload(configured_policy())
    except PolicyError as exc:
        return dependency_error(exc)


@app.get("/api/v1/simulations", response_model=None)
def list_simulations(engine_version: str | None = None) -> Response | dict[str, Any]:
    """List locally persisted comparison summaries."""

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
    """Load one canonical comparison by its content identity."""

    try:
        return RunStore().load(result_id)
    except PersistenceError as exc:
        return persistence_error(exc)


@app.post("/api/v1/simulations/compare", response_model=None)
def compare_simulations(payload: CompareRequest) -> Response | dict[str, Any]:
    """Run and persist one shared-tape policy-versus-baseline comparison."""

    try:
        policy = configured_policy()
    except PolicyError as exc:
        return dependency_error(exc)
    try:
        result = compare(payload.scenario, payload.seed, policy)
        return RunStore().save(result)
    except PolicyError as exc:
        return dependency_error(exc)
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
