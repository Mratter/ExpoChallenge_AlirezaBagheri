"""Minimal presentation runtime for the model-focused workbench.

This module intentionally does not import the legacy simulator, v4/v5 research,
or training stacks. The presentation profile needs only the sealed evidence
loader, FastAPI, static frontend files, and ONNX Runtime during preflight.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.workbench_service import WorkbenchEvidenceError, load_workbench_overview
from backend.app.workbench_toolbox import ToolboxEvaluationRequest, evaluate_toolbox

APP_VERSION: Final = "1.0.0"
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
FRONTEND_DIST: Final = REPOSITORY_ROOT / "frontend" / "dist"
NO_STORE_HEADERS: Final = {"Cache-Control": "no-store"}


class CanonicalJSONResponse(Response):
    """Serialize JSON deterministically for evidence-backed API responses."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def error_payload(
    code: str,
    message: str,
    *,
    details: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {"error": {"code": code, "details": details or [], "message": message}}


def _not_found(path: str) -> CanonicalJSONResponse:
    return CanonicalJSONResponse(
        status_code=404,
        content=error_payload("NOT_FOUND", f"No workbench resource exists at /{path}"),
        headers=NO_STORE_HEADERS,
    )


def _frontend_index_is_safe(index_path: Path) -> bool:
    return index_path.is_file() and not index_path.is_symlink()


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    """Create the isolated workbench API and static presentation app."""

    dist = FRONTEND_DIST if frontend_dist is None else frontend_dist.resolve()
    index_path = dist / "index.html"
    assets_path = dist / "assets"
    workbench_app = FastAPI(
        title="City Recovery Model Workbench",
        version=APP_VERSION,
        default_response_class=CanonicalJSONResponse,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @workbench_app.exception_handler(StarletteHTTPException)
    async def canonical_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> CanonicalJSONResponse:
        if exc.status_code == 404:
            return _not_found(request.url.path.lstrip("/"))
        headers = dict(exc.headers or {})
        headers.update(NO_STORE_HEADERS)
        return CanonicalJSONResponse(
            status_code=exc.status_code,
            content=error_payload("HTTP_ERROR", str(exc.detail)),
            headers=headers,
        )

    @workbench_app.exception_handler(RequestValidationError)
    async def canonical_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> CanonicalJSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": str(error["msg"]),
                "type": str(error["type"]),
            }
            for error in exc.errors()
        ]
        return CanonicalJSONResponse(
            status_code=422,
            content=error_payload(
                "INVALID_TOOLBOX_INPUT",
                "Toolbox input validation failed.",
                details=details,
            ),
            headers=NO_STORE_HEADERS,
        )

    @workbench_app.get("/health/live", response_model=None)
    def health_live() -> CanonicalJSONResponse:
        return CanonicalJSONResponse(
            content={"runtime": "model-workbench", "status": "live"},
            headers=NO_STORE_HEADERS,
        )

    @workbench_app.get("/health/ready", response_model=None)
    def health_ready() -> CanonicalJSONResponse:
        try:
            overview = load_workbench_overview()
        except WorkbenchEvidenceError as exc:
            return CanonicalJSONResponse(
                status_code=503,
                content=error_payload("WORKBENCH_EVIDENCE_NOT_READY", str(exc)),
                headers=NO_STORE_HEADERS,
            )
        return CanonicalJSONResponse(
            content={
                "benchmark_id": str(overview["benchmark"]["benchmark_id"]),
                "runtime": "model-workbench",
                "status": "ready",
            },
            headers=NO_STORE_HEADERS,
        )

    @workbench_app.get("/api/workbench/v1/overview", response_model=None)
    def get_workbench_overview() -> CanonicalJSONResponse:
        try:
            overview = load_workbench_overview()
        except WorkbenchEvidenceError as exc:
            return CanonicalJSONResponse(
                status_code=503,
                content=error_payload("WORKBENCH_EVIDENCE_NOT_READY", str(exc)),
                headers=NO_STORE_HEADERS,
            )
        return CanonicalJSONResponse(content=overview, headers=NO_STORE_HEADERS)

    @workbench_app.post("/api/workbench/v1/toolbox/evaluate", response_model=None)
    def evaluate_workbench_toolbox(
        request: ToolboxEvaluationRequest,
    ) -> CanonicalJSONResponse:
        try:
            result = evaluate_toolbox(request)
        except WorkbenchEvidenceError as exc:
            return CanonicalJSONResponse(
                status_code=503,
                content=error_payload("TOOLBOX_RUNTIME_NOT_READY", str(exc)),
                headers=NO_STORE_HEADERS,
            )
        return CanonicalJSONResponse(content=result, headers=NO_STORE_HEADERS)

    if assets_path.is_dir():
        workbench_app.mount(
            "/assets",
            StaticFiles(directory=assets_path),
            name="workbench-assets",
        )

    @workbench_app.get("/", response_model=None, include_in_schema=False)
    def frontend_index() -> Response:
        if not _frontend_index_is_safe(index_path):
            return CanonicalJSONResponse(
                status_code=503,
                content=error_payload(
                    "FRONTEND_NOT_READY",
                    "The workbench frontend build is missing; run scripts/setup.ps1.",
                ),
                headers=NO_STORE_HEADERS,
            )
        return FileResponse(index_path, headers=NO_STORE_HEADERS)

    @workbench_app.get("/{full_path:path}", response_model=None, include_in_schema=False)
    def frontend_fallback(full_path: str) -> Response:
        path = PurePosixPath(full_path)
        first_segment = path.parts[0] if path.parts else ""
        if first_segment in {"api", "assets", "health"} or path.suffix:
            return _not_found(full_path)
        if not _frontend_index_is_safe(index_path):
            return CanonicalJSONResponse(
                status_code=503,
                content=error_payload(
                    "FRONTEND_NOT_READY",
                    "The workbench frontend build is missing; run scripts/setup.ps1.",
                ),
                headers=NO_STORE_HEADERS,
            )
        return FileResponse(index_path, headers=NO_STORE_HEADERS)

    return workbench_app


app = create_app()


__all__ = [
    "APP_VERSION",
    "CanonicalJSONResponse",
    "FRONTEND_DIST",
    "NO_STORE_HEADERS",
    "app",
    "create_app",
    "error_payload",
]
