"""Minimal launch preflight for the presentation-only model workbench."""

from __future__ import annotations

import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.workbench_main import FRONTEND_DIST, app  # noqa: E402
from backend.app.workbench_service import (  # noqa: E402
    SHOWCASE_BENCHMARK_ID,
    SHOWCASE_CANDIDATE_MANIFEST_PATH,
    SHOWCASE_CHECKPOINT_PATH,
    SHOWCASE_MANIFEST_PATH,
    SHOWCASE_ONNX_PATH,
    SHOWCASE_TRAINING_RECEIPT_PATH,
    load_workbench_overview,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is missing, empty, or unsafe: {path}")


def _verify_frontend() -> list[Path]:
    index_path = FRONTEND_DIST / "index.html"
    _require_file(index_path, "workbench frontend index")
    assets = sorted((FRONTEND_DIST / "assets").glob("*"))
    files = [path for path in assets if path.is_file() and not path.is_symlink()]
    if not any(path.suffix == ".js" for path in files):
        raise RuntimeError("workbench frontend JavaScript bundle is missing")
    if not any(path.suffix == ".css" for path in files):
        raise RuntimeError("workbench frontend stylesheet bundle is missing")
    for path in files:
        _require_file(path, "workbench frontend asset")
    return files


def _verify_onnx_runtime() -> dict[str, Any]:
    session = ort.InferenceSession(
        str(SHOWCASE_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "observation" or inputs[0].shape[-1] != 21:
        raise RuntimeError("showcase ONNX input contract differs")
    if len(outputs) != 1 or outputs[0].name != "action_logits" or outputs[0].shape[-1] != 5:
        raise RuntimeError("showcase ONNX output contract differs")
    logits = session.run(None, {"observation": np.zeros((1, 21), dtype=np.float32)})[0]
    if logits.shape != (1, 5) or not np.isfinite(logits).all():
        raise RuntimeError("showcase ONNX smoke inference is invalid")
    return {
        "input_features": 21,
        "output_actions": 5,
        "providers": session.get_providers(),
    }


def _verify_http(overview: dict[str, Any], assets: list[Path]) -> None:
    with TestClient(app) as client:
        response = client.get("/api/workbench/v1/overview")
        if response.status_code != 200 or response.json() != overview:
            raise RuntimeError("workbench evidence endpoint differs from the verified loader")
        if response.content != _canonical_bytes(overview):
            raise RuntimeError("workbench evidence endpoint is not canonical JSON")
        if response.headers.get("cache-control") != "no-store":
            raise RuntimeError("workbench evidence endpoint is cacheable")
        ready = client.get("/health/ready")
        if ready.status_code != 200 or ready.json().get("status") != "ready":
            raise RuntimeError("workbench readiness endpoint failed")
        if ready.headers.get("cache-control") != "no-store":
            raise RuntimeError("workbench readiness endpoint is cacheable")
        index = client.get("/")
        if index.status_code != 200 or "text/html" not in index.headers.get("content-type", ""):
            raise RuntimeError("workbench frontend index smoke failed")
        if index.headers.get("cache-control") != "no-store":
            raise RuntimeError("workbench frontend index is cacheable")
        first_asset = assets[0]
        asset = client.get(f"/assets/{first_asset.name}")
        if asset.status_code != 200 or not asset.content:
            raise RuntimeError("workbench static asset smoke failed")
        toolbox = client.post(
            "/api/workbench/v1/toolbox/evaluate",
            json={
                "public_forecast_signal": [1.25, 0.1, -0.45, 0.35, -0.2],
                "visible_service_need": [0.25, 1.1, 0.2, 0.15, 0.1],
                "public_regime": 2,
                "current_service_health": [0.68, 0.71, 0.64, 0.73, 0.69],
                "phase_window": 1,
            },
        )
        toolbox_body = toolbox.json()
        if (
            toolbox.status_code != 200
            or toolbox_body.get("runtime", {}).get("real_model_inference") is not True
            or toolbox_body.get("model", {}).get("parameter_count") != 300_113
            or toolbox_body.get("model", {}).get("action_label") != "healthcare"
            or toolbox_body.get("heuristic", {}).get("action_label") != "housing"
        ):
            raise RuntimeError("workbench live toolbox inference smoke failed")
        if toolbox.content != _canonical_bytes(toolbox_body):
            raise RuntimeError("workbench live toolbox response is not canonical JSON")
        if toolbox.headers.get("cache-control") != "no-store":
            raise RuntimeError("workbench live toolbox response is cacheable")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version.split()[0]}")
    unexpected_training_packages = [
        name for name in ("stable_baselines3", "torch") if find_spec(name) is not None
    ]
    if unexpected_training_packages:
        raise RuntimeError(
            "CPU presentation environment unexpectedly includes training packages: "
            + ", ".join(unexpected_training_packages)
        )

    for path, label in (
        (SHOWCASE_MANIFEST_PATH, "showcase final manifest"),
        (SHOWCASE_CANDIDATE_MANIFEST_PATH, "showcase candidate manifest"),
        (SHOWCASE_TRAINING_RECEIPT_PATH, "showcase training receipt"),
        (SHOWCASE_ONNX_PATH, "showcase ONNX model"),
        (SHOWCASE_CHECKPOINT_PATH, "showcase checkpoint"),
    ):
        _require_file(path, label)

    overview = load_workbench_overview()
    benchmark = overview.get("benchmark", {})
    if benchmark.get("status") != "measured":
        raise RuntimeError("workbench benchmark is not measured evidence")
    if benchmark.get("benchmark_id") != SHOWCASE_BENCHMARK_ID:
        raise RuntimeError("workbench benchmark identity differs")

    assets = _verify_frontend()
    onnx_summary = _verify_onnx_runtime()
    _verify_http(overview, assets)

    forbidden_imports = sorted(
        name
        for name in sys.modules
        if name == "torch"
        or name == "backend.app.main"
        or name.startswith("backend.app.training_v4")
        or name.startswith("backend.app.v5")
    )
    if forbidden_imports:
        raise RuntimeError(
            "minimal workbench preflight imported legacy/training modules: "
            + ", ".join(forbidden_imports)
        )

    print(
        json.dumps(
            {
                "benchmark_id": SHOWCASE_BENCHMARK_ID,
                "frontend_assets": len(assets),
                "model": onnx_summary,
                "status": "workbench-preflight-passed",
                "training_packages_installed": False,
                "training_stack_imported": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
