from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.workbench_main import app, create_app
from backend.app.workbench_service import WorkbenchEvidenceError


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_minimal_workbench_api_returns_verified_canonical_evidence() -> None:
    with TestClient(app) as client:
        first = client.get("/api/workbench/v1/overview")
        second = client.get("/api/workbench/v1/overview")

    assert first.status_code == 200
    assert first.content == second.content
    assert first.headers["cache-control"] == "no-store"
    body = first.json()
    assert first.content == _canonical_bytes(body)
    assert body["benchmark"]["status"] == "measured"
    assert body["benchmark"]["benchmark_id"] == "adaptive-cascades-showcase-v2"
    assert body["benchmark"]["objective"]["learned_policy"]["passes"] == 38
    assert body["benchmark"]["objective"]["static_heuristic"]["passes"] == 20


def _toolbox_request() -> dict[str, object]:
    return {
        "public_forecast_signal": [1.25, 0.1, -0.45, 0.35, -0.2],
        "visible_service_need": [0.25, 1.1, 0.2, 0.15, 0.1],
        "public_regime": 2,
        "current_service_health": [0.68, 0.71, 0.64, 0.73, 0.69],
        "phase_window": 1,
    }


def test_toolbox_runs_real_v2_onnx_and_fixed_heuristic() -> None:
    with TestClient(app) as client:
        first = client.post("/api/workbench/v1/toolbox/evaluate", json=_toolbox_request())
        second = client.post("/api/workbench/v1/toolbox/evaluate", json=_toolbox_request())

    assert first.status_code == 200
    assert first.content == second.content
    assert first.content == _canonical_bytes(first.json())
    assert first.headers["cache-control"] == "no-store"
    body = first.json()
    assert body["schema_version"] == "model-toolbox-evaluation-v1"
    assert body["benchmark_id"] == "adaptive-cascades-showcase-v2"
    assert body["runtime"] == {
        "engine": "onnxruntime",
        "execution_provider": "CPUExecutionProvider",
        "onnx_sha256": "b3edf8007feb749ddc33fc3ebbb008a02ef98d561bd74cfde286dde030a4dae0",
        "real_model_inference": True,
    }
    assert len(body["input"]["feature_order"]) == 21
    assert len(body["input"]["vector"]) == 21
    assert len(body["input"]["normalization"]["mean"]) == 21
    assert len(body["input"]["normalization"]["scale"]) == 21
    assert body["input"]["vector"][10:14] == [0.0, 0.0, 1.0, 0.0]
    assert body["input"]["vector"][19:21] == pytest.approx([0.0, 1.0])

    model = body["model"]
    heuristic = body["heuristic"]
    assert model["id"] == "adaptive-cascade-mlp-v2-300k"
    assert model["parameter_count"] == 300_113
    assert len(model["logits"]) == 5
    assert len(model["probabilities"]) == 5
    assert sum(model["probabilities"]) == pytest.approx(1.0)
    assert model["action_index"] == max(range(5), key=model["probabilities"].__getitem__)
    assert model["action_label"] == "healthcare"
    assert heuristic["action_index"] == 1
    assert heuristic["action_label"] == "housing"
    assert heuristic["scores"] == pytest.approx(_toolbox_request()["visible_service_need"])
    assert body["comparison"] == {"same_action": False}
    assert body["benchmark_summary"] == {
        "scenario_total": 40,
        "objective_passes": {"model": 38, "heuristic": 20},
        "head_to_head": {"model_wins": 38, "heuristic_wins": 0, "ties": 2},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_forecast_signal", [0.0] * 4),
        ("visible_service_need", [0.0, 0.0, 0.0, 0.0, 1.51]),
        ("public_regime", 4),
        ("current_service_health", [0.5, 0.5, -0.1, 0.5, 0.5]),
        ("phase_window", 13),
        ("public_forecast_signal", [0.0, 0.0, True, 0.0, 0.0]),
    ],
)
def test_toolbox_rejects_invalid_public_input(field: str, value: object) -> None:
    request = _toolbox_request()
    request[field] = value

    with TestClient(app) as client:
        response = client.post("/api/workbench/v1/toolbox/evaluate", json=request)

    assert response.status_code == 422
    assert response.content == _canonical_bytes(response.json())
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["error"]["code"] == "INVALID_TOOLBOX_INPUT"
    assert any(detail["field"].startswith(field) for detail in response.json()["error"]["details"])


def test_toolbox_forbids_unknown_or_hidden_fields() -> None:
    request = _toolbox_request()
    request["hidden_target"] = 3

    with TestClient(app) as client:
        response = client.post("/api/workbench/v1/toolbox/evaluate", json=request)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_TOOLBOX_INPUT"
    assert response.json()["error"]["details"][0]["field"] == "hidden_target"


def test_toolbox_fails_closed_before_inference_when_evidence_is_invalid() -> None:
    with (
        patch(
            "backend.app.workbench_toolbox.load_workbench_overview",
            side_effect=WorkbenchEvidenceError("showcase ONNX model digest differs"),
        ),
        TestClient(app) as client,
    ):
        response = client.post("/api/workbench/v1/toolbox/evaluate", json=_toolbox_request())

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "TOOLBOX_RUNTIME_NOT_READY",
        "details": [],
        "message": "showcase ONNX model digest differs",
    }


def test_minimal_workbench_api_fails_closed_on_invalid_evidence() -> None:
    with (
        patch(
            "backend.app.workbench_main.load_workbench_overview",
            side_effect=WorkbenchEvidenceError("sealed result digest differs"),
        ),
        TestClient(app) as client,
    ):
        response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "WORKBENCH_EVIDENCE_NOT_READY",
            "details": [],
            "message": "sealed result digest differs",
        }
    }
    assert response.content == _canonical_bytes(response.json())


def test_minimal_workbench_serves_static_assets_and_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    index = b"<!doctype html><html><body><div id='root'></div></body></html>"
    script = b"console.log('workbench')"
    (dist / "index.html").write_bytes(index)
    (assets / "app.js").write_bytes(script)

    static_app = create_app(dist)
    with TestClient(static_app) as client:
        root = client.get("/")
        asset = client.get("/assets/app.js")
        spa = client.get("/models/showcase-adaptive-v2")

    assert root.status_code == 200
    assert root.content == index
    assert "text/html" in root.headers["content-type"]
    assert root.headers["cache-control"] == "no-store"
    assert asset.status_code == 200
    assert asset.content == script
    assert spa.status_code == 200
    assert spa.content == index
    assert spa.headers["cache-control"] == "no-store"


def test_minimal_workbench_404_is_canonical_and_does_not_fall_back_for_api() -> None:
    with TestClient(app) as client:
        missing_api = client.get("/api/workbench/v1/missing")
        missing_asset = client.get("/assets/does-not-exist.js")
        reserved_roots = [client.get(path) for path in ("/api", "/health", "/assets")]

    for response in (missing_api, missing_asset, *reserved_roots):
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"
        assert response.content == _canonical_bytes(response.json())
        assert response.headers["cache-control"] == "no-store"


def test_minimal_workbench_method_error_is_canonical() -> None:
    with TestClient(app) as client:
        response = client.post("/api/workbench/v1/overview")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "HTTP_ERROR"
    assert response.content == _canonical_bytes(response.json())
    assert response.headers["cache-control"] == "no-store"


def test_minimal_workbench_rejects_symlinked_frontend_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    index_path = dist / "index.html"
    index_path.write_text("<html>unsafe indirection</html>", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def report_index_as_symlink(path: Path) -> bool:
        return path == index_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_index_as_symlink)

    symlink_app = create_app(dist)
    with TestClient(symlink_app) as client:
        response = client.get("/")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "FRONTEND_NOT_READY"
    assert response.headers["cache-control"] == "no-store"


def test_minimal_workbench_reports_missing_frontend_without_masking_api(
    tmp_path: Path,
) -> None:
    missing_dist_app = create_app(tmp_path / "missing-dist")
    with TestClient(missing_dist_app) as client:
        frontend = client.get("/")
        api = client.get("/api/workbench/v1/overview")

    assert frontend.status_code == 503
    assert frontend.json()["error"]["code"] == "FRONTEND_NOT_READY"
    assert api.status_code == 200


def test_minimal_workbench_import_graph_excludes_training_stacks() -> None:
    source = (
        "import sys; import backend.app.workbench_main; "
        "forbidden=[name for name in sys.modules if name == 'torch' "
        "or name == 'backend.app.main' "
        "or name.startswith('backend.app.training_v4') "
        "or name.startswith('backend.app.v5')]; "
        "assert not forbidden, forbidden"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
