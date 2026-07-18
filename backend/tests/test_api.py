from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.artifact import ArtifactError
from backend.app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path / "state"))


def test_required_health_and_meta_endpoints() -> None:
    assert client.get("/health/live").json() == {"status": "live"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["policy_type"] == "stable_baselines3_ppo"
    meta = client.get("/api/v1/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["schema_version"] == "2.1.0"
    assert body["dataset"]["empirical"] is False
    assert body["dataset"]["schema_version"] == "2.0.0"
    assert body["model"]["artifact_type"] == "stable_baselines3_ppo"
    assert body["model"]["algorithm"] == "PPO"
    assert body["model"]["legacy_candidate"]["is_ppo"] is False
    assert body["baseline"]["library"] == "OR-Tools"


def test_feature_complete_fixture_is_canonical_deterministic_and_constrained() -> None:
    payload = {"seed": 424242, "scenario": {}}
    first = client.post("/api/v1/simulations/compare", json=payload)
    second = client.post("/api/v1/simulations/compare", json=payload)

    assert first.status_code == 200
    assert first.content == second.content
    body = first.json()
    assert body["schema_version"] == "2.1.0"
    assert body["scenario"]["forced_shocks"] == []
    expected_schedule_hash = "af3a57e9b378700a49a2da8d2042ebc9eb08178cc525cad93f4954306ae5ec81"
    assert body["shock_schedule_sha256"] == expected_schedule_hash
    assert body["candidate"]["rauc"] == 0.48599305
    assert body["baseline"]["rauc"] == 0.43592031
    assert body["shock_schedule"][4]["forced"] is True
    assert body["baseline_spec"]["library"] == "OR-Tools"
    assert body["policy"]["artifact_type"] == "stable_baselines3_ppo"
    assert body["policy"]["legacy_candidate"]["is_ppo"] is False
    for planner in ("baseline", "candidate"):
        assert body[planner]["constraint_violations"] == 0
        assert len(body[planner]["trajectory"]) == 14
        for day in body[planner]["trajectory"]:
            assert round(sum(day["allocation"]), 8) == day["available_budget"]
            assert day["projection"]["constraint_violations"] == 0
            assert not any(day["projection"]["violation_breakdown"].values())
            assert all(
                lower - 1e-7 <= value <= upper + 1e-7
                for lower, value, upper in zip(
                    day["lower_bounds"],
                    day["allocation"],
                    day["upper_bounds"],
                    strict=True,
                )
            )
    assert body["baseline"]["trajectory"][4]["shock"] == body["candidate"][
        "trajectory"
    ][4]["shock"]


def test_unseen_valid_input_changes_real_computation_and_persists() -> None:
    fixture = client.post(
        "/api/v1/simulations/compare",
        json={
            "seed": 900001,
            "scenario": {
                "name": "Unseen bridge outage",
                "horizon_days": 9,
                "daily_budget": 137,
                "initial_services": [0.22, 0.61, 0.37, 0.44, 0.28],
                "priorities": [1.8, 0.7, 1.1, 1.5, 0.9],
                "shock_probability": 0.17,
                "severity_min": 0.08,
                "severity_max": 0.24,
                "forced_shock": None,
            },
        },
    )
    assert fixture.status_code == 200
    result = fixture.json()
    assert len(result["candidate"]["trajectory"]) == 9
    assert result["scenario"]["daily_budget"] == 137
    assert result["shock_schedule_sha256"]
    restored = client.get(f"/api/v1/simulations/{result['result_id']}")
    assert restored.status_code == 200
    assert restored.content == fixture.content
    index = client.get("/api/v1/simulations").json()
    assert index["count"] == 1
    assert index["results"][0]["result_id"] == result["result_id"]


def test_invalid_input_has_structured_error() -> None:
    response = client.post(
        "/api/v1/simulations/compare",
        json={"seed": 1, "scenario": {"horizon_days": 99}},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_SCENARIO"
    assert response.json()["error"]["details"]


@pytest.mark.parametrize(
    "forced_shocks",
    [
        [{"day": 8, "type": "weather", "severity": 0.24}],
        [{"day": 7, "type": "utility", "severity": 0.20, "unknown": True}],
        [{"day": 4, "type": "meteor", "severity": 0.20}],
        [{"day": 4, "type": "supply", "severity": 0.049}],
        [{"day": 4, "type": "epidemic", "severity": 0.401}],
    ],
)
def test_forced_shocks_remain_strict_and_bounded_by_horizon(
    forced_shocks: list[dict[str, object]],
) -> None:
    response = client.post(
        "/api/v1/simulations/compare",
        json={
            "seed": 1,
            "scenario": {"horizon_days": 7, "forced_shocks": forced_shocks},
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_SCENARIO"
    assert response.json()["error"]["details"]


def test_corrupt_required_artifact_fails_closed() -> None:
    with patch("backend.app.main.load_policy", side_effect=ArtifactError("checksum mismatch")):
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_NOT_READY"


@pytest.mark.parametrize("artifact_state", ["missing", "corrupt"])
def test_live_onnx_loss_blocks_api_persistence_and_primary_ui(
    tmp_path: Path, artifact_state: str
) -> None:
    policy_path = tmp_path / "policy.onnx"
    if artifact_state == "corrupt":
        policy_path.write_bytes(b"corrupt")

    with patch("backend.app.artifact.ONNX_POLICY_PATH", policy_path):
        assert client.get("/health/live").status_code == 200
        responses = [
            client.get("/health/ready"),
            client.get("/api/v1/meta"),
            client.get("/api/v1/simulations"),
            client.post("/api/v1/simulations/compare", json={"seed": 1, "scenario": {}}),
            client.get("/"),
        ]

    for response in responses:
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "DEPENDENCY_NOT_READY"
        assert "candidate" not in response.json()
    assert b"Civic Relay" not in responses[-1].content


def test_corrupt_persisted_result_fails_explicitly(tmp_path: Path) -> None:
    response = client.post(
        "/api/v1/simulations/compare", json={"seed": 71, "scenario": {}}
    )
    result_id = response.json()["result_id"]
    persisted = tmp_path / "state" / "runs" / f"{result_id}.json"
    persisted.write_text("{}", encoding="utf-8")

    restored = client.get(f"/api/v1/simulations/{result_id}")
    assert restored.status_code == 500
    assert restored.json()["error"]["code"] == "PERSISTENCE_FAILED"
