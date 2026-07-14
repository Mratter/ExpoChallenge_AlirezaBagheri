from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.artifact import ArtifactError
from backend.app.main import app

client = TestClient(app)


def test_required_health_and_meta_endpoints() -> None:
    assert client.get("/health/live").json() == {"status": "live"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    meta = client.get("/api/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["dataset"]["empirical"] is False
    assert meta.json()["model"]["artifact_type"] == "deterministic_linear_policy_candidate"


def test_gate2_fixture_is_canonical_deterministic_and_constrained() -> None:
    payload = {"seed": 424242, "scenario": {}}
    first = client.post("/api/v1/simulations/compare", json=payload)
    second = client.post("/api/v1/simulations/compare", json=payload)

    assert first.status_code == 200
    assert first.content == second.content
    body = first.json()
    expected_schedule_hash = "af3a57e9b378700a49a2da8d2042ebc9eb08178cc525cad93f4954306ae5ec81"
    assert body["shock_schedule_sha256"] == expected_schedule_hash
    assert body["candidate"]["rauc"] == 0.49401335
    assert body["baseline"]["rauc"] == 0.49166123
    assert body["shock_schedule"][4]["forced"] is True
    assert body["baseline"]["constraint_violations"] == 0
    assert body["candidate"]["constraint_violations"] == 0
    for planner in ("baseline", "candidate"):
        assert len(body[planner]["trajectory"]) == 14
        for day in body[planner]["trajectory"]:
            assert round(sum(day["allocation"]), 8) == day["available_budget"]
            assert day["projection"]["constraint_violations"] == 0
    assert body["baseline"]["trajectory"][4]["shock"] == body["candidate"]["trajectory"][4]["shock"]


def test_unseen_valid_input_changes_real_computation() -> None:
    fixture = client.post("/api/v1/simulations/compare", json={"seed": 900001, "scenario": {
        "name": "Unseen bridge outage",
        "horizon_days": 9,
        "daily_budget": 137,
        "initial_services": [0.22, 0.61, 0.37, 0.44, 0.28],
        "priorities": [1.8, 0.7, 1.1, 1.5, 0.9],
        "shock_probability": 0.17,
        "severity_min": 0.08,
        "severity_max": 0.24,
        "forced_shock": None,
    }})
    assert fixture.status_code == 200
    result = fixture.json()
    assert len(result["candidate"]["trajectory"]) == 9
    assert result["scenario"]["daily_budget"] == 137
    assert result["shock_schedule_sha256"]


def test_invalid_input_has_structured_error() -> None:
    response = client.post(
        "/api/v1/simulations/compare",
        json={"seed": 1, "scenario": {"horizon_days": 99}},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_SCENARIO"
    assert response.json()["error"]["details"]


def test_corrupt_required_artifact_fails_closed() -> None:
    with patch("backend.app.main.load_policy", side_effect=ArtifactError("checksum mismatch")):
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_NOT_READY"
