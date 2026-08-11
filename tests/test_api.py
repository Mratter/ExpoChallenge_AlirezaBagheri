"""API and persistence tests for the single current runtime contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.city.environment import RAW_OBSERVATION_CONTRACT, policy_identity
from backend.app.persistence import RunStore, result_identity
from backend.app.shared_evidence import canonical_hash
from model.policy import PolicyError

POLICY_SHA256 = "a" * 64
SHIPPED_POLICY_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)


def _policy() -> SimpleNamespace:
    return SimpleNamespace(path=Path("/models/candidate.onnx"), sha256=POLICY_SHA256)


def _planner(rauc: float, solved: bool) -> dict[str, Any]:
    return {"rauc": rauc, "absolute_outcome": {"solved": solved}}


def _result(seed: int = 8) -> dict[str, Any]:
    policy = _policy()
    return {
        "schema_version": "4.0.0",
        "engine_version": "city-recovery-env-v3",
        "engine_spec_sha256": "7" * 64,
        "outcome_definition_sha256": "8" * 64,
        "seed": seed,
        "scenario": {"name": "primary", "horizon_days": 30},
        "policy": policy_identity(policy),
        "baseline_spec": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
        },
        "candidate": _planner(0.55, True),
        "baseline": _planner(0.47, False),
        "comparison": {
            "primary_metric": "independent_absolute_disaster_solved",
            "candidate_solved": True,
            "baseline_solved": False,
            "absolute_outcome_pair": "ppo_only",
        },
    }


def test_primary_routes_use_the_bundled_policy_without_environment_configuration(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv(main.POLICY_PATH_ENV, raising=False)
    monkeypatch.delenv(main.POLICY_SHA256_ENV, raising=False)
    client = TestClient(main.app)

    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    metadata = client.get("/api/v1/meta")

    assert hashlib.sha256(main.DEFAULT_POLICY_PATH.read_bytes()).hexdigest() == (
        SHIPPED_POLICY_SHA256
    )
    assert ready.status_code == 200
    assert ready.json()["policy_id"] == "city_recovery_ppo.v4"
    assert ready.json()["policy_path_stem"] == "city_recovery_ppo.v4"
    assert ready.json()["policy_sha256"] == SHIPPED_POLICY_SHA256
    assert ready.json()["observation_count"] == 73
    assert ready.json()["action_count"] == 22
    assert metadata.status_code == 200
    assert metadata.json()["model"]["id"] == "city_recovery_ppo.v4"
    assert metadata.json()["model"]["sha256"] == SHIPPED_POLICY_SHA256


def test_missing_bundled_policy_fails_closed_without_a_fixture_fallback(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(main.POLICY_PATH_ENV, raising=False)
    monkeypatch.delenv(main.POLICY_SHA256_ENV, raising=False)
    monkeypatch.setattr(main, "DEFAULT_POLICY_PATH", tmp_path / "missing-v4.onnx")

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_NOT_READY"
    assert "missing or unreadable" in response.json()["error"]["message"]


def test_environment_policy_and_hash_override_the_bundled_default(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    selected = _policy()

    def load(path: str, expected_sha256: str | None = None) -> Any:
        captured.update(path=path, expected_sha256=expected_sha256)
        return selected

    monkeypatch.setenv(main.POLICY_PATH_ENV, "C:/models/current.onnx")
    monkeypatch.setenv(main.POLICY_SHA256_ENV, "B" * 64)
    monkeypatch.setattr(main, "load_policy", load)

    assert main.configured_policy() is selected
    assert captured == {
        "path": "C:/models/current.onnx",
        "expected_sha256": "B" * 64,
    }


def test_metadata_is_lean_and_exposes_the_raw_observation_contract(
    monkeypatch: Any,
) -> None:
    policy = _policy()
    monkeypatch.setattr(main, "configured_policy", lambda: policy)

    response = TestClient(main.app).get("/api/v1/meta")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "app",
        "version",
        "schema_version",
        "default_seed",
        "services",
        "model",
        "environment",
        "outcome_definition",
        "outcome_definition_sha256",
        "baseline",
        "persistence",
        "determinism",
    }
    model = payload["model"]
    assert model["id"] == model["path_stem"] == "candidate"
    assert model["sha256"] == POLICY_SHA256
    assert model["runtime"] == "ONNX Runtime CPUExecutionProvider"
    assert model["observation_contract"] == RAW_OBSERVATION_CONTRACT
    assert model["observation_count"] == 73
    assert model["action_count"] == 22
    assert len(model["observation_order"]) == 73
    assert len(model["action_order"]) == 22
    assert "benchmark" not in payload
    forbidden = {
        "manifest_sha256",
        "selected_checkpoint_sha256",
        "scientific_source_sha256",
        "checkpoint_authorization",
    }
    assert forbidden.isdisjoint(model)


def test_health_ready_exposes_current_policy_identity(monkeypatch: Any) -> None:
    monkeypatch.setattr(main, "configured_policy", _policy)

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "engine_id": "CityRecoveryEnv-v3",
        "policy_id": "candidate",
        "policy_path_stem": "candidate",
        "policy_sha256": POLICY_SHA256,
        "policy_type": "onnx_policy",
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "observation_contract": RAW_OBSERVATION_CONTRACT,
        "observation_count": 73,
        "action_count": 22,
    }


def test_request_validation_happens_before_policy_loading(monkeypatch: Any) -> None:
    called = False

    def should_not_load() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(main, "configured_policy", should_not_load)
    response = TestClient(main.app).post(
        "/api/v1/simulations/compare",
        json={"seed": 1, "scenario": {"horizon_days": 14}},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_SCENARIO"
    assert called is False


def test_policy_load_failures_make_compare_unavailable(monkeypatch: Any) -> None:
    def unavailable() -> None:
        raise PolicyError("configured policy hash does not match")

    monkeypatch.setattr(main, "configured_policy", unavailable)
    response = TestClient(main.app).post("/api/v1/simulations/compare", json={})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_NOT_READY"


def test_compare_persists_the_current_physical_result(
    monkeypatch: Any, tmp_path: Path
) -> None:
    policy = _policy()
    monkeypatch.setattr(main, "configured_policy", lambda: policy)
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path))

    def compare(scenario: Any, seed: int, supplied_policy: Any) -> dict[str, Any]:
        assert scenario.horizon_days == 30
        assert supplied_policy is policy
        return _result(seed)

    monkeypatch.setattr(main, "compare", compare)
    response = TestClient(main.app).post(
        "/api/v1/simulations/compare", json={"seed": 8}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_id"] == result_identity(payload)
    assert payload["policy"]["path_stem"] == "candidate"
    assert payload["policy"]["observation_contract"] == RAW_OBSERVATION_CONTRACT
    assert payload["comparison"]["candidate_solved"] is True
    assert RunStore(tmp_path).load(payload["result_id"]) == payload


def test_persistence_uses_one_current_scientific_identity(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    result = _result()
    expected = canonical_hash(
        {
            "schema_version": result["schema_version"],
            "engine_version": result["engine_version"],
            "engine_spec_sha256": result["engine_spec_sha256"],
            "outcome_definition_sha256": result["outcome_definition_sha256"],
            "seed": result["seed"],
            "scenario": result["scenario"],
            "policy_sha256": result["policy"]["sha256"],
            "baseline_id": result["baseline_spec"]["id"],
            "baseline_version": result["baseline_spec"]["version"],
        }
    )

    saved = store.save(result)

    assert saved["result_id"] == expected
    assert store.load(expected) == saved
    summaries = store.list_summaries(engine_version="city-recovery-env-v3")
    assert len(summaries) == 1
    assert summaries[0]["candidate_solved"] is True
    assert summaries[0]["baseline_solved"] is False
    assert summaries[0]["outcome"] == "ppo_only"
