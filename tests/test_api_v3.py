from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.persistence import RunStore
from model.ppo_v3 import ArtifactError as V3ArtifactError


def _v3_bundle() -> SimpleNamespace:
    benchmark = {
        "artifact_type": "city_recovery_v3_final_benchmark",
        "status": "complete",
        "primary_metric": "independent_absolute_disasters_solved",
        "synthetic_case_count": 40,
        "candidate": {
            "id": "city-recovery-sb3-ppo-v3-selected",
            "solved_count": 30,
            "failed_count": 10,
            "solve_rate": 0.75,
        },
        "baseline": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "solved_count": 10,
            "failed_count": 30,
            "solve_rate": 0.25,
        },
        "paired_absolute_outcomes": {
            "both_solved": 8,
            "ppo_only": 22,
            "heuristic_only": 2,
            "neither": 8,
        },
        "secondary_head_to_head_resilience_auc": {
            "candidate_wins": 29,
            "baseline_wins": 11,
            "ties": 0,
            "candidate_mean_minus_baseline_mean": 0.08,
        },
        "invariants": {
            "all_rows_present": True,
            "unique_rows": True,
            "same_tapes": True,
            "deterministic_replay_mismatches": 0,
            "candidate_hard_violations": 0,
            "baseline_hard_violations": 0,
            "maximum_conservation_residual": 0.0,
        },
        "rows_sha256": "d" * 64,
        "ledger_sha256": "e" * 64,
    }
    metadata = {
        "id": "city-recovery-sb3-ppo-v3-selected",
        "version": "3.0.0",
        "artifact_type": "stable_baselines3_ppo",
        "architecture": {
            "trainable_parameters": 322733,
        },
        "training": {
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "requested_transitions": 645120,
            "completed_transitions": 92160,
        },
    }
    return SimpleNamespace(
        metadata=metadata,
        onnx_sha256="a" * 64,
        sb3_sha256="b" * 64,
        manifest_sha256="c" * 64,
        manifest_schema_version=1,
        parity_sha256="f" * 64,
        scientific_source_sha256="1" * 64,
        benchmark=benchmark,
        benchmark_sha256="2" * 64,
    )


def _planner(rauc: float, solved: bool | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"rauc": rauc}
    if solved is not None:
        value["absolute_outcome"] = {"solved": solved}
    return value


def test_primary_routes_fail_closed_without_selected_release(monkeypatch: Any) -> None:
    def unavailable() -> None:
        raise V3ArtifactError("selected V3 release is absent")

    monkeypatch.setattr(main, "load_policy_v3", unavailable)
    client = TestClient(main.app)

    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    metadata = client.get("/api/v1/meta")
    assert ready.status_code == 503
    assert metadata.status_code == 503
    assert ready.json()["error"]["code"] == "DEPENDENCY_NOT_READY"


def test_metadata_exposes_verified_counts_and_73_by_22_contract(
    monkeypatch: Any,
) -> None:
    bundle = _v3_bundle()
    monkeypatch.setattr(main, "load_policy_v3", lambda: bundle)
    response = TestClient(main.app).get("/api/v1/meta")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["id"] == "city-recovery-sb3-ppo-v3-selected"
    assert payload["model"]["observation_count"] == 73
    assert payload["model"]["action_count"] == 22
    assert len(payload["model"]["observation_order"]) == 73
    assert len(payload["model"]["action_order"]) == 22
    assert payload["model"]["onnx_sha256"] == bundle.onnx_sha256
    assert payload["model"]["selected_checkpoint_sha256"] == bundle.sb3_sha256
    assert payload["model"]["manifest_sha256"] == bundle.manifest_sha256
    assert payload["model"]["parity_sha256"] == bundle.parity_sha256
    assert (
        payload["model"]["scientific_source_sha256"] == bundle.scientific_source_sha256
    )
    assert payload["benchmark"]["artifact_sha256"] == bundle.benchmark_sha256
    assert payload["benchmark"]["status"] == "complete"
    assert (
        payload["benchmark"]["candidate"]["id"] == "city-recovery-sb3-ppo-v3-selected"
    )
    assert payload["benchmark"]["candidate"]["solved_count"] == 30
    assert payload["benchmark"]["baseline"]["solved_count"] == 10
    assert payload["benchmark"]["primary_metric"] == (
        "independent_absolute_disasters_solved"
    )


def test_health_ready_exposes_selected_policy_contract(monkeypatch: Any) -> None:
    bundle = _v3_bundle()
    monkeypatch.setattr(main, "load_policy_v3", lambda: bundle)

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["policy_id"] == "city-recovery-sb3-ppo-v3-selected"
    assert payload["policy_sha256"] == bundle.onnx_sha256
    assert payload["benchmark_sha256"] == bundle.benchmark_sha256
    assert payload["observation_count"] == 73
    assert payload["action_count"] == 22


def test_compare_request_requires_v3_fixed_horizon_before_loading_policy(
    monkeypatch: Any,
) -> None:
    called = False

    def should_not_load() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(main, "load_policy_v3", should_not_load)
    response = TestClient(main.app).post(
        "/api/v1/simulations/compare",
        json={"seed": 1, "scenario": {"horizon_days": 14}},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_SCENARIO"
    assert called is False


def test_valid_compare_uses_v3_and_persists_result(
    monkeypatch: Any, tmp_path: Path
) -> None:
    bundle = _v3_bundle()
    monkeypatch.setattr(main, "load_policy_v3", lambda: bundle)
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path))

    def compare_v3(scenario: Any, seed: int, supplied_bundle: Any) -> dict[str, Any]:
        assert scenario.horizon_days == 30
        assert supplied_bundle is bundle
        return {
            "schema_version": "4.0.0",
            "engine_version": "city-recovery-env-v3",
            "engine_spec_sha256": "7" * 64,
            "outcome_definition_sha256": "8" * 64,
            "seed": seed,
            "scenario": scenario.model_dump(mode="json"),
            "policy": {"sha256": bundle.onnx_sha256},
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

    monkeypatch.setattr(main, "compare_v3", compare_v3)
    response = TestClient(main.app).post(
        "/api/v1/simulations/compare", json={"seed": 8}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine_version"] == "city-recovery-env-v3"
    assert payload["scenario"]["horizon_days"] == 30
    assert payload["comparison"]["candidate_solved"] is True
    assert (
        RunStore(tmp_path).load(payload["result_id"])["result_id"]
        == payload["result_id"]
    )


def test_persistence_round_trips_and_summarizes_v3(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    v3 = {
        "schema_version": "4.0.0",
        "engine_version": "city-recovery-env-v3",
        "engine_spec_sha256": "7" * 64,
        "outcome_definition_sha256": "8" * 64,
        "seed": 8,
        "scenario": {"name": "primary", "horizon_days": 30},
        "policy": {"sha256": "9" * 64},
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
    saved_v3 = store.save(v3)

    assert store.load(saved_v3["result_id"])["engine_version"] == (
        "city-recovery-env-v3"
    )
    v3_summaries = store.list_summaries(engine_version="city-recovery-env-v3")
    assert len(v3_summaries) == 1
    assert v3_summaries[0]["candidate_solved"] is True
    assert v3_summaries[0]["baseline_solved"] is False
    assert v3_summaries[0]["outcome"] == "ppo_only"
