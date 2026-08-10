"""Focused replay, explanation, and counterfactual API tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.city.environment import (
    ACTION_ORDER,
    OBSERVATION_ORDER,
    compare,
)
from backend.app.models import Scenario
from backend.app.persistence import RunStore
from model.policy import load_policy


@pytest.fixture
def persisted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    policy_path = Path("tests/fixtures/legacy_policy.onnx").resolve()
    policy = load_policy(policy_path)
    result = RunStore(tmp_path).save(compare(Scenario(), 424242, policy))
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv(main.POLICY_PATH_ENV, str(policy_path))
    monkeypatch.delenv(main.POLICY_SHA256_ENV, raising=False)
    return result


def test_explanations_replay_exactly_and_cover_every_ordered_channel(
    persisted_run: dict[str, Any],
) -> None:
    response = TestClient(main.app).get(
        f"/api/v1/simulations/{persisted_run['result_id']}/explanations"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_id"] == persisted_run["result_id"]
    assert payload["policy"]["sha256"] == persisted_run["policy"]["sha256"]
    assert (
        payload["shock_schedule_sha256"]
        == persisted_run["shock_schedule_sha256"]
    )
    assert payload["future_tape_visible"] is False
    assert payload["method"]["causal"] is False
    assert "not a causal" in payload["method"]["interpretation"]
    assert payload["method"]["batch_size_per_day"] == 73
    assert payload["observation_order"] == list(OBSERVATION_ORDER)
    assert payload["action_order"] == list(ACTION_ORDER)
    assert payload["day_count"] == len(payload["days"]) == 30
    assert payload["observation_count"] == 73
    assert payload["action_count"] == 22

    for index, day in enumerate(payload["days"]):
        assert day["day"] == index + 1
        assert day["base_raw_action"] == persisted_run["candidate"]["trajectory"][
            index
        ]["raw_action"]
        assert len(day["channels"]) == 73
        assert [item["observation_name"] for item in day["channels"]] == list(
            OBSERVATION_ORDER
        )
        assert [item["observation_index"] for item in day["channels"]] == list(
            range(73)
        )
        assert sorted(item["influence_rank"] for item in day["channels"]) == list(
            range(1, 74)
        )
        for item in day["channels"]:
            assert all(
                math.isfinite(item[field])
                for field in (
                    "observed_value",
                    "mean_absolute_action_delta",
                    "normalized_influence",
                    "signed_action_delta",
                )
            )
            assert item["most_affected_action"] == ACTION_ORDER[
                item["most_affected_action_index"]
            ]
        total = sum(item["normalized_influence"] for item in day["channels"])
        assert total == pytest.approx(1.0, abs=5e-9)


def test_explanations_fail_closed_when_stored_trajectory_does_not_replay(
    persisted_run: dict[str, Any],
) -> None:
    persisted_run["candidate"]["trajectory"][0]["raw_action"][0] += 0.01
    RunStore().save(persisted_run)

    response = TestClient(main.app).get(
        f"/api/v1/simulations/{persisted_run['result_id']}/explanations"
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "REPLAY_MISMATCH"


def test_counterfactual_changes_only_the_selected_day_treatment_and_is_not_saved(
    persisted_run: dict[str, Any],
) -> None:
    store = RunStore()
    before = store.list_summaries()
    request = {
        "day": 5,
        "material_shares": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    client = TestClient(main.app)

    first = client.post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json=request,
    )
    second = client.post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json=request,
    )

    assert first.status_code == second.status_code == 200
    payload = first.json()
    assert second.json() == payload
    assert payload["analysis_only"] is True
    assert payload["persisted"] is False
    assert len(payload["analysis_id"]) == 64
    assert payload["same_disaster_tape"] is True
    assert payload["shock_schedule_sha256"] == persisted_run[
        "shock_schedule_sha256"
    ]
    assert payload["unchanged_prefix"]["days"] == 4
    assert payload["unchanged_prefix"]["matches"] is True
    assert (
        payload["unchanged_prefix"]["original_sha256"]
        == payload["unchanged_prefix"]["counterfactual_sha256"]
    )
    assert sum(payload["treatment"]["material_shares"]) == pytest.approx(1.0)
    assert payload["treatment"]["crew_shares"] is None
    realized = payload["selected_day_realized_allocations"]
    assert realized["counterfactual"]["material"] != realized["original"][
        "material"
    ]
    assert realized["counterfactual"]["crew"] == realized["original"]["crew"]
    assert len(payload["daily_deltas"]) == 30
    for day in payload["daily_deltas"][:4]:
        assert day["services_end"] == [0.0] * 5
        assert day["preparedness_end"] == [0.0] * 5
        assert day["resilience"] == 0.0
        assert day["reward"] == 0.0
    assert store.list_summaries() == before


def test_counterfactual_resubmitting_realized_shares_is_an_exact_no_op(
    persisted_run: dict[str, Any],
) -> None:
    selected = persisted_run["candidate"]["trajectory"][4]
    material = np.asarray(selected["material_allocation"], dtype=np.float64)
    crew = np.asarray(selected["crew_allocation"], dtype=np.float64)

    response = TestClient(main.app).post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json={
            "day": 5,
            # Match DecisionAnalysis.shareInputs(): percentages shown to four decimals,
            # then normalized again by the request builder.
            "material_shares": np.round(100.0 * material / material.sum(), 4).tolist(),
            "crew_shares": np.round(100.0 * crew / crew.sum(), 4).tolist(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert (
        payload["counterfactual"]["trajectory_sha256"]
        == payload["original"]["trajectory_sha256"]
    )
    assert payload["selected_day_realized_allocations"]["counterfactual"] == payload[
        "selected_day_realized_allocations"
    ]["original"]
    assert all(
        day["services_end"] == [0.0] * 5
        and day["preparedness_end"] == [0.0] * 5
        and day["resilience"] == 0.0
        and day["reward"] == 0.0
        for day in payload["daily_deltas"]
    )


def test_counterfactual_normalizes_large_finite_weights_without_overflow(
    persisted_run: dict[str, Any],
) -> None:
    response = TestClient(main.app).post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json={"day": 5, "material_shares": [1e308] * 5},
    )

    assert response.status_code == 200
    assert response.json()["treatment"]["material_shares"] == [0.2] * 5


@pytest.mark.parametrize(
    "body",
    [
        {"day": 5},
        {"day": 0, "material_shares": [1, 1, 1, 1, 1]},
        {"day": 5, "material_shares": [1, 1, 1, 1]},
        {"day": 5, "material_shares": [0, 0, 0, 0, 0]},
        {"day": 5, "crew_shares": [1, -1, 1, 1, 1]},
        {"day": 5, "crew_shares": [1, 1, "not-finite", 1, 1]},
    ],
)
def test_counterfactual_validation_is_field_local(
    persisted_run: dict[str, Any], body: dict[str, Any]
) -> None:
    response = TestClient(main.app).post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json=body,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_COUNTERFACTUAL"
    assert response.json()["error"]["details"]


def test_analysis_routes_report_not_found_and_policy_unavailable(
    persisted_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(main.app)
    missing = "f" * 64
    assert (
        client.get(f"/api/v1/simulations/{missing}/explanations").status_code
        == 404
    )

    monkeypatch.delenv(main.POLICY_PATH_ENV, raising=False)
    unavailable = client.post(
        f"/api/v1/simulations/{persisted_run['result_id']}/counterfactuals",
        json={"day": 5, "crew_shares": [1, 1, 1, 1, 1]},
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "DEPENDENCY_NOT_READY"
