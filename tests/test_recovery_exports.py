"""Focused deterministic CSV/PDF recovery-plan export tests."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.city.environment import compare
from backend.app.city.physics import SERVICES
from backend.app.models import Scenario
from backend.app.persistence import RunStore
from backend.app.recovery_exports import CSV_FIELDS, recovery_plan_csv
from model.policy import load_policy


@pytest.fixture
def persisted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    policy = load_policy(Path("tests/fixtures/legacy_policy.onnx").resolve())
    result = RunStore(tmp_path).save(compare(Scenario(), 424242, policy))
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path))
    return result


def test_csv_export_has_150_ordered_day_service_rows_and_auditable_fields(
    persisted_run: dict[str, Any],
) -> None:
    client = TestClient(main.app)
    path = (
        f"/api/v1/simulations/{persisted_run['result_id']}"
        "/recovery-plan?planner=candidate&format=csv"
    )

    first = client.get(path)
    second = client.get(path)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.headers["content-type"] == "text/csv; charset=utf-8"
    assert first.headers["content-disposition"].endswith('-candidate.csv"')
    assert first.content.endswith(b"\n")
    rows = list(csv.DictReader(io.StringIO(first.text)))
    assert tuple(rows[0]) == CSV_FIELDS
    assert len(rows) == 30 * len(SERVICES) == 150
    for day in range(1, 31):
        block = rows[(day - 1) * 5 : day * 5]
        assert [row["service"] for row in block] == list(SERVICES)
        assert {row["day"] for row in block} == {str(day)}
        assert {row["result_id"] for row in block} == {
            persisted_run["result_id"]
        }
        assert {row["policy_sha256"] for row in block} == {
            persisted_run["policy"]["sha256"]
        }
        assert {row["shock_schedule_sha256"] for row in block} == {
            persisted_run["shock_schedule_sha256"]
        }
    assert {
        "service_before",
        "service_after_shock",
        "service_target",
        "service_end",
        "material_allocation",
        "crew_allocation",
        "stock_release",
        "preparedness_investment",
        "resilience",
        "reward",
        "assessment_tail_active",
        "hard_violation_count",
        "conservation_residual",
    }.issubset(rows[0])


def test_csv_export_neutralizes_spreadsheet_formula_prefixes(
    persisted_run: dict[str, Any],
) -> None:
    for unsafe_name in ("=1+1", "+1", "-1", "@SUM(A1:A2)", "\t=1", "\r=1"):
        persisted_run["scenario"]["name"] = unsafe_name
        rows = list(
            csv.DictReader(
                io.StringIO(recovery_plan_csv(persisted_run, "candidate").decode())
            )
        )
        assert rows[0]["scenario_name"] == f"'{unsafe_name}"


def test_pdf_export_is_deterministic_paginated_and_includes_endpoints(
    persisted_run: dict[str, Any],
) -> None:
    client = TestClient(main.app)
    path = (
        f"/api/v1/simulations/{persisted_run['result_id']}"
        "/recovery-plan?planner=baseline&format=pdf"
    )

    first = client.get(path)
    second = client.get(path)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.headers["content-type"] == "application/pdf"
    assert first.headers["content-disposition"].endswith('-baseline.pdf"')
    assert first.content.startswith(b"%PDF-1.4")
    assert first.content.endswith(b"%%EOF\n")
    assert b"/Count 2" in first.content
    assert b"CITY RECOVERY PLAN / 30-DAY EVIDENCE BRIEF" in first.content
    assert b"SECTOR ENDPOINTS" in first.content
    for service in SERVICES:
        assert service.encode("ascii") in first.content


@pytest.mark.parametrize("tamper", ["trajectory", "schedule"])
def test_export_fails_closed_on_stale_evidence_hashes(
    persisted_run: dict[str, Any], tamper: str
) -> None:
    if tamper == "trajectory":
        persisted_run["candidate"]["trajectory"][0]["reward"] += 0.01
    else:
        persisted_run["shock_schedule"][0]["severity"] += 0.01
    RunStore().save(persisted_run)

    response = TestClient(main.app).get(
        f"/api/v1/simulations/{persisted_run['result_id']}"
        "/recovery-plan?planner=candidate&format=csv"
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "EXPORT_FAILED"
    assert "hash does not match" in response.json()["error"]["message"]


def test_export_route_returns_404_and_rejects_unknown_query_values(
    persisted_run: dict[str, Any],
) -> None:
    client = TestClient(main.app)
    missing = client.get(
        f"/api/v1/simulations/{'f' * 64}/recovery-plan?format=csv"
    )
    invalid_planner = client.get(
        f"/api/v1/simulations/{persisted_run['result_id']}"
        "/recovery-plan?planner=oracle&format=csv"
    )
    invalid_format = client.get(
        f"/api/v1/simulations/{persisted_run['result_id']}"
        "/recovery-plan?planner=candidate&format=xlsx"
    )
    malformed_id = client.get(
        "/api/v1/simulations/not-a-digest/recovery-plan?format=csv"
    )

    assert missing.status_code == 404
    assert invalid_planner.status_code == 422
    assert invalid_format.status_code == 422
    for response in (invalid_planner, invalid_format):
        assert response.json()["error"]["code"] == "INVALID_EXPORT"
        assert (
            response.json()["error"]["message"]
            == "Recovery-plan export validation failed."
        )
    assert malformed_id.status_code == 400
    assert malformed_id.json()["error"]["code"] == "PERSISTENCE_FAILED"
