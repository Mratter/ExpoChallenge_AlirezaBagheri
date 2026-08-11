"""Blocking served-path replay gate for the bundled v4 policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.city.scenarios import DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
from model import policy as policy_module

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_POLICY = ROOT / "artifacts" / "city_recovery_ppo.v4.onnx"
PARITY_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "city_recovery_ppo.v4.parity.json"
)
EXPECTED_POLICY_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)
EXPECTED_CASE_COUNT = 200
EXPECTED_SOLVED_COUNT = 178


@pytest.fixture
def served_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.delenv(main.POLICY_PATH_ENV, raising=False)
    monkeypatch.delenv(main.POLICY_SHA256_ENV, raising=False)
    monkeypatch.setenv("INNOVERSE_STATE_DIR", str(tmp_path))
    policy_module._load_cached.cache_clear()
    try:
        with TestClient(main.app) as client:
            yield client
    finally:
        policy_module._load_cached.cache_clear()


def test_bundled_policy_replays_development_parity_through_served_routes(
    served_client: TestClient,
) -> None:
    receipt = json.loads(PARITY_RECEIPT.read_text(encoding="utf-8"))
    parity = receipt["parity"]
    expected_rows = parity["rows"]
    ordered_row_ids = [
        f"{family.id}:{case_seed}"
        for family in DEVELOPMENT_FAMILIES
        for case_seed in DEVELOPMENT_SEEDS
    ]

    assert main.DEFAULT_POLICY_PATH.resolve() == BUNDLED_POLICY.resolve()
    assert hashlib.sha256(BUNDLED_POLICY.read_bytes()).hexdigest() == (
        EXPECTED_POLICY_SHA256
    )
    assert receipt["split"] == parity["split"] == "dev"
    assert receipt["final_split_used"] is parity["final_split_used"] is False
    assert receipt["development_case_count"] == EXPECTED_CASE_COUNT
    assert parity["case_count"] == parity["expected_case_count"] == (
        EXPECTED_CASE_COUNT
    )
    assert receipt["onnx_artifact"] == {
        "path": "artifacts/city_recovery_ppo.v4.onnx",
        "sha256": EXPECTED_POLICY_SHA256,
    }
    assert len(ordered_row_ids) == len(set(ordered_row_ids)) == EXPECTED_CASE_COUNT
    assert [row["row_id"] for row in expected_rows] == ordered_row_ids
    assert [row["onnx"]["row_id"] for row in expected_rows] == ordered_row_ids

    ready_response = served_client.get("/health/ready")
    metadata_response = served_client.get("/api/v1/meta")
    assert ready_response.status_code == 200, ready_response.text
    assert metadata_response.status_code == 200, metadata_response.text

    ready = ready_response.json()
    model = metadata_response.json()["model"]
    assert ready["status"] == "ready"
    assert ready["policy_id"] == ready["policy_path_stem"] == (
        "city_recovery_ppo.v4"
    )
    assert ready["policy_sha256"] == EXPECTED_POLICY_SHA256
    assert ready["observation_count"] == 73
    assert ready["action_count"] == 22
    assert model["id"] == model["path_stem"] == "city_recovery_ppo.v4"
    assert model["sha256"] == EXPECTED_POLICY_SHA256
    assert model["observation_count"] == 73
    assert model["action_count"] == 22

    replayed_row_ids: list[str] = []
    solved_count = 0
    hard_violation_count = 0
    maximum_conservation_residual = 0.0
    row_index = 0
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            row_id = f"{family.id}:{case_seed}"
            expected_row = expected_rows[row_index]
            expected_onnx = expected_row["onnx"]
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)

            comparison_response = served_client.post(
                "/api/v1/simulations/compare",
                json={
                    "scenario": scenario.model_dump(mode="json"),
                    "seed": tape_seed,
                },
            )
            assert comparison_response.status_code == 200, comparison_response.text
            comparison = comparison_response.json()
            persisted_response = served_client.get(
                f"/api/v1/simulations/{comparison['result_id']}"
            )
            assert persisted_response.status_code == 200, persisted_response.text
            persisted = persisted_response.json()
            assert persisted == comparison

            candidate = persisted["candidate"]
            candidate_solved = candidate["absolute_outcome"]["solved"]
            assert expected_row["row_id"] == expected_onnx["row_id"] == row_id
            assert expected_row["case_seed"] == expected_onnx["case_seed"] == (
                case_seed
            )
            assert expected_row["tape_seed"] == expected_onnx["tape_seed"] == (
                tape_seed
            )
            assert expected_row["tape_sha256"] == expected_onnx["tape_sha256"]
            assert persisted["seed"] == tape_seed
            assert persisted["shock_schedule_sha256"] == expected_onnx["tape_sha256"]
            assert candidate_solved == expected_onnx["solved"]
            assert persisted["comparison"]["candidate_solved"] == candidate_solved
            assert candidate["trajectory_sha256"] == expected_onnx[
                "trajectory_sha256"
            ]

            hard_violations = candidate["hard_violation_count"]
            assert hard_violations == expected_onnx["hard_violation_count"] == 0
            hard_violation_count += hard_violations
            conservation_residual = candidate[
                "max_logistics_conservation_residual"
            ]
            assert conservation_residual == expected_onnx[
                "maximum_conservation_residual"
            ]
            maximum_conservation_residual = max(
                maximum_conservation_residual,
                abs(conservation_residual),
            )

            replayed_row_ids.append(row_id)
            solved_count += int(candidate_solved)
            row_index += 1

    assert row_index == EXPECTED_CASE_COUNT
    assert replayed_row_ids == ordered_row_ids
    assert solved_count == parity["onnx_solved_count"] == EXPECTED_SOLVED_COUNT
    assert hard_violation_count == parity["onnx_hard_violation_count"] == 0
    assert maximum_conservation_residual == (
        parity["onnx_maximum_conservation_residual"]
    ) == 0.0
