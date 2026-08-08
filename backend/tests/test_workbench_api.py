from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import workbench_service
from backend.app.workbench_main import app
from backend.app.workbench_service import WorkbenchEvidenceError

client = TestClient(app)


def _track(body: dict[str, object], track_id: str) -> dict[str, object]:
    tracks = body["tracks"]
    assert isinstance(tracks, list)
    return next(track for track in tracks if track["id"] == track_id)


def _metrics(track: dict[str, object]) -> dict[str, dict[str, object]]:
    evaluation = track["evaluation"]
    assert isinstance(evaluation, dict)
    rows = evaluation["metrics"]
    assert isinstance(rows, list)
    return {row["id"]: row for row in rows}


def test_workbench_overview_exposes_five_distinct_evidence_classes() -> None:
    first = client.get("/api/workbench/v1/overview")
    second = client.get("/api/workbench/v1/overview")

    assert first.status_code == 200
    assert first.content == second.content
    body = first.json()
    assert body["schema_version"] == "model-workbench-v1"
    assert [track["id"] for track in body["tracks"]] == list(workbench_service.EXPECTED_TRACK_IDS)
    assert len({track["evidence_class"] for track in body["tracks"]}) == 5

    v2_metrics = _metrics(_track(body, "production-v2"))
    assert v2_metrics["scenario_wins"]["value"] == 32
    assert v2_metrics["scenario_total"]["value"] == 40
    assert v2_metrics["baseline_wins"]["value"] == 8
    assert v2_metrics["deterministic_executions"]["value"] == 200
    assert v2_metrics["deterministic_mismatches"]["value"] == 0
    assert _track(body, "production-v2")["training"]["unit"] == "transitions"

    showcase = _track(body, "showcase-adaptive-v2")
    assert showcase["status"] == "trained_evaluated"
    assert showcase["claim_eligible"] is True
    assert showcase["architecture"]["parameters"] == 300113
    assert showcase["architecture"]["runtime"] == "ONNX Runtime CPUExecutionProvider"
    assert showcase["training"]["started"] is True
    assert showcase["training"]["transitions"] == 9600
    assert showcase["training"]["unit"] == "labeled windows"
    assert showcase["training"]["hardware"] == "CPU"
    showcase_metrics = _metrics(showcase)
    assert showcase_metrics["objective_learned_passes"]["value"] == 38
    assert showcase_metrics["objective_static_heuristic_passes"]["value"] == 20
    assert showcase_metrics["head_to_head_learned_wins"]["value"] == 38
    assert showcase_metrics["head_to_head_heuristic_wins"]["value"] == 0
    assert showcase_metrics["head_to_head_ties"]["value"] == 2
    assert showcase_metrics["paired_mean_difference"]["value"] == pytest.approx(2.25)

    r22 = _track(body, "architecture-r22-v10")
    assert r22["status"] == "untrained_terminal_no_go"
    assert r22["claim_eligible"] is False
    assert r22["training"] == {
        "started": False,
        "transitions": 0,
        "unit": "transitions",
        "seed_count": 0,
        "hardware": "Training was planned for cuda:0 but never authorized",
        "note": (
            "R22 received zero optimizer updates and zero production-training transitions. "
            "V10 was a bounded search diagnostic, not neural training."
        ),
    }
    r22_metrics = _metrics(r22)
    assert r22_metrics["diagnostic_reduction_percent"]["value"] == pytest.approx(7.3426804414)
    assert r22_metrics["required_gate_percent"]["value"] == 40
    assert r22_metrics["exact_simulator_calls"]["value"] == 8406

    expected = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert first.content == expected


def test_workbench_benchmark_is_derived_from_the_sealed_showcase_result() -> None:
    response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 200
    benchmark = response.json()["benchmark"]
    assert benchmark["status"] == "measured"
    assert benchmark["benchmark_id"] == "adaptive-cascades-showcase-v2"
    assert benchmark["model_track_id"] == "showcase-adaptive-v2"
    assert benchmark["scenario_total"] == 40
    assert benchmark["synthetic_disclosure"] == (
        "Engineered synthetic benchmark of learnable observable patterns; "
        "not real-world validation."
    )

    objective = benchmark["objective"]
    assert objective["counts_are_independent_not_complementary"] is True
    assert objective["success_threshold"] == 10
    assert objective["learned_policy"]["passes"] == 38
    assert objective["learned_policy"]["misses"] == 2
    assert objective["static_heuristic"]["passes"] == 20
    assert objective["static_heuristic"]["misses"] == 20
    assert objective["learned_policy"]["passes"] + objective["learned_policy"]["misses"] == 40
    assert objective["static_heuristic"]["passes"] + objective["static_heuristic"]["misses"] == 40

    head_to_head = benchmark["head_to_head"]
    assert head_to_head["learned_wins"] == 38
    assert head_to_head["heuristic_wins"] == 0
    assert head_to_head["ties"] == 2
    assert (
        head_to_head["learned_wins"] + head_to_head["heuristic_wins"] + head_to_head["ties"] == 40
    )
    assert head_to_head["learned_mean"] == pytest.approx(11.5)
    assert head_to_head["heuristic_mean"] == pytest.approx(9.25)
    assert head_to_head["paired_mean_difference"] == pytest.approx(2.25)
    assert head_to_head["paired_bootstrap_ci95"] == pytest.approx([1.775, 2.75])

    assert benchmark["secondary"] == {
        "metric": {
            "id": "critical_service_deficit_auc",
            "label": "Critical service deficit AUC",
            "direction": "lower_is_better",
        },
        "learned_mean": pytest.approx(0.30383109070500003),
        "heuristic_mean": pytest.approx(0.33655319428),
    }


def test_workbench_copied_showcase_bundle_has_the_pinned_hashes() -> None:
    manifest_bytes = workbench_service.SHOWCASE_MANIFEST_PATH.read_bytes()
    result_bytes = (workbench_service.SHOWCASE_FINAL_ROOT / "result.json").read_bytes()
    candidate_manifest_bytes = workbench_service.SHOWCASE_CANDIDATE_MANIFEST_PATH.read_bytes()

    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        workbench_service.EXPECTED_SHOWCASE_MANIFEST_SHA256
    )
    assert hashlib.sha256(result_bytes).hexdigest() == (
        workbench_service.EXPECTED_SHOWCASE_RESULT_SHA256
    )
    assert hashlib.sha256(candidate_manifest_bytes).hexdigest() == (
        workbench_service.EXPECTED_SHOWCASE_CANDIDATE_MANIFEST_SHA256
    )


def test_workbench_preserves_and_verifies_archived_version_one_evidence() -> None:
    archived = workbench_service.ARCHIVED_SHOWCASE_V1_SPEC

    assert hashlib.sha256(archived.manifest_path.read_bytes()).hexdigest() == (
        archived.expected_manifest_sha256
    )
    assert hashlib.sha256((archived.final_root / "result.json").read_bytes()).hexdigest() == (
        archived.expected_result_sha256
    )
    assert hashlib.sha256(archived.candidate_manifest_path.read_bytes()).hexdigest() == (
        archived.expected_candidate_manifest_sha256
    )


def test_workbench_endpoint_fails_closed_on_tampered_overview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.v1.json"
    overview = tmp_path / "overview.v1.json"
    manifest.write_bytes(workbench_service.WORKBENCH_MANIFEST_PATH.read_bytes())
    source = workbench_service.WORKBENCH_MANIFEST_PATH.parent / "overview.v1.json"
    overview.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(workbench_service, "WORKBENCH_MANIFEST_PATH", manifest)

    response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WORKBENCH_EVIDENCE_NOT_READY"
    assert "digest differs" in response.json()["error"]["message"]


def test_workbench_endpoint_fails_closed_on_missing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workbench_service,
        "WORKBENCH_MANIFEST_PATH",
        tmp_path / "missing-manifest.json",
    )

    response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WORKBENCH_EVIDENCE_NOT_READY"
    assert "missing or unsafe" in response.json()["error"]["message"]


def test_workbench_endpoint_fails_closed_on_missing_showcase_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_spec = replace(
        workbench_service.ACTIVE_SHOWCASE_SPEC,
        manifest_path_override=tmp_path / "missing-showcase-manifest.json",
    )
    monkeypatch.setattr(workbench_service, "ACTIVE_SHOWCASE_SPEC", missing_spec)

    response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WORKBENCH_EVIDENCE_NOT_READY"
    assert "showcase final manifest is missing or unsafe" in response.json()["error"]["message"]


def test_workbench_endpoint_fails_closed_on_tampered_showcase_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tampered_onnx = tmp_path / "adaptive-cascade-mlp-v2-300k.onnx"
    tampered_onnx.write_bytes(workbench_service.SHOWCASE_ONNX_PATH.read_bytes() + b"tampered")
    tampered_spec = replace(
        workbench_service.ACTIVE_SHOWCASE_SPEC,
        onnx_path_override=tampered_onnx,
    )
    monkeypatch.setattr(workbench_service, "ACTIVE_SHOWCASE_SPEC", tampered_spec)

    response = client.get("/api/workbench/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WORKBENCH_EVIDENCE_NOT_READY"
    assert "ONNX model digest differs" in response.json()["error"]["message"]


def test_workbench_validator_prohibits_relabeling_r22_as_trained() -> None:
    document = workbench_service.load_workbench_overview()
    tampered = copy.deepcopy(document)
    r22 = _track(tampered, "architecture-r22-v10")
    training = r22["training"]
    assert isinstance(training, dict)
    training["started"] = True
    training["transitions"] = 1

    with pytest.raises(WorkbenchEvidenceError, match="R22 must remain explicitly untrained"):
        workbench_service._validate_overview(tampered)


def test_workbench_validator_rejects_hand_edited_showcase_claim() -> None:
    document = workbench_service.load_workbench_overview()
    tampered = copy.deepcopy(document)
    tampered["benchmark"]["objective"]["learned_policy"]["passes"] = 40

    with pytest.raises(
        WorkbenchEvidenceError,
        match="benchmark differs from sealed showcase result",
    ):
        workbench_service._validate_overview(tampered)
