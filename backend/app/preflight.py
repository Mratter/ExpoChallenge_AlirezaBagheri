from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.artifact import (
    ARTIFACT_LICENSE,
    MANIFEST_SCHEMA_VERSION,
    POLICY_FEATURE_ORDER,
    POLICY_SCHEMA_VERSION,
    POLICY_VERSION,
    PolicyBundle,
)
from backend.app.main import (
    API_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    metadata_payload,
)
from backend.app.simulator import SERVICES

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "evaluation" / "feature_complete_report.v1.json"
PROTOCOL_PATH = ROOT / "evaluation" / "protocol.v1.json"


def validate_exposed_metadata(metadata: dict[str, Any], bundle: PolicyBundle) -> None:
    expected_model = {
        "version": POLICY_VERSION,
        "schema_version": POLICY_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "observation_order": list(POLICY_FEATURE_ORDER),
        "license": ARTIFACT_LICENSE,
        "source": "scripts/train_policy.py",
        "onnx_sha256": bundle.onnx_sha256,
        "sb3_checkpoint_sha256": bundle.sb3_sha256,
        "parity_report_sha256": bundle.parity_sha256,
    }
    expected_dataset = {
        "version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "license": "CC0-1.0",
        "source": "backend/app/simulator.py and backend/app/scenarios.py",
        "service_order": list(SERVICES),
        "empirical": False,
    }
    if metadata.get("schema_version") != API_SCHEMA_VERSION:
        raise RuntimeError("exposed API schema version is invalid")
    model = metadata.get("model")
    dataset = metadata.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise RuntimeError("exposed model or dataset metadata is missing")
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise RuntimeError(f"exposed model metadata {field} is invalid")
    for field, expected in expected_dataset.items():
        if dataset.get(field) != expected:
            raise RuntimeError(f"exposed dataset metadata {field} is invalid")


def validate_evaluation(bundle: PolicyBundle) -> dict[str, Any]:
    report_payload = EVALUATION_PATH.read_bytes()
    report = json.loads(report_payload.decode("utf-8"))
    protocol_sha256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    if report.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("evaluation protocol checksum is inconsistent")
    if report.get("onnx_sha256") != bundle.onnx_sha256:
        raise RuntimeError("evaluation ONNX checksum is inconsistent")
    if report.get("sb3_checkpoint_sha256") != bundle.sb3_sha256:
        raise RuntimeError("evaluation SB3 checkpoint checksum is inconsistent")
    if report.get("evaluation_case_count") != 40 or report.get("held_out_family_count") != 5:
        raise RuntimeError("evaluation case coverage is incomplete")
    if report.get("determinism", {}).get("mismatches") != 0:
        raise RuntimeError("evaluation determinism evidence failed")
    if report.get("synthetic_only") is not True:
        raise RuntimeError("evaluation synthetic disclosure is missing")
    for planner in ("candidate", "baseline"):
        totals = report.get("violation_totals", {}).get(planner, {})
        if set(totals.values()) != {0}:
            raise RuntimeError(f"evaluation {planner} hard constraints failed")
    return report
