from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.artifact import (  # noqa: E402
    ARTIFACT_LICENSE,
    MANIFEST_SCHEMA_VERSION,
    POLICY_FEATURE_ORDER,
    POLICY_SCHEMA_VERSION,
    POLICY_VERSION,
    PolicyBundle,
    load_policy_bundle,
)
from backend.app.main import (  # noqa: E402
    API_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    metadata_payload,
)
from backend.app.models import Scenario  # noqa: E402
from backend.app.persistence import RunStore  # noqa: E402
from backend.app.simulator import SERVICES, canonical_json_bytes, compare  # noqa: E402

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


def validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != API_SCHEMA_VERSION:
        raise RuntimeError("comparison response schema version is invalid")
    schedule = result["shock_schedule"]
    for planner_name in ("baseline", "candidate"):
        planner = result[planner_name]
        if len(planner["trajectory"]) != 14 or planner["constraint_violations"] != 0:
            raise RuntimeError(f"{planner_name} smoke trajectory failed")
        measured_total = sum(
            day["projection"]["constraint_violations"] for day in planner["trajectory"]
        )
        if measured_total != planner["constraint_violations"]:
            raise RuntimeError(f"{planner_name} violation total is inconsistent")
        for day, shock in zip(planner["trajectory"], schedule, strict=True):
            if day["shock"] != shock:
                raise RuntimeError(f"{planner_name} did not receive the shared shock tape")
            allocation = np.asarray(day["allocation"])
            lower = np.asarray(day["lower_bounds"])
            upper = np.asarray(day["upper_bounds"])
            budget = day["available_budget"]
            if abs(float(allocation.sum()) - budget) > 1e-7:
                raise RuntimeError(f"{planner_name} allocation sum failed on day {day['day']}")
            if np.any(allocation < lower - 1e-7) or np.any(allocation > upper + 1e-7):
                raise RuntimeError(f"{planner_name} allocation bounds failed on day {day['day']}")
            if float(allocation.sum()) > budget + 1e-7:
                raise RuntimeError(f"{planner_name} allocation budget failed on day {day['day']}")
            if any(day["projection"]["violation_breakdown"].values()):
                raise RuntimeError(f"{planner_name} violation evidence is inconsistent")


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


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version.split()[0]}")

    bundle = load_policy_bundle()
    metadata = metadata_payload(bundle)
    validate_exposed_metadata(metadata, bundle)
    result = compare(Scenario(), 424242, bundle)
    validate_result(result)
    with tempfile.TemporaryDirectory(prefix="ai17-preflight-") as directory:
        store = RunStore(Path(directory))
        saved = store.save(result)
        restored = RunStore(Path(directory)).load(saved["result_id"])
        if canonical_json_bytes(saved) != canonical_json_bytes(restored):
            raise RuntimeError("persisted result did not restore byte-identically")
        if store.list_summaries()[0]["result_id"] != saved["result_id"]:
            raise RuntimeError("persisted result index is inconsistent")
    evaluation = validate_evaluation(bundle)
    if result["shock_schedule"][4]["type"] != "utility":
        raise RuntimeError("forced fixture shock is missing")
    print(
        json.dumps(
            {
                "api_schema_version": metadata["schema_version"],
                "baseline_rauc": result["baseline"]["rauc"],
                "candidate_rauc": result["candidate"]["rauc"],
                "dataset_schema_version": metadata["dataset"]["schema_version"],
                "evaluation_baseline_rauc": evaluation["aggregate"]["rauc"][
                    "baseline_mean"
                ],
                "evaluation_candidate_rauc": evaluation["aggregate"]["rauc"][
                    "candidate_mean"
                ],
                "evaluation_cases": evaluation["evaluation_case_count"],
                "manifest_schema_version": bundle.manifest_schema_version,
                "onnx_sha256": bundle.onnx_sha256,
                "parity_report_sha256": bundle.parity_sha256,
                "policy_schema_version": metadata["model"]["schema_version"],
                "policy_version": metadata["model"]["version"],
                "sb3_checkpoint_sha256": bundle.sb3_sha256,
                "schedule_sha256": result["shock_schedule_sha256"],
                "status": "preflight-smoke-passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
