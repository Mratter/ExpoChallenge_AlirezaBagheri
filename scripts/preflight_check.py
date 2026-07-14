from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.artifact import (  # noqa: E402
    ARTIFACT_LICENSE,
    ARTIFACT_SOURCE,
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
from backend.app.simulator import SERVICES, compare  # noqa: E402


def validate_exposed_metadata(metadata: dict[str, Any], bundle: PolicyBundle) -> None:
    expected_model = {
        "version": POLICY_VERSION,
        "schema_version": POLICY_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "feature_order": list(POLICY_FEATURE_ORDER),
        "license": ARTIFACT_LICENSE,
        "source": ARTIFACT_SOURCE,
        "sha256": bundle.sha256,
    }
    expected_dataset = {
        "version": DATASET_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "license": "CC0-1.0",
        "source": "backend/app/simulator.py",
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


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version.split()[0]}")

    bundle = load_policy_bundle()
    policy = bundle.content
    metadata = metadata_payload(policy, bundle.sha256)
    validate_exposed_metadata(metadata, bundle)
    result = compare(Scenario(), 424242, policy, bundle.sha256)
    if result.get("schema_version") != API_SCHEMA_VERSION:
        raise RuntimeError("comparison response schema version is invalid")
    for planner_name in ("baseline", "candidate"):
        planner = result[planner_name]
        if len(planner["trajectory"]) != 14 or planner["constraint_violations"] != 0:
            raise RuntimeError(f"{planner_name} smoke trajectory failed")
        measured_total = sum(
            day["projection"]["constraint_violations"] for day in planner["trajectory"]
        )
        if measured_total != planner["constraint_violations"]:
            raise RuntimeError(f"{planner_name} violation total is inconsistent")
        for day in planner["trajectory"]:
            if abs(sum(day["allocation"]) - day["available_budget"]) > 1e-7:
                raise RuntimeError(f"{planner_name} allocation sum failed on day {day['day']}")
    if result["shock_schedule"][4]["type"] != "utility":
        raise RuntimeError("forced fixture shock is missing")
    print(
        json.dumps(
            {
                "api_schema_version": metadata["schema_version"],
                "artifact_bytes": bundle.size_bytes,
                "artifact_license": bundle.license,
                "artifact_path": bundle.relative_path,
                "artifact_source": bundle.source,
                "candidate_rauc": result["candidate"]["rauc"],
                "dataset_schema_version": metadata["dataset"]["schema_version"],
                "dataset_version": metadata["dataset"]["version"],
                "manifest_schema_version": bundle.manifest_schema_version,
                "policy_schema_version": metadata["model"]["schema_version"],
                "policy_sha256": bundle.sha256,
                "policy_version": metadata["model"]["version"],
                "schedule_sha256": result["shock_schedule_sha256"],
                "status": "preflight-smoke-passed",
                "urgency_rauc": result["baseline"]["rauc"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
