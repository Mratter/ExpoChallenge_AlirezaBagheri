from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"
POLICY_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"

ARTIFACT_ID = "frozen-policy-v1"
ARTIFACT_LICENSE = "CC0-1.0"
ARTIFACT_RELATIVE_PATH = "artifacts/frozen_policy.v1.json"
ARTIFACT_SOURCE = "scripts/build_policy_artifact.py"
MANIFEST_SCHEMA_VERSION = 1
POLICY_ARTIFACT_TYPE = "deterministic_linear_policy_candidate"
POLICY_FEATURE_ORDER = (
    "priority_deficit",
    "criticality",
    "marginal_gain",
    "network_centrality",
)
POLICY_ID = "frozen-policy-candidate-v1"
POLICY_SCHEMA_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyBundle:
    content: dict[str, Any]
    sha256: str
    size_bytes: int
    license: str
    relative_path: str
    source: str
    manifest_schema_version: int


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ArtifactError("policy manifest root must be an object")
    if manifest.get("project") != "AI17":
        raise ArtifactError("policy manifest project must be AI17")
    if manifest.get("version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(
            f"policy manifest schema version must be {MANIFEST_SCHEMA_VERSION}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ArtifactError("policy manifest must contain exactly one artifact")
    record = artifacts[0]
    if not isinstance(record, dict):
        raise ArtifactError("policy manifest artifact record must be an object")

    expected_fields = {
        "id": ARTIFACT_ID,
        "license": ARTIFACT_LICENSE,
        "path": ARTIFACT_RELATIVE_PATH,
        "source": ARTIFACT_SOURCE,
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            raise ArtifactError(f"policy manifest {field} must be {expected}")
    if type(record.get("bytes")) is not int or record["bytes"] <= 0:
        raise ArtifactError("policy manifest bytes must be a positive integer")
    if not isinstance(record.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
        record["sha256"]
    ):
        raise ArtifactError("policy manifest sha256 must be 64 lowercase hex characters")
    return record


def _validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ArtifactError("frozen policy root must be an object")
    if policy.get("artifact_type") != POLICY_ARTIFACT_TYPE:
        raise ArtifactError("frozen policy artifact type is unsupported")
    if policy.get("id") != POLICY_ID:
        raise ArtifactError(f"frozen policy id must be {POLICY_ID}")
    if policy.get("version") != POLICY_VERSION:
        raise ArtifactError(f"frozen policy version must be {POLICY_VERSION}")
    if policy.get("feature_order") != list(POLICY_FEATURE_ORDER):
        raise ArtifactError("frozen policy feature order does not match the runtime schema")

    weights = policy.get("feature_weights")
    if not isinstance(weights, dict) or set(weights) != set(POLICY_FEATURE_ORDER):
        raise ArtifactError("frozen policy weights do not match the runtime feature schema")
    numeric_weights: list[float] = []
    for feature in POLICY_FEATURE_ORDER:
        value = weights[feature]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ArtifactError(f"frozen policy weight {feature} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ArtifactError(f"frozen policy weight {feature} must be between 0 and 1")
        numeric_weights.append(numeric)
    if not math.isclose(sum(numeric_weights), 1.0, abs_tol=1e-12):
        raise ArtifactError("frozen policy weights must sum to 1")

    calibration = policy.get("calibration")
    expected_calibration = {
        "candidate_count": 56,
        "objective": "mean weighted daily resilience AUC",
        "scenario_count": 5,
        "synthetic_only": True,
    }
    if not isinstance(calibration, dict):
        raise ArtifactError("frozen policy calibration metadata must be an object")
    for field, expected in expected_calibration.items():
        if calibration.get(field) != expected:
            raise ArtifactError(f"frozen policy calibration {field} is invalid")
    objective = calibration.get("winning_objective")
    if isinstance(objective, bool) or not isinstance(objective, int | float):
        raise ArtifactError("frozen policy winning objective must be numeric")
    if not math.isfinite(float(objective)):
        raise ArtifactError("frozen policy winning objective must be finite")
    if not isinstance(policy.get("disclosure"), str) or not policy["disclosure"].strip():
        raise ArtifactError("frozen policy disclosure must be non-empty")
    return policy


def load_policy_bundle(
    *, manifest_path: Path | None = None, policy_path: Path | None = None
) -> PolicyBundle:
    manifest = _read_json(manifest_path or MANIFEST_PATH, "policy manifest")
    record = _validate_manifest(manifest)

    artifact_path = policy_path or POLICY_PATH
    try:
        payload = artifact_path.read_bytes()
    except OSError as exc:
        raise ArtifactError("frozen policy artifact is missing or unreadable") from exc
    if len(payload) != record["bytes"]:
        raise ArtifactError("frozen policy byte count does not match the manifest")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != record["sha256"]:
        raise ArtifactError("frozen policy checksum does not match the manifest")
    try:
        policy = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ArtifactError("frozen policy artifact is invalid JSON") from exc

    return PolicyBundle(
        content=_validate_policy(policy),
        sha256=actual,
        size_bytes=len(payload),
        license=record["license"],
        relative_path=record["path"],
        source=record["source"],
        manifest_schema_version=manifest["version"],
    )


def load_policy() -> tuple[dict[str, Any], str]:
    bundle = load_policy_bundle()
    return bundle.content, bundle.sha256
