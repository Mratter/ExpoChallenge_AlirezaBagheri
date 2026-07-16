from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"
LEGACY_POLICY_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"
POLICY_PATH = LEGACY_POLICY_PATH
SB3_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.zip"
ONNX_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.onnx"
MODEL_CARD_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.metadata.json"
PARITY_PATH = ROOT / "evaluation" / "policy_parity.v1.json"

ARTIFACT_LICENSE = "CC0-1.0"
MANIFEST_SCHEMA_VERSION = 2
POLICY_SCHEMA_VERSION = "2.0.0"
POLICY_VERSION = "1.0.0"
POLICY_ID = "city-recovery-sb3-ppo-v1"
POLICY_ARTIFACT_TYPE = "stable_baselines3_ppo"
LEGACY_POLICY_SHA256 = "23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3"
POLICY_FEATURE_ORDER = (
    "service_transport",
    "service_housing",
    "service_food",
    "service_healthcare",
    "service_public_services",
    "priority_transport",
    "priority_housing",
    "priority_food",
    "priority_healthcare",
    "priority_public_services",
    "support_transport",
    "support_housing",
    "support_food",
    "support_healthcare",
    "support_public_services",
    "shock_impact_transport",
    "shock_impact_housing",
    "shock_impact_food",
    "shock_impact_healthcare",
    "shock_impact_public_services",
    "available_budget_fraction",
    "horizon_remaining_fraction",
    "shock_severity",
)
ACTION_ORDER = ("transport", "housing", "food", "healthcare", "public_services")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

RECORD_CONTRACT = {
    "accepted-linear-candidate-v1": {
        "path": "artifacts/frozen_policy.v1.json",
        "role": "accepted_legacy_linear_candidate",
        "source": "scripts/build_policy_artifact.py",
    },
    "city-recovery-ppo-v1-checkpoint": {
        "path": "artifacts/city_recovery_ppo.v1.zip",
        "role": "training_checkpoint",
        "source": "scripts/train_policy.py",
    },
    "city-recovery-ppo-v1-onnx": {
        "path": "artifacts/city_recovery_ppo.v1.onnx",
        "role": "runtime_policy",
        "source": "scripts/train_policy.py",
    },
    "city-recovery-ppo-v1-metadata": {
        "path": "artifacts/city_recovery_ppo.v1.metadata.json",
        "role": "policy_metadata",
        "source": "scripts/train_policy.py",
    },
    "city-recovery-ppo-v1-parity": {
        "path": "evaluation/policy_parity.v1.json",
        "role": "pytorch_onnx_parity_evidence",
        "source": "scripts/train_policy.py",
    },
}


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyBundle:
    metadata: dict[str, Any]
    session: ort.InferenceSession
    onnx_sha256: str
    sb3_sha256: str
    metadata_sha256: str
    parity_sha256: str
    legacy_sha256: str
    records: dict[str, dict[str, Any]]
    manifest_schema_version: int


def _read_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ArtifactError(f"{label} is invalid JSON") from exc


def _read_json(path: Path, label: str) -> Any:
    try:
        return _read_json_bytes(path.read_bytes(), label)
    except OSError as exc:
        raise ArtifactError(f"{label} is missing or unreadable") from exc


def _validate_manifest(manifest: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ArtifactError("policy manifest root must be an object")
    if manifest.get("project") != "AI17":
        raise ArtifactError("policy manifest project must be AI17")
    if manifest.get("version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(
            f"policy manifest schema version must be {MANIFEST_SCHEMA_VERSION}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(RECORD_CONTRACT):
        raise ArtifactError("policy manifest must contain the complete five-artifact bundle")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ArtifactError("policy manifest artifact record is invalid")
        artifact_id = record["id"]
        if artifact_id in records or artifact_id not in RECORD_CONTRACT:
            raise ArtifactError(f"policy manifest artifact id is invalid: {artifact_id}")
        contract = RECORD_CONTRACT[artifact_id]
        for field, expected in contract.items():
            if record.get(field) != expected:
                raise ArtifactError(f"policy manifest {artifact_id} {field} must be {expected}")
        if record.get("license") != ARTIFACT_LICENSE:
            raise ArtifactError(f"policy manifest {artifact_id} license is invalid")
        if type(record.get("bytes")) is not int or record["bytes"] <= 0:
            raise ArtifactError(f"policy manifest {artifact_id} byte count is invalid")
        if not isinstance(record.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
            record["sha256"]
        ):
            raise ArtifactError(f"policy manifest {artifact_id} sha256 is invalid")
        records[artifact_id] = record
    if set(records) != set(RECORD_CONTRACT):
        raise ArtifactError("policy manifest artifact set is incomplete")
    return records


def _default_paths() -> dict[str, Path]:
    return {
        "accepted-linear-candidate-v1": LEGACY_POLICY_PATH,
        "city-recovery-ppo-v1-checkpoint": SB3_POLICY_PATH,
        "city-recovery-ppo-v1-onnx": ONNX_POLICY_PATH,
        "city-recovery-ppo-v1-metadata": MODEL_CARD_PATH,
        "city-recovery-ppo-v1-parity": PARITY_PATH,
    }


def _verified_payload(record: dict[str, Any], path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        message = f"required policy artifact is missing or unreadable: {record['path']}"
        raise ArtifactError(message) from exc
    if len(payload) != record["bytes"]:
        raise ArtifactError(f"policy artifact byte count drifted: {record['path']}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise ArtifactError(f"policy artifact checksum drifted: {record['path']}")
    return payload, digest


def _validate_legacy(policy: Any, digest: str) -> None:
    if digest != LEGACY_POLICY_SHA256:
        raise ArtifactError("accepted legacy linear candidate checksum changed")
    if not isinstance(policy, dict):
        raise ArtifactError("accepted legacy linear candidate root is invalid")
    if policy.get("artifact_type") != "deterministic_linear_policy_candidate":
        raise ArtifactError("accepted legacy candidate was relabeled")
    if policy.get("id") != "frozen-policy-candidate-v1":
        raise ArtifactError("accepted legacy candidate identity changed")
    disclosure = policy.get("disclosure")
    if not isinstance(disclosure, str) or "not PPO" not in disclosure:
        raise ArtifactError("accepted legacy candidate non-PPO disclosure is missing")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtifactError(f"{label} must be finite")
    return numeric


def _validate_parity(report: Any, hashes: dict[str, str]) -> None:
    if not isinstance(report, dict) or report.get("schema_version") != "1.0.0":
        raise ArtifactError("policy parity report schema is invalid")
    if report.get("passed") is not True or report.get("cases", 0) < 20:
        raise ArtifactError("policy parity report did not pass enough cases")
    if report.get("onnx_sha256") != hashes["city-recovery-ppo-v1-onnx"]:
        raise ArtifactError("policy parity ONNX checksum is inconsistent")
    if report.get("sb3_checkpoint_sha256") != hashes["city-recovery-ppo-v1-checkpoint"]:
        raise ArtifactError("policy parity SB3 checksum is inconsistent")
    action_error = _finite_number(
        report.get("max_action_abs_error"), "policy parity action error"
    )
    projected_error = _finite_number(
        report.get("max_projected_allocation_abs_error"),
        "policy parity projected allocation error",
    )
    if action_error > _finite_number(report.get("action_tolerance"), "action tolerance"):
        raise ArtifactError("policy parity action tolerance is exceeded")
    if projected_error > _finite_number(
        report.get("projected_allocation_tolerance"), "projected allocation tolerance"
    ):
        raise ArtifactError("policy parity projected allocation tolerance is exceeded")


def _validate_metadata(
    metadata: Any, hashes: dict[str, str], parity_sha256: str
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ArtifactError("policy metadata root must be an object")
    expected = {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "id": POLICY_ID,
        "schema_version": POLICY_SCHEMA_VERSION,
        "version": POLICY_VERSION,
        "observation_order": list(POLICY_FEATURE_ORDER),
        "action_order": list(ACTION_ORDER),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ArtifactError(f"policy metadata {field} is invalid")
    training = metadata.get("training")
    if not isinstance(training, dict) or training.get("algorithm") != "PPO":
        raise ArtifactError("policy metadata training algorithm must be PPO")
    if training.get("library") != "stable-baselines3":
        raise ArtifactError("policy metadata training library is invalid")
    if training.get("synthetic_only") is not True or training.get("timesteps", 0) < 1:
        raise ArtifactError("policy metadata training provenance is incomplete")
    export = metadata.get("export")
    if not isinstance(export, dict):
        raise ArtifactError("policy metadata export contract is missing")
    if export.get("format") != "ONNX" or export.get("deterministic") is not True:
        raise ArtifactError("policy metadata ONNX export contract is invalid")
    if export.get("input_name") != "observation" or export.get("output_name") != "action":
        raise ArtifactError("policy metadata ONNX tensor names are invalid")
    if export.get("onnx_sha256") != hashes["city-recovery-ppo-v1-onnx"]:
        raise ArtifactError("policy metadata ONNX checksum is inconsistent")
    if metadata.get("sb3_checkpoint_sha256") != hashes["city-recovery-ppo-v1-checkpoint"]:
        raise ArtifactError("policy metadata SB3 checkpoint checksum is inconsistent")
    parity = metadata.get("parity")
    if not isinstance(parity, dict) or parity.get("report_sha256") != parity_sha256:
        raise ArtifactError("policy metadata parity checksum is inconsistent")
    legacy = metadata.get("legacy_candidate")
    if not isinstance(legacy, dict):
        raise ArtifactError("policy metadata legacy candidate disclosure is missing")
    if (
        legacy.get("artifact_type") != "deterministic_linear_policy_candidate"
        or legacy.get("sha256") != LEGACY_POLICY_SHA256
        or legacy.get("is_ppo") is not False
    ):
        raise ArtifactError("policy metadata relabels or changes the legacy candidate")
    disclosure = metadata.get("disclosure")
    if not isinstance(disclosure, str) or "synthetic" not in disclosure.lower():
        raise ArtifactError("policy metadata synthetic disclosure is missing")
    return metadata


def _create_session(payload: bytes) -> ort.InferenceSession:
    try:
        model = onnx.load_model_from_string(payload)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ArtifactError("ONNX policy cannot be parsed by the CPU runtime") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "observation" or inputs[0].type != "tensor(float)":
        raise ArtifactError("ONNX policy input schema is invalid")
    if len(outputs) != 1 or outputs[0].name != "action" or outputs[0].type != "tensor(float)":
        raise ArtifactError("ONNX policy output schema is invalid")
    try:
        result = session.run(
            ["action"], {"observation": np.zeros((1, len(POLICY_FEATURE_ORDER)), dtype=np.float32)}
        )[0]
    except Exception as exc:
        raise ArtifactError("ONNX policy smoke inference failed") from exc
    if np.asarray(result).shape != (1, 5) or not np.all(np.isfinite(result)):
        raise ArtifactError("ONNX policy smoke inference returned an invalid action")
    return session


def load_policy_bundle(
    *,
    manifest_path: Path | None = None,
    artifact_paths: dict[str, Path] | None = None,
) -> PolicyBundle:
    manifest = _read_json(manifest_path or MANIFEST_PATH, "policy manifest")
    records = _validate_manifest(manifest)
    paths = _default_paths()
    if artifact_paths:
        paths.update(artifact_paths)
    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for artifact_id, record in records.items():
        payloads[artifact_id], hashes[artifact_id] = _verified_payload(
            record, paths[artifact_id]
        )

    legacy = _read_json_bytes(
        payloads["accepted-linear-candidate-v1"], "accepted legacy linear candidate"
    )
    _validate_legacy(legacy, hashes["accepted-linear-candidate-v1"])
    parity = _read_json_bytes(
        payloads["city-recovery-ppo-v1-parity"], "policy parity report"
    )
    _validate_parity(parity, hashes)
    metadata = _read_json_bytes(
        payloads["city-recovery-ppo-v1-metadata"], "policy metadata"
    )
    metadata = _validate_metadata(
        metadata, hashes, hashes["city-recovery-ppo-v1-parity"]
    )
    session = _create_session(payloads["city-recovery-ppo-v1-onnx"])
    return PolicyBundle(
        metadata=metadata,
        session=session,
        onnx_sha256=hashes["city-recovery-ppo-v1-onnx"],
        sb3_sha256=hashes["city-recovery-ppo-v1-checkpoint"],
        metadata_sha256=hashes["city-recovery-ppo-v1-metadata"],
        parity_sha256=hashes["city-recovery-ppo-v1-parity"],
        legacy_sha256=hashes["accepted-linear-candidate-v1"],
        records=records,
        manifest_schema_version=manifest["version"],
    )


def load_policy() -> PolicyBundle:
    return load_policy_bundle()
