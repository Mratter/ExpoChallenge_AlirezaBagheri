from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from backend.app.artifact import ArtifactError
from backend.app.simulator_v2 import ENGINE_V2_SPEC_SHA256, OBSERVATION_ORDER_V2

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"
PROTOCOL_PATH = ROOT / "evaluation" / "protocol.v2.json"
LEGACY_POLICY_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"
V1_SB3_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.zip"
V1_ONNX_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.onnx"
V1_MODEL_CARD_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.metadata.json"
V1_PARITY_PATH = ROOT / "evaluation" / "policy_parity.v1.json"
SB3_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.zip"
ONNX_POLICY_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.onnx"
MODEL_CARD_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.metadata.json"
PARITY_PATH = ROOT / "evaluation" / "policy_parity.v2.json"

ARTIFACT_LICENSE = "CC0-1.0"
MANIFEST_SCHEMA_VERSION = 3
POLICY_SCHEMA_VERSION = "3.0.0"
POLICY_VERSION = "2.0.0"
POLICY_ID = "city-recovery-sb3-ppo-v2"
POLICY_ARTIFACT_TYPE = "stable_baselines3_ppo"
PARITY_ACTION_TOLERANCE = 1e-5
PARITY_ALLOCATION_TOLERANCE = 1e-4
LEGACY_POLICY_SHA256 = "23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3"
V1_POLICY_ONNX_SHA256 = "983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

ACTION_ORDER = ("transport", "housing", "food", "healthcare", "public_services")
V1_FEATURE_ORDER = (
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
POLICY_FEATURE_ORDER = (
    *V1_FEATURE_ORDER,
    *(f"depot_stock_fraction_{service}" for service in ACTION_ORDER),
    *(f"throughput_factor_{service}" for service in ACTION_ORDER),
)
if POLICY_FEATURE_ORDER != tuple(OBSERVATION_ORDER_V2):
    raise RuntimeError("artifact-v2 observation order drifted from CityRecoveryEnvV2")

# These released files are immutable inputs to the additive v2 bundle. The
# builder and loader both pin them so manifest-v3 cannot silently bless drift.
V1_PROVENANCE_SHA256 = {
    "artifacts/city_recovery_ppo.v1.metadata.json": (
        "becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e"
    ),
    "artifacts/city_recovery_ppo.v1.onnx": (
        "983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8"
    ),
    "artifacts/city_recovery_ppo.v1.zip": (
        "f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33"
    ),
    "artifacts/frozen_policy.v1.json": LEGACY_POLICY_SHA256,
    "evaluation/feature_complete_report.v1.json": (
        "fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351"
    ),
    "evaluation/gate2-evidence.json": (
        "82aa655ecff8c91db99d8db72ec561955ca23badf86689285424a6e90a5c74df"
    ),
    "evaluation/policy_parity.v1.json": (
        "20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82"
    ),
    "evaluation/protocol.v1.json": (
        "b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655"
    ),
}

RECORD_CONTRACT: dict[str, dict[str, Any]] = {
    "accepted-linear-candidate-v1": {
        "path": "artifacts/frozen_policy.v1.json",
        "role": "accepted_legacy_linear_candidate",
        "source": "scripts/build_policy_artifact.py",
        "bytes": 722,
        "sha256": LEGACY_POLICY_SHA256,
    },
    "city-recovery-ppo-v1-checkpoint": {
        "path": "artifacts/city_recovery_ppo.v1.zip",
        "role": "training_checkpoint",
        "source": "scripts/train_policy.py",
        "bytes": 80181,
        "sha256": "f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33",
    },
    "city-recovery-ppo-v1-onnx": {
        "path": "artifacts/city_recovery_ppo.v1.onnx",
        "role": "runtime_policy",
        "source": "scripts/train_policy.py",
        "bytes": 10469,
        "sha256": V1_POLICY_ONNX_SHA256,
    },
    "city-recovery-ppo-v1-metadata": {
        "path": "artifacts/city_recovery_ppo.v1.metadata.json",
        "role": "policy_metadata",
        "source": "scripts/train_policy.py",
        "bytes": 2530,
        "sha256": "becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e",
    },
    "city-recovery-ppo-v1-parity": {
        "path": "evaluation/policy_parity.v1.json",
        "role": "pytorch_onnx_parity_evidence",
        "source": "scripts/train_policy.py",
        "bytes": 631,
        "sha256": "20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82",
    },
    "city-recovery-ppo-v2-checkpoint": {
        "path": "artifacts/city_recovery_ppo.v2.zip",
        "role": "training_checkpoint",
        "source": "scripts/train_policy_v2.py",
    },
    "city-recovery-ppo-v2-onnx": {
        "path": "artifacts/city_recovery_ppo.v2.onnx",
        "role": "runtime_policy",
        "source": "scripts/train_policy_v2.py",
    },
    "city-recovery-ppo-v2-metadata": {
        "path": "artifacts/city_recovery_ppo.v2.metadata.json",
        "role": "policy_metadata",
        "source": "scripts/train_policy_v2.py",
    },
    "city-recovery-ppo-v2-parity": {
        "path": "evaluation/policy_parity.v2.json",
        "role": "pytorch_onnx_parity_evidence",
        "source": "scripts/train_policy_v2.py",
    },
}


@dataclass(frozen=True)
class PolicyBundle:
    metadata: dict[str, Any]
    session: ort.InferenceSession
    onnx_sha256: str
    sb3_sha256: str
    metadata_sha256: str
    parity_sha256: str
    protocol_sha256: str
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
    if set(manifest) != {"active_policy_id", "artifacts", "project", "version"}:
        raise ArtifactError("policy manifest root fields are invalid")
    if manifest.get("project") != "AI17":
        raise ArtifactError("policy manifest project must be AI17")
    if manifest.get("version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(
            f"policy manifest schema version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("active_policy_id") != POLICY_ID:
        raise ArtifactError(f"policy manifest active_policy_id must be {POLICY_ID}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(RECORD_CONTRACT):
        raise ArtifactError("policy manifest must contain the complete nine-artifact bundle")
    if [record.get("id") if isinstance(record, dict) else None for record in artifacts] != list(
        RECORD_CONTRACT
    ):
        raise ArtifactError("policy manifest artifact order is invalid")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ArtifactError("policy manifest artifact record is invalid")
        if set(record) != {"bytes", "id", "license", "path", "role", "sha256", "source"}:
            raise ArtifactError("policy manifest artifact record fields are invalid")
        artifact_id = record["id"]
        if artifact_id in records or artifact_id not in RECORD_CONTRACT:
            raise ArtifactError(f"policy manifest artifact id is invalid: {artifact_id}")
        contract = RECORD_CONTRACT[artifact_id]
        for field in ("path", "role", "source"):
            if record.get(field) != contract[field]:
                raise ArtifactError(
                    f"policy manifest {artifact_id} {field} must be {contract[field]}"
                )
        if record.get("license") != ARTIFACT_LICENSE:
            raise ArtifactError(f"policy manifest {artifact_id} license is invalid")
        if type(record.get("bytes")) is not int or record["bytes"] <= 0:
            raise ArtifactError(f"policy manifest {artifact_id} byte count is invalid")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ArtifactError(f"policy manifest {artifact_id} sha256 is invalid")
        for pinned_field in ("bytes", "sha256"):
            if pinned_field in contract and record.get(pinned_field) != contract[pinned_field]:
                raise ArtifactError(
                    f"policy manifest {artifact_id} released {pinned_field} changed"
                )
        records[artifact_id] = record
    if set(records) != set(RECORD_CONTRACT):
        raise ArtifactError("policy manifest artifact set is incomplete")
    return records


def _default_paths() -> dict[str, Path]:
    return {
        "accepted-linear-candidate-v1": LEGACY_POLICY_PATH,
        "city-recovery-ppo-v1-checkpoint": V1_SB3_POLICY_PATH,
        "city-recovery-ppo-v1-onnx": V1_ONNX_POLICY_PATH,
        "city-recovery-ppo-v1-metadata": V1_MODEL_CARD_PATH,
        "city-recovery-ppo-v1-parity": V1_PARITY_PATH,
        "city-recovery-ppo-v2-checkpoint": SB3_POLICY_PATH,
        "city-recovery-ppo-v2-onnx": ONNX_POLICY_PATH,
        "city-recovery-ppo-v2-metadata": MODEL_CARD_PATH,
        "city-recovery-ppo-v2-parity": PARITY_PATH,
    }


def _verified_payload(record: dict[str, Any], path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(
            f"required policy artifact is missing or unreadable: {record['path']}"
        ) from exc
    if len(payload) != record["bytes"]:
        raise ArtifactError(f"policy artifact byte count drifted: {record['path']}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise ArtifactError(f"policy artifact checksum drifted: {record['path']}")
    return payload, digest


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtifactError(f"{label} must be finite")
    return numeric


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


def _validate_protocol(protocol: Any, digest: str) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "2.0.0":
        raise ArtifactError("v2 evaluation protocol schema is invalid")
    candidate = protocol.get("candidate")
    if not isinstance(candidate, dict):
        raise ArtifactError("v2 evaluation protocol candidate contract is missing")
    if candidate.get("id") != POLICY_ID or candidate.get("observation_count") != 33:
        raise ArtifactError("v2 evaluation protocol candidate contract is invalid")
    if candidate.get("action_space_unchanged") is not True:
        raise ArtifactError("v2 evaluation protocol changed the action space")
    environment = protocol.get("environment")
    if not isinstance(environment, dict):
        raise ArtifactError("v2 evaluation protocol environment contract is missing")
    if environment.get("id") != "city-recovery-env-v2":
        raise ArtifactError("v2 evaluation protocol environment id is invalid")
    if environment.get("spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise ArtifactError("v2 evaluation protocol engine specification drifted")
    parity = protocol.get("parity_regimen")
    if not isinstance(parity, dict):
        raise ArtifactError("v2 evaluation protocol parity regimen is missing")
    if parity.get("action_absolute_tolerance") != PARITY_ACTION_TOLERANCE:
        raise ArtifactError("v2 evaluation protocol action tolerance drifted")
    if parity.get("projected_allocation_absolute_tolerance") != PARITY_ALLOCATION_TOLERANCE:
        raise ArtifactError("v2 evaluation protocol allocation tolerance drifted")
    if parity.get("onnx_opset") != 17 or parity.get("observation_case_count") != 32:
        raise ArtifactError("v2 evaluation protocol parity dimensions are invalid")
    if not SHA256_PATTERN.fullmatch(digest):
        raise ArtifactError("v2 evaluation protocol checksum is invalid")


def _validate_parity(
    report: Any, hashes: dict[str, str], protocol_sha256: str
) -> None:
    if not isinstance(report, dict) or report.get("schema_version") != "2.0.0":
        raise ArtifactError("v2 policy parity report schema is invalid")
    if report.get("passed") is not True or report.get("cases") != 32:
        raise ArtifactError("v2 policy parity report did not pass exactly 32 cases")
    if report.get("observation_count") != len(POLICY_FEATURE_ORDER):
        raise ArtifactError("v2 policy parity observation count is invalid")
    if report.get("environment") != "CityRecoveryEnv-v2":
        raise ArtifactError("v2 policy parity environment is invalid")
    if report.get("onnx_sha256") != hashes["city-recovery-ppo-v2-onnx"]:
        raise ArtifactError("v2 policy parity ONNX checksum is inconsistent")
    if report.get("sb3_checkpoint_sha256") != hashes["city-recovery-ppo-v2-checkpoint"]:
        raise ArtifactError("v2 policy parity SB3 checksum is inconsistent")
    if report.get("providers") != ["CPUExecutionProvider"]:
        raise ArtifactError("v2 policy parity provider must be CPUExecutionProvider")
    if report.get("engine_spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise ArtifactError("v2 policy parity engine specification drifted")
    if report.get("protocol_sha256") != protocol_sha256:
        raise ArtifactError("v2 policy parity protocol checksum is inconsistent")
    if report.get("onnxruntime_version") != ort.__version__:
        raise ArtifactError("v2 policy parity ONNX Runtime version is inconsistent")
    action_tolerance = _finite_number(report.get("action_tolerance"), "action tolerance")
    projected_tolerance = _finite_number(
        report.get("projected_allocation_tolerance"), "projected allocation tolerance"
    )
    if action_tolerance != PARITY_ACTION_TOLERANCE:
        raise ArtifactError("v2 policy parity action tolerance drifted")
    if projected_tolerance != PARITY_ALLOCATION_TOLERANCE:
        raise ArtifactError("v2 policy parity allocation tolerance drifted")
    action_error = _finite_number(
        report.get("max_action_abs_error"), "v2 action parity error"
    )
    if action_error > action_tolerance:
        raise ArtifactError("v2 policy parity action tolerance is exceeded")
    _finite_number(
        report.get("max_pre_projector_proposal_abs_error"),
        "v2 pre-projector proposal parity error",
    )
    if (
        _finite_number(
            report.get("max_projected_allocation_abs_error"),
            "v2 projected allocation parity error",
        )
        > projected_tolerance
    ):
        raise ArtifactError("v2 policy parity projected allocation tolerance is exceeded")


def _validate_checkpoint(payload: bytes) -> None:
    required_names = {
        "_stable_baselines3_version",
        "data",
        "policy.optimizer.pth",
        "policy.pth",
        "pytorch_variables.pth",
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as checkpoint:
            if checkpoint.testzip() is not None:
                raise ArtifactError("v2 SB3 checkpoint archive contains a corrupt member")
            if not required_names.issubset(checkpoint.namelist()):
                raise ArtifactError("v2 SB3 checkpoint archive is incomplete")
    except ArtifactError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactError("v2 SB3 checkpoint is not a valid archive") from exc


def _validate_metadata(
    metadata: Any,
    hashes: dict[str, str],
    protocol_sha256: str,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ArtifactError("v2 policy metadata root must be an object")
    expected = {
        "action_order": list(ACTION_ORDER),
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "id": POLICY_ID,
        "license": ARTIFACT_LICENSE,
        "observation_order": list(POLICY_FEATURE_ORDER),
        "schema_version": POLICY_SCHEMA_VERSION,
        "version": POLICY_VERSION,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ArtifactError(f"v2 policy metadata {field} is invalid")
    training = metadata.get("training")
    if not isinstance(training, dict):
        raise ArtifactError("v2 policy metadata training provenance is missing")
    if training.get("algorithm") != "PPO" or training.get("library") != "stable-baselines3":
        raise ArtifactError("v2 policy metadata training identity is invalid")
    if training.get("device") != "cpu" or training.get("synthetic_only") is not True:
        raise ArtifactError("v2 policy metadata training execution is invalid")
    if training.get("library_version") != "2.7.0":
        raise ArtifactError("v2 policy metadata training library version is invalid")
    if training.get("environment") != "CityRecoveryEnv-v2":
        raise ArtifactError("v2 policy metadata training environment is invalid")
    training_expected = {
        "family_ids": [
            "train_transit_cascade",
            "train_displacement",
            "train_supply_interrupt",
            "train_health_surge",
        ],
        "scenario_seed_count": 8,
        "scenario_unit_count": 32,
        "seed": 17017,
        "timesteps": 30_000,
    }
    for field, value in training_expected.items():
        if training.get(field) != value:
            raise ArtifactError(f"v2 policy metadata training {field} is invalid")
    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        raise ArtifactError("v2 policy metadata environment provenance is missing")
    environment_expected = {
        "change_log": "ENGINE_V2_CHANGELOG.md",
        "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        "id": "CityRecoveryEnv-v2",
        "observation_count": len(POLICY_FEATURE_ORDER),
        "protocol_path": "evaluation/protocol.v2.json",
        "protocol_sha256": protocol_sha256,
    }
    for field, value in environment_expected.items():
        if environment.get(field) != value:
            raise ArtifactError(f"v2 policy metadata environment {field} is invalid")
    export = metadata.get("export")
    if not isinstance(export, dict):
        raise ArtifactError("v2 policy metadata export contract is missing")
    export_expected = {
        "deterministic": True,
        "format": "ONNX",
        "input_name": "observation",
        "onnx_sha256": hashes["city-recovery-ppo-v2-onnx"],
        "opset": 17,
        "output_name": "action",
        "runtime_provider": "CPUExecutionProvider",
    }
    for field, value in export_expected.items():
        if export.get(field) != value:
            raise ArtifactError(f"v2 policy metadata export {field} is invalid")
    if metadata.get("sb3_checkpoint_sha256") != hashes["city-recovery-ppo-v2-checkpoint"]:
        raise ArtifactError("v2 policy metadata checkpoint checksum is inconsistent")
    parity = metadata.get("parity")
    if not isinstance(parity, dict):
        raise ArtifactError("v2 policy metadata parity provenance is missing")
    if parity.get("report_path") != "evaluation/policy_parity.v2.json":
        raise ArtifactError("v2 policy metadata parity path is invalid")
    if parity.get("report_sha256") != hashes["city-recovery-ppo-v2-parity"]:
        raise ArtifactError("v2 policy metadata parity checksum is inconsistent")
    if parity.get("action_tolerance") != PARITY_ACTION_TOLERANCE:
        raise ArtifactError("v2 policy metadata parity action tolerance drifted")
    if parity.get("projected_allocation_tolerance") != PARITY_ALLOCATION_TOLERANCE:
        raise ArtifactError("v2 policy metadata parity allocation tolerance drifted")
    predecessor = metadata.get("predecessor_policy")
    predecessor_expected = {
        "id": "city-recovery-sb3-ppo-v1",
        "onnx_sha256": V1_POLICY_ONNX_SHA256,
        "preserved": True,
        "version": "1.0.0",
    }
    if predecessor != predecessor_expected:
        raise ArtifactError("v2 policy metadata predecessor provenance is invalid")
    legacy = metadata.get("legacy_candidate")
    if not isinstance(legacy, dict):
        raise ArtifactError("v2 policy metadata legacy candidate provenance is missing")
    if (
        legacy.get("artifact_type") != "deterministic_linear_policy_candidate"
        or legacy.get("id") != "frozen-policy-candidate-v1"
        or legacy.get("sha256") != LEGACY_POLICY_SHA256
        or legacy.get("is_ppo") is not False
    ):
        raise ArtifactError("v2 policy metadata legacy candidate provenance is invalid")
    if metadata.get("v1_bundle_preserved") != dict(sorted(V1_PROVENANCE_SHA256.items())):
        raise ArtifactError("v2 policy metadata v1 preservation provenance is invalid")
    disclosure = metadata.get("disclosure")
    if not isinstance(disclosure, str) or "authored-synthetic" not in disclosure:
        raise ArtifactError("v2 policy metadata synthetic disclosure is missing")
    return metadata


def _create_session(payload: bytes) -> ort.InferenceSession:
    try:
        model = onnx.load_model_from_string(payload)
        onnx.checker.check_model(model)
        default_opsets = [
            item.version
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        ]
        if default_opsets != [17]:
            raise ArtifactError("v2 ONNX policy must use opset 17")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("v2 ONNX policy cannot be parsed by the CPU runtime") from exc
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ArtifactError("v2 ONNX runtime did not remain CPU-only")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "observation" or inputs[0].type != "tensor(float)":
        raise ArtifactError("v2 ONNX policy input schema is invalid")
    if len(outputs) != 1 or outputs[0].name != "action" or outputs[0].type != "tensor(float)":
        raise ArtifactError("v2 ONNX policy output schema is invalid")
    if inputs[0].shape[-1] != len(POLICY_FEATURE_ORDER):
        raise ArtifactError("v2 ONNX policy observation dimension is invalid")
    if outputs[0].shape[-1] != len(ACTION_ORDER):
        raise ArtifactError("v2 ONNX policy action dimension is invalid")
    try:
        result = session.run(
            ["action"],
            {
                "observation": np.zeros(
                    (1, len(POLICY_FEATURE_ORDER)), dtype=np.float32
                )
            },
        )[0]
    except Exception as exc:
        raise ArtifactError("v2 ONNX policy smoke inference failed") from exc
    if np.asarray(result).shape != (1, len(ACTION_ORDER)) or not np.all(np.isfinite(result)):
        raise ArtifactError("v2 ONNX policy smoke inference returned an invalid action")
    return session


def load_policy_bundle(
    *,
    manifest_path: Path | None = None,
    artifact_paths: dict[str, Path] | None = None,
    protocol_path: Path | None = None,
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
    protocol_file = protocol_path or PROTOCOL_PATH
    try:
        protocol_payload = protocol_file.read_bytes()
    except OSError as exc:
        raise ArtifactError("v2 evaluation protocol is missing or unreadable") from exc
    protocol_sha256 = hashlib.sha256(protocol_payload).hexdigest()
    protocol = _read_json_bytes(protocol_payload, "v2 evaluation protocol")
    _validate_protocol(protocol, protocol_sha256)
    parity = _read_json_bytes(
        payloads["city-recovery-ppo-v2-parity"], "v2 policy parity report"
    )
    _validate_checkpoint(payloads["city-recovery-ppo-v2-checkpoint"])
    _validate_parity(parity, hashes, protocol_sha256)
    metadata = _read_json_bytes(
        payloads["city-recovery-ppo-v2-metadata"], "v2 policy metadata"
    )
    metadata = _validate_metadata(metadata, hashes, protocol_sha256)
    session = _create_session(payloads["city-recovery-ppo-v2-onnx"])
    return PolicyBundle(
        metadata=metadata,
        session=session,
        onnx_sha256=hashes["city-recovery-ppo-v2-onnx"],
        sb3_sha256=hashes["city-recovery-ppo-v2-checkpoint"],
        metadata_sha256=hashes["city-recovery-ppo-v2-metadata"],
        parity_sha256=hashes["city-recovery-ppo-v2-parity"],
        protocol_sha256=protocol_sha256,
        legacy_sha256=hashes["accepted-linear-candidate-v1"],
        records=records,
        manifest_schema_version=manifest["version"],
    )


def load_policy() -> PolicyBundle:
    return load_policy_bundle()
