import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.artifact import (
    MANIFEST_PATH,
    ArtifactError,
    load_policy_bundle,
)
from backend.app.main import metadata_payload
from backend.app.preflight import validate_exposed_metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(manifest: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    return next(item for item in manifest["artifacts"] if item["id"] == artifact_id)


def _replace_artifact(
    tmp_path: Path, artifact_id: str, value: Any
) -> tuple[Path, dict[str, Path]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    replacement = tmp_path / f"{artifact_id}.json"
    _write_json(replacement, value)
    payload = replacement.read_bytes()
    record = _record(manifest, artifact_id)
    record["bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, {artifact_id: replacement}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "artifacts/other.onnx", "path"),
        ("bytes", 1, "byte count"),
        ("sha256", "0" * 64, "checksum"),
        ("license", "Proprietary", "license"),
        ("source", "unknown.py", "source"),
        ("role", "fallback", "role"),
    ],
)
def test_manifest_contract_drift_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _record(manifest, "city-recovery-ppo-v1-onnx")[field] = value
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactError, match=message):
        load_policy_bundle(manifest_path=manifest_path)


def test_policy_observation_order_drift_is_rejected_after_integrity_checks(
    tmp_path: Path,
) -> None:
    bundle = load_policy_bundle()
    metadata = copy.deepcopy(bundle.metadata)
    metadata["observation_order"] = list(reversed(metadata["observation_order"]))
    manifest_path, paths = _replace_artifact(
        tmp_path, "city-recovery-ppo-v1-metadata", metadata
    )

    with pytest.raises(ArtifactError, match="observation_order"):
        load_policy_bundle(manifest_path=manifest_path, artifact_paths=paths)


def test_failed_parity_report_is_rejected_after_integrity_checks(tmp_path: Path) -> None:
    report_path = Path("evaluation/policy_parity.v1.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = False
    manifest_path, paths = _replace_artifact(
        tmp_path, "city-recovery-ppo-v1-parity", report
    )

    with pytest.raises(ArtifactError, match="parity report"):
        load_policy_bundle(manifest_path=manifest_path, artifact_paths=paths)


def test_corrupt_checksum_valid_onnx_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    replacement = tmp_path / "policy.onnx"
    replacement.write_bytes(b"not-an-onnx-model")
    record = _record(manifest, "city-recovery-ppo-v1-onnx")
    record["bytes"] = replacement.stat().st_size
    record["sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="parity ONNX|ONNX policy"):
        load_policy_bundle(
            manifest_path=manifest_path,
            artifact_paths={"city-recovery-ppo-v1-onnx": replacement},
        )


@pytest.mark.parametrize("section", ["model", "dataset"])
def test_preflight_rejects_exposed_schema_drift(section: str) -> None:
    bundle = load_policy_bundle()
    metadata = copy.deepcopy(metadata_payload(bundle))
    metadata[section]["schema_version"] = "9.9.9"

    with pytest.raises(RuntimeError, match=f"exposed {section} metadata schema_version"):
        validate_exposed_metadata(metadata, bundle)
