import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.artifact import (
    MANIFEST_PATH,
    POLICY_PATH,
    ArtifactError,
    load_policy_bundle,
)
from backend.app.main import metadata_payload
from scripts.preflight_check import validate_exposed_metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "artifacts/other.json", "path"),
        ("bytes", 721, "byte count"),
        ("sha256", "0" * 64, "checksum"),
        ("license", "Proprietary", "license"),
        ("source", "unknown.py", "source"),
    ],
)
def test_manifest_contract_drift_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["artifacts"][0][field] = value
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactError, match=message):
        load_policy_bundle(manifest_path=manifest_path, policy_path=POLICY_PATH)


def test_policy_feature_order_drift_is_rejected_after_integrity_checks(
    tmp_path: Path,
) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["feature_order"] = list(reversed(policy["feature_order"]))
    policy_path = tmp_path / "policy.json"
    _write_json(policy_path, policy)
    payload = policy_path.read_bytes()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["bytes"] = len(payload)
    manifest["artifacts"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="feature order"):
        load_policy_bundle(manifest_path=manifest_path, policy_path=policy_path)


@pytest.mark.parametrize("section", ["model", "dataset"])
def test_preflight_rejects_exposed_schema_drift(section: str) -> None:
    bundle = load_policy_bundle()
    metadata = copy.deepcopy(metadata_payload(bundle.content, bundle.sha256))
    metadata[section]["schema_version"] = "9.9.9"

    with pytest.raises(RuntimeError, match=f"exposed {section} metadata schema_version"):
        validate_exposed_metadata(metadata, bundle)
