from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"
POLICY_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy() -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        record = next(item for item in manifest["artifacts"] if item["id"] == "frozen-policy-v1")
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        raise ArtifactError("policy manifest is missing or invalid") from exc

    actual = sha256_file(POLICY_PATH)
    if actual != record.get("sha256"):
        raise ArtifactError("frozen policy checksum does not match the manifest")
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError("frozen policy artifact is unreadable") from exc
    if policy.get("artifact_type") != "deterministic_linear_policy_candidate":
        raise ArtifactError("frozen policy artifact type is unsupported")
    return policy, actual

