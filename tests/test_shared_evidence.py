from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.shared_evidence import (
    canonical_bytes,
    canonical_hash,
    file_sha256,
    fsync_parent,
    function_source_sha256,
    load_json_object,
    split_contract,
    wilson_interval,
)
def test_canonical_hash_matches_the_published_serialization_contract() -> None:
    value = {"z": [3, 2, 1], "ascii": "yes", "finite": 0.25}
    assert canonical_bytes(value) == b'{"ascii":"yes","finite":0.25,"z":[3,2,1]}'
    assert canonical_hash(value) == (
        "ccc1f14287eca7669052691932dbe00f4b315f78d28e4f3b79c295732e817a8b"
    )


def test_file_and_json_helpers_validate_bytes_and_root(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_sha256(path) == expected
    assert load_json_object(path, "evidence", expected_sha256=expected) == {"ok": True}
    with pytest.raises(ValueError, match="hash mismatch"):
        load_json_object(path, "evidence", expected_sha256="0" * 64)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_json_object(path, "evidence")


def test_wilson_split_and_function_source_helpers(tmp_path: Path) -> None:
    assert wilson_interval(35, 40) == [0.7388788, 0.945405]
    assert wilson_interval(35, 40, digits=10) == [0.7388788016, 0.9454049975]
    with pytest.raises(ValueError, match="0 <= successes"):
        wilson_interval(41, 40)

    families = (SimpleNamespace(id="a"), SimpleNamespace(id="b"))
    assert split_contract("dev", families, (4, 5))["cartesian_case_count"] == 4
    with pytest.raises(ValueError, match="contiguous"):
        split_contract("dev", families, (4, 6))

    source = tmp_path / "sample.py"
    source.write_text("def bound():\n    return 3\n\n", encoding="utf-8")
    expected = hashlib.sha256(b"def bound():\n    return 3\n").hexdigest()
    assert function_source_sha256(tmp_path, "sample.py", "bound") == expected
    fsync_parent(source)


def test_json_loader_supports_domain_specific_errors(tmp_path: Path) -> None:
    class DomainError(RuntimeError):
        pass

    missing = tmp_path / "missing.json"
    with pytest.raises(DomainError, match="missing or invalid"):
        load_json_object(missing, "domain evidence", error_type=DomainError)


def test_canonical_bytes_rejects_nonfinite_json() -> None:
    with pytest.raises(ValueError):
        canonical_bytes({"bad": float("nan")})
    assert json.loads(canonical_bytes({"ok": 1})) == {"ok": 1}
