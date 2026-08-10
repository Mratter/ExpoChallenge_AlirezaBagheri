from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "artifacts" / "city_recovery_ppo.v3.metadata.json"
SOURCE_SEAL = ROOT / "training" / "v3" / "source-seal.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_v3_scientific_sources_and_sealing_tool_are_byte_identical() -> None:
    assert _sha256(METADATA) == (
        "a7a5a8a549f05febee0906dc45cc9d73109ab25d9f768167bf7f050c0494c895"
    )
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    source_seal = json.loads(SOURCE_SEAL.read_text(encoding="utf-8"))
    assert metadata["source_identity"] == source_seal["scientific_source"]
    assert metadata["source_identity"]["semantic_sha256"] == (
        "f0fd873a075f86e418eb4841a87f964e090189eca6f04df01255aa5a3b2bb3d9"
    )
    expected = metadata["source_identity"]["files_sha256"]
    assert len(expected) == 13
    assert {relative: _sha256(ROOT / relative) for relative in expected} == expected

    sealing_tool = source_seal["sealing_tool"]
    assert _sha256(ROOT / sealing_tool["path"]) == sealing_tool["sha256"]


def test_duplicate_evidence_helpers_exist_only_in_frozen_boundaries_or_shared_module() -> None:
    allowed = {
        "backend/app/shared_evidence.py",
        "backend/app/simulator_core.py",
        "model/ppo_v3.py",
        "scripts/evaluate_policy_v3.py",
        "scripts/select_policy_v3.py",
        "scripts/train_policy_v3.py",
        "scripts/v3_protocol.py",
    }
    helper_names = {
        "_sha256_file",
        "sha256_file",
        "file_sha256",
        "canonical_hash",
        "_wilson_interval",
        "wilson_interval",
        "wilson_95",
        "_split_contract",
        "split_contract",
        "_load_object",
        "load_json_object",
        "_function_source_sha256",
        "function_source_sha256",
        "_fsync_parent",
        "fsync_parent",
    }
    definitions: dict[str, list[str]] = {}
    for top_level in ("backend", "model", "scripts"):
        for path in (ROOT / top_level).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names:
                    definitions.setdefault(node.name, []).append(relative)
    unexpected = {
        name: paths
        for name, paths in definitions.items()
        if any(path not in allowed for path in paths)
    }
    assert unexpected == {}
    assert "backend/app/shared_evidence.py" in {
        path for paths in definitions.values() for path in paths
    }


def test_v4_path_reuses_frozen_functions_except_the_documented_projection_delta() -> None:
    frozen_paths = (
        ROOT / "backend" / "app" / "simulator_core.py",
        ROOT / "backend" / "app" / "simulator_v2.py",
        ROOT / "backend" / "app" / "simulator_v3.py",
    )
    v4_paths = (
        ROOT / "backend" / "app" / "simulator_core_v4.py",
        ROOT / "backend" / "app" / "simulator_v4.py",
        ROOT / "scripts" / "train_policy_v4.py",
        ROOT / "scripts" / "headroom_probe_v4.py",
    )

    def function_names(paths: tuple[Path, ...]) -> set[str]:
        names: set[str] = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names.update(
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        return names

    assert function_names(frozen_paths) & function_names(v4_paths) == {
        "project_capped_simplex"
    }
