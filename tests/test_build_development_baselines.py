from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.shared_evidence import canonical_bytes, file_sha256, wilson_interval
from scripts.build_development_baselines import (
    BaselineError,
    _write_new,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "development-baselines-200.json"
)
MARKDOWN = ROOT / "benchmarks" / "v4" / "development-baselines-200.md"


def test_committed_200_case_development_evidence_is_complete_and_current() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert RECEIPT.read_bytes() == canonical_bytes(payload)
    assert payload["status"] == "development_baselines_200_nonauthorizing"
    assert payload["split"] == "dev"
    assert payload["final_split_used"] is False
    assert payload["case_count"] == 200
    assert payload["split_contract"]["cartesian_case_count"] == 200
    assert all(payload["invariants"].values())

    expected_solves = {
        "heuristic": 91,
        "teacher": 151,
        "tuned": 160,
        "onnx:tests/fixtures/legacy_policy.onnx": 141,
    }
    assert {
        policy["id"]: policy["solved_count"] for policy in payload["policies"]
    } == expected_solves
    for policy in payload["policies"]:
        assert policy["case_count"] == 200
        assert policy["hard_violation_count"] == 0
        assert policy["maximum_conservation_residual"] == 0.0
        assert policy["wilson_95"] == wilson_interval(
            policy["solved_count"], 200, digits=10
        )

    ordered_ids: list[str] | None = None
    ordered_tapes: list[int] | None = None
    for policy_id in payload["policy_order"]:
        rows = payload["rows"][policy_id]
        assert len(rows) == 200
        assert len({row["row_id"] for row in rows}) == 200
        ids = [row["row_id"] for row in rows]
        tapes = [row["tape_seed"] for row in rows]
        if ordered_ids is None:
            ordered_ids, ordered_tapes = ids, tapes
        else:
            assert ids == ordered_ids
            assert tapes == ordered_tapes

    for relative_path, expected_sha256 in payload["source_identity"].items():
        assert file_sha256(ROOT / relative_path) == expected_sha256

    historical = payload["historical_40_case_evidence"]
    assert historical["scope"] == "original_40_case_development_subset"
    assert historical["case_count"] == 40
    assert historical["privileged_oracle"]["solved_count"] == 37
    assert historical["privileged_oracle"]["case_count"] == 40
    for record in (historical["receipt"], historical["markdown"]):
        assert file_sha256(ROOT / record["path"]) == record["sha256"]

    assert MARKDOWN.read_text(encoding="utf-8") == render_markdown(payload)
    assert "37/40 on that original subset only" in MARKDOWN.read_text(
        encoding="utf-8"
    )


def test_evidence_writer_is_create_new(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    _write_new(output, b"first")
    assert output.read_bytes() == b"first"

    with pytest.raises(BaselineError, match="refusing to overwrite"):
        _write_new(output, b"second")
    assert output.read_bytes() == b"first"
