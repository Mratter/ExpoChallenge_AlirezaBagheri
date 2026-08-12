"""Receipt-only checks for the descriptive final-family results supplement."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.city.scenarios import FINAL_FAMILIES


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_RECEIPT = (
    ROOT / "internal" / "evaluation_runs" / "v4" / "final-evaluation-200.success.json"
)
ORACLE_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "clairvoyant-oracle-200-final.json"
)
FAMILY_REPORT = ROOT / "benchmarks" / "v4" / "final-family-analysis-200.md"

EXPECTED_SHIPPED = {
    "v3_final_food_access": 38,
    "v3_final_coastal_isolation": 34,
    "v3_final_public_health": 34,
    "v3_final_grid_cascade": 31,
    "v3_final_aftershock_corridor": 26,
}
EXPECTED_TUNED = {
    "v3_final_food_access": 38,
    "v3_final_coastal_isolation": 30,
    "v3_final_public_health": 29,
    "v3_final_grid_cascade": 30,
    "v3_final_aftershock_corridor": 20,
}
EXPECTED_TEACHER = {
    "v3_final_food_access": 39,
    "v3_final_coastal_isolation": 26,
    "v3_final_public_health": 30,
    "v3_final_grid_cascade": 28,
    "v3_final_aftershock_corridor": 16,
}
EXPECTED_CONSTRUCTION = {
    "v3_final_food_access": (144, 0.23, 0.31),
    "v3_final_coastal_isolation": (157, 0.28, 0.35),
    "v3_final_public_health": (198, 0.30, 0.36),
    "v3_final_grid_cascade": (168, 0.26, 0.34),
    "v3_final_aftershock_corridor": (136, 0.30, 0.36),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_family_supplement_matches_retained_evidence() -> None:
    """Read receipts and definitions only; never construct or evaluate final cases."""

    success = _load(SUCCESS_RECEIPT)
    oracle = _load(ORACLE_RECEIPT)

    shipped = Counter(
        row["family_id"] for row in success["rows"] if row["solved"]
    )
    tuned = Counter(
        row["family_id"] for row in oracle["rows"] if row["tuned_rule"]["solved"]
    )
    assert dict(shipped) == EXPECTED_SHIPPED
    assert dict(tuned) == EXPECTED_TUNED

    regression = success["bound_evidence"]["regression_gate"]
    assert sum(EXPECTED_TEACHER.values()) == regression[
        "preparedness_teacher_solved_count"
    ] == 139

    construction = {
        family.id: (
            family.budget_center,
            family.shock_probability,
            family.severity_max,
        )
        for family in FINAL_FAMILIES
    }
    assert construction == EXPECTED_CONSTRUCTION


def test_final_family_supplement_keeps_single_use_sources_byte_bound() -> None:
    success = _load(SUCCESS_RECEIPT)
    bound_oracle = success["bound_evidence"]["oracle_receipt"]
    assert bound_oracle["sha256"] == _sha256(ORACLE_RECEIPT)

    source_identity = success["source_identity"]
    for relative_path in (
        "backend/app/city/scenarios.py",
        "scripts/publish_final_evaluation_v4.py",
    ):
        assert source_identity[relative_path] == _sha256(ROOT / relative_path)


def test_final_family_supplement_states_the_observation_without_causal_overclaim() -> None:
    report = FAMILY_REPORT.read_text(encoding="utf-8")
    expected_rows = (
        "| Food access | **38 / 40** | **38 / 40** | 39 / 40 | 144 | 0.23 | 0.31 |",
        "| Coastal isolation | **34 / 40** | 30 / 40 | 26 / 40 | 157 | 0.28 | 0.35 |",
        "| Public health | **34 / 40** | 29 / 40 | 30 / 40 | 198 | 0.30 | 0.36 |",
        "| Grid cascade | **31 / 40** | 30 / 40 | 28 / 40 | 168 | 0.26 | 0.34 |",
        "| Aftershock corridor | **26 / 40** | 20 / 40 | 16 / 40 | **136** | **0.30** | **0.36** |",
    )
    assert all(row in report for row in expected_rows)
    assert "**+6** cases over the tuned rule" in report
    assert "**+10** over the teacher" in report
    assert "ties the policy exactly at **38 / 40**" in report
    assert "not a causal estimate" in report
    assert "No planner was rerun" in report
