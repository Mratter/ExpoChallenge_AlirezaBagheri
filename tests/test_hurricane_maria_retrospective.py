from __future__ import annotations

import ast
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.shared_evidence import canonical_hash
from scripts import hurricane_maria_retrospective as maria


def _independent_interpolation(
    anchors: list[tuple[int, float]], day: int
) -> float:
    anchors = sorted(anchors)
    if day <= anchors[0][0]:
        return anchors[0][1]
    if day >= anchors[-1][0]:
        return anchors[-1][1]
    left = max(anchor for anchor in anchors if anchor[0] <= day)
    right = min(anchor for anchor in anchors if anchor[0] >= day)
    if left[0] == right[0]:
        return left[1]
    fraction = (day - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])


def test_prepared_contract_is_complete_valid_scenario_and_preplanner() -> None:
    contract = maria.build_prepared_contract()

    maria.validate_prepared_contract(contract)
    assert contract["policy_loaded_during_preparation"] is False
    assert contract["final_split_used"] is False
    assert contract["reconstruction"]["days"] == list(range(31))
    assert len(contract["reconstruction"]["dates"]) == 31
    assert len(contract["scenario"]["name"]) <= 64
    assert contract["tape_contract"] == {
        "day_count": 30,
        "all_days_no_shock": True,
        "public_risk_all_zero": True,
        "maria_encoded_only_as_initial_condition": True,
    }
    assert all(
        len(contract["reconstruction"]["services"][service]) == 31
        for service in maria.SERVICE_ORDER
    )
    assert contract["archived_source_bytes"]["verified_before_freeze"] is False
    with pytest.raises(maria.RetrospectiveError, match="not verified"):
        maria.validate_prepared_contract(contract, require_archive_verified=True)


def test_every_observation_date_conversion_and_decision_is_validated() -> None:
    manifest = maria._read_object(maria.SOURCE_MANIFEST)
    observations = maria._read_object(maria.OBSERVATIONS)
    rows = maria.validate_observations(observations, manifest)
    start = date.fromisoformat(observations["day_zero"])

    assert len({row["id"] for row in rows}) == len(rows)
    assert all(date.fromisoformat(row["date"]) == start + timedelta(days=row["day"]) for row in rows)
    assert all(row["selected"] or row["rejection_reason"] for row in rows)
    selected = {row["id"]: row for row in rows if row["selected"]}
    assert selected["public_cell_day29"]["normalized_value"] == 0.302
    assert selected["public_cell_day30_carry"]["evidence_class"] == "project_estimate"
    assert "rejected_fema_cell_day30" not in selected
    assert selected["healthcare_day30_estimate"]["normalized_value"] == pytest.approx(
        (59 / 69) + ((65 / 67) - (59 / 69)) * (20 / 27), abs=5e-8
    )
    assert selected["food_grocery_day30_estimate"]["normalized_value"] == pytest.approx(
        0.49 + (0.89 - 0.49) * (20 / 37), abs=5e-8
    )


def test_every_component_interpolation_service_mean_and_total_recomputes() -> None:
    observations = maria._read_object(maria.OBSERVATIONS)
    crosswalk = maria._read_object(maria.CROSSWALK)
    reconstruction = maria.reconstruct(observations, crosswalk)
    selected = [row for row in observations["observations"] if row["selected"]]

    for service in maria.SERVICE_ORDER:
        for component, weight in crosswalk["services"][service]["components"].items():
            anchors = [
                (row["day"], row["normalized_value"])
                for row in selected
                if row["service"] == service and row["component"] == component
            ]
            for day in range(31):
                assert reconstruction["components"][service][component][day] == pytest.approx(
                    _independent_interpolation(anchors, day), abs=5e-8
                )
        for day in range(31):
            expected_service = sum(
                weight * reconstruction["components"][service][component][day]
                for component, weight in crosswalk["services"][service]["components"].items()
            )
            assert reconstruction["services"][service][day] == pytest.approx(
                expected_service, abs=5e-8
            )
    for day in range(31):
        expected_total = sum(
            reconstruction["services"][service][day]
            for service in maria.SERVICE_ORDER
        ) / 5
        assert reconstruction["total"][day] == pytest.approx(expected_total, abs=5e-8)


def test_selected_official_marker_days_exclude_project_estimate_anchors() -> None:
    observations = maria._read_object(maria.OBSERVATIONS)["observations"]
    selected = [row for row in observations if row["selected"]]
    contract = maria.build_prepared_contract()
    marker_days = contract["reconstruction"]["observation_days"]
    for service, days in marker_days.items():
        assert days == sorted(
            {
                row["day"]
                for row in selected
                if row["service"] == service
                and row["evidence_class"] == "direct_official_observation"
            }
        )


def test_source_manifest_and_normalized_web_facts_are_reproducible() -> None:
    manifest = maria._read_object(maria.SOURCE_MANIFEST)
    facts = maria._read_object(maria.WEB_FACTS)
    maria.validate_sources(manifest)
    maria.validate_web_facts(manifest, facts)
    fact_map = {fact["source_id"]: fact for fact in facts["facts"]}
    for source in manifest["sources"]:
        archived = source["archive_filename"] is not None
        assert archived == (source["size_bytes"] is not None)
        assert archived == (source["sha256"] is not None)
        if source.get("verified_excerpt_sha256"):
            fact = fact_map[source["id"]]
            assert hashlib.sha256(fact["canonical_fact_text"].encode()).hexdigest() == source[
                "verified_excerpt_sha256"
            ]


def test_archive_verifier_checks_all_identities_size_and_digest(tmp_path: Path) -> None:
    payload = b"official-source-fixture"
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(payload)
    manifest = {
        "sources": [
            {
                "archive_filename": source_path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }
    assert maria.verify_archive_root(manifest, tmp_path) == 1
    source_path.write_bytes(payload + b"drift")
    with pytest.raises(maria.RetrospectiveError, match="size mismatch"):
        maria.verify_archive_root(manifest, tmp_path)


def test_module_never_imports_final_roster_or_evaluator() -> None:
    source_text = Path(maria.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    )
    assert "scripts.evaluate" not in imported_modules
    assert "scripts.publish_final_evaluation_v4" not in imported_modules
    scenario_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "backend.app.city.scenarios"
    ]
    assert all({alias.name for alias in node.names} == {"Shock"} for node in scenario_imports)
    assert "FINAL_FAMILIES" not in source_text
    assert "FINAL_SEEDS" not in source_text
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_prepared_contract"
    )
    assert "load_policy" not in {node.id for node in ast.walk(build) if isinstance(node, ast.Name)}


def test_run_uses_canonical_explicit_tape_helpers_not_custom_environment() -> None:
    source_text = Path(maria.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_retrospective"
    )
    names = {node.id for node in ast.walk(run) if isinstance(node, ast.Name)}
    assert {"rollout_policy", "rollout_baseline"}.issubset(names)
    assert "CityRecoveryEnv" not in names


def test_replay_mode_reruns_both_helpers_and_never_publishes() -> None:
    source_text = Path(maria.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    replay = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "replay_outputs"
    )
    names = {node.id for node in ast.walk(replay) if isinstance(node, ast.Name)}
    calls = {
        node.func.id
        for node in ast.walk(replay)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"rollout_policy", "rollout_baseline"}.issubset(calls)
    assert "publish_create_new_bundle" not in calls
    assert "_write_json" not in calls
    assert "load_policy" in names


def test_no_shock_tape_is_explicit_and_public_risk_free() -> None:
    tape = maria._no_shock_tape()
    assert [item.day for item in tape] == list(range(1, 31))
    assert all(item.type is None and item.severity == 0.0 for item in tape)
    assert all(item.impact == [0.0] * 5 for item in tape)
    assert all(item.public_risk_before == [0.0] * 5 for item in tape)
    assert all(item.public_risk_next == [0.0] * 5 for item in tape)
    assert [item.assessment_tail for item in tape] == [False] * 27 + [True] * 3


def test_publication_is_create_new_and_rolls_back_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.ts"
    maria.publish_create_new_bundle({first: "one", second: "two"})
    assert first.read_text() == "one"
    assert second.read_text() == "two"
    with pytest.raises(maria.RetrospectiveError, match="already exists"):
        maria.publish_create_new_bundle({first: "overwrite"})

    first.unlink()
    second.unlink()
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-publication failure")
        real_link(source, destination)

    monkeypatch.setattr(maria.os, "link", fail_second_link)
    with pytest.raises(maria.RetrospectiveError, match="create-new publication failed"):
        maria.publish_create_new_bundle({first: "one", second: "two"})
    assert not first.exists()
    assert not second.exists()


def test_shipped_artifact_and_canonical_benchmark_identities_are_bound() -> None:
    assert maria.file_sha256(maria.ARTIFACT) == maria.EXPECTED_ARTIFACT_SHA256
    benchmark = maria.load_canonical_benchmark_rows()
    rows = benchmark["rows"]
    assert [(row["solved"], row["total"]) for row in rows] == [
        (182, 200),
        (163, 200),
        (147, 200),
        (139, 200),
        (135, 200),
        (125, 200),
        (72, 200),
    ]
    assert "Privileged" in rows[0]["classification"]
    assert "not a submission baseline" in rows[0]["classification"]
    for evidence in benchmark["evidence"].values():
        path = maria.ROOT / evidence["path"]
        assert maria.file_sha256(path) == evidence["sha256"]


def test_frontend_render_is_stable_across_sorted_receipt_roundtrip() -> None:
    service_values = {service: [0.1] * 31 for service in maria.SERVICE_ORDER}
    series = {"total": [0.1] * 31, "services": service_values}
    receipt = {
        "methodology_label": "fixed test methodology",
        "historical": {
            "dates": [f"2017-09-{day + 1:02d}" for day in range(30)] + ["2017-10-01"],
            "days": list(range(31)),
            "service_order": list(maria.SERVICE_ORDER),
            "service_labels": {service: service for service in maria.SERVICE_ORDER},
            "observation_days": {service: [] for service in maria.SERVICE_ORDER},
            "total": [0.1] * 31,
            "services": service_values,
        },
        "planners": {
            "v4": {"series": series},
            "reactive": {"series": series},
        },
        "receipt_sha256": "0" * 64,
        "synthetic_benchmark": {"rows": []},
    }
    sorted_roundtrip = json.loads(json.dumps(receipt, sort_keys=True))
    assert maria.render_frontend(receipt) == maria.render_frontend(sorted_roundtrip)


@pytest.mark.skipif(not maria.RECEIPT.exists(), reason="retrospective not run yet")
def test_published_receipt_and_generated_outputs_are_self_consistent() -> None:
    maria.verify_outputs()
    receipt = maria._read_object(maria.RECEIPT)
    assert canonical_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ) == receipt["receipt_sha256"]
    assert receipt["final_split_used"] is False
    assert receipt["invariants"]["canonical_explicit_tape_helpers_used"]
