from __future__ import annotations

import ast
import builtins
import copy
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


def _pre_rebind_fixture_from_published_receipt() -> dict[str, object]:
    """Reverse only the durable correction record for helper-level tests."""

    published = maria._read_object(maria.RECEIPT)
    old = copy.deepcopy(published)
    correction = old.pop("provenance_corrections")[0]
    contract = old["frozen_inputs"]
    old["receipt_sha256"] = correction["previous_receipt_sha256"]
    old["prepared_contract"] = {
        "path": maria.PREPARED_CONTRACT.relative_to(maria.ROOT).as_posix(),
        "file_sha256": correction["previous_output_file_sha256"][
            maria.PREPARED_CONTRACT.relative_to(maria.ROOT).as_posix()
        ],
        "contract_sha256": correction["previous_contract_sha256"],
    }
    manifest = contract["source_manifest_snapshot"]
    sources = {source["id"]: source for source in manifest["sources"]}
    for change in correction["source_manifest_changes"]:
        sources[change["source_id"]][change["field"]] = change["old"]
    observations = contract["observation_table_snapshot"]
    observations.pop("event_observations")
    added_ids = set(
        correction["healthcare_crosswalk_change"][
            "added_project_estimate_observation_ids"
        ]
    )
    observations["observations"] = [
        row for row in observations["observations"] if row["id"] not in added_ids
    ]
    crosswalk = contract["crosswalk_snapshot"]
    crosswalk["services"]["healthcare"] = correction[
        "healthcare_crosswalk_change"
    ]["old"]
    disclosure = next(
        item
        for item in crosswalk["disclosures"]
        if "health-center operational series" in item
    )
    crosswalk["disclosures"].remove(disclosure)
    # Original tracked-input file hashes are deliberately not reconstructible
    # from sorted JSON snapshots. They remain bound by the previous contract and
    # output identities in the durable correction record; helper-level tests do
    # not invoke the strong on-disk pre-publication validator.
    contract["contract_sha256"] = correction["previous_contract_sha256"]
    old["historical"] = copy.deepcopy(published["historical"])
    return old


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
    health_center_rows = [
        row
        for row in selected.values()
        if row["component"] == "health_center_availability"
    ]
    assert [row["day"] for row in health_center_rows] == [0, 10, 30]
    assert all(row["evidence_class"] == "project_estimate" for row in health_center_rows)
    assert all("unavailable-data" in row["conversion"] for row in health_center_rows)
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


@pytest.mark.skipif(not maria.RECEIPT.exists(), reason="retrospective not run yet")
def test_report_component_review_includes_health_center_estimate() -> None:
    receipt = maria._read_object(maria.RECEIPT)
    rendered = maria.render_report(receipt)
    expected_header = (
        "| Day | Date | Transport proxy | Housing proxy | Water | Grocery | "
        "Hospitals | Health centers (estimate) | Power | Cell sites |"
    )
    assert expected_header in rendered

    historical = receipt["historical"]
    for day in range(31):
        health_center = historical["components"]["healthcare"][
            "health_center_availability"
        ][day]
        hospital = historical["components"]["healthcare"][
            "operational_hospital_availability"
        ][day]
        expected_fragment = f"| {hospital:.8f} | {health_center:.8f} |"
        assert expected_fragment in rendered


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
    sources = {source["id"]: source for source in manifest["sources"]}
    assert sources["nhc_maria_tcr"]["publication_date"] == "2023-01-04"
    assert sources["nhc_maria_tcr"]["locators"][0] == (
        "page 2, Puerto Rico landfall chronology"
    )
    assert sources["fcc_2017_10_19"]["url"].endswith("DOC-347339A1.pdf")
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


def test_landfall_timing_is_a_machine_readable_nonservice_observation() -> None:
    observations = maria._read_object(maria.OBSERVATIONS)
    assert observations["event_observations"] == [
        {
            "id": "maria_puerto_rico_landfall",
            "date": "2017-09-20",
            "time_utc": "10:15:00",
            "time_local": "06:15:00",
            "local_timezone": "Atlantic Standard Time (UTC-04:00)",
            "location": "southeast coast of Puerto Rico near Yabucoa",
            "source_ids": ["nhc_maria_tcr"],
            "locator": "PDF page 2, Synoptic History, Puerto Rico landfall paragraph",
            "raw_value": "1015 UTC 20 September 2017",
            "raw_units": (
                "date-time when Maria's center crossed the southeast coast of Puerto Rico"
            ),
            "conversion": "1015 UTC - 4 hours = 0615 Atlantic Standard Time",
            "evidence_class": "direct_official_observation",
            "selected": True,
            "reconstruction_role": (
                "event-window anchor only; not folded into any service index"
            ),
        }
    ]


def test_healthcare_crosswalk_explicitly_covers_hospitals_and_health_centers() -> None:
    crosswalk = maria._read_object(maria.CROSSWALK)
    healthcare = crosswalk["services"]["healthcare"]
    assert healthcare["components"] == {
        "operational_hospital_availability": 0.5,
        "health_center_availability": 0.5,
    }
    assert "unavailable-data project estimate" in healthcare["interpretation"]
    assert "not a measured health-center percentage" in healthcare["interpretation"]


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


def test_official_urls_and_locators_are_semantically_bound_to_archived_evidence() -> None:
    manifest = maria._read_object(maria.SOURCE_MANIFEST)
    observations = maria._read_object(maria.OBSERVATIONS)
    sources = {source["id"]: source for source in manifest["sources"]}

    assert sources["fcc_2017_10_19"] == {
        **sources["fcc_2017_10_19"],
        "url": "https://docs.fcc.gov/public/attachments/DOC-347339A1.pdf",
        "archive_filename": "fcc-2017-10-19.pdf",
        "size_bytes": 488641,
        "sha256": "24028ef1db142c21938e26905231efc4e2eb74dce311307e66c1d2ab665b4927",
    }
    nhc = sources["nhc_maria_tcr"]
    assert nhc["publication_date"] == "2023-01-04"
    assert nhc["locators"][0] == "page 2, Puerto Rico landfall chronology"
    event = observations["event_observations"][0]
    assert event["source_ids"] == ["nhc_maria_tcr"]
    assert event["locator"].startswith("PDF page 2")
    assert event["raw_value"] == "1015 UTC 20 September 2017"

    selected = [row for row in observations["observations"] if row["selected"]]
    for row in selected:
        assert row["locator"].strip()
        assert row["raw_units"].strip()
        if row["evidence_class"] == "project_estimate":
            assert row["conversion"] != "not used"
        else:
            assert row["denominator"] is not None
            assert row["conversion"] != "not used"


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


def test_replace_exact_bundle_rejects_drift_symlinks_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.ts"
    first.write_text("old-one", encoding="utf-8")
    second.write_text("old-two", encoding="utf-8")
    identities = {path: maria.file_sha256(path) for path in (first, second)}

    drifted = dict(identities)
    drifted[first] = "0" * 64
    with pytest.raises(maria.RetrospectiveError, match="drifted before staging"):
        maria.publish_replace_exact_bundle(
            {first: "new-one", second: "new-two"}, drifted
        )
    assert first.read_text(encoding="utf-8") == "old-one"
    assert second.read_text(encoding="utf-8") == "old-two"

    symlink = tmp_path / "linked.json"
    try:
        symlink.symlink_to(first)
    except OSError:
        pass
    else:
        with pytest.raises(maria.RetrospectiveError, match="missing or unsafe"):
            maria.publish_replace_exact_bundle(
                {symlink: "new"}, {symlink: maria.file_sha256(symlink)}
            )

    real_replace = os.replace
    calls = 0

    def fail_second_new_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        # Staged replacements end in .rebind-new; rollback sources end in old.
        if str(source).endswith(".rebind-new"):
            calls += 1
            if calls == 2:
                raise OSError("injected second replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(maria.os, "replace", fail_second_new_replace)
    with pytest.raises(maria.RetrospectiveError, match="publication failed"):
        maria.publish_replace_exact_bundle(
            {first: "new-one", second: "new-two"}, identities
        )
    assert first.read_text(encoding="utf-8") == "old-one"
    assert second.read_text(encoding="utf-8") == "old-two"
    assert maria.file_sha256(first) == identities[first]
    assert maria.file_sha256(second) == identities[second]


def test_rebind_requires_exact_expected_corrections_and_retains_results() -> None:
    old_receipt = _pre_rebind_fixture_from_published_receipt()
    corrected = maria.build_prepared_contract()
    rebound = maria.build_provenance_rebound_receipt(
        old_receipt, corrected, maria.PRE_REBIND_IDENTITIES
    )

    assert rebound["artifact"] == old_receipt["artifact"]
    assert rebound["tape"] == old_receipt["tape"]
    assert rebound["planners"] == old_receipt["planners"]
    assert rebound["invariants"] == old_receipt["invariants"]
    assert {
        planner: rebound["planners"][planner]["summary"]["trajectory_sha256"]
        for planner in ("v4", "reactive")
    } == {
        "v4": "cf20c8bf17dd671bc517e3395170ff4c7c7bac527c0ed2bab42ba6eaf6074f32",
        "reactive": "157551ed5bf86d036798e7d1822309d68441cdadac697f6253c320cb8d480c17",
    }
    correction = rebound["provenance_corrections"][0]
    assert correction["execution"] == {
        "planner_rerun": False,
        "policy_loaded": False,
        "rollout_helpers_called": False,
        "statement": (
            "No planner was rerun; this rebind corrects provenance metadata "
            "and a numerically neutral crosswalk disclosure only."
        ),
    }
    assert not any(correction["numerical_effect"].values())

    mutated = copy.deepcopy(corrected)
    mutated["reconstruction"]["total"][0] += 0.001
    with pytest.raises(maria.RetrospectiveError, match="historical total"):
        maria.build_provenance_rebound_receipt(
            old_receipt, mutated, maria.PRE_REBIND_IDENTITIES
        )


def test_rebind_code_has_no_policy_or_rollout_helper_dependency() -> None:
    source_text = Path(maria.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    rebind_function_names = {
        "_validate_embedded_pre_rebind_contract",
        "_planner_series_without_helpers",
        "validate_pre_rebind_publication",
        "_assert_expected_provenance_changes",
        "_provenance_correction_record",
        "build_provenance_rebound_receipt",
        "rebind_provenance",
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in rebind_function_names
    }
    assert set(functions) == rebind_function_names
    forbidden = {
        "load_policy",
        "rollout_policy",
        "rollout_baseline",
        "summarize_trajectory",
        "_planner_payload",
        "run_retrospective",
        "replay_outputs",
        "verify_outputs",
        "validate_receipt",
    }
    for name, function in functions.items():
        referenced = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
        calls = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (forbidden & (referenced | calls)), name


def test_pre_rebind_validation_and_rebound_builder_bomb_policy_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_receipt = _pre_rebind_fixture_from_published_receipt()
    corrected = maria.build_prepared_contract()
    real_import = builtins.__import__
    forbidden_prefixes = (
        "model.policy",
        "backend.app.city.environment",
        "backend.app.city.outcome",
    )

    def bomb_planner_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith(forbidden_prefixes):
            raise AssertionError(f"planner dependency imported during rebind: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", bomb_planner_import)
    rebound = maria.build_provenance_rebound_receipt(
        old_receipt, corrected, maria.PRE_REBIND_IDENTITIES
    )
    assert rebound["planners"] == old_receipt["planners"]


def test_rebind_rejects_unapproved_source_or_planner_mutation() -> None:
    old_receipt = _pre_rebind_fixture_from_published_receipt()
    corrected = maria.build_prepared_contract()
    unexpected_source = copy.deepcopy(corrected)
    unexpected_source["source_manifest_snapshot"]["sources"][1]["title"] += " drift"
    with pytest.raises(maria.RetrospectiveError, match="unexpected source drift"):
        maria.build_provenance_rebound_receipt(
            old_receipt, unexpected_source, maria.PRE_REBIND_IDENTITIES
        )

    mutated_old_receipt = copy.deepcopy(old_receipt)
    mutated_old_receipt["planners"]["v4"]["series"]["total"][0] += 0.001
    # The strong published-bundle validator catches this through its exact file
    # identity in production; the helper-level check still refuses a forged
    # retained planner result if given one directly.
    with pytest.raises(maria.RetrospectiveError, match="historical|evidence"):
        maria.build_provenance_rebound_receipt(
            mutated_old_receipt, corrected, maria.PRE_REBIND_IDENTITIES
        )


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
        "scenario": {"name": "fixed test scenario"},
        "invariants": {"observation_count": 73, "action_count": 22},
        "receipt_sha256": "0" * 64,
        "synthetic_benchmark": {
            "rows": [
                {
                    "classification": "Test evidence",
                    "detail": "Stable-render fixture.",
                    "id": "test",
                    "label": "Test planner",
                    "rate": 0.005,
                    "solved": 1,
                    "total": 200,
                }
            ]
        },
    }
    sorted_roundtrip = json.loads(json.dumps(receipt, sort_keys=True))
    assert maria.render_frontend(receipt) == maria.render_frontend(sorted_roundtrip)


@pytest.mark.skipif(not maria.RECEIPT.exists(), reason="retrospective not run yet")
def test_frontend_payload_derives_landing_metadata_from_frozen_receipt() -> None:
    receipt = maria._read_object(maria.RECEIPT)
    payload = maria.frontend_payload(receipt)
    historical = receipt["historical"]

    assert payload["display"] == {
        "milestoneDays": [0, 10, 20, 30],
        "dayZeroLabel": "Sep 20",
        "dayEndLabel": "Oct 20, 2017",
        "horizonStart": historical["days"][0],
        "dayEnd": historical["days"][-1],
        "dayCount": len(historical["days"]),
        "indexMin": 0,
        "indexMax": 100,
    }
    assert payload["scenarioCount"] == 1
    benchmark_totals = {
        row["total"] for row in receipt["synthetic_benchmark"]["rows"]
    }
    assert benchmark_totals == {payload["syntheticBenchmarkCaseCount"]}
    assert payload["interface"] == {
        "observationCount": receipt["invariants"]["observation_count"],
        "actionCount": receipt["invariants"]["action_count"],
    }


@pytest.mark.skipif(not maria.RECEIPT.exists(), reason="retrospective not run yet")
def test_published_receipt_and_generated_outputs_are_self_consistent() -> None:
    maria.verify_outputs()
    receipt = maria._read_object(maria.RECEIPT)
    assert canonical_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ) == receipt["receipt_sha256"]
    assert receipt["final_split_used"] is False
    assert receipt["invariants"]["canonical_explicit_tape_helpers_used"]
