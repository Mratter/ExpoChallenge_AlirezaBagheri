#!/usr/bin/env python3
"""Build and validate the fixed Hurricane Maria retrospective receipt.

The evidence/reconstruction phase is deliberately independent of planner loading.
Use ``--prepare`` to validate and freeze the historical-data contract. The one-time
``--run`` phase refuses to start unless that exact contract has been prepared.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import canonical_hash, file_sha256  # noqa: E402

SOURCE_MANIFEST = ROOT / "benchmarks/v4/hurricane-maria-source-manifest.json"
OBSERVATIONS = ROOT / "benchmarks/v4/hurricane-maria-observations.json"
CROSSWALK = ROOT / "benchmarks/v4/hurricane-maria-crosswalk.json"
WEB_FACTS = ROOT / "benchmarks/v4/hurricane-maria-web-facts.json"
PREPARED_CONTRACT = ROOT / "internal/retrospectives/hurricane-maria-inputs.json"
RECEIPT = ROOT / "internal/retrospectives/hurricane-maria-30d.json"
REPORT = ROOT / "benchmarks/v4/hurricane-maria-retrospective.md"
FRONTEND = ROOT / "frontend/src/generated/mariaRetrospective.ts"
ARTIFACT = ROOT / "artifacts/city_recovery_ppo.v4.onnx"
FINAL_SUCCESS_EVIDENCE = ROOT / "internal/evaluation_runs/v4/final-evaluation-200.success.json"
ORACLE_EVIDENCE = ROOT / "internal/developmental_runs/v4/clairvoyant-oracle-200-final.json"
REGRESSION_GATE_EVIDENCE = ROOT / "tests/test_consolidation_gate.py"
EXPECTED_ARTIFACT_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)
SERVICE_ORDER = (
    "transport",
    "housing",
    "food",
    "healthcare",
    "public_services",
)
CAPTION = (
    "The historical line is a project-derived index from official records. "
    "Policy lines are simulated alternatives in the frozen model, not observed "
    "or causal real-world outcomes."
)

# This migration is deliberately bound to the one published bundle whose
# provenance metadata needed correction.  It is not a general-purpose way to
# overwrite retrospective evidence, and it cannot be run a second time.
PRE_REBIND_IDENTITIES = {
    PREPARED_CONTRACT: "5702fca129386ae46e92688da10c60280c7bc964bed91b91769b8e0ec645ea14",
    RECEIPT: "643a19eb4803aecce0935b3eb71c462a59ca6a30754d83706a0936038aa72828",
    FRONTEND: "2d9290f56125d0a1077998ea230e3e4aad7767cffeca8ed31b2fcaf8b284de79",
    REPORT: "51f60838cf54b6368891edff3f473e947d9cbb401d9b72e7e8415bea93d1c9cc",
}
PRE_REBIND_RECEIPT_SHA256 = (
    "66eafd97e8336e2ad9e0a6fae1ba11dfe9cf3e03b0f1cc25a0888a24525329d4"
)
PRE_REBIND_CONTRACT_SHA256 = (
    "e52f0f5de499d73e078518c9a42a3a82e305dea67b4b2c18a46c294d7c56950d"
)
PROVENANCE_CORRECTION_ID = "2026-08-13-official-source-provenance-correction"


class RetrospectiveError(RuntimeError):
    """Raised when evidence or the frozen retrospective contract is invalid."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RetrospectiveError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RetrospectiveError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.is_symlink():
        raise RetrospectiveError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_json_text(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass


def _json_text(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"


def _assert_regular_absent(path: Path) -> None:
    if path.is_symlink():
        raise RetrospectiveError(f"publication target must not be a symlink: {path}")
    if path.exists():
        raise RetrospectiveError(f"publication target already exists: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RetrospectiveError(f"publication parent is unsafe: {parent}")


def preflight_publication_targets() -> None:
    """Refuse any existing/symlink target before policy code can be loaded."""

    for path in (RECEIPT, FRONTEND, REPORT):
        _assert_regular_absent(path)


def publish_create_new_bundle(outputs: dict[Path, str]) -> None:
    """Publish a create-new bundle, rolling back every created target on failure."""

    staged: dict[Path, Path] = {}
    created: list[Path] = []
    try:
        for destination, payload in outputs.items():
            _assert_regular_absent(destination)
            descriptor, temporary = tempfile.mkstemp(
                dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
            )
            temporary_path = Path(temporary)
            staged[destination] = temporary_path
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temporary_path in staged.items():
            try:
                os.link(temporary_path, destination)
            except OSError as exc:
                raise RetrospectiveError(
                    f"create-new publication failed for {destination}"
                ) from exc
            created.append(destination)
    except Exception as exc:
        rollback_failures: list[str] = []
        for destination in reversed(created):
            try:
                destination.unlink()
            except OSError as rollback_exc:
                rollback_failures.append(f"{destination}: {rollback_exc}")
        if rollback_failures:
            raise RetrospectiveError(
                "publication failed and rollback was incomplete; manual recovery "
                "requires removing only these create-new targets: "
                + "; ".join(rollback_failures)
            ) from exc
        raise
    finally:
        for temporary_path in staged.values():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def publish_replace_exact_bundle(
    outputs: dict[Path, str], expected_sha256: dict[Path, str]
) -> None:
    """Replace one exact regular-file bundle and roll it all back on failure.

    The caller must bind every destination to its pre-migration digest.  Staged
    payloads and hard-linked backups live beside each destination, so each
    individual replacement is atomic and a later failure can restore the exact
    previous inode contents.
    """

    if set(outputs) != set(expected_sha256) or not outputs:
        raise RetrospectiveError("replacement bundle identities are incomplete")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for destination, payload in outputs.items():
            if destination.is_symlink() or not destination.is_file():
                raise RetrospectiveError(
                    f"replacement target missing or unsafe: {destination}"
                )
            if destination.parent.is_symlink() or not destination.parent.is_dir():
                raise RetrospectiveError(
                    f"replacement parent is unsafe: {destination.parent}"
                )
            if file_sha256(destination) != expected_sha256[destination]:
                raise RetrospectiveError(
                    f"replacement target drifted before staging: {destination}"
                )
            descriptor, temporary = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".rebind-new",
            )
            temporary_path = Path(temporary)
            staged[destination] = temporary_path
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            backup_descriptor, backup = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".rebind-old",
            )
            os.close(backup_descriptor)
            backup_path = Path(backup)
            backup_path.unlink()
            os.link(destination, backup_path)
            backups[destination] = backup_path

        # Detect edits made while payloads and backups were being staged.
        for destination, expected in expected_sha256.items():
            if destination.is_symlink() or file_sha256(destination) != expected:
                raise RetrospectiveError(
                    f"replacement target drifted during staging: {destination}"
                )

        for destination, temporary_path in staged.items():
            os.replace(temporary_path, destination)
            replaced.append(destination)
    except Exception as exc:
        rollback_failures: list[str] = []
        for destination in reversed(replaced):
            backup = backups[destination]
            try:
                os.replace(backup, destination)
            except OSError as rollback_exc:
                rollback_failures.append(f"{destination}: {rollback_exc}")
        if rollback_failures:
            raise RetrospectiveError(
                "provenance rebind failed and rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from exc
        if isinstance(exc, RetrospectiveError):
            raise
        raise RetrospectiveError("provenance rebind publication failed") from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _round(value: float) -> float:
    return round(float(value), 8)


def _interpolate(anchors: Iterable[tuple[int, float]]) -> list[float]:
    ordered = sorted((int(day), float(value)) for day, value in anchors)
    if not ordered or ordered[0][0] < 0 or ordered[-1][0] > 30:
        raise RetrospectiveError("component anchors must span valid Day-0..30 dates")
    if len({day for day, _ in ordered}) != len(ordered):
        raise RetrospectiveError("component anchors contain a duplicate day")
    result: list[float] = []
    for day in range(31):
        if day <= ordered[0][0]:
            value = ordered[0][1]
        elif day >= ordered[-1][0]:
            value = ordered[-1][1]
        else:
            right_index = next(
                index for index, (anchor_day, _) in enumerate(ordered) if day < anchor_day
            )
            left_day, left_value = ordered[right_index - 1]
            right_day, right_value = ordered[right_index]
            fraction = (day - left_day) / (right_day - left_day)
            value = left_value + fraction * (right_value - left_value)
        result.append(_round(value))
    return result


def _validate_conversion(row: dict[str, Any]) -> None:
    conversion = str(row.get("conversion", ""))
    value = float(row["normalized_value"])
    raw = row.get("raw_value")
    if conversion == "not used":
        if row.get("selected") is not False:
            raise RetrospectiveError(f"selected row cannot use 'not used': {row['id']}")
        return
    if conversion == "percent / 100":
        expected = float(raw) / 100.0
    elif conversion == "1 - outage_percent / 100":
        expected = 1.0 - float(raw) / 100.0
    elif re.fullmatch(r"\d+ / \d+", conversion):
        numerator, denominator = (float(part) for part in conversion.split(" / "))
        expected = numerator / denominator
    elif conversion == "0.49 + (0.89 - 0.49) * (20 / 37)":
        expected = 0.49 + (0.89 - 0.49) * (20 / 37)
    elif conversion == "(59/69) + ((65/67) - (59/69)) * (20 / 27)":
        expected = (59 / 69) + ((65 / 67) - (59 / 69)) * (20 / 27)
    elif conversion == "carry Day-2 0.046 backward two days":
        expected = 0.046
    elif conversion == "carry Day-29 0.302 forward one day":
        expected = 0.302
    elif conversion.startswith("fixed "):
        if row.get("evidence_class") != "project_estimate":
            raise RetrospectiveError(
                f"fixed conversion is not marked project estimate: {row['id']}"
            )
        return
    else:
        raise RetrospectiveError(f"unsupported conversion for {row['id']}: {conversion}")
    if abs(expected - value) > 5e-8:
        raise RetrospectiveError(
            f"conversion result differs for {row['id']}: {expected} != {value}"
        )


def validate_observations(
    observations: dict[str, Any],
    manifest: dict[str, Any],
    *,
    require_event_observation: bool = True,
) -> list[dict[str, Any]]:
    rows = observations.get("observations")
    if not isinstance(rows, list) or not rows:
        raise RetrospectiveError("observation table is empty")
    start = date.fromisoformat(observations["day_zero"])
    source_entries = {
        source["id"]: source for source in manifest["sources"]
    }
    event_rows = observations.get("event_observations")
    if require_event_observation and (not isinstance(event_rows, list) or not event_rows):
        raise RetrospectiveError("event observation table is empty")
    if event_rows is None:
        event_rows = []
    if not isinstance(event_rows, list):
        raise RetrospectiveError("event observation table is invalid")
    event_ids: set[str] = set()
    for event in event_rows:
        event_id = event.get("id")
        if not isinstance(event_id, str) or event_id in event_ids:
            raise RetrospectiveError(f"invalid or duplicate event observation id: {event_id}")
        event_ids.add(event_id)
        referenced = event.get("source_ids")
        if not isinstance(referenced, list) or not referenced:
            raise RetrospectiveError(f"event observation lacks sources: {event_id}")
        if not set(referenced).issubset(source_entries):
            raise RetrospectiveError(f"unknown event source in {event_id}")
        if not event.get("locator") or not event.get("raw_units"):
            raise RetrospectiveError(f"event observation lacks locator or units: {event_id}")
        if event.get("date") != observations["day_zero"]:
            raise RetrospectiveError(f"event observation does not anchor Day 0: {event_id}")
        if event.get("reconstruction_role") != (
            "event-window anchor only; not folded into any service index"
        ):
            raise RetrospectiveError(f"event observation role is invalid: {event_id}")
    ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            raise RetrospectiveError(f"invalid or duplicate observation id: {row_id}")
        ids.add(row_id)
        if row.get("service") not in SERVICE_ORDER:
            raise RetrospectiveError(f"unknown service in {row_id}")
        referenced = row.get("source_ids")
        if not isinstance(referenced, list) or not referenced:
            raise RetrospectiveError(f"observation lacks sources: {row_id}")
        if not set(referenced).issubset(source_entries):
            raise RetrospectiveError(f"unknown source in {row_id}")
        day = int(row.get("day", -1))
        if not 0 <= day <= 30:
            raise RetrospectiveError(f"invalid day in {row_id}")
        if date.fromisoformat(row["date"]) != start + timedelta(days=day):
            raise RetrospectiveError(f"date/day mismatch in {row_id}")
        value = row.get("normalized_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RetrospectiveError(f"invalid normalized value in {row_id}")
        if not 0 <= float(value) <= 1:
            raise RetrospectiveError(f"normalized value out of range in {row_id}")
        if not row.get("locator") or not row.get("raw_units"):
            raise RetrospectiveError(f"observation lacks locator or units: {row_id}")
        if row.get("evidence_class") != "project_estimate" and row.get("denominator") is None:
            raise RetrospectiveError(f"official observation lacks denominator: {row_id}")
        if row.get("selected") is False and not row.get("rejection_reason"):
            raise RetrospectiveError(f"rejected alternative lacks reason: {row_id}")
        if row.get("selected") is True:
            for source_id in referenced:
                source = source_entries[source_id]
                if source.get("sha256") is None and len(
                    str(source.get("verified_excerpt_sha256", ""))
                ) != 64:
                    raise RetrospectiveError(
                        f"selected web source {source_id} lacks a fact hash"
                    )
        _validate_conversion(row)
    selected = [row for row in rows if row.get("selected") is True]
    if not selected or not any(row.get("selected") is False for row in rows):
        raise RetrospectiveError("observations need selected and rejected alternatives")
    for service in SERVICE_ORDER:
        components = {
            row["component"] for row in selected if row["service"] == service
        }
        for component in components:
            days = {
                int(row["day"])
                for row in selected
                if row["service"] == service and row["component"] == component
            }
            if 0 not in days or 30 not in days:
                raise RetrospectiveError(
                    f"selected component lacks Day-0/30 anchors: {service}/{component}"
                )
    return rows


def validate_sources(manifest: dict[str, Any]) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) < 10:
        raise RetrospectiveError("source manifest is incomplete")
    ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise RetrospectiveError("source manifest entry is invalid")
        source_id = source["id"]
        if source_id in ids:
            raise RetrospectiveError(f"duplicate source id: {source_id}")
        ids.add(source_id)
        for key in ("agency", "title", "publication_date", "url", "locators"):
            if not source.get(key):
                raise RetrospectiveError(f"source {source_id} lacks {key}")
        archived = source.get("archive_filename") is not None
        archive_fields = (
            source.get("size_bytes") is not None,
            source.get("sha256") is not None,
        )
        if archived != all(archive_fields):
            raise RetrospectiveError(
                f"source {source_id} has a partial archive identity"
            )
        if archived:
            if source["size_bytes"] <= 0 or len(source["sha256"]) != 64:
                raise RetrospectiveError(f"source {source_id} archive identity invalid")
        elif not source.get("retrieval_status"):
            raise RetrospectiveError(
                f"source {source_id} has neither archive identity nor retrieval status"
            )


def validate_web_facts(manifest: dict[str, Any], web_facts: dict[str, Any]) -> None:
    facts = web_facts.get("facts")
    if not isinstance(facts, list):
        raise RetrospectiveError("normalized web-fact table is invalid")
    by_source = {fact.get("source_id"): fact for fact in facts}
    if len(by_source) != len(facts):
        raise RetrospectiveError("normalized web-fact source ids must be unique")
    for source in manifest["sources"]:
        expected = source.get("verified_excerpt_sha256")
        if expected is None:
            continue
        fact = by_source.get(source["id"])
        if fact is None:
            raise RetrospectiveError(f"normalized fact missing for {source['id']}")
        text = fact.get("canonical_fact_text")
        if not isinstance(text, str) or not text:
            raise RetrospectiveError(f"normalized fact text invalid for {source['id']}")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected or fact.get("sha256") != expected:
            raise RetrospectiveError(f"normalized fact hash mismatch for {source['id']}")


def verify_archive_root(manifest: dict[str, Any], archive_root: Path) -> int:
    """Verify every downloaded official byte object without requiring network access."""

    checked = 0
    for source in manifest["sources"]:
        filename = source.get("archive_filename")
        if filename is None:
            continue
        path = archive_root / filename
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise RetrospectiveError(f"archived source is missing: {path}") from exc
        if size != source["size_bytes"]:
            raise RetrospectiveError(f"archived source size mismatch: {path}")
        if file_sha256(path) != source["sha256"]:
            raise RetrospectiveError(f"archived source SHA-256 mismatch: {path}")
        checked += 1
    if checked == 0:
        raise RetrospectiveError("no archived official source bytes were checked")
    return checked


def load_canonical_benchmark_rows() -> dict[str, Any]:
    """Derive presentation rows from retained, hash-bound aggregate evidence only."""

    final_receipt = _read_object(FINAL_SUCCESS_EVIDENCE)
    if not (
        final_receipt.get("status") == "complete_owner_authorized_final_evaluation"
        and final_receipt.get("owner_authorized_execution") is True
        and final_receipt.get("case_count") == 200
        and final_receipt.get("artifact", {}).get("sha256")
        == EXPECTED_ARTIFACT_SHA256
    ):
        raise RetrospectiveError("canonical final success evidence is invalid")
    bound = final_receipt.get("bound_evidence", {})
    oracle_identity = bound.get("oracle_receipt", {})
    if (
        oracle_identity.get("path")
        != ORACLE_EVIDENCE.relative_to(ROOT).as_posix()
        or oracle_identity.get("sha256") != file_sha256(ORACLE_EVIDENCE)
    ):
        raise RetrospectiveError("canonical oracle evidence identity changed")
    regression = bound.get("regression_gate", {})
    if (
        regression.get("path")
        != REGRESSION_GATE_EVIDENCE.relative_to(ROOT).as_posix()
        or regression.get("sha256") != file_sha256(REGRESSION_GATE_EVIDENCE)
    ):
        raise RetrospectiveError("canonical regression-gate evidence identity changed")
    oracle_receipt = _read_object(ORACLE_EVIDENCE)
    aggregates = oracle_receipt.get("planner_aggregates", {})
    counts = {
        "oracle": aggregates.get("clairvoyant_oracle_cem", {}).get("solved_count"),
        "v4": final_receipt.get("aggregate", {}).get("solved_count"),
        "tuned": aggregates.get("tuned_rule", {}).get("solved_count"),
        "teacher": regression.get("preparedness_teacher_solved_count"),
        "mpc": aggregates.get("selected_mpc_k5", {}).get("solved_count"),
        "legacy": regression.get("legacy_onnx_fixture_solved_count"),
        "reactive": regression.get("reactive_heuristic_solved_count"),
    }
    if counts != {
        "oracle": 182,
        "v4": 163,
        "tuned": 147,
        "teacher": 139,
        "mpc": 135,
        "legacy": 125,
        "reactive": 72,
    }:
        raise RetrospectiveError(f"canonical benchmark counts changed: {counts}")
    labels = {
        "oracle": (
            "Clairvoyant oracle",
            "Privileged; not a submission baseline",
            "Complete future-tape knowledge; anytime achieved lower bound.",
        ),
        "v4": (
            "v4 PPO (shipped)",
            "Shipped policy",
            "Single owner-authorized final evaluation.",
        ),
        "tuned": (
            "Tuned constant rule",
            "Hand-coded planner",
            "Strongest hand-coded comparator.",
        ),
        "teacher": (
            "Preparedness teacher",
            "Public deterministic regression",
            "Original behavior-cloning teacher.",
        ),
        "mpc": (
            "Selected MPC",
            "Causal diagnostic",
            "Selected receding-horizon planner, k=5.",
        ),
        "legacy": (
            "Legacy shipped policy",
            "Retired regression fixture",
            "Legacy ONNX comparator.",
        ),
        "reactive": (
            "Reactive heuristic",
            "Public deterministic regression",
            "Simple reactive allocation heuristic.",
        ),
    }
    rows = []
    for identifier in ("oracle", "v4", "tuned", "teacher", "mpc", "legacy", "reactive"):
        label, classification, detail = labels[identifier]
        solved = int(counts[identifier])
        rows.append(
            {
                "id": identifier,
                "label": label,
                "solved": solved,
                "total": 200,
                "rate": solved / 200,
                "classification": classification,
                "detail": detail,
            }
        )
    return {
        "rows": rows,
        "evidence": {
            "final_success": {
                "path": FINAL_SUCCESS_EVIDENCE.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(FINAL_SUCCESS_EVIDENCE),
            },
            "oracle": {
                "path": ORACLE_EVIDENCE.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(ORACLE_EVIDENCE),
            },
            "regression_gate": {
                "path": REGRESSION_GATE_EVIDENCE.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(REGRESSION_GATE_EVIDENCE),
            },
        },
        "derivation": "Aggregate counts read from retained evidence; no final roster was imported or evaluated.",
    }


def _reconstruct_from_snapshots(
    observations: dict[str, Any],
    crosswalk: dict[str, Any],
    manifest: dict[str, Any],
    *,
    require_event_observation: bool = True,
) -> dict[str, Any]:
    rows = validate_observations(
        observations,
        manifest,
        require_event_observation=require_event_observation,
    )
    selected = [row for row in rows if row.get("selected") is True]
    if tuple(crosswalk.get("service_order", ())) != SERVICE_ORDER:
        raise RetrospectiveError("crosswalk service order differs from runtime contract")

    services: dict[str, list[float]] = {}
    components: dict[str, dict[str, list[float]]] = {}
    observation_days: dict[str, list[int]] = {}
    for service in SERVICE_ORDER:
        spec = crosswalk["services"][service]
        weights = spec["components"]
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
            raise RetrospectiveError(f"{service} component weights do not sum to one")
        component_curves: dict[str, list[float]] = {}
        for component in weights:
            anchors = [
                (row["day"], row["normalized_value"])
                for row in selected
                if row["service"] == service and row["component"] == component
            ]
            component_curves[component] = _interpolate(anchors)
        components[service] = component_curves
        services[service] = [
            _round(
                sum(
                    float(weights[component]) * component_curves[component][day]
                    for component in weights
                )
            )
            for day in range(31)
        ]
        observation_days[service] = sorted(
            {
                int(row["day"])
                for row in selected
                if row["service"] == service
                and row["evidence_class"] == "direct_official_observation"
            }
        )
        if observation_days[service] != crosswalk["official_observation_markers"][service]:
            raise RetrospectiveError(
                f"official marker days differ from selected facts for {service}"
            )
    total = [
        _round(sum(services[service][day] for service in SERVICE_ORDER) / 5.0)
        for day in range(31)
    ]
    start = date.fromisoformat(observations["day_zero"])
    dates = [(start + timedelta(days=day)).isoformat() for day in range(31)]
    return {
        "dates": dates,
        "days": list(range(31)),
        "service_order": list(SERVICE_ORDER),
        "service_labels": crosswalk["service_labels"],
        "observation_days": observation_days,
        "components": components,
        "services": services,
        "total": total,
    }


def reconstruct(
    observations: dict[str, Any], crosswalk: dict[str, Any]
) -> dict[str, Any]:
    return _reconstruct_from_snapshots(
        observations, crosswalk, _read_object(SOURCE_MANIFEST)
    )


def _archive_identities(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "filename": source["archive_filename"],
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
        }
        for source in manifest["sources"]
        if source.get("archive_filename") is not None
    ]


def build_prepared_contract(archive_root: Path | None = None) -> dict[str, Any]:
    manifest = _read_object(SOURCE_MANIFEST)
    observations = _read_object(OBSERVATIONS)
    crosswalk = _read_object(CROSSWALK)
    web_facts = _read_object(WEB_FACTS)
    validate_sources(manifest)
    validate_web_facts(manifest, web_facts)
    archived_identities = _archive_identities(manifest)
    verified_count = (
        verify_archive_root(manifest, archive_root.resolve())
        if archive_root is not None
        else 0
    )
    reconstruction = reconstruct(observations, crosswalk)
    initial_raw = [reconstruction["services"][name][0] for name in SERVICE_ORDER]
    initial_clipped = [_round(np.clip(value, 0.05, 0.95)) for value in initial_raw]
    clipping = [
        {
            "service": name,
            "reconstructed": initial_raw[index],
            "scenario_value": initial_clipped[index],
            "clipped": initial_raw[index] != initial_clipped[index],
        }
        for index, name in enumerate(SERVICE_ORDER)
    ]
    scenario = {
        key: value
        for key, value in crosswalk["scenario"].items()
        if key not in {"initial_services", "post_landfall_tape"}
    }
    scenario["initial_services"] = initial_clipped
    from backend.app.models import Scenario

    scenario = Scenario.model_validate(scenario).model_dump(mode="json")
    synthetic_benchmark = load_canonical_benchmark_rows()
    contract = {
        "schema_version": "hurricane-maria-prepared-inputs-v1",
        "kind": "frozen_preplanner_historical_data_contract",
        "policy_loaded_during_preparation": False,
        "final_split_used": False,
        "methodology_label": crosswalk["methodology_label"],
        "source_manifest": {
            "path": SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(SOURCE_MANIFEST),
        },
        "observations": {
            "path": OBSERVATIONS.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(OBSERVATIONS),
        },
        "crosswalk": {
            "path": CROSSWALK.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(CROSSWALK),
        },
        "normalized_web_facts": {
            "path": WEB_FACTS.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(WEB_FACTS),
        },
        "archived_source_bytes": {
            "required_object_count": len(archived_identities),
            "verified_object_count": verified_count,
            "verified_before_freeze": verified_count == len(archived_identities),
            "identities_sha256": canonical_hash(archived_identities),
            "verification_command": (
                "python scripts/hurricane_maria_retrospective.py "
                "--verify-archive-root <out-of-repository-archive-root>"
            ),
        },
        "synthetic_benchmark": synthetic_benchmark,
        "source_manifest_snapshot": manifest,
        "observation_table_snapshot": observations,
        "crosswalk_snapshot": crosswalk,
        "normalized_web_facts_snapshot": web_facts,
        "reconstruction": reconstruction,
        "reconstruction_sha256": canonical_hash(reconstruction),
        "scenario": scenario,
        "initial_service_clipping": clipping,
        "tape_contract": {
            "day_count": 30,
            "all_days_no_shock": True,
            "public_risk_all_zero": True,
            "maria_encoded_only_as_initial_condition": True,
        },
        "caption": CAPTION,
    }
    contract["contract_sha256"] = canonical_hash(contract)
    return contract


def validate_prepared_contract(
    contract: dict[str, Any], *, require_archive_verified: bool = False
) -> None:
    if contract.get("kind") != "frozen_preplanner_historical_data_contract":
        raise RetrospectiveError("prepared-input contract kind is invalid")
    expected_hash = contract.get("contract_sha256")
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if expected_hash != canonical_hash(unsigned):
        raise RetrospectiveError("prepared-input contract hash mismatch")
    for field, path in (
        ("source_manifest", SOURCE_MANIFEST),
        ("observations", OBSERVATIONS),
        ("crosswalk", CROSSWALK),
        ("normalized_web_facts", WEB_FACTS),
    ):
        if contract[field]["sha256"] != file_sha256(path):
            raise RetrospectiveError(f"{field} changed after the contract was frozen")
    if contract.get("final_split_used") is not False:
        raise RetrospectiveError("retrospective contract must not use the final split")
    validate_sources(_read_object(SOURCE_MANIFEST))
    validate_web_facts(_read_object(SOURCE_MANIFEST), _read_object(WEB_FACTS))
    expected_reconstruction = reconstruct(
        _read_object(OBSERVATIONS), _read_object(CROSSWALK)
    )
    if contract.get("reconstruction") != expected_reconstruction:
        raise RetrospectiveError("prepared reconstruction differs from source inputs")
    if contract.get("reconstruction_sha256") != canonical_hash(expected_reconstruction):
        raise RetrospectiveError("prepared reconstruction hash mismatch")
    if contract.get("synthetic_benchmark") != load_canonical_benchmark_rows():
        raise RetrospectiveError("prepared benchmark evidence changed")
    for snapshot, path in (
        ("source_manifest_snapshot", SOURCE_MANIFEST),
        ("observation_table_snapshot", OBSERVATIONS),
        ("crosswalk_snapshot", CROSSWALK),
        ("normalized_web_facts_snapshot", WEB_FACTS),
    ):
        if contract.get(snapshot) != _read_object(path):
            raise RetrospectiveError(f"prepared {snapshot} differs from tracked input")
    archive = contract.get("archived_source_bytes", {})
    identities = _archive_identities(_read_object(SOURCE_MANIFEST))
    if (
        archive.get("required_object_count") != len(identities)
        or archive.get("identities_sha256") != canonical_hash(identities)
    ):
        raise RetrospectiveError("archived source identity contract changed")
    verified = (
        archive.get("verified_before_freeze") is True
        and archive.get("verified_object_count") == len(identities)
    )
    if require_archive_verified and not verified:
        raise RetrospectiveError("archived source bytes were not verified before freeze")
    from backend.app.models import Scenario

    Scenario.model_validate(contract["scenario"])


def _validate_embedded_pre_rebind_contract(contract: dict[str, Any]) -> None:
    """Validate the previous contract from its snapshots, not drifted inputs.

    This intentionally avoids every policy, rollout, outcome-summary, and city
    environment import.  The exact known contract digest binds byte-era
    provenance while the checks below independently recompute its historical
    reconstruction and archive/source relationships.
    """

    expected_hash = contract.get("contract_sha256")
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if expected_hash != canonical_hash(unsigned):
        raise RetrospectiveError("pre-rebind prepared-input contract hash mismatch")
    if not (
        contract.get("schema_version") == "hurricane-maria-prepared-inputs-v1"
        and contract.get("kind") == "frozen_preplanner_historical_data_contract"
        and contract.get("policy_loaded_during_preparation") is False
        and contract.get("final_split_used") is False
    ):
        raise RetrospectiveError("pre-rebind prepared-input scope is invalid")

    manifest = contract.get("source_manifest_snapshot")
    observations = contract.get("observation_table_snapshot")
    crosswalk = contract.get("crosswalk_snapshot")
    web_facts = contract.get("normalized_web_facts_snapshot")
    if not all(isinstance(value, dict) for value in (manifest, observations, crosswalk, web_facts)):
        raise RetrospectiveError("pre-rebind prepared-input snapshots are invalid")
    validate_sources(manifest)
    validate_web_facts(manifest, web_facts)
    reconstruction = _reconstruct_from_snapshots(
        observations, crosswalk, manifest, require_event_observation=False
    )
    if contract.get("reconstruction") != reconstruction:
        raise RetrospectiveError("pre-rebind reconstruction does not recompute")
    if contract.get("reconstruction_sha256") != canonical_hash(reconstruction):
        raise RetrospectiveError("pre-rebind reconstruction hash mismatch")
    identities = _archive_identities(manifest)
    archive = contract.get("archived_source_bytes", {})
    if not (
        archive.get("required_object_count") == len(identities)
        and archive.get("verified_object_count") == len(identities)
        and archive.get("verified_before_freeze") is True
        and archive.get("identities_sha256") == canonical_hash(identities)
    ):
        raise RetrospectiveError("pre-rebind archive verification identity is invalid")
    if contract.get("synthetic_benchmark") != load_canonical_benchmark_rows():
        raise RetrospectiveError("pre-rebind benchmark evidence changed")


def _planner_series_without_helpers(
    planner: dict[str, Any], initial_services: list[float]
) -> dict[str, Any]:
    summary = planner.get("summary", {})
    trajectory = summary.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 30:
        raise RetrospectiveError("pre-rebind planner trajectory length is invalid")
    if summary.get("trajectory_sha256") != canonical_hash(trajectory):
        raise RetrospectiveError("pre-rebind planner trajectory hash mismatch")
    services = [list(map(float, initial_services))]
    for row in trajectory:
        day_services = row.get("services_end")
        if not isinstance(day_services, list) or len(day_services) != 5:
            raise RetrospectiveError("pre-rebind planner service state is invalid")
        if not isinstance(row.get("raw_action"), list) or len(row["raw_action"]) != 22:
            raise RetrospectiveError("pre-rebind planner action is not 22-dimensional")
        if row.get("hard_violation_count") != 0:
            raise RetrospectiveError("pre-rebind planner contains a hard violation")
        residuals = row.get("logistics", {}).get("conservation_residual")
        if not isinstance(residuals, list) or any(value != 0.0 for value in residuals):
            raise RetrospectiveError("pre-rebind conservation residual changed")
        services.append(list(map(float, day_services)))
    return {
        "total": [_round(sum(values) / 5.0) for values in services],
        "services": {
            service: [_round(values[index]) for values in services]
            for index, service in enumerate(SERVICE_ORDER)
        },
    }


def validate_pre_rebind_publication(
    expected_identities: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the exact known pre-correction bundle is present."""

    identities = PRE_REBIND_IDENTITIES if expected_identities is None else expected_identities
    if set(identities) != {PREPARED_CONTRACT, RECEIPT, FRONTEND, REPORT}:
        raise RetrospectiveError("pre-rebind identity set is incomplete")
    for path, expected in identities.items():
        if path.is_symlink() or not path.is_file():
            raise RetrospectiveError(f"pre-rebind output missing or unsafe: {path}")
        if file_sha256(path) != expected:
            raise RetrospectiveError(f"pre-rebind output identity changed: {path}")

    receipt = _read_object(RECEIPT)
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if not (
        receipt.get("receipt_sha256") == PRE_REBIND_RECEIPT_SHA256
        and canonical_hash(unsigned_receipt) == PRE_REBIND_RECEIPT_SHA256
    ):
        raise RetrospectiveError("pre-rebind receipt identity changed")
    if not (
        receipt.get("schema_version") == "hurricane-maria-retrospective-receipt-v1"
        and receipt.get("kind") == "project_reconstruction_from_official_records"
        and receipt.get("authorizing") is False
        and receipt.get("final_split_used") is False
        and receipt.get("model_selection_used") is False
    ):
        raise RetrospectiveError("pre-rebind receipt evidence scope is invalid")
    contract = receipt.get("frozen_inputs")
    if not isinstance(contract, dict):
        raise RetrospectiveError("pre-rebind receipt lacks its prepared-input snapshot")
    if contract.get("contract_sha256") != PRE_REBIND_CONTRACT_SHA256:
        raise RetrospectiveError("pre-rebind contract identity changed")
    _validate_embedded_pre_rebind_contract(contract)
    prepared = _read_object(PREPARED_CONTRACT)
    prepared_identity = receipt.get("prepared_contract", {})
    if not (
        prepared == contract
        and prepared_identity.get("path")
        == PREPARED_CONTRACT.relative_to(ROOT).as_posix()
        and prepared_identity.get("file_sha256") == identities[PREPARED_CONTRACT]
        and prepared_identity.get("contract_sha256") == PRE_REBIND_CONTRACT_SHA256
    ):
        raise RetrospectiveError("pre-rebind prepared-contract binding changed")
    if receipt.get("scenario") != contract.get("scenario"):
        raise RetrospectiveError("pre-rebind scenario differs from frozen inputs")
    if receipt.get("historical") != contract.get("reconstruction"):
        raise RetrospectiveError("pre-rebind historical reconstruction changed")
    if receipt.get("synthetic_benchmark") != contract.get("synthetic_benchmark"):
        raise RetrospectiveError("pre-rebind synthetic benchmark changed")
    if receipt.get("artifact") != {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_ARTIFACT_SHA256,
        "runtime": "onnxruntime-cpu",
    } or file_sha256(ARTIFACT) != EXPECTED_ARTIFACT_SHA256:
        raise RetrospectiveError("pre-rebind artifact identity changed")

    tape = receipt.get("tape", {})
    days = tape.get("days")
    if not isinstance(days, list) or len(days) != 30 or tape.get("sha256") != canonical_hash(days):
        raise RetrospectiveError("pre-rebind tape identity changed")
    for day, shock in enumerate(days, start=1):
        if not (
            shock.get("day") == day
            and shock.get("type") is None
            and shock.get("severity") == 0.0
            and shock.get("impact") == [0.0] * 5
            and shock.get("public_risk_before") == [0.0] * 5
            and shock.get("public_risk_next") == [0.0] * 5
            and shock.get("assessment_tail") == (day >= 28)
        ):
            raise RetrospectiveError(f"pre-rebind tape Day {day} changed")
    expected_labels = {
        "v4": "onnx_policy",
        "reactive": "reactive_public_state_heuristic",
    }
    for planner_id, label in expected_labels.items():
        planner = receipt.get("planners", {}).get(planner_id, {})
        if planner.get("summary", {}).get("planner") != label:
            raise RetrospectiveError(f"pre-rebind {planner_id} helper label changed")
        series = _planner_series_without_helpers(
            planner, list(map(float, receipt["scenario"]["initial_services"]))
        )
        trajectory = planner["summary"]["trajectory"]
        if not (
            planner.get("series") == series
            and planner.get("trajectory_tape_sha256") == tape["sha256"]
            and [row.get("shock") for row in trajectory] == days
            and planner["summary"].get("hard_violation_count") == 0
            and planner["summary"].get("max_logistics_conservation_residual") == 0.0
        ):
            raise RetrospectiveError(f"pre-rebind {planner_id} evidence changed")
    if receipt.get("invariants", {}).get("observation_count") != 73 or receipt.get(
        "invariants", {}
    ).get("action_count") != 22:
        raise RetrospectiveError("pre-rebind interface invariant changed")
    return receipt


def _source_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {source["id"]: source for source in manifest["sources"]}


def _assert_expected_provenance_changes(
    old_receipt: dict[str, Any], new_contract: dict[str, Any]
) -> None:
    """Require exactly the audited metadata/crosswalk correction and no result drift."""

    old_contract = old_receipt["frozen_inputs"]
    old_manifest = old_contract["source_manifest_snapshot"]
    new_manifest = new_contract["source_manifest_snapshot"]
    old_sources = _source_by_id(old_manifest)
    new_sources = _source_by_id(new_manifest)
    if set(old_sources) != set(new_sources):
        raise RetrospectiveError("source roster changed during provenance correction")
    allowed_source_fields = {
        "nhc_maria_tcr": {
            "publication_date": ("2019-01-04", "2023-01-04"),
            "locators": (
                ["page 1, landfall chronology", "pages 7-8, Puerto Rico impacts"],
                [
                    "page 2, Puerto Rico landfall chronology",
                    "pages 7-8, Puerto Rico impacts",
                ],
            ),
        },
        "fcc_2017_10_19": {
            "url": (
                "https://docs.fcc.gov/public/attachments/DOC-347339A2.pdf",
                "https://docs.fcc.gov/public/attachments/DOC-347339A1.pdf",
            )
        },
    }
    for source_id in old_sources:
        old_source = old_sources[source_id]
        new_source = new_sources[source_id]
        allowed = allowed_source_fields.get(source_id, {})
        for key in set(old_source) | set(new_source):
            if key in allowed:
                if (old_source.get(key), new_source.get(key)) != allowed[key]:
                    raise RetrospectiveError(
                        f"unexpected {source_id} {key} provenance correction"
                    )
            elif old_source.get(key) != new_source.get(key):
                raise RetrospectiveError(
                    f"unexpected source drift during provenance correction: {source_id}/{key}"
                )

    old_observations = old_contract["observation_table_snapshot"]
    new_observations = new_contract["observation_table_snapshot"]
    if "event_observations" in old_observations:
        raise RetrospectiveError("pre-rebind observations unexpectedly contain event metadata")
    event_rows = new_observations.get("event_observations")
    if not isinstance(event_rows, list) or len(event_rows) != 1:
        raise RetrospectiveError("corrected landfall event metadata is missing")
    event = event_rows[0]
    if not (
        event.get("id") == "maria_puerto_rico_landfall"
        and event.get("raw_value") == "1015 UTC 20 September 2017"
        and event.get("time_local") == "06:15:00"
        and event.get("local_timezone") == "Atlantic Standard Time (UTC-04:00)"
        and event.get("reconstruction_role")
        == "event-window anchor only; not folded into any service index"
    ):
        raise RetrospectiveError("corrected landfall event metadata is invalid")
    added_health_ids = {
        "healthcare_centers_day0_estimate",
        "healthcare_centers_day10_estimate",
        "healthcare_centers_day30_estimate",
    }
    new_rows = new_observations["observations"]
    health_rows = [row for row in new_rows if row.get("id") in added_health_ids]
    retained_rows = [row for row in new_rows if row.get("id") not in added_health_ids]
    if retained_rows != old_observations["observations"] or {
        row.get("id") for row in health_rows
    } != added_health_ids:
        raise RetrospectiveError("observation rows changed beyond the audited additions")
    expected_health_points = {0: 0.25, 10: 0.85507246, 30: 0.94031453}
    if any(
        row.get("component") != "health_center_availability"
        or row.get("evidence_class") != "project_estimate"
        or row.get("selected") is not True
        or row.get("normalized_value") != expected_health_points.get(row.get("day"))
        for row in health_rows
    ):
        raise RetrospectiveError("health-center project-estimate additions changed")
    old_without_rows = {key: value for key, value in old_observations.items()}
    new_without_rows = {
        key: value
        for key, value in new_observations.items()
        if key != "event_observations"
    }
    old_without_rows["observations"] = retained_rows
    new_without_rows["observations"] = retained_rows
    if old_without_rows != new_without_rows:
        raise RetrospectiveError("observation-table metadata changed unexpectedly")

    old_crosswalk = old_contract["crosswalk_snapshot"]
    new_crosswalk = copy.deepcopy(new_contract["crosswalk_snapshot"])
    healthcare_disclosure = (
        "No commensurable island-wide health-center operational series was found "
        "for the window; its explicitly marked project-estimate component follows "
        "the nearest official hospital trajectory and is not a measured "
        "health-center percentage."
    )
    healthcare = new_crosswalk["services"]["healthcare"]
    if healthcare.get("components") != {
        "operational_hospital_availability": 0.5,
        "health_center_availability": 0.5,
    } or healthcare.get("interpretation") != (
        "Equal mean of hospital operational availability and a disclosed "
        "health-center unavailable-data project estimate. No commensurable "
        "island-wide health-center operational series was found for the window, "
        "so the health-center component follows the nearest official hospital "
        "trajectory; it is not a measured health-center percentage."
    ):
        raise RetrospectiveError("healthcare crosswalk correction changed")
    healthcare["components"] = {
        "operational_hospital_availability": 1.0,
    }
    healthcare["interpretation"] = (
        "Hospital operational-availability proxy; partially operational and fully "
        "operational facilities both count as operational."
    )
    if new_crosswalk.get("disclosures", []).count(healthcare_disclosure) != 1:
        raise RetrospectiveError("healthcare crosswalk disclosure changed")
    new_crosswalk["disclosures"].remove(healthcare_disclosure)
    if new_crosswalk != old_crosswalk:
        raise RetrospectiveError("crosswalk changed beyond the audited healthcare structure")
    if new_contract["normalized_web_facts_snapshot"] != old_contract[
        "normalized_web_facts_snapshot"
    ]:
        raise RetrospectiveError("normalized web facts changed during provenance correction")

    old_historical = old_receipt["historical"]
    new_historical = new_contract["reconstruction"]
    for field in (
        "dates",
        "days",
        "service_order",
        "service_labels",
        "observation_days",
        "services",
        "total",
    ):
        if new_historical.get(field) != old_historical.get(field):
            raise RetrospectiveError(
                f"provenance correction changed historical {field}"
            )
    if new_historical["components"]["healthcare"][
        "health_center_availability"
    ] != new_historical["components"]["healthcare"][
        "operational_hospital_availability"
    ]:
        raise RetrospectiveError("health-center estimate altered the healthcare curve")
    if new_contract.get("scenario") != old_receipt.get("scenario"):
        raise RetrospectiveError("provenance correction changed the scenario")
    if new_contract.get("synthetic_benchmark") != old_receipt.get(
        "synthetic_benchmark"
    ):
        raise RetrospectiveError("provenance correction changed the benchmark table")


def _provenance_correction_record(
    old_receipt: dict[str, Any],
    new_contract: dict[str, Any],
    old_file_identities: dict[Path, str],
) -> dict[str, Any]:
    old_crosswalk = old_receipt["frozen_inputs"]["crosswalk_snapshot"]
    new_crosswalk = new_contract["crosswalk_snapshot"]
    return {
        "id": PROVENANCE_CORRECTION_ID,
        "kind": "provenance_only_rebind",
        "recorded_on": "2026-08-13",
        "previous_receipt_sha256": old_receipt["receipt_sha256"],
        "previous_contract_sha256": old_receipt["frozen_inputs"]["contract_sha256"],
        "previous_output_file_sha256": {
            path.relative_to(ROOT).as_posix(): digest
            for path, digest in old_file_identities.items()
        },
        "corrected_contract_sha256": new_contract["contract_sha256"],
        "source_manifest_changes": [
            {
                "source_id": "fcc_2017_10_19",
                "field": "url",
                "old": "https://docs.fcc.gov/public/attachments/DOC-347339A2.pdf",
                "new": "https://docs.fcc.gov/public/attachments/DOC-347339A1.pdf",
            },
            {
                "source_id": "nhc_maria_tcr",
                "field": "publication_date",
                "old": "2019-01-04",
                "new": "2023-01-04",
            },
            {
                "source_id": "nhc_maria_tcr",
                "field": "locators",
                "old": ["page 1, landfall chronology", "pages 7-8, Puerto Rico impacts"],
                "new": [
                    "page 2, Puerto Rico landfall chronology",
                    "pages 7-8, Puerto Rico impacts",
                ],
            },
        ],
        "event_metadata_change": {
            "old": None,
            "new": new_contract["observation_table_snapshot"]["event_observations"][0],
        },
        "healthcare_crosswalk_change": {
            "old": old_crosswalk["services"]["healthcare"],
            "new": new_crosswalk["services"]["healthcare"],
            "added_project_estimate_observation_ids": [
                "healthcare_centers_day0_estimate",
                "healthcare_centers_day10_estimate",
                "healthcare_centers_day30_estimate",
            ],
        },
        "execution": {
            "planner_rerun": False,
            "policy_loaded": False,
            "rollout_helpers_called": False,
            "statement": (
                "No planner was rerun; this rebind corrects provenance metadata "
                "and a numerically neutral crosswalk disclosure only."
            ),
        },
        "numerical_effect": {
            "historical_service_curves_changed": False,
            "historical_total_changed": False,
            "scenario_changed": False,
            "tape_changed": False,
            "planner_summaries_changed": False,
            "planner_series_changed": False,
            "trajectory_hashes_changed": False,
            "invariants_changed": False,
        },
    }


def build_provenance_rebound_receipt(
    old_receipt: dict[str, Any],
    new_contract: dict[str, Any],
    old_file_identities: dict[Path, str],
) -> dict[str, Any]:
    """Rebind corrected evidence while retaining every recorded planner result."""

    _assert_expected_provenance_changes(old_receipt, new_contract)
    tape = old_receipt.get("tape", {})
    initial_services = list(map(float, old_receipt["scenario"]["initial_services"]))
    for planner_id in ("v4", "reactive"):
        planner = old_receipt.get("planners", {}).get(planner_id, {})
        trajectory = planner.get("summary", {}).get("trajectory", [])
        if not (
            planner.get("series")
            == _planner_series_without_helpers(planner, initial_services)
            and planner.get("trajectory_tape_sha256") == tape.get("sha256")
            and [row.get("shock") for row in trajectory] == tape.get("days")
        ):
            raise RetrospectiveError(
                f"pre-rebind retained {planner_id} evidence is invalid"
            )
    rebound = copy.deepcopy(old_receipt)
    prepared_text = _json_text(new_contract).encode("utf-8")
    rebound["prepared_contract"] = {
        "path": PREPARED_CONTRACT.relative_to(ROOT).as_posix(),
        "file_sha256": hashlib.sha256(prepared_text).hexdigest(),
        "contract_sha256": new_contract["contract_sha256"],
    }
    rebound["frozen_inputs"] = new_contract
    rebound["methodology_label"] = new_contract["methodology_label"]
    rebound["caption"] = new_contract["caption"]
    rebound["scenario"] = new_contract["scenario"]
    rebound["historical"] = new_contract["reconstruction"]
    rebound["synthetic_benchmark"] = new_contract["synthetic_benchmark"]
    rebound["provenance_corrections"] = [
        _provenance_correction_record(old_receipt, new_contract, old_file_identities)
    ]
    rebound.pop("receipt_sha256", None)
    rebound["receipt_sha256"] = canonical_hash(rebound)

    for field in ("artifact", "tape", "planners", "invariants"):
        if rebound[field] != old_receipt[field]:
            raise RetrospectiveError(f"provenance rebind changed retained {field}")
    for planner_id in ("v4", "reactive"):
        old_planner = old_receipt["planners"][planner_id]
        new_planner = rebound["planners"][planner_id]
        if not (
            new_planner["summary"] == old_planner["summary"]
            and new_planner["series"] == old_planner["series"]
            and new_planner["summary"]["trajectory_sha256"]
            == old_planner["summary"]["trajectory_sha256"]
        ):
            raise RetrospectiveError(f"provenance rebind changed {planner_id} evidence")
    return rebound


def rebind_provenance(archive_root: Path) -> dict[str, Any]:
    """Perform the one authorized provenance-only correction, without planners."""

    old_receipt = validate_pre_rebind_publication()
    new_contract = build_prepared_contract(archive_root.resolve())
    validate_prepared_contract(new_contract, require_archive_verified=True)
    rebound = build_provenance_rebound_receipt(
        old_receipt, new_contract, PRE_REBIND_IDENTITIES
    )
    outputs = {
        PREPARED_CONTRACT: _json_text(new_contract),
        RECEIPT: _json_text(rebound),
        FRONTEND: render_frontend(rebound),
        REPORT: render_report(rebound),
    }
    publish_replace_exact_bundle(outputs, PRE_REBIND_IDENTITIES)
    for path, payload in outputs.items():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise RetrospectiveError(
                f"provenance rebind post-publication verification failed: {path}"
            )
    return rebound


def _no_shock_tape() -> list[Any]:
    from backend.app.city.scenarios import Shock

    return [
        Shock(
            day=day,
            type=None,
            severity=0.0,
            impact=[0.0] * 5,
            budget_factor=0.0,
            forced=False,
            occurrence_probability=0.0,
            occurrence_draw=1.0,
            public_risk_before=[0.0] * 5,
            public_risk_next=[0.0] * 5,
            assessment_tail=day >= 28,
        )
        for day in range(1, 31)
    ]


def _planner_payload(
    planner: str, summary: dict[str, Any], scenario: Any
) -> dict[str, Any]:
    trajectory = summary.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 30:
        raise RetrospectiveError(f"{planner} helper returned an invalid trajectory")
    if summary.get("trajectory_sha256") != canonical_hash(trajectory):
        raise RetrospectiveError(f"{planner} trajectory hash mismatch")
    services = [list(map(float, scenario.initial_services))] + [
        list(map(float, row["services_end"])) for row in trajectory
    ]
    total = [
        _round(sum(day_values) / len(day_values)) for day_values in services
    ]
    return {
        "planner": planner,
        "summary": summary,
        "trajectory_tape_sha256": canonical_hash(
            [row["shock"] for row in trajectory]
        ),
        "series": {
            "total": total,
            "services": {
                name: [_round(day_values[index]) for day_values in services]
                for index, name in enumerate(SERVICE_ORDER)
            },
        },
    }


def run_retrospective(contract: dict[str, Any]) -> dict[str, Any]:
    validate_prepared_contract(contract, require_archive_verified=True)
    if file_sha256(ARTIFACT) != EXPECTED_ARTIFACT_SHA256:
        raise RetrospectiveError("shipped artifact SHA-256 mismatch")
    from backend.app.city.environment import (
        ACTION_ORDER,
        OBSERVATION_ORDER,
        rollout_baseline,
        rollout_policy,
    )
    from backend.app.models import Scenario
    from model.policy import ACTION_COUNT, OBSERVATION_COUNT, load_policy

    if (
        len(OBSERVATION_ORDER) != OBSERVATION_COUNT
        or len(ACTION_ORDER) != ACTION_COUNT
        or OBSERVATION_COUNT != 73
        or ACTION_COUNT != 22
    ):
        raise RetrospectiveError("policy/environment tensor interface changed")

    scenario = Scenario.model_validate(contract["scenario"])
    schedule = _no_shock_tape()
    tape_payload = [dataclasses.asdict(item) for item in schedule]
    policy = load_policy(ARTIFACT, expected_sha256=EXPECTED_ARTIFACT_SHA256)
    v4_summary = rollout_policy(scenario, 0, policy.predict, schedule=schedule)
    reactive_summary = rollout_baseline(scenario, 0, schedule=schedule)
    v4 = _planner_payload("v4", v4_summary, scenario)
    reactive = _planner_payload("reactive", reactive_summary, scenario)
    tape_hash = canonical_hash(tape_payload)
    for rollout in (v4, reactive):
        if rollout["trajectory_tape_sha256"] != tape_hash:
            raise RetrospectiveError("planner did not receive the frozen tape")
        if rollout["summary"]["hard_violation_count"] != 0:
            raise RetrospectiveError("retrospective rollout has a hard violation")
        if rollout["summary"]["max_logistics_conservation_residual"] != 0.0:
            raise RetrospectiveError("retrospective conservation residual is not 0.0")
    receipt = {
        "schema_version": "hurricane-maria-retrospective-receipt-v1",
        "kind": "project_reconstruction_from_official_records",
        "authorizing": False,
        "final_split_used": False,
        "model_selection_used": False,
        "prepared_contract": {
            "path": PREPARED_CONTRACT.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(PREPARED_CONTRACT),
            "contract_sha256": contract["contract_sha256"],
        },
        "frozen_inputs": contract,
        "methodology_label": contract["methodology_label"],
        "caption": CAPTION,
        "artifact": {
            "path": ARTIFACT.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_ARTIFACT_SHA256,
            "runtime": "onnxruntime-cpu",
        },
        "scenario": contract["scenario"],
        "historical": contract["reconstruction"],
        "synthetic_benchmark": contract["synthetic_benchmark"],
        "tape": {
            "sha256": tape_hash,
            "days": tape_payload,
        },
        "planners": {"v4": v4, "reactive": reactive},
        "invariants": {
            "historical_data_frozen_before_planner_load": True,
            "policy_and_heuristic_received_identical_tape": True,
            "tape_has_no_secondary_shocks": True,
            "artifact_sha256_matched": True,
            "observation_count": 73,
            "action_count": 22,
            "all_hard_violation_counts_zero": True,
            "all_conservation_residuals_exactly_zero": True,
            "all_trajectory_hashes_verified": True,
            "canonical_explicit_tape_helpers_used": True,
            "deterministic_replay_bound_by_artifact_scenario_and_tape_hashes": True,
            "final_split_used": False,
        },
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)
    return receipt


def _benchmark_rows() -> list[dict[str, Any]]:
    return load_canonical_benchmark_rows()["rows"]


def frontend_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    historical = receipt["historical"]
    source_manifest = _read_object(SOURCE_MANIFEST)
    benchmark_totals = {
        int(row["total"]) for row in receipt["synthetic_benchmark"]["rows"]
    }
    if len(benchmark_totals) != 1:
        raise RetrospectiveError("synthetic benchmark rows use different denominators")
    day_zero = date.fromisoformat(historical["dates"][0])
    day_end_date = date.fromisoformat(historical["dates"][-1])
    milestone_days = [
        day
        for day in (historical["days"][0], 10, 20, historical["days"][-1])
        if day in historical["days"]
    ]
    return {
        "methodologyLabel": receipt["methodology_label"],
        "caption": CAPTION,
        "dates": historical["dates"],
        "days": historical["days"],
        "serviceOrder": historical["service_order"],
        "serviceLabels": {
            service: historical["service_labels"][service]
            for service in sorted(SERVICE_ORDER)
        },
        "observationDays": {
            service: historical["observation_days"][service]
            for service in sorted(SERVICE_ORDER)
        },
        "series": {
            "historical": {
                "label": "Project reconstruction",
                "evidenceType": "Project-derived index from official records",
                "total": historical["total"],
                "services": {
                    service: historical["services"][service]
                    for service in sorted(SERVICE_ORDER)
                },
            },
            "v4": {
                "label": "Shipped v4",
                "evidenceType": "Simulated alternative",
                "total": receipt["planners"]["v4"]["series"]["total"],
                "services": {
                    service: receipt["planners"]["v4"]["series"]["services"][service]
                    for service in SERVICE_ORDER
                },
            },
            "reactive": {
                "label": "Reactive heuristic",
                "evidenceType": "Simulated alternative",
                "total": receipt["planners"]["reactive"]["series"]["total"],
                "services": {
                    service: receipt["planners"]["reactive"]["series"]["services"][service]
                    for service in SERVICE_ORDER
                },
            },
        },
        "receiptSha256": receipt["receipt_sha256"],
        "sourceManifestSha256": file_sha256(SOURCE_MANIFEST),
        "reconstructionSha256": canonical_hash(historical),
        "artifactSha256": EXPECTED_ARTIFACT_SHA256,
        "sourceCount": len(source_manifest["sources"]),
        "display": {
            "milestoneDays": milestone_days,
            "dayZeroLabel": day_zero.strftime("%b %d").replace(" 0", " "),
            "dayEndLabel": day_end_date.strftime("%b %d, %Y").replace(" 0", " "),
            "horizonStart": historical["days"][0],
            "dayEnd": historical["days"][-1],
            "dayCount": len(historical["days"]),
            "indexMin": 0,
            "indexMax": 100,
        },
        "scenarioCount": 1,
        "syntheticBenchmarkCaseCount": benchmark_totals.pop(),
        "interface": {
            "observationCount": receipt["invariants"]["observation_count"],
            "actionCount": receipt["invariants"]["action_count"],
        },
        "benchmarkRows": [
            {
                "id": row["id"],
                "label": row["label"],
                "classification": row["classification"],
                "solved": row["solved"],
                "total": row["total"],
                "rate": row["rate"],
                "detail": row["detail"],
            }
            for row in receipt["synthetic_benchmark"]["rows"]
        ],
    }


def render_frontend(receipt: dict[str, Any]) -> str:
    payload = json.dumps(frontend_payload(receipt), ensure_ascii=True, indent=2)
    return (
        "// Generated by scripts/hurricane_maria_retrospective.py. Do not edit.\n"
        "export type MariaServiceId =\n"
        "  | 'transport'\n"
        "  | 'housing'\n"
        "  | 'food'\n"
        "  | 'healthcare'\n"
        "  | 'public_services'\n\n"
        f"export const mariaRetrospective = {payload} as const\n"
    )


def render_report(receipt: dict[str, Any]) -> str:
    historical = receipt["historical"]
    v4 = receipt["planners"]["v4"]
    reactive = receipt["planners"]["reactive"]
    frozen = receipt["frozen_inputs"]
    manifest = frozen["source_manifest_snapshot"]
    event_observations = frozen["observation_table_snapshot"].get(
        "event_observations", []
    )
    observations = frozen["observation_table_snapshot"]["observations"]
    checkpoints = (0, 10, 20, 30)

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Hurricane Maria 30-day retrospective",
        "",
        "This is one fixed **project reconstruction from official records**. It is not an official FEMA restoration percentage, an inverse fit, or a sensitivity study.",
        "",
        f"> {CAPTION}",
        "",
        "| Series | Evidence type | Day 0 | Day 10 | Day 20 | Day 30 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, evidence, values in (
        ("Project reconstruction", "official records + disclosed estimates", historical["total"]),
        ("Shipped v4", "simulated alternative", v4["series"]["total"]),
        ("Reactive heuristic", "simulated alternative", reactive["series"]["total"]),
    ):
        cells = " | ".join(f"{100 * values[day]:.1f}" for day in checkpoints)
        lines.append(f"| {label} | {evidence} | {cells} |")
    lines.extend(
        [
            "",
            "Values are derived-recovery-index points on a 0-100 display scale.",
            "",
            "## Frozen scenario and replay",
            "",
            f"- Prepared-input contract SHA-256: `{receipt['prepared_contract']['contract_sha256']}`",
            f"- Receipt SHA-256: `{receipt['receipt_sha256']}`",
            f"- Shipped ONNX SHA-256: `{receipt['artifact']['sha256']}`",
            f"- Explicit no-secondary-shock tape SHA-256: `{receipt['tape']['sha256']}`",
            "- Daily budget: 180 abstract units; daily crew pool: 150 abstract units.",
            "- Maria is encoded only in the post-landfall Day-0 service state.",
            "- Both planners received the same explicit 30-day tape.",
            "- Hard violations: 0 for both planners.",
            "- Maximum conservation residual: exactly 0.0 for both planners.",
            "- The canonical final split was not imported or evaluated.",
            "",
            "## Service crosswalk",
            "",
            "- Transport: official road/airport/port milestones; the Day-30 cross-mode value is a project estimate.",
            "- Housing: qualitative official damage/recovery evidence converted to disclosed project-estimate anchors.",
            "- Food and water: equal mean of potable-water and grocery availability.",
            "- Healthcare: equal mean of operational-hospital availability and a clearly marked unavailable-data health-center project estimate that follows the hospital trajectory.",
            "- Public services: equal mean of restored electricity and operational cellular-site share.",
            "",
            "The source manifest, raw observation table, selected/rejected alternatives, conversion formulas, and interpolation contract are tracked alongside this report.",
            "",
        ]
    )
    if event_observations:
        lines.extend(
            [
                "## Event-window anchor",
                "",
                "| Event | Date | UTC | AST | Location | Source locator | Reconstruction role |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for event in event_observations:
            lines.append(
                "| "
                + " | ".join(
                    cell(value)
                    for value in (
                        event["id"],
                        event["date"],
                        event["time_utc"],
                        event["time_local"],
                        event["location"],
                        event["locator"],
                        event["reconstruction_role"],
                    )
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Official source manifest",
            "",
            "| Agency | Title | Published | Retrieved | URL | Page/table locator | Archive filename | Bytes | SHA-256 / normalized-fact SHA-256 |",
            "|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    for source in manifest["sources"]:
        digest = source.get("sha256") or source.get("verified_excerpt_sha256") or "—"
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    source["agency"],
                    source["title"],
                    source["publication_date"],
                    manifest["retrieved_on"],
                    source["url"],
                    "; ".join(source["locators"]),
                    source.get("archive_filename"),
                    source.get("size_bytes"),
                    digest,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Blocked FEMA/Defense.gov pages use the exact locator plus a tracked, reproducibly hashed normalized project fact record. These records are not quotations. Every downloaded byte object was verified before the input contract was frozen.",
            "",
            "## Raw-statistic, conversion, and selection review",
            "",
            "| ID | Date | Service / component | Sources | Reported statistic | Units | Denominator | Conversion | Final point | Decision | Selection reason |",
            "|---|---|---|---|---|---|---|---|---:|---|---|",
        ]
    )
    for row in observations:
        decision = (
            "selected official observation"
            if row["selected"] and row["evidence_class"] == "direct_official_observation"
            else "selected project estimate"
            if row["selected"]
            else "rejected alternative"
        )
        reason = (
            row.get("rejection_reason")
            or "Selected under the frozen denominator/coverage/date rules."
        )
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    row["id"],
                    row["date"],
                    f"{row['service']} / {row['component']}",
                    ", ".join(row["source_ids"]),
                    row["raw_value"],
                    row["raw_units"],
                    row.get("denominator"),
                    row["conversion"],
                    f"{row['normalized_value']:.8f}",
                    decision,
                    reason,
                )
            )
            + " |"
        )
    component_columns = [
        ("transport", "transport_access_proxy", "Transport proxy"),
        ("housing", "shelter_and_habitability_proxy", "Housing proxy"),
        ("food", "potable_water_access", "Water"),
        ("food", "grocery_availability", "Grocery"),
        ("healthcare", "operational_hospital_availability", "Hospitals"),
        ("healthcare", "health_center_availability", "Health centers (estimate)"),
        ("public_services", "electricity", "Power"),
        ("public_services", "cellular", "Cell sites"),
    ]
    lines.extend(
        [
            "",
            "## Complete component interpolation, Day 0–30",
            "",
            "Piecewise-linear interpolation is applied independently between selected anchors; the fixed carry-back/carry-forward values are explicitly marked as project estimates above.",
            "",
            "| Day | Date | "
            + " | ".join(label for _, _, label in component_columns)
            + " |",
            "|---:|---|" + "---:|" * len(component_columns),
        ]
    )
    for day, day_date in zip(historical["days"], historical["dates"], strict=True):
        values = [
            historical["components"][service][component][day]
            for service, component, _ in component_columns
        ]
        lines.append(
            f"| {day} | {day_date} | "
            + " | ".join(f"{value:.8f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Complete final project reconstruction, Day 0–30",
            "",
            "Each service is the disclosed component-weighted mean; Total is the equal arithmetic mean of all five services.",
            "",
            "| Day | Date | Transport | Housing | Food and water | Healthcare | Public services | Total |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for day, day_date in zip(historical["days"], historical["dates"], strict=True):
        service_values = [historical["services"][name][day] for name in SERVICE_ORDER]
        lines.append(
            f"| {day} | {day_date} | "
            + " | ".join(f"{value:.8f}" for value in service_values)
            + f" | {historical['total'][day]:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Initial-condition clipping review",
            "",
            "| Service | Reconstructed Day 0 | Scenario Day 0 | Clipped |",
            "|---|---:|---:|---|",
        ]
    )
    for clipping in frozen["initial_service_clipping"]:
        lines.append(
            f"| {clipping['service']} | {clipping['reconstructed']:.8f} | "
            f"{clipping['scenario_value']:.8f} | {str(clipping['clipped']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic benchmark results (separate evidence scope)",
            "",
            "These rows are derived from the retained canonical aggregate evidence bound in the frozen input contract; this retrospective did not import or evaluate the final roster.",
            "",
            "| Method | Solved | Rate | Classification | Detail |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in receipt["synthetic_benchmark"]["rows"]:
        lines.append(
            f"| {cell(row['label'])} | {row['solved']}/{row['total']} | "
            f"{row['rate']:.3f} | {cell(row['classification'])} | "
            f"{cell(row['detail'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_receipt(receipt: dict[str, Any]) -> None:
    expected_hash = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if expected_hash != canonical_hash(unsigned):
        raise RetrospectiveError("receipt hash mismatch")
    if not (
        receipt.get("kind") == "project_reconstruction_from_official_records"
        and receipt.get("final_split_used") is False
        and receipt.get("model_selection_used") is False
        and receipt.get("authorizing") is False
    ):
        raise RetrospectiveError("retrospective evidence-scope flags are invalid")
    prepared = _read_object(PREPARED_CONTRACT)
    validate_prepared_contract(prepared, require_archive_verified=True)
    if receipt.get("frozen_inputs") != prepared:
        raise RetrospectiveError("receipt does not embed the exact prepared contract")
    prepared_identity = receipt.get("prepared_contract", {})
    if (
        prepared_identity.get("path") != PREPARED_CONTRACT.relative_to(ROOT).as_posix()
        or prepared_identity.get("file_sha256") != file_sha256(PREPARED_CONTRACT)
        or prepared_identity.get("contract_sha256") != prepared["contract_sha256"]
    ):
        raise RetrospectiveError("prepared-contract receipt identity changed")
    if receipt.get("artifact") != {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_ARTIFACT_SHA256,
        "runtime": "onnxruntime-cpu",
    } or file_sha256(ARTIFACT) != EXPECTED_ARTIFACT_SHA256:
        raise RetrospectiveError("receipt artifact identity is invalid")
    if receipt.get("historical") != prepared["reconstruction"]:
        raise RetrospectiveError("receipt historical reconstruction changed")
    historical = receipt["historical"]
    if (
        historical.get("days") != list(range(31))
        or len(historical.get("dates", [])) != 31
        or tuple(historical.get("service_order", ())) != SERVICE_ORDER
    ):
        raise RetrospectiveError("historical Day-0..30 contract is invalid")
    for service in SERVICE_ORDER:
        if len(historical["services"][service]) != 31:
            raise RetrospectiveError(f"historical {service} series length is invalid")
    expected_total = [
        _round(sum(historical["services"][service][day] for service in SERVICE_ORDER) / 5)
        for day in range(31)
    ]
    if historical["total"] != expected_total:
        raise RetrospectiveError("historical total is not the equal service mean")
    tape = receipt.get("tape", {})
    tape_days = tape.get("days")
    if not isinstance(tape_days, list) or len(tape_days) != 30:
        raise RetrospectiveError("receipt tape does not have 30 explicit days")
    if tape.get("sha256") != canonical_hash(tape_days):
        raise RetrospectiveError("receipt tape hash mismatch")
    for expected_day, shock in enumerate(tape_days, start=1):
        if not (
            shock.get("day") == expected_day
            and shock.get("type") is None
            and shock.get("severity") == 0.0
            and shock.get("impact") == [0.0] * 5
            and shock.get("public_risk_before") == [0.0] * 5
            and shock.get("public_risk_next") == [0.0] * 5
            and shock.get("assessment_tail") == (expected_day >= 28)
        ):
            raise RetrospectiveError(f"receipt tape Day {expected_day} is not frozen no-shock")
    from backend.app.city.environment import ACTION_ORDER, OBSERVATION_ORDER
    from backend.app.city.outcome import summarize_trajectory
    from backend.app.models import Scenario
    from model.policy import ACTION_COUNT, OBSERVATION_COUNT, load_policy

    scenario = Scenario.model_validate(receipt["scenario"])
    if scenario.model_dump(mode="json") != prepared["scenario"]:
        raise RetrospectiveError("receipt scenario differs from prepared scenario")
    if not (
        len(OBSERVATION_ORDER) == OBSERVATION_COUNT == 73
        and len(ACTION_ORDER) == ACTION_COUNT == 22
    ):
        raise RetrospectiveError("runtime tensor interface is not 73-in/22-out")
    load_policy(ARTIFACT, expected_sha256=EXPECTED_ARTIFACT_SHA256)
    for planner_id, expected_label in (
        ("v4", "onnx_policy"),
        ("reactive", "reactive_public_state_heuristic"),
    ):
        planner = receipt.get("planners", {}).get(planner_id, {})
        summary = planner.get("summary", {})
        trajectory = summary.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != 30:
            raise RetrospectiveError(f"{planner_id} trajectory length is invalid")
        if summary.get("planner") != expected_label:
            raise RetrospectiveError(f"{planner_id} canonical helper label changed")
        recomputed_summary = summarize_trajectory(expected_label, trajectory, scenario)
        if summary != recomputed_summary:
            raise RetrospectiveError(f"{planner_id} canonical summary does not recompute")
        if [row.get("shock") for row in trajectory] != tape_days:
            raise RetrospectiveError(f"{planner_id} trajectory used a different tape")
        recomputed_payload = _planner_payload(planner_id, summary, scenario)
        if planner.get("series") != recomputed_payload["series"]:
            raise RetrospectiveError(f"{planner_id} displayed series does not recompute")
        if planner.get("trajectory_tape_sha256") != tape["sha256"]:
            raise RetrospectiveError(f"{planner_id} trajectory tape hash differs")
        if any(row.get("hard_violation_count") != 0 for row in trajectory):
            raise RetrospectiveError(f"{planner_id} contains a hard violation")
        residuals = [
            value
            for row in trajectory
            for value in row["logistics"]["conservation_residual"]
        ]
        if any(value != 0.0 for value in residuals):
            raise RetrospectiveError(f"{planner_id} conservation residual is not exactly 0.0")
        if any(len(row.get("raw_action", [])) != 22 for row in trajectory):
            raise RetrospectiveError(f"{planner_id} action evidence is not 22-dimensional")
    if receipt.get("synthetic_benchmark") != load_canonical_benchmark_rows():
        raise RetrospectiveError("receipt synthetic benchmark differs from bound evidence")
    expected_invariants = {
        "historical_data_frozen_before_planner_load": True,
        "policy_and_heuristic_received_identical_tape": True,
        "tape_has_no_secondary_shocks": True,
        "artifact_sha256_matched": True,
        "observation_count": 73,
        "action_count": 22,
        "all_hard_violation_counts_zero": True,
        "all_conservation_residuals_exactly_zero": True,
        "all_trajectory_hashes_verified": True,
        "canonical_explicit_tape_helpers_used": True,
        "deterministic_replay_bound_by_artifact_scenario_and_tape_hashes": True,
        "final_split_used": False,
    }
    if receipt.get("invariants") != expected_invariants:
        raise RetrospectiveError("receipt invariant block differs from recomputed checks")


def verify_outputs() -> None:
    for path in (RECEIPT, FRONTEND, REPORT):
        if path.is_symlink() or not path.is_file():
            raise RetrospectiveError(f"published output missing or unsafe: {path}")
    receipt = _read_object(RECEIPT)
    validate_receipt(receipt)
    expected_frontend = render_frontend(receipt)
    if FRONTEND.read_text(encoding="utf-8") != expected_frontend:
        raise RetrospectiveError("generated frontend retrospective data drifted")
    if REPORT.read_text(encoding="utf-8") != render_report(receipt):
        raise RetrospectiveError("retrospective report drifted")


def replay_outputs() -> None:
    """Rerun both canonical helpers and require byte-identical recorded summaries."""

    verify_outputs()
    receipt = _read_object(RECEIPT)
    from backend.app.city.environment import rollout_baseline, rollout_policy
    from backend.app.city.scenarios import Shock
    from backend.app.models import Scenario
    from model.policy import load_policy

    scenario = Scenario.model_validate(receipt["scenario"])
    schedule = [Shock(**item) for item in receipt["tape"]["days"]]
    policy = load_policy(ARTIFACT, expected_sha256=EXPECTED_ARTIFACT_SHA256)
    replayed = {
        "v4": rollout_policy(scenario, 0, policy.predict, schedule=schedule),
        "reactive": rollout_baseline(scenario, 0, schedule=schedule),
    }
    for planner_id, summary in replayed.items():
        recorded = receipt["planners"][planner_id]["summary"]
        if summary != recorded:
            raise RetrospectiveError(
                f"deterministic replay differs for {planner_id}; publication unchanged"
            )
        if summary["trajectory_sha256"] != recorded["trajectory_sha256"]:
            raise RetrospectiveError(
                f"deterministic replay trajectory hash differs for {planner_id}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-root",
        type=Path,
        help=(
            "out-of-repository official-source archive; required with --prepare "
            "or --rebind-provenance"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--rebind-provenance", action="store_true")
    mode.add_argument("--verify-archive-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_archive_root is not None:
        manifest = _read_object(SOURCE_MANIFEST)
        validate_sources(manifest)
        checked = verify_archive_root(manifest, args.verify_archive_root.resolve())
        print(f"Verified {checked} archived official source byte objects.")
        return 0
    if args.prepare:
        if args.archive_root is None:
            raise RetrospectiveError("--prepare requires --archive-root")
        if any(path.exists() or path.is_symlink() for path in (RECEIPT, FRONTEND, REPORT)):
            raise RetrospectiveError("cannot refreeze inputs after retrospective publication")
        contract = build_prepared_contract(args.archive_root)
        _write_json(PREPARED_CONTRACT, contract)
        print(
            json.dumps(
                {
                    "prepared": PREPARED_CONTRACT.relative_to(ROOT).as_posix(),
                    "contract_sha256": contract["contract_sha256"],
                    "policy_loaded": False,
                    "final_split_used": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.rebind_provenance:
        if args.archive_root is None:
            raise RetrospectiveError("--rebind-provenance requires --archive-root")
        receipt = rebind_provenance(args.archive_root)
        print(
            json.dumps(
                {
                    "receipt": RECEIPT.relative_to(ROOT).as_posix(),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "provenance_correction": PROVENANCE_CORRECTION_ID,
                    "planner_rerun": False,
                    "policy_loaded": False,
                    "final_split_used": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.run:
        if args.archive_root is not None:
            raise RetrospectiveError("--archive-root is only valid with --prepare")
        preflight_publication_targets()
        if PREPARED_CONTRACT.is_symlink() or not PREPARED_CONTRACT.is_file():
            raise RetrospectiveError("prepared contract is missing or unsafe")
        contract = _read_object(PREPARED_CONTRACT)
        receipt = run_retrospective(contract)
        validate_receipt(receipt)
        publish_create_new_bundle(
            {
                RECEIPT: _json_text(receipt),
                FRONTEND: render_frontend(receipt),
                REPORT: render_report(receipt),
            }
        )
        print(
            json.dumps(
                {
                    "receipt": RECEIPT.relative_to(ROOT).as_posix(),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "final_split_used": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.replay:
        replay_outputs()
        print("Hurricane Maria retrospective deterministic replay matched exactly.")
        return 0
    verify_outputs()
    print("Hurricane Maria retrospective receipt and generated outputs verified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetrospectiveError as exc:
        raise SystemExit(f"hurricane-maria retrospective error: {exc}") from exc
