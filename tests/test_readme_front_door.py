from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MARIA_REPORT = ROOT / "benchmarks" / "v4" / "hurricane-maria-retrospective.md"
LANDING_SOURCE = ROOT / "frontend" / "src" / "LandingPage.tsx"
LANDING_SCREENSHOT = "docs/screenshots/hurricane-maria-landing.png"
LANDING_SCREENSHOT_MANIFEST = ROOT / "docs/screenshots/hurricane-maria-landing.manifest.json"

SECTION_ORDER = (
    "## Quick Start",
    "## What Opens",
    "## Hurricane Maria Reconstruction",
    "## Measured Synthetic Results",
    "## How One Decision Becomes a Result",
    "## Runtime, API, and Documentation",
    "## Evidence and Reproducibility Boundary",
    "## Three Common Problems",
)

FINAL_COMPARATORS = (
    "Privileged future-aware CEM",
    "Shipped v4 PPO",
    "Tuned constant rule",
    "Preparedness teacher",
    "Causal MPC, k=5",
    "Legacy ONNX fixture",
    "Reactive heuristic",
)

INLINE_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
IMAGE_LINK = re.compile(r"!\[[^]]*]\(([^)]+)\)")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _section(readme: str, heading: str) -> str:
    start = readme.index(heading)
    next_heading = readme.find("\n## ", start + len(heading))
    return readme[start:] if next_heading == -1 else readme[start:next_heading]


def _table_first_column(section: str) -> tuple[str, ...]:
    rows = [line for line in section.splitlines() if line.startswith("|")]
    body = rows[2:]
    return tuple(
        re.sub(r"[*`]", "", row.split("|", maxsplit=2)[1]).strip()
        for row in body
    )


def _markdown_links_outside_fences(markdown: str) -> list[str]:
    links: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            links.extend(match.group(1) for match in INLINE_LINK.finditer(line))
    return links


def _link_destination(raw_destination: str) -> str:
    destination = raw_destination.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")]
    return destination.split(maxsplit=1)[0]


def _github_heading_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    occurrences: dict[str, int] = {}
    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match is None:
            continue
        heading = re.sub(r"<[^>]+>", "", match.group(1))
        heading = re.sub(r"!?\[([^]]+)]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"[^\w\- ]", "", heading.lower(), flags=re.UNICODE)
        base = re.sub(r"\s+", "-", heading.strip())
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def test_readme_is_a_short_runnable_front_door() -> None:
    readme = README.read_text(encoding="utf-8")
    lines = readme.splitlines()

    assert len(lines) <= 320
    assert lines.index("## Quick Start") + 1 <= 8

    fence_line = lines.index("```powershell") + 1
    command_line = next(
        index
        for index, line in enumerate(lines[fence_line:], start=fence_line + 1)
        if line.strip()
    )
    assert fence_line <= 15
    assert command_line <= 15

    headings = tuple(line for line in lines if line.startswith("## "))
    assert headings == SECTION_ORDER


def test_readme_uses_only_the_completed_landing_screenshot() -> None:
    readme = README.read_text(encoding="utf-8")
    images = IMAGE_LINK.findall(readme)

    assert images == [LANDING_SCREENSHOT]
    assert (ROOT / LANDING_SCREENSHOT).is_file()
    manifest = json.loads(LANDING_SCREENSHOT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["path"] == LANDING_SCREENSHOT
    assert manifest["sha256"] == hashlib.sha256(
        (ROOT / LANDING_SCREENSHOT).read_bytes()
    ).hexdigest()
    receipt = json.loads(
        (ROOT / "internal/retrospectives/hurricane-maria-30d.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["retrospective_receipt_sha256"] == receipt["receipt_sha256"]
    assert manifest["artifact_sha256"] == receipt["artifact"]["sha256"]

    legacy_content = (
        "docs/screenshots/3d-city.png",
        "docs/screenshots/trajectory.png",
        "docs/screenshots/dispatch-manifest.png",
        "docs/screenshots/decision-log.png",
        "docs/screenshots/decision-support.png",
        "these interface captures use the legacy regression fixture",
        "not the policy shown in the screenshots",
    )
    lowered = readme.lower()
    for legacy_item in legacy_content:
        assert legacy_item not in lowered


def test_readme_distinguishes_reconstruction_from_policy_simulations() -> None:
    maria_section = _section(
        README.read_text(encoding="utf-8"), "## Hurricane Maria Reconstruction"
    ).lower()

    assert "linear interpolation" in maria_section
    assert "shipped v4 and reactive-heuristic paths are simulation outputs" in maria_section
    assert "day-by-day service, shock, allocation, and recovery lines are simulation outputs" not in maria_section


def test_landing_source_uses_generated_metadata_for_substantive_numbers() -> None:
    source = LANDING_SOURCE.read_text(encoding="utf-8")
    stale_literals = (
        "const milestoneDays = [0, 10, 20, 30]",
        "Sep 20–Oct 20, 2017",
        "synthetic 200-case benchmark",
        "<b>1</b> frozen scenario",
        "Day 0–30 · 0–100",
        "full 73-input, 22-action trace",
        "days 0 through 30",
        "zero-to-100 scale",
    )
    for literal in stale_literals:
        assert literal not in source

    required_bindings = (
        "mariaRetrospective.display.milestoneDays",
        "mariaRetrospective.display.dayZeroLabel",
        "mariaRetrospective.display.dayEndLabel",
        "mariaRetrospective.display.horizonStart",
        "mariaRetrospective.display.dayEnd",
        "mariaRetrospective.display.dayCount",
        "mariaRetrospective.display.indexMin",
        "mariaRetrospective.display.indexMax",
        "mariaRetrospective.scenarioCount",
        "mariaRetrospective.syntheticBenchmarkCaseCount",
        "mariaRetrospective.interface.observationCount",
        "mariaRetrospective.interface.actionCount",
    )
    for binding in required_bindings:
        assert binding in source


def test_readme_final_comparator_table_has_all_seven_rows() -> None:
    readme = README.read_text(encoding="utf-8")
    results = _section(readme, "## Measured Synthetic Results")

    assert _table_first_column(results) == FINAL_COMPARATORS


def test_local_links_in_front_door_documents_resolve() -> None:
    documents = (README, *sorted((ROOT / "docs").glob("*.md")), MARIA_REPORT)
    failures: list[str] = []

    for document in documents:
        markdown = document.read_text(encoding="utf-8")
        for raw_destination in _markdown_links_outside_fences(markdown):
            destination = _link_destination(raw_destination)
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc:
                continue

            relative_path = unquote(parsed.path)
            target = document if not relative_path else document.parent / relative_path
            target = target.resolve()
            if not target.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {destination}")
                continue

            if parsed.fragment and target.suffix.lower() == ".md":
                anchors = _github_heading_anchors(target.read_text(encoding="utf-8"))
                if unquote(parsed.fragment).lower() not in anchors:
                    failures.append(
                        f"{document.relative_to(ROOT)} -> {destination} (missing anchor)"
                    )

    assert failures == []
