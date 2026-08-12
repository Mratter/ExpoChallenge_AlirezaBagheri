from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import scripts.generate_frontend_contract as generator
from backend.app.city.environment import (
    ACTION_GROUPS,
    ACTION_ORDER,
    OBSERVATION_ORDER,
)
from backend.app.city.physics import SERVICES, SHOCK_IMPACTS, SHOCKS
from backend.app.models import CompareRequest, ForcedShock, Scenario


def _constant(source: str, name: str) -> object:
    declaration = re.search(
        rf"export const {re.escape(name)}(?:[^=\n]*)? = ",
        source,
    )
    assert declaration is not None
    start = declaration.end()
    value, _ = json.JSONDecoder().raw_decode(source[start:])
    return value


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return sum(
        weight * channel
        for weight, channel in zip((0.2126, 0.7152, 0.0722), linear)
    )


def _contrast_ratio(first: str, second: str) -> float:
    darker, lighter = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (lighter + 0.05) / (darker + 0.05)


def test_tracked_frontend_contract_matches_deterministic_generator() -> None:
    expected = generator.render_contract().encode("utf-8")
    assert generator.OUTPUT.read_bytes() == expected
    stylesheet = generator.STYLES_OUTPUT.read_text(encoding="utf-8")
    assert generator.render_stylesheet(stylesheet) == stylesheet


def test_generated_contract_preserves_backend_order_orientation_and_limits() -> None:
    source = generator.render_contract()

    assert _constant(source, "services") == list(SERVICES)
    assert _constant(source, "shockTypes") == list(SHOCKS)
    assert _constant(source, "SHOCK_IMPACTS") == {
        shock: SHOCK_IMPACTS[index].tolist()
        for index, shock in enumerate(SHOCKS)
    }
    palette = _constant(source, "sectorPalette")
    assert palette == {
        "transport": {"accent": "#5a8290", "body": "#6e8790", "ui": "#54676d"},
        "housing": {"accent": "#bd6b52", "body": "#b98269", "ui": "#8d6350"},
        "food": {"accent": "#92906a", "body": "#7b7f4a", "ui": "#5d6138"},
        "healthcare": {"accent": "#e6e2d8", "body": "#aab9b4", "ui": "#818d89"},
        "public_services": {"accent": "#71866a", "body": "#8b9a7f", "ui": "#6a7561"},
    }
    stylesheet = generator.STYLES_OUTPUT.read_text(encoding="utf-8")
    for service, colors in palette.items():
        body_channels = [
            int(colors["body"][index : index + 2], 16) for index in (1, 3, 5)
        ]
        expected_ui = "#" + "".join(
            f"{round(channel * generator.SECTOR_UI_DARKEN_FACTOR):02x}"
            for channel in body_channels
        )
        assert colors["ui"] == expected_ui
        assert _contrast_ratio(colors["ui"], generator.PAPER_COLOR) >= 3.0
        css_name = service.replace("_", "-")
        assert f"--sector-{css_name}: {colors['ui']};" in stylesheet
    assert _constant(source, "observationOrder") == list(OBSERVATION_ORDER)
    assert _constant(source, "actionOrder") == list(ACTION_ORDER)
    assert _constant(source, "actionGroups") == list(ACTION_GROUPS)
    assert _constant(source, "scenarioFieldOrder") == list(Scenario.model_fields)
    assert _constant(source, "forcedShockFieldOrder") == list(ForcedShock.model_fields)
    assert _constant(source, "compareRequestFieldOrder") == list(
        CompareRequest.model_fields
    )
    assert _constant(source, "requestLimits") == {
        "seed": {"minimum": 0, "maximum": 4_294_967_295},
        "name": {"minimumLength": 1, "maximumLength": 64},
        "horizonDays": {"constant": 30},
        "dailyBudget": {"minimum": 50.0, "maximum": 500.0},
        "shockProbability": {"minimum": 0.0, "maximum": 0.35},
        "severityMin": {"minimum": 0.05, "maximum": 0.25},
        "severityMax": {"minimum": 0.10, "maximum": 0.40},
        "dailyCrewPool": {"minimum": 50.0, "maximum": 300.0},
        "forcedShockDay": {"minimum": 1, "maximum": 30},
        "forcedShockSeverity": {"minimum": 0.05, "maximum": 0.40},
        "assessmentTailDays": {"constant": 3},
        "initialServices": {"length": 5, "minimum": 0.05, "maximum": 0.95},
        "priorities": {"length": 5, "minimum": 0.5, "maximum": 2.0},
        "recoveryTargets": {"length": 5, "minimum": 0.45, "maximum": 0.75},
    }
    assert _constant(source, "defaultScenario") == Scenario().model_dump(mode="json")
    assert _constant(source, "defaultCompareRequest") == CompareRequest().model_dump(
        mode="json"
    )


def test_cli_can_create_check_and_reject_stale_external_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "nested" / "backendContract.ts"
    styles_output = tmp_path / "nested" / "styles.css"
    styles_output.parent.mkdir(parents=True)
    styles_output.write_text(
        f":root {{\n{generator.SECTOR_CSS_START}\n{generator.SECTOR_CSS_END}\n}}\n",
        encoding="utf-8",
    )
    arguments = [
        "--output",
        str(output),
        "--styles-output",
        str(styles_output),
    ]

    assert generator.main(["--write", *arguments]) == 0
    assert output.read_text(encoding="utf-8") == generator.render_contract()
    rendered_styles = styles_output.read_text(encoding="utf-8")
    assert generator.render_stylesheet(rendered_styles) == rendered_styles
    assert generator.main(["--check", *arguments]) == 0

    styles_output.write_text(
        styles_output.read_text(encoding="utf-8").replace("#54676d", "#000000"),
        encoding="utf-8",
    )
    assert generator.main(["--check", *arguments]) == 1
    assert "frontend sector styles drifted" in capsys.readouterr().err
    assert generator.main(["--write", *arguments]) == 0

    output.write_text(output.read_text(encoding="utf-8") + "// stale\n", encoding="utf-8")
    assert generator.main(["--check", *arguments]) == 1
    assert "frontend contract drifted" in capsys.readouterr().err


def test_service_ledger_consumes_sector_tokens_and_shows_target_marker() -> None:
    app = (generator.ROOT / "frontend" / "src" / "App.tsx").read_text(
        encoding="utf-8"
    )
    stylesheet = generator.STYLES_OUTPUT.read_text(encoding="utf-8")

    assert 'className="service-ledger-row" data-service={service}' in app
    assert "border-left: 3px solid var(--sector-row)" in stylesheet
    assert "background: var(--sector-row)" in stylesheet
    assert ".state-cell i::after" in stylesheet
    assert "left: 55%" in stylesheet


def test_generator_rejects_non_finite_shock_impacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impacts = SHOCK_IMPACTS.tolist()
    impacts[0][0] = float("nan")

    class NonFiniteImpacts:
        @staticmethod
        def tolist() -> list[list[float]]:
            return impacts

    monkeypatch.setattr(generator, "SHOCK_IMPACTS", NonFiniteImpacts())
    with pytest.raises(generator.ContractGenerationError, match="finite"):
        generator.render_contract()
