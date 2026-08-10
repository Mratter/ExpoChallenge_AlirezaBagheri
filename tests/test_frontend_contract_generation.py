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


def test_tracked_frontend_contract_matches_deterministic_generator() -> None:
    expected = generator.render_contract().encode("utf-8")
    assert generator.OUTPUT.read_bytes() == expected


def test_generated_contract_preserves_backend_order_orientation_and_limits() -> None:
    source = generator.render_contract()

    assert _constant(source, "services") == list(SERVICES)
    assert _constant(source, "shockTypes") == list(SHOCKS)
    assert _constant(source, "SHOCK_IMPACTS") == {
        shock: SHOCK_IMPACTS[index].tolist()
        for index, shock in enumerate(SHOCKS)
    }
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

    assert generator.main(["--write", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == generator.render_contract()
    assert generator.main(["--check", "--output", str(output)]) == 0

    output.write_text(output.read_text(encoding="utf-8") + "// stale\n", encoding="utf-8")
    assert generator.main(["--check", "--output", str(output)]) == 1
    assert "frontend contract drifted" in capsys.readouterr().err


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
