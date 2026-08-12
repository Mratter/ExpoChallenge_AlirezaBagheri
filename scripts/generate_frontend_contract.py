#!/usr/bin/env python3
"""Generate the frontend's simulator contract from canonical Python values.

The backend is the source of truth. This development utility emits deterministic
TypeScript so browser consumers cannot drift from the runtime physics contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, get_args

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models import (  # noqa: E402
    CompareRequest,
    ForcedShock,
    Scenario,
    ServiceName,
    ShockName,
)
from backend.app.city.environment import (  # noqa: E402
    ACTION_GROUPS,
    ACTION_ORDER,
    ACTION_SIZE,
    ENGINE_ID,
    ENGINE_VERSION,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    RESULT_SCHEMA,
)
from backend.app.city.outcome import (  # noqa: E402
    CONSERVATION_TOLERANCE,
    CRITICAL_SERVICE_FLOOR,
    CRITICAL_SERVICE_RATE_CAP,
    SOLVED_RAUC_FLOOR,
    TERMINAL_PENDING_CAPACITY_MULTIPLIER,
)
from backend.app.city.physics import SERVICES, SHOCK_IMPACTS, SHOCKS  # noqa: E402

OUTPUT = ROOT / "frontend" / "src" / "generated" / "backendContract.ts"
STYLES_OUTPUT = ROOT / "frontend" / "src" / "styles.css"

PAPER_COLOR = "#f7f8f8"
MIN_SECTOR_UI_CONTRAST = 3.0
SECTOR_UI_DARKEN_FACTOR = 0.76
SECTOR_PALETTE = {
    "transport": {"accent": "#5a8290", "body": "#6e8790"},
    "housing": {"accent": "#bd6b52", "body": "#b98269"},
    "food": {"accent": "#92906a", "body": "#7b7f4a"},
    "healthcare": {"accent": "#e6e2d8", "body": "#aab9b4"},
    "public_services": {"accent": "#71866a", "body": "#8b9a7f"},
}
SECTOR_CSS_START = "  /* BEGIN GENERATED SECTOR PALETTE — do not hand-edit. */"
SECTOR_CSS_END = "  /* END GENERATED SECTOR PALETTE. */"

SCENARIO_FIELDS = (
    "name",
    "horizon_days",
    "daily_budget",
    "initial_services",
    "priorities",
    "shock_probability",
    "severity_min",
    "severity_max",
    "forced_shock",
    "forced_shocks",
    "daily_crew_pool",
    "recovery_targets",
    "assessment_tail_days",
)
FORCED_SHOCK_FIELDS = ("day", "type", "severity")
COMPARE_REQUEST_FIELDS = ("seed", "scenario")

# Pydantic's JSON schema does not expose bounds enforced by model validators.
# These values are accepted/rejected against Scenario at generation time below.
RUNTIME_VECTOR_LIMITS = {
    "initialServices": ("initial_services", 0.05, 0.95),
    "priorities": ("priorities", 0.5, 2.0),
    "recoveryTargets": ("recovery_targets", 0.45, 0.75),
}


class ContractGenerationError(RuntimeError):
    """Raised when Python and TypeScript contracts cannot be reconciled."""


def _validate_runtime_vector_limits(defaults: dict[str, Any]) -> None:
    epsilon = 1e-9
    for label, (field, minimum, maximum) in RUNTIME_VECTOR_LIMITS.items():
        for value in (minimum, maximum):
            Scenario.model_validate({**defaults, field: [value] * 5})
        for value in (minimum - epsilon, maximum + epsilon):
            try:
                Scenario.model_validate({**defaults, field: [value] * 5})
            except ValueError:
                continue
            raise ContractGenerationError(
                f"{label} runtime bounds no longer match Scenario validators"
            )


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)


def _darken_hex(color: str, factor: float) -> str:
    """Scale sRGB channels to derive a darker, hue-preserving UI variant."""
    if not (len(color) == 7 and color.startswith("#")):
        raise ContractGenerationError(f"expected a six-digit hex color: {color}")
    try:
        channels = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    except ValueError as error:
        raise ContractGenerationError(f"expected a six-digit hex color: {color}") from error
    return "#" + "".join(f"{round(channel * factor):02x}" for channel in channels)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    luminances = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (luminances[1] + 0.05) / (luminances[0] + 0.05)


def _sector_palette(services: Sequence[str]) -> dict[str, dict[str, str]]:
    if set(SECTOR_PALETTE) != set(services):
        raise ContractGenerationError("sector palette must define every service exactly once")
    palette = {
        service: {
            **SECTOR_PALETTE[service],
            "ui": _darken_hex(
                SECTOR_PALETTE[service]["body"],
                SECTOR_UI_DARKEN_FACTOR,
            ),
        }
        for service in services
    }
    for service, colors in palette.items():
        contrast = _contrast_ratio(colors["ui"], PAPER_COLOR)
        if contrast < MIN_SECTOR_UI_CONTRAST:
            raise ContractGenerationError(
                f"{service} UI color must clear {MIN_SECTOR_UI_CONTRAST}:1 "
                f"against {PAPER_COLOR}; got {contrast:.3f}:1"
            )
    return palette


def render_sector_css() -> str:
    """Render the generated custom-property block embedded in styles.css."""
    palette = _sector_palette(list(SERVICES))
    declarations = "\n".join(
        f"  --sector-{service.replace('_', '-')}: {colors['ui']};"
        for service, colors in palette.items()
    )
    return f"{SECTOR_CSS_START}\n{declarations}\n{SECTOR_CSS_END}"


def render_stylesheet(source: str) -> str:
    """Replace only the generator-owned sector-token block in a stylesheet."""
    if source.count(SECTOR_CSS_START) != 1 or source.count(SECTOR_CSS_END) != 1:
        raise ContractGenerationError(
            "styles.css must contain one generated sector palette marker pair"
        )
    start = source.index(SECTOR_CSS_START)
    end = source.index(SECTOR_CSS_END, start) + len(SECTOR_CSS_END)
    return source[:start] + render_sector_css() + source[end:]


def _action_slices() -> dict[str, tuple[int, int]]:
    prefixes = (
        ("materialShares", "material_share_"),
        ("materialUtilization", "material_utilization"),
        ("crewShares", "crew_share_"),
        ("crewUtilization", "crew_utilization"),
        ("stockRelease", "stock_release_"),
        ("preparednessInvestment", "preparedness_investment_"),
    )
    slices: dict[str, tuple[int, int]] = {}
    for label, prefix in prefixes:
        indexes = [
            index
            for index, name in enumerate(ACTION_ORDER)
            if name == prefix or name.startswith(prefix)
        ]
        if not indexes or indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise ContractGenerationError(f"action group is not contiguous: {prefix}")
        slices[label] = (indexes[0], indexes[-1] + 1)
    return slices


def render_contract() -> str:
    services = list(SERVICES)
    shocks = list(SHOCKS)
    if len(services) != 5 or len(shocks) != 5:
        raise ContractGenerationError("the frontend Vector5 contract requires five services and shocks")
    if services != list(get_args(ServiceName)):
        raise ContractGenerationError("ServiceName literal order drifted")
    if shocks != list(get_args(ShockName)):
        raise ContractGenerationError("ShockName literal order drifted")
    impacts = SHOCK_IMPACTS.tolist()
    if len(impacts) != len(shocks) or any(len(row) != len(services) for row in impacts):
        raise ContractGenerationError("SHOCK_IMPACTS must be shock-by-service 5x5")
    if any(not math.isfinite(float(value)) for row in impacts for value in row):
        raise ContractGenerationError("SHOCK_IMPACTS must contain only finite values")
    if len(OBSERVATION_ORDER) != 73 or OBSERVATION_SIZE != 73:
        raise ContractGenerationError("observation contract drifted")
    if len(ACTION_ORDER) != 22 or ACTION_SIZE != 22:
        raise ContractGenerationError("action contract drifted")

    field_orders = {
        "Scenario": tuple(Scenario.model_fields),
        "ForcedShock": tuple(ForcedShock.model_fields),
        "CompareRequest": tuple(CompareRequest.model_fields),
    }
    expected_field_orders = {
        "Scenario": SCENARIO_FIELDS,
        "ForcedShock": FORCED_SHOCK_FIELDS,
        "CompareRequest": COMPARE_REQUEST_FIELDS,
    }
    for model_name, expected in expected_field_orders.items():
        if field_orders[model_name] != expected:
            raise ContractGenerationError(
                f"{model_name} field contract drifted: {field_orders[model_name]}"
            )

    impact_map = {shock: impacts[index] for index, shock in enumerate(shocks)}
    scenario_defaults = Scenario().model_dump(mode="json")
    request_defaults = CompareRequest().model_dump(mode="json")
    _validate_runtime_vector_limits(scenario_defaults)
    scenario_schema = Scenario.model_json_schema()
    forced_schema = ForcedShock.model_json_schema()
    request_schema = CompareRequest.model_json_schema()
    action_slices = {
        key: {"start": start, "end": end}
        for key, (start, end) in _action_slices().items()
    }
    limits = {
        "seed": {
            "minimum": request_schema["properties"]["seed"]["minimum"],
            "maximum": request_schema["properties"]["seed"]["maximum"],
        },
        "name": {
            "minimumLength": scenario_schema["properties"]["name"]["minLength"],
            "maximumLength": scenario_schema["properties"]["name"]["maxLength"],
        },
        "horizonDays": {
            "constant": scenario_schema["properties"]["horizon_days"]["const"],
        },
        "dailyBudget": {
            "minimum": scenario_schema["properties"]["daily_budget"]["minimum"],
            "maximum": scenario_schema["properties"]["daily_budget"]["maximum"],
        },
        "shockProbability": {
            "minimum": scenario_schema["properties"]["shock_probability"]["minimum"],
            "maximum": scenario_schema["properties"]["shock_probability"]["maximum"],
        },
        "severityMin": {
            "minimum": scenario_schema["properties"]["severity_min"]["minimum"],
            "maximum": scenario_schema["properties"]["severity_min"]["maximum"],
        },
        "severityMax": {
            "minimum": scenario_schema["properties"]["severity_max"]["minimum"],
            "maximum": scenario_schema["properties"]["severity_max"]["maximum"],
        },
        "dailyCrewPool": {
            "minimum": scenario_schema["properties"]["daily_crew_pool"]["minimum"],
            "maximum": scenario_schema["properties"]["daily_crew_pool"]["maximum"],
        },
        "forcedShockDay": {
            "minimum": forced_schema["properties"]["day"]["minimum"],
            "maximum": forced_schema["properties"]["day"]["maximum"],
        },
        "forcedShockSeverity": {
            "minimum": forced_schema["properties"]["severity"]["minimum"],
            "maximum": forced_schema["properties"]["severity"]["maximum"],
        },
        "assessmentTailDays": {
            "constant": scenario_schema["properties"]["assessment_tail_days"]["const"],
        },
    }
    for label, (field, minimum, maximum) in RUNTIME_VECTOR_LIMITS.items():
        field_schema = scenario_schema["properties"][field]
        if field_schema["minItems"] != 5 or field_schema["maxItems"] != 5:
            raise ContractGenerationError(f"{field} must remain a five-element vector")
        limits[label] = {
            "length": field_schema["minItems"],
            "minimum": minimum,
            "maximum": maximum,
        }
    sector_palette = _sector_palette(services)
    source = f"""/* This file is generated by scripts/generate_frontend_contract.py. */
/* Do not hand-edit it; run the generator and commit the deterministic output. */

export const services = {_json(services)} as const
export type Service = (typeof services)[number]

export const shockTypes = {_json(shocks)} as const
export type ShockType = (typeof shockTypes)[number]

export type Vector5 = [number, number, number, number, number]
export type Vector22 = [
  number, number, number, number, number, number,
  number, number, number, number, number, number,
  number, number, number, number, number,
  number, number, number, number, number,
]

export const SHOCK_IMPACTS: Record<ShockType, Vector5> = {_json(impact_map)}

/** Shared diorama colors plus body-derived, contrast-safe light-panel variants. */
export const sectorPalette: Record<Service, {{ accent: string; body: string; ui: string }}> = {_json(sector_palette)}

export const observationOrder = {_json(list(OBSERVATION_ORDER))} as const
export const actionOrder = {_json(list(ACTION_ORDER))} as const
export const actionGroups = {_json(list(ACTION_GROUPS))} as const
export const actionSlices = {_json(action_slices)} as const

export const environmentContract = {_json({
        "id": ENGINE_ID,
        "version": ENGINE_VERSION,
        "schemaVersion": RESULT_SCHEMA,
        "observationCount": OBSERVATION_SIZE,
        "actionCount": ACTION_SIZE,
        "assessmentTailDays": scenario_defaults["assessment_tail_days"],
        "resilienceAucFloor": SOLVED_RAUC_FLOOR,
        "criticalServiceFloor": CRITICAL_SERVICE_FLOOR,
        "criticalServiceDayRateCap": CRITICAL_SERVICE_RATE_CAP,
        "conservationTolerance": CONSERVATION_TOLERANCE,
        "terminalPendingCapacityMultiplier": TERMINAL_PENDING_CAPACITY_MULTIPLIER,
    })} as const

export const scenarioFieldOrder = {_json(list(field_orders["Scenario"]))} as const
export const forcedShockFieldOrder = {_json(list(field_orders["ForcedShock"]))} as const
export const compareRequestFieldOrder = {_json(list(field_orders["CompareRequest"]))} as const
export const requestLimits = {_json(limits)} as const

export type ForcedShock = {{
  day: number
  type: ShockType
  severity: number
}}

/** Fully materialized Scenario returned to and maintained by the client. */
export type Scenario = {{
  name: string
  horizon_days: {scenario_defaults["horizon_days"]}
  daily_budget: number
  initial_services: Vector5
  priorities: Vector5
  shock_probability: number
  severity_min: number
  severity_max: number
  forced_shock: ForcedShock | null
  forced_shocks: ForcedShock[]
  daily_crew_pool: number
  recovery_targets: Vector5
  assessment_tail_days: {scenario_defaults["assessment_tail_days"]}
}}

export type CompareRequest = {{ seed: number; scenario: Scenario }}

export const defaultScenario: Scenario = {_json(scenario_defaults)}
export const defaultCompareRequest: CompareRequest = {_json(request_defaults)}
"""
    return source.replace("\r\n", "\n").rstrip() + "\n"


def _write_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_contract(path: Path = OUTPUT) -> None:
    _write_payload(path, render_contract().encode("utf-8"))


def write_stylesheet(path: Path = STYLES_OUTPUT) -> None:
    source = path.read_text(encoding="utf-8")
    _write_payload(path, render_stylesheet(source).encode("utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--styles-output", type=Path, default=STYLES_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    styles_output = args.styles_output.resolve()
    expected = render_contract().encode("utf-8")
    if not styles_output.is_file():
        if args.check:
            print(f"frontend styles missing: {styles_output}", file=sys.stderr)
            return 1
        raise ContractGenerationError(f"frontend styles missing: {styles_output}")
    expected_styles = render_stylesheet(
        styles_output.read_text(encoding="utf-8")
    ).encode("utf-8")
    if args.check:
        stale = False
        if not output.is_file() or output.read_bytes() != expected:
            print(f"frontend contract drifted: {output}", file=sys.stderr)
            stale = True
        if styles_output.read_bytes() != expected_styles:
            print(f"frontend sector styles drifted: {styles_output}", file=sys.stderr)
            stale = True
        if stale:
            return 1
        print(
            "frontend contract verified: "
            f"{_display_path(output)} and {_display_path(styles_output)}"
        )
        return 0
    write_contract(output)
    _write_payload(styles_output, expected_styles)
    print(
        "frontend contract generated: "
        f"{_display_path(output)} and {_display_path(styles_output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
