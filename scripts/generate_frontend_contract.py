#!/usr/bin/env python3
"""Generate the frontend's v3 simulator contract from frozen Python values.

The Python release remains the source of truth. This development utility emits
TypeScript only; it never edits or seals the immutable v3 modules it imports.
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
    CompareRequestV3,
    ForcedShock,
    ScenarioV3,
    ServiceName,
    ShockName,
)
from backend.app.simulator_core import SERVICES, SHOCK_IMPACTS, SHOCKS  # noqa: E402
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_GROUPS_V3,
    ACTION_ORDER_V3,
    ACTION_SIZE_V3,
    CONSERVATION_TOLERANCE_V3,
    CRITICAL_SERVICE_FLOOR,
    CRITICAL_SERVICE_RATE_CAP,
    ENGINE_V3_ID,
    ENGINE_V3_VERSION,
    OBSERVATION_ORDER_V3,
    OBSERVATION_SIZE_V3,
    RESULT_SCHEMA_V3,
    SOLVED_RAUC_FLOOR,
    TERMINAL_PENDING_CAPACITY_MULTIPLIER,
)

OUTPUT = ROOT / "frontend" / "src" / "generated" / "backendContract.ts"

SCENARIO_V3_FIELDS = (
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
COMPARE_REQUEST_V3_FIELDS = ("seed", "scenario")

# Pydantic's JSON schema does not expose bounds enforced by model validators.
# These values are accepted/rejected against ScenarioV3 at generation time below.
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
            ScenarioV3.model_validate({**defaults, field: [value] * 5})
        for value in (minimum - epsilon, maximum + epsilon):
            try:
                ScenarioV3.model_validate({**defaults, field: [value] * 5})
            except ValueError:
                continue
            raise ContractGenerationError(
                f"{label} runtime bounds no longer match ScenarioV3 validators"
            )


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)


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
            for index, name in enumerate(ACTION_ORDER_V3)
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
    if len(OBSERVATION_ORDER_V3) != 73 or OBSERVATION_SIZE_V3 != 73:
        raise ContractGenerationError("v3 observation contract drifted")
    if len(ACTION_ORDER_V3) != 22 or ACTION_SIZE_V3 != 22:
        raise ContractGenerationError("v3 action contract drifted")

    field_orders = {
        "ScenarioV3": tuple(ScenarioV3.model_fields),
        "ForcedShock": tuple(ForcedShock.model_fields),
        "CompareRequestV3": tuple(CompareRequestV3.model_fields),
    }
    expected_field_orders = {
        "ScenarioV3": SCENARIO_V3_FIELDS,
        "ForcedShock": FORCED_SHOCK_FIELDS,
        "CompareRequestV3": COMPARE_REQUEST_V3_FIELDS,
    }
    for model_name, expected in expected_field_orders.items():
        if field_orders[model_name] != expected:
            raise ContractGenerationError(
                f"{model_name} field contract drifted: {field_orders[model_name]}"
            )

    impact_map = {shock: impacts[index] for index, shock in enumerate(shocks)}
    scenario_defaults = ScenarioV3().model_dump(mode="json")
    request_defaults = CompareRequestV3().model_dump(mode="json")
    _validate_runtime_vector_limits(scenario_defaults)
    scenario_schema = ScenarioV3.model_json_schema()
    forced_schema = ForcedShock.model_json_schema()
    request_schema = CompareRequestV3.model_json_schema()
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

export const observationOrderV3 = {_json(list(OBSERVATION_ORDER_V3))} as const
export const actionOrderV3 = {_json(list(ACTION_ORDER_V3))} as const
export const actionGroupsV3 = {_json(list(ACTION_GROUPS_V3))} as const
export const actionSlicesV3 = {_json(action_slices)} as const

export const environmentContractV3 = {_json({
        "id": ENGINE_V3_ID,
        "version": ENGINE_V3_VERSION,
        "schemaVersion": RESULT_SCHEMA_V3,
        "observationCount": OBSERVATION_SIZE_V3,
        "actionCount": ACTION_SIZE_V3,
        "assessmentTailDays": scenario_defaults["assessment_tail_days"],
        "resilienceAucFloor": SOLVED_RAUC_FLOOR,
        "criticalServiceFloor": CRITICAL_SERVICE_FLOOR,
        "criticalServiceDayRateCap": CRITICAL_SERVICE_RATE_CAP,
        "conservationTolerance": CONSERVATION_TOLERANCE_V3,
        "terminalPendingCapacityMultiplier": TERMINAL_PENDING_CAPACITY_MULTIPLIER,
    })} as const

export const scenarioV3FieldOrder = {_json(list(field_orders["ScenarioV3"]))} as const
export const forcedShockFieldOrder = {_json(list(field_orders["ForcedShock"]))} as const
export const compareRequestV3FieldOrder = {_json(list(field_orders["CompareRequestV3"]))} as const
export const requestLimitsV3 = {_json(limits)} as const

export type ForcedShock = {{
  day: number
  type: ShockType
  severity: number
}}

/** Fully materialized ScenarioV3 returned to and maintained by the client. */
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

export type CompareRequestV3 = {{ seed: number; scenario: Scenario }}

export const defaultScenarioV3: Scenario = {_json(scenario_defaults)}
export const defaultCompareRequestV3: CompareRequestV3 = {_json(request_defaults)}
"""
    return source.replace("\r\n", "\n").rstrip() + "\n"


def write_contract(path: Path = OUTPUT) -> None:
    payload = render_contract().encode("utf-8")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    expected = render_contract().encode("utf-8")
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"frontend contract drifted: {output}", file=sys.stderr)
            return 1
        print(f"frontend contract verified: {_display_path(output)}")
        return 0
    write_contract(output)
    print(f"frontend contract generated: {_display_path(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
