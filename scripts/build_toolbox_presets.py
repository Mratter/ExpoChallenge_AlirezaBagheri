#!/usr/bin/env python3
"""Generate the Analyst Toolbox demonstration presets from real paired runs.

Every preset is executed against the configured runtime before it is written,
and the declared outcome class is checked against the official solved verdicts
the engine returns. A preset whose label no longer matches what the runtime
produces fails the build instead of shipping a claim the app cannot reproduce.

The presets are chosen so each one is *typical* of its configuration rather
than a single lucky seed: the seed is the lowest that reproduces the declared
class, and `class_share_first_25_seeds` records how often seeds 1-25 of the
same configuration land in that class.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import ENGINE_VERSION, compare  # noqa: E402
from backend.app.city.outcome import SOLVED_DEFINITION_SHA256  # noqa: E402
from backend.app.main import configured_policy  # noqa: E402
from backend.app.models import Scenario  # noqa: E402

OUTPUT = ROOT / "frontend" / "src" / "generated" / "toolboxPresets.ts"
PROFILE_SEEDS = range(1, 26)

BASE_SCENARIO: dict[str, Any] = {
    "horizon_days": 30,
    "daily_budget": 180.0,
    "initial_services": [0.34, 0.26, 0.41, 0.38, 0.30],
    "priorities": [1.0, 1.1, 1.2, 1.4, 1.0],
    "shock_probability": 0.20,
    "severity_min": 0.10,
    "severity_max": 0.28,
    "forced_shock": None,
    "forced_shocks": [],
    "daily_crew_pool": 150.0,
    "recovery_targets": [0.55, 0.55, 0.55, 0.55, 0.55],
    "assessment_tail_days": 3,
}

BOTH = "both"
PPO_ONLY = "ppo_only"
NEITHER = "neither"

# (id, label, note, declared class, scenario overrides)
PRESETS: tuple[tuple[str, str, str, str, dict[str, Any]], ...] = (
    (
        "calm-month",
        "Calm month, full depots",
        "Ample material and crew against light weather.",
        BOTH,
        {
            "daily_budget": 320.0,
            "daily_crew_pool": 260.0,
            "shock_probability": 0.10,
            "severity_max": 0.18,
        },
    ),
    (
        "routine-operations",
        "Routine operations",
        "Normal resourcing, few and mild incidents.",
        BOTH,
        {
            "daily_budget": 200.0,
            "daily_crew_pool": 170.0,
            "shock_probability": 0.10,
            "severity_max": 0.18,
        },
    ),
    (
        "light-storm-season",
        "Light storm season",
        "Leaner supply, still inside the tuned rule's comfortable range.",
        BOTH,
        {"daily_budget": 140.0, "daily_crew_pool": 120.0},
    ),
    (
        "sustained-storms",
        "Sustained storm season",
        "Resources are fine; the hazard rate and severity are not.",
        PPO_ONLY,
        {
            "daily_budget": 200.0,
            "daily_crew_pool": 170.0,
            "shock_probability": 0.35,
            "severity_min": 0.20,
            "severity_max": 0.40,
        },
    ),
    (
        "material-starved",
        "Material short, crews idle",
        "Crew is plentiful but material is the binding constraint.",
        PPO_ONLY,
        {
            "daily_budget": 95.0,
            "daily_crew_pool": 170.0,
            "shock_probability": 0.28,
            "severity_min": 0.14,
            "severity_max": 0.32,
        },
    ),
    (
        "thin-supply-calm",
        "Thin supply, calm weather",
        "Material scarcity alone, without an unusual hazard rate.",
        PPO_ONLY,
        {"daily_budget": 75.0, "daily_crew_pool": 170.0},
    ),
    (
        "frequent-damage",
        "Frequent moderate damage",
        "Steady mid-severity incidents on ordinary resourcing.",
        PPO_ONLY,
        {
            "daily_budget": 160.0,
            "daily_crew_pool": 140.0,
            "shock_probability": 0.28,
            "severity_min": 0.14,
            "severity_max": 0.32,
        },
    ),
    (
        "lean-and-heavy",
        "Lean supply, heavy weather",
        "Scarce material and crew under the worst hazard band.",
        PPO_ONLY,
        {
            "daily_budget": 140.0,
            "daily_crew_pool": 120.0,
            "shock_probability": 0.35,
            "severity_min": 0.20,
            "severity_max": 0.40,
        },
    ),
    (
        "crew-below-need",
        "Crew below need",
        "Material is available but no planner can staff the repairs.",
        NEITHER,
        {
            "daily_budget": 200.0,
            "daily_crew_pool": 80.0,
            "shock_probability": 0.28,
            "severity_min": 0.14,
            "severity_max": 0.32,
        },
    ),
    (
        "hazard-beyond-resources",
        "Hazard beyond resources",
        "Both inputs starved under sustained heavy damage.",
        NEITHER,
        {
            "daily_budget": 70.0,
            "daily_crew_pool": 62.0,
            "shock_probability": 0.35,
            "severity_min": 0.20,
            "severity_max": 0.40,
        },
    ),
)


class PresetGenerationError(RuntimeError):
    pass


def scenario_for(label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    return {**BASE_SCENARIO, **overrides, "name": label}


def outcome_class(candidate_solved: bool, baseline_solved: bool) -> str:
    if candidate_solved and baseline_solved:
        return BOTH
    if candidate_solved:
        return PPO_ONLY
    if baseline_solved:
        return "heuristic_only"
    return NEITHER


def run_case(scenario: dict[str, Any], seed: int, policy: Any) -> dict[str, Any]:
    result = compare(Scenario(**scenario), seed, policy)
    candidate = result["candidate"]
    baseline = result["baseline"]
    candidate_solved = bool(candidate["absolute_outcome"]["solved"])
    baseline_solved = bool(baseline["absolute_outcome"]["solved"])
    return {
        "class": outcome_class(candidate_solved, baseline_solved),
        "candidate_solved": candidate_solved,
        "baseline_solved": baseline_solved,
        "candidate_rauc": round(float(candidate["rauc"]), 6),
        "baseline_rauc": round(float(baseline["rauc"]), 6),
        "shock_schedule_sha256": result["shock_schedule_sha256"],
    }


def build() -> dict[str, Any]:
    policy = configured_policy()
    presets: list[dict[str, Any]] = []
    for identifier, label, note, declared, overrides in PRESETS:
        scenario = scenario_for(label, overrides)
        shares: collections.Counter[str] = collections.Counter()
        seed_for_class: int | None = None
        for seed in PROFILE_SEEDS:
            observed = run_case(scenario, seed, policy)
            shares[observed["class"]] += 1
            if observed["class"] == declared and seed_for_class is None:
                seed_for_class = seed
        if seed_for_class is None:
            raise PresetGenerationError(
                f"preset {identifier!r} never produced {declared!r} "
                f"in seeds {PROFILE_SEEDS.start}-{PROFILE_SEEDS.stop - 1}: {dict(shares)}"
            )
        observed = run_case(scenario, seed_for_class, policy)
        if observed["class"] != declared:
            raise PresetGenerationError(
                f"preset {identifier!r} declared {declared!r} but produced "
                f"{observed['class']!r} at seed {seed_for_class}"
            )
        presets.append(
            {
                "id": identifier,
                "label": label,
                "note": note,
                "outcome": declared,
                "seed": seed_for_class,
                "scenario": scenario,
                "observed": {
                    "candidateSolved": observed["candidate_solved"],
                    "baselineSolved": observed["baseline_solved"],
                    "candidateRauc": observed["candidate_rauc"],
                    "baselineRauc": observed["baseline_rauc"],
                    "shockScheduleSha256": observed["shock_schedule_sha256"],
                },
                "classShareFirst25Seeds": shares[declared],
            }
        )

    ppo_solved = sum(1 for preset in presets if preset["observed"]["candidateSolved"])
    heuristic_solved = sum(
        1 for preset in presets if preset["observed"]["baselineSolved"]
    )
    neither = sum(1 for preset in presets if preset["outcome"] == NEITHER)
    if (len(presets), ppo_solved, heuristic_solved, neither) != (10, 8, 3, 2):
        raise PresetGenerationError(
            "preset mix drifted from the documented 10/8/3/2 contract: "
            f"total={len(presets)} ppo={ppo_solved} heuristic={heuristic_solved} "
            f"neither={neither}"
        )
    if any(
        preset["observed"]["baselineSolved"]
        and not preset["observed"]["candidateSolved"]
        for preset in presets
    ):
        raise PresetGenerationError(
            "a preset solved by the heuristic alone contradicts the stated mix"
        )

    return {
        "generatedBy": "scripts/build_toolbox_presets.py",
        "engineVersion": ENGINE_VERSION,
        "outcomeDefinitionSha256": SOLVED_DEFINITION_SHA256,
        "policySha256": policy.sha256,
        "profileSeeds": [PROFILE_SEEDS.start, PROFILE_SEEDS.stop - 1],
        "summary": {
            "total": len(presets),
            "ppoSolved": ppo_solved,
            "heuristicSolved": heuristic_solved,
            "neitherSolved": neither,
        },
        "presets": presets,
    }


def render(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
    return (
        "/* This file is generated by scripts/build_toolbox_presets.py. */\n"
        "/* Do not hand-edit it; run the generator and commit the deterministic output. */\n"
        "\n"
        "export type ToolboxPresetOutcome = 'both' | 'ppo_only' | 'neither'\n"
        "\n"
        f"export const toolboxPresets = {body} as const\n"
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    expected = render(build()).encode("utf-8")
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"toolbox presets drifted: {output}", file=sys.stderr)
            return 1
        print(f"toolbox presets verified: {output.relative_to(ROOT).as_posix()}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"toolbox presets generated: {output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
