#!/usr/bin/env python3
"""Compare City Recovery policies on shared development or final tapes.

This is deliberately a nonauthorizing diagnostic. It reads policies but never
creates or changes governance files. Use the development split for diagnostics;
the final split exists only for an explicitly authorized reproducibility gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    ACTION_ORDER,
    OBSERVATION_ORDER,
    CityRecoveryEnv,
)
from backend.app.city.outcome import summarize_trajectory  # noqa: E402
from backend.app.city.planners import (  # noqa: E402
    preparedness_teacher_action,
    reactive_heuristic_action,
    tuned_rule_action,
)
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    generate_disaster_tape,
)
from model.policy import (  # noqa: E402
    ACTION_COUNT,
    OBSERVATION_COUNT,
    PolicyError,
    load_policy,
)

DEFAULT_ONNX_PATH = ROOT / "tests" / "fixtures" / "legacy_policy.onnx"
DEFAULT_POLICIES = (
    "heuristic",
    "teacher",
    f"onnx:{DEFAULT_ONNX_PATH.relative_to(ROOT).as_posix()}",
)

PolicyFn = Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]


class ProbeError(RuntimeError):
    """Raised when a requested probe cannot be run honestly."""


@dataclass(frozen=True)
class ProbeCase:
    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    scenario: Any
    schedule: tuple[Any, ...]


@dataclass(frozen=True)
class Policy:
    label: str
    kind: str
    action: PolicyFn


@dataclass(frozen=True)
class ProbeRow:
    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    solved: bool
    status: str
    reason_codes: tuple[str, ...]
    resilience_auc: float
    minimum_tail_margin: float
    critical_service_days: int
    hard_violation_count: int
    max_conservation_residual: float
    trajectory_sha256: str


def _resolve_artifact_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _onnx_policy(path: Path) -> Policy:
    if (
        len(OBSERVATION_ORDER) != OBSERVATION_COUNT
        or len(ACTION_ORDER) != ACTION_COUNT
    ):
        raise ProbeError("policy tensor contract does not match the environment")
    try:
        loaded_policy = load_policy(path)
    except PolicyError as exc:
        raise ProbeError(f"ONNX policy is incompatible: {path}: {exc}") from exc

    def action(observation: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            result = loaded_policy.predict(observation)
        except PolicyError as exc:
            raise ProbeError(f"ONNX policy emitted an invalid action: {path}") from exc
        return result, {"runtime": "onnxruntime-cpu", "path": str(path)}

    try:
        display_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(path)
    return Policy(f"onnx:{display_path}", "onnx", action)


def resolve_policy(spec: str) -> Policy:
    if spec == "heuristic":
        return Policy(spec, spec, reactive_heuristic_action)
    if spec == "teacher":
        return Policy(spec, spec, preparedness_teacher_action)
    if spec == "tuned":
        return Policy(spec, spec, tuned_rule_action)
    if spec.startswith("onnx:"):
        value = spec.removeprefix("onnx:")
        if not value:
            raise ProbeError("onnx:<path> requires a nonempty artifact path")
        return _onnx_policy(_resolve_artifact_path(value))
    if spec in {"mpc", "oracle"}:
        raise ProbeError(
            f"{spec!r} is reserved by this harness but is not implemented until "
            "Step 6; refusing to substitute a different planner"
        )
    raise ProbeError(
        f"unknown policy {spec!r}; expected heuristic, teacher, tuned, "
        "onnx:<path>, mpc, or oracle"
    )


def build_cases(split: str) -> list[ProbeCase]:
    if split == "dev":
        families, seeds = DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
    elif split == "final":
        # Keep the single-use final contract out of development-only imports.
        from backend.app.city.scenarios import FINAL_FAMILIES, FINAL_SEEDS

        families, seeds = FINAL_FAMILIES, FINAL_SEEDS
    else:  # Defensive for callers that bypass argparse.
        raise ProbeError(f"unsupported split: {split}")
    cases: list[ProbeCase] = []
    for family in families:
        for case_seed in seeds:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = tuple(generate_disaster_tape(scenario, tape_seed))
            cases.append(
                ProbeCase(
                    row_id=f"{family.id}:{case_seed}",
                    family_id=family.id,
                    case_seed=case_seed,
                    tape_seed=tape_seed,
                    scenario=scenario,
                    schedule=schedule,
                )
            )
    if len(cases) != 40:
        raise ProbeError(f"{split} split produced {len(cases)} cases, expected 40")
    return cases


def rollout(case: ProbeCase, policy: Policy) -> ProbeRow:
    env = CityRecoveryEnv(case.scenario, case.tape_seed, case.schedule)
    observation, _ = env.reset(seed=case.tape_seed)
    terminated = False
    while not terminated:
        action, evidence = policy.action(observation)
        if policy.kind == "onnx":
            observation, _, terminated, truncated, _ = env.step(action)
        else:
            # Match the published heuristic rollout, whose transparent decision
            # evidence is part of the sealed trajectory hash. Evidence does not
            # participate in physics or outcome scoring.
            observation, _, terminated, truncated, _ = env.step_with_evidence(
                action, evidence
            )
        if truncated:
            raise ProbeError(f"unexpected truncated episode for {case.row_id}")
    summary = summarize_trajectory(policy.label, env.trajectory, case.scenario)
    outcome = summary["absolute_outcome"]
    minimum_tail_margin = float(
        np.min(
            np.asarray(outcome["tail_minimum_services"], dtype=np.float64)
            - np.asarray(outcome["recovery_targets"], dtype=np.float64)
        )
    )
    return ProbeRow(
        row_id=case.row_id,
        family_id=case.family_id,
        case_seed=case.case_seed,
        tape_seed=case.tape_seed,
        solved=bool(outcome["solved"]),
        status=str(outcome["status"]),
        reason_codes=tuple(outcome["reason_codes"]),
        resilience_auc=float(summary["rauc"]),
        minimum_tail_margin=minimum_tail_margin,
        critical_service_days=int(summary["critical_service_days"]),
        hard_violation_count=int(summary["hard_violation_count"]),
        max_conservation_residual=float(
            summary["max_logistics_conservation_residual"]
        ),
        trajectory_sha256=str(summary["trajectory_sha256"]),
    )


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Return the exact two-sided binomial McNemar p-value."""

    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def aggregate(rows: Sequence[ProbeRow]) -> dict[str, Any]:
    if not rows:
        raise ProbeError("cannot aggregate an empty probe")
    reasons = Counter(
        reason
        for row in rows
        if not row.solved
        for reason in row.reason_codes
    )
    return {
        "case_count": len(rows),
        "solved_count": sum(row.solved for row in rows),
        "solve_rate": sum(row.solved for row in rows) / len(rows),
        "mean_resilience_auc": round(fmean(row.resilience_auc for row in rows), 10),
        "mean_minimum_tail_margin": round(
            fmean(row.minimum_tail_margin for row in rows), 10
        ),
        "hard_violation_count": sum(row.hard_violation_count for row in rows),
        "maximum_conservation_residual": max(
            row.max_conservation_residual for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
    }


def paired_contingency(
    left_rows: Sequence[ProbeRow], right_rows: Sequence[ProbeRow]
) -> dict[str, Any]:
    if len(left_rows) != len(right_rows):
        raise ProbeError("paired policies have different case counts")
    if [row.row_id for row in left_rows] != [row.row_id for row in right_rows]:
        raise ProbeError("paired policies are not aligned on the same ordered tapes")
    both = sum(left.solved and right.solved for left, right in zip(left_rows, right_rows))
    left_only = sum(
        left.solved and not right.solved for left, right in zip(left_rows, right_rows)
    )
    right_only = sum(
        right.solved and not left.solved for left, right in zip(left_rows, right_rows)
    )
    neither = len(left_rows) - both - left_only - right_only
    return {
        "both_solved": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "table": [[both, left_only], [right_only, neither]],
        "discordant_count": left_only + right_only,
        "exact_mcnemar_p_two_sided": round(
            exact_mcnemar_p(left_only, right_only), 12
        ),
    }


def run_probe(split: str, policies: Sequence[Policy]) -> dict[str, Any]:
    if len({policy.label for policy in policies}) != len(policies):
        raise ProbeError("policy labels must be unique")
    cases = build_cases(split)
    rows_by_policy: dict[str, list[ProbeRow]] = {
        policy.label: [] for policy in policies
    }
    # Case-major ordering guarantees every policy receives the exact same tape.
    for case in cases:
        for policy in policies:
            rows_by_policy[policy.label].append(rollout(case, policy))
    comparisons: dict[str, Any] = {}
    for left, right in combinations(policies, 2):
        comparisons[f"{left.label} vs {right.label}"] = paired_contingency(
            rows_by_policy[left.label], rows_by_policy[right.label]
        )
    return {
        "schema_version": 1,
        "tool": "evaluate",
        "authorizing": False,
        "split": split,
        "same_tapes": True,
        "policies": {
            policy.label: aggregate(rows_by_policy[policy.label])
            for policy in policies
        },
        "paired_comparisons": comparisons,
        "rows": rows_by_policy,
    }


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "rows"}


def print_human(result: dict[str, Any]) -> None:
    print(
        f"evaluate split={result['split']} cases=40 "
        f"authorizing={str(result['authorizing']).lower()} same_tapes=true"
    )
    for label, metrics in result["policies"].items():
        print(
            f"\n[{label}] solved={metrics['solved_count']}/{metrics['case_count']} "
            f"rate={metrics['solve_rate']:.3f} "
            f"mean_resilience_auc={metrics['mean_resilience_auc']:.10f} "
            f"hard_violations={metrics['hard_violation_count']} "
            f"max_conservation_residual={metrics['maximum_conservation_residual']}"
        )
        print(
            "failure_reason_codes="
            + json.dumps(metrics["failure_reason_code_histogram"], sort_keys=True)
        )
    print("\n[paired exact McNemar]")
    for label, values in result["paired_comparisons"].items():
        print(
            f"{label}: both={values['both_solved']} "
            f"left_only={values['left_only']} right_only={values['right_only']} "
            f"neither={values['neither']} "
            f"p={values['exact_mcnemar_p_two_sided']:.12g}"
        )
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "final"), required=True)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "repeatable: heuristic, teacher, tuned, onnx:<path>, mpc, or oracle; "
            "defaults to heuristic+teacher+the legacy ONNX regression fixture"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable aggregate JSON instead of the text report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = args.policy or list(DEFAULT_POLICIES)
        policies = [resolve_policy(spec) for spec in specs]
        result = run_probe(args.split, policies)
    except (ProbeError, OSError, ValueError) as exc:
        print(f"evaluate: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(serializable_result(result), indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
