#!/usr/bin/env python3
"""Run the authorized development-only seed sweep and matched ablations.

The study is deliberately sequential because each training arm owns a 20-lane
process tree. It never imports or evaluates the final split. Baseline seed
receipts are reused as the matched controls for the three-seed ablations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_policy.py"
DEFAULT_OUTPUT_ROOT = ROOT / "internal" / "developmental_runs" / "v4" / "study-2m"
POLICY_SEEDS = (37_017, 47_017, 57_017, 67_017, 77_017)
ABLATION_SEEDS = POLICY_SEEDS[:3]


class StudyError(RuntimeError):
    """Raised when a training arm or its evidence violates the study contract."""


@dataclass(frozen=True)
class Arm:
    """One training treatment with all non-treatment settings made explicit."""

    name: str
    transitions: int = 2_000_000
    reward_profile: str = "v3_equivalent"
    preparedness_alignment_coefficient: float | None = 10.0
    bc_warm_start: bool = True
    vec_normalize: bool = True


BASELINE = Arm(name="adopted_v3_equivalent_2m")
ABLATIONS = (
    Arm(name="no_bc_warm_start", bc_warm_start=False),
    Arm(
        name="risk_averse_reward",
        reward_profile="risk_averse",
        preparedness_alignment_coefficient=2.0,
    ),
    Arm(name="no_vec_normalize", vec_normalize=False),
    Arm(name="preparedness_alignment_2", preparedness_alignment_coefficient=2.0),
    Arm(name="budget_645k", transitions=645_000),
)


def arm_directory(output_root: Path, arm: Arm, seed: int) -> Path:
    """Return the stable create-new evidence directory for one arm."""

    return output_root / arm.name / f"seed-{seed}"


def training_command(output_root: Path, arm: Arm, seed: int) -> list[str]:
    """Build the exact trainer invocation for one registered arm."""

    directory = arm_directory(output_root, arm, seed)
    command = [
        sys.executable,
        str(TRAINER),
        "--transitions",
        str(arm.transitions),
        "--lanes",
        "20",
        "--n-steps",
        "250",
        "--batch-size",
        "500",
        "--policy-seed",
        str(seed),
        "--learning-rate",
        "0.000075",
        "--target-kl",
        "0.02",
        "--ent-coef",
        "0.003",
        "--critic-warmup-min-transitions",
        "50000",
        "--critic-warmup-max-transitions",
        "100000",
        "--freeze-observation-rms",
        "--reward-profile",
        arm.reward_profile,
        "--preparedness-alignment-coefficient",
        str(arm.preparedness_alignment_coefficient),
        "--checkpoint-dir",
        str(directory / "checkpoints"),
        "--json-output",
        str(directory / "training-receipt.json"),
    ]
    command.append("--bc-warm-start" if arm.bc_warm_start else "--no-bc-warm-start")
    command.append("--vec-normalize" if arm.vec_normalize else "--no-vec-normalize")
    return command


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise StudyError(f"JSON evidence must be an object: {path}")
    return payload


def validate_training_receipt(path: Path, arm: Arm, seed: int) -> dict[str, Any]:
    """Fail closed on incomplete, mismatched, unsafe, or final-split evidence."""

    payload = _load_json(path)
    config = payload.get("config", {})
    checks = payload.get("checks", {})
    if (
        payload.get("status") != "complete"
        or payload.get("final_split_used") is not False
        or config.get("policy_seed") != seed
        or config.get("active_actor_critic_transitions") != arm.transitions
        or config.get("reward_profile") != arm.reward_profile
        or config.get("preparedness_alignment_coefficient")
        != arm.preparedness_alignment_coefficient
        or config.get("bc_warm_start") is not arm.bc_warm_start
        or config.get("vec_normalize") is not arm.vec_normalize
        or not checks.get("training_complete")
        or not checks.get("development_hard_violations_zero")
        or not checks.get("development_conservation_residuals_zero")
    ):
        raise StudyError(f"training receipt does not match registered arm: {path}")
    development = payload.get("development", {})
    if (
        development.get("case_count") != 40
        or not isinstance(development.get("solved_count"), int)
    ):
        raise StudyError(f"development result is incomplete: {path}")
    return payload


def run_arm(output_root: Path, arm: Arm, seed: int) -> dict[str, Any]:
    """Run or validate one arm without silently overwriting partial evidence."""

    directory = arm_directory(output_root, arm, seed)
    receipt_path = directory / "training-receipt.json"
    if receipt_path.exists():
        print(f"[study] reusing verified {arm.name} seed {seed}", flush=True)
        return validate_training_receipt(receipt_path, arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "trainer.log"
    if log_path.exists() or (directory / "checkpoints").exists():
        raise StudyError(
            f"partial arm exists without a receipt; inspect before retrying: {directory}"
        )
    print(
        f"[study] starting {arm.name} seed {seed} ({arm.transitions:,} active)",
        flush=True,
    )
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            training_command(output_root, arm, seed),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise StudyError(
            f"trainer failed for {arm.name} seed {seed}; see {log_path}"
        )
    payload = validate_training_receipt(receipt_path, arm, seed)
    print(
        f"[study] finished {arm.name} seed {seed}: "
        f"{payload['development']['solved_count']}/40",
        flush=True,
    )
    return payload


def _receipt_row(
    output_root: Path,
    arm: Arm,
    seed: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    development = payload["development"]
    return {
        "arm": arm.name,
        "seed": seed,
        "active_actor_critic_transitions": arm.transitions,
        "solved_count": development["solved_count"],
        "solve_rate": development["solve_rate"],
        "mean_resilience_auc": development["mean_resilience_auc"],
        "mean_minimum_tail_margin": development["mean_minimum_tail_margin"],
        "hard_violation_count": development["hard_violation_count"],
        "maximum_conservation_residual": development[
            "maximum_conservation_residual"
        ],
        "receipt": str(
            arm_directory(output_root, arm, seed) / "training-receipt.json"
        ),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a multi-seed arm with explicit sample standard deviations."""

    if len(rows) < 2:
        raise StudyError("multi-seed summary requires at least two rows")
    solved = [float(row["solved_count"]) for row in rows]
    rates = [float(row["solve_rate"]) for row in rows]
    return {
        "seed_count": len(rows),
        "mean_solved_count": fmean(solved),
        "sample_std_solved_count": stdev(solved),
        "mean_solve_rate": fmean(rates),
        "sample_std_solve_rate": stdev(rates),
        "minimum_solved_count": int(min(solved)),
        "maximum_solved_count": int(max(solved)),
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise StudyError(f"refusing to overwrite study summary: {path}") from exc


def run_seed_sweep(output_root: Path) -> dict[str, Any]:
    rows = [
        _receipt_row(
            output_root,
            BASELINE,
            seed,
            run_arm(output_root, BASELINE, seed),
        )
        for seed in POLICY_SEEDS
    ]
    payload = {
        "schema_version": 1,
        "tool": "run_training_study.py",
        "phase": "seed_sweep",
        "split": "dev",
        "final_split_used": False,
        "registered_policy_seeds": list(POLICY_SEEDS),
        "baseline": asdict(BASELINE),
        "rows": rows,
        "aggregate": summarize_rows(rows),
    }
    _write_new_json(output_root / "seed-sweep-summary.json", payload)
    print(json.dumps(payload["aggregate"], sort_keys=True), flush=True)
    return payload


def run_ablations(output_root: Path) -> dict[str, Any]:
    baseline_rows: dict[int, dict[str, Any]] = {}
    for seed in ABLATION_SEEDS:
        baseline_rows[seed] = _receipt_row(
            output_root,
            BASELINE,
            seed,
            validate_training_receipt(
                arm_directory(output_root, BASELINE, seed)
                / "training-receipt.json",
                BASELINE,
                seed,
            ),
        )
    comparisons: list[dict[str, Any]] = []
    for arm in ABLATIONS:
        treatment_rows = [
            _receipt_row(
                output_root,
                arm,
                seed,
                run_arm(output_root, arm, seed),
            )
            for seed in ABLATION_SEEDS
        ]
        paired = [
            {
                "seed": seed,
                "control_solved_count": baseline_rows[seed]["solved_count"],
                "treatment_solved_count": treatment["solved_count"],
                "treatment_minus_control_solved": (
                    treatment["solved_count"]
                    - baseline_rows[seed]["solved_count"]
                ),
            }
            for seed, treatment in zip(ABLATION_SEEDS, treatment_rows, strict=True)
        ]
        comparisons.append(
            {
                "treatment": asdict(arm),
                "control": asdict(BASELINE),
                "treatment_rows": treatment_rows,
                "treatment_aggregate": summarize_rows(treatment_rows),
                "paired_rows": paired,
                "mean_treatment_minus_control_solved": fmean(
                    row["treatment_minus_control_solved"] for row in paired
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "tool": "run_training_study.py",
        "phase": "ablations",
        "split": "dev",
        "final_split_used": False,
        "registered_policy_seeds": list(ABLATION_SEEDS),
        "baseline_receipts_reused": True,
        "comparisons": comparisons,
    }
    _write_new_json(output_root / "ablation-summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("seed-sweep", "ablations"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.phase == "seed-sweep":
        run_seed_sweep(output_root)
    else:
        run_ablations(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
