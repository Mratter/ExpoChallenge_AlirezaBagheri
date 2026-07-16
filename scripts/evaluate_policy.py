from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.artifact import load_policy_bundle  # noqa: E402
from backend.app.scenarios import (  # noqa: E402
    HELD_OUT_FAMILIES,
    HELD_OUT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
)
from backend.app.simulator import canonical_hash, canonical_json_bytes, compare  # noqa: E402

PROTOCOL_PATH = ROOT / "evaluation" / "protocol.v1.json"
REPORT_PATH = ROOT / "evaluation" / "feature_complete_report.v1.json"
BOOTSTRAP_SEED = 1717
BOOTSTRAP_SAMPLES = 5_000


def read_protocol() -> tuple[dict[str, Any], str]:
    payload = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    expected_families = [family.id for family in HELD_OUT_FAMILIES]
    if protocol.get("held_out_scenario_families") != expected_families:
        raise RuntimeError("protocol held-out scenario families drifted from authored code")
    if protocol.get("evaluation_seeds") != list(HELD_OUT_SEEDS):
        raise RuntimeError("protocol evaluation seeds drifted from authored code")
    training_ids = {family.id for family in TRAINING_FAMILIES}
    if training_ids.intersection(expected_families):
        raise RuntimeError("training and held-out family ids overlap")
    if set(TRAINING_SEEDS).intersection(HELD_OUT_SEEDS):
        raise RuntimeError("training and held-out scenario seeds overlap")
    if protocol.get("training_exclusions", {}).get("seeds") != list(HELD_OUT_SEEDS):
        raise RuntimeError("protocol training exclusions are incomplete")
    return protocol, hashlib.sha256(payload).hexdigest()


def assert_invariants(result: dict[str, Any]) -> None:
    schedule = result["shock_schedule"]
    for planner_name in ("candidate", "baseline"):
        planner = result[planner_name]
        if planner["constraint_violations"] != 0:
            raise RuntimeError(f"{planner_name} reported a hard-constraint violation")
        for day, shock in zip(planner["trajectory"], schedule, strict=True):
            if day["shock"] != shock:
                raise RuntimeError(f"{planner_name} received a different shock tape")
            allocation = np.asarray(day["allocation"])
            lower = np.asarray(day["lower_bounds"])
            upper = np.asarray(day["upper_bounds"])
            budget = day["available_budget"]
            if abs(float(allocation.sum()) - budget) > 1e-7:
                raise RuntimeError(f"{planner_name} allocation sum invariant failed")
            if np.any(allocation < lower - 1e-7):
                raise RuntimeError(f"{planner_name} allocation lower invariant failed")
            if np.any(allocation > upper + 1e-7):
                raise RuntimeError(f"{planner_name} allocation upper invariant failed")
            if float(allocation.sum()) > budget + 1e-7:
                raise RuntimeError(f"{planner_name} allocation budget invariant failed")
            breakdown = day["projection"]["violation_breakdown"]
            if any(value != 0 for value in breakdown.values()):
                raise RuntimeError(f"{planner_name} serialized violation evidence is nonzero")


def mean(values: list[float]) -> float:
    return round(float(np.mean(np.asarray(values, dtype=np.float64))), 8)


def paired_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "method": "paired nonparametric bootstrap over complete scenario-seed units",
        "samples": BOOTSTRAP_SAMPLES,
        "mean": round(float(array.mean()), 8),
        "lower_95": round(float(np.quantile(means, 0.025)), 8),
        "upper_95": round(float(np.quantile(means, 0.975)), 8),
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rauc",
        "post_shock_recovery_shortfall_auc",
        "days_to_pre_shock_recovery_after_largest_loss",
        "critical_service_days",
    )
    result: dict[str, Any] = {}
    for metric in metrics:
        candidate = [float(case["candidate"][metric]) for case in cases]
        baseline = [float(case["baseline"][metric]) for case in cases]
        deltas = [left - right for left, right in zip(candidate, baseline, strict=True)]
        result[metric] = {
            "baseline_mean": mean(baseline),
            "candidate_mean": mean(candidate),
            "candidate_minus_baseline": mean(deltas),
        }
        if metric == "rauc":
            result[metric]["paired_95_interval"] = paired_interval(deltas)
    return result


def main() -> None:
    protocol, protocol_sha256 = read_protocol()
    bundle = load_policy_bundle()
    repeats = int(protocol["determinism"]["repeats"])
    cases: list[dict[str, Any]] = []
    determinism_mismatches = 0
    violation_totals = {
        "candidate": defaultdict(int),
        "baseline": defaultdict(int),
    }
    for family in HELD_OUT_FAMILIES:
        for seed in HELD_OUT_SEEDS:
            scenario = family.build(seed)
            results = [compare(scenario, seed, bundle) for _ in range(repeats)]
            hashes = [canonical_hash(result) for result in results]
            if len(set(hashes)) != 1:
                determinism_mismatches += 1
            first = results[0]
            assert_invariants(first)
            for planner_name in ("candidate", "baseline"):
                for key, value in first[planner_name]["violation_breakdown"].items():
                    violation_totals[planner_name][key] += value
            cases.append(
                {
                    "baseline": {
                        "critical_service_days": first["baseline"]["critical_service_days"],
                        "days_to_pre_shock_recovery_after_largest_loss": first["baseline"][
                            "days_to_pre_shock_recovery_after_largest_loss"
                        ],
                        "post_shock_recovery_shortfall_auc": first["baseline"][
                            "post_shock_recovery_shortfall_auc"
                        ],
                        "rauc": first["baseline"]["rauc"],
                        "trajectory_sha256": first["baseline"]["trajectory_sha256"],
                    },
                    "candidate": {
                        "critical_service_days": first["candidate"]["critical_service_days"],
                        "days_to_pre_shock_recovery_after_largest_loss": first["candidate"][
                            "days_to_pre_shock_recovery_after_largest_loss"
                        ],
                        "post_shock_recovery_shortfall_auc": first["candidate"][
                            "post_shock_recovery_shortfall_auc"
                        ],
                        "rauc": first["candidate"]["rauc"],
                        "trajectory_sha256": first["candidate"]["trajectory_sha256"],
                    },
                    "family_id": family.id,
                    "result_sha256": hashes[0],
                    "scenario_sha256": canonical_hash(scenario.model_dump(mode="json")),
                    "seed": seed,
                    "shock_schedule_sha256": first["shock_schedule_sha256"],
                }
            )
    if determinism_mismatches:
        raise RuntimeError(f"determinism failed for {determinism_mismatches} held-out cases")
    aggregate_metrics = aggregate(cases)
    by_family = {
        family.id: aggregate([case for case in cases if case["family_id"] == family.id])
        for family in HELD_OUT_FAMILIES
    }
    rauc_delta = aggregate_metrics["rauc"]["candidate_minus_baseline"]
    recovery_delta = aggregate_metrics["post_shock_recovery_shortfall_auc"][
        "candidate_minus_baseline"
    ]
    if rauc_delta > 0:
        outcome = "measured_resilience_improvement"
        outcome_statement = (
            "The fixed PPO/ONNX candidate has higher mean resilience AUC on this "
            "synthetic held-out protocol. Recovery deltas are reported separately and "
            "are not hidden when they favor the baseline."
        )
    else:
        outcome = "measured_tradeoff_or_baseline_advantage"
        outcome_statement = (
            "The fixed PPO/ONNX candidate does not improve mean resilience AUC on this "
            "synthetic held-out protocol; the measured resilience and recovery trade-off "
            "is reported without changing the baseline or holdout."
        )
    wins = sum(case["candidate"]["rauc"] > case["baseline"]["rauc"] for case in cases)
    losses = sum(case["candidate"]["rauc"] < case["baseline"]["rauc"] for case in cases)
    report = {
        "aggregate": aggregate_metrics,
        "baseline_id": "ortools-glop-visible-v1",
        "by_family": by_family,
        "candidate_id": bundle.metadata["id"],
        "cases": cases,
        "determinism": {
            "canonical_result_bytes": "identical",
            "mismatches": determinism_mismatches,
            "repeats_per_case": repeats,
        },
        "evaluation_case_count": len(cases),
        "held_out_family_count": len(HELD_OUT_FAMILIES),
        "limitations": [
            (
                "Every scenario, shock, coefficient, and policy training input is "
                "synthetic and non-empirical."
            ),
            (
                "The paired interval describes this authored finite protocol; it is not "
                "population or causal uncertainty."
            ),
            "No municipal effectiveness, equity, safety, or deployment claim is supported.",
        ],
        "onnx_sha256": bundle.onnx_sha256,
        "outcome": outcome,
        "outcome_statement": outcome_statement,
        "protocol_sha256": protocol_sha256,
        "recovery_shortfall_candidate_minus_baseline": recovery_delta,
        "resilience_case_counts": {
            "baseline_higher": losses,
            "candidate_higher": wins,
            "ties": len(cases) - wins - losses,
        },
        "sb3_checkpoint_sha256": bundle.sb3_sha256,
        "schema_version": "1.0.0",
        "split": {
            "evaluation_seeds": list(HELD_OUT_SEEDS),
            "held_out_family_ids": [family.id for family in HELD_OUT_FAMILIES],
            "training_family_ids": [family.id for family in TRAINING_FAMILIES],
            "training_seeds": list(TRAINING_SEEDS),
            "unit": protocol["split_unit"],
        },
        "synthetic_only": True,
        "violation_totals": {
            planner: dict(values) for planner, values in violation_totals.items()
        },
    }
    REPORT_PATH.write_bytes(canonical_json_bytes(report) + b"\n")
    print(
        json.dumps(
            {
                "baseline_rauc": aggregate_metrics["rauc"]["baseline_mean"],
                "candidate_rauc": aggregate_metrics["rauc"]["candidate_mean"],
                "cases": len(cases),
                "determinism_mismatches": determinism_mismatches,
                "outcome": outcome,
                "rauc_delta": rauc_delta,
                "report_sha256": hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest(),
                "violations": sum(sum(values.values()) for values in violation_totals.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
