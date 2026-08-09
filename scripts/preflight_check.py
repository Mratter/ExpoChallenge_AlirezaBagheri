"""Fail-closed portable-runtime check for the selected PPO-v3 release.

This launcher check never trains, selects, or evaluates reserved final cases. It
loads the already authorized deployment and frozen aggregate evidence, performs
one ONNX smoke inference, and runs one ordinary operator scenario.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.models import ScenarioV3  # noqa: E402
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_ORDER_V3,
    OBSERVATION_ORDER_V3,
    compare_v3,
)
from model.ppo_v3 import load_policy_v3  # noqa: E402

SHA256 = re.compile(r"[0-9a-f]{64}")


def _smoke_action(bundle: object, observation_count: int, action_count: int) -> None:
    session = bundle.session  # type: ignore[attr-defined]
    output = session.run(
        ["action"],
        {"observation": np.zeros((1, observation_count), dtype=np.float32)},
    )[0]
    action = np.asarray(output)
    if action.shape != (1, action_count) or not np.all(np.isfinite(action)):
        raise RuntimeError("ONNX smoke inference returned an invalid action tensor.")


def _validate_v3_comparison(result: dict[str, object]) -> None:
    if result.get("engine_version") != "city-recovery-env-v3":
        raise RuntimeError("The V3 smoke comparison returned the wrong engine.")
    scenario = result.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("horizon_days") != 30:
        raise RuntimeError("The V3 smoke comparison returned an invalid scenario.")
    for planner_name in ("candidate", "baseline"):
        planner = result.get(planner_name)
        if not isinstance(planner, dict):
            raise RuntimeError(f"{planner_name} summary is missing.")
        trajectory = planner.get("trajectory")
        residual = planner.get("max_logistics_conservation_residual")
        if not isinstance(trajectory, list) or len(trajectory) != 30:
            raise RuntimeError(f"{planner_name} produced an incomplete trajectory.")
        if planner.get("hard_violation_count") != 0:
            raise RuntimeError(f"{planner_name} violated the V3 hard contract.")
        if (
            not isinstance(residual, (int, float))
            or isinstance(residual, bool)
            or not 0.0 <= float(residual) <= 1e-6
        ):
            raise RuntimeError(f"{planner_name} failed conservation verification.")
        trajectory_hash = planner.get("trajectory_sha256")
        if (
            not isinstance(trajectory_hash, str)
            or SHA256.fullmatch(trajectory_hash) is None
        ):
            raise RuntimeError(f"{planner_name} trajectory hash is invalid.")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required; found {sys.version.split()[0]}.")
    if len(OBSERVATION_ORDER_V3) != 73 or len(ACTION_ORDER_V3) != 22:
        raise RuntimeError("The PPO-v3 observation or action contract has changed.")

    bundle_v3 = load_policy_v3()
    _smoke_action(bundle_v3, len(OBSERVATION_ORDER_V3), len(ACTION_ORDER_V3))
    benchmark = bundle_v3.benchmark
    if (
        benchmark.get("status") != "complete"
        or benchmark.get("synthetic_case_count") != 40
        or benchmark.get("invariants", {}).get("candidate_hard_violations") != 0
        or benchmark.get("invariants", {}).get("baseline_hard_violations") != 0
    ):
        raise RuntimeError("The frozen V3 benchmark is incomplete or unsafe.")

    result = compare_v3(
        ScenarioV3(name="Portable runtime preflight"),
        424242,
        bundle_v3,
    )
    _validate_v3_comparison(result)

    print(
        json.dumps(
            {
                "benchmark_sha256": bundle_v3.benchmark_sha256,
                "manifest_schema_version": bundle_v3.manifest_schema_version,
                "manifest_sha256": bundle_v3.manifest_sha256,
                "model_id": bundle_v3.metadata["id"],
                "observation_count": len(OBSERVATION_ORDER_V3),
                "action_count": len(ACTION_ORDER_V3),
                "onnx_sha256": bundle_v3.onnx_sha256,
                "selected_checkpoint_sha256": bundle_v3.sb3_sha256,
                "scientific_source_sha256": bundle_v3.scientific_source_sha256,
                "smoke_outcome_pair": result["comparison"]["absolute_outcome_pair"],
                "status": "ppo-v3-portable-preflight-passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
