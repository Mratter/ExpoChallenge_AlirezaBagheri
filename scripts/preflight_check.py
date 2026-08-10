"""Validate one explicitly configured policy against the portable runtime."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.city.environment import (  # noqa: E402
    ACTION_SIZE,
    ENGINE_ID,
    ENGINE_SPEC_SHA256,
    OBSERVATION_SIZE,
    compare,
    policy_identity,
)
from backend.app.city.outcome import SOLVED_DEFINITION_SHA256  # noqa: E402
from backend.app.models import Scenario  # noqa: E402
from model.policy import (  # noqa: E402
    ACTION_COUNT,
    OBSERVATION_COUNT,
    Policy,
    load_policy,
)

POLICY_PATH_ENV = "INNOVERSE_POLICY_PATH"
POLICY_SHA256_ENV = "INNOVERSE_POLICY_SHA256"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _load_configured_policy() -> Policy:
    """Load only the ONNX artifact explicitly selected by the operator."""

    policy_path = os.environ.get(POLICY_PATH_ENV, "").strip()
    if not policy_path:
        raise RuntimeError(
            f"{POLICY_PATH_ENV} is required; point it to the ONNX policy to serve."
        )
    expected_sha256 = os.environ.get(POLICY_SHA256_ENV)
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip() or None
    return load_policy(policy_path, expected_sha256=expected_sha256)


def _smoke_action(policy: Policy) -> None:
    """Require one finite, bounded action through the public policy interface."""

    action = policy.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float64))
    if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
        raise RuntimeError("Policy smoke inference returned an invalid action.")
    if not np.all(np.abs(action) <= 1.0):
        raise RuntimeError("Policy smoke inference returned an out-of-bounds action.")


def _validate_comparison(result: dict[str, Any]) -> None:
    """Check the ordinary smoke comparison's public safety invariants."""

    environment = result.get("environment")
    if not isinstance(environment, dict) or environment.get("id") != ENGINE_ID:
        raise RuntimeError("The smoke comparison returned the wrong environment.")
    scenario = result.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("horizon_days") != 30:
        raise RuntimeError("The smoke comparison returned an invalid scenario.")
    for planner_name in ("candidate", "baseline"):
        planner = result.get(planner_name)
        if not isinstance(planner, dict):
            raise RuntimeError(f"{planner_name} summary is missing.")
        trajectory = planner.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != 30:
            raise RuntimeError(f"{planner_name} produced an incomplete trajectory.")
        if planner.get("hard_violation_count") != 0:
            raise RuntimeError(f"{planner_name} violated the runtime hard contract.")
        if planner.get("max_logistics_conservation_residual") != 0.0:
            raise RuntimeError(f"{planner_name} failed exact conservation verification.")
        trajectory_hash = planner.get("trajectory_sha256")
        if (
            not isinstance(trajectory_hash, str)
            or SHA256.fullmatch(trajectory_hash) is None
        ):
            raise RuntimeError(f"{planner_name} trajectory hash is invalid.")


def main() -> None:
    """Run the local runtime gate without training or reserved-case evaluation."""

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required; found {sys.version.split()[0]}.")
    if (
        OBSERVATION_SIZE != OBSERVATION_COUNT
        or ACTION_SIZE != ACTION_COUNT
        or OBSERVATION_SIZE != 73
        or ACTION_SIZE != 22
    ):
        raise RuntimeError("The environment and ONNX policy contracts disagree.")

    policy = _load_configured_policy()
    _smoke_action(policy)
    result = compare(Scenario(name="Portable runtime preflight"), 424242, policy)
    _validate_comparison(result)
    identity = policy_identity(policy)

    print(
        json.dumps(
            {
                "action_count": ACTION_SIZE,
                "engine_id": ENGINE_ID,
                "engine_spec_sha256": ENGINE_SPEC_SHA256,
                "observation_count": OBSERVATION_SIZE,
                "outcome_definition_sha256": SOLVED_DEFINITION_SHA256,
                "policy_id": identity["id"],
                "policy_path": str(policy.path),
                "policy_sha256": policy.sha256,
                "smoke_outcome_pair": result["comparison"][
                    "absolute_outcome_pair"
                ],
                "status": "portable-runtime-preflight-passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
