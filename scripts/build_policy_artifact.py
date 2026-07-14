from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.models import Scenario  # noqa: E402
from backend.app.simulator import _run_planner, generate_shock_schedule  # noqa: E402

ARTIFACT_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"
FEATURES = ("priority_deficit", "criticality", "marginal_gain", "network_centrality")


def calibration_scenarios() -> list[tuple[Scenario, int]]:
    return [
        (
            Scenario(
                name=f"Synthetic calibration {index + 1}",
                horizon_days=10 + index,
                daily_budget=120 + 20 * index,
                initial_services=services,
                priorities=priorities,
                shock_probability=0.14 + 0.02 * index,
                severity_min=0.08,
                severity_max=0.22 + 0.01 * index,
                forced_shock=None,
            ),
            8100 + index,
        )
        for index, (services, priorities) in enumerate(
            [
                ([0.18, 0.52, 0.46, 0.31, 0.42], [1.4, 0.9, 1.1, 1.5, 1.0]),
                ([0.55, 0.21, 0.38, 0.47, 0.29], [0.8, 1.3, 1.2, 1.4, 1.1]),
                ([0.32, 0.36, 0.20, 0.28, 0.58], [1.1, 1.0, 1.6, 1.5, 0.8]),
                ([0.41, 0.48, 0.44, 0.17, 0.26], [1.0, 0.9, 1.1, 1.8, 1.3]),
                ([0.25, 0.29, 0.57, 0.51, 0.33], [1.3, 1.2, 0.8, 1.0, 1.4]),
            ]
        )
    ]


def candidate_weights() -> list[dict[str, float]]:
    candidates = []
    for parts in itertools.product(range(6), repeat=4):
        if sum(parts) == 5:
            candidates.append(dict(zip(FEATURES, (part / 5 for part in parts), strict=True)))
    return candidates


def score(weights: dict[str, float]) -> float:
    values = []
    for scenario, seed in calibration_scenarios():
        schedule = generate_shock_schedule(scenario, seed)
        result = _run_planner("frozen_policy", scenario, schedule, weights)
        values.append(result["rauc"])
    return float(np.mean(values))


def main() -> None:
    scored = (
        (score(weights), tuple(weights[name] for name in FEATURES), weights)
        for weights in candidate_weights()
    )
    ranked = sorted(
        scored,
        key=lambda item: (-item[0], item[1]),
    )
    best_score, _, best_weights = ranked[0]
    artifact = {
        "artifact_type": "deterministic_linear_policy_candidate",
        "calibration": {
            "candidate_count": len(ranked),
            "objective": "mean weighted daily resilience AUC",
            "scenario_count": len(calibration_scenarios()),
            "synthetic_only": True,
            "winning_objective": round(best_score, 8),
        },
        "disclosure": (
            "Deterministic grid-selected linear heuristic on synthetic scenarios; "
            "not PPO, not empirically trained, and not operational guidance."
        ),
        "feature_order": list(FEATURES),
        "feature_weights": best_weights,
        "id": "frozen-policy-candidate-v1",
        "version": "1.0.0",
    }
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ARTIFACT_PATH.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "artifacts": [
            {
                "bytes": len(payload),
                "id": "frozen-policy-v1",
                "license": "CC0-1.0",
                "path": "artifacts/frozen_policy.v1.json",
                "sha256": digest,
                "source": "scripts/build_policy_artifact.py",
            }
        ],
        "project": "AI17",
        "version": 1,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_PATH.name} sha256={digest}")


if __name__ == "__main__":
    main()
