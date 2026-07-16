from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"
EXPECTED_SHA256 = "23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3"


def accepted_legacy_artifact() -> dict[str, object]:
    """Reproduce the accepted Gate 2 linear candidate without touching manifest v2."""
    return {
        "artifact_type": "deterministic_linear_policy_candidate",
        "calibration": {
            "candidate_count": 56,
            "objective": "mean weighted daily resilience AUC",
            "scenario_count": 5,
            "synthetic_only": True,
            "winning_objective": 0.51479517,
        },
        "disclosure": (
            "Deterministic grid-selected linear heuristic on synthetic scenarios; "
            "not PPO, not empirically trained, and not operational guidance."
        ),
        "feature_order": [
            "priority_deficit",
            "criticality",
            "marginal_gain",
            "network_centrality",
        ],
        "feature_weights": {
            "criticality": 0.2,
            "marginal_gain": 0.8,
            "network_centrality": 0.0,
            "priority_deficit": 0.0,
        },
        "id": "frozen-policy-candidate-v1",
        "version": "1.0.0",
    }


def main() -> None:
    payload = (
        json.dumps(accepted_legacy_artifact(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError("legacy artifact reproduction changed accepted bytes")
    ARTIFACT_PATH.write_bytes(payload)
    print(f"wrote accepted non-PPO legacy artifact sha256={digest}")


if __name__ == "__main__":
    main()
