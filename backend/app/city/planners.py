"""Produce causal public-state actions for city-recovery policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.app.city.physics import SHOCK_IMPACTS, round_vector

_PUBLIC_OBSERVATION_SIZE = 73


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    """Map positive relative weights into the policy's bounded logit space."""

    logs = np.log(np.maximum(np.asarray(weights, dtype=np.float64), 1e-9))
    centered = logs - float(np.mean(logs))
    scale = max(float(np.max(np.abs(centered))), 1e-9)
    return np.clip(centered / scale, -1.0, 1.0)


def reactive_heuristic_action(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the transparent causal baseline for the public observation."""

    public = np.asarray(observation, dtype=np.float64).reshape(-1)
    if public.shape != (_PUBLIC_OBSERVATION_SIZE,) or not np.all(np.isfinite(public)):
        raise ValueError(
            "heuristic requires the same 73 finite public inputs as PPO"
        )
    shocked = public[0:5]
    supplied_priorities = public[5:10] * 2.0
    stock_fraction = public[20:25]
    pending_pressure = public[25:30]
    throughput = public[30:35]
    recovery_targets = public[45:50]
    preparedness = public[55:60]
    public_risk_next = public[68:73]
    target_gap = np.maximum(0.01, recovery_targets - shocked)
    expected_impact = SHOCK_IMPACTS.T @ public_risk_next
    material_weights = (
        target_gap**0.45 + 0.12 * expected_impact
    ) * supplied_priorities**0.40
    crew_weights = (
        (target_gap**0.45 + 0.06 * expected_impact)
        * supplied_priorities**0.40
        * np.maximum(stock_fraction, 0.05) ** 0.15
        * np.maximum(throughput, 0.10) ** -0.25
    )
    relative_gap = float(np.mean(target_gap / np.maximum(recovery_targets, 0.01)))
    material_utilization = float(
        np.clip(
            0.99
            + 0.08 * relative_gap
            - 0.03 * float(np.mean(pending_pressure))
            - 0.01 * float(np.mean(stock_fraction)),
            0.90,
            1.0,
        )
    )
    crew_utilization = float(
        np.clip(
            0.99
            + 0.08 * relative_gap
            + 0.02 * float(np.mean(stock_fraction))
            - 0.02 * float(np.mean(pending_pressure)),
            0.90,
            1.0,
        )
    )
    release = np.clip(
        0.78 + 0.28 * target_gap / np.maximum(recovery_targets, 0.01),
        0.78,
        1.0,
    )
    preparedness_investment = np.clip(
        0.50 * expected_impact * (1.0 - preparedness),
        0.0,
        0.25,
    )
    action = np.concatenate(
        (
            weights_to_logits(material_weights),
            np.asarray([2.0 * material_utilization - 1.0]),
            weights_to_logits(crew_weights),
            np.asarray([2.0 * crew_utilization - 1.0]),
            2.0 * release - 1.0,
            2.0 * preparedness_investment - 1.0,
        )
    )
    return np.asarray(action, dtype=np.float64), {
        "baseline_id": "reactive-public-state-heuristic-v3",
        "baseline_version": "3.0.0",
        "uses_public_risk_signal": True,
        "future_tape_visible": False,
        "material_weights": round_vector(material_weights),
        "crew_weights": round_vector(crew_weights),
        "expected_public_impact": round_vector(expected_impact),
        "material_utilization": round(material_utilization, 8),
        "crew_utilization": round(crew_utilization, 8),
        "release_targets": round_vector(release),
        "preparedness_targets": round_vector(preparedness_investment),
    }


def preparedness_teacher_action(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the stronger public-only teacher used to warm-start the policy."""

    public = np.asarray(observation, dtype=np.float64).reshape(-1)
    action, _ = reactive_heuristic_action(public)
    preparedness = public[55:60]
    public_risk_next = public[68:73]
    expected_impact = SHOCK_IMPACTS.T @ public_risk_next
    preparedness_investment = np.clip(
        6.0 * expected_impact * (1.0 - preparedness),
        0.0,
        0.40,
    )
    action[17:22] = 2.0 * preparedness_investment - 1.0
    return np.asarray(action, dtype=np.float64), {
        "teacher_id": "public-preparedness-curriculum-v3",
        "teacher_version": "1.0.0",
        "uses_exact_public_observation": True,
        "future_tape_visible": False,
        "expected_public_impact": round_vector(expected_impact),
        "preparedness_targets": round_vector(preparedness_investment),
    }


def tuned_rule_action(
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the published public rule with its diagnosed preparedness constants."""

    public = np.asarray(observation, dtype=np.float64).reshape(-1)
    action, _ = reactive_heuristic_action(public)
    preparedness = public[55:60]
    public_risk_next = public[68:73]
    expected_impact = SHOCK_IMPACTS.T @ public_risk_next
    preparedness_investment = np.clip(
        10.0 * expected_impact * (1.0 - preparedness),
        0.0,
        0.50,
    )
    action[17:22] = 2.0 * preparedness_investment - 1.0
    return np.asarray(action, dtype=np.float64), {
        "planner_id": "tuned-constant-public-rule-v3",
        "preparedness_multiplier": 10.0,
        "preparedness_cap": 0.50,
        "future_tape_visible": False,
    }


__all__ = (
    "preparedness_teacher_action",
    "reactive_heuristic_action",
    "tuned_rule_action",
    "weights_to_logits",
)
