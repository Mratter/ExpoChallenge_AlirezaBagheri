"""Replay persisted policy runs for local explanations and what-if analysis.

The helpers in this module are deliberately analysis-only.  They reuse the
canonical environment, disaster tape, policy, and outcome implementation and
never write a derived run to persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.city.environment import (
    ACTION_ORDER,
    ACTION_SIZE,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    CityRecoveryEnv,
)
from backend.app.city.outcome import summarize_trajectory
from backend.app.city.physics import SERVICES, round_vector
from backend.app.city.scenarios import Shock
from backend.app.models import Scenario as ScenarioModel
from backend.app.shared_evidence import canonical_hash
from model.policy import Policy, PolicyError


class AnalysisError(RuntimeError):
    """Raised when persisted evidence cannot be reproduced exactly."""


class CounterfactualRequest(BaseModel):
    """One allocation-only intervention applied to a single simulated day."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1, le=30)
    material_shares: list[float] | None = Field(
        default=None, min_length=5, max_length=5
    )
    crew_shares: list[float] | None = Field(
        default=None, min_length=5, max_length=5
    )

    @model_validator(mode="after")
    def validate_shares(self) -> "CounterfactualRequest":
        """Require at least one usable, finite five-service allocation."""

        if self.material_shares is None and self.crew_shares is None:
            raise ValueError("material_shares or crew_shares is required")
        for name, supplied in (
            ("material_shares", self.material_shares),
            ("crew_shares", self.crew_shares),
        ):
            if supplied is None:
                continue
            values = np.asarray(supplied, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(values < 0.0):
                raise ValueError(f"{name} cannot contain negative values")
            if float(values.max()) <= 0.0:
                raise ValueError(f"{name} must have a positive total")
        return self


@dataclass(frozen=True)
class VerifiedReplay:
    """The exact public observations and actions behind a persisted result."""

    scenario: ScenarioModel
    schedule: tuple[Shock, ...]
    observations: tuple[np.ndarray, ...]
    raw_actions: tuple[np.ndarray, ...]
    summary: dict[str, Any]


def _schedule_from_result(result: dict[str, Any]) -> tuple[Shock, ...]:
    try:
        schedule_payload = result["shock_schedule"]
        expected_hash = result["shock_schedule_sha256"]
        schedule = tuple(Shock(**item) for item in schedule_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("persisted disaster tape is invalid") from exc
    if canonical_hash(schedule_payload) != expected_hash:
        raise AnalysisError("persisted disaster tape hash does not match")
    return schedule


def _validate_policy_identity(result: dict[str, Any], policy: Policy) -> None:
    try:
        expected_sha256 = result["policy"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise AnalysisError("persisted policy identity is invalid") from exc
    if policy.sha256 != expected_sha256:
        raise PolicyError(
            "configured policy does not match the persisted result policy SHA-256"
        )


def verified_replay(result: dict[str, Any], policy: Policy) -> VerifiedReplay:
    """Replay a stored candidate and require byte-equivalent trajectory evidence."""

    _validate_policy_identity(result, policy)
    try:
        scenario = ScenarioModel.model_validate(result["scenario"])
        seed = int(result["seed"])
        persisted_candidate = result["candidate"]
        persisted_trajectory = persisted_candidate["trajectory"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("persisted candidate evidence is incomplete") from exc
    schedule = _schedule_from_result(result)
    if len(schedule) != scenario.horizon_days:
        raise AnalysisError("persisted disaster tape length does not match the scenario")

    environment = CityRecoveryEnv(scenario, seed, schedule)
    observation, reset_evidence = environment.reset(seed=seed)
    if reset_evidence.get("shock_schedule_sha256") != result["shock_schedule_sha256"]:
        raise AnalysisError("replay did not consume the persisted disaster tape")

    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    terminated = False
    while not terminated:
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        raw_action = np.asarray(policy.predict(observation), dtype=np.float64)
        if raw_action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(raw_action)):
            raise PolicyError("configured policy returned an invalid replay action")
        raw_actions.append(np.clip(raw_action, -1.0, 1.0))
        observation, _, terminated, _, _ = environment.step(raw_action)

    replay_summary = summarize_trajectory(
        "onnx_policy", environment.trajectory, scenario
    )
    if replay_summary["trajectory"] != persisted_trajectory:
        raise AnalysisError(
            "configured policy replay does not match the persisted candidate trajectory"
        )
    if replay_summary["trajectory_sha256"] != persisted_candidate.get(
        "trajectory_sha256"
    ):
        raise AnalysisError("persisted candidate trajectory hash does not match replay")
    for index, raw_action in enumerate(raw_actions):
        if round_vector(raw_action) != persisted_trajectory[index].get("raw_action"):
            raise AnalysisError(
                f"persisted candidate raw action does not match replay on day {index + 1}"
            )
    return VerifiedReplay(
        scenario=scenario,
        schedule=schedule,
        observations=tuple(observations),
        raw_actions=tuple(raw_actions),
        summary=replay_summary,
    )


def _predict_occlusion_batch(policy: Policy, observation: np.ndarray) -> np.ndarray:
    batch = np.repeat(observation.reshape(1, OBSERVATION_SIZE), OBSERVATION_SIZE, axis=0)
    batch[np.arange(OBSERVATION_SIZE), np.arange(OBSERVATION_SIZE)] = 0.0
    try:
        predicted = policy.session.run(
            ["action"], {"observation": batch.astype(np.float32, copy=False)}
        )[0]
    except Exception as exc:
        raise PolicyError("ONNX policy batched explanation inference failed") from exc
    actions = np.asarray(predicted, dtype=np.float64)
    if actions.shape != (OBSERVATION_SIZE, ACTION_SIZE):
        raise PolicyError(
            "batched explanation inference must return shape "
            f"({OBSERVATION_SIZE}, {ACTION_SIZE})"
        )
    if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
        raise PolicyError("batched explanation inference returned invalid actions")
    return actions


def build_explanations(result: dict[str, Any], policy: Policy) -> dict[str, Any]:
    """Return ordered, per-day single-channel action sensitivities."""

    replay = verified_replay(result, policy)
    days: list[dict[str, Any]] = []
    for day_index, (observation, base_action) in enumerate(
        zip(replay.observations, replay.raw_actions, strict=True)
    ):
        occluded_actions = _predict_occlusion_batch(policy, observation)
        action_deltas = occluded_actions - base_action.reshape(1, ACTION_SIZE)
        sensitivity = np.mean(np.abs(action_deltas), axis=1)
        total = float(sensitivity.sum())
        normalized = (
            sensitivity / total if total > 0.0 else np.zeros_like(sensitivity)
        )
        descending = np.argsort(-sensitivity, kind="stable")
        ranks = np.empty(OBSERVATION_SIZE, dtype=np.int64)
        ranks[descending] = np.arange(1, OBSERVATION_SIZE + 1)
        channels: list[dict[str, Any]] = []
        for observation_index, name in enumerate(OBSERVATION_ORDER):
            deltas = action_deltas[observation_index]
            action_index = int(np.argmax(np.abs(deltas)))
            channels.append(
                {
                    "observation_index": observation_index,
                    "observation_name": name,
                    "observed_value": round(float(observation[observation_index]), 10),
                    "mean_absolute_action_delta": round(
                        float(sensitivity[observation_index]), 10
                    ),
                    "normalized_influence": round(
                        float(normalized[observation_index]), 10
                    ),
                    "influence_rank": int(ranks[observation_index]),
                    "most_affected_action_index": action_index,
                    "most_affected_action": ACTION_ORDER[action_index],
                    "signed_action_delta": round(float(deltas[action_index]), 10),
                }
            )
        days.append(
            {
                "day": day_index + 1,
                "base_raw_action": round_vector(base_action),
                "channels": channels,
            }
        )

    return {
        "schema_version": "1.0.0",
        "result_id": result["result_id"],
        "method": {
            "id": "single-channel-zero-occlusion-action-sensitivity-v1",
            "description": (
                "Each observed channel is set to zero independently and the mean "
                "absolute change across the 22 raw policy actions is measured."
            ),
            "interpretation": (
                "This is a local action-sensitivity diagnostic, not a causal "
                "attribution or proof of why an outcome occurred."
            ),
            "causal": False,
            "occlusion_value": 0.0,
            "batch_size_per_day": OBSERVATION_SIZE,
            "normalization": "within-day sum of mean absolute action deltas",
            "future_tape_visible": False,
        },
        "policy": {
            "id": result["policy"]["id"],
            "sha256": result["policy"]["sha256"],
        },
        "shock_schedule_sha256": result["shock_schedule_sha256"],
        "future_tape_visible": False,
        "day_count": len(days),
        "observation_count": OBSERVATION_SIZE,
        "action_count": ACTION_SIZE,
        "observation_order": list(OBSERVATION_ORDER),
        "action_order": list(ACTION_ORDER),
        "days": days,
    }


def _normalized(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    supplied = np.asarray(values, dtype=np.float64)
    scaled = supplied / float(supplied.max())
    return scaled / float(scaled.sum())


def _shares_to_bounded_logits(shares: np.ndarray) -> np.ndarray:
    """Encode relative shares in the environment's bounded softmax coordinates."""

    logs = np.log(np.maximum(np.asarray(shares, dtype=np.float64), 1e-12))
    midpoint = 0.5 * (float(logs.min()) + float(logs.max()))
    return np.clip(logs - midpoint, -1.0, 1.0)


def _matches_realized_shares(shares: np.ndarray, allocation: list[float]) -> bool:
    """Recognize the unchanged persisted form values as an exact no-op."""

    realized = np.asarray(allocation, dtype=np.float64)
    total = float(realized.sum())
    return total > 0.0 and bool(
        np.allclose(shares, realized / total, rtol=0.0, atol=1e-6)
    )


def _summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "solved": summary["absolute_outcome"]["solved"],
        "rauc": summary["rauc"],
        "final_resilience": summary["final_resilience"],
        "minimum_resilience": summary["minimum_resilience"],
        "critical_service_days": summary["critical_service_days"],
        "hard_violation_count": summary["hard_violation_count"],
        "absolute_outcome": summary["absolute_outcome"],
        "trajectory_sha256": summary["trajectory_sha256"],
    }


def build_counterfactual(
    result: dict[str, Any], policy: Policy, request: CounterfactualRequest
) -> dict[str, Any]:
    """Replay one allocation override and let the same policy act thereafter."""

    replay = verified_replay(result, policy)
    material = _normalized(request.material_shares)
    crew = _normalized(request.crew_shares)
    environment = CityRecoveryEnv(
        replay.scenario, int(result["seed"]), replay.schedule
    )
    observation, reset_evidence = environment.reset(seed=int(result["seed"]))
    if reset_evidence.get("shock_schedule_sha256") != result["shock_schedule_sha256"]:
        raise AnalysisError("counterfactual did not consume the persisted disaster tape")

    original_selected = replay.summary["trajectory"][request.day - 1]

    terminated = False
    while not terminated:
        day = len(environment.trajectory) + 1
        action = np.asarray(policy.predict(observation), dtype=np.float64)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise PolicyError("configured policy returned an invalid counterfactual action")
        if day == request.day:
            action = action.copy()
            if material is not None and not _matches_realized_shares(
                material, original_selected["material_allocation"]
            ):
                action[:5] = _shares_to_bounded_logits(material)
            if crew is not None and not _matches_realized_shares(
                crew, original_selected["crew_allocation"]
            ):
                action[6:11] = _shares_to_bounded_logits(crew)
        observation, _, terminated, _, _ = environment.step(action)

    counterfactual = summarize_trajectory(
        "onnx_policy_counterfactual", environment.trajectory, replay.scenario
    )
    original_trajectory = replay.summary["trajectory"]
    counterfactual_trajectory = counterfactual["trajectory"]
    prefix_end = request.day - 1
    original_prefix_hash = canonical_hash(original_trajectory[:prefix_end])
    counterfactual_prefix_hash = canonical_hash(
        counterfactual_trajectory[:prefix_end]
    )
    if original_prefix_hash != counterfactual_prefix_hash:
        raise AnalysisError("counterfactual changed evidence before the selected day")

    daily_deltas: list[dict[str, Any]] = []
    for original_day, changed_day in zip(
        original_trajectory, counterfactual_trajectory, strict=True
    ):
        service_delta = np.asarray(changed_day["services_end"]) - np.asarray(
            original_day["services_end"]
        )
        preparedness_delta = np.asarray(changed_day["preparedness_end"]) - np.asarray(
            original_day["preparedness_end"]
        )
        daily_deltas.append(
            {
                "day": original_day["day"],
                "services_end": round_vector(service_delta),
                "preparedness_end": round_vector(preparedness_delta),
                "resilience": round(
                    changed_day["resilience"] - original_day["resilience"], 8
                ),
                "reward": round(changed_day["reward"] - original_day["reward"], 8),
            }
        )

    selected_index = request.day - 1
    original_selected = original_trajectory[selected_index]
    changed_selected = counterfactual_trajectory[selected_index]
    treatment = {
        "day": request.day,
        "material_shares": None if material is None else round_vector(material),
        "crew_shares": None if crew is None else round_vector(crew),
    }
    derived_hash = canonical_hash(
        {
            "result_id": result["result_id"],
            "policy_sha256": result["policy"]["sha256"],
            "shock_schedule_sha256": result["shock_schedule_sha256"],
            "treatment": treatment,
            "counterfactual_trajectory_sha256": counterfactual["trajectory_sha256"],
        }
    )
    return {
        "schema_version": "1.0.0",
        "result_id": result["result_id"],
        "analysis_id": derived_hash,
        "analysis_only": True,
        "persisted": False,
        "policy_sha256": result["policy"]["sha256"],
        "shock_schedule_sha256": result["shock_schedule_sha256"],
        "same_disaster_tape": True,
        "future_tape_visible": False,
        "treatment": treatment,
        "unchanged_prefix": {
            "days": prefix_end,
            "original_sha256": original_prefix_hash,
            "counterfactual_sha256": counterfactual_prefix_hash,
            "matches": True,
        },
        "selected_day_realized_allocations": {
            "services": list(SERVICES),
            "original": {
                "material": original_selected["material_allocation"],
                "crew": original_selected["crew_allocation"],
            },
            "counterfactual": {
                "material": changed_selected["material_allocation"],
                "crew": changed_selected["crew_allocation"],
            },
        },
        "original": _summary(replay.summary),
        "counterfactual": _summary(counterfactual),
        "daily_deltas": daily_deltas,
    }


__all__ = (
    "AnalysisError",
    "CounterfactualRequest",
    "build_counterfactual",
    "build_explanations",
    "verified_replay",
)
