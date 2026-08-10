"""Optimized parallel v4 simulator built around the frozen v3 contract.

V3 remains byte-identical for provenance and fallback use. V4 subclasses its
unchanged scenario generation, observations, physics helpers, and absolute
outcome definition; only action projection, evidence collection, and (in Step
3) reward assembly diverge here. The deliberate isolation prevents a v4 edit
from invalidating the sealed v3 model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np

from backend.app.models import ScenarioV3
from backend.app.simulator_core_v4 import (
    _round_vector,
    action_to_proposal,
    measure_constraints,
    project_capped_simplex,
)
from backend.app.simulator_v3 import (
    ACTION_ORDER_V3,
    ACTION_SIZE_V3,
    CONSERVATION_TOLERANCE_V3,
    CONSTRAINT_TOLERANCE,
    CREW_PRODUCTIVITY,
    CRITICAL_SERVICE_FLOOR,
    DELTA,
    DEPOT_CAPACITY,
    ETA,
    FOOD_SPOILAGE_RATE,
    IMMEDIATE_DELIVERY_FRACTION,
    OBSERVATION_ORDER_V3,
    OBSERVATION_SIZE_V3,
    PREPAREDNESS_DAILY_GAIN_CAP,
    PREPAREDNESS_DECAY,
    RESERVE_DRAW_FRACTION,
    SHOCK_IMPACTS,
    SOLVED_RAUC_FLOOR,
    CityRecoveryEnvV3,
    DayContextV3,
    InterventionV3,
    ShockV3,
    _land_capped_v3,
    _summarize_v3,
    absolute_outcome_v3,
    generate_disaster_tape_v3,
)

# The public v4 interface is the frozen v3 tensor contract by construction.
ACTION_ORDER_V4 = ACTION_ORDER_V3
ACTION_SIZE_V4 = ACTION_SIZE_V3
OBSERVATION_ORDER_V4 = OBSERVATION_ORDER_V3
OBSERVATION_SIZE_V4 = OBSERVATION_SIZE_V3
ENGINE_V4_ID = "CityRecoveryEnv-v4"
ENGINE_V4_VERSION = "4.0.0"
REWARD_PROFILES_V4 = ("v3_equivalent", "risk_averse")


def decode_action_v4(
    action: np.ndarray,
    context: DayContextV3,
    *,
    collect_evidence: bool = True,
) -> InterventionV3:
    supplied = np.asarray(action, dtype=np.float64).reshape(-1)
    if supplied.shape != (ACTION_SIZE_V3,) or not np.all(np.isfinite(supplied)):
        raise ValueError(
            f"v4 policy action must contain {ACTION_SIZE_V3} finite values"
        )
    raw = np.clip(supplied, -1.0, 1.0)
    material_minimum = float(context.material_lower.sum())
    material_gate = float((raw[5] + 1.0) / 2.0)
    material_used = material_minimum + material_gate * (
        context.available_budget - material_minimum
    )
    crew_minimum = float(context.crew_lower.sum())
    crew_gate = float((raw[11] + 1.0) / 2.0)
    crew_used = crew_minimum + crew_gate * (context.available_crew - crew_minimum)
    material_proposal = action_to_proposal(raw[:5], material_used)
    crew_proposal = action_to_proposal(raw[6:11], crew_used)
    material, material_projection = project_capped_simplex(
        material_proposal,
        material_used,
        context.material_lower,
        context.material_upper,
        collect_evidence=collect_evidence,
    )
    crew, crew_projection = project_capped_simplex(
        crew_proposal,
        crew_used,
        context.crew_lower,
        context.crew_upper,
        collect_evidence=collect_evidence,
    )
    stock_release = np.clip((raw[12:17] + 1.0) / 2.0, 0.0, 1.0)
    preparedness_requested = np.clip((raw[17:22] + 1.0) / 2.0, 0.0, 1.0)
    material_preparedness_cap = np.where(
        material > CONSTRAINT_TOLERANCE,
        1.0 - context.material_lower / np.maximum(material, CONSTRAINT_TOLERANCE),
        np.where(context.material_lower <= CONSTRAINT_TOLERANCE, 1.0, 0.0),
    )
    crew_preparedness_cap = np.where(
        crew > CONSTRAINT_TOLERANCE,
        1.0 - context.crew_lower / np.maximum(crew, CONSTRAINT_TOLERANCE),
        np.where(context.crew_lower <= CONSTRAINT_TOLERANCE, 1.0, 0.0),
    )
    preparedness_cap = np.clip(
        np.minimum(material_preparedness_cap, crew_preparedness_cap), 0.0, 1.0
    )
    preparedness_investment = np.minimum(preparedness_requested, preparedness_cap)
    return InterventionV3(
        material=material,
        crew=crew,
        stock_release=stock_release,
        preparedness_requested=preparedness_requested,
        preparedness_investment=preparedness_investment,
        material_used=material_used,
        material_unspent=context.available_budget - material_used,
        crew_used=crew_used,
        crew_idle=context.available_crew - crew_used,
        raw_action=raw,
        material_projection=material_projection,
        crew_projection=crew_projection,
    )

class CityRecoveryEnvV4(CityRecoveryEnvV3):
    """V4 transition system with optional compact training records."""

    def __init__(
        self,
        scenario: ScenarioV3,
        shock_seed: int = 0,
        schedule: Sequence[ShockV3] | None = None,
        *,
        collect_evidence: bool = True,
        reward_profile: str = "risk_averse",
    ):
        if not isinstance(collect_evidence, bool):
            raise TypeError("collect_evidence must be a boolean")
        if reward_profile not in REWARD_PROFILES_V4:
            raise ValueError(
                f"reward_profile must be one of {', '.join(REWARD_PROFILES_V4)}"
            )
        super().__init__(scenario, shock_seed, schedule)
        self.collect_evidence = collect_evidence
        self.reward_profile = reward_profile

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.collect_evidence:
            return super().reset(seed=seed, options=options)
        # Gym's base reset owns RNG initialization. The rest mirrors the frozen
        # transition reset while omitting the trainer-unused serialized tape
        # hash and engine evidence.
        gym.Env.reset(self, seed=seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
            if not isinstance(scenario, ScenarioV3):
                raise TypeError("reset option scenario must be a ScenarioV3")
            self.scenario = scenario
            self._provided_schedule = None
        if options and "shock_seed" in options:
            self.shock_seed = int(options["shock_seed"])
            self._provided_schedule = None
        elif seed is not None and self._provided_schedule is None:
            self.shock_seed = seed
        if self._provided_schedule is not None:
            if len(self._provided_schedule) != self.scenario.horizon_days:
                raise ValueError("precomputed v4 tape must match scenario horizon")
            self.schedule = list(self._provided_schedule)
        else:
            self.schedule = generate_disaster_tape_v3(
                self.scenario, self.shock_seed
            )
        self.trajectory = []
        self._q = np.asarray(self.scenario.initial_services, dtype=np.float64)
        self._stocks = DEPOT_CAPACITY * np.clip(self._q, 0.05, 0.50)
        self._pending = np.zeros(5, dtype=np.float64)
        self._damage_peak = np.zeros(5, dtype=np.float64)
        self._damage_duration = np.zeros(5, dtype=np.int64)
        self._damage_remaining = np.zeros(5, dtype=np.int64)
        self._priorities = np.asarray(self.scenario.priorities, dtype=np.float64)
        self._normalized_priorities = self._priorities / float(self._priorities.sum())
        self._targets = np.asarray(self.scenario.recovery_targets, dtype=np.float64)
        self._critical_streak = np.where(self._q < CRITICAL_SERVICE_FLOOR, 1, 0)
        self._preparedness = np.zeros(5, dtype=np.float64)
        self._days_since_last_shock = self.scenario.horizon_days
        self._previous_resilience = float(self._normalized_priorities @ self._q)
        self._day_index = 0
        self._terminated = False
        self._context = self._make_context()
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        intervention = decode_action_v4(
            action, self.current_context(), collect_evidence=self.collect_evidence
        )
        return self._advance(intervention, planner_evidence=None)

    def step_with_evidence(
        self, action: np.ndarray, planner_evidence: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        intervention = decode_action_v4(
            action, self.current_context(), collect_evidence=self.collect_evidence
        )
        return self._advance(intervention, planner_evidence=planner_evidence)

    def _advance(
        self,
        intervention: InterventionV3,
        *,
        planner_evidence: dict[str, Any] | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        context = self.current_context()
        material = intervention.material
        crew = intervention.crew
        material_measurements = measure_constraints(
            material,
            intervention.material_used,
            context.material_lower,
            context.material_upper,
        )
        crew_measurements = measure_constraints(
            crew,
            intervention.crew_used,
            context.crew_lower,
            context.crew_upper,
        )
        material_measurements["availability_violations"] = int(
            intervention.material_used < -CONSTRAINT_TOLERANCE
            or intervention.material_used
            > context.available_budget + CONSTRAINT_TOLERANCE
        )
        crew_measurements["availability_violations"] = int(
            intervention.crew_used < -CONSTRAINT_TOLERANCE
            or intervention.crew_used > context.available_crew + CONSTRAINT_TOLERANCE
        )
        material_measurements["total"] = sum(
            value for key, value in material_measurements.items() if key != "total"
        )
        crew_measurements["total"] = sum(
            value for key, value in crew_measurements.items() if key != "total"
        )
        if self.collect_evidence:
            intervention.material_projection["constraint_violations"] = (
                material_measurements["total"]
            )
            intervention.material_projection["violation_breakdown"] = (
                material_measurements
            )
            intervention.crew_projection["constraint_violations"] = crew_measurements[
                "total"
            ]
            intervention.crew_projection["violation_breakdown"] = crew_measurements

        preparedness_material_requested = (
            material * intervention.preparedness_investment
        )
        preparedness_crew = crew * intervention.preparedness_investment
        repair_material = material - preparedness_material_requested
        repair_crew = crew - preparedness_crew
        same_day_scheduled = material * IMMEDIATE_DELIVERY_FRACTION
        delayed_scheduled = material - same_day_scheduled
        stock_with_delivery, same_day_landed, same_day_held = _land_capped_v3(
            context.stock_ready, same_day_scheduled
        )
        preparedness_crew_capacity_effective = (
            preparedness_crew * CREW_PRODUCTIVITY * context.support
        )
        preparedness_crew_capacity_physical = (
            preparedness_crew_capacity_effective / np.maximum(context.throughput, 0.05)
        )
        total_release_budget = intervention.stock_release * stock_with_delivery
        preparedness_material = np.minimum.reduce(
            (
                stock_with_delivery,
                preparedness_material_requested,
                preparedness_crew_capacity_physical,
                total_release_budget,
            )
        )
        preparedness_material = np.maximum(0.0, preparedness_material)
        preparedness_effective_work = preparedness_material * context.throughput
        preparedness_crew_utilized = np.minimum(
            preparedness_crew,
            preparedness_effective_work
            / np.maximum(CREW_PRODUCTIVITY * context.support, 1e-9),
        )
        stock_after_preparedness = stock_with_delivery - preparedness_material
        release_remaining = np.maximum(
            0.0, total_release_budget - preparedness_material
        )
        repair_reserve = (
            RESERVE_DRAW_FRACTION * DEPOT_CAPACITY * (1.0 - context.shocked)
        )
        work_order_limit = repair_material + repair_reserve
        release_limit = release_remaining
        crew_capacity_effective = repair_crew * CREW_PRODUCTIVITY * context.support
        crew_capacity_physical = crew_capacity_effective / np.maximum(
            context.throughput, 0.05
        )
        physical_dispatch = np.minimum.reduce(
            (
                stock_after_preparedness,
                work_order_limit,
                release_limit,
                crew_capacity_physical,
            )
        )
        physical_dispatch = np.maximum(0.0, physical_dispatch)
        effective_repair = physical_dispatch * context.throughput
        stock_before_spoilage = np.maximum(
            0.0, stock_after_preparedness - physical_dispatch
        )
        spoilage = np.zeros(5, dtype=np.float64)
        spoilage[2] = FOOD_SPOILAGE_RATE * stock_before_spoilage[2]
        stock_end = stock_before_spoilage - spoilage
        pending_next = self._pending + same_day_held + delayed_scheduled
        capacity_overflow = context.pending_held + same_day_held
        backlog_pressure = float(
            np.mean(pending_next / np.maximum(pending_next + DEPOT_CAPACITY, 1e-9))
        )
        material_preparedness_ratio = preparedness_effective_work / max(
            0.20 * context.available_budget,
            1e-9,
        )
        crew_preparedness_ratio = preparedness_crew_utilized / max(
            0.20 * context.available_crew,
            1e-9,
        )
        preparedness_gain_requested = np.minimum(
            PREPAREDNESS_DAILY_GAIN_CAP,
            PREPAREDNESS_DAILY_GAIN_CAP
            * np.sqrt(
                np.maximum(0.0, material_preparedness_ratio)
                * np.maximum(0.0, crew_preparedness_ratio)
            )
            * context.support,
        )
        preparedness_base = np.clip(
            PREPAREDNESS_DECAY * context.preparedness_after_hazard, 0.0, 1.0
        )
        preparedness_gain = np.minimum(
            preparedness_gain_requested, 1.0 - preparedness_base
        )
        preparedness_end = preparedness_base + preparedness_gain
        gain = (
            ETA
            * np.sqrt(effective_repair / 200.0)
            * context.support
            * (1.0 - context.shocked)
        )
        strain = (
            DELTA
            * np.maximum(0.0, CRITICAL_SERVICE_FLOOR + 0.05 - context.shocked)
            * (1.0 - np.clip(effective_repair / context.available_budget, 0.0, 1.0))
        )
        end = np.clip(context.shocked + gain - strain, 0.0, 1.0)
        resilience = float(self._normalized_priorities @ end)
        shocked_resilience = float(self._normalized_priorities @ context.shocked)
        critical_count = int(np.count_nonzero(end < CRITICAL_SERVICE_FLOOR))
        target_shortfalls = np.maximum(0.0, context.recovery_targets - end)
        mean_target_gap = float(np.mean(target_shortfalls))
        worst_target_gap = float(np.max(target_shortfalls))
        target_margin = float(np.min(end - context.recovery_targets))
        critical_shortfall = float(
            np.mean(np.maximum(0.0, CRITICAL_SERVICE_FLOOR - end))
        )
        assessment_target_penalty = (
            2.0 * mean_target_gap if context.shock.assessment_tail else 0.0
        )
        expected_next_impact = SHOCK_IMPACTS.T @ context.public_risk_next
        preparedness_alignment = float(expected_next_impact @ preparedness_gain)
        projection_cost = 0.0001 * (
            intervention.material_projection["distance"] / context.available_budget
            + intervention.crew_projection["distance"] / context.available_crew
        )
        if self.reward_profile == "v3_equivalent":
            reward = (
                1.50 * resilience
                + 0.75 * (resilience - shocked_resilience)
                - 0.35 * critical_count
                - 1.00 * mean_target_gap
                - 1.20 * critical_shortfall
                - assessment_target_penalty
                - 0.25 * backlog_pressure
                + 10.0 * preparedness_alignment
                - projection_cost
            )
        else:
            # CVaR_(1/5): with five services, the one-sector lower tail is the
            # worst-served sector. Emergency recovery should protect that tail,
            # not let surplus in one service average away another's shortfall.
            reward = (
                1.50 * resilience
                + 0.75 * (resilience - shocked_resilience)
                - 0.35 * critical_count
                - 0.30 * mean_target_gap
                - 2.50 * worst_target_gap
                + 0.60 * float(np.clip(target_margin, -0.10, 0.05))
                - 1.20 * critical_shortfall
                - assessment_target_penalty
                - 0.25 * backlog_pressure
                + 2.00 * preparedness_alignment
                - projection_cost
            )
        conservation_residual = (
            context.stock_before
            + context.pending_arrivals
            + material
            + context.transfer_net
            - preparedness_material
            - physical_dispatch
            - spoilage
            - stock_end
            - pending_next
        )
        stock_bound_violations = int(
            np.count_nonzero(stock_end < -CONSTRAINT_TOLERANCE)
            + np.count_nonzero(stock_end > DEPOT_CAPACITY + CONSTRAINT_TOLERANCE)
            + np.count_nonzero(pending_next < -CONSTRAINT_TOLERANCE)
        )
        conservation_violation = int(
            float(np.max(np.abs(conservation_residual))) > CONSERVATION_TOLERANCE_V3
        )
        repair_floor_violations = int(
            np.count_nonzero(
                repair_material + CONSTRAINT_TOLERANCE < context.material_lower
            )
            + np.count_nonzero(repair_crew + CONSTRAINT_TOLERANCE < context.crew_lower)
        )
        hard_violation_count = (
            material_measurements["total"]
            + crew_measurements["total"]
            + stock_bound_violations
            + conservation_violation
            + repair_floor_violations
        )
        # The outcome calculation needs only these three rounded vectors. PPO
        # workers do not consume the much larger explanatory evidence record.
        # Keeping the compact fields rounded through the same helper makes the
        # reward and terminal outcome bit-for-bit identical in both modes.
        services_end_record = _round_vector(end)
        outcome_logistics = {
            "pending_next_day": _round_vector(pending_next),
            "conservation_residual": _round_vector(conservation_residual),
        }
        resilience_record = round(resilience, 8)
        reward_record = round(float(reward), 8)
        if self.collect_evidence:
            logistics = {
                "depot_capacity": _round_vector(DEPOT_CAPACITY),
                "depot_stock_before": _round_vector(context.stock_before),
                "pending_arrivals": _round_vector(context.pending_arrivals),
                "pending_arrivals_landed": _round_vector(context.pending_landed),
                "pending_arrivals_held": _round_vector(context.pending_held),
                "depot_stock_after_pending": _round_vector(
                    context.stock_after_pending
                ),
                "depot_damage_penalty": _round_vector(context.damage_penalty),
                "depot_damage_days_remaining": [
                    int(value) for value in context.damage_days_remaining
                ],
                "depot_damage_factor": _round_vector(context.depot_factor),
                "road_capacity": round(context.road_capacity, 8),
                "throughput_factor": _round_vector(context.throughput),
                "mutual_aid_transfers": [asdict(event) for event in context.transfers],
                "mutual_aid_net": _round_vector(context.transfer_net),
                "depot_stock_ready": _round_vector(context.stock_ready),
                "preparedness_material_requested": _round_vector(
                    preparedness_material_requested
                ),
                "preparedness_material_consumed": _round_vector(
                    preparedness_material
                ),
                "preparedness_effective_work": _round_vector(
                    preparedness_effective_work
                ),
                "depot_stock_after_preparedness": _round_vector(
                    stock_after_preparedness
                ),
                "repair_material_committed": _round_vector(repair_material),
                "preparedness_crew_assigned": _round_vector(preparedness_crew),
                "preparedness_crew_utilized": _round_vector(
                    preparedness_crew_utilized
                ),
                "preparedness_crew_capacity_effective": _round_vector(
                    preparedness_crew_capacity_effective
                ),
                "preparedness_crew_capacity_physical": _round_vector(
                    preparedness_crew_capacity_physical
                ),
                "repair_crew_assigned": _round_vector(repair_crew),
                "same_day_delivery_scheduled": _round_vector(same_day_scheduled),
                "same_day_delivery_landed": _round_vector(same_day_landed),
                "same_day_delivery_held": _round_vector(same_day_held),
                "delayed_delivery_scheduled": _round_vector(delayed_scheduled),
                "repair_reserve": _round_vector(repair_reserve),
                "repair_request": _round_vector(work_order_limit),
                "total_stock_release_budget": _round_vector(total_release_budget),
                "stock_release_remaining_after_preparedness": _round_vector(
                    release_remaining
                ),
                "stock_release_limit": _round_vector(release_limit),
                "crew_capacity_effective": _round_vector(crew_capacity_effective),
                "crew_capacity_physical": _round_vector(crew_capacity_physical),
                "repair_dispatch": _round_vector(physical_dispatch),
                "repair_supply": _round_vector(effective_repair),
                "spoilage": _round_vector(spoilage),
                "depot_stock_end": _round_vector(stock_end),
                "pending_next_day": outcome_logistics["pending_next_day"],
                "capacity_overflow": _round_vector(capacity_overflow),
                "conservation_residual": outcome_logistics[
                    "conservation_residual"
                ],
            }
            record = {
                "day": context.shock.day,
                "shock": asdict(context.shock),
                "available_budget": round(context.available_budget, 8),
                "available_crew": round(context.available_crew, 8),
                "material_used": round(intervention.material_used, 8),
                "material_unspent": round(intervention.material_unspent, 8),
                "crew_used": round(intervention.crew_used, 8),
                "crew_idle": round(intervention.crew_idle, 8),
                "services_before": _round_vector(context.before),
                "services_after_shock": _round_vector(context.shocked),
                "raw_action": _round_vector(intervention.raw_action),
                "allocation": _round_vector(material),
                "material_allocation": _round_vector(material),
                "crew_allocation": _round_vector(crew),
                "stock_release": _round_vector(intervention.stock_release),
                "preparedness_requested": _round_vector(
                    intervention.preparedness_requested
                ),
                "preparedness_investment": _round_vector(
                    intervention.preparedness_investment
                ),
                "preparedness_before": _round_vector(context.preparedness_before),
                "preparedness_after_hazard": _round_vector(
                    context.preparedness_after_hazard
                ),
                "preparedness_gain_requested": _round_vector(
                    preparedness_gain_requested
                ),
                "preparedness_gain": _round_vector(preparedness_gain),
                "preparedness_end": _round_vector(preparedness_end),
                "preparedness_alignment_reward": round(preparedness_alignment, 8),
                "backlog_pressure": round(backlog_pressure, 8),
                "lower_bounds": _round_vector(context.material_lower),
                "upper_bounds": _round_vector(context.material_upper),
                "crew_lower_bounds": _round_vector(context.crew_lower),
                "crew_upper_bounds": _round_vector(context.crew_upper),
                "projection": intervention.material_projection,
                "crew_projection": intervention.crew_projection,
                "planner_evidence": planner_evidence,
                "support": _round_vector(context.support),
                "throughput": _round_vector(context.throughput),
                "public_next_day_risk": _round_vector(context.public_risk_next),
                "gain": _round_vector(gain),
                "strain": _round_vector(strain),
                "services_end": services_end_record,
                "resilience": resilience_record,
                "reward": reward_record,
                "hard_violation_count": hard_violation_count,
                "hard_violation_breakdown": {
                    "material": material_measurements,
                    "crew": crew_measurements,
                    "stock_bounds": stock_bound_violations,
                    "conservation": conservation_violation,
                    "repair_floors": repair_floor_violations,
                },
                "logistics": logistics,
            }
        else:
            record = {
                "day": context.shock.day,
                "services_end": services_end_record,
                "resilience": resilience_record,
                "reward": reward_record,
                "hard_violation_count": hard_violation_count,
                "logistics": outcome_logistics,
            }
        self.trajectory.append(record)
        self._q = end
        self._stocks = stock_end
        self._pending = pending_next
        self._preparedness = preparedness_end
        self._damage_remaining = np.maximum(0, self._damage_remaining - 1)
        self._critical_streak = np.where(
            end < CRITICAL_SERVICE_FLOOR, self._critical_streak + 1, 0
        )
        self._previous_resilience = resilience
        self._day_index += 1
        self._terminated = self._day_index >= self.scenario.horizon_days
        if self._terminated:
            outcome = absolute_outcome_v3(
                self.trajectory,
                self.scenario.recovery_targets,
                self.scenario.assessment_tail_days,
            )
            if self.reward_profile == "v3_equivalent":
                tail_shortfall = float(
                    np.sum(
                        np.maximum(
                            0.0,
                            np.asarray(outcome["recovery_targets"])
                            - np.asarray(outcome["tail_minimum_services"]),
                        )
                    )
                )
            critical_excess = max(
                0,
                outcome["critical_service_days"] - outcome["critical_service_day_cap"],
            )
            if self.reward_profile == "v3_equivalent":
                terminal_bonus = (
                    12.0
                    if outcome["solved"]
                    else -(
                        8.0 * tail_shortfall
                        + 0.20 * critical_excess
                        + 5.0
                        * max(0.0, SOLVED_RAUC_FLOOR - outcome["resilience_auc"])
                        + 1.5 * len(outcome["reason_codes"])
                    )
                )
            else:
                realized_minimum_margin = float(
                    np.min(
                        np.asarray(outcome["tail_minimum_services"])
                        - np.asarray(outcome["recovery_targets"])
                    )
                )
                tail_targets_met = bool(
                    outcome["checks"]["assessment_tail_targets_met"]
                )
                target_component = (
                    12.0
                    + 40.0 * float(np.clip(realized_minimum_margin, 0.0, 0.05))
                    if tail_targets_met
                    else -12.0
                    - 40.0 * float(np.clip(-realized_minimum_margin, 0.0, 0.10))
                )
                non_target_failures = sum(
                    reason != "assessment_tail_targets_met"
                    for reason in outcome["reason_codes"]
                )
                terminal_bonus = (
                    target_component
                    - 0.20 * critical_excess
                    - 5.0
                    * max(0.0, SOLVED_RAUC_FLOOR - outcome["resilience_auc"])
                    - 1.5 * non_target_failures
                )
                record["terminal_minimum_target_margin"] = round(
                    realized_minimum_margin, 8
                )
                record["terminal_tail_targets_met"] = tail_targets_met
            reward += terminal_bonus
            record["terminal_bonus"] = round(float(terminal_bonus), 8)
            record["reward"] = round(float(reward), 8)
            record["absolute_outcome"] = outcome
            self._context = None
            observation = np.zeros(OBSERVATION_SIZE_V3, dtype=np.float32)
        else:
            self._context = self._make_context()
            observation = self._observation()
        return observation, float(reward), self._terminated, False, {"day": record}

    def render(self) -> list[dict[str, Any]]:
        return list(self.trajectory)


class CyclingScenarioEnvV4(gym.Env[np.ndarray, np.ndarray]):
    """Cycle deterministic training cases without constructing full evidence."""

    def __init__(
        self,
        scenarios: list[tuple[ScenarioV3, int]],
        *,
        collect_evidence: bool = True,
        reward_profile: str = "risk_averse",
    ):
        if not scenarios:
            raise ValueError("at least one v4 training scenario is required")
        # Training revisits this fixed authored set for millions of transitions.
        # Materialize each deterministic tape once per worker instead of
        # rebuilding and serializing it at every 30-step auto-reset.
        self.scenarios = [
            (
                scenario,
                shock_seed,
                tuple(generate_disaster_tape_v3(scenario, shock_seed)),
            )
            for scenario, shock_seed in scenarios
        ]
        self.index = 0
        first, first_seed, first_schedule = self.scenarios[0]
        self.inner = CityRecoveryEnvV4(
            first,
            first_seed,
            first_schedule,
            collect_evidence=collect_evidence,
            reward_profile=reward_profile,
        )
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scenario, shock_seed, schedule = self.scenarios[
            self.index % len(self.scenarios)
        ]
        self.index += 1
        self.inner.scenario = scenario
        self.inner.shock_seed = shock_seed
        self.inner._provided_schedule = schedule
        return self.inner.reset(seed=shock_seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.inner.step(action)

    def render(self) -> list[dict[str, Any]]:
        return self.inner.render()


def rollout_candidate_v4(
    scenario: ScenarioV3,
    seed: int,
    action_provider: Callable[[np.ndarray], np.ndarray],
    schedule: Sequence[ShockV3] | None = None,
    *,
    collect_evidence: bool = True,
    reward_profile: str = "risk_averse",
) -> dict[str, Any]:
    shared_schedule = (
        generate_disaster_tape_v3(scenario, seed)
        if schedule is None
        else list(schedule)
    )
    env = CityRecoveryEnvV4(
        scenario,
        seed,
        shared_schedule,
        collect_evidence=collect_evidence,
        reward_profile=reward_profile,
    )
    observation, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action = action_provider(observation)
        observation, _, terminated, _, _ = env.step(action)
    return _summarize_v3("ppo_v4", env.trajectory, scenario)
