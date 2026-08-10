"""Canonical deterministic environment and public tensor contract.

The environment combines the stable city physics with the exact breakpoint
allocation projection introduced for efficient training. It exposes one
73-value observation contract, one 22-value action contract, and the original
published reward used by the shipped model.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from backend.app.city.outcome import (
    CONSERVATION_TOLERANCE,
    CRITICAL_SERVICE_FLOOR,
    SOLVED_DEFINITION,
    SOLVED_DEFINITION_SHA256,
    SOLVED_RAUC_FLOOR,
    absolute_outcome,
    summarize_trajectory,
)
from backend.app.city.physics import (
    CONSTRAINT_TOLERANCE,
    DELTA,
    DEPENDENCIES,
    DEPOT_CAPACITY,
    ETA,
    FOOD_SPOILAGE_RATE,
    IMMEDIATE_DELIVERY_FRACTION,
    RESERVE_DRAW_FRACTION,
    SERVICES,
    SHOCK_IMPACTS,
    SHOCKS,
    Transfer,
    action_to_proposal,
    apply_depot_damage,
    deterministic_transfer,
    land_capped,
    measure_constraints,
    project_capped_simplex,
    round_vector,
    throughput_factors,
)
from backend.app.city.scenarios import (
    PUBLIC_RISK_EVENT_PULSE,
    PUBLIC_RISK_NOISE,
    PUBLIC_RISK_PERSISTENCE,
    PUBLIC_RISK_TOTAL_CAP,
    Shock,
    generate_disaster_tape,
)
from backend.app.city.planners import reactive_heuristic_action
from backend.app.models import Scenario as ScenarioModel
from backend.app.shared_evidence import canonical_hash

if TYPE_CHECKING:
    from model.policy import Policy

ENGINE_ID = "CityRecoveryEnv-v3"
ENGINE_VERSION = "3.0.0"
RESULT_SCHEMA = "4.0.0"
ACTION_GROUPS = (
    "material_share",
    "material_utilization",
    "crew_share",
    "crew_utilization",
    "stock_release",
    "preparedness_investment",
)
ACTION_ORDER = (
    *(f"material_share_{service}" for service in SERVICES),
    "material_utilization",
    *(f"crew_share_{service}" for service in SERVICES),
    "crew_utilization",
    *(f"stock_release_{service}" for service in SERVICES),
    *(f"preparedness_investment_{service}" for service in SERVICES),
)
ACTION_SIZE = len(ACTION_ORDER)
CREW_PRODUCTIVITY = np.array([1.18, 1.02, 1.2, 1.28, 1.08], dtype=np.float64)
PREPAREDNESS_DECAY = 0.92
PREPAREDNESS_SHOCK_REDUCTION = 0.85
PREPAREDNESS_SHOCK_CONSUMPTION = 0.45
PREPAREDNESS_DAILY_GAIN_CAP = 0.4
OBSERVATION_ORDER = (
    *(f"service_{name}" for name in SERVICES),
    *(f"priority_{name}" for name in SERVICES),
    *(f"support_{name}" for name in SERVICES),
    *(f"shock_impact_{name}" for name in SERVICES),
    *(f"depot_stock_fraction_{name}" for name in SERVICES),
    *(f"pending_arrival_pressure_{name}" for name in SERVICES),
    *(f"throughput_factor_{name}" for name in SERVICES),
    *(f"depot_damage_penalty_{name}" for name in SERVICES),
    *(f"depot_damage_days_fraction_{name}" for name in SERVICES),
    *(f"recovery_target_{name}" for name in SERVICES),
    *(f"prior_day_critical_streak_fraction_{name}" for name in SERVICES),
    *(f"preparedness_level_{name}" for name in SERVICES),
    "available_material_budget_fraction",
    "available_crew_pool_fraction",
    "horizon_remaining_fraction",
    "assessment_tail_active",
    "current_shock_severity",
    "road_capacity",
    "days_since_last_shock_fraction",
    "previous_weighted_resilience",
    *(f"public_next_day_risk_{shock}" for shock in SHOCKS),
)
OBSERVATION_SIZE = len(OBSERVATION_ORDER)
RAW_OBSERVATION_CONTRACT: dict[str, Any] = {
    "source": "raw_environment_observation",
    "input_name": "observation",
    "dtype": "float32",
    "shape": [OBSERVATION_SIZE],
    "normalization": "embedded_in_onnx",
}
ENGINE_SPEC: dict[str, Any] = {
    "id": ENGINE_ID,
    "version": ENGINE_VERSION,
    "observation_size": OBSERVATION_SIZE,
    "action_size": ACTION_SIZE,
    "action_groups": list(ACTION_GROUPS),
    "policy_neutral_transition": True,
    "future_tape_visible": False,
    "public_risk": {
        "description": "causal public indicators that determine, but do not reveal, next-day hazard odds",
        "persistence": PUBLIC_RISK_PERSISTENCE,
        "noise": PUBLIC_RISK_NOISE,
        "event_pulse": PUBLIC_RISK_EVENT_PULSE,
        "total_probability_cap": PUBLIC_RISK_TOTAL_CAP,
    },
    "material": {
        "depot_capacity": DEPOT_CAPACITY.tolist(),
        "same_day_fraction": IMMEDIATE_DELIVERY_FRACTION,
        "next_day_fraction": 1.0 - IMMEDIATE_DELIVERY_FRACTION,
        "physical_dispatch_consumes_stock": True,
    },
    "crew": {
        "productivity_by_service": CREW_PRODUCTIVITY.tolist(),
        "daily_pool_does_not_carry": True,
    },
    "action_layout": {
        "material_shares": 5,
        "material_utilization_gate": 1,
        "crew_shares": 5,
        "crew_utilization_gate": 1,
        "stock_release_gates": 5,
        "preparedness_investment_gates": 5,
    },
    "idle_resources_are_explicit": True,
    "preparedness": {
        "uses_current_material_and_crew": True,
        "daily_decay": PREPAREDNESS_DECAY,
        "shock_reduction_at_full_readiness": PREPAREDNESS_SHOCK_REDUCTION,
        "shock_consumption": PREPAREDNESS_SHOCK_CONSUMPTION,
        "daily_gain_cap": PREPAREDNESS_DAILY_GAIN_CAP,
    },
    "stock_release": "five independent gates in [0,1] controlling combined preparedness and repair dispatch from available stock",
    "solved_definition_sha256": SOLVED_DEFINITION_SHA256,
}
ENGINE_SPEC_SHA256 = canonical_hash(ENGINE_SPEC)


@dataclass(frozen=True)
class Intervention:
    material: np.ndarray
    crew: np.ndarray
    stock_release: np.ndarray
    preparedness_requested: np.ndarray
    preparedness_investment: np.ndarray
    material_used: float
    material_unspent: float
    crew_used: float
    crew_idle: float
    raw_action: np.ndarray
    material_projection: dict[str, Any]
    crew_projection: dict[str, Any]


@dataclass(frozen=True)
class DayContext:
    before: np.ndarray
    shocked: np.ndarray
    support: np.ndarray
    available_budget: float
    available_crew: float
    material_lower: np.ndarray
    material_upper: np.ndarray
    crew_lower: np.ndarray
    crew_upper: np.ndarray
    shock: Shock
    stock_before: np.ndarray
    pending_arrivals: np.ndarray
    pending_landed: np.ndarray
    pending_held: np.ndarray
    stock_after_pending: np.ndarray
    damage_penalty: np.ndarray
    damage_days_remaining: np.ndarray
    depot_factor: np.ndarray
    road_capacity: float
    throughput: np.ndarray
    transfers: tuple[Transfer, ...]
    transfer_net: np.ndarray
    stock_ready: np.ndarray
    public_risk_next: np.ndarray
    recovery_targets: np.ndarray
    critical_streak: np.ndarray
    days_since_last_shock: int
    previous_resilience: float
    preparedness_before: np.ndarray
    preparedness_after_hazard: np.ndarray


def decode_action(
    action: np.ndarray, context: DayContext, *, collect_evidence: bool = True
) -> Intervention:
    supplied = np.asarray(action, dtype=np.float64).reshape(-1)
    if supplied.shape != (ACTION_SIZE,) or not np.all(np.isfinite(supplied)):
        raise ValueError(f"policy action must contain {ACTION_SIZE} finite values")
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
    return Intervention(
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


class CityRecoveryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic city-recovery environment with compact training evidence."""

    metadata = {"render_modes": ["trajectory"], "render_fps": 1}

    def __init__(
        self,
        scenario: ScenarioModel,
        shock_seed: int = 0,
        schedule: Sequence[Shock] | None = None,
        *,
        collect_evidence: bool = True,
    ):
        if not isinstance(collect_evidence, bool):
            raise TypeError("collect_evidence must be a boolean")
        super().__init__()
        self.collect_evidence = collect_evidence
        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            high=np.ones(OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.scenario = scenario
        self.shock_seed = shock_seed
        self._provided_schedule = None if schedule is None else tuple(schedule)
        self.schedule: list[Shock] = []
        self.trajectory: list[dict[str, Any]] = []
        self._q = np.zeros(5, dtype=np.float64)
        self._stocks = np.zeros(5, dtype=np.float64)
        self._pending = np.zeros(5, dtype=np.float64)
        self._damage_peak = np.zeros(5, dtype=np.float64)
        self._damage_duration = np.zeros(5, dtype=np.int64)
        self._damage_remaining = np.zeros(5, dtype=np.int64)
        self._priorities = np.ones(5, dtype=np.float64)
        self._normalized_priorities = np.full(5, 0.2, dtype=np.float64)
        self._targets = np.full(5, 0.55, dtype=np.float64)
        self._critical_streak = np.zeros(5, dtype=np.int64)
        self._preparedness = np.zeros(5, dtype=np.float64)
        self._days_since_last_shock = 0
        self._previous_resilience = 0.0
        self._day_index = 0
        self._context: DayContext | None = None
        self._terminated = False

    def set_scenario(self, scenario: ScenarioModel, shock_seed: int) -> None:
        """Select the scenario and deterministic tape seed for the next reset."""
        self.scenario = scenario
        self.shock_seed = shock_seed
        self._provided_schedule = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
            if not isinstance(scenario, ScenarioModel):
                raise TypeError("reset option scenario must be a Scenario")
            self.scenario = scenario
            self._provided_schedule = None
        if options and "shock_seed" in options:
            self.shock_seed = int(options["shock_seed"])
            self._provided_schedule = None
        elif seed is not None and self._provided_schedule is None:
            self.shock_seed = seed
        if self._provided_schedule is not None:
            if len(self._provided_schedule) != self.scenario.horizon_days:
                raise ValueError("precomputed tape must match scenario horizon")
            self.schedule = list(self._provided_schedule)
        else:
            self.schedule = generate_disaster_tape(self.scenario, self.shock_seed)
        self.trajectory = []
        self._q = np.asarray(self.scenario.initial_services, dtype=np.float64)
        self._stocks = DEPOT_CAPACITY * np.clip(self._q, 0.05, 0.5)
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
        if not self.collect_evidence:
            return (self._observation(), {})
        tape_payload = [asdict(item) for item in self.schedule]
        return (
            self._observation(),
            {
                "shock_schedule_sha256": canonical_hash(tape_payload),
                "shock_seed": self.shock_seed,
                "engine_spec_sha256": ENGINE_SPEC_SHA256,
            },
        )

    def _make_context(self) -> DayContext:
        shock = self.schedule[self._day_index]
        before = self._q.copy()
        stock_before = self._stocks.copy()
        pending_arrivals = self._pending.copy()
        stock_after_pending, pending_landed, pending_held = land_capped(
            stock_before, pending_arrivals
        )
        self._stocks = stock_after_pending
        self._pending = pending_held
        impact = np.asarray(shock.impact, dtype=np.float64)
        preparedness_before = self._preparedness.copy()
        mitigated_impact = (
            shock.severity
            * impact
            * (1.0 - PREPAREDNESS_SHOCK_REDUCTION * preparedness_before)
        )
        shocked = np.clip(before * (1.0 - mitigated_impact), 0.0, 1.0)
        preparedness_after_hazard = preparedness_before * (
            1.0 - PREPAREDNESS_SHOCK_CONSUMPTION * shock.severity * impact
        )
        support = 0.55 + 0.45 * (DEPENDENCIES @ shocked)
        available_budget = self.scenario.daily_budget * (
            1.0 - shock.severity * shock.budget_factor
        )
        available_crew = self.scenario.daily_crew_pool * (
            1.0 - 0.45 * shock.severity * shock.budget_factor
        )
        material_lower = np.where(
            shocked < CRITICAL_SERVICE_FLOOR, 0.04 * available_budget, 0.0
        )
        material_upper = np.full(5, 0.5 * available_budget, dtype=np.float64)
        crew_lower = np.where(
            shocked < CRITICAL_SERVICE_FLOOR, 0.04 * available_crew, 0.0
        )
        crew_upper = np.full(5, 0.5 * available_crew, dtype=np.float64)
        (
            self._damage_peak,
            self._damage_duration,
            self._damage_remaining,
            damage_penalty,
        ) = apply_depot_damage(
            shock, self._damage_peak, self._damage_duration, self._damage_remaining
        )
        depot_factor, road_capacity, throughput = throughput_factors(
            shocked, damage_penalty
        )
        stock_ready, transfer_net, transfers = deterministic_transfer(
            stock_after_pending, throughput
        )
        self._stocks = stock_ready
        self._days_since_last_shock = (
            0
            if shock.type is not None
            else min(self.scenario.horizon_days, self._days_since_last_shock + 1)
        )
        return DayContext(
            before=before,
            shocked=shocked,
            support=support,
            available_budget=float(available_budget),
            available_crew=float(available_crew),
            material_lower=material_lower,
            material_upper=material_upper,
            crew_lower=crew_lower,
            crew_upper=crew_upper,
            shock=shock,
            stock_before=stock_before,
            pending_arrivals=pending_arrivals,
            pending_landed=pending_landed,
            pending_held=pending_held,
            stock_after_pending=stock_after_pending,
            damage_penalty=damage_penalty,
            damage_days_remaining=self._damage_remaining.copy(),
            depot_factor=depot_factor,
            road_capacity=road_capacity,
            throughput=throughput,
            transfers=transfers,
            transfer_net=transfer_net,
            stock_ready=stock_ready,
            public_risk_next=np.asarray(shock.public_risk_next, dtype=np.float64),
            recovery_targets=self._targets.copy(),
            critical_streak=self._critical_streak.copy(),
            days_since_last_shock=self._days_since_last_shock,
            previous_resilience=self._previous_resilience,
            preparedness_before=preparedness_before,
            preparedness_after_hazard=preparedness_after_hazard,
        )

    def _observation(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("environment must be reset before observation")
        context = self._context
        remaining = (self.scenario.horizon_days - self._day_index) / float(
            self.scenario.horizon_days
        )
        values = np.concatenate(
            (
                context.shocked,
                self._priorities / 2.0,
                context.support,
                np.asarray(context.shock.impact, dtype=np.float64),
                context.stock_ready / DEPOT_CAPACITY,
                context.pending_arrivals
                / np.maximum(context.pending_arrivals + DEPOT_CAPACITY, 1e-09),
                context.throughput,
                context.damage_penalty,
                np.clip(context.damage_days_remaining / 10.0, 0.0, 1.0),
                context.recovery_targets,
                np.clip(context.critical_streak / self.scenario.horizon_days, 0.0, 1.0),
                context.preparedness_after_hazard,
                np.array(
                    [
                        context.available_budget / 500.0,
                        context.available_crew / 300.0,
                        remaining,
                        float(context.shock.assessment_tail),
                        context.shock.severity,
                        context.road_capacity,
                        context.days_since_last_shock / self.scenario.horizon_days,
                        context.previous_resilience,
                    ],
                    dtype=np.float64,
                ),
                context.public_risk_next,
            )
        )
        if values.shape != (OBSERVATION_SIZE,):
            raise RuntimeError("observation assembly drifted")
        return np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float32)

    def current_context(self) -> DayContext:
        if self._context is None or self._terminated:
            raise RuntimeError("environment has no active day")
        return self._context

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        intervention = decode_action(
            action, self.current_context(), collect_evidence=self.collect_evidence
        )
        return self._advance(intervention, planner_evidence=None)

    def step_with_evidence(
        self, action: np.ndarray, planner_evidence: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        intervention = decode_action(
            action, self.current_context(), collect_evidence=self.collect_evidence
        )
        return self._advance(intervention, planner_evidence=planner_evidence)

    def _advance(
        self, intervention: Intervention, *, planner_evidence: dict[str, Any] | None
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
            crew, intervention.crew_used, context.crew_lower, context.crew_upper
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
            (value for key, value in material_measurements.items() if key != "total")
        )
        crew_measurements["total"] = sum(
            (value for key, value in crew_measurements.items() if key != "total")
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
        stock_with_delivery, same_day_landed, same_day_held = land_capped(
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
            / np.maximum(CREW_PRODUCTIVITY * context.support, 1e-09),
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
            np.mean(pending_next / np.maximum(pending_next + DEPOT_CAPACITY, 1e-09))
        )
        material_preparedness_ratio = preparedness_effective_work / max(
            0.2 * context.available_budget, 1e-09
        )
        crew_preparedness_ratio = preparedness_crew_utilized / max(
            0.2 * context.available_crew, 1e-09
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
        reward = (
            1.5 * resilience
            + 0.75 * (resilience - shocked_resilience)
            - 0.35 * critical_count
            - 1.0 * mean_target_gap
            - 1.2 * critical_shortfall
            - assessment_target_penalty
            - 0.25 * backlog_pressure
            + 10.0 * preparedness_alignment
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
            float(np.max(np.abs(conservation_residual))) > CONSERVATION_TOLERANCE
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
        services_end_record = round_vector(end)
        outcome_logistics = {
            "pending_next_day": round_vector(pending_next),
            "conservation_residual": round_vector(conservation_residual),
        }
        resilience_record = round(resilience, 8)
        reward_record = round(float(reward), 8)
        if self.collect_evidence:
            logistics = {
                "depot_capacity": round_vector(DEPOT_CAPACITY),
                "depot_stock_before": round_vector(context.stock_before),
                "pending_arrivals": round_vector(context.pending_arrivals),
                "pending_arrivals_landed": round_vector(context.pending_landed),
                "pending_arrivals_held": round_vector(context.pending_held),
                "depot_stock_after_pending": round_vector(context.stock_after_pending),
                "depot_damage_penalty": round_vector(context.damage_penalty),
                "depot_damage_days_remaining": [
                    int(value) for value in context.damage_days_remaining
                ],
                "depot_damage_factor": round_vector(context.depot_factor),
                "road_capacity": round(context.road_capacity, 8),
                "throughput_factor": round_vector(context.throughput),
                "mutual_aid_transfers": [asdict(event) for event in context.transfers],
                "mutual_aid_net": round_vector(context.transfer_net),
                "depot_stock_ready": round_vector(context.stock_ready),
                "preparedness_material_requested": round_vector(
                    preparedness_material_requested
                ),
                "preparedness_material_consumed": round_vector(preparedness_material),
                "preparedness_effective_work": round_vector(
                    preparedness_effective_work
                ),
                "depot_stock_after_preparedness": round_vector(
                    stock_after_preparedness
                ),
                "repair_material_committed": round_vector(repair_material),
                "preparedness_crew_assigned": round_vector(preparedness_crew),
                "preparedness_crew_utilized": round_vector(preparedness_crew_utilized),
                "preparedness_crew_capacity_effective": round_vector(
                    preparedness_crew_capacity_effective
                ),
                "preparedness_crew_capacity_physical": round_vector(
                    preparedness_crew_capacity_physical
                ),
                "repair_crew_assigned": round_vector(repair_crew),
                "same_day_delivery_scheduled": round_vector(same_day_scheduled),
                "same_day_delivery_landed": round_vector(same_day_landed),
                "same_day_delivery_held": round_vector(same_day_held),
                "delayed_delivery_scheduled": round_vector(delayed_scheduled),
                "repair_reserve": round_vector(repair_reserve),
                "repair_request": round_vector(work_order_limit),
                "total_stock_release_budget": round_vector(total_release_budget),
                "stock_release_remaining_after_preparedness": round_vector(
                    release_remaining
                ),
                "stock_release_limit": round_vector(release_limit),
                "crew_capacity_effective": round_vector(crew_capacity_effective),
                "crew_capacity_physical": round_vector(crew_capacity_physical),
                "repair_dispatch": round_vector(physical_dispatch),
                "repair_supply": round_vector(effective_repair),
                "spoilage": round_vector(spoilage),
                "depot_stock_end": round_vector(stock_end),
                "pending_next_day": outcome_logistics["pending_next_day"],
                "capacity_overflow": round_vector(capacity_overflow),
                "conservation_residual": outcome_logistics["conservation_residual"],
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
                "services_before": round_vector(context.before),
                "services_after_shock": round_vector(context.shocked),
                "raw_action": round_vector(intervention.raw_action),
                "allocation": round_vector(material),
                "material_allocation": round_vector(material),
                "crew_allocation": round_vector(crew),
                "stock_release": round_vector(intervention.stock_release),
                "preparedness_requested": round_vector(
                    intervention.preparedness_requested
                ),
                "preparedness_investment": round_vector(
                    intervention.preparedness_investment
                ),
                "preparedness_before": round_vector(context.preparedness_before),
                "preparedness_after_hazard": round_vector(
                    context.preparedness_after_hazard
                ),
                "preparedness_gain_requested": round_vector(
                    preparedness_gain_requested
                ),
                "preparedness_gain": round_vector(preparedness_gain),
                "preparedness_end": round_vector(preparedness_end),
                "preparedness_alignment_reward": round(preparedness_alignment, 8),
                "backlog_pressure": round(backlog_pressure, 8),
                "lower_bounds": round_vector(context.material_lower),
                "upper_bounds": round_vector(context.material_upper),
                "crew_lower_bounds": round_vector(context.crew_lower),
                "crew_upper_bounds": round_vector(context.crew_upper),
                "projection": intervention.material_projection,
                "crew_projection": intervention.crew_projection,
                "planner_evidence": planner_evidence,
                "support": round_vector(context.support),
                "throughput": round_vector(context.throughput),
                "public_next_day_risk": round_vector(context.public_risk_next),
                "gain": round_vector(gain),
                "strain": round_vector(strain),
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
            outcome = absolute_outcome(
                self.trajectory,
                self.scenario.recovery_targets,
                self.scenario.assessment_tail_days,
            )
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
            terminal_bonus = (
                12.0
                if outcome["solved"]
                else -(
                    8.0 * tail_shortfall
                    + 0.2 * critical_excess
                    + 5.0 * max(0.0, SOLVED_RAUC_FLOOR - outcome["resilience_auc"])
                    + 1.5 * len(outcome["reason_codes"])
                )
            )
            reward += terminal_bonus
            record["terminal_bonus"] = round(float(terminal_bonus), 8)
            record["reward"] = round(float(reward), 8)
            record["absolute_outcome"] = outcome
            self._context = None
            observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        else:
            self._context = self._make_context()
            observation = self._observation()
        return (observation, float(reward), self._terminated, False, {"day": record})

    def render(self) -> list[dict[str, Any]]:
        return list(self.trajectory)


class CyclingScenarioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Cycle deterministic training cases without constructing full evidence."""

    def __init__(
        self,
        scenarios: list[tuple[ScenarioModel, int]],
        *,
        collect_evidence: bool = True,
    ):
        if not scenarios:
            raise ValueError("at least one training scenario is required")
        self.scenarios = [
            (scenario, shock_seed, tuple(generate_disaster_tape(scenario, shock_seed)))
            for scenario, shock_seed in scenarios
        ]
        self.index = 0
        first, first_seed, first_schedule = self.scenarios[0]
        self.inner = CityRecoveryEnv(
            first,
            first_seed,
            first_schedule,
            collect_evidence=collect_evidence,
        )
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
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


def rollout_policy(
    scenario: ScenarioModel,
    seed: int,
    action_provider: Callable[[np.ndarray], np.ndarray],
    schedule: Sequence[Shock] | None = None,
) -> dict[str, Any]:
    """Run one policy against a supplied or deterministically generated tape."""

    shared_schedule = (
        generate_disaster_tape(scenario, seed) if schedule is None else list(schedule)
    )
    environment = CityRecoveryEnv(scenario, seed, shared_schedule)
    observation, _ = environment.reset(seed=seed)
    terminated = False
    while not terminated:
        action = action_provider(observation)
        observation, _, terminated, _, _ = environment.step(action)
    return summarize_trajectory("onnx_policy", environment.trajectory, scenario)


def rollout_baseline(
    scenario: ScenarioModel,
    seed: int,
    schedule: Sequence[Shock] | None = None,
) -> dict[str, Any]:
    """Run the causal public-state baseline on the same environment contract."""

    shared_schedule = (
        generate_disaster_tape(scenario, seed) if schedule is None else list(schedule)
    )
    environment = CityRecoveryEnv(scenario, seed, shared_schedule)
    observation, _ = environment.reset(seed=seed)
    terminated = False
    while not terminated:
        action, evidence = reactive_heuristic_action(observation)
        observation, _, terminated, _, _ = environment.step_with_evidence(
            action, evidence
        )
    return summarize_trajectory(
        "reactive_public_state_heuristic", environment.trajectory, scenario
    )


def policy_identity(policy: "Policy") -> dict[str, Any]:
    """Describe the exact artifact and inference contract used for a rollout."""

    return {
        "id": policy.path.stem,
        "path_stem": policy.path.stem,
        "artifact_type": "onnx_policy",
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "sha256": policy.sha256,
        "observation_contract": dict(RAW_OBSERVATION_CONTRACT),
    }


def compare(
    scenario: ScenarioModel,
    seed: int,
    policy: "Policy",
) -> dict[str, Any]:
    """Compare one ONNX policy and the baseline on one immutable shared tape."""

    schedule = generate_disaster_tape(scenario, seed)
    schedule_payload = [asdict(shock) for shock in schedule]
    baseline = rollout_baseline(scenario, seed, schedule)
    candidate = rollout_policy(scenario, seed, policy.predict, schedule)
    candidate_shocks = [day["shock"] for day in candidate["trajectory"]]
    baseline_shocks = [day["shock"] for day in baseline["trajectory"]]
    if candidate_shocks != schedule_payload or baseline_shocks != schedule_payload:
        raise RuntimeError("planners did not consume the same precomputed tape")

    candidate_solved = bool(candidate["absolute_outcome"]["solved"])
    baseline_solved = bool(baseline["absolute_outcome"]["solved"])
    if candidate_solved and baseline_solved:
        outcome_pair = "both_solved"
    elif candidate_solved:
        outcome_pair = "ppo_only"
    elif baseline_solved:
        outcome_pair = "heuristic_only"
    else:
        outcome_pair = "neither"

    return {
        "schema_version": RESULT_SCHEMA,
        "engine_version": "city-recovery-env-v3",
        "environment": {
            "id": ENGINE_ID,
            "version": ENGINE_VERSION,
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "spec_sha256": ENGINE_SPEC_SHA256,
        },
        "engine_spec": ENGINE_SPEC,
        "engine_spec_sha256": ENGINE_SPEC_SHA256,
        "outcome_definition": SOLVED_DEFINITION,
        "outcome_definition_sha256": SOLVED_DEFINITION_SHA256,
        "seed": seed,
        "generator": "numpy.PCG64",
        "scenario": scenario.model_dump(mode="json"),
        "services": list(SERVICES),
        "observation_order": list(OBSERVATION_ORDER),
        "action_order": list(ACTION_ORDER),
        "shock_schedule": schedule_payload,
        "shock_schedule_sha256": canonical_hash(schedule_payload),
        "policy": policy_identity(policy),
        "baseline_spec": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "uses_same_observation_contract": True,
            "uses_same_action_contract": True,
            "uses_public_risk_signal": True,
            "future_tape_visible": False,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "primary_metric": "independent_absolute_disaster_solved",
            "candidate_solved": candidate_solved,
            "baseline_solved": baseline_solved,
            "absolute_outcome_pair": outcome_pair,
            "secondary_rauc_candidate_minus_baseline": round(
                candidate["rauc"] - baseline["rauc"], 8
            ),
        },
    }


__all__ = (
    "ACTION_GROUPS",
    "ACTION_ORDER",
    "ACTION_SIZE",
    "CityRecoveryEnv",
    "CyclingScenarioEnv",
    "DayContext",
    "ENGINE_ID",
    "ENGINE_SPEC",
    "ENGINE_SPEC_SHA256",
    "ENGINE_VERSION",
    "Intervention",
    "OBSERVATION_ORDER",
    "OBSERVATION_SIZE",
    "RAW_OBSERVATION_CONTRACT",
    "RESULT_SCHEMA",
    "compare",
    "decode_action",
    "policy_identity",
    "rollout_baseline",
    "rollout_policy",
)
