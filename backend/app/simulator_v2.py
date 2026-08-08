from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import gymnasium as gym
import numpy as np
import ortools
from gymnasium import spaces
from ortools.linear_solver import pywraplp

from backend.app.models import Scenario
from backend.app.simulator import (
    CONSTRAINT_TOLERANCE,
    DELTA,
    DEPENDENCIES,
    ETA,
    SERVICES,
    SHOCK_BUDGET_FACTORS,
    SHOCK_IMPACTS,
    SHOCK_TYPE_PROBABILITIES,
    SHOCKS,
    _round_vector,
    action_to_proposal,
    canonical_hash,
    measure_constraints,
    project_capped_simplex,
)
from backend.app.simulator import (
    OBSERVATION_ORDER as OBSERVATION_ORDER_V1,
)

# The v2 coefficients are preregistered, centralized, and hashed into every compare
# result.  Keeping them here also prevents the training and runtime paths from
# silently drifting apart.
DEPOT_CAPACITY = np.full(5, 400.0, dtype=np.float64)
IMMEDIATE_DELIVERY_FRACTION = 0.65
DELAYED_DELIVERY_FRACTION = 0.35
FOOD_SPOILAGE_RATE = 0.006
RESERVE_DRAW_FRACTION = 0.04
ROAD_CAPACITY_FLOOR = 0.40
DEPOT_THROUGHPUT_FLOOR = 0.30
DEPOT_DAMAGE_SCALE = 1.50
DEPOT_DAMAGE_PENALTY_CAP = 0.72
TRANSFER_STARVED_FRACTION = 0.15
TRANSFER_SURPLUS_FRACTION = 0.42
TRANSFER_DONOR_RESERVE_FRACTION = 0.35
TRANSFER_RECEIVER_TARGET_FRACTION = 0.30
TRANSFER_DAILY_CAP_FRACTION = 0.06
TRANSFER_MIN_THROUGHPUT = 0.55
AFTERSHOCK_DAY_ONE_SCALE = 0.28
AFTERSHOCK_DAY_TWO_SCALE = 0.12

ENGINE_V2_SPEC: dict[str, Any] = {
    "id": "city-recovery-env-v2",
    "observation_size": 33,
    "action_size": 5,
    "depot_capacity": [400.0] * 5,
    "initial_stock_fraction": "clip(initial_service, 0.05, 0.50)",
    "delivery": {
        "same_day_fraction": IMMEDIATE_DELIVERY_FRACTION,
        "next_day_fraction": DELAYED_DELIVERY_FRACTION,
        "capacity_overflow": "held in the deterministic pending queue",
    },
    "repair": {
        "reserve_draw_fraction": RESERVE_DRAW_FRACTION,
        "request": "allocation + capacity * reserve_fraction * service_deficit",
        "dispatch": "min(stock_ready + same_day_landed, request)",
        "effective_supply": "dispatch * throughput_factor",
    },
    "throughput": {
        "road": "0.40 + 0.60 * shocked_transport_service",
        "transport": "transport_depot_factor",
        "other_services": "service_depot_factor * road_capacity",
        "depot_floor": DEPOT_THROUGHPUT_FLOOR,
    },
    "depot_damage": {
        "penalty": "clip(1.50 * severity * typed_impact, 0, 0.72)",
        "duration_days": "ceil(2 + 8 * severity * typed_impact)",
        "recovery": "linear; stronger penalty and longer remaining duration win",
    },
    "mutual_aid": {
        "max_events_per_day": 1,
        "receiver_below_fraction": TRANSFER_STARVED_FRACTION,
        "donor_above_fraction": TRANSFER_SURPLUS_FRACTION,
        "donor_reserve_fraction": TRANSFER_DONOR_RESERVE_FRACTION,
        "receiver_target_fraction": TRANSFER_RECEIVER_TARGET_FRACTION,
        "daily_cap_fraction": TRANSFER_DAILY_CAP_FRACTION,
        "minimum_effective_throughput": TRANSFER_MIN_THROUGHPUT,
    },
    "food_spoilage_fraction_per_day": FOOD_SPOILAGE_RATE,
    "aftershock_cluster": {
        "day_one_severity_scale": AFTERSHOCK_DAY_ONE_SCALE,
        "day_two_severity_scale": AFTERSHOCK_DAY_TWO_SCALE,
        "probability": (
            "clip(base + 0.28 * prior_quake_severity + 0.12 * two_day_quake_severity, 0, 1)"
        ),
        "randomness": "the precomputed PCG64 ambient occurrence draw",
    },
}
ENGINE_V2_SPEC_SHA256 = canonical_hash(ENGINE_V2_SPEC)

OBSERVATION_ORDER_V2 = (
    *OBSERVATION_ORDER_V1,
    *(f"depot_stock_fraction_{name}" for name in SERVICES),
    *(f"throughput_factor_{name}" for name in SERVICES),
)
OBSERVATION_SIZE_V2 = len(OBSERVATION_ORDER_V2)
ACTION_ORDER_V2 = SERVICES


@dataclass(frozen=True)
class ShockV2:
    day: int
    type: str | None
    severity: float
    impact: list[float]
    budget_factor: float
    forced: bool
    clustered: bool
    cluster_parent_days: tuple[int, ...]
    ambient_occurrence_probability: float
    ambient_occurrence_draw: float
    cluster_hazard: float


@dataclass(frozen=True)
class TransferV2:
    from_service: str
    to_service: str
    units: float
    donor_stock_fraction_before: float
    receiver_stock_fraction_before: float


@dataclass(frozen=True)
class DayContextV2:
    before: np.ndarray
    shocked: np.ndarray
    support: np.ndarray
    available_budget: float
    lower: np.ndarray
    upper: np.ndarray
    shock: ShockV2
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
    transfers: tuple[TransferV2, ...]
    transfer_net: np.ndarray
    stock_ready: np.ndarray


def generate_shock_schedule_v2(scenario: Scenario, seed: int) -> list[ShockV2]:
    """Precompute one deterministic tape, including two-day quake clustering.

    The function consumes the same three ambient draws per day as v1.  A quake in
    the final authored tape (including a forced one) raises the occurrence
    threshold on the next two days.  Only an occurrence in the added probability
    interval is typed as a clustered aftershock; the original ambient interval
    retains its independently drawn type.  Forced overrides are resolved before
    the loop in the established singular-then-list, last-wins order.
    """

    forced_by_day: dict[int, Any] = {}
    if scenario.forced_shock is not None:
        forced_by_day[scenario.forced_shock.day] = scenario.forced_shock
    for forced in scenario.forced_shocks:
        forced_by_day[forced.day] = forced

    rng = np.random.Generator(np.random.PCG64(seed))
    schedule: list[ShockV2] = []
    for day in range(1, scenario.horizon_days + 1):
        occurrence_draw = float(rng.random())
        ambient_index = int(rng.choice(len(SHOCKS), p=SHOCK_TYPE_PROBABILITIES))
        severity_draw = scenario.severity_min + (
            scenario.severity_max - scenario.severity_min
        ) * float(rng.beta(2.0, 5.0))

        prior = schedule[-1] if schedule else None
        two_back = schedule[-2] if len(schedule) >= 2 else None
        prior_severity = prior.severity if prior is not None and prior.type == "aftershock" else 0.0
        two_back_severity = (
            two_back.severity if two_back is not None and two_back.type == "aftershock" else 0.0
        )
        cluster_hazard = (
            AFTERSHOCK_DAY_ONE_SCALE * prior_severity + AFTERSHOCK_DAY_TWO_SCALE * two_back_severity
        )
        occurrence_probability = float(
            np.clip(scenario.shock_probability + cluster_hazard, 0.0, 1.0)
        )
        occurs = occurrence_draw < occurrence_probability
        clustered = bool(
            occurs and cluster_hazard > 0.0 and occurrence_draw >= scenario.shock_probability
        )
        shock_type = "aftershock" if clustered else SHOCKS[ambient_index] if occurs else None
        forced = forced_by_day.get(day)
        if forced is not None:
            shock_type = forced.type
            severity = float(forced.severity)
            forced_flag = True
            clustered = False
        elif shock_type is not None:
            severity = float(round(severity_draw, 8))
            forced_flag = False
        else:
            severity = 0.0
            forced_flag = False

        if shock_type is None:
            impact = [0.0] * 5
            budget_factor = 0.0
        else:
            shock_index = SHOCKS.index(shock_type)
            impact = _round_vector(SHOCK_IMPACTS[shock_index])
            budget_factor = float(SHOCK_BUDGET_FACTORS[shock_index])
        parents = tuple(
            item.day for item in (two_back, prior) if item is not None and item.type == "aftershock"
        )
        schedule.append(
            ShockV2(
                day=day,
                type=shock_type,
                severity=severity,
                impact=impact,
                budget_factor=budget_factor,
                forced=forced_flag,
                clustered=clustered,
                cluster_parent_days=parents,
                ambient_occurrence_probability=float(round(occurrence_probability, 8)),
                ambient_occurrence_draw=float(round(occurrence_draw, 8)),
                cluster_hazard=float(round(cluster_hazard, 8)),
            )
        )
    return schedule


def apply_depot_damage_v2(
    shock: ShockV2,
    peak_penalty: np.ndarray,
    duration_days: np.ndarray,
    remaining_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply typed depot damage while preserving the stronger/longer condition."""

    peaks = np.asarray(peak_penalty, dtype=np.float64).copy()
    durations = np.asarray(duration_days, dtype=np.int64).copy()
    remaining = np.asarray(remaining_days, dtype=np.int64).copy()
    current = np.divide(
        peaks * remaining,
        durations,
        out=np.zeros(5, dtype=np.float64),
        where=durations > 0,
    )
    if shock.type is not None and shock.severity > 0.0:
        impact = np.asarray(shock.impact, dtype=np.float64)
        candidate_penalty = np.clip(
            DEPOT_DAMAGE_SCALE * shock.severity * impact,
            0.0,
            DEPOT_DAMAGE_PENALTY_CAP,
        )
        candidate_duration = np.array(
            [ceil(2.0 + 8.0 * shock.severity * value) for value in impact],
            dtype=np.int64,
        )
        for index in range(5):
            if candidate_penalty[index] <= 0.0:
                continue
            combined_penalty = max(float(current[index]), float(candidate_penalty[index]))
            combined_remaining = max(int(remaining[index]), int(candidate_duration[index]))
            peaks[index] = combined_penalty
            durations[index] = combined_remaining
            remaining[index] = combined_remaining
            current[index] = combined_penalty
    return peaks, durations, remaining, current


def throughput_factors_v2(
    shocked_services: np.ndarray, damage_penalty: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    depot_factor = np.clip(
        1.0 - np.asarray(damage_penalty, dtype=np.float64),
        DEPOT_THROUGHPUT_FLOOR,
        1.0,
    )
    shocked = np.asarray(shocked_services, dtype=np.float64)
    road_capacity = float(
        np.clip(ROAD_CAPACITY_FLOOR + (1.0 - ROAD_CAPACITY_FLOOR) * shocked[0], 0.0, 1.0)
    )
    throughput = depot_factor.copy()
    throughput[1:] *= road_capacity
    return depot_factor, road_capacity, np.clip(throughput, 0.0, 1.0)


def deterministic_transfer_v2(
    stock: np.ndarray,
    throughput: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[TransferV2, ...]]:
    """Move at most one bounded generic-supply load between eligible depots."""

    adjusted = np.asarray(stock, dtype=np.float64).copy()
    factors = np.asarray(throughput, dtype=np.float64)
    fractions = adjusted / DEPOT_CAPACITY
    receivers = [
        index
        for index in range(5)
        if fractions[index] < TRANSFER_STARVED_FRACTION
        and factors[index] >= TRANSFER_MIN_THROUGHPUT
    ]
    if not receivers:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    receiver = min(receivers, key=lambda index: (fractions[index], index))
    donors = [
        index
        for index in range(5)
        if index != receiver
        and fractions[index] > TRANSFER_SURPLUS_FRACTION
        and factors[index] >= TRANSFER_MIN_THROUGHPUT
    ]
    if not donors:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    donor = max(donors, key=lambda index: (fractions[index], -index))
    amount = min(
        TRANSFER_DAILY_CAP_FRACTION * DEPOT_CAPACITY[receiver],
        adjusted[donor] - TRANSFER_DONOR_RESERVE_FRACTION * DEPOT_CAPACITY[donor],
        TRANSFER_RECEIVER_TARGET_FRACTION * DEPOT_CAPACITY[receiver] - adjusted[receiver],
    )
    amount = float(max(0.0, amount))
    if amount <= CONSTRAINT_TOLERANCE:
        return adjusted, np.zeros(5, dtype=np.float64), ()
    net = np.zeros(5, dtype=np.float64)
    adjusted[donor] -= amount
    adjusted[receiver] += amount
    net[donor] = -amount
    net[receiver] = amount
    event = TransferV2(
        from_service=SERVICES[donor],
        to_service=SERVICES[receiver],
        units=float(round(amount, 8)),
        donor_stock_fraction_before=float(round(fractions[donor], 8)),
        receiver_stock_fraction_before=float(round(fractions[receiver], 8)),
    )
    return adjusted, net, (event,)


def _land_capped(
    stock: np.ndarray, arrivals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room = np.maximum(0.0, DEPOT_CAPACITY - stock)
    landed = np.minimum(np.asarray(arrivals, dtype=np.float64), room)
    held = np.asarray(arrivals, dtype=np.float64) - landed
    return stock + landed, landed, held


class CityRecoveryEnvV2(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic depot-buffer and delivery-latency recovery environment."""

    metadata = {"render_modes": ["trajectory"], "render_fps": 1}

    def __init__(
        self,
        scenario: Scenario,
        shock_seed: int = 0,
        schedule: Sequence[ShockV2] | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_SIZE_V2, dtype=np.float32),
            high=np.ones(OBSERVATION_SIZE_V2, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.scenario = scenario
        self.shock_seed = shock_seed
        self._provided_schedule = None if schedule is None else tuple(schedule)
        self.schedule: list[ShockV2] = []
        self.trajectory: list[dict[str, Any]] = []
        self._q = np.zeros(5, dtype=np.float64)
        self._stocks = np.zeros(5, dtype=np.float64)
        self._pending = np.zeros(5, dtype=np.float64)
        self._damage_peak = np.zeros(5, dtype=np.float64)
        self._damage_duration = np.zeros(5, dtype=np.int64)
        self._damage_remaining = np.zeros(5, dtype=np.int64)
        self._priorities = np.ones(5, dtype=np.float64)
        self._normalized_priorities = np.full(5, 0.2, dtype=np.float64)
        self._day_index = 0
        self._context: DayContextV2 | None = None
        self._terminated = False

    def set_scenario(self, scenario: Scenario, shock_seed: int) -> None:
        self.scenario = scenario
        self.shock_seed = shock_seed
        self._provided_schedule = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
            if not isinstance(scenario, Scenario):
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
                raise ValueError("precomputed v2 shock schedule must match scenario horizon")
            self.schedule = list(self._provided_schedule)
        else:
            self.schedule = generate_shock_schedule_v2(self.scenario, self.shock_seed)
        self.trajectory = []
        self._q = np.asarray(self.scenario.initial_services, dtype=np.float64)
        initial_fraction = np.clip(self._q, 0.05, 0.50)
        self._stocks = DEPOT_CAPACITY * initial_fraction
        self._pending = np.zeros(5, dtype=np.float64)
        self._damage_peak = np.zeros(5, dtype=np.float64)
        self._damage_duration = np.zeros(5, dtype=np.int64)
        self._damage_remaining = np.zeros(5, dtype=np.int64)
        self._priorities = np.asarray(self.scenario.priorities, dtype=np.float64)
        self._normalized_priorities = self._priorities / float(self._priorities.sum())
        self._day_index = 0
        self._terminated = False
        self._context = self._make_context()
        schedule_payload = [asdict(item) for item in self.schedule]
        return self._observation(), {
            "shock_schedule_sha256": canonical_hash(schedule_payload),
            "shock_seed": self.shock_seed,
            "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        }

    def _make_context(self) -> DayContextV2:
        shock = self.schedule[self._day_index]
        before = self._q.copy()
        stock_before = self._stocks.copy()
        pending_arrivals = self._pending.copy()
        stock_after_pending, pending_landed, pending_held = _land_capped(
            stock_before, pending_arrivals
        )
        self._stocks = stock_after_pending
        self._pending = pending_held

        impact = np.asarray(shock.impact, dtype=np.float64)
        shocked = np.clip(before * (1.0 - shock.severity * impact), 0.0, 1.0)
        support = 0.55 + 0.45 * (DEPENDENCIES @ shocked)
        available_budget = self.scenario.daily_budget * (1.0 - shock.severity * shock.budget_factor)
        lower = np.where(shocked < 0.30, 0.04 * available_budget, 0.0)
        upper = np.full(5, 0.50 * available_budget, dtype=np.float64)

        (
            self._damage_peak,
            self._damage_duration,
            self._damage_remaining,
            damage_penalty,
        ) = apply_depot_damage_v2(
            shock,
            self._damage_peak,
            self._damage_duration,
            self._damage_remaining,
        )
        depot_factor, road_capacity, throughput = throughput_factors_v2(shocked, damage_penalty)
        stock_ready, transfer_net, transfers = deterministic_transfer_v2(
            stock_after_pending, throughput
        )
        self._stocks = stock_ready
        return DayContextV2(
            before=before,
            shocked=shocked,
            support=support,
            available_budget=float(available_budget),
            lower=lower,
            upper=upper,
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
        )

    def _observation(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("environment must be reset before observation")
        context = self._context
        remaining = (self.scenario.horizon_days - self._day_index) / float(
            self.scenario.horizon_days
        )
        prefix = np.concatenate(
            (
                context.shocked,
                self._priorities / 2.0,
                context.support,
                np.asarray(context.shock.impact, dtype=np.float64),
                np.array(
                    [
                        context.available_budget / 500.0,
                        remaining,
                        context.shock.severity,
                    ],
                    dtype=np.float64,
                ),
            )
        )
        values = np.concatenate((prefix, context.stock_ready / DEPOT_CAPACITY, context.throughput))
        return np.asarray(values, dtype=np.float32)

    def current_context(self) -> DayContextV2:
        if self._context is None or self._terminated:
            raise RuntimeError("environment has no active day")
        return self._context

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        context = self.current_context()
        supplied = np.asarray(action, dtype=np.float64).reshape(-1)
        proposal = action_to_proposal(supplied, context.available_budget)
        raw_action = np.clip(supplied, -1.0, 1.0)
        return self._advance(proposal, raw_action=raw_action, planner_evidence=None)

    def step_proposal(
        self, proposal: np.ndarray, planner_evidence: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._advance(
            np.asarray(proposal, dtype=np.float64),
            raw_action=None,
            planner_evidence=planner_evidence,
        )

    def _advance(
        self,
        proposal: np.ndarray,
        *,
        raw_action: np.ndarray | None,
        planner_evidence: dict[str, Any] | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        context = self.current_context()
        allocation, projection = project_capped_simplex(
            proposal, context.available_budget, context.lower, context.upper
        )
        measurements = measure_constraints(
            allocation, context.available_budget, context.lower, context.upper
        )
        projection["constraint_violations"] = measurements["total"]
        projection["violation_breakdown"] = measurements

        same_day_scheduled = allocation * IMMEDIATE_DELIVERY_FRACTION
        delayed_scheduled = allocation - same_day_scheduled
        stock_with_delivery, same_day_landed, same_day_held = _land_capped(
            context.stock_ready, same_day_scheduled
        )
        repair_reserve = RESERVE_DRAW_FRACTION * DEPOT_CAPACITY * (1.0 - context.shocked)
        repair_request = allocation + repair_reserve
        repair_dispatch = np.minimum(stock_with_delivery, repair_request)
        repair_supply = repair_dispatch * context.throughput
        stock_before_spoilage = np.maximum(0.0, stock_with_delivery - repair_supply)
        spoilage = np.zeros(5, dtype=np.float64)
        spoilage[2] = FOOD_SPOILAGE_RATE * stock_before_spoilage[2]
        stock_end = stock_before_spoilage - spoilage
        pending_next = self._pending + same_day_held + delayed_scheduled
        capacity_overflow = context.pending_held + same_day_held

        gain = (
            ETA
            * np.sqrt(repair_dispatch / 200.0)
            * context.support
            * context.throughput
            * (1.0 - context.shocked)
        )
        strain = (
            DELTA
            * np.maximum(0.0, 0.35 - context.shocked)
            * (1.0 - np.clip(repair_supply / context.available_budget, 0.0, 1.0))
        )
        end = np.clip(context.shocked + gain - strain, 0.0, 1.0)
        resilience = float(self._normalized_priorities @ end)
        shocked_resilience = float(self._normalized_priorities @ context.shocked)
        reward = resilience + 0.35 * (resilience - shocked_resilience)
        reward -= 0.0001 * projection["distance"] / context.available_budget

        conservation_residual = (
            context.stock_before
            + context.pending_arrivals
            + allocation
            + context.transfer_net
            - repair_supply
            - spoilage
            - stock_end
            - pending_next
        )
        logistics = {
            "depot_capacity": _round_vector(DEPOT_CAPACITY),
            "depot_stock_before": _round_vector(context.stock_before),
            "pending_arrivals": _round_vector(context.pending_arrivals),
            "pending_arrivals_landed": _round_vector(context.pending_landed),
            "pending_arrivals_held": _round_vector(context.pending_held),
            "depot_stock_after_pending": _round_vector(context.stock_after_pending),
            "depot_damage_penalty": _round_vector(context.damage_penalty),
            "depot_damage_days_remaining": [
                int(value) for value in context.damage_days_remaining.tolist()
            ],
            "depot_damage_factor": _round_vector(context.depot_factor),
            "road_capacity": round(context.road_capacity, 8),
            "throughput_factor": _round_vector(context.throughput),
            "mutual_aid_transfers": [asdict(event) for event in context.transfers],
            "mutual_aid_net": _round_vector(context.transfer_net),
            "depot_stock_ready": _round_vector(context.stock_ready),
            "same_day_delivery_scheduled": _round_vector(same_day_scheduled),
            "same_day_delivery_landed": _round_vector(same_day_landed),
            "same_day_delivery_held": _round_vector(same_day_held),
            "delayed_delivery_scheduled": _round_vector(delayed_scheduled),
            "repair_reserve": _round_vector(repair_reserve),
            "repair_request": _round_vector(repair_request),
            "repair_dispatch": _round_vector(repair_dispatch),
            "repair_supply": _round_vector(repair_supply),
            "spoilage": _round_vector(spoilage),
            "depot_stock_end": _round_vector(stock_end),
            "pending_next_day": _round_vector(pending_next),
            "capacity_overflow": _round_vector(capacity_overflow),
            "conservation_residual": _round_vector(conservation_residual),
        }
        record = {
            "day": context.shock.day,
            "shock": asdict(context.shock),
            "available_budget": round(context.available_budget, 8),
            "services_before": _round_vector(context.before),
            "services_after_shock": _round_vector(context.shocked),
            "raw_action": None if raw_action is None else _round_vector(raw_action),
            "raw_proposal": _round_vector(proposal),
            "lower_bounds": _round_vector(context.lower),
            "upper_bounds": _round_vector(context.upper),
            "allocation": _round_vector(allocation),
            "projection": projection,
            "planner_evidence": planner_evidence,
            "support": _round_vector(context.support),
            "throughput": _round_vector(context.throughput),
            "gain": _round_vector(gain),
            "strain": _round_vector(strain),
            "services_end": _round_vector(end),
            "resilience": round(resilience, 8),
            "reward": round(float(reward), 8),
            "logistics": logistics,
        }
        self.trajectory.append(record)
        self._q = end
        self._stocks = stock_end
        self._pending = pending_next
        self._damage_remaining = np.maximum(0, self._damage_remaining - 1)
        self._day_index += 1
        self._terminated = self._day_index >= self.scenario.horizon_days
        if self._terminated:
            self._context = None
            observation = np.zeros(OBSERVATION_SIZE_V2, dtype=np.float32)
        else:
            self._context = self._make_context()
            observation = self._observation()
        return observation, float(reward), self._terminated, False, {"day": record}

    def render(self) -> list[dict[str, Any]]:
        return list(self.trajectory)


class CyclingScenarioEnvV2(gym.Env[np.ndarray, np.ndarray]):
    """Deterministically cycle complete scenario/seed units for SB3 v2 training."""

    def __init__(self, scenarios: list[tuple[Scenario, int]]):
        if not scenarios:
            raise ValueError("at least one training scenario is required")
        self.scenarios = scenarios
        self.index = 0
        first, first_seed = scenarios[0]
        self.inner = CityRecoveryEnvV2(first, first_seed)
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scenario, shock_seed = self.scenarios[self.index % len(self.scenarios)]
        self.index += 1
        self.inner.set_scenario(scenario, shock_seed)
        return self.inner.reset(seed=shock_seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.inner.step(action)

    def render(self) -> list[dict[str, Any]]:
        return self.inner.render()


def ortools_proposal_v2(
    context: DayContextV2, priorities: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the disclosed fair v2 x-allocation/y-dispatch GLOP baseline."""

    centrality = DEPENDENCIES.sum(axis=0)
    reserve = RESERVE_DRAW_FRACTION * DEPOT_CAPACITY * (1.0 - context.shocked)
    coefficients = (
        np.asarray(priorities, dtype=np.float64)
        * (1.0 - context.shocked)
        * (ETA * context.support + 0.04 * centrality)
        * context.throughput
    )
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver is unavailable")
    solver.SetNumThreads(1)
    allocations = [
        solver.NumVar(
            float(context.lower[index]), float(context.upper[index]), f"x_{SERVICES[index]}"
        )
        for index in range(5)
    ]
    dispatches = [
        solver.NumVar(0.0, solver.infinity(), f"y_{SERVICES[index]}") for index in range(5)
    ]
    solver.Add(sum(allocations) == context.available_budget)
    for index in range(5):
        solver.Add(
            dispatches[index]
            <= float(context.stock_ready[index]) + IMMEDIATE_DELIVERY_FRACTION * allocations[index]
        )
        solver.Add(dispatches[index] <= allocations[index] + float(reserve[index]))
    solver.Maximize(sum(float(coefficients[index]) * dispatches[index] for index in range(5)))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools v2 baseline failed with status {status}")
    proposal = np.array([variable.solution_value() for variable in allocations])
    dispatch_solution = np.array([variable.solution_value() for variable in dispatches])
    return proposal, {
        "library": "OR-Tools",
        "library_version": ortools.__version__,
        "solver": "GLOP",
        "status": "OPTIMAL",
        "objective": (
            "maximize sum(priority * service_deficit * "
            "(eta * support + 0.04 * dependency_centrality) * "
            "throughput_factor * dispatch_y)"
        ),
        "decision_variables": {
            "x": "five allocation units; sum(x) equals the full available daily budget",
            "y": (
                "five raw repair dispatch units; y <= stock_ready + 0.65*x "
                "and y <= x + 0.04*capacity*deficit"
            ),
        },
        "objective_coefficients": _round_vector(coefficients),
        "allocation_solution": _round_vector(proposal),
        "dispatch_solution": _round_vector(dispatch_solution),
        "future_shocks_visible": False,
    }


def _summarize_v2(
    planner: str,
    trajectory: list[dict[str, Any]],
    normalized_priorities: np.ndarray,
) -> dict[str, Any]:
    resilience = np.array([day["resilience"] for day in trajectory], dtype=np.float64)
    before_resilience = np.array(
        [normalized_priorities @ np.asarray(day["services_before"]) for day in trajectory]
    )
    shocked_resilience = np.array(
        [normalized_priorities @ np.asarray(day["services_after_shock"]) for day in trajectory]
    )
    largest_loss_index = int(np.argmax(before_resilience - shocked_resilience))
    recovery_target = float(before_resilience[largest_loss_index])
    recovery_day = len(trajectory) + 1
    for index in range(largest_loss_index, len(trajectory)):
        if resilience[index] >= recovery_target - CONSTRAINT_TOLERANCE:
            recovery_day = index - largest_loss_index
            break
    violations = sum(day["projection"]["constraint_violations"] for day in trajectory)
    breakdown = {
        name: sum(day["projection"]["violation_breakdown"][name] for day in trajectory)
        for name in (
            "sum_violations",
            "budget_violations",
            "lower_violations",
            "upper_violations",
        )
    }
    max_conservation_residual = max(
        abs(value) for day in trajectory for value in day["logistics"]["conservation_residual"]
    )
    return {
        "planner": planner,
        "rauc": round(float(np.mean(resilience)), 8),
        "final_resilience": round(float(resilience[-1]), 8),
        "minimum_resilience": round(float(np.min(resilience)), 8),
        "post_shock_recovery_shortfall_auc": round(
            float(np.mean(np.maximum(0.0, recovery_target - resilience[largest_loss_index:]))),
            8,
        ),
        "days_to_pre_shock_recovery_after_largest_loss": recovery_day,
        "largest_shock_loss_day": largest_loss_index + 1,
        "critical_service_days": sum(
            int(value < 0.30) for day in trajectory for value in day["services_end"]
        ),
        "total_projection_distance": round(
            sum(day["projection"]["distance"] for day in trajectory), 8
        ),
        "constraint_violations": violations,
        "violation_breakdown": breakdown,
        "total_food_spoilage": round(sum(day["logistics"]["spoilage"][2] for day in trajectory), 8),
        "total_mutual_aid_units": round(
            sum(
                event["units"]
                for day in trajectory
                for event in day["logistics"]["mutual_aid_transfers"]
            ),
            8,
        ),
        "max_logistics_conservation_residual": round(float(max_conservation_residual), 8),
        "final_depot_stock": trajectory[-1]["logistics"]["depot_stock_end"],
        "final_pending_arrivals": trajectory[-1]["logistics"]["pending_next_day"],
        "trajectory_sha256": canonical_hash(trajectory),
        "trajectory": trajectory,
    }


def rollout_candidate_v2(
    scenario: Scenario,
    seed: int,
    action_provider: Callable[[np.ndarray], np.ndarray],
    schedule: Sequence[ShockV2] | None = None,
) -> dict[str, Any]:
    shared_schedule = (
        generate_shock_schedule_v2(scenario, seed) if schedule is None else list(schedule)
    )
    env = CityRecoveryEnvV2(scenario, seed, shared_schedule)
    observation, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action = action_provider(observation)
        observation, _, terminated, _, _ = env.step(action)
    priorities = np.asarray(scenario.priorities, dtype=np.float64)
    normalized_priorities = priorities / priorities.sum()
    return _summarize_v2("stable_baselines3_ppo_v2_onnx", env.trajectory, normalized_priorities)


def rollout_baseline_v2(
    scenario: Scenario,
    seed: int,
    schedule: Sequence[ShockV2] | None = None,
) -> dict[str, Any]:
    shared_schedule = (
        generate_shock_schedule_v2(scenario, seed) if schedule is None else list(schedule)
    )
    env = CityRecoveryEnvV2(scenario, seed, shared_schedule)
    env.reset(seed=seed)
    priorities = np.asarray(scenario.priorities, dtype=np.float64)
    terminated = False
    while not terminated:
        proposal, evidence = ortools_proposal_v2(env.current_context(), priorities)
        _, _, terminated, _, _ = env.step_proposal(proposal, evidence)
    normalized_priorities = priorities / priorities.sum()
    return _summarize_v2("ortools_glop_baseline_v2", env.trajectory, normalized_priorities)


def compare_v2(scenario: Scenario, seed: int, policy_bundle: Any) -> dict[str, Any]:
    """Compare PPO v2 and GLOP v2 on the exact same precomputed dynamics tape."""

    schedule = generate_shock_schedule_v2(scenario, seed)
    schedule_payload = [asdict(shock) for shock in schedule]
    metadata = policy_bundle.metadata
    input_name = metadata["export"]["input_name"]
    output_name = metadata["export"]["output_name"]

    def action_provider(observation: np.ndarray) -> np.ndarray:
        outputs = policy_bundle.session.run(
            [output_name], {input_name: observation.reshape(1, -1).astype(np.float32)}
        )
        return np.asarray(outputs[0][0], dtype=np.float64)

    baseline = rollout_baseline_v2(scenario, seed, schedule)
    candidate = rollout_candidate_v2(scenario, seed, action_provider, schedule)
    baseline_shocks = [day["shock"] for day in baseline["trajectory"]]
    candidate_shocks = [day["shock"] for day in candidate["trajectory"]]
    if baseline_shocks != schedule_payload or candidate_shocks != schedule_payload:
        raise RuntimeError("v2 planners did not consume the shared precomputed shock tape")
    delta = candidate["rauc"] - baseline["rauc"]
    if delta > 1e-8:
        outcome = "candidate_higher_rauc"
    elif delta < -1e-8:
        outcome = "baseline_higher_rauc"
    else:
        outcome = "rauc_tie"
    return {
        "schema_version": "3.0.0",
        "engine_version": "city-recovery-env-v2",
        "environment": {
            "id": "CityRecoveryEnv-v2",
            "version": "2.0.0",
            "observation_count": OBSERVATION_SIZE_V2,
            "action_count": 5,
            "spec_sha256": ENGINE_V2_SPEC_SHA256,
        },
        "engine_spec": ENGINE_V2_SPEC,
        "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        "seed": seed,
        "generator": "numpy.PCG64",
        "scenario": scenario.model_dump(mode="json"),
        "services": list(SERVICES),
        "observation_order": list(OBSERVATION_ORDER_V2),
        "action_order": list(ACTION_ORDER_V2),
        "shock_schedule": schedule_payload,
        "shock_schedule_sha256": canonical_hash(schedule_payload),
        "policy": {
            "id": metadata["id"],
            "artifact_type": metadata["artifact_type"],
            "algorithm": metadata["training"]["algorithm"],
            "runtime": "ONNX Runtime CPUExecutionProvider",
            "sha256": policy_bundle.onnx_sha256,
            "sb3_checkpoint_sha256": policy_bundle.sb3_sha256,
            "parity_report_sha256": policy_bundle.parity_sha256,
            "disclosure": metadata["disclosure"],
            "predecessor_policy": metadata.get("predecessor_policy"),
            "legacy_candidate": metadata.get("legacy_candidate"),
        },
        "baseline_spec": {
            "id": "ortools-glop-visible-v2",
            "library": "OR-Tools",
            "library_version": ortools.__version__,
            "solver": "GLOP",
            "objective": (
                "maximize priority-weighted, deficit-weighted feasible repair dispatch "
                "under the same depot stock, latency, throughput, bounds, and "
                "full-budget constraints"
            ),
            "future_shocks_visible": False,
            "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "primary_metric": "weighted_daily_resilience_auc",
            "candidate_minus_baseline": round(delta, 8),
            "recovery_shortfall_candidate_minus_baseline": round(
                candidate["post_shock_recovery_shortfall_auc"]
                - baseline["post_shock_recovery_shortfall_auc"],
                8,
            ),
            "recovery_days_candidate_minus_baseline": (
                candidate["days_to_pre_shock_recovery_after_largest_loss"]
                - baseline["days_to_pre_shock_recovery_after_largest_loss"]
            ),
            "outcome": outcome,
        },
        "limitations": [
            (
                "All dynamics, depot coefficients, authored scenario families, and "
                "training inputs are synthetic and non-empirical."
            ),
            (
                "The SB3 PPO policy is a local simulation candidate, not a forecast or "
                "municipal deployment recommendation."
            ),
            (
                "The v1 policy and environment remain separately versioned for exact "
                "restoration of historical results."
            ),
        ],
    }
