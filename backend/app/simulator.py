from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import ortools
from gymnasium import spaces
from ortools.linear_solver import pywraplp

from backend.app.models import Scenario
from backend.app.recommendations import run_recommendations

SERVICES = ("transport", "housing", "food", "healthcare", "public_services")
SHOCKS = ("aftershock", "supply", "epidemic", "utility", "weather")
OBSERVATION_ORDER = (
    *(f"service_{name}" for name in SERVICES),
    *(f"priority_{name}" for name in SERVICES),
    *(f"support_{name}" for name in SERVICES),
    *(f"shock_impact_{name}" for name in SERVICES),
    "available_budget_fraction",
    "horizon_remaining_fraction",
    "shock_severity",
)
OBSERVATION_SIZE = len(OBSERVATION_ORDER)
ACTION_ORDER = SERVICES
SHOCK_TYPE_PROBABILITIES = np.array([0.24, 0.22, 0.18, 0.20, 0.16])
SHOCK_IMPACTS = np.array(
    [
        [0.65, 1.00, 0.20, 0.35, 0.45],
        [0.35, 0.05, 1.00, 0.55, 0.10],
        [0.10, 0.20, 0.25, 1.00, 0.35],
        [0.30, 0.35, 0.45, 0.70, 1.00],
        [0.75, 0.55, 0.50, 0.40, 0.60],
    ],
    dtype=np.float64,
)
SHOCK_BUDGET_FACTORS = np.array([0.15, 0.25, 0.10, 0.30, 0.25])
DEPENDENCIES = np.array(
    [
        [0.00, 0.10, 0.10, 0.20, 0.60],
        [0.30, 0.00, 0.15, 0.10, 0.45],
        [0.45, 0.10, 0.00, 0.15, 0.30],
        [0.30, 0.10, 0.20, 0.00, 0.40],
        [0.35, 0.20, 0.15, 0.30, 0.00],
    ],
    dtype=np.float64,
)
ETA = np.array([0.18, 0.16, 0.20, 0.22, 0.17])
DELTA = np.array([0.010, 0.012, 0.015, 0.018, 0.010])
CONSTRAINT_TOLERANCE = 1e-7


@dataclass(frozen=True)
class Shock:
    day: int
    type: str | None
    severity: float
    impact: list[float]
    budget_factor: float
    forced: bool


@dataclass(frozen=True)
class DayContext:
    before: np.ndarray
    shocked: np.ndarray
    support: np.ndarray
    available_budget: float
    lower: np.ndarray
    upper: np.ndarray
    shock: Shock


def _round_vector(values: np.ndarray) -> list[float]:
    return [float(round(value, 8)) for value in values.tolist()]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def generate_shock_schedule(scenario: Scenario, seed: int) -> list[Shock]:
    """Generate the complete PCG64 shock tape before either planner runs."""
    rng = np.random.Generator(np.random.PCG64(seed))
    schedule: list[Shock] = []
    for day in range(1, scenario.horizon_days + 1):
        occurs = bool(rng.random() < scenario.shock_probability)
        shock_index = int(rng.choice(len(SHOCKS), p=SHOCK_TYPE_PROBABILITIES))
        severity_draw = scenario.severity_min + (
            scenario.severity_max - scenario.severity_min
        ) * float(rng.beta(2.0, 5.0))
        if occurs:
            schedule.append(
                Shock(
                    day=day,
                    type=SHOCKS[shock_index],
                    severity=float(round(severity_draw, 8)),
                    impact=_round_vector(SHOCK_IMPACTS[shock_index]),
                    budget_factor=float(SHOCK_BUDGET_FACTORS[shock_index]),
                    forced=False,
                )
            )
        else:
            schedule.append(Shock(day, None, 0.0, [0.0] * 5, 0.0, False))

    forced_overrides = [] if scenario.forced_shock is None else [scenario.forced_shock]
    forced_overrides.extend(scenario.forced_shocks)
    for forced in forced_overrides:
        shock_index = SHOCKS.index(forced.type)
        schedule[forced.day - 1] = Shock(
            day=forced.day,
            type=forced.type,
            severity=float(forced.severity),
            impact=_round_vector(SHOCK_IMPACTS[shock_index]),
            budget_factor=float(SHOCK_BUDGET_FACTORS[shock_index]),
            forced=True,
        )
    return schedule


def project_capped_simplex(
    proposal: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project onto the common bounded budget simplex with deterministic rounding."""
    proposal = np.asarray(proposal, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if proposal.shape != (5,) or not np.all(np.isfinite(proposal)):
        raise ValueError("planner proposal must contain five finite allocations")
    if float(lower.sum()) > total + 1e-9 or float(upper.sum()) < total - 1e-9:
        raise ValueError("allocation constraints are infeasible")
    lo = float(np.min(proposal - upper))
    hi = float(np.max(proposal - lower))
    for _ in range(64):
        midpoint = (lo + hi) / 2.0
        candidate = np.clip(proposal - midpoint, lower, upper)
        if float(candidate.sum()) > total:
            lo = midpoint
        else:
            hi = midpoint
    projected = np.clip(proposal - ((lo + hi) / 2.0), lower, upper)
    rounded = np.round(projected, 8)
    residual = round(float(total - rounded.sum()), 8)
    if residual:
        order = (
            np.argsort(-(upper - rounded))
            if residual > 0
            else np.argsort(-(rounded - lower))
        )
        for index in order:
            capacity = (
                upper[index] - rounded[index]
                if residual > 0
                else rounded[index] - lower[index]
            )
            adjustment = np.sign(residual) * min(abs(residual), float(capacity))
            rounded[index] = round(float(rounded[index] + adjustment), 8)
            residual = round(float(residual - adjustment), 8)
            if residual == 0:
                break
    bindings = [
        {
            "service": SERVICES[index],
            "lower": bool(abs(rounded[index] - lower[index]) <= CONSTRAINT_TOLERANCE),
            "upper": bool(abs(rounded[index] - upper[index]) <= CONSTRAINT_TOLERANCE),
        }
        for index in range(5)
    ]
    return rounded, {
        "distance": round(float(np.linalg.norm(rounded - proposal)), 8),
        "bindings": bindings,
        "sum": round(float(rounded.sum()), 8),
    }


def action_to_proposal(action: np.ndarray, budget: float) -> np.ndarray:
    """Convert five bounded policy logits into a positive budget proposal."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (5,) or not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain five finite values")
    clipped = np.clip(action, -1.0, 1.0)
    exponentials = np.exp(clipped - float(np.max(clipped)))
    return budget * exponentials / float(exponentials.sum())


def measure_constraints(
    allocation: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray
) -> dict[str, int]:
    allocation_sum = float(allocation.sum())
    measurements = {
        "sum_violations": int(abs(allocation_sum - total) > CONSTRAINT_TOLERANCE),
        "budget_violations": int(allocation_sum > total + CONSTRAINT_TOLERANCE),
        "lower_violations": int(
            np.count_nonzero(allocation < lower - CONSTRAINT_TOLERANCE)
        ),
        "upper_violations": int(
            np.count_nonzero(allocation > upper + CONSTRAINT_TOLERANCE)
        ),
    }
    measurements["total"] = sum(measurements.values())
    return measurements


class CityRecoveryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic five-resource recovery environment used by training and runtime."""

    metadata = {"render_modes": ["trajectory"], "render_fps": 1}

    def __init__(self, scenario: Scenario, shock_seed: int = 0):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            high=np.ones(OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.scenario = scenario
        self.shock_seed = shock_seed
        self.schedule: list[Shock] = []
        self.trajectory: list[dict[str, Any]] = []
        self._q = np.zeros(5, dtype=np.float64)
        self._priorities = np.ones(5, dtype=np.float64)
        self._normalized_priorities = np.full(5, 0.2, dtype=np.float64)
        self._day_index = 0
        self._context: DayContext | None = None
        self._terminated = False

    def set_scenario(self, scenario: Scenario, shock_seed: int) -> None:
        self.scenario = scenario
        self.shock_seed = shock_seed

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
        if options and "shock_seed" in options:
            self.shock_seed = int(options["shock_seed"])
        elif seed is not None:
            self.shock_seed = seed
        self.schedule = generate_shock_schedule(self.scenario, self.shock_seed)
        self.trajectory = []
        self._q = np.asarray(self.scenario.initial_services, dtype=np.float64)
        self._priorities = np.asarray(self.scenario.priorities, dtype=np.float64)
        self._normalized_priorities = self._priorities / float(self._priorities.sum())
        self._day_index = 0
        self._terminated = False
        self._context = self._make_context()
        return self._observation(), {
            "shock_schedule_sha256": canonical_hash([asdict(item) for item in self.schedule]),
            "shock_seed": self.shock_seed,
        }

    def _make_context(self) -> DayContext:
        shock = self.schedule[self._day_index]
        before = self._q.copy()
        impact = np.asarray(shock.impact, dtype=np.float64)
        shocked = np.clip(before * (1.0 - shock.severity * impact), 0.0, 1.0)
        support = 0.55 + 0.45 * (DEPENDENCIES @ shocked)
        available_budget = self.scenario.daily_budget * (
            1.0 - shock.severity * shock.budget_factor
        )
        lower = np.where(shocked < 0.30, 0.04 * available_budget, 0.0)
        upper = np.full(5, 0.50 * available_budget, dtype=np.float64)
        return DayContext(
            before=before,
            shocked=shocked,
            support=support,
            available_budget=float(available_budget),
            lower=lower,
            upper=upper,
            shock=shock,
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
                np.array(
                    [
                        context.available_budget / 500.0,
                        remaining,
                        context.shock.severity,
                    ]
                ),
            )
        )
        return np.asarray(values, dtype=np.float32)

    def current_context(self) -> DayContext:
        if self._context is None or self._terminated:
            raise RuntimeError("environment has no active day")
        return self._context

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        context = self.current_context()
        raw_action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        proposal = action_to_proposal(raw_action, context.available_budget)
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
        gain = (
            ETA
            * np.sqrt(allocation / 200.0)
            * context.support
            * (1.0 - context.shocked)
        )
        strain = (
            DELTA
            * np.maximum(0.0, 0.35 - context.shocked)
            * (1.0 - allocation / context.available_budget)
        )
        end = np.clip(context.shocked + gain - strain, 0.0, 1.0)
        resilience = float(self._normalized_priorities @ end)
        shocked_resilience = float(self._normalized_priorities @ context.shocked)
        reward = resilience + 0.35 * (resilience - shocked_resilience)
        reward -= 0.0001 * projection["distance"] / context.available_budget
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
            "gain": _round_vector(gain),
            "strain": _round_vector(strain),
            "services_end": _round_vector(end),
            "resilience": round(resilience, 8),
            "reward": round(float(reward), 8),
        }
        self.trajectory.append(record)
        self._q = end
        self._day_index += 1
        self._terminated = self._day_index >= self.scenario.horizon_days
        if self._terminated:
            self._context = None
            observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        else:
            self._context = self._make_context()
            observation = self._observation()
        return observation, float(reward), self._terminated, False, {"day": record}

    def render(self) -> list[dict[str, Any]]:
        return list(self.trajectory)


class CyclingScenarioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministically cycles whole scenario/seed units during SB3 training."""

    def __init__(self, scenarios: list[tuple[Scenario, int]]):
        if not scenarios:
            raise ValueError("at least one training scenario is required")
        self.scenarios = scenarios
        self.index = 0
        first, first_seed = scenarios[0]
        self.inner = CityRecoveryEnv(first, first_seed)
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

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.inner.step(action)

    def render(self) -> list[dict[str, Any]]:
        return self.inner.render()


def ortools_proposal(
    context: DayContext, priorities: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the visible one-day linear recovery allocation with OR-Tools GLOP."""
    centrality = DEPENDENCIES.sum(axis=0)
    coefficients = priorities * (1.0 - context.shocked) * (
        ETA * context.support + 0.04 * centrality
    )
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver is unavailable")
    solver.SetNumThreads(1)
    allocations = [
        solver.NumVar(float(context.lower[index]), float(context.upper[index]), SERVICES[index])
        for index in range(5)
    ]
    solver.Add(sum(allocations) == context.available_budget)
    solver.Maximize(sum(float(coefficients[index]) * allocations[index] for index in range(5)))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools baseline failed with status {status}")
    proposal = np.array([variable.solution_value() for variable in allocations])
    return proposal, {
        "library": "OR-Tools",
        "library_version": ortools.__version__,
        "solver": "GLOP",
        "status": "OPTIMAL",
        "objective": (
            "maximize sum(priority * deficit * "
            "(eta * support + 0.04 * dependency_centrality) * allocation)"
        ),
        "objective_coefficients": _round_vector(coefficients),
    }


def _summarize(
    planner: str, trajectory: list[dict[str, Any]], normalized_priorities: np.ndarray
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
    return {
        "planner": planner,
        "rauc": round(float(np.mean(resilience)), 8),
        "final_resilience": round(float(resilience[-1]), 8),
        "minimum_resilience": round(float(np.min(resilience)), 8),
        "post_shock_recovery_shortfall_auc": round(
            float(
                np.mean(
                    np.maximum(
                        0.0, recovery_target - resilience[largest_loss_index:]
                    )
                )
            ),
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
        "trajectory_sha256": canonical_hash(trajectory),
        "trajectory": trajectory,
    }


def rollout_candidate(
    scenario: Scenario,
    seed: int,
    action_provider: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    env = CityRecoveryEnv(scenario, seed)
    observation, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action = action_provider(observation)
        observation, _, terminated, _, _ = env.step(action)
    priorities = np.asarray(scenario.priorities)
    normalized_priorities = priorities / priorities.sum()
    return _summarize(
        "stable_baselines3_ppo_onnx", env.trajectory, normalized_priorities
    )


def rollout_baseline(scenario: Scenario, seed: int) -> dict[str, Any]:
    env = CityRecoveryEnv(scenario, seed)
    env.reset(seed=seed)
    priorities = np.asarray(scenario.priorities, dtype=np.float64)
    terminated = False
    while not terminated:
        proposal, evidence = ortools_proposal(env.current_context(), priorities)
        _, _, terminated, _, _ = env.step_proposal(proposal, evidence)
    normalized_priorities = priorities / priorities.sum()
    return _summarize("ortools_glop_baseline", env.trajectory, normalized_priorities)


def compare(scenario: Scenario, seed: int, policy_bundle: Any) -> dict[str, Any]:
    schedule = generate_shock_schedule(scenario, seed)
    schedule_payload = [asdict(shock) for shock in schedule]
    metadata = policy_bundle.metadata
    input_name = metadata["export"]["input_name"]
    output_name = metadata["export"]["output_name"]

    def action_provider(observation: np.ndarray) -> np.ndarray:
        outputs = policy_bundle.session.run(
            [output_name], {input_name: observation.reshape(1, -1).astype(np.float32)}
        )
        return np.asarray(outputs[0][0], dtype=np.float64)

    baseline = rollout_baseline(scenario, seed)
    candidate = rollout_candidate(scenario, seed, action_provider)
    delta = candidate["rauc"] - baseline["rauc"]
    if delta > 1e-8:
        outcome = "candidate_higher_rauc"
    elif delta < -1e-8:
        outcome = "baseline_higher_rauc"
    else:
        outcome = "rauc_tie"
    return {
        "schema_version": "2.2.0",
        "seed": seed,
        "generator": "numpy.PCG64",
        "scenario": scenario.model_dump(mode="json"),
        "services": list(SERVICES),
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
            "legacy_candidate": metadata["legacy_candidate"],
        },
        "baseline_spec": {
            "id": "ortools-glop-visible-v1",
            "library": "OR-Tools",
            "library_version": ortools.__version__,
            "solver": "GLOP",
            "objective": (
                "maximize immediate priority-weighted deficit recovery under the same "
                "daily bounds and budget"
            ),
            "future_shocks_visible": False,
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
        "recommendations": run_recommendations({
            "candidate": candidate,
            "baseline": baseline,
            "comparison": {
                "outcome": outcome,
                "candidate_minus_baseline": round(delta, 8),
            },
            "scenario": scenario.model_dump(mode="json"),
            "shock_schedule": schedule_payload,
        }, SERVICES),
        "limitations": [
            (
                "All dynamics, authored scenario families, and training inputs are "
                "synthetic and non-empirical."
            ),
            (
                "The SB3 PPO policy is a local simulation candidate, not a forecast or "
                "municipal deployment recommendation."
            ),
            (
                "The accepted linear candidate remains disclosed as a separate legacy "
                "non-PPO artifact and is not used as this policy."
            ),
        ],
    }
