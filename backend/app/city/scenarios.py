"""Canonical scenario families, split rosters, and deterministic disaster tapes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from backend.app.models import ForcedShock, Scenario as ScenarioModel, ShockName
from backend.app.city.physics import (
    SHOCK_BUDGET_FACTORS,
    SHOCK_IMPACTS,
    SHOCK_TYPE_PROBABILITIES,
    SHOCKS,
    round_vector,
)

PUBLIC_RISK_PERSISTENCE = 0.72
PUBLIC_RISK_NOISE = 0.012
PUBLIC_RISK_EVENT_PULSE = 0.055
PUBLIC_RISK_TOTAL_CAP = 0.48


@dataclass(frozen=True)
class Shock:
    day: int
    type: str | None
    severity: float
    impact: list[float]
    budget_factor: float
    forced: bool
    occurrence_probability: float
    occurrence_draw: float
    public_risk_before: list[float]
    public_risk_next: list[float]
    assessment_tail: bool


@dataclass(frozen=True)
class ScenarioFamily:
    id: str
    label: str
    initial_center: tuple[float, float, float, float, float]
    priorities: tuple[float, float, float, float, float]
    budget_center: float
    crew_center: float
    shock_probability: float
    severity_min: float
    severity_max: float
    primary_type: ShockName
    secondary_type: ShockName
    target: float = 0.55

    def tape_seed(self, case_seed: int) -> int:
        """Derive an independent PCG64 tape seed for this family/case block."""

        payload = f"city-recovery-v3:{self.id}:{case_seed}:tape".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

    def build(self, seed: int) -> ScenarioModel:
        family_key = int.from_bytes(
            hashlib.sha256(self.id.encode("ascii")).digest()[:8], "little"
        )
        rng = np.random.Generator(np.random.PCG64(seed ^ family_key))
        initial = np.clip(
            np.asarray(self.initial_center) + rng.uniform(-0.045, 0.045, size=5),
            0.05,
            0.95,
        )
        priorities = np.clip(
            np.asarray(self.priorities) + rng.uniform(-0.075, 0.075, size=5),
            0.5,
            2.0,
        )
        budget = float(np.clip(self.budget_center + rng.integers(-16, 17), 50, 500))
        crew = float(np.clip(self.crew_center + rng.integers(-14, 15), 50, 300))
        first_day = int(rng.integers(8, 13))
        second_day = int(rng.integers(18, 24))
        first_severity = float(
            round(rng.uniform(self.severity_min, self.severity_max), 8)
        )
        second_severity = float(
            round(rng.uniform(self.severity_min, self.severity_max), 8)
        )
        probability = float(
            round(
                np.clip(
                    self.shock_probability + rng.uniform(-0.025, 0.025), 0.08, 0.38
                ),
                8,
            )
        )
        return ScenarioModel(
            name=f"{self.label} / {seed}",
            horizon_days=30,
            daily_budget=budget,
            daily_crew_pool=crew,
            initial_services=[round(float(value), 8) for value in initial],
            priorities=[round(float(value), 8) for value in priorities],
            shock_probability=probability,
            severity_min=self.severity_min,
            severity_max=self.severity_max,
            forced_shock=ForcedShock(
                day=first_day,
                type=self.primary_type,
                severity=first_severity,
            ),
            forced_shocks=[
                ForcedShock(
                    day=second_day,
                    type=self.secondary_type,
                    severity=second_severity,
                )
            ],
            recovery_targets=[self.target] * 5,
            assessment_tail_days=3,
        )


def _bounded_risk(values: np.ndarray) -> np.ndarray:
    risk = np.clip(np.asarray(values, dtype=np.float64), 0.0, 0.22)
    total = float(risk.sum())
    if total > PUBLIC_RISK_TOTAL_CAP:
        risk *= PUBLIC_RISK_TOTAL_CAP / total
    return risk


def generate_disaster_tape(scenario: ScenarioModel, seed: int) -> list[Shock]:
    """Precompute a shared tape from public causal risk signals and hidden draws."""

    forced_by_day: dict[int, ForcedShock] = {}
    if scenario.forced_shock is not None:
        forced_by_day[scenario.forced_shock.day] = scenario.forced_shock
    for forced in scenario.forced_shocks:
        forced_by_day[forced.day] = forced

    rng = np.random.Generator(np.random.PCG64(seed ^ 0xC17E_C0DE))
    base_risk = scenario.shock_probability * SHOCK_TYPE_PROBABILITIES
    initial_noise = scenario.shock_probability * rng.uniform(-0.03, 0.03, size=5)
    risk = _bounded_risk(base_risk + initial_noise)
    schedule: list[Shock] = []
    tail_start = scenario.horizon_days - scenario.assessment_tail_days + 1
    for day in range(1, scenario.horizon_days + 1):
        risk_before = risk.copy()
        occurrence_draw = float(rng.random())
        type_draw = float(rng.random())
        severity_draw = scenario.severity_min + (
            scenario.severity_max - scenario.severity_min
        ) * float(rng.beta(2.0, 4.5))
        risk_noise = scenario.shock_probability * rng.uniform(
            -PUBLIC_RISK_NOISE,
            PUBLIC_RISK_NOISE,
            size=5,
        )
        assessment_tail = day >= tail_start
        forced = forced_by_day.get(day)
        occurrence_probability = 0.0 if assessment_tail else float(risk_before.sum())

        shock_type: str | None = None
        forced_flag = False
        if not assessment_tail and forced is not None:
            shock_type = forced.type
            severity = float(forced.severity)
            forced_flag = True
        elif not assessment_tail and occurrence_draw < occurrence_probability:
            cumulative = np.cumsum(risk_before / max(occurrence_probability, 1e-12))
            shock_index = int(np.searchsorted(cumulative, type_draw, side="right"))
            shock_type = SHOCKS[min(shock_index, len(SHOCKS) - 1)]
            severity = float(round(severity_draw, 8))
        else:
            severity = 0.0

        if shock_type is None:
            impact = np.zeros(5, dtype=np.float64)
            budget_factor = 0.0
        else:
            shock_index = SHOCKS.index(shock_type)
            impact = SHOCK_IMPACTS[shock_index]
            budget_factor = float(SHOCK_BUDGET_FACTORS[shock_index])

        next_risk = (
            base_risk
            + PUBLIC_RISK_PERSISTENCE * (risk_before - base_risk)
            + risk_noise
        )
        if shock_type is not None:
            shock_index = SHOCKS.index(shock_type)
            next_risk[shock_index] += PUBLIC_RISK_EVENT_PULSE
            if shock_type == "aftershock":
                next_risk[0] += 0.035
            elif shock_type in ("weather", "utility"):
                next_risk[1] += 0.015
        if day + 1 >= tail_start:
            next_risk = np.zeros(5, dtype=np.float64)
        else:
            next_risk = _bounded_risk(next_risk)
        risk = next_risk
        schedule.append(
            Shock(
                day=day,
                type=shock_type,
                severity=severity,
                impact=round_vector(impact),
                budget_factor=budget_factor,
                forced=forced_flag,
                occurrence_probability=float(round(occurrence_probability, 8)),
                occurrence_draw=float(round(occurrence_draw, 8)),
                public_risk_before=round_vector(risk_before),
                public_risk_next=round_vector(next_risk),
                assessment_tail=assessment_tail,
            )
        )
    return schedule


TRAINING_FAMILIES = (
    ScenarioFamily(
        "v3_train_transit_nexus",
        "Transit nexus cascade",
        (0.22, 0.47, 0.38, 0.43, 0.30),
        (1.70, 0.85, 1.10, 1.35, 1.05),
        154,
        126,
        0.24,
        0.09,
        0.30,
        "aftershock",
        "weather",
    ),
    ScenarioFamily(
        "v3_train_displacement",
        "Compound displacement",
        (0.44, 0.18, 0.34, 0.36, 0.27),
        (0.95, 1.75, 1.20, 1.35, 1.15),
        176,
        142,
        0.23,
        0.09,
        0.30,
        "utility",
        "epidemic",
    ),
    ScenarioFamily(
        "v3_train_supply_chain",
        "Regional supply interruption",
        (0.40, 0.37, 0.17, 0.45, 0.34),
        (1.05, 1.10, 1.80, 1.40, 0.90),
        142,
        122,
        0.22,
        0.08,
        0.29,
        "supply",
        "utility",
    ),
    ScenarioFamily(
        "v3_train_health_surge",
        "Healthcare capacity surge",
        (0.37, 0.42, 0.35, 0.16, 0.32),
        (0.90, 0.95, 1.15, 1.95, 1.20),
        191,
        148,
        0.26,
        0.10,
        0.33,
        "epidemic",
        "supply",
    ),
    ScenarioFamily(
        "v3_train_grid_failure",
        "Grid and civic failure",
        (0.36, 0.32, 0.40, 0.29, 0.15),
        (1.10, 1.20, 1.20, 1.55, 1.80),
        166,
        132,
        0.25,
        0.10,
        0.32,
        "utility",
        "aftershock",
    ),
    ScenarioFamily(
        "v3_train_weather_isolation",
        "Weather isolation",
        (0.25, 0.40, 0.31, 0.38, 0.23),
        (1.55, 1.00, 1.30, 1.25, 1.40),
        160,
        136,
        0.28,
        0.11,
        0.34,
        "weather",
        "supply",
    ),
)


DEVELOPMENT_FAMILIES = (
    ScenarioFamily(
        "v3_dev_river_flood",
        "River flood corridor",
        (0.24, 0.39, 0.29, 0.36, 0.25),
        (1.50, 1.05, 1.35, 1.30, 1.35),
        158,
        134,
        0.27,
        0.10,
        0.34,
        "weather",
        "aftershock",
    ),
    ScenarioFamily(
        "v3_dev_industrial_outage",
        "Industrial utility outage",
        (0.38, 0.34, 0.37, 0.27, 0.18),
        (1.10, 1.20, 1.25, 1.55, 1.70),
        171,
        130,
        0.24,
        0.10,
        0.32,
        "utility",
        "supply",
    ),
    ScenarioFamily(
        "v3_dev_logistics_strike",
        "Logistics strike",
        (0.42, 0.36, 0.16, 0.38, 0.35),
        (1.00, 1.10, 1.85, 1.45, 0.95),
        145,
        120,
        0.23,
        0.09,
        0.30,
        "supply",
        "weather",
    ),
    ScenarioFamily(
        "v3_dev_seismic_cluster",
        "Seismic cluster",
        (0.17, 0.30, 0.44, 0.34, 0.39),
        (1.80, 1.35, 0.95, 1.35, 1.15),
        139,
        124,
        0.29,
        0.12,
        0.35,
        "aftershock",
        "utility",
    ),
    ScenarioFamily(
        "v3_dev_health_compound",
        "Health compound event",
        (0.31, 0.39, 0.30, 0.20, 0.26),
        (1.00, 1.05, 1.30, 1.90, 1.45),
        196,
        150,
        0.29,
        0.12,
        0.35,
        "epidemic",
        "utility",
    ),
)


FINAL_FAMILIES = (
    ScenarioFamily(
        "v3_final_coastal_isolation",
        "Coastal isolation",
        (0.26, 0.41, 0.30, 0.37, 0.22),
        (1.55, 1.00, 1.35, 1.25, 1.40),
        157,
        132,
        0.28,
        0.11,
        0.35,
        "weather",
        "supply",
    ),
    ScenarioFamily(
        "v3_final_grid_cascade",
        "Regional grid cascade",
        (0.35, 0.30, 0.39, 0.27, 0.15),
        (1.10, 1.25, 1.20, 1.55, 1.80),
        168,
        128,
        0.26,
        0.10,
        0.34,
        "utility",
        "aftershock",
    ),
    ScenarioFamily(
        "v3_final_food_access",
        "Food access disruption",
        (0.43, 0.35, 0.14, 0.34, 0.38),
        (0.90, 1.20, 1.90, 1.45, 1.00),
        144,
        118,
        0.23,
        0.09,
        0.31,
        "supply",
        "weather",
    ),
    ScenarioFamily(
        "v3_final_aftershock_corridor",
        "Aftershock corridor",
        (0.15, 0.28, 0.45, 0.33, 0.39),
        (1.90, 1.40, 0.90, 1.35, 1.10),
        136,
        121,
        0.30,
        0.12,
        0.36,
        "aftershock",
        "utility",
    ),
    ScenarioFamily(
        "v3_final_public_health",
        "Public health compound event",
        (0.31, 0.39, 0.30, 0.20, 0.25),
        (1.00, 1.05, 1.30, 1.90, 1.45),
        198,
        148,
        0.30,
        0.12,
        0.36,
        "epidemic",
        "supply",
    ),
)


TRAINING_SEEDS = tuple(range(810000, 810032))
DEVELOPMENT_SEEDS = tuple(range(820000, 820040))
FINAL_SEEDS = tuple(range(830000, 830040))

__all__ = (
    "DEVELOPMENT_FAMILIES",
    "DEVELOPMENT_SEEDS",
    "FINAL_FAMILIES",
    "FINAL_SEEDS",
    "PUBLIC_RISK_EVENT_PULSE",
    "PUBLIC_RISK_NOISE",
    "PUBLIC_RISK_PERSISTENCE",
    "PUBLIC_RISK_TOTAL_CAP",
    "ScenarioFamily",
    "Shock",
    "TRAINING_FAMILIES",
    "TRAINING_SEEDS",
    "generate_disaster_tape",
)
