"""Disjoint authored scenario families for v3 training, development, and final evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from backend.app.models import ForcedShock, ScenarioV3, ShockName


@dataclass(frozen=True)
class ScenarioFamilyV3:
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

    def build(self, seed: int) -> ScenarioV3:
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
        return ScenarioV3(
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


TRAINING_FAMILIES_V3 = (
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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


DEVELOPMENT_FAMILIES_V3 = (
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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


FINAL_FAMILIES_V3 = (
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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
    ScenarioFamilyV3(
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


TRAINING_SEEDS_V3 = tuple(range(810000, 810032))
DEVELOPMENT_SEEDS_V3 = tuple(range(820000, 820008))
FINAL_SEEDS_V3 = tuple(range(830000, 830008))
