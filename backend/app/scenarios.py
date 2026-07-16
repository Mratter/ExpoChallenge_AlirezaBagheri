from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from backend.app.models import ForcedShock, Scenario, ShockName


@dataclass(frozen=True)
class AuthoredScenarioFamily:
    id: str
    label: str
    initial_center: tuple[float, float, float, float, float]
    priorities: tuple[float, float, float, float, float]
    budget_center: float
    horizon_center: int
    shock_probability: float
    severity_min: float
    severity_max: float
    forced_type: ShockName

    def build(self, seed: int) -> Scenario:
        """Create a bounded member without using Python's process-randomized hash."""
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
            np.asarray(self.priorities) + rng.uniform(-0.08, 0.08, size=5),
            0.5,
            2.0,
        )
        horizon = int(np.clip(self.horizon_center + rng.integers(-2, 3), 7, 30))
        budget = float(np.clip(self.budget_center + rng.integers(-18, 19), 50, 500))
        forced_day = int(np.clip(horizon // 2 + rng.integers(-1, 2), 1, horizon))
        forced_severity = float(
            round(rng.uniform(self.severity_min, self.severity_max), 8)
        )
        return Scenario(
            name=f"{self.label} / {seed}",
            horizon_days=horizon,
            daily_budget=budget,
            initial_services=[round(float(value), 8) for value in initial],
            priorities=[round(float(value), 8) for value in priorities],
            shock_probability=self.shock_probability,
            severity_min=self.severity_min,
            severity_max=self.severity_max,
            forced_shock=ForcedShock(
                day=forced_day,
                type=self.forced_type,
                severity=forced_severity,
            ),
        )


TRAINING_FAMILIES = (
    AuthoredScenarioFamily(
        "train_transit_cascade",
        "Transit cascade",
        (0.22, 0.48, 0.39, 0.44, 0.31),
        (1.65, 0.85, 1.10, 1.35, 1.05),
        152,
        15,
        0.20,
        0.08,
        0.25,
        "aftershock",
    ),
    AuthoredScenarioFamily(
        "train_displacement",
        "Housing displacement",
        (0.45, 0.19, 0.34, 0.37, 0.28),
        (0.95, 1.70, 1.20, 1.35, 1.15),
        176,
        18,
        0.22,
        0.09,
        0.27,
        "utility",
    ),
    AuthoredScenarioFamily(
        "train_supply_interrupt",
        "Supply interruption",
        (0.41, 0.38, 0.18, 0.46, 0.35),
        (1.05, 1.10, 1.75, 1.40, 0.90),
        138,
        14,
        0.18,
        0.07,
        0.24,
        "supply",
    ),
    AuthoredScenarioFamily(
        "train_health_surge",
        "Healthcare surge",
        (0.38, 0.43, 0.36, 0.17, 0.33),
        (0.90, 0.95, 1.15, 1.90, 1.20),
        194,
        20,
        0.24,
        0.10,
        0.30,
        "epidemic",
    ),
)


HELD_OUT_FAMILIES = (
    AuthoredScenarioFamily(
        "holdout_coastal_weather",
        "Coastal weather isolation",
        (0.27, 0.42, 0.32, 0.39, 0.24),
        (1.50, 1.00, 1.30, 1.25, 1.35),
        164,
        19,
        0.27,
        0.11,
        0.33,
        "weather",
    ),
    AuthoredScenarioFamily(
        "holdout_blackout",
        "Extended utility blackout",
        (0.36, 0.31, 0.41, 0.28, 0.16),
        (1.10, 1.25, 1.20, 1.55, 1.75),
        187,
        21,
        0.25,
        0.10,
        0.32,
        "utility",
    ),
    AuthoredScenarioFamily(
        "holdout_food_access",
        "Food access disruption",
        (0.44, 0.36, 0.15, 0.35, 0.39),
        (0.85, 1.20, 1.90, 1.45, 1.00),
        146,
        16,
        0.21,
        0.08,
        0.28,
        "supply",
    ),
    AuthoredScenarioFamily(
        "holdout_aftershock",
        "Aftershock corridor",
        (0.16, 0.29, 0.46, 0.34, 0.40),
        (1.85, 1.40, 0.90, 1.35, 1.10),
        132,
        13,
        0.29,
        0.12,
        0.35,
        "aftershock",
    ),
    AuthoredScenarioFamily(
        "holdout_public_health",
        "Public health compound event",
        (0.32, 0.40, 0.31, 0.21, 0.27),
        (1.00, 1.05, 1.30, 1.85, 1.45),
        208,
        23,
        0.30,
        0.12,
        0.36,
        "epidemic",
    ),
)


TRAINING_SEEDS = tuple(range(170100, 170108))
HELD_OUT_SEEDS = tuple(range(271700, 271708))


def scenario_family(family_id: str) -> AuthoredScenarioFamily:
    for family in (*TRAINING_FAMILIES, *HELD_OUT_FAMILIES):
        if family.id == family_id:
            return family
    raise KeyError(f"unknown authored scenario family: {family_id}")
