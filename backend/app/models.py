"""Validated public request models for the city-recovery runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ServiceName = Literal["transport", "housing", "food", "healthcare", "public_services"]
ShockName = Literal["aftershock", "supply", "epidemic", "utility", "weather"]


class StrictModel(BaseModel):
    """Reject unknown fields so operator mistakes fail at the API boundary."""

    model_config = ConfigDict(extra="forbid")


class ForcedShock(StrictModel):
    """One operator-specified shock in the public 30-day scenario."""

    day: int = Field(ge=1, le=30)
    type: ShockName
    severity: float = Field(ge=0.05, le=0.40)


class Scenario(StrictModel):
    """The single 30-day scientific scenario accepted by the runtime."""

    name: str = Field(default="Operator scenario", min_length=1, max_length=64)
    horizon_days: Literal[30] = 30
    daily_budget: float = Field(default=180.0, ge=50.0, le=500.0)
    initial_services: list[float] = Field(
        default_factory=lambda: [0.34, 0.26, 0.41, 0.38, 0.30],
        min_length=5,
        max_length=5,
    )
    priorities: list[float] = Field(
        default_factory=lambda: [1.0, 1.1, 1.2, 1.4, 1.0],
        min_length=5,
        max_length=5,
    )
    shock_probability: float = Field(default=0.20, ge=0.0, le=0.35)
    severity_min: float = Field(default=0.10, ge=0.05, le=0.25)
    severity_max: float = Field(default=0.28, ge=0.10, le=0.40)
    forced_shock: ForcedShock | None = Field(
        default=ForcedShock(day=5, type="utility", severity=0.26)
    )
    forced_shocks: list[ForcedShock] = Field(default_factory=list)
    daily_crew_pool: float = Field(default=150.0, ge=50.0, le=300.0)
    recovery_targets: list[float] = Field(
        default_factory=lambda: [0.55] * 5,
        min_length=5,
        max_length=5,
    )
    assessment_tail_days: Literal[3] = 3

    @model_validator(mode="after")
    def validate_scenario(self) -> "Scenario":
        """Enforce coupled bounds that JSON Schema cannot express alone."""

        if any(not 0.05 <= value <= 0.95 for value in self.initial_services):
            raise ValueError("each initial service must be between 0.05 and 0.95")
        if any(not 0.5 <= value <= 2.0 for value in self.priorities):
            raise ValueError("each priority must be between 0.5 and 2.0")
        if self.severity_min >= self.severity_max:
            raise ValueError("severity_min must be less than severity_max")
        if any(not 0.45 <= value <= 0.75 for value in self.recovery_targets):
            raise ValueError("each recovery target must be between 0.45 and 0.75")

        forced = ([self.forced_shock] if self.forced_shock else []) + self.forced_shocks
        if any(shock.day > self.horizon_days for shock in forced):
            raise ValueError("each forced shock day must be within horizon_days")
        tail_start = self.horizon_days - self.assessment_tail_days + 1
        if any(shock.day >= tail_start for shock in forced):
            raise ValueError("forced shocks cannot occur in the assessment tail")
        return self


class CompareRequest(StrictModel):
    """A deterministic scenario comparison request."""

    seed: int = Field(default=424242, ge=0, le=4_294_967_295)
    scenario: Scenario = Field(default_factory=Scenario)


# Temporary source-compatibility aliases for legacy modules awaiting deletion.
# New code must import the neutral names above.
ScenarioV3 = Scenario
CompareRequestV3 = CompareRequest


__all__ = (
    "CompareRequest",
    "CompareRequestV3",
    "ForcedShock",
    "Scenario",
    "ScenarioV3",
    "ServiceName",
    "ShockName",
)
