from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ServiceName = Literal["transport", "housing", "food", "healthcare", "public_services"]
ShockName = Literal["aftershock", "supply", "epidemic", "utility", "weather"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForcedShock(StrictModel):
    day: int = Field(ge=1, le=30)
    type: ShockName
    severity: float = Field(ge=0.05, le=0.40)


class Scenario(StrictModel):
    name: str = Field(default="Operator scenario", min_length=1, max_length=64)
    horizon_days: int = Field(default=14, ge=7, le=30)
    daily_budget: float = Field(default=180.0, ge=50.0, le=500.0)
    initial_services: list[float] = Field(
        default=[0.34, 0.26, 0.41, 0.38, 0.30], min_length=5, max_length=5
    )
    priorities: list[float] = Field(
        default=[1.0, 1.1, 1.2, 1.4, 1.0], min_length=5, max_length=5
    )
    shock_probability: float = Field(default=0.20, ge=0.0, le=0.35)
    severity_min: float = Field(default=0.10, ge=0.05, le=0.25)
    severity_max: float = Field(default=0.28, ge=0.10, le=0.40)
    forced_shock: ForcedShock | None = Field(
        default=ForcedShock(day=5, type="utility", severity=0.26)
    )
    forced_shocks: list[ForcedShock] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scenario(self) -> "Scenario":
        if any(not 0.05 <= value <= 0.95 for value in self.initial_services):
            raise ValueError("each initial service must be between 0.05 and 0.95")
        if any(not 0.5 <= value <= 2.0 for value in self.priorities):
            raise ValueError("each priority must be between 0.5 and 2.0")
        if self.severity_min >= self.severity_max:
            raise ValueError("severity_min must be less than severity_max")
        if self.forced_shock and self.forced_shock.day > self.horizon_days:
            raise ValueError("forced_shock.day must be within horizon_days")
        if any(shock.day > self.horizon_days for shock in self.forced_shocks):
            raise ValueError("each forced_shocks day must be within horizon_days")
        return self


class CompareRequest(StrictModel):
    seed: int = Field(default=424242, ge=0, le=4_294_967_295)
    scenario: Scenario = Field(default_factory=Scenario)
