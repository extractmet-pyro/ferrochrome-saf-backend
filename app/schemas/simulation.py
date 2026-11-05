from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SimCreate(BaseModel):
    mva: float = Field(45.0, ge=3, le=120, description="transformer rating MVA (capacity class)")
    enclosure: Literal["open", "semi-closed", "closed"] = "closed"
    scenario: str = Field("A", pattern="^[A-L]$")
    seed: int = Field(42, ge=0, le=2 ** 31)
    overrides: dict[str, float] = Field(default_factory=dict, description="geometry overrides, e.g. hearth_inner_diameter_m")
    run: bool = Field(True, description="start real-time-accelerated ticking")
    speed_steps: int = Field(2, ge=1, le=60, description="simulation steps (60 s each) per 0.5 s tick")


class StepIn(BaseModel):
    steps: int = Field(1, ge=1, le=720, description="number of 60 s steps (max 12 h per call)")


class ActionIn(BaseModel):
    action: Literal["set_tap", "set_resistance", "set_recipe", "set_tap_interval", "disturb", "tap_now"]
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field("what-if", max_length=300)


class SpeedIn(BaseModel):
    running: bool = True
    speed_steps: int = Field(2, ge=1, le=60)


class ScenarioCompareIn(BaseModel):
    scenarios: list[str] = Field(["A", "B"], min_length=1, max_length=12)
    hours: float | None = Field(None, gt=0, le=72)


class ScenarioSave(BaseModel):
    name: str = Field(min_length=2, max_length=128)
    definition: dict[str, Any]
