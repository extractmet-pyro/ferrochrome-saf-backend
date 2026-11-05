from typing import Literal

from pydantic import BaseModel, Field


class RecommendIn(BaseModel):
    session_id: str
    hybrid: bool = Field(False, description="add in-distribution ML residual corrections to SEC / Cr recovery")


class DecisionIn(BaseModel):
    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=3, max_length=500, description="operator reason (mandatory, audited)")


class CompareIn(BaseModel):
    scenario: str = Field("D", pattern="^[A-L]$")
    hours: float = Field(12.0, gt=1, le=48)
    hybrid: bool = True
