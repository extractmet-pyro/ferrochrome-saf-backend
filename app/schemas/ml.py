from typing import Literal

from pydantic import BaseModel, Field


class TrainIn(BaseModel):
    target: Literal["sec_residual", "cr_rec_residual", "slag_cr2o3_residual"]
    kind: Literal["ridge", "gbr"] = "ridge"
    seed: int = Field(0, ge=0)


class ActivateIn(BaseModel):
    reason: str = Field(min_length=5, max_length=500, description="change-management justification (audited)")
