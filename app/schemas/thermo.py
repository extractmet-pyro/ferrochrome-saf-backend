from pydantic import BaseModel, Field


class SlagIn(BaseModel):
    wt_pct: dict[str, float] = Field(description="slag oxides wt% (liquid)")
    T_K: float = Field(1973.15, ge=1300, le=2500)
    solid_fraction: float = Field(0.0, ge=0, le=0.7)
