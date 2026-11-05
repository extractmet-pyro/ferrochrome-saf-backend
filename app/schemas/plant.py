from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlantUpdate(BaseModel):
    changes: dict[str, Any] = Field(description="parameter name -> new value (units as listed in GET /config/plant)")
    reason: str = Field(min_length=3, max_length=500)


class CapacityQuery(BaseModel):
    mva: float = Field(45.0, ge=3, le=120, description="transformer rating, MVA")
    enclosure: Literal["open", "semi-closed", "closed"] = "closed"
