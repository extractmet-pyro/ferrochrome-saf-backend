from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AssayIn(BaseModel):
    assay_wt_pct_dry: dict[str, float] = Field(description="dry-basis wt% (must total 100 +/- 2); explicit speciation")
    moisture_wet_wt_pct: float = Field(0.0, ge=0, le=40, description="free moisture, wt% of wet (as-received) mass")
    basis: str = Field("wt_pct", description="only 'wt_pct' accepted; mass fractions are rejected")

    @field_validator("assay_wt_pct_dry")
    @classmethod
    def _nonempty(cls, v):
        if not v:
            raise ValueError("empty assay")
        return v


class Recipe(BaseModel):
    ore_blend: dict[str, float] = Field(description="ore key -> mass fraction of the ore blend (normalised)")
    COKE_MET: float = Field(172.0, ge=0, le=400, description="kg wet per t ore")
    COAL_LA: float = Field(40.0, ge=0, le=300, description="kg wet per t ore")
    COKE_LOWQ: float = Field(0.0, ge=0, le=400)
    QUARTZITE: float = Field(85.0, ge=0, le=300)
    LIMESTONE: float = Field(0.0, ge=0, le=200)
    DOLOMITE: float = Field(0.0, ge=0, le=200)
    METAL_RECYCLE: float = Field(0.0, ge=0, le=200)


class BalanceIn(BaseModel):
    recipe: Recipe
    conditions: dict[str, float] = Field(default_factory=dict, description="T_metal_K, T_slag_K, T_gas_K, T_feed_K, heat_loss_MW, ...")
    prices: dict[str, float] = Field(default_factory=dict)


class OptimiseIn(BaseModel):
    base_recipe: Recipe
    ores: list[str] = ["ORE_A_LUMPY", "ORE_B_PELLET", "ORE_C_FRIABLE"]
    reductants: list[str] = ["COKE_MET", "COAL_LA"]
    fluxes: list[str] = ["QUARTZITE", "LIMESTONE"]
    constraints: dict[str, float] = Field(default_factory=dict)
    prices: dict[str, float] = Field(default_factory=dict)
    conditions: dict[str, float] = Field(default_factory=dict)
    bounds: dict[str, list[float]] = Field(default_factory=dict, description="var -> [lo, hi]; ores as blend fraction, others kg/t ore")
