"""Economics & ESG accounting (DEMO prices/factors - user configurable, never universal).

Boundary: Scope 1 = direct process carbon (reductants, electrode paste, carbonates) leaving
as CO/CO2 (CO assumed ultimately oxidised); Scope 2 = purchased electricity x grid factor;
optional Scope 3 = cradle-to-gate factors of purchased materials if configured.
Avoided emissions from furnace-gas energy use are reported as a SEPARATE memo item
(displaced fuel x its factor) and are NOT subtracted from Scope 1 (no double counting).
"""
from __future__ import annotations

import copy

import numpy as np

from .material_streams import default_materials
from .static_balance import DEFAULT_PRICES, static_balance

DEFAULT_ESG = {"grid_ef_t_per_mwh": 0.71, "displaced_fuel_ef_t_per_gj": 0.0561, "gas_utilisation_frac": 0.0,
               "scope3": {}}


def kpis(recipe: dict, prices: dict | None = None, esg: dict | None = None, cond: dict | None = None) -> dict:
    p = dict(DEFAULT_PRICES) | (prices or {})
    e = dict(DEFAULT_ESG) | (esg or {})
    c = dict(cond or {})
    c["co2_ef_grid_t_per_mwh"] = e["grid_ef_t_per_mwh"]
    m = default_materials()
    b = static_balance(recipe, m, c, p)
    gas_GJ = b["gas"]["chemical_energy_GJ_per_t_fecr"]
    avoided = gas_GJ * e["gas_utilisation_frac"] * e["displaced_fuel_ef_t_per_gj"]
    s3 = 0.0
    tm = b["metal_kg_per_t_ore"] / 1000
    for k, kg in b["masses_in_kg_per_t_ore"].items():
        s3 += kg / 1000 * e["scope3"].get(k, 0.0) / tm
    return {"cost": b["cost"], "sec_kwh_per_t": b["energy"]["sec_kwh_per_t_fecr"], "cr_recovery_pct": b["cr_recovery_pct"],
            "emissions_t_co2e_per_t": {"scope1": b["emissions"]["scope1_t_co2_per_t_fecr"], "scope2": b["emissions"]["scope2_t_co2_per_t_fecr"],
                                       "scope3_optional": s3, "total_scope1_2": b["emissions"]["scope1_t_co2_per_t_fecr"] + b["emissions"]["scope2_t_co2_per_t_fecr"],
                                       "memo_avoided_by_gas_use": avoided, "furnace_gas_chemical_energy_GJ_per_t": gas_GJ},
            "assumptions": {"prices": p, "esg": e}, "synthetic": True}


def breakeven_tariff(recipe: dict, fecr_price_inr_per_t: float = 110000.0) -> dict:
    tariffs = np.linspace(3000, 11000, 17)
    rows = []
    for tr in tariffs:
        k = kpis(recipe, {"electricity_inr_per_mwh": float(tr)})
        rows.append({"tariff_inr_per_mwh": float(tr), "cost_inr_per_t": k["cost"]["total_inr_per_t_fecr"],
                     "margin_inr_per_t": fecr_price_inr_per_t - k["cost"]["total_inr_per_t_fecr"]})
    be = None
    for a, b_ in zip(rows, rows[1:]):
        if a["margin_inr_per_t"] >= 0 > b_["margin_inr_per_t"]:
            f = a["margin_inr_per_t"] / (a["margin_inr_per_t"] - b_["margin_inr_per_t"])
            be = a["tariff_inr_per_mwh"] + f * (b_["tariff_inr_per_mwh"] - a["tariff_inr_per_mwh"])
    return {"rows": rows, "breakeven_tariff_inr_per_mwh": be, "fecr_price_inr_per_t": fecr_price_inr_per_t}


def pareto(recipe: dict) -> list[dict]:
    """Cost vs CO2 trade-off over coke rate and ore blend (pellet share)."""
    out = []
    for pel in (0.2, 0.35, 0.5, 0.65, 0.8):
        for coke in (160, 172, 185):
            r = copy.deepcopy(recipe)
            r["ore_blend"] = {"ORE_A_LUMPY": 1 - pel, "ORE_B_PELLET": pel}
            r["COKE_MET"] = coke
            k = kpis(r)
            out.append({"pellet_share": pel, "coke_kg_t": coke, "cost_inr_per_t": k["cost"]["total_inr_per_t_fecr"],
                        "co2_t_per_t": k["emissions_t_co2e_per_t"]["total_scope1_2"], "sec": k["sec_kwh_per_t"],
                        "cr_recovery": k["cr_recovery_pct"]})
    return out
