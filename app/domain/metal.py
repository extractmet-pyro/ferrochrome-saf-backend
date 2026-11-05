"""Fe-Cr-C-Si-Mn-P-S alloy chemistry surrogates (HC ferrochrome).

Carbon solubility: Neumann-Schenck-type linear correlation for Fe-C extended with Cr and Si
interaction terms; HC FeCr typically lands at 6-8.5 wt% C (ISO 5448 grade ranges).
Liquidus: linear surrogate for HC FeCr (approx. 1450-1600 °C reported range).
Both CALIBRATION-REQUIRED against site metal assays and/or a licensed database.
"""
from __future__ import annotations

ATOMIC = {"Fe": 55.845, "Cr": 51.996, "C": 12.011, "Si": 28.085, "Mn": 54.938, "P": 30.974, "S": 32.06}


def c_saturation_wt(T_K: float, cr_wt: float, si_wt: float) -> float:
    TC = T_K - 273.15
    return max(1.0, 1.34 + 2.54e-3 * TC + 0.040 * cr_wt - 0.30 * si_wt)


def liquidus_C(cr_wt: float, c_wt: float, si_wt: float) -> float:
    return 1520.0 + 3.0 * (cr_wt - 60.0) - 20.0 * (c_wt - 7.0) - 8.0 * (si_wt - 2.0)


def density_kg_m3(cr_wt: float, c_wt: float, T_K: float) -> float:
    return (7000.0 - 6.0 * (cr_wt - 50.0) - 60.0 * (c_wt - 6.0)) * (1 - 1.0e-4 * (T_K - 1873.0))


def element_wt(mass_by_element: dict[str, float]) -> dict[str, float]:
    tot = sum(mass_by_element.values())
    return {k: 100 * v / tot for k, v in mass_by_element.items()} if tot > 0 else {}


def metal_properties(el_wt: dict[str, float], T_K: float) -> dict:
    cr, c, si = el_wt.get("Cr", 0.0), el_wt.get("C", 0.0), el_wt.get("Si", 0.0)
    tl = liquidus_C(cr, c, si)
    csat = c_saturation_wt(T_K, cr, si)
    cr_fe = cr / el_wt["Fe"] if el_wt.get("Fe", 0) > 0 else float("nan")
    grade = "HC FeCr (charge chrome)" if cr < 60 else "HC FeCr"
    return {
        "wt_pct": el_wt, "liquidus_C": tl, "liquidus_K": tl + 273.15, "superheat_K": T_K - (tl + 273.15),
        "c_saturation_wt_pct": csat, "c_saturation_ratio": c / csat if csat > 0 else float("nan"),
        "density_kg_m3": density_kg_m3(cr, c, T_K), "Cr_to_Fe": cr_fe, "grade_label": grade,
        "model": "linear surrogates (CALIBRATION-REQUIRED)",
    }
