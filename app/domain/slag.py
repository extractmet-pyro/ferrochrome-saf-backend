"""Slag chemistry and property surrogates for ferrochrome SAF slags.

Basicity indices are kept as *distinct named definitions* (never interchanged):
  B1 = (CaO + MgO) / SiO2                    (wt%)
  B2 = (CaO + MgO) / (SiO2 + Al2O3)          (wt%)
  MgO/Al2O3 ratio is reported separately (key for spinel-saturated FeCr slag practice).

Viscosity: Urbain model (Urbain, Cambier, Deletter & Anseau, Trans. J. Br. Ceram. Soc. 80
(1981) 139; summarised in Mills, Slag Atlas 2nd ed., 1995), with Einstein-Roscoe correction
for suspended solids (undissolved chromite / spinel).
Liquidus: linear surrogate for the MgO-Al2O3-SiO2(-CaO-CrOx) system calibrated to the general
1650-1800 °C range reported for HC FeCr slags (Visser et al., Minerals 2023, 13, 809).
=> CALIBRATION-REQUIRED; extrapolation flags are raised outside the stated domain.
"""
from __future__ import annotations

import math

MW_OX = {"SiO2": 60.08, "Al2O3": 101.96, "MgO": 40.30, "CaO": 56.08, "FeO": 71.84, "Fe2O3": 159.69,
         "Cr2O3": 151.99, "CrO": 68.0, "MnO": 70.94, "TiO2": 79.87, "P2O5": 141.94, "CaS": 72.14}

LIQUIDUS_DOMAIN = {"SiO2": (22, 50), "MgO": (15, 42), "Al2O3": (10, 35), "CaO": (0, 20), "Cr2O3eq": (1, 20)}
LIQUIDUS_COEF = {"base": 1340.0, "MgO": 10.5, "Al2O3": 3.0, "SiO2": -4.0, "CaO": -6.0, "Cr2O3eq": 12.0, "FeO": -5.0}


def cr2o3_equivalent(wt: dict[str, float]) -> float:
    """Cr2O3-equivalent wt%: CrO converted to Cr2O3 basis (2 CrO + 1/2 O2 -> Cr2O3)."""
    return wt.get("Cr2O3", 0.0) + wt.get("CrO", 0.0) * (MW_OX["Cr2O3"] / (2 * MW_OX["CrO"]))


def normalise(wt: dict[str, float]) -> dict[str, float]:
    tot = sum(v for v in wt.values() if v > 0)
    return {k: 100 * v / tot for k, v in wt.items() if v > 0} if tot > 0 else {}


def basicities(wt: dict[str, float]) -> dict[str, float]:
    c, m, s, a = wt.get("CaO", 0), wt.get("MgO", 0), wt.get("SiO2", 0), wt.get("Al2O3", 0)
    return {
        "B1_(CaO+MgO)/SiO2": (c + m) / s if s > 0 else float("nan"),
        "B2_(CaO+MgO)/(SiO2+Al2O3)": (c + m) / (s + a) if (s + a) > 0 else float("nan"),
        "MgO/Al2O3": m / a if a > 0 else float("nan"),
        "CaO/SiO2": c / s if s > 0 else float("nan"),
    }


def liquidus_C(wt: dict[str, float]) -> tuple[float, list[str]]:
    w = dict(wt)
    w["Cr2O3eq"] = cr2o3_equivalent(wt)
    T = LIQUIDUS_COEF["base"]
    for k, c in LIQUIDUS_COEF.items():
        if k != "base":
            T += c * w.get(k, 0.0)
    flags = [f"{k}={w.get(k, 0):.1f} outside [{lo},{hi}] wt%" for k, (lo, hi) in LIQUIDUS_DOMAIN.items()
             if not (lo <= w.get(k, 0.0) <= hi)]
    return T, flags


def urbain_viscosity_Pa_s(wt: dict[str, float], T_K: float) -> float:
    mol = {k: v / MW_OX[k] for k, v in wt.items() if k in MW_OX and v > 0}
    tot = sum(mol.values())
    if tot <= 0:
        return float("nan")
    x = {k: v / tot for k, v in mol.items()}
    XG = x.get("SiO2", 0) + x.get("P2O5", 0)
    XM = sum(x.get(k, 0) for k in ("CaO", "MgO", "FeO", "MnO", "CrO", "CaS")) + x.get("TiO2", 0)
    XA = x.get("Al2O3", 0) + x.get("Fe2O3", 0) + x.get("Cr2O3", 0)
    alpha = XM / (XM + XA) if (XM + XA) > 0 else 0.5
    B0 = 13.8 + 39.9355 * alpha - 44.049 * alpha ** 2
    B1 = 30.481 - 117.1505 * alpha + 129.9978 * alpha ** 2
    B2 = -40.9429 + 234.0486 * alpha - 300.04 * alpha ** 2
    B3 = 60.7619 - 153.9276 * alpha + 211.1616 * alpha ** 2
    B = B0 + B1 * XG + B2 * XG ** 2 + B3 * XG ** 3
    lnA = -(0.2693 * B + 11.6725)
    eta_poise = math.exp(lnA) * T_K * math.exp(1000.0 * B / T_K)
    return 0.1 * eta_poise


def einstein_roscoe(eta_liq: float, solid_frac: float) -> float:
    f = min(max(solid_frac, 0.0), 0.70)
    return eta_liq * (1.0 - 1.35 * f) ** -2.5


DENSITY_OX = {"SiO2": 2.20, "Al2O3": 3.05, "MgO": 3.50, "CaO": 3.30, "FeO": 4.80, "Fe2O3": 4.6, "Cr2O3": 4.3,
              "CrO": 4.3, "MnO": 4.6, "TiO2": 3.8, "P2O5": 2.4, "CaS": 2.5}  # partial molten densities, g/cm3


def density_kg_m3(wt: dict[str, float], T_K: float) -> float:
    tot = sum(wt.values())
    if tot <= 0:
        return 2900.0
    inv = sum((v / tot) / DENSITY_OX.get(k, 3.0) for k, v in wt.items())
    rho_1773 = 1000.0 / inv
    return rho_1773 * (1 - 1.0e-4 * (T_K - 1773.0))


def cr2o3_solubility_wt(T_K: float, B2: float) -> float:
    """Saturation Cr2O3 in molten FeCr slag (surrogate, CALIBRATION-REQUIRED).

    Literature (e.g. MgO-Al2O3-SiO2-CrOx studies) reports a few wt% Cr2O3 solubility rising with
    temperature; reported tap-slag Cr2O3 above this is largely undissolved chromite/spinel.
    """
    TC = T_K - 273.15
    b = 1.0 if not math.isfinite(B2) else B2
    return max(1.0, 3.0 + 0.025 * (TC - 1550.0) + 2.0 * (b - 1.0))


def slag_properties(wt_raw: dict[str, float], T_K: float, solid_frac: float = 0.0) -> dict:
    wt = normalise({k: v for k, v in wt_raw.items() if k in MW_OX})
    if not wt:
        return {"valid": False}
    Tl, flags = liquidus_C(wt)
    eta = urbain_viscosity_Pa_s(wt, T_K)
    b = basicities(wt)
    return {
        "valid": True,
        "wt_pct": wt,
        "basicity": b,
        "cr2o3_equiv_wt_pct": cr2o3_equivalent(wt),
        "liquidus_C": Tl,
        "liquidus_K": Tl + 273.15,
        "superheat_K": T_K - (Tl + 273.15),
        "viscosity_liquid_Pa_s": eta,
        "viscosity_effective_Pa_s": einstein_roscoe(eta, solid_frac),
        "solid_fraction": solid_frac,
        "density_kg_m3": density_kg_m3(wt, T_K),
        "cr2o3_solubility_wt_pct": cr2o3_solubility_wt(T_K, b["B2_(CaO+MgO)/(SiO2+Al2O3)"]),
        "extrapolation_flags": flags,
        "model": "Urbain viscosity + Einstein-Roscoe; linear liquidus surrogate (CALIBRATION-REQUIRED)",
    }
