"""Rate laws: kinetics, mass transfer, dissolution and transport (separate from equilibrium).

General form for reaction j in zone z:
    r_j = k_j(T) * n_lim * S_j * DF_j          [mol of reaction / s]
    k_j(T) = A_j exp(-E_j / R T)               (Arrhenius, physically clipped)
    S_j    = contact / surface / carbon-availability factor (dimensionless, 0..1+)
    DF_j   = max(0, 1 - Q_j / K_j)              thermodynamic driving force (affinity-limited)
Mass-transfer-controlled steps (chromite dissolution, carburisation, droplet settling) use
explicit transport laws (shrinking particle with k_m, Nernst-Brunner, Stokes).
All pre-exponential factors are CALIBRATION-REQUIRED priors chosen so that the synthetic
nominal case lands inside published HC FeCr ranges; activation energies are of the order
reported for carbothermic chromite reduction (approx. 200-400 kJ/mol; e.g. Hayes, "Process
Selection in Extractive Metallurgy"; Niayesh & Dippenaar; Ding & Warner studies) and are flagged.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import species as sp
from .reactions import RXN
from .thermodynamics import R_GAS


@dataclass
class KParam:
    A: float
    E: float  # J/mol
    provenance: str = "CALIBRATION-REQUIRED"
    note: str = ""


K_DEFAULT: dict[str, KParam] = {
    "drying": KParam(1 / 200.0, 0.0, "DEMO-ONLY", "heat-limited drying, logistic onset at ~350 K"),
    "devol": KParam(2.0e3, 90e3, note="volatile release, first order"),
    "magnesite": KParam(1.0e8, 180e3),
    "calcite": KParam(5.0e7, 190e3, note="calcination, first order with CO2 back-pressure"),
    "boudouard": KParam(5.0e6, 250e3, note="coke solution loss; scale with coke reactivity index"),
    "wgs": KParam(50.0, 60e3, note="approach-to-equilibrium factor in burden gas"),
    "ind_Fe2O3": KParam(50.0, 60e3, note="hematite->wustite by CO"),
    "ind_FeO": KParam(5.0, 80e3, note="free wustite->Fe by CO"),
    "ss_FeCr2O4": KParam(2.0e4, 300e3, note="solid-state carbothermic reduction, Fe-chromite"),
    "ss_MgCr2O4": KParam(1.0e4, 330e3, note="solid-state carbothermic reduction, Mg-chromite"),
    "ss_Cr2O3": KParam(4.0e4, 310e3),
    "dir_FeO": KParam(1.0e5, 230e3, note="direct reduction of free FeO/Fe2O3 by carbon"),
    "sl_FeO": KParam(4.0e5, 200e3, note="slag FeO reduction at coke-bed interface"),
    "sl_Cr2O3": KParam(2.5e4, 230e3, note="slag Cr2O3 reduction at coke-bed interface (overall)"),
    "sl_Cr2O3_CrO": KParam(1.5e4, 200e3),
    "sl_CrO": KParam(2.5e4, 220e3),
    "sl_SiO2": KParam(4.5e7, 470e3, note="strongly T-sensitive SiO2 reduction -> Si in alloy"),
    "sl_MnO": KParam(2.0e5, 220e3),
    "sl_P2O5": KParam(5.0e5, 200e3),
    "desulf": KParam(2.0e-3, 0.0, note="relaxation toward sulfur partition L_S"),
    "silicothermic": KParam(5.0e3, 150e3, note="3Si + 2Cr2O3 -> 3SiO2 + 4Cr on droplets"),
    "carbide": KParam(1 / 600.0, 0.0, note="relaxation to carbide-bound C fraction"),
    "carburise": KParam(1 / 150.0, 0.0, note="Nernst-Brunner droplet carburisation"),
    "melt": KParam(1 / 1500.0, 0.0, note="burden melting in coke bed above slag liquidus"),
    "km_chromite": KParam(0.8e-6, 120e3, note="mass-transfer coefficient (m/s) at 1973 K, Arrhenius-scaled"),
    "ch4_crack": KParam(1.5e8, 220e3, note="volatile cracking in hot burden"),
    "c_h2o": KParam(2.0e6, 230e3, note="steam gasification of coke (reactivity-scaled)"),
    "electrode": KParam(1 / 60.0, 0.0, "DEMO-ONLY", "electrode tip carbon joins coke bed"),
}

GAMMA = {"FeO": 2.0, "Cr2O3": 1.2, "CrO": 1.5, "SiO2": 1.0, "MnO": 1.5, "P2O5": 1e-4}
GAMMA_METAL = {"Fe": 1.0, "Cr": 1.0, "Si": 0.01, "Mn": 1.0, "P": 1e-3}


def arrh(p: KParam, T: float) -> float:
    if p.E == 0:
        return p.A
    return p.A * math.exp(-p.E / (R_GAS * max(T, 200.0)))


def arrh_ref(p: KParam, T: float, Tref: float = 1973.15) -> float:
    """Arrhenius scaling relative to a reference temperature (A is the value at Tref)."""
    return p.A * math.exp(-p.E / R_GAS * (1.0 / max(T, 200.0) - 1.0 / Tref))


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 60), -60)))


def mole_frac(n: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    tot = float(n[mask].sum())
    if tot <= 0:
        return {}
    return {sp.KEYS[i]: n[i] / tot for i in np.where(mask)[0]}


def driving_force(rxn_key: str, T: float, act_react: float, act_prod: float) -> float:
    """DF = max(0, 1 - Q/K) with Q = prod/react activities (pressures in atm)."""
    K = RXN[rxn_key].K(T)
    if act_react <= 0:
        return 0.0
    Q = act_prod / act_react
    return max(0.0, 1.0 - Q / K) if K > 0 else 0.0


def stokes_velocity(r_m: float, d_rho: float, eta: float) -> float:
    return 2.0 * 9.81 * max(d_rho, 0.0) * r_m * r_m / (9.0 * max(eta, 1e-3))


def limit_extents(n: np.ndarray, nu: np.ndarray, xi: np.ndarray, frac: float = 0.95) -> np.ndarray:
    """Scale extents (mol) so no reactant inventory goes negative (handles shared reactants)."""
    xi = np.maximum(xi, 0.0)
    for _ in range(6):
        cons = np.maximum(-nu, 0.0).T @ xi  # consumption per species
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cons > 0, frac * np.maximum(n, 0.0) / cons, np.inf)
        if np.all(ratio >= 1.0):
            return xi
        s = np.ones_like(xi)
        for j in range(len(xi)):
            idx = np.where(nu[j] < 0)[0]
            if len(idx):
                s[j] = min(1.0, float(np.min(ratio[idx])))
        xi = xi * s
    return xi


def params_table() -> list[dict]:
    return [dict(key=k, **asdict(v)) for k, v in K_DEFAULT.items()]
