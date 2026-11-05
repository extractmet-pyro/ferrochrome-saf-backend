"""Off-gas, freeboard combustion, dust, pressure drop and analyzer models.

Normal reference conditions: 273.15 K, 101.325 kPa (Nm3 = 0.0224140 m3/mol ideal gas).
Dry basis excludes H2O. Wet and dry bases are never mixed: every reported composition carries
its basis label. CO is a life-safety hazard: the twin calculates advisories only; it never
disables gas detection, ventilation or interlocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import species as sp
from .thermodynamics import h_species

NM3_PER_MOL = 0.022414
GAS_KEYS = ["CO", "CO2", "H2", "H2O", "N2", "O2", "CH4"]
LHV_J_PER_MOL = {"CO": 282_980.0, "H2": 241_820.0, "CH4": 802_300.0}
AIR_O2 = 0.2095


def gas_vector(n: np.ndarray) -> dict[str, float]:
    return {k: float(n[sp.IDX[k]]) for k in GAS_KEYS}


def composition(n: np.ndarray, dry: bool) -> dict[str, float]:
    g = gas_vector(n)
    if dry:
        g.pop("H2O")
    tot = sum(g.values())
    return {k: 100 * v / tot for k, v in g.items()} if tot > 0 else {k: 0.0 for k in g}


def flows_nm3_per_h(n_rate_mol_s: np.ndarray) -> dict[str, float]:
    g = gas_vector(n_rate_mol_s)
    wet = sum(g.values()) * NM3_PER_MOL * 3600
    dry = (sum(g.values()) - g["H2O"]) * NM3_PER_MOL * 3600
    return {"wet_nm3_per_h": wet, "dry_nm3_per_h": dry}


def chemical_energy_W(n_rate_mol_s: np.ndarray) -> float:
    g = gas_vector(n_rate_mol_s)
    return sum(LHV_J_PER_MOL[k] * g[k] for k in LHV_J_PER_MOL)


def freeboard(n_gas_rate: np.ndarray, H_gas_rate: float, n_dust_rate: np.ndarray, H_dust_rate: float,
              air_nm3_per_nm3: float, T_amb: float, G_roof_W_K: float, T_roof_face: float, q_from_burden_W: float,
              combustion_eff: float, T_guess: float) -> dict:
    """Mix furnace gas + dust + infiltrated air, burn combustibles, return exit state (rates per s)."""
    n = n_gas_rate.copy()
    n_air = air_nm3_per_nm3 * float(n[sp.GAS_MASK].sum())
    n[sp.IDX["O2"]] += AIR_O2 * n_air
    n[sp.IDX["N2"]] += (1 - AIR_O2) * n_air
    H = H_gas_rate + H_dust_rate + n_air * (AIR_O2 * h_species(T_amb)[sp.IDX["O2"]] + (1 - AIR_O2) * h_species(T_amb)[sp.IDX["N2"]])
    H_air = n_air * (AIR_O2 * h_species(T_amb)[sp.IDX["O2"]] + (1 - AIR_O2) * h_species(T_amb)[sp.IDX["N2"]])
    burned = {}
    O2 = n[sp.IDX["O2"]]
    for key, o2_per, fuel in (("R25_CO_comb", 0.5, "CO"), ("R26_H2_comb", 0.5, "H2"), ("R27_CH4_comb", 2.0, "CH4")):
        avail = n[sp.IDX[fuel]] * combustion_eff
        x = min(avail, O2 / o2_per) if o2_per > 0 else 0.0
        x = max(x, 0.0)
        n[sp.IDX[fuel]] -= x
        n[sp.IDX["O2"]] -= o2_per * x
        if fuel == "CO":
            n[sp.IDX["CO2"]] += x
        elif fuel == "H2":
            n[sp.IDX["H2O"]] += x
        else:
            n[sp.IDX["CO2"]] += x
            n[sp.IDX["H2O"]] += 2 * x
        O2 = n[sp.IDX["O2"]]
        burned[fuel] = x
    H_avail = H + q_from_burden_W
    tot = n + n_dust_rate
    # implicit roof coupling: H_mix(T) + G (T - T_face) = H_avail  (monotone in T -> bisection)
    hs = h_species
    lo, hi = 260.0, 3500.0
    f = lambda T: float(tot @ hs(T)) + G_roof_W_K * (T - T_roof_face) - H_avail
    if f(lo) > 0:
        T = lo
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        T = 0.5 * (lo + hi)
    q_roof = G_roof_W_K * (T - T_roof_face)
    H_exit = H_avail - q_roof
    return dict(n_rate=n, T=T, H_rate=H_exit, q_roof_W=q_roof, n_air_rate=n_air, H_air_rate=H_air, burned_mol_s=burned)


def ergun_dp_pa(u: float, d_p: float, voidage: float, height: float, mu: float = 4.5e-5, rho: float = 0.25) -> float:
    e = min(max(voidage, 0.15), 0.6)
    d = max(d_p, 1e-4)
    return height * (150 * mu * u * (1 - e) ** 2 / (e ** 3 * d ** 2) + 1.75 * rho * u * u * (1 - e) / (e ** 3 * d))


def voidage_from_fines(fines_pct: float, base: float = 0.42) -> float:
    return max(0.18, base - 0.006 * fines_pct)


@dataclass
class Analyzer:
    tau_s: float = 90.0
    bias: dict = field(default_factory=dict)
    stuck: bool = False
    value: dict = field(default_factory=dict)

    def update(self, true_dry: dict, dt: float) -> dict:
        if not self.value:
            self.value = dict(true_dry)
        if self.stuck:
            return dict(self.value)
        a = 1 - math.exp(-dt / max(self.tau_s, 1e-3))
        for k, v in true_dry.items():
            self.value[k] = self.value.get(k, v) + a * (v - self.value.get(k, v))
        return {k: v + self.bias.get(k, 0.0) for k, v in self.value.items()}
