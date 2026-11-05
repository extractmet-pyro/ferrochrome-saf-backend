"""Enthalpy bookkeeping, temperature inversion and reaction thermochemistry.

Energy bookkeeping basis (no double counting):
  Every zone carries its *total* enthalpy H [J] referenced to the elements at 298.15 K:
      H_zone(T, n) = sum_i n_i [dHf_i + int_298^T Cp_i dT] + sum_g L_g * m_g * f_g(T)
  where g are fusion groups (molten slag oxides, molten metal) with latent heat L_g [J/kg]
  and liquid fraction f_g(T) ramping linearly between solidus and liquidus.
  Reaction heats are therefore *implicit* in the formation enthalpies: a reaction changes n at
  constant H and the temperature follows by inversion. Reaction heat terms are reported only
  as diagnostics (never added a second time).

Equilibrium basis (feasibility, separate from rates):
  dG°(T) = A + B*T  [J per mole of reaction as written] from critically-reviewed fits
  (Turkdogan 1980; Kubaschewski et al. 1993). K = exp(-dG°/RT).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import species as sp

R_GAS = 8.314462618
T_REF = 298.15


def h_species(T: float) -> np.ndarray:
    """Molar enthalpy (J/mol) of all species at T, incl. formation enthalpy."""
    a, b, c = sp.CP_A, sp.CP_B, sp.CP_C
    return sp.DHF + a * (T - T_REF) + 0.5 * b * (T * T - T_REF * T_REF) - c * (1.0 / T - 1.0 / T_REF)


def cp_species(T: float) -> np.ndarray:
    return sp.CP_A + sp.CP_B * T + sp.CP_C / (T * T)


@dataclass
class FusionGroup:
    name: str
    mask: np.ndarray  # species included in the group
    latent_J_per_kg: float
    t_sol: float
    t_liq: float

    def frac(self, T: float) -> float:
        if T <= self.t_sol:
            return 0.0
        if T >= self.t_liq:
            return 1.0
        return (T - self.t_sol) / max(self.t_liq - self.t_sol, 1.0)

    def dfrac(self, T: float) -> float:
        if self.t_sol < T < self.t_liq:
            return 1.0 / max(self.t_liq - self.t_sol, 1.0)
        return 0.0

    def mass(self, n: np.ndarray) -> float:
        return float(np.sum(n[self.mask] * sp.MW[self.mask]))


def zone_enthalpy(n: np.ndarray, T: float, groups: list[FusionGroup] | None = None) -> float:
    H = float(n @ h_species(T))
    for g in groups or []:
        H += g.latent_J_per_kg * g.mass(n) * g.frac(T)
    return H


def zone_heat_capacity(n: np.ndarray, T: float, groups: list[FusionGroup] | None = None) -> float:
    C = float(n @ cp_species(T))
    for g in groups or []:
        C += g.latent_J_per_kg * g.mass(n) * g.dfrac(T)
    return C


class TemperatureInversionError(RuntimeError):
    pass


def invert_temperature(H: float, n: np.ndarray, groups: list[FusionGroup] | None, T_guess: float,
                       T_lo: float = 250.0, T_hi: float = 4000.0, tol: float = 1e-6) -> float:
    """Solve zone_enthalpy(n, T) = H for T. Safeguarded Newton with bisection fallback."""
    if not np.isfinite(H):
        raise TemperatureInversionError("non-finite enthalpy")
    f_lo = zone_enthalpy(n, T_lo, groups) - H
    f_hi = zone_enthalpy(n, T_hi, groups) - H
    if f_lo > 0:
        return T_lo
    if f_hi < 0:
        raise TemperatureInversionError(f"enthalpy above T_hi bound (H={H:.3e})")
    T = min(max(T_guess, T_lo + 1), T_hi - 1)
    lo, hi = T_lo, T_hi
    for _ in range(60):
        f = zone_enthalpy(n, T, groups) - H
        if abs(f) <= tol * max(1.0, abs(H)) * 1e-6 + 1e-3:
            return T
        if f > 0:
            hi = T
        else:
            lo = T
        dfdT = zone_heat_capacity(n, T, groups)
        Tn = T - f / dfdT if dfdT > 0 else 0.5 * (lo + hi)
        if not (lo < Tn < hi):
            Tn = 0.5 * (lo + hi)
        if abs(Tn - T) < 1e-7:
            return Tn
        T = Tn
    return T


# ---------------------------------------------------------------- reaction thermochemistry
def reaction_dH(nu: np.ndarray, T: float) -> float:
    """Reaction enthalpy at T from species data (J per mole of reaction)."""
    return float(nu @ h_species(T))


def dG_linear(A: float, B: float, T: float) -> float:
    return A + B * T


def K_eq(A: float, B: float, T: float) -> float:
    x = -(A + B * T) / (R_GAS * T)
    return float(np.exp(np.clip(x, -700, 700)))


# ---------------------------------------------------------------- provider adapter
class ThermoProvider(Protocol):
    name: str

    def equilibrium(self, inputs: dict) -> dict: ...


class BuiltinCorrelationProvider:
    """Open fallback: published correlations + calibrated surrogates (larger uncertainty)."""

    name = "builtin-correlations"

    def equilibrium(self, inputs: dict) -> dict:
        from .slag import slag_properties
        return {"provider": self.name, "slag": slag_properties(inputs.get("slag_wt", {}), inputs.get("T_K", 1973.0)),
                "uncertainty": "large - correlation domain limited, see ASSUMPTIONS_AND_LIMITATIONS.md"}


class ExternalProviderNotConfigured(RuntimeError):
    pass


class FactSageAdapter:
    """Adapter boundary for a licensed FactSage/ChemApp (or Thermo-Calc) installation.

    No proprietary database is bundled. A site installation implements `equilibrium` by calling
    the licensed library (e.g. via ChemApp for Python) and returning the same dictionary shape
    as BuiltinCorrelationProvider. Until configured it raises ExternalProviderNotConfigured.
    """

    name = "factsage-adapter"

    def __init__(self, library_path: str | None = None, database: str | None = None):
        self.library_path = library_path
        self.database = database

    def equilibrium(self, inputs: dict) -> dict:
        raise ExternalProviderNotConfigured(
            "FactSage/ChemApp not configured: set THERMO_PROVIDER=factsage and THERMO_LIBRARY_PATH on a licensed host")
