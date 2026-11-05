"""Lumped-zone state container with exact enthalpy-carrying inter-zone transfers.

Zones (phase inventories):
  B1  upper burden: drying, devolatilisation, magnesite decomposition, WGS, preheating
  B2  lower burden: calcination, indirect (gas-solid) reduction, Boudouard, onset of
      solid-state carbothermic reduction
  CB  coke bed / reaction zone around submerged electrode tips: solid-state chromite
      reduction, burden melting, electrode carbon
  SL  molten slag incl. undissolved chromite spinel and entrained metal droplets; interfacial
      reduction of dissolved oxides by coke-bed carbon, chromite dissolution, carburisation
  MB  metal bath (HC FeCr), carbide formation
Gas (CO-rich) is quasi-steady: generated in CB/SL/MB, passes counter-currently through B2, B1,
then the freeboard (FB) where infiltrated air may burn combustibles.

State per zone: species moles n [mol], total enthalpy H [J]; temperature is derived by
inversion. Mass changes only through `transfer` (enthalpy carried at source state) or
`react` (constant H, element-conserving stoichiometry) or boundary `add`/`remove` calls whose
enthalpy is booked in the global ledger -> mass and energy closure are exact by construction
and are *independently* verified by the ledger (phase_inventory.py / energy.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import species as sp
from .thermodynamics import FusionGroup, h_species, invert_temperature, zone_enthalpy

L_SLAG = 4.2e5  # J/kg latent heat of fusion of slag (approx.; CALIBRATION-REQUIRED)
L_METAL = 2.7e5  # J/kg latent heat of fusion of HC FeCr (approx.; CALIBRATION-REQUIRED)


@dataclass
class Zone:
    name: str
    n: np.ndarray
    T: float
    H: float = 0.0
    fusion: dict = field(default_factory=dict)  # name -> (mask, L, t_sol, t_liq)

    def groups(self) -> list[FusionGroup]:
        return [FusionGroup(k, m, L, ts, tl) for k, (m, L, ts, tl) in self.fusion.items()]

    def set_T(self, T: float):
        self.T = T
        self.H = zone_enthalpy(self.n, T, self.groups())

    def refresh_T(self):
        self.T = invert_temperature(self.H, self.n, self.groups(), self.T)

    def set_liquidus(self, group: str, t_liq: float, width: float):
        if group in self.fusion:
            m, L, _, _ = self.fusion[group]
            self.fusion[group] = (m, L, t_liq - width, t_liq)

    def mass(self, mask: np.ndarray | None = None) -> float:
        m = self.n * sp.MW
        return float(m[mask].sum() if mask is not None else m.sum())

    def carried_enthalpy(self, dn: np.ndarray) -> float:
        H = float(dn @ h_species(self.T))
        for g in self.groups():
            H += g.latent_J_per_kg * float(np.sum(dn[g.mask] * sp.MW[g.mask])) * g.frac(self.T)
        return H


def transfer(src: Zone, dst: Zone, dn: np.ndarray) -> float:
    dn = np.minimum(np.maximum(dn, 0.0), np.maximum(src.n, 0.0))
    if not np.any(dn > 0):
        return 0.0
    H = src.carried_enthalpy(dn)
    src.n = src.n - dn
    src.H -= H
    dst.n = dst.n + dn
    dst.H += H
    return H


def react(z: Zone, nu: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Apply extents xi (mol) of reactions with stoichiometry rows nu; constant H."""
    dn = nu.T @ xi
    z.n = z.n + dn
    z.n[np.abs(z.n) < 1e-12] = 0.0
    return dn


def remove_gas(z: Zone) -> tuple[np.ndarray, float]:
    g = np.where(sp.GAS_MASK, z.n, 0.0)
    H = float(g @ h_species(z.T))
    z.n = z.n - g
    z.H -= H
    return g, H


def make_zone(name: str, n: np.ndarray, T: float, slag_liq: float = 1990.0, metal_liq: float = 1790.0) -> Zone:
    fus = {}
    if name in ("SL",):
        fus["slag"] = (sp.SLAG_OXIDE_MASK, L_SLAG, slag_liq - 180.0, slag_liq)
        fus["metal"] = (sp.METAL_MASK, L_METAL, metal_liq - 120.0, metal_liq)
    if name in ("MB", "CB"):
        fus["metal"] = (sp.METAL_MASK, L_METAL, metal_liq - 120.0, metal_liq)
    z = Zone(name, n.astype(float).copy(), T, 0.0, fus)
    z.set_T(T)
    return z
