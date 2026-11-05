"""Global mass & energy ledgers (independent verification of closure).

Mass:   residual_e = sum(in_e) - sum(out_e) - (inventory_e(t) - inventory_e(t0))   per element
Energy: residual   = E_in - E_out - (H_zones(t) - H_zones(t0)) - (E_wall(t) - E_wall(t0))
E_in  = feed enthalpy (incl. formation) + electrode carbon + furnace electrical energy + infiltrated air
E_out = off-gas (sensible + chemical, i.e. total enthalpy) + dust + tapped metal + tapped slag
        + shell/roof/hearth losses to ambient and cooling water
Network (busbar/electrode column) losses are outside the furnace boundary and reported separately.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import species as sp
from .thermodynamics import h_species


class Ledger:
    def __init__(self, n_inv0: np.ndarray, H0: float, Ewall0: float):
        self.inv0 = n_inv0.copy()
        self.H0 = H0
        self.Ewall0 = Ewall0
        self.n_in: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(sp.NS))
        self.n_out: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(sp.NS))
        self.e_in: dict[str, float] = defaultdict(float)
        self.e_out: dict[str, float] = defaultdict(float)
        self.e_info: dict[str, float] = defaultdict(float)  # diagnostics (not part of balance)
        self.h298_in: dict[str, float] = defaultdict(float)  # formation-only enthalpy of booked streams
        self.h298_out: dict[str, float] = defaultdict(float)

    def book_in(self, name: str, dn: np.ndarray, H: float | None = None):
        self.n_in[name] = self.n_in[name] + dn
        if H is not None:
            self.e_in[name] += H
            self.h298_in[name] += float(dn @ h_species(298.15))

    def book_out(self, name: str, dn: np.ndarray, H: float | None = None):
        self.n_out[name] = self.n_out[name] + dn
        if H is not None:
            self.e_out[name] += H
            self.h298_out[name] += float(dn @ h_species(298.15))

    def energy_in(self, name: str, J: float):
        self.e_in[name] += J

    def energy_out(self, name: str, J: float):
        self.e_out[name] += J

    def info(self, name: str, J: float):
        self.e_info[name] += J

    def totals(self):
        nin = sum(self.n_in.values(), np.zeros(sp.NS))
        nout = sum(self.n_out.values(), np.zeros(sp.NS))
        return nin, nout

    def mass_residual(self, n_inv: np.ndarray, elapsed_s: float) -> dict:
        nin, nout = self.totals()
        el_in = sp.element_mass_kg(nin)
        el_out = sp.element_mass_kg(nout)
        dinv = sp.element_mass_kg(n_inv) - sp.element_mass_kg(self.inv0)
        res = el_in - el_out - dinv
        hrs = max(elapsed_s / 3600.0, 1e-9)
        out = {}
        for i, e in enumerate(sp.ELEMENTS):
            denom = max(el_in[i], 1e-12)
            out[e] = {"in_kg": float(el_in[i]), "out_kg": float(el_out[i]), "accum_kg": float(dinv[i]),
                      "residual_kg": float(res[i]), "residual_kg_per_h": float(res[i] / hrs),
                      "residual_pct_of_input": float(100 * res[i] / denom) if el_in[i] > 0 else 0.0}
        tot_in = float(np.sum(nin * sp.MW))
        tot_res = float(np.sum(nin * sp.MW) - np.sum(nout * sp.MW) - (np.sum(n_inv * sp.MW) - np.sum(self.inv0 * sp.MW)))
        out["_total"] = {"in_kg": tot_in, "residual_kg": tot_res, "residual_kg_per_h": tot_res / hrs,
                         "residual_pct_of_input": 100 * tot_res / max(tot_in, 1e-12),
                         "denominator": "cumulative total mass input since ledger start"}
        return out

    def energy_residual(self, H_zones: float, E_wall: float, elapsed_s: float) -> dict:
        Ein = sum(self.e_in.values())
        Eout = sum(self.e_out.values())
        dS = (H_zones - self.H0) + (E_wall - self.Ewall0)
        res = Ein - Eout - dS
        # engineering distribution of the electrical input (exact identity, no normalisation):
        # E_el = sum(sensible out) - sum(sensible in) + losses + dStorage + [H298(out) - H298(in)]  (+ residual)
        sens_out = {k: self.e_out[k] - self.h298_out.get(k, 0.0) for k in self.e_out if k in self.h298_out}
        sens_in = {k: self.e_in[k] - self.h298_in.get(k, 0.0) for k in self.e_in if k in self.h298_in}
        chem = sum(self.h298_out.values()) - sum(self.h298_in.values())
        dist = {"reaction_enthalpy_net": chem, **{f"sensible_out:{k}": v for k, v in sens_out.items()},
                **{f"sensible_in:{k}": -v for k, v in sens_in.items()},
                "wall_losses": self.e_out.get("wall_losses", 0.0), "storage_change": dS}
        return {"distribution_J": dist, "E_in_J": Ein, "E_out_J": Eout, "storage_change_J": dS, "residual_J": res,
                "residual_MW": res / max(elapsed_s, 1e-9) / 1e6,
                "residual_pct_of_electric": 100 * res / max(self.e_in.get("electric_furnace", 0.0), 1e-9),
                "in_breakdown_J": dict(self.e_in), "out_breakdown_J": dict(self.e_out),
                "diagnostics_J": dict(self.e_info),
                "basis": "total enthalpy incl. formation enthalpy at 298.15 K; no post-hoc normalisation"}
