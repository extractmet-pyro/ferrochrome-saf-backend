"""Reaction network for HC ferrochrome SAF smelting.

Each reaction carries: stoichiometry (species coefficients), equilibrium fit dG° = A + B*T
(J per mole of reaction), zone of occurrence, temperature validity domain and provenance.
Equilibrium (feasibility) is kept strictly separate from the rate law (kinetics.py).

dG° fits: Turkdogan, Physical Chemistry of High Temperature Technology (1980) and
Kubaschewski, Alcock & Spencer, Materials Thermochemistry (1993); combinations derived by
Hess's law in this file (see comments). Spinel-formation terms and CrO are surrogates
(CALIBRATION-REQUIRED / HYPOTHESIS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import species as sp
from .thermodynamics import K_eq, dG_linear, reaction_dH


@dataclass(frozen=True)
class Reaction:
    key: str
    equation: str
    nu: dict[str, float]
    A: float
    B: float
    zone: str  # burden_upper | burden_lower | reaction | slag | metal | freeboard
    t_min: float
    t_max: float
    provenance: str = "SOURCE-VERIFIED"
    rate_only: bool = False  # no equilibrium constraint applied (bookkeeping/transfer step)
    lumped: bool = False
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def nu_vec(self) -> np.ndarray:
        v = np.zeros(sp.NS)
        for k, c in self.nu.items():
            v[sp.IDX[k]] = c
        return v

    def dG(self, T: float) -> float:
        return dG_linear(self.A, self.B, T)

    def K(self, T: float) -> float:
        return K_eq(self.A, self.B, T)

    def dH(self, T: float) -> float:
        return reaction_dH(self.nu_vec, T)

    def in_domain(self, T: float) -> bool:
        return self.t_min <= T <= self.t_max


_R = Reaction
REACTIONS: list[Reaction] = [
    _R("R01_Fe2O3_C", "Fe2O3 + C -> 2 FeO + CO", {"Fe2O3": -1, "C": -1, "FeO": 2, "CO": 1}, 172_600, -207.6,
       "reaction", 900, 2300, notes="illustrative overall iron-oxide step (Hess: Fe2O3/FeO + CO formation)"),
    _R("R02_FeO_C", "FeO + C -> Fe + CO", {"FeO": -1, "C": -1, "Fe": 1, "CO": 1}, 149_600, -150.4,
       "reaction", 1000, 2300),
    _R("R03_Cr2O3_C", "Cr2O3 + 3 C -> 2 Cr + 3 CO", {"Cr2O3": -1, "C": -3, "Cr": 2, "CO": 3}, 766_940, -504.63,
       "reaction", 1400, 2300, lumped=True, notes="LUMPED overall net; not a mechanism"),
    _R("R04_Cr2O3_CrO", "Cr2O3 + C -> 2 CrO + CO", {"Cr2O3": -1, "C": -1, "CrO": 2, "CO": 1}, 327_740, -205.49,
       "reaction", 1500, 2300, "HYPOTHESIS", notes="Cr(III)->Cr(II) in slag; CrO formation data estimated"),
    _R("R05_CrO_C", "CrO + C -> Cr + CO", {"CrO": -1, "C": -1, "Cr": 1, "CO": 1}, 219_600, -149.57,
       "reaction", 1500, 2300, "HYPOTHESIS"),
    _R("R06_SiO2_C", "SiO2 + 2 C -> Si + 2 CO", {"SiO2": -1, "C": -2, "Si": 1, "CO": 2}, 678_300, -346.5,
       "reaction", 1600, 2400),
    _R("R07_MnO_C", "MnO + C -> Mn + CO", {"MnO": -1, "C": -1, "Mn": 1, "CO": 1}, 274_500, -162.07,
       "reaction", 1300, 2300),
    _R("R08_P2O5_C", "P2O5 + 5 C -> 2 P + 5 CO", {"P2O5": -1, "C": -5, "P": 2, "CO": 5}, 920_000, -908.85,
       "reaction", 1200, 2300, "CALIBRATION-REQUIRED"),
    _R("R09_Boudouard", "C + CO2 -> 2 CO", {"C": -1, "CO2": -1, "CO": 2}, 170_700, -174.5,
       "burden_lower", 800, 2300),
    _R("R10_WGS", "CO + H2O -> CO2 + H2", {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}, -36_000, 32.0,
       "burden_upper", 500, 1500, notes="water-gas shift, approached in upper burden"),
    _R("R11_calcite", "CaCO3 -> CaO + CO2", {"CaCO3": -1, "CaO": 1, "CO2": 1}, 161_300, -137.2,
       "burden_lower", 900, 1600),
    _R("R12_magnesite", "MgCO3 -> MgO + CO2", {"MgCO3": -1, "MgO": 1, "CO2": 1}, 117_600, -170.0,
       "burden_upper", 600, 1400, "CALIBRATION-REQUIRED"),
    _R("R13_drying", "H2O(l) -> H2O(g)", {"H2O_l": -1, "H2O": 1}, 41_000, -110.0,
       "burden_upper", 290, 700, rate_only=True, notes="heat-limited drying"),
    _R("R14_devol", "VOL -> CH4(g)", {"VOL": -1, "CH4": 1}, 0.0, -1.0, "burden_upper", 600, 1300,
       "HYPOTHESIS", rate_only=True, notes="volatile release as CH4-equivalent"),
    _R("R15_carbide", "7 Cr + 3 C[m] -> Cr7C3", {"Cr": -7, "C_m": -3, "Cr7C3": 1}, -228_000, -17.1,
       "metal", 1300, 2200, "CALIBRATION-REQUIRED", notes="M7C3 surrogate for carbide-bound carbon"),
    _R("R16_C_dissolution", "C -> C[m]", {"C": -1, "C_m": 1}, 0.0, -1.0, "reaction", 1400, 2200,
       rate_only=True, notes="Nernst-Brunner carburization toward C saturation"),
    _R("R17_FeCr2O4_C", "FeCr2O4 + 4 C -> Fe + 2 Cr + 4 CO", {"FeCr2O4": -1, "C": -4, "Fe": 1, "Cr": 2, "CO": 4},
       968_540, -667.03, "reaction", 1300, 2300, "CALIBRATION-REQUIRED",
       notes="solid-state carbothermic reduction of chromite spinel (Fe end-member)"),
    _R("R18_MgCr2O4_C", "MgCr2O4 + 3 C -> MgO + 2 Cr + 3 CO", {"MgCr2O4": -1, "C": -3, "MgO": 1, "Cr": 2, "CO": 3},
       808_940, -510.63, "reaction", 1400, 2300, "CALIBRATION-REQUIRED"),
    _R("R19_dis_FeCr2O4", "FeCr2O4 -> FeO + Cr2O3 (into slag)", {"FeCr2O4": -1, "FeO": 1, "Cr2O3": 1},
       52_000, -12.0, "slag", 1700, 2300, "CALIBRATION-REQUIRED", rate_only=True,
       notes="chromite dissolution, mass-transfer controlled toward Cr2O3 solubility"),
    _R("R20_dis_MgCr2O4", "MgCr2O4 -> MgO + Cr2O3 (into slag)", {"MgCr2O4": -1, "MgO": 1, "Cr2O3": 1},
       42_000, -6.0, "slag", 1700, 2300, "CALIBRATION-REQUIRED", rate_only=True),
    _R("R21_dis_MgAl2O4", "MgAl2O4 -> MgO + Al2O3 (into slag)", {"MgAl2O4": -1, "MgO": 1, "Al2O3": 1},
       35_600, 2.1, "slag", 1700, 2300, "CALIBRATION-REQUIRED", rate_only=True),
    _R("R22_Fe2O3_CO", "Fe2O3 + CO -> 2 FeO + CO2", {"Fe2O3": -1, "CO": -1, "FeO": 2, "CO2": 1}, 1_900, -33.1,
       "burden_lower", 700, 1500, notes="indirect gas-solid reduction (Hess: R01 - R09)"),
    _R("R23_FeO_CO", "FeO + CO -> Fe + CO2", {"FeO": -1, "CO": -1, "Fe": 1, "CO2": 1}, -21_100, 24.1,
       "burden_lower", 800, 1400, notes="limited by CO/CO2 equilibrium"),
    _R("R24_desulf", "CaO + S + C -> CaS + CO", {"CaO": -1, "S": -1, "C": -1, "CaS": 1, "CO": 1}, -50_000, -50.0,
       "reaction", 1600, 2300, "CALIBRATION-REQUIRED", notes="represents sulfur capacity of slag"),
    _R("R25_CO_comb", "2 CO + O2 -> 2 CO2", {"CO": -2, "O2": -1, "CO2": 2}, -564_800, 173.6, "freeboard", 300, 2500),
    _R("R26_H2_comb", "2 H2 + O2 -> 2 H2O", {"H2": -2, "O2": -1, "H2O": 2}, -492_900, 109.6, "freeboard", 300, 2500),
    _R("R27_CH4_comb", "CH4 + 2 O2 -> CO2 + 2 H2O", {"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}, -802_000, 0.0,
       "freeboard", 300, 2500),
    _R("R29_silicothermic", "3 Si + 2 Cr2O3 -> 3 SiO2 + 4 Cr", {"Si": -3, "Cr2O3": -2, "SiO2": 3, "Cr": 4},
       -501_020, 30.36, "slag", 1500, 2300, notes="slag-metal exchange on droplets (Hess from formation fits)"),
    _R("R30_CH4_crack", "CH4 -> C + 2 H2", {"CH4": -1, "C": 1, "H2": 2}, 91_040, -110.7, "burden_lower", 900, 2300,
       "SOURCE-VERIFIED", notes="thermal cracking of volatiles released in the hot lower burden"),
    _R("R31_C_H2O", "C + H2O -> CO + H2", {"C": -1, "H2O": -1, "CO": 1, "H2": 1}, 134_700, -142.3, "burden_lower", 900, 2300,
       notes="steam-carbon (water-gas) reaction"),
    _R("R28_electrode", "C(electrode) -> C (reactive)", {"C_el": -1, "C": 1}, 0.0, -1.0, "reaction", 1500, 3500,
       rate_only=True, notes="electrode tip consumption feeds carbon to the reaction zone"),
]

RXN: dict[str, Reaction] = {r.key: r for r in REACTIONS}
NU = np.array([r.nu_vec for r in REACTIONS])  # (n_rxn, n_species)


def element_imbalance(r: Reaction) -> np.ndarray:
    """Atom imbalance per element (should be exactly zero)."""
    return sp.E_MATRIX @ r.nu_vec


def check_all_balanced(tol: float = 1e-12) -> dict[str, float]:
    return {r.key: float(np.max(np.abs(element_imbalance(r)))) for r in REACTIONS
            if np.max(np.abs(element_imbalance(r))) > tol}


def table(T: float = 1973.15) -> list[dict]:
    out = []
    for r in REACTIONS:
        out.append(dict(key=r.key, equation=r.equation, zone=r.zone, dG_J=r.dG(T), K=r.K(T), dH_J=r.dH(T),
                        T_eq_K=(-r.A / r.B if r.B not in (0.0,) and not r.rate_only and -r.A / r.B > 0 else None),
                        domain=[r.t_min, r.t_max], provenance=r.provenance, lumped=r.lumped,
                        rate_only=r.rate_only, notes=r.notes,
                        max_atom_imbalance=float(np.max(np.abs(element_imbalance(r))))))
    return out


def ellingham_series(t_lo: float = 900, t_hi: float = 2300, n: int = 29) -> dict:
    T = np.linspace(t_lo, t_hi, n)
    keys = ["R02_FeO_C", "R03_Cr2O3_C", "R05_CrO_C", "R06_SiO2_C", "R07_MnO_C", "R09_Boudouard",
            "R17_FeCr2O4_C", "R18_MgCr2O4_C", "R11_calcite"]
    return {"T_K": T.tolist(), "series": {k: [RXN[k].dG(t) / 1000.0 for t in T] for k in keys}, "units": "kJ/mol rxn"}
