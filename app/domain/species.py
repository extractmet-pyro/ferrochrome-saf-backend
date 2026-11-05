"""Species registry: elemental formulas, molar masses, formation enthalpies and heat capacities.

All thermochemical constants are *approximate standard-state data* compiled from the
open literature (NIST-JANAF tables; Kubaschewski, Alcock & Spencer, "Materials
Thermochemistry"; Turkdogan, "Physical Chemistry of High Temperature Technology").
Heat capacities use the Maier-Kelley form  Cp = a + b*T + c/T^2  [J/(mol K)], fitted
over roughly 298-2000 K. Values are flagged per species:

* SOURCE-VERIFIED      - standard compilation value, rounded
* CALIBRATION-REQUIRED - estimated / extrapolated, must be checked against a licensed
                         database (e.g. FactSage) before plant use
* HYPOTHESIS           - surrogate species introduced for bookkeeping

Enthalpies of mixing in slag and metal are neglected at this model fidelity (stated
limitation). Fusion is handled per phase group (see thermodynamics.py), not per species.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ELEMENTS: list[str] = ["Cr", "Fe", "C", "O", "Si", "Mg", "Al", "Ca", "Mn", "Ti", "S", "P", "H", "N"]
ATOMIC_MASS: dict[str, float] = {
    "Cr": 51.9961, "Fe": 55.845, "C": 12.011, "O": 15.999, "Si": 28.085, "Mg": 24.305,
    "Al": 26.982, "Ca": 40.078, "Mn": 54.938, "Ti": 47.867, "S": 32.06, "P": 30.974,
    "H": 1.008, "N": 14.007,
}


@dataclass(frozen=True)
class Species:
    key: str
    name: str
    formula: dict[str, float]
    dHf298: float  # J/mol
    cp: tuple[float, float, float]  # a, b, c  (Cp = a + bT + c/T^2) J/mol/K
    kind: str  # oxide | spinel | metal | carbon | gas | flux | water | sulfide | carbide | volatile
    provenance: str = "SOURCE-VERIFIED"
    rho: float = 3000.0  # kg/m3 condensed density (approx.)
    notes: str = ""

    @property
    def molar_mass(self) -> float:  # kg/mol
        return sum(ATOMIC_MASS[e] * n for e, n in self.formula.items()) / 1000.0


_S = Species
SPECIES_LIST: list[Species] = [
    # --- oxides (slag / ore / ash) ---
    _S("Cr2O3", "chromium(III) oxide", {"Cr": 2, "O": 3}, -1_139_700, (119.37, 9.30e-3, -15.65e5), "oxide", rho=5220),
    _S("CrO", "chromium(II) oxide (slag)", {"Cr": 1, "O": 1}, -340_000, (52.0, 6.0e-3, -4.0e5), "oxide",
       "HYPOTHESIS", rho=4800, notes="CrO is not stable as a pure solid; used as a slag-component surrogate"),
    _S("FeO", "wustite", {"Fe": 1, "O": 1}, -272_000, (50.80, 8.61e-3, -3.31e5), "oxide", rho=5700),
    _S("Fe2O3", "hematite", {"Fe": 2, "O": 3}, -824_200, (98.28, 77.82e-3, -14.85e5), "oxide", rho=5240,
       notes="Cp fit above 950 K is extrapolated"),
    _S("SiO2", "silica", {"Si": 1, "O": 2}, -910_700, (53.50, 16.0e-3, -13.0e5), "oxide", rho=2650),
    _S("Al2O3", "alumina", {"Al": 2, "O": 3}, -1_675_700, (114.80, 12.80e-3, -35.44e5), "oxide", rho=3950),
    _S("MgO", "periclase", {"Mg": 1, "O": 1}, -601_600, (48.98, 3.14e-3, -11.44e5), "oxide", rho=3580),
    _S("CaO", "lime", {"Ca": 1, "O": 1}, -635_100, (49.62, 4.52e-3, -6.95e5), "oxide", rho=3340),
    _S("MnO", "manganosite", {"Mn": 1, "O": 1}, -385_200, (46.48, 8.12e-3, -3.68e5), "oxide", rho=5370),
    _S("TiO2", "rutile", {"Ti": 1, "O": 2}, -944_000, (75.19, 1.17e-3, -18.20e5), "oxide", rho=4230),
    _S("P2O5", "phosphorus pentoxide (slag basis)", {"P": 2, "O": 5}, -1_504_900, (150.0, 30.0e-3, -30.0e5), "oxide",
       "CALIBRATION-REQUIRED", rho=2390),
    # --- spinels (chromite mineralogy) ---
    _S("FeCr2O4", "ferrous chromite spinel", {"Fe": 1, "Cr": 2, "O": 4}, -1_444_700, (163.0, 22.3e-3, -31.0e5), "spinel",
       "CALIBRATION-REQUIRED", rho=4790),
    _S("MgCr2O4", "magnesiochromite spinel", {"Mg": 1, "Cr": 2, "O": 4}, -1_783_000, (167.8, 13.6e-3, -42.0e5), "spinel",
       "CALIBRATION-REQUIRED", rho=4430),
    _S("MgAl2O4", "spinel (Mg-Al)", {"Mg": 1, "Al": 2, "O": 4}, -2_299_000, (153.9, 26.8e-3, -41.3e5), "spinel",
       rho=3580),
    # --- carbonates / water / volatiles ---
    _S("CaCO3", "calcite", {"Ca": 1, "C": 1, "O": 3}, -1_207_600, (104.52, 21.92e-3, -25.94e5), "flux", rho=2710),
    _S("MgCO3", "magnesite (carbonate)", {"Mg": 1, "C": 1, "O": 3}, -1_095_800, (77.91, 57.74e-3, -17.41e5), "flux",
       rho=3000),
    _S("H2O_l", "free moisture", {"H": 2, "O": 1}, -285_830, (75.3, 0.0, 0.0), "water", rho=1000,
       notes="liquid-water Cp held constant; moisture is evaporated before ~400 K in practice"),
    _S("VOL", "coal volatile matter (CH4-equivalent)", {"C": 1, "H": 4}, -74_870, (23.64, 47.86e-3, -1.92e5),
       "volatile", "HYPOTHESIS", rho=1300, notes="volatile matter bound in reductant, released as CH4"),
    # --- carbon ---
    _S("C", "carbon (coke/coal fixed C)", {"C": 1}, 0.0, (17.15, 4.27e-3, -8.79e5), "carbon", rho=1900),
    _S("C_el", "electrode carbon", {"C": 1}, 0.0, (17.15, 4.27e-3, -8.79e5), "carbon", rho=1550,
       notes="Soderberg electrode carbon tracked separately for the carbon balance"),
    _S("C_m", "carbon dissolved in metal", {"C": 1}, 22_600, (20.0, 4.0e-3, 0.0), "metal", "CALIBRATION-REQUIRED",
       rho=7000, notes="partial molar enthalpy of solution of C in Fe"),
    # --- metals / carbide ---
    _S("Fe", "iron", {"Fe": 1}, 0.0, (24.50, 8.0e-3, 0.0), "metal", rho=7000),
    _S("Cr", "chromium", {"Cr": 1}, 0.0, (24.43, 9.87e-3, -3.68e5), "metal", rho=6300),
    _S("Si", "silicon (dissolved)", {"Si": 1}, 0.0, (22.82, 3.86e-3, -3.54e5), "metal", rho=2570),
    _S("Mn", "manganese", {"Mn": 1}, 0.0, (23.85, 14.14e-3, -1.57e5), "metal", rho=6430),
    _S("P", "phosphorus (dissolved)", {"P": 1}, 0.0, (19.1, 15.8e-3, 0.0), "metal", "CALIBRATION-REQUIRED", rho=2000),
    _S("S", "sulfur", {"S": 1}, 0.0, (22.6, 0.0, 0.0), "metal", "CALIBRATION-REQUIRED", rho=2000),
    _S("Cr7C3", "chromium carbide (M7C3 surrogate)", {"Cr": 7, "C": 3}, -228_000, (228.9, 60.4e-3, -41.9e5), "carbide",
       "CALIBRATION-REQUIRED", rho=6920, notes="M7C3 with Fe substitution not resolved"),
    _S("CaS", "calcium sulfide (slag)", {"Ca": 1, "S": 1}, -482_400, (42.7, 15.9e-3, 0.0), "sulfide", rho=2590),
    # --- gases ---
    _S("CO", "carbon monoxide", {"C": 1, "O": 1}, -110_530, (28.41, 4.10e-3, -0.46e5), "gas"),
    _S("CO2", "carbon dioxide", {"C": 1, "O": 2}, -393_510, (44.14, 9.04e-3, -8.54e5), "gas"),
    _S("H2", "hydrogen", {"H": 2}, 0.0, (27.28, 3.26e-3, 0.50e5), "gas"),
    _S("H2O", "water vapour", {"H": 2, "O": 1}, -241_830, (30.00, 10.71e-3, 0.33e5), "gas"),
    _S("N2", "nitrogen", {"N": 2}, 0.0, (27.87, 4.27e-3, 0.0), "gas"),
    _S("O2", "oxygen", {"O": 2}, 0.0, (29.96, 4.18e-3, -1.67e5), "gas"),
    _S("CH4", "methane", {"C": 1, "H": 4}, -74_870, (23.64, 47.86e-3, -1.92e5), "gas"),
]

SPECIES: dict[str, Species] = {s.key: s for s in SPECIES_LIST}
KEYS: list[str] = [s.key for s in SPECIES_LIST]
IDX: dict[str, int] = {k: i for i, k in enumerate(KEYS)}
NS = len(KEYS)
MW = np.array([s.molar_mass for s in SPECIES_LIST])  # kg/mol
DHF = np.array([s.dHf298 for s in SPECIES_LIST])
CP_A = np.array([s.cp[0] for s in SPECIES_LIST])
CP_B = np.array([s.cp[1] for s in SPECIES_LIST])
CP_C = np.array([s.cp[2] for s in SPECIES_LIST])

# element matrix  E[e, s] = atoms of element e in species s
E_MATRIX = np.zeros((len(ELEMENTS), NS))
for _j, _s in enumerate(SPECIES_LIST):
    for _e, _n in _s.formula.items():
        E_MATRIX[ELEMENTS.index(_e), _j] = _n


def kind_mask(*kinds: str) -> np.ndarray:
    return np.array([s.kind in kinds for s in SPECIES_LIST])


GAS_MASK = kind_mask("gas")
METAL_MASK = kind_mask("metal", "carbide")
SLAG_OXIDE_MASK = kind_mask("oxide", "sulfide")
SPINEL_MASK = kind_mask("spinel")
CARBON_MASK = kind_mask("carbon")


def vec(**kw: float) -> np.ndarray:
    v = np.zeros(NS)
    for k, val in kw.items():
        v[IDX[k]] = val
    return v


def mass_kg(n_mol: np.ndarray) -> np.ndarray:
    return n_mol * MW


def element_mass_kg(n_mol: np.ndarray) -> np.ndarray:
    """Return kg of each element (ELEMENTS order) contained in a species-mole vector."""
    am = np.array([ATOMIC_MASS[e] for e in ELEMENTS]) / 1000.0
    return (E_MATRIX @ n_mol) * am


def from_mass_kg(mass: dict[str, float]) -> np.ndarray:
    v = np.zeros(NS)
    for k, m in mass.items():
        v[IDX[k]] += m / SPECIES[k].molar_mass
    return v


def wt_pct(n_mol: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    m = mass_kg(n_mol)
    if mask is not None:
        m = np.where(mask, m, 0.0)
    tot = m.sum()
    if tot <= 0:
        return {}
    return {KEYS[i]: 100.0 * m[i] / tot for i in range(NS) if m[i] > 0}


@dataclass
class SpeciesInfo:
    key: str
    name: str
    formula: dict[str, float]
    molar_mass_kg_per_mol: float
    dHf298_J_per_mol: float
    cp_coeffs: tuple[float, float, float]
    kind: str
    provenance: str
    notes: str = ""
    extra: dict = field(default_factory=dict)


def registry_table() -> list[dict]:
    return [
        dict(key=s.key, name=s.name, formula=s.formula, molar_mass_kg_per_mol=s.molar_mass,
             dHf298_J_per_mol=s.dHf298, cp=list(s.cp), kind=s.kind, provenance=s.provenance, notes=s.notes)
        for s in SPECIES_LIST
    ]
