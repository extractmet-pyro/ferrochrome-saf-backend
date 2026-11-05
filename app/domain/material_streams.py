"""Raw-material definitions, assay validation, mineralogy allocation and species conversion.

Assays are wt% on a DRY basis; free moisture is wt% on a WET (as-received) basis. Bases are
never mixed silently: every conversion is explicit (wet -> dry -> species moles).

Chromite mineralogy: the bulk oxide analysis is allocated to a spinel (Fe,Mg)(Cr,Al)2O4
represented by end-members FeCr2O4, MgCr2O4, MgAl2O4 plus residual (free) oxides. This keeps
elemental book-keeping (exact) distinct from mineral structure (which drives dissolution
and solid-state reduction kinetics). The allocation rule is a documented HYPOTHESIS; a site
should replace it with QEMSCAN/XRD-based mineralogy where available.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import species as sp

ASSAY_KEYS = {
    "Cr2O3", "FeO", "Fe2O3", "Al2O3", "MgO", "SiO2", "CaO", "MnO", "TiO2", "P2O5", "CaCO3", "MgCO3",
    "C", "VOL", "S", "LOI", "Fe_met", "Cr_met", "Si_met", "C_met",
}
_ASSAY_TO_SPECIES = {"LOI": "H2O_l", "Fe_met": "Fe", "Cr_met": "Cr", "Si_met": "Si", "C_met": "C_m"}


class AssayError(ValueError):
    pass


@dataclass
class RawMaterial:
    key: str
    name: str
    category: str  # ore | reductant | flux | recycle | electrode
    assay_dry_wt_pct: dict[str, float]
    moisture_wet_wt_pct: float = 0.0
    price_inr_per_t: float = 0.0
    co2_factor_t_per_t: float = 0.0  # cradle-to-gate, user-configurable (Scope 3); 0 = not set
    d50_mm: float = 20.0
    fines_pct: float = 5.0  # < 6 mm
    bulk_density_t_m3: float = 2.0
    porosity: float = 0.15
    prereduction_fe: float = 0.0
    prereduction_cr: float = 0.0
    spinel_al_fraction: float = 0.85
    feed_temperature_k: float = 303.15
    provenance: str = "DEMO-ONLY"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_assay(assay: dict[str, float], basis: str = "wt_pct", tol: float = 2.0) -> list[str]:
    errs = []
    unknown = set(assay) - ASSAY_KEYS
    if unknown:
        errs.append(f"unknown assay species {sorted(unknown)} (speciation must be explicit)")
    vals = list(assay.values())
    if any((not isinstance(v, (int, float))) or (not math.isfinite(v)) for v in vals):
        errs.append("non-finite assay value")
        return errs
    if any(v < 0 for v in vals):
        errs.append("negative assay value")
    tot = sum(vals)
    if basis != "wt_pct":
        errs.append("only wt% basis accepted; convert mass fractions explicitly before upload")
    if 0.9 <= tot <= 1.1 and max(vals) <= 1.0:
        errs.append(f"assay sums to {tot:.3f}: looks like mass fractions, not wt% (mixing bases rejected)")
    elif abs(tot - 100.0) > tol:
        errs.append(f"assay total {tot:.2f} wt% outside 100 +/- {tol} (reconcile oxides/LOI explicitly)")
    return errs


def allocate_mineralogy(assay: dict[str, float], spinel_al_fraction: float = 0.85) -> dict[str, float]:
    """Convert a bulk dry oxide assay (wt%) to a species mass split (kg per 100 kg dry)."""
    a = dict(assay)
    mol = {k: a.get(k, 0.0) / (sp.SPECIES[k].molar_mass * 1000.0)
           for k in ("Cr2O3", "FeO", "MgO", "Al2O3")}  # kmol per 100 kg
    out_mol: dict[str, float] = {}
    fecr = min(mol["FeO"], mol["Cr2O3"])
    out_mol["FeCr2O4"] = fecr
    cr_rem = mol["Cr2O3"] - fecr
    fe_rem = mol["FeO"] - fecr
    al_sp = mol["Al2O3"] * spinel_al_fraction
    mg_for_al = min(al_sp, mol["MgO"])
    mg_rem = mol["MgO"] - mg_for_al
    mgcr = min(cr_rem, mg_rem)
    out_mol["MgCr2O4"] = mgcr
    cr_rem -= mgcr
    mg_rem -= mgcr
    out_mol["MgAl2O4"] = mg_for_al
    al_rem = mol["Al2O3"] - mg_for_al
    out_mol["Cr2O3"] = cr_rem
    out_mol["FeO"] = fe_rem
    out_mol["MgO"] = mg_rem
    out_mol["Al2O3"] = al_rem
    res = {k: v * sp.SPECIES[k].molar_mass * 1000.0 for k, v in out_mol.items() if v > 0}
    for k, v in a.items():
        if k in ("Cr2O3", "FeO", "MgO", "Al2O3"):
            continue
        res[_ASSAY_TO_SPECIES.get(k, k)] = res.get(_ASSAY_TO_SPECIES.get(k, k), 0.0) + v
    return res


def species_moles(mat: RawMaterial, wet_mass_kg: float) -> np.ndarray:
    """Species mole vector for `wet_mass_kg` of material (free moisture added as H2O_l)."""
    dry = wet_mass_kg * (1.0 - mat.moisture_wet_wt_pct / 100.0)
    water = wet_mass_kg - dry
    split = allocate_mineralogy(mat.assay_dry_wt_pct, mat.spinel_al_fraction) if mat.category == "ore" else \
        {_ASSAY_TO_SPECIES.get(k, k): v for k, v in mat.assay_dry_wt_pct.items()}
    tot = sum(split.values())
    v = np.zeros(sp.NS)
    for k, pct in split.items():
        v[sp.IDX[k]] += dry * (pct / tot) / sp.SPECIES[k].molar_mass
    v[sp.IDX["H2O_l"]] += water / sp.SPECIES["H2O_l"].molar_mass
    return v


def cr_fe_ratio(assay: dict[str, float]) -> float:
    cr = assay.get("Cr2O3", 0) * 2 * 51.996 / 151.99 + assay.get("Cr_met", 0)
    fe = assay.get("FeO", 0) * 55.845 / 71.844 + assay.get("Fe2O3", 0) * 2 * 55.845 / 159.69 + assay.get("Fe_met", 0)
    return cr / fe if fe > 0 else float("nan")


def prereduce(base: RawMaterial, x_fe: float, x_cr: float, key: str | None = None) -> RawMaterial:
    """Create a pre-reduced variant: fractions x_fe, x_cr of Fe/Cr oxides converted to metal (dry basis)."""
    a = dict(base.assay_dry_wt_pct)
    fe_ox = a.get("FeO", 0) * 55.845 / 71.844 + a.get("Fe2O3", 0) * 2 * 55.845 / 159.69
    cr_ox = a.get("Cr2O3", 0) * 2 * 51.996 / 151.99
    a["FeO"] = a.get("FeO", 0) * (1 - x_fe)
    a["Fe2O3"] = a.get("Fe2O3", 0) * (1 - x_fe)
    a["Cr2O3"] = a.get("Cr2O3", 0) * (1 - x_cr)
    a["Fe_met"] = fe_ox * x_fe
    a["Cr_met"] = cr_ox * x_cr
    a.pop("LOI", None)
    tot = sum(a.values())
    a = {k: 100 * v / tot for k, v in a.items() if v > 0}
    return RawMaterial(key=key or f"{base.key}_PR", name=f"{base.name} (pre-reduced Fe {x_fe:.0%}, Cr {x_cr:.0%})",
                       category="ore", assay_dry_wt_pct=a, moisture_wet_wt_pct=0.0, price_inr_per_t=base.price_inr_per_t * 1.25,
                       d50_mm=12, fines_pct=4, bulk_density_t_m3=2.4, porosity=0.25, prereduction_fe=x_fe,
                       prereduction_cr=x_cr, spinel_al_fraction=base.spinel_al_fraction, feed_temperature_k=303.15,
                       provenance="DEMO-ONLY", notes="pre-reduced pellet variant generated by prereduce()")


def default_materials() -> dict[str, RawMaterial]:
    M = {}

    def add(m: RawMaterial):
        errs = validate_assay(m.assay_dry_wt_pct)
        if errs:
            raise AssayError(f"{m.key}: {errs}")
        M[m.key] = m

    add(RawMaterial("ORE_A_LUMPY", "Chromite lumpy ore A (synthetic, Cr/Fe~2.7)", "ore",
                    {"Cr2O3": 46.0, "Fe2O3": 6.0, "FeO": 9.5, "Al2O3": 10.5, "MgO": 11.2, "SiO2": 12.0, "CaO": 0.6,
                     "MnO": 0.3, "TiO2": 0.4, "P2O5": 0.01, "LOI": 3.49}, 4.0, 9500, 0.0, 40, 8, 2.4, 0.08))
    add(RawMaterial("ORE_B_PELLET", "Sintered chromite concentrate pellets B (synthetic)", "ore",
                    {"Cr2O3": 50.0, "Fe2O3": 4.0, "FeO": 12.5, "Al2O3": 12.99, "MgO": 11.2, "SiO2": 8.0, "CaO": 0.5,
                     "MnO": 0.3, "TiO2": 0.5, "P2O5": 0.01}, 0.5, 13500, 0.0, 12, 3, 2.3, 0.22))
    add(RawMaterial("ORE_C_FRIABLE", "Low-grade friable chromite C (synthetic, high gangue)", "ore",
                    {"Cr2O3": 38.0, "Fe2O3": 9.0, "FeO": 10.0, "Al2O3": 9.0, "MgO": 12.0, "SiO2": 16.0, "CaO": 1.0,
                     "MnO": 0.3, "TiO2": 0.3, "P2O5": 0.02, "LOI": 4.38}, 7.0, 6500, 0.0, 8, 25, 2.1, 0.12))
    add(RawMaterial("COKE_MET", "Metallurgical nut coke (synthetic)", "reductant",
                    {"C": 84.0, "VOL": 1.6, "SiO2": 6.5, "Al2O3": 3.25, "Fe2O3": 1.3, "CaO": 0.65, "MgO": 0.26,
                     "P2O5": 0.06, "S": 0.6, "TiO2": 0.08, "LOI": 1.7}, 6.0, 32000, 0.0, 15, 6, 0.5, 0.45))
    add(RawMaterial("COAL_LA", "Low-ash coal / anthracite (synthetic)", "reductant",
                    {"C": 76.0, "VOL": 10.5, "SiO2": 6.0, "Al2O3": 3.0, "Fe2O3": 1.0, "CaO": 0.6, "MgO": 0.3,
                     "P2O5": 0.03, "S": 0.5, "LOI": 2.07}, 8.0, 18000, 0.0, 20, 10, 0.75, 0.10))
    add(RawMaterial("COKE_LOWQ", "High-ash, lower fixed-C coke (synthetic)", "reductant",
                    {"C": 75.0, "VOL": 2.0, "SiO2": 11.0, "Al2O3": 6.0, "Fe2O3": 1.8, "CaO": 1.0, "MgO": 0.4,
                     "P2O5": 0.09, "S": 0.8, "TiO2": 0.11, "LOI": 1.8}, 9.0, 24000, 0.0, 10, 14, 0.5, 0.48))
    add(RawMaterial("QUARTZITE", "Quartzite flux (synthetic)", "flux",
                    {"SiO2": 97.0, "Al2O3": 1.4, "Fe2O3": 0.8, "CaO": 0.3, "MgO": 0.2, "LOI": 0.3}, 2.0, 1800, 0.0, 30, 3, 1.5, 0.05))
    add(RawMaterial("LIMESTONE", "Limestone (synthetic)", "flux",
                    {"CaCO3": 94.0, "MgCO3": 2.5, "SiO2": 2.0, "Al2O3": 0.8, "Fe2O3": 0.4, "LOI": 0.3}, 2.0, 1500, 0.0, 30, 4, 1.5, 0.05))
    add(RawMaterial("DOLOMITE", "Dolomite (synthetic)", "flux",
                    {"CaCO3": 54.0, "MgCO3": 42.0, "SiO2": 2.5, "Al2O3": 0.8, "Fe2O3": 0.7}, 2.0, 1700, 0.0, 30, 4, 1.6, 0.05))
    add(RawMaterial("METAL_RECYCLE", "Recovered FeCr metal from slag jigging (synthetic)", "recycle",
                    {"Cr_met": 52.0, "Fe_met": 36.0, "C_met": 6.8, "Si_met": 2.5, "SiO2": 1.5, "MgO": 0.7, "Al2O3": 0.5},
                    1.0, 60000, 0.0, 10, 5, 3.5, 0.05))
    pr = prereduce(M["ORE_B_PELLET"], 0.80, 0.45, key="ORE_B_PREREDUCED")
    M[pr.key] = pr
    return M


DEFAULT_RECIPE = {  # wet kg per t of total ore (blend) charged — DEMO-ONLY nominal
    "ore_blend": {"ORE_A_LUMPY": 0.55, "ORE_B_PELLET": 0.45},
    "COKE_MET": 163.0, "COAL_LA": 40.0, "QUARTZITE": 85.0, "LIMESTONE": 0.0, "DOLOMITE": 0.0, "METAL_RECYCLE": 0.0,
}


def recipe_species_per_t_ore(recipe: dict, mats: dict[str, RawMaterial]) -> tuple[np.ndarray, dict[str, float]]:
    """Species moles and wet mass of each material per 1 t (wet) of ore blend."""
    v = np.zeros(sp.NS)
    masses: dict[str, float] = {}
    blend = recipe["ore_blend"]
    s = sum(blend.values())
    if s <= 0:
        raise AssayError("empty ore blend")
    for k, f in blend.items():
        m = 1000.0 * f / s
        masses[k] = m
        v += species_moles(mats[k], m)
    for k, kg in recipe.items():
        if k == "ore_blend" or kg <= 0:
            continue
        masses[k] = kg
        v += species_moles(mats[k], kg)
    return v, masses
