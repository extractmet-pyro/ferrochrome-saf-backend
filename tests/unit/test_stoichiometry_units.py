import numpy as np
import pytest

from app.core.units import UnitError, actual_m3_from_nm3, convert, dry_from_wet
from app.domain import reactions as rx
from app.domain import species as sp
from app.domain.material_streams import (AssayError, allocate_mineralogy, default_materials, species_moles, validate_assay)
from app.domain.offgas import NM3_PER_MOL, composition


@pytest.mark.parametrize("r", rx.REACTIONS, ids=lambda r: r.key)
def test_every_reaction_atom_balanced(r):
    assert np.max(np.abs(rx.element_imbalance(r))) < 1e-12


def test_molar_masses():
    assert abs(sp.SPECIES["Cr2O3"].molar_mass - 0.15199) < 1e-4
    assert abs(sp.SPECIES["FeO"].molar_mass - 0.071844) < 1e-5
    assert abs(sp.SPECIES["Fe2O3"].molar_mass - 0.15969) < 1e-4


def test_oxide_to_element_conversion():
    n = sp.from_mass_kg({"Cr2O3": 100.0})
    cr = sp.element_mass_kg(n)[sp.ELEMENTS.index("Cr")]
    assert abs(cr - 100 * 2 * 51.9961 / 151.99) < 1e-3


def test_mineralogy_preserves_elements():
    m = default_materials()["ORE_A_LUMPY"]
    split = allocate_mineralogy(m.assay_dry_wt_pct)
    bulk = sp.element_mass_kg(sp.from_mass_kg({k: v for k, v in m.assay_dry_wt_pct.items() if k != "LOI"} | {"H2O_l": m.assay_dry_wt_pct["LOI"]}))
    mine = sp.element_mass_kg(sp.from_mass_kg(split))
    assert np.allclose(bulk, mine, atol=1e-9)
    assert split["FeCr2O4"] > 0 and split["MgCr2O4"] > 0  # spinel distinct from bulk oxides


def test_wet_dry_basis():
    m = default_materials()["ORE_A_LUMPY"]
    n = species_moles(m, 1000.0)
    water = n[sp.IDX["H2O_l"]] * sp.MW[sp.IDX["H2O_l"]]
    loi = 1000 * (1 - m.moisture_wet_wt_pct / 100) * m.assay_dry_wt_pct["LOI"] / 100
    assert abs(water - (1000 * m.moisture_wet_wt_pct / 100 + loi)) < 1e-6
    assert abs(float(np.sum(n * sp.MW)) - 1000.0) < 1e-6


def test_assay_validation_rejects_fractions_and_unknown():
    assert validate_assay({"Cr2O3": 0.5, "SiO2": 0.5})
    assert validate_assay({"Cr2O3": 50, "Unobtainium": 50})
    assert validate_assay({"Cr2O3": 40, "SiO2": 40})  # total 80
    assert not validate_assay({"Cr2O3": 60, "SiO2": 40})


def test_units_and_gas_basis():
    assert convert(1, "MWh", "MJ") == 3600.0
    assert convert(2, "t", "kg") == 2000.0
    with pytest.raises(UnitError):
        convert(1, "kWh", "kg")
    assert abs(actual_m3_from_nm3(1.0, 546.3, 101.325) - 2.0) < 1e-6
    d = dry_from_wet({"CO": 70, "CO2": 10, "H2O": 20})
    assert abs(d["CO"] - 87.5) < 1e-9
    with pytest.raises(UnitError):
        dry_from_wet({"CO": 80, "CO2": 20})
    n = sp.vec(CO=0.7, CO2=0.1, H2O=0.2)
    assert abs(composition(n, dry=True)["CO"] - 87.5) < 1e-9
    assert abs(NM3_PER_MOL - 0.022414) < 1e-9
