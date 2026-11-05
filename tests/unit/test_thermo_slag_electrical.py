import math

import numpy as np

from app.domain import electrical as el
from app.domain import plant_config as pc
from app.domain import slag as sg
from app.domain import species as sp
from app.domain.refractory import Wall
from app.domain.thermodynamics import FusionGroup, invert_temperature, zone_enthalpy


def test_enthalpy_inversion_roundtrip():
    n = sp.from_mass_kg({"SiO2": 300, "MgO": 250, "Al2O3": 250, "Cr2O3": 50})
    g = [FusionGroup("slag", sp.SLAG_OXIDE_MASK, 4.2e5, 1800, 1980)]
    for T in (900, 1850, 1900, 2050):
        H = zone_enthalpy(n, T, g)
        assert abs(invert_temperature(H, n, g, 1500) - T) < 1e-3


def test_basicity_definitions_distinct():
    wt = {"CaO": 2, "MgO": 25, "SiO2": 33, "Al2O3": 27, "Cr2O3": 8}
    b = sg.basicities(wt)
    assert abs(b["B1_(CaO+MgO)/SiO2"] - 27 / 33) < 1e-12
    assert abs(b["B2_(CaO+MgO)/(SiO2+Al2O3)"] - 27 / 60) < 1e-12
    assert b["B1_(CaO+MgO)/SiO2"] != b["B2_(CaO+MgO)/(SiO2+Al2O3)"]


def test_slag_extrapolation_flagged():
    p = sg.slag_properties({"SiO2": 70, "MgO": 5, "Al2O3": 20, "CaO": 5}, 1973)
    assert p["extrapolation_flags"]
    p2 = sg.slag_properties({"SiO2": 33, "MgO": 26, "Al2O3": 27, "CaO": 2, "Cr2O3": 3}, 1973)
    assert not p2["extrapolation_flags"]
    assert 0.05 < p2["viscosity_liquid_Pa_s"] < 5


def test_viscosity_decreases_with_temperature_and_solids_increase():
    wt = {"SiO2": 33, "MgO": 26, "Al2O3": 27, "CaO": 2, "Cr2O3": 3}
    assert sg.urbain_viscosity_Pa_s(wt, 1850) > sg.urbain_viscosity_Pa_s(wt, 1950)
    assert sg.einstein_roscoe(1.0, 0.2) > 1.0


def _design():
    f = pc.flat(pc.build_default_config(45))
    d = el.design_from_config(f)
    el.calibrate(d, f, 2050)
    return d, f


def test_three_phase_balanced_power():
    d, f = _design()
    R = np.full(3, el.nominal_resistance(f))
    d.x_phase = np.full(3, d.x_phase[0])
    tap = int(round((f["nominal_line_voltage_v"] - f["tap_min_v"]) / f["tap_step_v"])) + 1
    c = el.solve_circuit(d, tap, R)
    Vph = d.v_line(tap) / math.sqrt(3)
    Z = abs(complex(d.r_net + R[0], d.x_phase[0]))
    I = Vph / Z
    assert abs(c["i_phase_A"][0] - I) / I < 1e-9
    assert abs(c["p_total_W"] - 3 * I * I * (d.r_net + R[0])) / c["p_total_W"] < 1e-9
    assert c["current_unbalance_pct"] < 1e-9 and c["neutral_shift_V"] < 1e-6
    assert abs(c["p_total_W"] / 1e6 - f["nominal_power_mw"]) / f["nominal_power_mw"] < 0.05


def test_phase_unbalance_detected():
    d, f = _design()
    R = np.array([0.5, 1.0, 1.0]) * el.nominal_resistance(f)
    c = el.solve_circuit(d, 11, R)
    assert c["current_unbalance_pct"] > 5 and c["neutral_shift_V"] > 1 and c["negative_sequence_pct"] > 1


def test_wall_energy_closure_and_losses():
    w = Wall("t", "SL", [("magnesia_brick", 0.46), ("high_alumina", 0.23), ("steel", 0.03)], "cyl", r_inner=5, height=1,
             h_in=150, outer_mode="natural")
    w.init_steady(1900)
    E0 = w.stored_J()
    r = w.step(1950, 600)
    assert abs((r["q_in_W"] - r["q_out_W"]) * 600 - (w.stored_J() - E0)) < 1e-6 * abs(r["q_in_W"] * 600) + 1
    parts = r["parts_W"]
    assert parts["convection"] > 0 and parts["radiation"] > 0
    assert r["shell_T_K"] < r["hot_face_T_K"]
