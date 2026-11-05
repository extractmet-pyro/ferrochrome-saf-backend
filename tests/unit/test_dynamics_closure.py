import math

import numpy as np

from app.domain import species as sp


def test_mass_and_energy_closure(sim_factory):
    s = sim_factory()
    s.reset_ledger()
    for _ in range(240):
        snap = s.step()
    mb = snap["mass_balance"]
    for e in ("Cr", "Fe", "C", "O", "Si"):
        assert abs(mb[e]["residual_kg"]) < 1e-6 * max(mb[e]["in_kg"], 1), e
    assert abs(mb["_total"]["residual_pct_of_input"]) < 1e-9
    assert abs(snap["energy_balance"]["residual_MW"]) < 1e-6
    assert snap["status"] == "ok"


def test_nonnegative_inventories_and_finite(sim_factory):
    s = sim_factory()
    for _ in range(120):
        s.step()
        for z in s.z.values():
            assert np.all(z.n >= 0) and math.isfinite(z.T)


def test_recovery_and_sec_denominators(sim_factory):
    s = sim_factory()
    snap = None
    for _ in range(int(5 * 3600 / s.dt)):
        snap = s.step()
    tap = s.taps[-1]
    assert abs(tap["sec_kwh_per_t"] - tap["electric_MWh_since_last_tap"] * 1000 / tap["metal_t"]) < 1e-6
    cr = snap["cr_recovery_pct"]
    assert cr is not None and 0 < cr < 100


def test_reproducible_with_seed(sim_factory):
    a, b = sim_factory(), sim_factory()
    for _ in range(60):
        sa, sb = a.step(), b.step()
    assert sa["power_mw"] == sb["power_mw"] and sa["fecr_cr_wt_pct"] == sb["fecr_cr_wt_pct"]


def test_time_step_convergence(sim_factory):
    a, b = sim_factory(), sim_factory()
    b.dt, b.nsub = 30.0, 2
    for _ in range(60):
        sa = a.step()
    for _ in range(120):
        sb = b.step()
    assert abs(sa["zone_temperature_k"]["SL"] - sb["zone_temperature_k"]["SL"]) < 15
    assert abs(sa["fecr_cr_wt_pct"] - sb["fecr_cr_wt_pct"]) < 0.5


def test_solver_rejects_invalid_and_freezes(sim_factory):
    s = sim_factory()
    s.step()
    s.z["SL"].H = float("nan")
    out = s.step()
    assert s.status == "frozen_last_good" and out["status"] == "frozen_last_good"
    assert np.isfinite(s.z["SL"].T)


def test_tap_lockout(sim_factory):
    s = sim_factory()
    s.apply_action("set_tap", {"tap": s.sp.tap - 1})
    try:
        s.apply_action("set_tap", {"tap": s.sp.tap - 1})
        assert False, "second tap change inside dwell must be locked out"
    except ValueError:
        pass


def test_species_mask_consistency():
    assert not np.any(sp.GAS_MASK & sp.METAL_MASK)
