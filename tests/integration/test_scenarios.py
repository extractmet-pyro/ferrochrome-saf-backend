"""Scenario A-L checks (shortened horizons for CI; full-length runs via scripts/run_demo.py)."""
import pytest

from app.twin.events import SCENARIOS, resolve_events


def _run(sim_factory, sid, hours):
    s = sim_factory()
    s.schedule(resolve_events(SCENARIOS[sid]["events"], s.sp.tap))
    for k, v in SCENARIOS[sid].get("initial_disturb", {}).items():
        setattr(s.dist, k, v)
    s.reset_ledger()
    snaps = [s.step() for _ in range(int(hours * 3600 / s.dt))]
    return s, snaps


@pytest.mark.parametrize("sid", list("ABCDEFGHIJKL"))
def test_scenario_runs_with_closure(sim_factory, sid):
    s, snaps = _run(sim_factory, sid, 4.0)
    last = snaps[-1]
    assert last["status"] == "ok"
    assert abs(last["energy_balance_residual_mw"]) < 1e-5
    assert abs(last["mass_balance"]["_total"]["residual_pct_of_input"]) < 1e-8


def test_G_power_curtailment(sim_factory):
    s, snaps = _run(sim_factory, "G", 4.0)
    assert snaps[-1]["power_mw"] < 0.92 * snaps[0]["power_mw"]


def test_H_unbalance_alarm(sim_factory):
    s, snaps = _run(sim_factory, "H", 3.0)
    assert any(a["code"] == "PHASE_UNBALANCE" for a in snaps[-1]["alarms"])
    assert snaps[-1]["operating_state"] in ("transformer_electrode_abnormality", "tapping")


def test_I_permeability(sim_factory):
    s, snaps = _run(sim_factory, "I", 3.0)
    assert snaps[-1]["permeability_index"] < 0.8
    assert any(a["code"] == "BURDEN_PERMEABILITY" for a in snaps[-1]["alarms"])


def test_C_moisture_dust(sim_factory):
    s, snaps = _run(sim_factory, "C", 5.0)
    assert snaps[-1]["dust_kg_per_h"] > snaps[0]["dust_kg_per_h"]


def test_E_quartz_lowers_liquidus(sim_factory):
    s, snaps = _run(sim_factory, "E", 8.0)
    assert snaps[-1]["liquidus_K"]["slag"] < snaps[0]["liquidus_K"]["slag"]
