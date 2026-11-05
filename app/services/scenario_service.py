"""Scenario comparison (A-L or user-defined) run as background jobs."""
from __future__ import annotations

import numpy as np

from ..twin.events import SCENARIOS, resolve_events
from .jsonable import jsonable
from .simulation_service import base_sim


def run_scenario(sid: str, hours: float | None = None, mva: float = 45.0, enclosure: str = "closed", progress=None,
                 sample_h: float = 0.5) -> dict:
    sc = SCENARIOS[sid]
    sim = base_sim(mva, enclosure, sid)
    sim.schedule(resolve_events(sc["events"], sim.sp.tap))
    H = hours or sc["duration_h"]
    n = int(H * 3600 / sim.dt)
    every = max(int(sample_h * 3600 / sim.dt), 1)
    series = []
    states = {}
    alarms = {}
    for i in range(n):
        s = sim.step()
        states[s["operating_state"]] = states.get(s["operating_state"], 0) + 1
        for a in s["alarms"]:
            alarms[a["code"]] = alarms.get(a["code"], 0) + 1
        if i % every == 0 or i == n - 1:
            series.append(sim.history[-1])
        if progress and i % 20 == 0:
            progress(i / n)
    taps = sim.taps
    metal = sum(t["metal_t"] for t in taps)
    E = sum(t["electric_MWh_since_last_tap"] for t in taps)
    crm = sum(t["cr_in_metal_kg"] for t in taps)
    crs = sum(t["cr_in_slag_kg"] for t in taps)
    return jsonable({"scenario": sid, "title": sc["title"], "hours": H, "expected": sc["expected"], "assumptions": sc["assumptions"],
                     "summary": {"metal_t": metal, "sec_kwh_per_t": E * 1000 / metal if metal else None,
                                 "cr_recovery_pct": 100 * crm / (crm + crs) if crm + crs else None,
                                 "mean_metal_cr": float(np.mean([t["metal_wt_pct"]["Cr"] for t in taps])) if taps else None,
                                 "mean_metal_si": float(np.mean([t["metal_wt_pct"].get("Si", 0) for t in taps])) if taps else None,
                                 "mean_slag_cr2o3": float(np.mean([t["slag_cr2o3_equiv_wt_pct"] for t in taps])) if taps else None,
                                 "max_abs_energy_residual_mw": max(abs(r["energy_balance_residual_mw"] or 0) for r in series),
                                 "max_abs_cr_residual_kg_h": max(abs(r["mass_balance_cr_residual_kg_per_h"] or 0) for r in series),
                                 "operating_states_steps": states, "alarm_counts": alarms, "solver_status": sim.status},
                     "series": series, "taps": taps, "events": sim.events[-200:], "synthetic": True})


def compare(ids: list[str], hours: float | None = None, progress=None) -> dict:
    out = {}
    for i, sid in enumerate(ids):
        out[sid] = run_scenario(sid, hours, progress=(lambda f, i=i: progress((i + f) / len(ids))) if progress else None)
    return {"scenarios": out, "synthetic": True}
