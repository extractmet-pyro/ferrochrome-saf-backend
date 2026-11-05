"""Comparative closed-loop *simulation* of baseline policy vs physics-MPC vs hybrid-MPC under
IDENTICAL disturbance schedules (same seed, same events). MPC moves are applied to the
SIMULATED furnace only (this is an offline study, not plant control). Results are SYNTHETIC.
"""
from __future__ import annotations

import time

import numpy as np

from ..twin.events import SCENARIOS, resolve_events
from .constraints import default_mpc_config
from .mpc import SupervisoryMPC
from .policy import BaselinePolicy


def _apply(sim, rec):
    for m in rec["moves"]:
        if abs(m["delta"]) < 1e-9:
            continue
        if m["name"] == "tap":
            sim.apply_action("set_tap", {"tap": int(round(m["new"]))}, source="mpc_sim")
        elif m["name"] == "r_setpoint_mohm":
            sim.apply_action("set_resistance", {"r_mohm": m["new"]}, source="mpc_sim")
        elif m["name"] == "coke_kg_t":
            from .mpc import _coke_key
            sim.apply_action("set_recipe", {_coke_key(sim): m["new"]}, source="mpc_sim")
        elif m["name"] == "quartz_kg_t":
            sim.apply_action("set_recipe", {"QUARTZITE": m["new"]}, source="mpc_sim")


def compare(sim_factory, scenario: str = "D", hours: float = 24.0, mpc_every_h: float = 3.0, hybrid_offsets=None,
            fast: bool = False, progress=None) -> dict:
    out = {}
    policies = ["baseline", "physics_mpc"] + (["hybrid_mpc"] if hybrid_offsets is not None else [])
    for pi, pol in enumerate(policies):
        sim = sim_factory()
        ev = resolve_events(SCENARIOS[scenario]["events"], sim.sp.tap)
        sim.schedule(ev)
        base = BaselinePolicy()
        mpc = None
        if pol != "baseline":
            cfg = default_mpc_config(sim.f, sim.sp)
            if fast:
                cfg.horizon_h, cfg.sim_dt_s, cfg.sim_substeps = 3.0, 600.0, 8
            mpc = SupervisoryMPC(sim, cfg)
        last = -1e9
        ctime = []
        viol = 0
        traj = {"t_h": [], "T_slag_K": [], "si": [], "fill": [], "power": []}
        moves = []
        n = int(hours * 3600 / sim.dt)
        for i in range(n):
            t = sim.t / 3600
            if pol == "baseline":
                a = base.act(sim)
                if a:
                    moves.append(a)
            elif t - last >= mpc_every_h:
                last = t
                t0 = time.time()
                offs = None
                if pol == "hybrid_mpc":
                    offs = hybrid_offsets(sim) if callable(hybrid_offsets) else hybrid_offsets
                rec = mpc.recommend(sim, offs)
                ctime.append(time.time() - t0)
                if rec["solver"]["success"] and not rec["verification"]["hard_violations"]:
                    _apply(sim, rec)
                    moves.append({"t_h": t, "moves": {m["name"]: m["new"] for m in rec["moves"]}})
            s = sim.step()
            if s["slag_props"].get("superheat_K", 99) < 0 or s["fecr_si_wt_pct"] > sim.f["max_si_wt_pct"] or \
                    max(s["electrical"]["i_phase_A"]) > sim.design.i_max:
                viol += 1
            if i % max(int(1800 / sim.dt), 1) == 0:
                traj["t_h"].append(s["t_h"])
                traj["T_slag_K"].append(s["zone_temperature_k"]["SL"])
                traj["si"].append(s["fecr_si_wt_pct"])
                traj["fill"].append(s["coke_bed_fill"])
                traj["power"].append(s["power_mw"])
            if progress:
                progress((pi + i / n) / len(policies))
        taps = sim.taps
        metal = sum(t["metal_t"] for t in taps)
        E = sum(t["electric_MWh_since_last_tap"] for t in taps)
        crm = sum(t["cr_in_metal_kg"] for t in taps)
        crs = sum(t["cr_in_slag_kg"] for t in taps)
        out[pol] = {"metal_t": metal, "sec_kwh_per_t": E * 1000 / metal if metal else None,
                    "cr_recovery_pct": 100 * crm / (crm + crs) if crm + crs else None,
                    "cr_loss_slag_t": crs / 1000, "constraint_violation_steps": viol,
                    "mean_si_wt_pct": float(np.mean([t["metal_wt_pct"].get("Si", 0) for t in taps])) if taps else None,
                    "mpc_calls": len(ctime), "mean_compute_s": float(np.mean(ctime)) if ctime else 0.0,
                    "moves": moves, "trajectory": traj}
    return {"scenario": scenario, "hours": hours, "results": out, "synthetic": True,
            "note": "identical disturbance schedule and seed for every policy; SYNTHETIC / NOT PLANT-VALIDATED"}
