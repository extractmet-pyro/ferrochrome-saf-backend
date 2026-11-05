"""Supervisory MPC (linearised step-response / DMC-type, identified online from the physics twin).

Prediction model: for the current state, the twin is cloned and simulated over the horizon
(i) with current setpoints (free response) and (ii) with a step on each MV (identification
runs). Step-response coefficients S_i(k) give the linear prediction
        y(k) = y_free(k) + sum_i S_i(k) du_i         (single move at t0, held: move blocking M = 1)
which is optimised with SLSQP under MV bounds / move limits, discrete tap enumeration and
slack-penalised soft CV limits. This is a LINEARISED MPC; nonlinearity is handled by
(a) re-identification each execution (receding horizon) and (b) verification of the chosen
moves on the full NONLINEAR twin before the safety gate. Hybrid mode adds ML residual
corrections (SEC, Cr recovery) when the active models are in-distribution.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import time
import uuid

import numpy as np

from .constraints import MPCConfig, default_mpc_config
from .objectives import objective
from .optimizers import solve

CV_NAMES = ["T_slag_K", "slag_superheat_K", "si_wt_pct", "c_wt_pct", "cr_wt_pct", "coke_bed_fill", "power_mw", "i_max_ka",
            "slag_cr2o3_eq", "T_metal_K"]


def _cv(s: dict) -> dict:
    return {"T_slag_K": s["zone_temperature_k"]["SL"], "slag_superheat_K": s["slag_props"].get("superheat_K", 0.0),
            "si_wt_pct": s["fecr_si_wt_pct"], "c_wt_pct": s["fecr_c_wt_pct"], "cr_wt_pct": s["fecr_cr_wt_pct"],
            "coke_bed_fill": s["coke_bed_fill"], "power_mw": s["power_mw"], "i_max_ka": max(s["electrical"]["i_phase_A"]) / 1000,
            "slag_cr2o3_eq": s["slag_cr2o3_equiv_wt_pct"], "T_metal_K": s["zone_temperature_k"]["MB"]}


def _coke_key(sim) -> str:
    """The coke grade currently charged (the MV acts on the main coke line)."""
    r = sim.sp.recipe
    return max(("COKE_MET", "COKE_LOWQ"), key=lambda k: float(r.get(k, 0.0)))


def _mv_get(sim, name):
    return {"tap": float(sim.sp.tap), "r_setpoint_mohm": sim.sp.r_setpoint_mohm,
            "coke_kg_t": float(sim.sp.recipe.get(_coke_key(sim), 0.0)), "quartz_kg_t": float(sim.sp.recipe.get("QUARTZITE", 0.0))}[name]


def _mv_set(sim, name, v):
    if name == "tap":
        sim.sp.tap = int(round(v))
    elif name == "r_setpoint_mohm":
        sim.sp.r_setpoint_mohm = float(v)
    elif name == "coke_kg_t":
        sim.sp.recipe[_coke_key(sim)] = float(v)
    elif name == "quartz_kg_t":
        sim.sp.recipe["QUARTZITE"] = float(v)


def run_horizon(sim, cfg: MPCConfig, moves: dict | None = None) -> dict:
    s = sim.clone()
    s.dt, s.nsub = cfg.sim_dt_s, cfg.sim_substeps
    for k, v in (moves or {}).items():
        _mv_set(s, k, v)
    n_per = max(int(round(cfg.sample_h * 3600 / s.dt)), 1)
    n_s = int(round(cfg.horizon_h / cfg.sample_h))
    traj = {k: [] for k in CV_NAMES}
    t = []
    m0 = s.z["MB"].mass()
    taps0 = len(s.taps)
    E = 0.0
    for _ in range(n_s):
        for _ in range(n_per):
            snap = s.step()
            E += snap["power_mw"] * s.dt / 3600
        cv = _cv(snap)
        for k in CV_NAMES:
            traj[k].append(float(cv[k]))
        t.append(snap["t_h"])
    taps = s.taps[taps0:]
    metal_t = (s.z["MB"].mass() - m0) / 1000 + sum(tp["metal_t"] for tp in taps)
    crm = sum(tp["cr_in_metal_kg"] for tp in taps)
    crs = sum(tp["cr_in_slag_kg"] for tp in taps)
    return {"t_h": t, "traj": traj, "sec_avg": E * 1000 / max(metal_t, 1e-6), "metal_t": metal_t,
            "cr_rec_avg": 100 * crm / (crm + crs) if crm + crs > 0 else float("nan"), "energy_MWh": E,
            "status": s.status}


class SupervisoryMPC:
    def __init__(self, sim, cfg: MPCConfig | None = None):
        self.cfg = cfg or default_mpc_config(sim.f, sim.sp)
        self.model = None
        self.secret = b"change-me"

    def identify(self, sim) -> dict:
        t0 = time.time()
        free = run_horizon(sim, self.cfg)
        S = {}
        for mv in self.cfg.mvs:
            cur = _mv_get(sim, mv.name)
            d = mv.step_for_id if cur + mv.step_for_id <= mv.hi else -mv.step_for_id
            r = run_horizon(sim, self.cfg, {mv.name: cur + d})
            S[mv.name] = {"traj": {k: ((np.array(r["traj"][k]) - np.array(free["traj"][k])) / d).tolist() for k in CV_NAMES},
                          "sec": (r["sec_avg"] - free["sec_avg"]) / d,
                          "cr": ((r["cr_rec_avg"] - free["cr_rec_avg"]) / d) if np.isfinite(r["cr_rec_avg"]) and np.isfinite(free["cr_rec_avg"]) else 0.0,
                          "step": d}
        self.model = {"free": free, "S": S, "t_identified_h": sim.t / 3600, "wall_s": time.time() - t0,
                      "type": "linearised step-response (identified from nonlinear twin)"}
        return self.model

    def predict(self, du: dict, offsets: dict | None = None) -> dict:
        free, S = self.model["free"], self.model["S"]
        traj = {k: np.array(free["traj"][k], float) for k in CV_NAMES}
        sec = free["sec_avg"]
        cr = free["cr_rec_avg"] if np.isfinite(free["cr_rec_avg"]) else 90.0
        for name, d in du.items():
            for k in CV_NAMES:
                traj[k] = traj[k] + np.array(S[name]["traj"][k]) * d
            sec += S[name]["sec"] * d
            cr += S[name]["cr"] * d
        if offsets:
            sec += offsets.get("sec", 0.0)
            cr += offsets.get("cr", 0.0)
        return {"traj": {k: v.tolist() for k, v in traj.items()}, "sec_avg": sec, "cr_rec_avg": cr, "t_h": free["t_h"]}

    def recommend(self, sim, hybrid_offsets: dict | None = None, reidentify: bool = True) -> dict:
        t_start = time.time()
        deadline = t_start + self.cfg.timeout_s
        if reidentify or self.model is None:
            self.identify(sim)
        mvs = self.cfg.mvs
        cont = [m for m in mvs if not m.discrete]
        cur = {m.name: _mv_get(sim, m.name) for m in mvs}
        scales = np.array([m.scale for m in cont])
        bounds = []
        for m in cont:
            lo_, hi_ = max(m.lo - cur[m.name], -m.max_move), min(m.hi - cur[m.name], m.max_move)
            if lo_ > hi_:  # current value already outside the MV range: only moves back toward the range are allowed
                lo_, hi_ = (0.0, m.max_move) if cur[m.name] < m.lo else (-m.max_move, 0.0)
            bounds.append((lo_, hi_))
        tap_opts = [0]
        locked = sim.t - sim.last_tap_change_t < sim.f["tap_min_dwell_s"]
        if not locked:
            tap_opts = [d for d in (-1, 0, 1) if 1 <= cur["tap"] + d <= sim.design.tap_count]
        results = []
        for dtap in tap_opts:
            def fun(x, dtap=dtap):
                du = {m.name: float(x[i]) for i, m in enumerate(cont)}
                du["tap"] = dtap
                p = self.predict(du, hybrid_offsets)
                J, _ = objective(p, np.concatenate([x, [dtap]]), np.concatenate([scales, [1.0]]), self.cfg, self.cfg.cvs)
                return J
            x0 = np.zeros(len(cont))
            starts = [np.array([b[0] * 0.5 + b[1] * 0.5 for b in bounds])]
            r = solve(fun, x0, bounds, deadline, starts)
            results.append((r, dtap))
        ok = [x for x in results if x[0]["success"]]
        r, dtap = min(ok or results, key=lambda x: x[0]["fun"])
        du = {m.name: float(r["x"][i]) for i, m in enumerate(cont)}
        du["tap"] = float(dtap)
        pred = self.predict(du, hybrid_offsets)
        J, slacks = objective(pred, np.concatenate([r["x"], [dtap]]), np.concatenate([scales, [1.0]]), self.cfg, self.cfg.cvs)
        base_pred = self.predict({}, hybrid_offsets)
        J0, _ = objective(base_pred, np.zeros(len(cont) + 1), np.concatenate([scales, [1.0]]), self.cfg, self.cfg.cvs)
        moves = [{"name": m.name, "units": m.units, "current": cur[m.name], "new": cur[m.name] + du[m.name], "delta": du[m.name],
                  "lo": m.lo, "hi": m.hi, "max_move": m.max_move, "discrete": m.discrete, "description": m.description} for m in mvs]
        # verification on the nonlinear twin
        new_mv = {m["name"]: m["new"] for m in moves}
        ver = run_horizon(sim, self.cfg, new_mv)
        hard = []
        for cv in self.cfg.cvs:
            y = np.array(ver["traj"].get(cv.name, []))
            if y.size == 0:
                continue
            if cv.hard_lo is not None and np.min(y) < cv.hard_lo:
                hard.append(f"{cv.name} min {np.min(y):.2f} < {cv.hard_lo}")
            if cv.hard_hi is not None and np.max(y) > cv.hard_hi:
                hard.append(f"{cv.name} max {np.max(y):.2f} > {cv.hard_hi}")
        mism = {k: float(np.max(np.abs(np.array(ver["traj"][k]) - np.array(pred["traj"][k])))) for k in CV_NAMES}
        margins = {}
        for cv in self.cfg.cvs:
            y = np.array(ver["traj"][cv.name])
            if cv.soft_lo is not None:
                margins[f"{cv.name}>= {cv.soft_lo}"] = float(np.min(y) - cv.soft_lo)
            if cv.soft_hi is not None:
                margins[f"{cv.name}<= {cv.soft_hi}"] = float(cv.soft_hi - np.max(y))
            if cv.hard_hi is not None:
                margins[f"{cv.name}<= {cv.hard_hi} (hard)"] = float(cv.hard_hi - np.max(y))
        free = self.model["free"]
        rec = {
            "id": f"rec-{uuid.uuid4().hex[:10]}", "created_t_h": sim.t / 3600, "advisory_only": True,
            "mode": "hybrid" if hybrid_offsets else "physics",
            "moves": moves, "objective": {"J": J, "J_no_change": J0, "slacks": slacks},
            "predicted": pred, "predicted_no_change": base_pred,
            "verification": {"nonlinear": ver, "hard_violations": hard, "model_mismatch_max_abs": mism},
            "expected_effect": {"sec_kwh_per_t": ver["sec_avg"] - free["sec_avg"], "cr_recovery_pct": (ver["cr_rec_avg"] - free["cr_rec_avg"])
                                if np.isfinite(ver["cr_rec_avg"]) and np.isfinite(free["cr_rec_avg"]) else None,
                                "production_t": ver["metal_t"] - free["metal_t"]},
            "constraint_margins": margins,
            "solver": {"success": bool(r["success"]), "message": r["message"], "iterations": r["nit"], "timed_out": bool(r.get("timed_out", False)),
                       "starts": r.get("starts_tried"), "tap_options": tap_opts, "tap_locked": bool(locked),
                       "wall_time_s": time.time() - t_start, "identification_wall_s": self.model["wall_s"],
                       "method": "SLSQP on linearised step-response model + nonlinear verification"},
            "confidence": "medium" if max(mism.values()) < 50 else "low",
            "explanation": self._explain(moves, slacks, ver, free),
        }
        payload = json.dumps({k: rec[k] for k in ("id", "created_t_h", "moves")}, sort_keys=True, default=float).encode()
        rec["audit_signature"] = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
        return rec

    def _explain(self, moves, slacks, ver, free) -> list[str]:
        out = []
        for m in moves:
            if abs(m["delta"]) > 1e-6:
                out.append(f"{m['name']}: {m['current']:.3g} -> {m['new']:.3g} {m['units']}")
        if not out:
            out.append("No change recommended: current setpoints are near the constrained optimum.")
        out.append(f"Predicted SEC over horizon {ver['sec_avg']:.0f} vs {free['sec_avg']:.0f} kWh/t (no change).")
        if slacks:
            out.append("Soft limits still active: " + ", ".join(slacks))
        return out
