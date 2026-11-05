"""Hidden-state estimation: Extended Kalman Filter with out-of-sequence (delayed) lab assays.

Primary: EKF on a reduced-order observer model (9 states). Delayed assays are fused at their
*sample* time by rewinding to the stored prior, updating, and re-propagating stored inputs and
in-between measurements forward ("re-filtering"). Only measurements whose *reported* time is
<= the current time are ever used, so historical replay has no future leakage.
Fallback: exponentially-weighted complementary observer (always available, no covariance),
activated when the EKF covariance loses positive definiteness or innovations are persistently
inconsistent (NIS outside the chi-square 99.9 % band for 10 consecutive updates).

States: 0 M_metal [t], 1 M_slag [t], 2 Cr_metal [wt%], 3 C_metal [wt%], 4 Si_metal [wt%],
        5 T_metal [K], 6 Cr2O3eq_slag [wt%], 7 theta = metal productivity [t/MWh],
        8 coke-bed fill [-]
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..domain import electrical as el

NAMES = ["metal_inventory_t", "slag_inventory_t", "metal_cr_wt_pct", "metal_c_wt_pct", "metal_si_wt_pct",
         "metal_T_K", "slag_cr2o3eq_wt_pct", "productivity_t_per_mwh", "coke_bed_fill"]
OBSERVABILITY = {"metal_inventory_t": "inferred (tap masses)", "slag_inventory_t": "inferred (tap masses)",
                 "metal_cr_wt_pct": "measured (delayed lab)", "metal_c_wt_pct": "measured (delayed lab)",
                 "metal_si_wt_pct": "measured (delayed lab)", "metal_T_K": "measured at tap only",
                 "slag_cr2o3eq_wt_pct": "measured (delayed lab)", "productivity_t_per_mwh": "inferred (parameter)",
                 "coke_bed_fill": "inferred (electrical resistance) - weakly observable"}


@dataclass
class ObserverModel:
    targets: dict  # static-model predictions for composition targets
    heel_m_t: float
    heel_s_t: float
    slag_to_metal: float
    tau_mix_h: float = 6.0
    tau_T_h: float = 1.5
    T_nom: float = 1910.0
    P_nom: float = 35.4
    kT: float = 120.0
    c_ratio: float = 1.0
    kappa_fill: float = 0.8  # fill change per h per unit carbon-ratio excess

    def f(self, x: np.ndarray, u: dict, dt_h: float) -> np.ndarray:
        P = u["P_MW"]
        y = x.copy()
        prod = max(x[7], 0.05) * P
        y[0] += prod * dt_h
        y[1] += prod * self.slag_to_metal * dt_h
        a = min(dt_h / self.tau_mix_h, 1.0)
        y[2] += a * (self.targets["Cr"] - x[2])
        y[3] += a * (self.targets["C"] - x[3])
        y[4] += a * (self.targets["Si"] - x[4])
        y[6] += a * (self.targets["Cr2O3eq"] - x[6])
        Tt = self.T_nom + self.kT * (P / self.P_nom - 1)
        y[5] += min(dt_h / self.tau_T_h, 1.0) * (Tt - x[5])
        y[8] += self.kappa_fill * (self.c_ratio - 1.0) * dt_h
        return y


@dataclass
class EKF:
    model: ObserverModel
    x: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    t_h: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=600))
    innov: deque = field(default_factory=lambda: deque(maxlen=500))
    bad_nis: int = 0
    mode: str = "EKF"

    def predict(self, u: dict, dt_h: float):
        F = self._jac(lambda s: self.model.f(s, u, dt_h), self.x)
        self.x = self.model.f(self.x, u, dt_h)
        self.P = F @ self.P @ F.T + self.Q * dt_h
        self.t_h += dt_h

    def _jac(self, fn, x, eps=1e-5):
        fx = fn(x)
        J = np.zeros((len(fx), len(x)))
        for i in range(len(x)):
            d = np.zeros_like(x)
            h = eps * max(1.0, abs(x[i]))
            d[i] = h
            J[:, i] = (fn(x + d) - fx) / h
        return J

    def update(self, hfun, z: np.ndarray, Rm: np.ndarray, label: str):
        H = self._jac(hfun, self.x)
        yhat = hfun(self.x)
        nu = z - yhat
        S = H @ self.P @ H.T + Rm
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ nu
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T  # Joseph form
        nis = float(nu @ Sinv @ nu)
        chi = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47}.get(len(z), 20.0)
        self.bad_nis = self.bad_nis + 1 if nis > chi else 0
        rec = {"t_h": self.t_h, "label": label, "innovation": nu.tolist(), "nis": nis, "nis_limit_99_9": chi}
        self.innov.append(rec)
        return rec


class StateEstimator:
    """Wraps the EKF with re-filtering for delayed assays and a complementary fallback."""

    def __init__(self, sim, static_bal: dict | None = None):
        f = sim.f
        tg = static_bal or {}
        mw = tg.get("metal_wt_pct", {"Cr": 63.5, "C": 6.8, "Si": 3.6})
        self.design = sim.design
        self.f = f
        model = ObserverModel(targets={"Cr": mw.get("Cr", 63.5), "C": mw.get("C", 6.8), "Si": mw.get("Si", 3.6),
                                       "Cr2O3eq": tg.get("slag_cr2o3_equiv_wt_pct", 9.0)},
                              heel_m_t=sim.M_metal_heel / 1000, heel_s_t=sim.M_slag_heel / 1000,
                              slag_to_metal=tg.get("slag_to_metal", 0.95), P_nom=f["nominal_power_mw"])
        x0 = np.array([sim.M_metal_heel / 1000 + 5, sim.M_slag_heel / 1000 + 5, 63.0, 6.8, 3.6, 1905.0, 9.0, 0.30, 1.0])
        P0 = np.diag([15.0, 20.0, 2.0, 0.5, 0.8, 40.0, 2.0, 0.03, 0.2]) ** 2
        Q = np.diag([0.5, 0.8, 0.3, 0.1, 0.2, 15.0, 0.4, 0.002, 0.03]) ** 2
        self.ekf = EKF(model, x0, P0, Q)
        self.fallback = x0.copy()
        self.sim_ref = sim
        self.last_lab_tap = 0
        self.events: list[dict] = []
        self.rng = np.random.default_rng(11)

    # --- measurement functions
    def _h_tap_metal(self, x):
        m = self.ekf.model
        return np.array([self.f["tap_fraction_metal"] * (x[0] - m.heel_m_t)])

    def _h_tap_slag(self, x):
        m = self.ekf.model
        return np.array([self.f["tap_fraction_slag"] * (x[1] - m.heel_s_t)])

    def _h_R(self, x, h_tip):
        rho = el.rho_coke_bed(self.design.rho_cb0, x[5] + 45.0, x[8])
        return np.array([el.phase_load(self.design, h_tip, rho, self.design.rho_bur)["R_load"] * 1e3])

    def step(self, t_h: float, tags: dict, tap: dict | None, labs: list[dict], dt_h: float) -> dict:
        P = _v(tags, "EL_P_TOTAL")
        u = {"P_MW": P if P is not None else self.ekf.model.P_nom}
        rec = {"t_h": t_h, "u": u, "meas": []}
        self.ekf.history.append((copy.deepcopy(self.ekf.x), copy.deepcopy(self.ekf.P), self.ekf.t_h, u, dt_h, rec))
        self.ekf.predict(u, dt_h)
        # electrical resistance -> coke-bed fill (continuous)
        Rs = [_v(tags, f"EL_R{k}") for k in (1, 2, 3)]
        hs = [_v(tags, f"ELD_TIP{k}") for k in (1, 2, 3)]
        if all(v is not None for v in Rs + hs):
            Rbar, hbar = float(np.mean(Rs)), float(np.mean(hs))
            m = ("R", (lambda x, h=hbar: self._h_R(x, h)), np.array([Rbar]), np.array([[0.02 ** 2]]))
            rec["meas"].append(m)
            self.ekf.update(m[1], m[2], m[3], "electrical_R")
        if tap is not None:
            # tap removes mass (known input) after using it as a measurement of inventory
            zm = tap["metal_t"] + self.rng.normal(0, 0.3)  # ladle weigh-bridge
            zs = tap["slag_t"] + self.rng.normal(0, 1.0)  # slag-pot count estimate
            mm = ("tap_metal", self._h_tap_metal, np.array([zm]), np.array([[0.4 ** 2]]))
            ms = ("tap_slag", self._h_tap_slag, np.array([zs]), np.array([[1.5 ** 2]]))
            mT = ("tap_T", lambda x: np.array([x[5]]), np.array([tap["metal_T_K"] + self.rng.normal(0, 8)]), np.array([[10.0 ** 2]]))
            for m in (mm, ms, mT):
                rec["meas"].append(m)
                self.ekf.update(m[1], m[2], m[3], m[0])
            self.ekf.x[0] -= zm
            self.ekf.x[1] -= zs
            rec["tap_removal"] = (zm, zs)
        for lab in labs:  # delayed assays: re-filter from sample time
            self._delayed_update(lab)
        if self.ekf.bad_nis > 10 or np.any(np.linalg.eigvalsh(0.5 * (self.ekf.P + self.ekf.P.T)) <= 0):
            self.ekf.mode = "FALLBACK"
        # complementary fallback observer
        a = min(dt_h / 2.0, 1.0)
        self.fallback = self.fallback + a * (self.ekf.model.f(self.fallback, u, dt_h) - self.fallback)
        if tap is not None:
            self.fallback[0] -= rec["tap_removal"][0]
            self.fallback[1] -= rec["tap_removal"][1]
        return self.state()

    def _delayed_update(self, lab: dict):
        ts = lab["sample_t_h"]
        hist = list(self.ekf.history)
        idx = None
        for i, (x, P, t, u, dt, rec) in enumerate(hist):
            if t <= ts < t + dt + 1e-9:
                idx = i
        if idx is None:
            self.events.append({"t_h": self.ekf.t_h, "msg": f"lab for tap {lab['tap_no']} older than buffer - discarded"})
            return
        x, P, t, u, dt, rec = hist[idx]
        self.ekf.x, self.ekf.P, self.ekf.t_h = x.copy(), P.copy(), t
        z = np.array([lab["metal"].get("Cr", np.nan), lab["metal"].get("C", np.nan), lab["metal"].get("Si", np.nan),
                      lab["slag"].get("Cr2O3", np.nan) + lab["slag"].get("CrO", 0.0) * 1.117])
        ok = np.isfinite(z)
        sel = np.where(ok)[0]
        idxs = [2, 3, 4, 6]
        hfun = lambda xx: np.array([xx[idxs[i]] for i in sel])
        Rm = np.diag(np.array([0.3, 0.08, 0.08, 0.4])[sel] ** 2)
        # replay: re-predict and re-apply stored measurements from the sample step forward,
        # refreshing the stored priors so later delayed assays rebase on the corrected trajectory
        new_hist = hist[:idx]
        for j in range(idx, len(hist)):
            _, _, tj, uj, dtj, recj = hist[j]
            new_hist.append((self.ekf.x.copy(), self.ekf.P.copy(), self.ekf.t_h, uj, dtj, recj))
            self.ekf.predict(uj, dtj)
            for m in recj["meas"]:
                self.ekf.update(m[1], m[2], m[3], m[0] + "_replay")
            if "tap_removal" in recj:
                self.ekf.x[0] -= recj["tap_removal"][0]
                self.ekf.x[1] -= recj["tap_removal"][1]
            if j == idx:
                self.ekf.update(hfun, z[sel], Rm, f"lab_tap{lab['tap_no']}")
        self.ekf.history = deque(new_hist, maxlen=self.ekf.history.maxlen)
        self.events.append({"t_h": self.ekf.t_h, "msg": f"lab assay (tap {lab['tap_no']}, sampled {ts:.2f} h) fused by re-filtering"})

    def state(self) -> dict:
        x = self.ekf.x if self.ekf.mode == "EKF" else self.fallback
        sd = np.sqrt(np.maximum(np.diag(self.ekf.P), 0))
        innov = list(self.ekf.innov)[-50:]
        nis = [r["nis"] for r in innov]
        return {"mode": self.ekf.mode, "t_h": self.ekf.t_h,
                "states": [{"name": n, "value": float(x[i]), "sd": float(sd[i]), "ci95": [float(x[i] - 1.96 * sd[i]), float(x[i] + 1.96 * sd[i])],
                            "observability": OBSERVABILITY[n]} for i, n in enumerate(NAMES)],
                "innovations": innov[-20:], "nis_mean": float(np.mean(nis)) if nis else None,
                "nis_consistency": (float(np.mean([r["nis"] <= r["nis_limit_99_9"] for r in innov])) if innov else None),
                "events": self.events[-10:], "primary": "EKF (reduced-order, Joseph form, out-of-sequence re-filtering)",
                "fallback": "complementary exponential observer"}


def _v(tags: dict, k: str):
    t = tags.get(k)
    if not t or t.get("value") is None or t.get("quality") in ("BAD",):
        return None
    return float(t["value"])
