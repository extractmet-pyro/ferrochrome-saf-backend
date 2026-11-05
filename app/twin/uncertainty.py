"""Uncertainty propagation, sensitivity (tornado) and parameter identifiability.

* Monte Carlo on the static balance: assays, moisture, recovery surrogates, temperatures.
* One-at-a-time sensitivity of SEC / Cr recovery / cost to plant drivers.
* Identifiability: finite-difference Jacobian of observable outputs w.r.t. kinetic parameters
  from a short dynamic run, scaled; SVD singular values and condition number. Parameters that
  load mainly on small singular values are reported as NOT identifiable from the available
  instrumentation and their dependent states are flagged as uncertain.
"""
from __future__ import annotations

import copy

import numpy as np

from ..domain import kinetics as kn
from ..domain.material_streams import default_materials
from ..domain.static_balance import DEFAULT_PRICES, static_balance


def monte_carlo(recipe: dict, n: int = 200, seed: int = 1, cond: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    base_m = default_materials()
    out = {"sec": [], "cr_rec": [], "metal_cr": [], "slag_cr2o3": [], "cost": []}
    for _ in range(n):
        m = copy.deepcopy(base_m)
        for mat in m.values():
            a = mat.assay_dry_wt_pct
            for k in list(a):
                a[k] = max(a[k] * (1 + rng.normal(0, 0.02)), 0)
            tot = sum(a.values())
            for k in a:
                a[k] *= 100 / tot
            mat.moisture_wet_wt_pct = max(mat.moisture_wet_wt_pct + rng.normal(0, 0.5), 0)
        c = dict(cond or {})
        c["T_slag_K"] = c.get("T_slag_K", 1905) + rng.normal(0, 20)
        c["T_metal_K"] = c.get("T_metal_K", 1910) + rng.normal(0, 20)
        c["heat_loss_MW"] = max(c.get("heat_loss_MW", 1.2) * (1 + rng.normal(0, 0.2)), 0.2)
        b = static_balance(recipe, m, c, DEFAULT_PRICES)
        out["sec"].append(b["energy"]["sec_kwh_per_t_fecr"])
        out["cr_rec"].append(b["cr_recovery_pct"])
        out["metal_cr"].append(b["metal_wt_pct"]["Cr"])
        out["slag_cr2o3"].append(b["slag_cr2o3_equiv_wt_pct"])
        out["cost"].append(b["cost"]["total_inr_per_t_fecr"])
    summ = {k: {"p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                "mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in out.items()}
    return {"n": n, "summary": summ, "samples": {k: [float(x) for x in v[:200]] for k, v in out.items()},
            "label": "SYNTHETIC uncertainty propagation (assay +/-2 % rel, moisture +/-0.5, T +/-20 K, losses +/-20 %)"}


SENS_DRIVERS = {
    "electricity_price": ("price", "electricity_inr_per_mwh", 0.2),
    "ore_A_moisture": ("moisture", "ORE_A_LUMPY", 2.0),
    "coke_fixed_C": ("assay", ("COKE_MET", "C"), -5.0),
    "quartzite_rate": ("recipe", "QUARTZITE", 25.0),
    "coke_rate": ("recipe", "COKE_MET", 15.0),
    "slag_temperature": ("cond", "T_slag_K", 30.0),
    "feed_temperature": ("cond", "T_feed_K", 200.0),
    "heat_losses": ("cond", "heat_loss_MW", 0.5),
}


def sensitivity(recipe: dict) -> dict:
    m0 = default_materials()
    base = static_balance(recipe, m0, None, DEFAULT_PRICES)
    rows = []
    for name, (kind, key, d) in SENS_DRIVERS.items():
        res = []
        for sgn in (-1, 1):
            m = copy.deepcopy(m0)
            r = copy.deepcopy(recipe)
            c = {}
            p = dict(DEFAULT_PRICES)
            if kind == "price":
                p[key] = p[key] * (1 + sgn * d)
            elif kind == "moisture":
                m[key].moisture_wet_wt_pct = max(m[key].moisture_wet_wt_pct + sgn * d, 0)
            elif kind == "assay":
                mk, ak = key
                m[mk].assay_dry_wt_pct[ak] += sgn * d
                m[mk].assay_dry_wt_pct["SiO2"] -= sgn * d
            elif kind == "recipe":
                r[key] = max(r.get(key, 0) + sgn * d, 0)
            elif kind == "cond":
                from ..domain.static_balance import DEFAULT_COND
                c[key] = DEFAULT_COND[key] + sgn * d
            b = static_balance(r, m, c, p)
            res.append(b)
        rows.append({"driver": name, "step": d,
                     "sec_low": res[0]["energy"]["sec_kwh_per_t_fecr"], "sec_high": res[1]["energy"]["sec_kwh_per_t_fecr"],
                     "cost_low": res[0]["cost"]["total_inr_per_t_fecr"], "cost_high": res[1]["cost"]["total_inr_per_t_fecr"],
                     "crrec_low": res[0]["cr_recovery_pct"], "crrec_high": res[1]["cr_recovery_pct"]})
    return {"base": {"sec": base["energy"]["sec_kwh_per_t_fecr"], "cost": base["cost"]["total_inr_per_t_fecr"],
                     "cr_rec": base["cr_recovery_pct"]}, "rows": rows}


IDENT_PARAMS = ["sl_Cr2O3", "sl_SiO2", "km_chromite", "ss_FeCr2O4", "carburise", "boudouard"]


def identifiability(sim_factory, hours: float = 4.0, rel_step: float = 0.2) -> dict:
    """Jacobian of measurable outputs (tap Cr, Si, C, slag Cr2O3, off-gas CO, power) w.r.t. kinetic parameters."""
    def outputs(sim):
        s = None
        for _ in range(int(hours * 3600 / sim.dt)):
            s = sim.step()
        return np.array([s["fecr_cr_wt_pct"], s["fecr_si_wt_pct"], s["fecr_c_wt_pct"], s["slag_cr2o3_equiv_wt_pct"],
                         s["offgas_co_dry_vol_pct"], s["power_mw"], s["coke_bed_fill"]])

    base_sim = sim_factory()
    y0 = outputs(base_sim.clone())
    J = []
    for p in IDENT_PARAMS:
        s = base_sim.clone()
        s.k = copy.deepcopy(base_sim.k)
        s.k[p] = kn.KParam(s.k[p].A * (1 + rel_step), s.k[p].E, s.k[p].provenance)
        y = outputs(s)
        J.append((y - y0) / rel_step / np.maximum(np.abs(y0), 1e-6))
    J = np.array(J).T  # outputs x params (relative sensitivities)
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    cond = float(sv[0] / max(sv[-1], 1e-12))
    weak = [IDENT_PARAMS[int(np.argmax(np.abs(Vt[-1])))]] if sv[-1] < 1e-3 * sv[0] else []
    return {"params": IDENT_PARAMS, "outputs": ["metal_Cr", "metal_Si", "metal_C", "slag_Cr2O3eq", "offgas_CO", "power", "coke_bed_fill"],
            "relative_jacobian": J.tolist(), "singular_values": sv.tolist(), "condition_number": cond,
            "weakly_identifiable": weak,
            "interpretation": "large condition number => some kinetic parameter combinations cannot be separated by plant measurements; "
                              "report dependent states (e.g. undissolved chromite fraction) with uncertainty, not as exact"}
