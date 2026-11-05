"""Synthetic training-campaign generation and chronological training of residual models.

The synthetic "plant" is the dynamic twin with a HIDDEN mismatch that the static physics
model does not contain (chromite mass transfer degrading with fines, altered slag current
path, coke reactivity drift). Targets are residuals actual - physics on a per-tap basis.
All data are SYNTHETIC; real plant data must replace them before any site use.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import pandas as pd

from ..domain import kinetics as kn
from ..domain.material_streams import default_materials
from ..domain.static_balance import DEFAULT_PRICES, static_balance
from . import features as F
from .drift import OODDetector, drift_report
from .explain import importance
from .models import HP_GRID, make_model


def _plant(seed: int, spinup_h: float = 20.0):
    from ..twin.simulator import FeCrSAFSimulator
    sim = FeCrSAFSimulator(seed=seed, dt_s=180.0, substeps=4, spinup_h=spinup_h)
    sim.design.slag_path_frac = 0.50  # hidden mismatch (model assumes 0.55)
    return sim


def generate_campaign(days: float = 15.0, seed: int = 11, change_every_h: float = 8.0, progress=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim = _plant(seed)
    mats = default_materials()
    base_k = copy.deepcopy(sim.k)
    base_tap = sim.sp.tap
    rows = []
    acc = {"n": 0}
    keys = ["power_mw", "coke_bed_fill", "burden_dp_kpa", "offgas_co_dry_vol_pct"]
    last_tap = None
    n_steps = int(days * 24 * 3600 / sim.dt)
    next_change = 0.0
    from ..control.policy import BaselinePolicy
    pol = BaselinePolicy(period_h=3.0, step=8.0, deadband_m=0.3)  # shift practice keeps the coke bed bounded
    for i in range(n_steps):
        t_h = sim.t / 3600
        if t_h >= next_change:
            blend = rng.dirichlet([3, 3, 1.2])
            rec = {"ore_blend": {"ORE_A_LUMPY": blend[0], "ORE_B_PELLET": blend[1], "ORE_C_FRIABLE": blend[2]},
                   "COKE_MET": float(np.clip(sim.sp.recipe.get("COKE_MET", 172) + rng.normal(0, 8), 145, 200)),
                   "COAL_LA": float(rng.uniform(25, 50)),
                   "QUARTZITE": float(rng.uniform(55, 125))}
            sim.apply_action("set_recipe", rec, source="schedule")
            sim.apply_action("set_tap", {"tap": int(np.clip(base_tap + rng.integers(-3, 3), 1, sim.design.tap_count))}, source="schedule")
            sim.apply_action("disturb", {"fines_extra_pct": float(rng.uniform(0, 14)), "moisture_extra_pct": float(rng.uniform(0, 3)),
                                         "coke_reactivity_factor": float(rng.uniform(0.8, 1.5))}, source="schedule")
            next_change += change_every_h * float(rng.uniform(0.7, 1.3))
        # hidden mismatch: chromite mass transfer degrades with fines (not in physics model)
        fines = sim._fines_pct()
        sim.k["km_chromite"] = kn.KParam(base_k["km_chromite"].A * float(np.clip(1 - 0.02 * (fines - 8), 0.5, 1.2)),
                                         base_k["km_chromite"].E)
        pol.act(sim)
        s = sim.step()
        for k in keys:
            acc[k] = acc.get(k, 0.0) + float(s[k])
        acc["depth"] = acc.get("depth", 0.0) + float(np.mean(s["electrode_depth_m"]))
        acc["n"] += 1
        tap = s["tap_now"]
        if tap is not None:
            n = max(acc["n"], 1)
            r = sim.sp.recipe
            bl = r["ore_blend"]
            sb = sum(bl.values())
            mats_d = sim._mats_with_dist()
            cond = {"T_slag_K": last_tap["slag_T_K"] if last_tap else 1905.0, "T_metal_K": last_tap["metal_T_K"] if last_tap else 1910.0,
                    "furnace_power_MW": acc["power_mw"] / n * 0.98, "T_feed_K": sim.dist.feed_temperature_k}
            phys = static_balance(r, mats_d, cond, DEFAULT_PRICES)
            crm, crs = tap["cr_in_metal_kg"], tap["cr_in_slag_kg"]
            rows.append({
                "t_h": tap["t_h"], "power_mw": acc["power_mw"] / n, "coke_kg_t": r.get("COKE_MET", 0), "coal_kg_t": r.get("COAL_LA", 0),
                "quartz_kg_t": r.get("QUARTZITE", 0), "frac_ore_A": bl.get("ORE_A_LUMPY", 0) / sb, "frac_ore_B": bl.get("ORE_B_PELLET", 0) / sb,
                "frac_ore_C": bl.get("ORE_C_FRIABLE", 0) / sb, "fines_pct": fines, "moisture_extra_pct": sim.dist.moisture_extra_pct,
                "feed_T_K": sim.dist.feed_temperature_k, "coke_bed_fill": acc["coke_bed_fill"] / n, "electrode_depth_m": acc["depth"] / n,
                "burden_dp_kpa": acc["burden_dp_kpa"] / n, "offgas_co_pct": acc["offgas_co_dry_vol_pct"] / n,
                "tap_interval_h": sim.sp.tap_interval_h, "tap_T_K": tap["metal_T_K"],
                "sec_actual": tap["sec_kwh_per_t"], "cr_rec_actual": 100 * crm / (crm + crs), "slag_cr2o3_actual": tap["slag_cr2o3_equiv_wt_pct"],
                "metal_si_actual": tap["metal_wt_pct"].get("Si", 0), "metal_cr_actual": tap["metal_wt_pct"].get("Cr", 0),
                "physics_sec": phys["energy"]["sec_kwh_per_t_fecr"], "physics_cr_rec": phys["cr_recovery_pct"],
                "physics_slag_cr2o3": phys["slag_cr2o3_equiv_wt_pct"], "synthetic": True})
            last_tap = tap
            acc = {"n": 0}
        if progress and i % 500 == 0:
            progress(i / n_steps)
    return pd.DataFrame(rows)


def _metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_target(df: pd.DataFrame, target: str, kind: str, seed: int = 0) -> dict:
    df = F.add_lags(df)
    tr, va, te = F.chrono_split(df)
    Xtr, Xva, Xte = F.matrix(tr), F.matrix(va), F.matrix(te)
    ytr, yva, yte = tr[target].to_numpy(), va[target].to_numpy(), te[target].to_numpy()
    best = None
    for hp in HP_GRID[kind]:
        m = make_model(kind, seed, **hp).fit(Xtr, ytr)
        s = _metrics(yva, m.predict(Xva))["RMSE"]
        if best is None or s < best[0]:
            best = (s, hp)
    hp = best[1]
    model = make_model(kind, seed, **hp).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    # split-conformal style interval from validation residuals of the train-only model
    m_tr = make_model(kind, seed, **hp).fit(Xtr, ytr)
    q90 = float(np.quantile(np.abs(yva - m_tr.predict(Xva)), 0.9))
    pte = model.predict(Xte)
    a, p = F.TARGETS[target]
    phys_te = te[p].to_numpy()
    act_te = te[a].to_numpy()
    cover = float(np.mean(np.abs(yte - pte) <= q90))
    ood = OODDetector(np.vstack([Xtr, Xva]))
    return {
        "model": model, "kind": kind, "target": target, "hyperparameters": hp,
        "metrics": {"validation_RMSE_best": best[0], "test_residual": _metrics(yte, pte),
                    "test_physics_only": _metrics(act_te, phys_te), "test_hybrid": _metrics(act_te, phys_te + pte),
                    "interval_90_halfwidth": q90, "interval_90_test_coverage": cover},
        "split": {"train": [float(tr.t_h.min()), float(tr.t_h.max()), len(tr)], "val": [float(va.t_h.min()), float(va.t_h.max()), len(va)],
                  "test": [float(te.t_h.min()), float(te.t_h.max()), len(te)], "method": "chronological 60/20/20, no shuffling"},
        "ood": ood.to_dict(), "importance": importance(model, Xte, yte, F.FEATURES, seed)[:10],
        "drift_test_vs_train": drift_report(Xtr, Xte, F.FEATURES),
        "x_ref": np.vstack([Xtr, Xva]).mean(axis=0).tolist(),
        "test_series": {"t_h": te.t_h.tolist(), "actual": act_te.tolist(), "physics": phys_te.tolist(), "hybrid": (phys_te + pte).tolist(),
                        "lo": (phys_te + pte - q90).tolist(), "hi": (phys_te + pte + q90).tolist()},
    }


def manifest(df: pd.DataFrame, target: str, kind: str, seed: int) -> dict:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
    return {"dataset_sha256_16": h, "rows": len(df), "features": F.FEATURES, "target": target, "kind": kind, "seed": seed,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "synthetic": True,
            "quality_filters": "drop first tap (no lag), drop rows with missing lab lag", "data_lineage": "app/data/demo/campaign_taps.csv"}
