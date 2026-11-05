"""Hybrid prediction = physics + ML residual, with OOD fallback and physical clipping."""
from __future__ import annotations

import numpy as np

from . import features as F
from .drift import OODDetector
from .explain import local_reasons

CLIP = {"sec_residual": (-800.0, 800.0), "cr_rec_residual": (-15.0, 15.0), "slag_cr2o3_residual": (-6.0, 6.0)}
PHYS_BOUNDS = {"sec_residual": (1500.0, 7000.0), "cr_rec_residual": (40.0, 99.5), "slag_cr2o3_residual": (0.2, 40.0)}


def hybrid_predict(registry, target: str, x: dict, physics_value: float, force_ood: bool = False) -> dict:
    act = registry.active(target)
    if act is None:
        return {"value": physics_value, "physics": physics_value, "correction": 0.0, "mode": "physics-only (no active model)"}
    mid, meta = act
    xv = np.array([float(x.get(f, np.nan)) for f in F.FEATURES])
    if np.any(~np.isfinite(xv)):
        return {"value": physics_value, "physics": physics_value, "correction": 0.0, "mode": "physics-only (missing features)",
                "model_id": mid}
    ood = OODDetector.from_dict(meta["ood"])
    d = ood.distance(xv)
    if force_ood or d > ood.threshold:
        return {"value": physics_value, "physics": physics_value, "correction": 0.0, "mode": "physics-only (OOD fallback)",
                "ood_distance": d, "ood_threshold": ood.threshold, "model_id": mid}
    model = registry.load(mid)
    corr = float(model.predict(xv[None, :])[0])
    lo, hi = CLIP[target]
    corr_c = float(np.clip(corr, lo, hi))
    v = float(np.clip(physics_value + corr_c, *PHYS_BOUNDS[target]))
    hw = meta["metrics"]["interval_90_halfwidth"]
    return {"value": v, "physics": physics_value, "correction": corr_c, "raw_correction": corr, "clipped": corr != corr_c,
            "interval_90": [v - hw, v + hw], "mode": "hybrid", "ood_distance": d, "ood_threshold": ood.threshold, "model_id": mid,
            "reasons": local_reasons(model, xv, np.array(meta["x_ref"]), F.FEATURES)}
