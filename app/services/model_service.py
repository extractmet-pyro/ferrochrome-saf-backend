"""ML lifecycle: training jobs (chronological), registration (shadow), authorised activation,
metrics and a demo bootstrap that trains on the shipped SYNTHETIC campaign."""
from __future__ import annotations

import os

import pandas as pd

from ..core.config import settings
from ..ml.registry import ModelRegistry
from ..ml.train import manifest, train_target

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo", "campaign_taps.csv")
TARGETS = ["sec_residual", "cr_rec_residual", "slag_cr2o3_residual"]

registry = ModelRegistry(settings.model_dir)


def load_campaign() -> pd.DataFrame:
    return pd.read_csv(DATA)


def train(target: str, kind: str, seed: int = 0, progress=None) -> dict:
    df = load_campaign()
    if progress:
        progress(0.1)
    tr = train_target(df, target, kind, seed)
    mid = registry.register(tr, manifest(df, target, kind, seed))
    if progress:
        progress(1.0)
    return {"model_id": mid, "metrics": tr["metrics"], "status": "shadow"}


def bootstrap():
    """If no model is active for a target, train ridge+gbr, activate the better test-hybrid RMSE (demo only)."""
    if not os.path.exists(DATA):
        return
    for t in TARGETS:
        if registry.active(t):
            continue
        best = None
        for kind in ("ridge", "gbr"):
            r = train(t, kind)
            rm = r["metrics"]["test_hybrid"]["RMSE"]
            if best is None or rm < best[1]:
                best = (r["model_id"], rm)
        registry.activate(best[0], "system-bootstrap (demo)")
