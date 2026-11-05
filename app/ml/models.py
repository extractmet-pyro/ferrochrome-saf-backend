"""Model zoo: ridge baseline and gradient-boosted trees (residual correction to physics)."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, seed: int = 0, **hp):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=hp.get("alpha", 3.0)))
    if kind == "gbr":
        return GradientBoostingRegressor(n_estimators=hp.get("n_estimators", 250), max_depth=hp.get("max_depth", 3),
                                         learning_rate=hp.get("learning_rate", 0.05), subsample=0.8, random_state=seed)
    raise ValueError(f"unknown model kind {kind}")


HP_GRID = {"ridge": [{"alpha": a} for a in (0.3, 1.0, 3.0, 10.0, 30.0)],
           "gbr": [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05} for n in (150, 300) for d in (2, 3)]}
