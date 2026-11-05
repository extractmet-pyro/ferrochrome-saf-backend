"""Explainability: permutation importance and per-prediction linear attribution."""
from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


def importance(model, X, y, names, seed=0) -> list[dict]:
    r = permutation_importance(model, X, y, n_repeats=8, random_state=seed)
    order = np.argsort(-r.importances_mean)
    return [{"feature": names[i], "importance": float(r.importances_mean[i]), "sd": float(r.importances_std[i])} for i in order]


def local_reasons(model, x: np.ndarray, x_ref: np.ndarray, names: list[str], top: int = 4) -> list[dict]:
    """Occlusion attribution: change in prediction when each feature is replaced by the training mean."""
    base = float(model.predict(x[None, :])[0])
    out = []
    for i, n in enumerate(names):
        z = x.copy()
        z[i] = x_ref[i]
        out.append({"feature": n, "contribution": base - float(model.predict(z[None, :])[0])})
    out.sort(key=lambda d: -abs(d["contribution"]))
    return out[:top]
