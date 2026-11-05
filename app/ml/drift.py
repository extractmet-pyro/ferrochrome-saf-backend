"""Out-of-distribution (Mahalanobis) detection and concept/data drift (PSI)."""
from __future__ import annotations

import numpy as np


class OODDetector:
    def __init__(self, X: np.ndarray, q: float = 0.99):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.cov_inv = np.linalg.pinv(np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1]))
        d = np.sqrt(np.einsum("ij,jk,ik->i", Z, self.cov_inv, Z))
        self.threshold = float(np.quantile(d, q) * 1.25)

    def distance(self, x: np.ndarray) -> float:
        z = (np.asarray(x, dtype=float) - self.mu) / self.sd
        return float(np.sqrt(z @ self.cov_inv @ z))

    def is_ood(self, x: np.ndarray) -> bool:
        return self.distance(x) > self.threshold

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "cov_inv": self.cov_inv.tolist(), "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d):
        o = cls.__new__(cls)
        o.mu, o.sd, o.cov_inv, o.threshold = np.array(d["mu"]), np.array(d["sd"]), np.array(d["cov_inv"]), d["threshold"]
        return o


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, qs)[0] / max(len(expected), 1) + 1e-4
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / max(len(actual), 1) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_train: np.ndarray, X_recent: np.ndarray, names: list[str]) -> dict:
    vals = {n: psi(X_train[:, i], X_recent[:, i]) for i, n in enumerate(names)}
    return {"psi": vals, "max_psi": max(vals.values()) if vals else 0.0,
            "drifted": [n for n, v in vals.items() if v > 0.25], "rule": "PSI > 0.25 = significant drift; 0.1-0.25 = moderate"}
