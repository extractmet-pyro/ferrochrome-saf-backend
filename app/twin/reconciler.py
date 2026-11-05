"""Data reconciliation of chromium / iron / mass balances and gross-error detection.

Weighted least squares with linear balance constraints A x = 0:
    x_hat = x - S A^T (A S A^T)^-1 A x ,   S = diag(sigma^2)
Global test: gamma = r^T (A S A^T)^-1 r ~ chi2(m) with r = A x (balance residuals).
Individual measurement tests use normalised adjustments |x_hat_i - x_i| / sqrt(V_ii).
Used for the Cr balance audit over a window (feed Cr, metal Cr, slag Cr, dust Cr, inventory).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def reconcile(values: dict[str, float], sigmas: dict[str, float], A_rows: list[dict[str, float]]) -> dict:
    keys = list(values)
    x = np.array([values[k] for k in keys], dtype=float)
    S = np.diag([max(sigmas[k], 1e-9) ** 2 for k in keys])
    A = np.array([[row.get(k, 0.0) for k in keys] for row in A_rows], dtype=float)
    r = A @ x
    M = A @ S @ A.T
    Minv = np.linalg.pinv(M)
    adj = S @ A.T @ Minv @ r
    xh = x - adj
    V = S @ A.T @ Minv @ A @ S
    gamma = float(r @ Minv @ r)
    m = A.shape[0]
    crit = float(stats.chi2.ppf(0.95, m))
    z = {k: float(abs(adj[i]) / np.sqrt(V[i, i])) if V[i, i] > 0 else 0.0 for i, k in enumerate(keys)}
    return {"measured": dict(zip(keys, x.tolist())), "reconciled": dict(zip(keys, xh.tolist())),
            "adjustments": dict(zip(keys, (-adj).tolist())), "balance_residuals_before": r.tolist(),
            "global_test": {"gamma": gamma, "chi2_95": crit, "gross_error_suspected": gamma > crit},
            "normalised_adjustments": z, "suspect": [k for k, v in z.items() if v > 1.96]}


def cr_balance_window(sim, hours: float = 24.0) -> dict:
    """Reconcile the Cr balance over the last `hours` using tap records, feed and dust."""
    from ..domain import species as sp
    led = sim.ledger
    ci = sp.ELEMENTS.index("Cr")
    feed = float(sp.element_mass_kg(led.n_in.get("charge", np.zeros(sp.NS)))[ci])
    metal = float(sp.element_mass_kg(led.n_out.get("tapped_metal", np.zeros(sp.NS)))[ci])
    slag = float(sp.element_mass_kg(led.n_out.get("tapped_slag", np.zeros(sp.NS)))[ci])
    dust = float(sp.element_mass_kg(led.n_out.get("dust", np.zeros(sp.NS)))[ci])
    inv = float(sp.element_mass_kg(sim.total_inventory())[ci] - sp.element_mass_kg(led.inv0)[ci])
    rng = np.random.default_rng(3)
    meas = {"feed_Cr": feed * (1 + rng.normal(0, 0.01)), "metal_Cr": metal * (1 + rng.normal(0, 0.005)),
            "slag_Cr": slag * (1 + rng.normal(0, 0.04)), "dust_Cr": dust * (1 + rng.normal(0, 0.15)), "inventory_dCr": inv}
    sig = {"feed_Cr": 0.015 * feed + 1, "metal_Cr": 0.006 * metal + 1, "slag_Cr": 0.05 * slag + 1, "dust_Cr": 0.2 * dust + 1,
           "inventory_dCr": 0.03 * abs(inv) + 0.02 * metal + 50}
    res = reconcile(meas, sig, [{"feed_Cr": 1, "metal_Cr": -1, "slag_Cr": -1, "dust_Cr": -1, "inventory_dCr": -1}])
    res["true_values_synthetic"] = {"feed_Cr": feed, "metal_Cr": metal, "slag_Cr": slag, "dust_Cr": dust, "inventory_dCr": inv}
    res["window"] = "since ledger start" if hours <= 0 else f"ledger window (session)"
    res["sigma"] = sig
    return res
