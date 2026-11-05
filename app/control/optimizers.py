"""Bounded-time SLSQP with multi-start, warm start and deterministic no-solution handling."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize


class _Timeout(Exception):
    pass


def solve(fun, x0, bounds, deadline: float, starts: list | None = None, maxiter: int = 80) -> dict:
    best = None
    tried = 0
    for xs in [x0] + (starts or []):
        if time.time() > deadline:
            break
        tried += 1

        def cb(_x):
            if time.time() > deadline:
                raise _Timeout()
        try:
            r = minimize(fun, np.clip(xs, [b[0] for b in bounds], [b[1] for b in bounds]), method="SLSQP", bounds=bounds,
                         callback=cb, options={"maxiter": maxiter, "ftol": 1e-7})
            cand = {"x": r.x, "fun": float(r.fun), "success": bool(r.success), "message": str(r.message), "nit": int(r.nit)}
        except _Timeout:
            cand = {"x": np.asarray(xs, float), "fun": float(fun(np.asarray(xs, float))), "success": False,
                    "message": "time budget exhausted", "nit": 0, "timed_out": True}
        if best is None or (cand["success"] and (not best["success"] or cand["fun"] < best["fun"])):
            best = cand
    if best is None:
        best = {"x": np.asarray(x0, float), "fun": float("nan"), "success": False, "message": "no start evaluated", "nit": 0,
                "timed_out": True}
    best["starts_tried"] = tried
    return best
