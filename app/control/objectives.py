"""MPC objective (normalised, dimensionless):
J = sum_k sum_cv w_cv ((y_k - y_ref)/scale)^2 + w_move sum_i (du_i/scale_i)^2
    + w_sec * SEC_avg/1000 + w_cr * (100 - CrRec_avg)/10 + w_co2 * CO2_avg + slack' S slack
Hard MV bounds/rates are optimisation bounds; CV soft limits become slack-penalised; hard CV
limits are checked on the nonlinear verification run (safety gate)."""
from __future__ import annotations

import numpy as np


def objective(pred: dict, du: np.ndarray, mv_scales: np.ndarray, cfg, cv_defs) -> tuple[float, dict]:
    J = 0.0
    slacks = {}
    for cv in cv_defs:
        y = pred["traj"].get(cv.name)
        if y is None:
            continue
        y = np.asarray(y)
        if cv.target is not None and cv.weight > 0:
            J += cv.weight * float(np.mean(((y - cv.target) / cv.scale) ** 2))
        s = 0.0
        if cv.soft_lo is not None:
            s += float(np.sum(np.maximum(cv.soft_lo - y, 0) / cv.scale))
        if cv.soft_hi is not None:
            s += float(np.sum(np.maximum(y - cv.soft_hi, 0) / cv.scale))
        if s > 0:
            slacks[cv.name] = s
            J += cfg.slack_penalty * s / len(y)
    J += cfg.w_move * float(np.sum((du / mv_scales) ** 2))
    J += cfg.w_sec * pred["sec_avg"] / 1000.0
    J += cfg.w_cr_loss * (100.0 - pred["cr_rec_avg"]) / 10.0
    J += cfg.w_co2 * pred.get("co2_t_per_t", 0.0)
    return J, slacks
