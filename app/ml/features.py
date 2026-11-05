"""Feature engineering for hybrid residual models (per-tap granularity).

Features are computed ONLY from information available before the target tap is reported:
recipe and setpoints over the tap interval, interval-mean process signals, and lagged lab
results with their reporting delay respected (lag >= 1 tap). No look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["power_mw", "coke_kg_t", "coal_kg_t", "quartz_kg_t", "frac_ore_A", "frac_ore_B", "frac_ore_C", "fines_pct",
            "moisture_extra_pct", "feed_T_K", "coke_bed_fill", "electrode_depth_m", "burden_dp_kpa", "offgas_co_pct",
            "tap_interval_h", "lag1_slag_cr2o3", "lag1_metal_si", "lag1_tap_T_K", "physics_sec", "physics_cr_rec",
            "physics_slag_cr2o3"]
TARGETS = {"sec_residual": ("sec_actual", "physics_sec"), "cr_rec_residual": ("cr_rec_actual", "physics_cr_rec"),
           "slag_cr2o3_residual": ("slag_cr2o3_actual", "physics_slag_cr2o3")}


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("t_h").reset_index(drop=True)
    df["lag1_slag_cr2o3"] = df["slag_cr2o3_actual"].shift(1)
    df["lag1_metal_si"] = df["metal_si_actual"].shift(1)
    df["lag1_tap_T_K"] = df["tap_T_K"].shift(1)
    for k, (a, p) in TARGETS.items():
        df[k] = df[a] - df[p]
    return df.dropna(subset=["lag1_slag_cr2o3"]).reset_index(drop=True)


def chrono_split(df: pd.DataFrame, f_train=0.6, f_val=0.2):
    n = len(df)
    a, b = int(n * f_train), int(n * (f_train + f_val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)
