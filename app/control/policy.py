"""Baseline operating policy (what a typical shift would do without the twin).

Rule set (DEMO-ONLY, documented operator practice): every 4 h adjust coke by +/-5 kg/t ore
when the mean electrode tip depth deviates from nominal by more than 0.4 m (deep electrodes
=> carbon-deficient coke bed => add coke; shallow => remove). Tap and resistance setpoints
held. Electrode regulator, tap-changer and gas loops are the existing PLC/DCS loops modelled
inside the twin.
"""
from __future__ import annotations

import numpy as np


class BaselinePolicy:
    name = "baseline operating policy"

    def __init__(self, period_h: float = 4.0, step: float = 5.0, deadband_m: float = 0.4):
        self.period_h, self.step, self.deadband = period_h, step, deadband_m
        self.last = -1e9

    def act(self, sim) -> dict | None:
        t = sim.t / 3600
        if t - self.last < self.period_h:
            return None
        self.last = t
        dev = float(np.mean(sim.electrodes.tip_depth_m)) - sim.f["electrode_tip_depth_nominal_m"]
        if abs(dev) < self.deadband:
            return None
        from .mpc import _coke_key
        key = _coke_key(sim)
        coke = sim.sp.recipe.get(key, 0.0) + (self.step if dev > 0 else -self.step)
        coke = float(np.clip(coke, 130, 230))
        sim.apply_action("set_recipe", {key: coke}, source="baseline_policy")
        return {"t_h": t, key: coke}
