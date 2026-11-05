"""Reproducible synthetic scenarios A-L (DEMO-ONLY; values are not plant design limits).

Each scenario lists disclosed assumptions, a timed action schedule, expected qualitative
behaviour and the automated checks applied in tests/integration/test_scenarios.py.
"""
from __future__ import annotations

import copy

from ..domain.material_streams import DEFAULT_RECIPE

LIFECYCLE_STATES = {
    "cold_startup": {"models": "energy/refractory heat-up only", "mpc": False},
    "warm_startup": {"models": "full physics, wide uncertainty", "mpc": False},
    "normal_steady_smelting": {"models": "full physics + hybrid ML", "mpc": True},
    "ore_recipe_transition": {"models": "full physics; ML flagged OOD-sensitive", "mpc": True},
    "power_curtailment": {"models": "full physics", "mpc": True},
    "transformer_electrode_abnormality": {"models": "physics; electrical advisories", "mpc": False},
    "burden_instability": {"models": "physics; permeability advisories", "mpc": False},
    "tapping": {"models": "physics with tap impulse", "mpc": False},
    "recovery": {"models": "full physics", "mpc": True},
    "maintenance": {"models": "none (frozen)", "mpc": False},
    "emergency_fallback": {"models": "frozen last-good", "mpc": False},
}


def _r(**kw):
    r = copy.deepcopy(DEFAULT_RECIPE)
    for k, v in kw.items():
        r[k] = v
    return r


SCENARIOS: dict[str, dict] = {
    "A": dict(title="Nominal stable HC FeCr, 3-electrode AC", duration_h=24, events=[],
              assumptions="45 MVA closed furnace, 55/45 lumpy/pellet blend, nominal tap, level-controlled charging",
              expected="Quasi-steady temperatures, metal ~63-65 % Cr, slag Cr2O3-eq ~8-10 %, closure residuals at round-off"),
    "B": dict(title="Lower-grade chromite (higher gangue, lower Cr:Fe)", duration_h=24,
              events=[{"t_h": 2, "action": "set_recipe", "params": {"ore_blend": {"ORE_C_FRIABLE": 0.6, "ORE_A_LUMPY": 0.4}}}],
              assumptions="Ore C (Cr2O3 38 %, SiO2 16 %) replaces pellets", expected="Lower metal Cr %, more slag, higher SEC, higher Cr loss"),
    "C": dict(title="Elevated ore moisture and pellet degradation / fines", duration_h=24,
              events=[{"t_h": 2, "action": "disturb", "params": {"moisture_extra_pct": 4.0, "fines_extra_pct": 15.0}}],
              assumptions="+4 % free moisture, +15 % fines on ore", expected="Top-gas cools, permeability falls, dust and Cr-to-dust rise"),
    "D": dict(title="Higher-ash / lower fixed-C reductant, higher reactivity", duration_h=24,
              events=[{"t_h": 2, "action": "set_recipe", "params": {"COKE_MET": 0.0, "COKE_LOWQ": 205.0}},
                      {"t_h": 2, "action": "disturb", "params": {"coke_reactivity_factor": 1.8}}],
              assumptions="Coke replaced by 75 % FC, 21 % ash coke", expected="More slag, coke-bed carbon falls, electrodes go deeper, Cr recovery drops"),
    "E": dict(title="Quartzite flux step (slag liquidus/viscosity shift)", duration_h=24,
              events=[{"t_h": 2, "action": "set_recipe", "params": {"QUARTZITE": 135.0}}],
              assumptions="+50 kg quartzite / t ore", expected="Lower slag liquidus, lower viscosity, more slag, Si pick-up rises"),
    "F": dict(title="Lower pre-reduction / hot-to-cold feed transition", duration_h=24,
              initial_recipe=_r(ore_blend={"ORE_A_LUMPY": 0.55, "ORE_B_PREREDUCED": 0.45}, COKE_MET=135.0),
              initial_disturb={"feed_temperature_k": 773.15},
              events=[{"t_h": 3, "action": "set_recipe", "params": {"ore_blend": {"ORE_A_LUMPY": 0.55, "ORE_B_PELLET": 0.45}, "COKE_MET": 172.0}},
                      {"t_h": 3, "action": "disturb", "params": {"feed_temperature_k": 303.15}}],
              assumptions="Pre-reduced (Fe 80 %, Cr 45 %) pellets charged hot at 500 °C, then cold unreduced pellets",
              expected="SEC rises markedly after transition; production falls at constant power"),
    "G": dict(title="Transformer tap / power reduction (tariff curtailment)", duration_h=24,
              events=[{"t_h": 2, "action": "set_tap", "params": {"tap": -5}}, {"t_h": 10, "action": "set_tap", "params": {"tap": +5}}],
              assumptions="5 taps down for 8 h (peak tariff), then restored", expected="Power drops ~20 %, temperatures and production fall, then recover"),
    "H": dict(title="Electrical phase unbalance / regulator disturbance (advisory only)", duration_h=12,
              events=[{"t_h": 2, "action": "disturb", "params": {"rho_cb_phase_factor": [0.55, 1.0, 1.0], "stuck_electrode": 0}},
                      {"t_h": 7, "action": "disturb", "params": {"rho_cb_phase_factor": [1.0, 1.0, 1.0], "stuck_electrode": -1}}],
              assumptions="Phase 1 coke-bed resistivity drops 45 % and electrode-1 hoist fault (regulator cannot correct)", expected="Current unbalance alarm, neutral shift, power redistribution"),
    "I": dict(title="Burden permeability decline / gas-flow anomaly and CO warning", duration_h=12,
              events=[{"t_h": 2, "action": "disturb", "params": {"fines_extra_pct": 35.0}}],
              assumptions="Severe fines generation", expected="Permeability index < 0.8, burden-instability state, CO-leak risk advisory"),
    "J": dict(title="Delayed tapping, delayed assay, inventory uncertainty", duration_h=16,
              events=[{"t_h": 1, "action": "set_tap_interval", "params": {"h": 4.0}},
                      {"t_h": 1, "action": "disturb", "params": {"lab_extra_delay_min": 90.0}}],
              assumptions="Tap every 4 h instead of 2 h, lab +90 min", expected="Metal/slag inventories rise, estimator uncertainty grows between assays"),
    "K": dict(title="Sensor bias, analyzer stuck-value, missing tags, delayed lab", duration_h=12,
              events=[{"t_h": 2, "action": "disturb", "params": {"analyzer_stuck": True, "thermocouple_bias_k": 40.0,
                                                                "missing_tags": ["GAS_FLOW_DRY", "TC_SHELL_SLAG"], "lab_extra_delay_min": 30.0}}],
              assumptions="Analyzer freezes, +40 K TC bias, two tags lost", expected="Data-quality events, recommendation issuance disabled"),
    "L": dict(title="Out-of-distribution operation forcing ML/MPC safe fallback", duration_h=12,
              events=[{"t_h": 1, "action": "set_recipe", "params": {"ore_blend": {"ORE_C_FRIABLE": 1.0}, "QUARTZITE": 220.0}},
                      {"t_h": 1, "action": "disturb", "params": {"ood_flag": True}}],
              assumptions="100 % low-grade ore and very high flux (outside training domain)", expected="OOD detector trips, hybrid correction disabled, MPC falls back"),
}


def resolve_events(events: list[dict], base_tap: int) -> list[dict]:
    """Relative tap moves (negative/positive ints in set_tap with sign) are resolved against the base tap."""
    out = []
    cur = base_tap
    for e in events:
        e = copy.deepcopy(e)
        if e["action"] == "set_tap":
            cur = cur + int(e["params"]["tap"])
            e["params"]["tap"] = cur
        out.append(e)
    return out


def scenario_table() -> list[dict]:
    return [dict(id=k, **{kk: vv for kk, vv in v.items() if kk not in ("initial_recipe",)}) for k, v in SCENARIOS.items()]
