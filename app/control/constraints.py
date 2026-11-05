"""Manipulated / controlled variable definitions and constraint sets for the supervisory MPC.

Hierarchy: the MPC sits ABOVE the existing PLC/DCS electrode regulator, transformer tap
changer and gas-pressure loops. Emergency trips, hard interlocks and human authority are
independent and always take precedence. Values are DEMO-ONLY defaults; site values are set
at commissioning (HAZOP/LOPA, FAT/SAT) and must not be assumed universally safe.
Protected / procedural functions that are NEVER manipulated variables: electrode slipping and
baking, breakers, gas safety & ventilation, cooling-water systems, refractory protection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MV:
    name: str
    units: str
    lo: float
    hi: float
    max_move: float  # per MPC execution
    step_for_id: float
    scale: float
    discrete: bool = False
    description: str = ""


@dataclass
class CV:
    name: str
    units: str
    target: float | None = None
    soft_lo: float | None = None
    soft_hi: float | None = None
    hard_lo: float | None = None
    hard_hi: float | None = None
    scale: float = 1.0
    weight: float = 0.0
    description: str = ""


@dataclass
class MPCConfig:
    horizon_h: float = 6.0
    sample_h: float = 0.5
    sim_dt_s: float = 300.0
    sim_substeps: int = 6
    timeout_s: float = 25.0
    w_sec: float = 1.0
    w_cr_loss: float = 1.0
    w_co2: float = 0.0
    w_move: float = 0.3
    slack_penalty: float = 200.0
    advisory_only: bool = True
    mvs: list = field(default_factory=list)
    cvs: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def default_mpc_config(f: dict, sp) -> MPCConfig:
    R0 = sp.r_setpoint_mohm
    mvs = [
        MV("tap", "-", 1, f["tap_count"], 1, 1, 1, True, "Transformer tap (discrete; dwell lockout; via tap changer PLC)"),
        MV("r_setpoint_mohm", "mOhm", 0.6 * R0, 1.4 * R0, 0.06 * R0, 0.05 * R0, 0.05 * R0, False,
           "Electrode regulator resistance setpoint (existing regulator executes)"),
        MV("coke_kg_t", "kg/t ore", 130, 230, 10, 10, 10, False, "Coke addition per t ore (batching recipe)"),
        MV("quartz_kg_t", "kg/t ore", 40, 160, 20, 20, 20, False, "Quartzite flux per t ore (batching recipe)"),
    ]
    cvs = [
        CV("T_slag_K", "K", target=1905, soft_lo=1870, soft_hi=1960, scale=20, weight=1.0, description="slag temperature band"),
        CV("slag_superheat_K", "K", soft_lo=40, hard_lo=0, scale=10, description="liquidus margin"),
        CV("si_wt_pct", "wt%", target=3.5, soft_hi=f["max_si_wt_pct"], scale=0.5, weight=0.6),
        CV("c_wt_pct", "wt%", soft_hi=f["max_c_wt_pct"], scale=0.3),
        CV("cr_wt_pct", "wt%", soft_lo=f["target_cr_wt_pct"], scale=1.0),
        CV("coke_bed_fill", "-", target=1.0, soft_lo=0.7, soft_hi=1.35, scale=0.1, weight=0.8, description="carbon balance"),
        CV("power_mw", "MW", target=f["nominal_power_mw"], scale=0.03 * f["nominal_power_mw"], weight=0.5),
        CV("i_max_ka", "kA", hard_hi=f["max_phase_current_ka"], scale=2.0, description="electrode current limit"),
    ]
    return MPCConfig(mvs=mvs, cvs=cvs)
