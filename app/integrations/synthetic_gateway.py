"""Synthetic read-only telemetry gateway: maps twin truth to instrument tags with realistic
noise, bias, lag, quantisation, missing data and quality codes.

NO live equipment connection exists in this module. A plant deployment replaces it with the
read-only OPC UA adapter (opcua_readonly_adapter.py) running on an on-prem gateway.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TagDef:
    tag: str
    description: str
    units: str
    lo: float
    hi: float
    sigma: float  # measurement standard deviation
    rate_s: float
    source: str  # DCS | PLC | ANALYZER | LAB | CALC
    path: str  # dotted path into the simulator snapshot (or callable key)
    delay_s: float = 0.0
    permission: str = "read"
    outlier_policy: str = "reject >5 sigma, hold last good"
    fallback: str = "hold last good, flag STALE after 3 samples"
    calibration: str = "synthetic - none"


def _p(s: dict, path: str):
    cur = s
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


TAGS: list[TagDef] = [
    TagDef("EL_P_TOTAL", "Total active power (transformer secondary)", "MW", 0, 120, 0.05, 1, "DCS", "power_mw"),
    TagDef("EL_P_FURNACE", "Active power into furnace (calc.)", "MW", 0, 120, 0.05, 1, "CALC", "furnace_power_mw"),
    TagDef("EL_PF", "Power factor", "-", 0, 1, 0.003, 1, "DCS", "electrical.power_factor"),
    TagDef("EL_S", "Apparent power", "MVA", 0, 150, 0.08, 1, "DCS", "electrical.s_VA", calibration="scaled 1e-6"),
    TagDef("EL_Q", "Reactive power", "Mvar", 0, 120, 0.08, 1, "DCS", "electrical.q_total_var", calibration="scaled 1e-6"),
    TagDef("EL_TAP", "Transformer tap position", "-", 1, 40, 0.0, 1, "PLC", "electrical.tap"),
    TagDef("EL_VLL", "Secondary line voltage at tap", "V", 0, 600, 0.5, 1, "DCS", "electrical.v_line_V"),
    TagDef("EL_UNBAL", "Current unbalance", "%", 0, 100, 0.2, 1, "CALC", "electrical.current_unbalance_pct"),
    TagDef("EL_NEG_SEQ", "Negative-sequence current", "%", 0, 100, 0.2, 1, "CALC", "electrical.negative_sequence_pct"),
    TagDef("EL_NSHIFT", "Furnace star-point shift voltage", "V", 0, 200, 0.5, 1, "CALC", "electrical.neutral_shift_V"),
]
for _k in range(3):
    TAGS += [
        TagDef(f"EL_I{_k + 1}", f"Electrode {_k + 1} RMS current", "kA", 0, 250, 0.3, 1, "DCS", f"electrical.i_phase_A.{_k}", calibration="scaled 1e-3"),
        TagDef(f"EL_V{_k + 1}", f"Electrode {_k + 1} voltage to hearth", "V", 0, 400, 0.6, 1, "DCS", f"electrical.v_phase_V.{_k}"),
        TagDef(f"EL_R{_k + 1}", f"Electrode {_k + 1} load resistance", "mOhm", 0, 10, 0.01, 1, "CALC", f"electrical.r_load_mohm.{_k}"),
        TagDef(f"EL_P{_k + 1}", f"Electrode {_k + 1} active power", "MW", 0, 50, 0.05, 1, "DCS", f"electrical.p_load_W.{_k}", calibration="scaled 1e-6"),
        TagDef(f"ELD_POS{_k + 1}", f"Electrode {_k + 1} holder position", "m", 0, 2.4, 0.005, 1, "PLC", f"electrical.holder_m.{_k}"),
        TagDef(f"ELD_TIP{_k + 1}", f"Electrode {_k + 1} estimated tip depth", "m", 0, 6, 0.05, 10, "CALC", f"electrical.tip_depth_m.{_k}"),
        TagDef(f"ELD_SLIP{_k + 1}", f"Electrode {_k + 1} slip counter", "-", 0, 1e6, 0.0, 10, "PLC", f"electrical.slips.{_k}"),
    ]
TAGS += [
    TagDef("TC_OFFGAS", "Off-gas temperature (duct)", "K", 273, 1800, 4.0, 5, "DCS", "offgas_T_K"),
    TagDef("TC_SHELL_METAL", "Shell thermocouple, metal band", "K", 273, 900, 1.5, 10, "DCS", "walls.metal_band.shell_T_K"),
    TagDef("TC_SHELL_SLAG", "Shell thermocouple, slag band", "K", 273, 900, 1.5, 10, "DCS", "walls.slag_band.shell_T_K"),
    TagDef("TC_SHELL_BURDEN", "Shell thermocouple, burden band", "K", 273, 900, 1.5, 10, "DCS", "walls.burden_lower.shell_T_K"),
    TagDef("TC_HEARTH_BOTTOM", "Hearth bottom plate thermocouple", "K", 273, 900, 1.5, 10, "DCS", "walls.hearth.shell_T_K"),
    TagDef("TC_HEARTH_HOTFACE", "Hearth carbon hot-face (embedded TC est.)", "K", 273, 2500, 6.0, 10, "DCS", "walls.hearth.hot_face_T_K"),
    TagDef("TC_ROOF", "Roof panel temperature", "K", 273, 900, 1.5, 10, "DCS", "walls.roof.shell_T_K"),
    TagDef("CW_HEAT", "Cooling water heat removal", "MW", 0, 20, 0.02, 10, "CALC", "cooling_water_heat_MW"),
    TagDef("CW_DT", "Cooling water temperature rise", "K", 0, 40, 0.1, 10, "DCS", "cooling_water_dT_K"),
    TagDef("GAS_FLOW_DRY", "Off-gas flow (dry, Nm3/h)", "Nm3/h", 0, 2e5, 150, 5, "DCS", "dry_offgas_nm3_per_h"),
    TagDef("GAS_CO", "Off-gas CO (dry, analyzer)", "vol%", 0, 100, 0.4, 5, "ANALYZER", "analyzer_dry_vol_pct.CO", delay_s=90),
    TagDef("GAS_CO2", "Off-gas CO2 (dry, analyzer)", "vol%", 0, 100, 0.3, 5, "ANALYZER", "analyzer_dry_vol_pct.CO2", delay_s=90),
    TagDef("GAS_H2", "Off-gas H2 (dry, analyzer)", "vol%", 0, 100, 0.3, 5, "ANALYZER", "analyzer_dry_vol_pct.H2", delay_s=90),
    TagDef("GAS_O2", "Off-gas O2 (dry, analyzer)", "vol%", 0, 25, 0.05, 5, "ANALYZER", "offgas_dry_vol_pct.O2", delay_s=60),
    TagDef("GAS_N2", "Off-gas N2 (dry, by difference)", "vol%", 0, 100, 0.5, 5, "CALC", "offgas_dry_vol_pct.N2"),
    TagDef("GAS_CH4", "Off-gas CH4 (dry)", "vol%", 0, 20, 0.05, 5, "ANALYZER", "offgas_dry_vol_pct.CH4", delay_s=90),
    TagDef("DUST_LOAD", "Dust loading to gas cleaning", "kg/h", 0, 5000, 5, 60, "CALC", "dust_kg_per_h"),
    TagDef("BURDEN_DP", "Burden pressure-drop proxy", "kPa", 0, 50, 0.05, 5, "CALC", "burden_dp_kpa"),
    TagDef("FEED_RATE", "Charge feed rate (weigh-feeders)", "t/h", 0, 200, 0.2, 10, "PLC", "feed_t_per_h"),
    TagDef("BURDEN_UPPER", "Upper burden inventory (level-derived)", "t", 0, 2000, 2.0, 30, "CALC", "burden_upper_t"),
    TagDef("METAL_TAP_T", "Metal tap temperature (pyrometer, at tap)", "K", 1500, 2300, 8.0, 3600, "LAB", "last_tap.metal_T_K"),
    TagDef("METAL_TAP_MASS", "Tapped metal mass (ladle weigh)", "t", 0, 200, 0.3, 3600, "PLC", "last_tap.metal_t"),
    TagDef("SLAG_TAP_MASS", "Tapped slag mass (pot count est.)", "t", 0, 400, 1.0, 3600, "PLC", "last_tap.slag_t"),
    TagDef("LAB_METAL_CR", "Lab metal Cr", "wt%", 0, 100, 0.25, 7200, "LAB", "latest_lab.metal.Cr", delay_s=2700),
    TagDef("LAB_METAL_C", "Lab metal C", "wt%", 0, 12, 0.05, 7200, "LAB", "latest_lab.metal.C", delay_s=2700),
    TagDef("LAB_METAL_SI", "Lab metal Si", "wt%", 0, 15, 0.05, 7200, "LAB", "latest_lab.metal.Si", delay_s=2700),
    TagDef("LAB_SLAG_CR2O3", "Lab slag Cr2O3 (total, XRF)", "wt%", 0, 40, 0.2, 7200, "LAB", "latest_lab.slag.Cr2O3", delay_s=2700),
    TagDef("LAB_SLAG_SIO2", "Lab slag SiO2", "wt%", 0, 60, 0.2, 7200, "LAB", "latest_lab.slag.SiO2", delay_s=2700),
    TagDef("LAB_SLAG_MGO", "Lab slag MgO", "wt%", 0, 60, 0.2, 7200, "LAB", "latest_lab.slag.MgO", delay_s=2700),
    TagDef("LAB_SLAG_AL2O3", "Lab slag Al2O3", "wt%", 0, 60, 0.2, 7200, "LAB", "latest_lab.slag.Al2O3", delay_s=2700),
]

SCALE = {"EL_S": 1e-6, "EL_Q": 1e-6, **{f"EL_I{k}": 1e-3 for k in (1, 2, 3)}, **{f"EL_P{k}": 1e-6 for k in (1, 2, 3)}}


class SyntheticGateway:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.last_good: dict[str, float] = {}
        self.stale_count: dict[str, int] = {}

    def read(self, snap: dict, dist: dict | None = None) -> dict:
        dist = dist or {}
        missing = set(dist.get("missing_tags", []))
        tc_bias = float(dist.get("thermocouple_bias_k", 0.0))
        out = {}
        for td in TAGS:
            raw = _p(snap, td.path)
            q = "GOOD"
            if td.tag in missing or raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
                self.stale_count[td.tag] = self.stale_count.get(td.tag, 0) + 1
                val = self.last_good.get(td.tag)
                q = "STALE" if self.stale_count[td.tag] <= 3 and val is not None else "BAD"
                out[td.tag] = {"value": val, "quality": q, "units": td.units, "source": td.source, "t_s": snap["t_s"]}
                continue
            self.stale_count[td.tag] = 0
            val = float(raw) * SCALE.get(td.tag, 1.0)
            if td.sigma > 0:
                val += self.rng.normal(0, td.sigma)
            if td.tag.startswith("TC_"):
                val += tc_bias
            if dist.get("analyzer_stuck") and td.source == "ANALYZER":
                q = "SUSPECT_STUCK"
            if td.lo - 3 * td.sigma <= val < td.lo or td.hi < val <= td.hi + 3 * td.sigma:
                val = min(max(val, td.lo), td.hi)  # noise excursion at a physical bound: clamp (documented)
            elif not (td.lo <= val <= td.hi):
                q = "OUT_OF_RANGE"
            self.last_good[td.tag] = val
            out[td.tag] = {"value": val, "quality": q, "units": td.units, "source": td.source, "t_s": snap["t_s"],
                           "delay_s": td.delay_s}
        return out


def tags_csv() -> str:
    buf = io.StringIO()
    fields = list(asdict(TAGS[0]).keys())
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for t in TAGS:
        w.writerow(asdict(t))
    return buf.getvalue()


def tag_table() -> list[dict]:
    return [asdict(t) for t in TAGS]
