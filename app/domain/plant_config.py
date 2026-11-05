"""Versioned, editable plant specification with full parameter metadata.

Every parameter carries: name, symbol, description, units, physical bounds, nominal value,
provenance, uncertainty, revision, timestamp, source system, reviewer, effective date.
Values here are DEMO-ONLY example priors for a synthetic "Industry-X" plant; they are not
design limits for any real furnace. Capacity presets scale geometry and electrical design
with documented engineering rules-of-thumb (CALIBRATION-REQUIRED per site).
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

NOW = datetime(2026, 10, 29, tzinfo=timezone.utc).isoformat()


@dataclass
class Param:
    name: str
    symbol: str
    description: str
    units: str
    lo: float
    hi: float
    value: Any
    provenance: str = "DEMO-ONLY"
    uncertainty: str = "+/-20 % (example prior)"
    revision: int = 1
    timestamp: str = NOW
    source_system: str = "synthetic-config"
    reviewer: str = "unreviewed"
    effective_date: str = NOW[:10]
    group: str = "general"

    def validate(self) -> list[str]:
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            if not math.isfinite(self.value):
                return [f"{self.name}: non-finite"]
            if not (self.lo <= self.value <= self.hi):
                return [f"{self.name}={self.value} {self.units} outside [{self.lo}, {self.hi}]"]
        return []


CAPACITY_PRESETS = {
    # MVA : electrode dia (m), nominal line voltage (V), hearth depth to burden top (m)
    9.0: dict(d_e=0.90, burden_h=3.4), 16.5: dict(d_e=1.10, burden_h=3.9), 30.0: dict(d_e=1.40, burden_h=4.6),
    45.0: dict(d_e=1.55, burden_h=5.1), 60.0: dict(d_e=1.70, burden_h=5.5), 81.0: dict(d_e=1.90, burden_h=6.0),
}

REFRACTORY_MATERIALS = {
    # k(T) = k0 + k1 (T - 300)  W/m/K ; rho kg/m3 ; cp J/kg/K ; max service T (K)
    "carbon_block": dict(k0=14.0, k1=-0.004, rho=1600, cp=1250, t_max=3000, label="Carbon/semi-graphite block"),
    "ramming_paste": dict(k0=5.0, k1=0.0, rho=1500, cp=1200, t_max=3000, label="Carbon ramming paste"),
    "magnesia_brick": dict(k0=12.0, k1=-0.0055, rho=2950, cp=1150, t_max=2300, label="Magnesia (MgO) brick"),
    "chrome_magnesite": dict(k0=4.0, k1=-0.0012, rho=3050, cp=1000, t_max=2100, label="Magnesia-chrome brick"),
    "high_alumina": dict(k0=2.2, k1=-0.0002, rho=2600, cp=1050, t_max=2000, label="High-alumina (70 % Al2O3)"),
    "fireclay": dict(k0=1.0, k1=0.0002, rho=2100, cp=1000, t_max=1700, label="Fireclay brick"),
    "insulating": dict(k0=0.25, k1=0.0002, rho=800, cp=1000, t_max=1500, label="Insulating brick"),
    "castable": dict(k0=1.6, k1=0.0, rho=2400, cp=1000, t_max=1900, label="Alumina castable"),
    "steel": dict(k0=48.0, k1=-0.02, rho=7850, cp=500, t_max=900, label="Steel shell"),
    "frozen_slag": dict(k0=1.5, k1=0.0, rho=3000, cp=1100, t_max=2100, label="Frozen slag (freeze lining)"),
}


def default_lining() -> dict:
    return {
        "metal_band": [("carbon_block", 0.50), ("ramming_paste", 0.06), ("fireclay", 0.115), ("steel", 0.032)],
        "slag_band": [("magnesia_brick", 0.46), ("high_alumina", 0.23), ("insulating", 0.115), ("steel", 0.032)],
        "burden_band": [("high_alumina", 0.35), ("fireclay", 0.23), ("steel", 0.025)],
        "hearth": [("carbon_block", 1.00), ("magnesia_brick", 0.46), ("fireclay", 0.30), ("steel", 0.040)],
        "roof": [("castable", 0.15), ("steel", 0.020)],
    }


def derive_geometry(mva: float, overrides: dict | None = None) -> dict:
    """Scale geometry/electrical design from transformer rating (rules of thumb, CALIBRATION-REQUIRED)."""
    keys = sorted(CAPACITY_PRESETS)
    m = min(keys, key=lambda k: abs(k - mva))
    pre = CAPACITY_PRESETS[m]
    s = (mva / m) ** 0.3
    d_e = pre["d_e"] * s
    p_mw = 0.80 * mva  # nominal active power ~ 0.8 x rating
    v_line = 68.0 * p_mw ** (1 / 3)  # Westly-type: V proportional to P^(1/3)
    pcd = 2.6 * d_e
    d_hearth = pcd + 2.2 * d_e + 3.0
    g = dict(
        transformer_mva=mva, electrode_diameter_m=round(d_e, 3), pitch_circle_diameter_m=round(pcd, 3),
        hearth_inner_diameter_m=round(d_hearth, 2), burden_height_m=pre["burden_h"],
        metal_bath_design_depth_m=0.6, slag_layer_design_depth_m=0.9, freeboard_height_m=1.6,
        nominal_power_mw=round(p_mw, 2), nominal_line_voltage_v=round(v_line, 1),
        tap_count=19, tap_step_v=round(v_line * 0.022, 2),
        current_density_limit_a_per_cm2=6.5, power_factor_design=0.82,
    )
    g["tap_min_v"] = round(v_line - 10 * g["tap_step_v"], 1)
    g["max_phase_current_ka"] = round(g["current_density_limit_a_per_cm2"] * math.pi * (100 * d_e) ** 2 / 4 / 1000, 1)
    if overrides:
        g.update(overrides)
    return g


def build_default_config(mva: float = 45.0, enclosure: str = "closed", overrides: dict | None = None) -> dict:
    g = derive_geometry(mva, overrides)
    P = []
    add = P.append
    G = "furnace"
    add(Param("furnace_id", "-", "Furnace identifier", "-", 0, 0, "Industry-X FCF-1", group=G))
    add(Param("site_timezone", "-", "Site timezone (UTC stored, local displayed)", "-", 0, 0, "Asia/Kolkata", group=G))
    add(Param("furnace_type", "-", "Electrical configuration", "-", 0, 0, "AC-3-electrode-SAF", group=G))
    add(Param("enclosure", "-", "Roof enclosure: open | semi-closed | closed", "-", 0, 0, enclosure, group=G))
    add(Param("transformer_mva", "S_T", "Furnace transformer rating", "MVA", 3, 120, g["transformer_mva"], group="electrical"))
    add(Param("nominal_power_mw", "P_nom", "Nominal active furnace power", "MW", 2, 100, g["nominal_power_mw"], group="electrical"))
    add(Param("nominal_line_voltage_v", "V_LL", "Nominal secondary line voltage", "V", 60, 600, g["nominal_line_voltage_v"], group="electrical"))
    add(Param("tap_count", "N_tap", "Number of on-load tap positions", "-", 3, 40, g["tap_count"], group="electrical"))
    add(Param("tap_step_v", "dV_tap", "Line-voltage step per tap", "V", 0.5, 20, g["tap_step_v"], group="electrical"))
    add(Param("tap_min_v", "V_tap,min", "Line voltage at tap 1", "V", 40, 500, g["tap_min_v"], group="electrical"))
    add(Param("tap_min_dwell_s", "t_dwell", "Minimum dwell between tap changes", "s", 30, 3600, 300, group="electrical"))
    add(Param("max_phase_current_ka", "I_max", "Electrode current limit (current-density based)", "kA", 10, 250, g["max_phase_current_ka"], group="electrical"))
    add(Param("power_factor_design", "PF_d", "Design power factor at nominal load", "-", 0.5, 0.98, g["power_factor_design"], group="electrical"))
    add(Param("short_network_resistance_mohm", "R_net", "Busbar/flexible/electrode-column resistance per phase", "mOhm", 0.01, 0.5, 0.06, group="electrical"))
    add(Param("electrode_diameter_m", "D_e", "Soderberg electrode diameter", "m", 0.5, 2.2, g["electrode_diameter_m"], group="electrodes"))
    add(Param("pitch_circle_diameter_m", "PCD", "Electrode pitch-circle diameter", "m", 1, 7, g["pitch_circle_diameter_m"], group="electrodes"))
    add(Param("electrode_tip_depth_nominal_m", "h_tip", "Nominal tip depth below burden surface", "m", 0.3, 4.0, round(0.55 * g["burden_height_m"], 2), group="electrodes"))
    add(Param("electrode_tip_depth_min_m", "h_tip,min", "Minimum allowed tip depth (immersion)", "m", 0.1, 3.0, round(0.30 * g["burden_height_m"], 2), group="electrodes"))
    add(Param("electrode_tip_depth_max_m", "h_tip,max", "Maximum tip depth (hearth clearance)", "m", 0.5, 5.0, round(0.80 * g["burden_height_m"], 2), group="electrodes"))
    add(Param("electrode_max_speed_m_per_min", "v_e", "Electrode hoist max speed", "m/min", 0.05, 2.0, 0.40, group="electrodes"))
    add(Param("electrode_paste_kg_per_mwh", "e_paste", "Electrode paste consumption", "kg/MWh", 1, 10, 3.0, group="electrodes", provenance="CALIBRATION-REQUIRED"))
    add(Param("slip_length_mm", "l_slip", "Electrode slip length per event", "mm", 5, 40, 15, group="electrodes"))
    add(Param("hearth_inner_diameter_m", "D_h", "Hearth inner diameter", "m", 3, 20, g["hearth_inner_diameter_m"], group="geometry"))
    add(Param("burden_height_m", "H_b", "Burden column height above slag", "m", 1.5, 9, g["burden_height_m"], group="geometry"))
    add(Param("metal_bath_design_depth_m", "H_m", "Metal bath design depth (after tap heel)", "m", 0.2, 1.5, g["metal_bath_design_depth_m"], group="geometry"))
    add(Param("slag_layer_design_depth_m", "H_s", "Slag layer design depth", "m", 0.3, 2.0, g["slag_layer_design_depth_m"], group="geometry"))
    add(Param("freeboard_height_m", "H_fb", "Freeboard height under roof", "m", 0.5, 4, g["freeboard_height_m"], group="geometry"))
    add(Param("ambient_temperature_k", "T_amb", "Ambient air temperature", "K", 263, 323, 303.15, group="thermal"))
    add(Param("shell_emissivity", "eps_sh", "Shell surface emissivity", "-", 0.3, 0.95, 0.80, group="thermal"))
    add(Param("sidewall_spray_cooling", "-", "Shell water-spray cooling on metal/slag band", "-", 0, 0, True, group="thermal"))
    add(Param("spray_htc_w_m2k", "h_spray", "Shell spray-cooling heat-transfer coefficient", "W/m2/K", 50, 3000, 600, group="thermal", provenance="CALIBRATION-REQUIRED"))
    add(Param("hearth_air_htc_w_m2k", "h_air,b", "Under-hearth forced-air HTC", "W/m2/K", 5, 200, 35, group="thermal"))
    add(Param("roof_water_htc_w_m2k", "h_roof", "Roof panel cooling-water HTC", "W/m2/K", 100, 5000, 1500, group="thermal"))
    add(Param("cooling_water_inlet_k", "T_cw,in", "Cooling-water inlet temperature", "K", 280, 320, 305.15, group="thermal"))
    add(Param("refractory_lining", "-", "Lining layup per wall region (material, thickness m)", "-", 0, 0, default_lining(), group="refractory"))
    add(Param("tap_interval_h", "t_tap", "Scheduled interval between taps", "h", 0.5, 8, 2.0, group="tapping"))
    add(Param("tap_fraction_metal", "phi_m", "Fraction of metal above heel removed per tap", "-", 0.3, 1.0, 0.85, group="tapping"))
    add(Param("tap_fraction_slag", "phi_s", "Fraction of slag above heel removed per tap", "-", 0.3, 1.0, 0.80, group="tapping"))
    add(Param("lab_delay_min", "t_lab", "Laboratory reporting delay (metal/slag assays)", "min", 5, 240, 45, group="tapping"))
    add(Param("air_infiltration_nm3_per_nm3", "lambda_air", "Air ingress per Nm3 furnace gas (open/semi-closed)", "Nm3/Nm3",
              0, 60, {"closed": 0.02, "semi-closed": 1.5, "open": 25.0}[enclosure], group="offgas"))
    add(Param("co_alarm_ppm", "-", "Workplace CO advisory threshold at roof seal (example)", "ppm", 10, 500, 50, group="offgas"))
    add(Param("analyzer_lag_s", "tau_an", "Gas analyzer first-order lag", "s", 5, 600, 90, group="offgas"))
    add(Param("target_cr_wt_pct", "-", "Product Cr target (min)", "wt%", 45, 75, 52.0, group="quality"))
    add(Param("max_si_wt_pct", "-", "Product Si maximum", "wt%", 0.5, 8, 4.0, group="quality"))
    add(Param("max_c_wt_pct", "-", "Product C maximum", "wt%", 4, 10, 8.5, group="quality"))
    add(Param("max_p_wt_pct", "-", "Product P maximum", "wt%", 0.01, 0.1, 0.035, group="quality"))
    cfg = {"revision": 1, "created_at": NOW, "synthetic": True, "params": {p.name: asdict(p) for p in P},
           "geometry_derivation": "app.domain.plant_config.derive_geometry (rules of thumb, CALIBRATION-REQUIRED)"}
    return cfg


def value(cfg: dict, name: str):
    return cfg["params"][name]["value"]


def flat(cfg: dict) -> dict:
    return {k: v["value"] for k, v in cfg["params"].items()}


def validate_config(cfg: dict) -> list[str]:
    errs = []
    for k, d in cfg["params"].items():
        p = Param(**d)
        errs += p.validate()
    f = flat(cfg)
    if f["electrode_tip_depth_min_m"] >= f["electrode_tip_depth_max_m"]:
        errs.append("electrode_tip_depth_min_m must be < electrode_tip_depth_max_m")
    if f["pitch_circle_diameter_m"] + f["electrode_diameter_m"] >= f["hearth_inner_diameter_m"]:
        errs.append("electrodes do not fit inside hearth (PCD + D_e >= D_hearth)")
    if f["enclosure"] not in ("open", "semi-closed", "closed"):
        errs.append("enclosure must be open | semi-closed | closed")
    for region, layers in f["refractory_lining"].items():
        for mat, th in layers:
            if mat not in REFRACTORY_MATERIALS:
                errs.append(f"{region}: unknown material {mat}")
            if not (0.001 <= th <= 3.0):
                errs.append(f"{region}: layer thickness {th} m out of range")
    return errs


def update_config(cfg: dict, changes: dict, reviewer: str = "unreviewed") -> dict:
    new = copy.deepcopy(cfg)
    for k, v in changes.items():
        if k not in new["params"]:
            raise KeyError(f"unknown parameter {k}")
        new["params"][k]["value"] = v
        new["params"][k]["revision"] += 1
        new["params"][k]["timestamp"] = datetime.now(timezone.utc).isoformat()
        new["params"][k]["reviewer"] = reviewer
    new["revision"] = cfg["revision"] + 1
    errs = validate_config(new)
    if errs:
        raise ValueError("; ".join(errs))
    return new


def capacity_table() -> list[dict]:
    return [derive_geometry(m) for m in sorted(CAPACITY_PRESETS)]
