"""Reduced-order three-phase AC submerged-arc-furnace circuit model.

Topology: three single-phase (or one three-phase) transformer secondary windings in star,
each feeding one Soderberg electrode through the short network (R_net + jX). The furnace
hearth/burden forms a floating star point (neutral not bonded to the transformer star), so
phase currents are found with Millman's theorem using complex phasors:

    V_n = sum(V_k Y_k) / sum(Y_k),  I_k = (V_k - V_n) Y_k,  Y_k = 1 / (R_net + R_load,k + j X_k)

Per-phase active power into the furnace = |I_k|^2 R_load,k (real phase-wise power, not a DC
formula). Network loss = |I_k|^2 R_net. Reactive power = |I_k|^2 X_k.

Load resistance per phase couples electrode immersion and burden/coke-bed state:
    R_cb   = rho_cb(T, coke-bed fill) / (2 pi r_e (1 + h/r_e))        tip path via coke bed -> slag
    R_bur  = rho_bur / (2 pi h) * ln(PCD / D_e)                       side path through burden
    R_load = R_cb || R_bur
Current through the tip path is split into coke-bed and slag heating by a slag-path fraction.
The coke-bed resistivity prior rho_cb0 is CALIBRATED at commissioning so the nominal
operating point reproduces the design resistance (see calibrate()).
References for structure (not parameter values): Westly (1975) design rules; Dhainaut (2004);
Tesfahunegn et al., Metall. Mater. Trans. B 52 (2021) metamodelling of SAF electrical conditions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SQRT3 = math.sqrt(3.0)
A_PH = np.array([1.0, np.exp(-2j * np.pi / 3), np.exp(2j * np.pi / 3)])


@dataclass
class ElectricalDesign:
    v_tap_min: float
    tap_step: float
    tap_count: int
    r_net: float  # ohm
    x_phase: np.ndarray  # ohm (3,)
    d_e: float
    pcd: float
    i_max: float  # A
    s_rated: float  # VA
    rho_cb0: float = 3.0e-3  # ohm m (calibrated)
    rho_bur: float = 0.25  # ohm m
    slag_path_frac: float = 0.55
    arc_frac: float = 0.05

    def v_line(self, tap: int) -> float:
        tap = int(min(max(tap, 1), self.tap_count))
        return self.v_tap_min + (tap - 1) * self.tap_step


def rho_coke_bed(rho0: float, T_cb: float, fill: float, coke_size_factor: float = 1.0) -> float:
    """Effective coke-bed resistivity: falls with temperature and coke-bed fill (carbon inventory)."""
    fT = math.exp(1500.0 * (1.0 / T_cb - 1.0 / 1950.0))  # mild NTC behaviour of coke beds
    ff = 1.0 / max(fill, 0.15) ** 0.8
    return rho0 * fT * ff * coke_size_factor


def phase_load(design: ElectricalDesign, h_tip: float, rho_cb: float, rho_bur: float) -> dict:
    r_e = design.d_e / 2
    h = max(h_tip, 0.05)
    R_cb = rho_cb / (2 * math.pi * r_e * (1.0 + h / r_e))
    R_bur = rho_bur / (2 * math.pi * h) * math.log(design.pcd / design.d_e)
    R_load = R_cb * R_bur / (R_cb + R_bur)
    f_tip = R_bur / (R_cb + R_bur)  # current fraction through tip path
    return dict(R_cb=R_cb, R_bur=R_bur, R_load=R_load, f_tip_current=f_tip)


def solve_circuit(design: ElectricalDesign, tap: int, R_load: np.ndarray) -> dict:
    VLL = design.v_line(tap)
    Vph = VLL / SQRT3 * A_PH
    Z = design.r_net + R_load + 1j * design.x_phase
    Y = 1.0 / Z
    Vn = np.sum(Vph * Y) / np.sum(Y)
    I = (Vph - Vn) * Y
    Iabs = np.abs(I)
    P_load = Iabs ** 2 * R_load
    P_net = Iabs ** 2 * design.r_net
    Q = Iabs ** 2 * design.x_phase
    S_ph = (Vph - Vn) * np.conj(I)  # complex power into each branch (load side of transformer star)
    S_tr = Vph * np.conj(I)
    P_tot = float(np.sum(P_load + P_net))
    S_abs = float(np.abs(np.sum(S_tr)))
    pf = P_tot / math.sqrt(P_tot ** 2 + float(np.sum(Q)) ** 2) if P_tot > 0 else 0.0
    i_mean = float(np.mean(Iabs))
    unbalance = float((np.max(Iabs) - np.min(Iabs)) / i_mean * 100) if i_mean > 0 else 0.0
    # negative-sequence current ratio (IEC style unbalance indicator)
    a = np.exp(2j * np.pi / 3)
    I2 = (I[0] + a * a * I[1] + a * I[2]) / 3
    I1 = (I[0] + a * I[1] + a * a * I[2]) / 3
    return dict(
        tap=int(tap), v_line_V=VLL, v_phase_V=(np.abs(Vph - Vn)).tolist(), neutral_shift_V=float(abs(Vn)),
        i_phase_A=Iabs.tolist(), i_angle_deg=np.degrees(np.angle(I)).tolist(),
        p_load_W=P_load.tolist(), p_network_W=P_net.tolist(), q_var=Q.tolist(),
        p_total_W=P_tot, p_furnace_W=float(np.sum(P_load)), q_total_var=float(np.sum(Q)), s_VA=S_abs,
        power_factor=pf, current_unbalance_pct=unbalance,
        negative_sequence_pct=float(abs(I2) / abs(I1) * 100) if abs(I1) > 0 else 0.0,
        r_load_mohm=(R_load * 1e3).tolist(), x_mohm=(design.x_phase * 1e3).tolist(),
        phase_s_VA=np.abs(S_ph).tolist(), phase_pf=(np.real(S_tr) / np.maximum(np.abs(S_tr), 1)).tolist(),
        electrical_efficiency=float(np.sum(P_load)) / P_tot if P_tot > 0 else 0.0,
    )


def design_from_config(f: dict) -> ElectricalDesign:
    P_ph = f["nominal_power_mw"] * 1e6 / 3
    V_ph = f["nominal_line_voltage_v"] / SQRT3
    pf = f["power_factor_design"]
    R_tot = V_ph ** 2 * pf ** 2 / P_ph
    X = R_tot * math.tan(math.acos(pf))
    return ElectricalDesign(
        v_tap_min=f["tap_min_v"], tap_step=f["tap_step_v"], tap_count=int(f["tap_count"]),
        r_net=f["short_network_resistance_mohm"] * 1e-3, x_phase=np.array([X, X * 1.03, X * 0.97]),
        d_e=f["electrode_diameter_m"], pcd=f["pitch_circle_diameter_m"], i_max=f["max_phase_current_ka"] * 1e3,
        s_rated=f["transformer_mva"] * 1e6)


def nominal_resistance(f: dict) -> float:
    P_ph = f["nominal_power_mw"] * 1e6 / 3
    V_ph = f["nominal_line_voltage_v"] / SQRT3
    pf = f["power_factor_design"]
    return V_ph ** 2 * pf ** 2 / P_ph - f["short_network_resistance_mohm"] * 1e-3


def calibrate(design: ElectricalDesign, f: dict, T_cb: float = 1950.0, fill: float = 1.0) -> float:
    """Find rho_cb0 so the nominal tip depth reproduces the design load resistance."""
    R_target = nominal_resistance(f)
    h = f["electrode_tip_depth_nominal_m"]
    lo, hi = 1e-5, 1.0
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        rho = rho_coke_bed(mid, T_cb, fill)
        R = phase_load(design, h, rho, design.rho_bur)["R_load"]
        if R > R_target:
            hi = mid
        else:
            lo = mid
    design.rho_cb0 = math.sqrt(lo * hi)
    return design.rho_cb0


@dataclass
class ElectrodeState:
    tip_depth_m: np.ndarray
    holder_m: np.ndarray = field(default_factory=lambda: np.full(3, 1.2))  # holder travel (0 = top stop, 2.4 = bottom stop)
    slips: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    unslipped_burn_m: np.ndarray = field(default_factory=lambda: np.zeros(3))


def regulator_step(design: ElectricalDesign, es: ElectrodeState, R_meas: np.ndarray, R_sp: np.ndarray, I_meas: np.ndarray,
                   dt: float, v_max_m_s: float, h_min: float, h_max: float, gain: float = 0.6,
                   consumption_m_s: np.ndarray | None = None, slip_len_m: float = 0.015) -> list[str]:
    """Existing-PLC electrode regulator model (impedance control with current limiting).

    Modelled to reproduce plant behaviour; the twin never commands the real regulator.
    Tip burn-off is compensated by holder travel; when the burnt length since the last slip
    reaches the slip length a (procedural) slip event is logged and the holder regrips.
    """
    events = []
    burn = consumption_m_s * dt if consumption_m_s is not None else np.zeros(3)
    err = (R_meas - R_sp) / np.maximum(R_sp, 1e-9)
    v = np.clip(gain * err, -1, 1) * v_max_m_s  # R too high -> lower electrode (deeper)
    over = I_meas > design.i_max
    v = np.where(over, -v_max_m_s, v)  # current limiting raises electrode
    new_h = np.clip(es.tip_depth_m + v * dt, h_min, h_max)
    es.holder_m = es.holder_m + (new_h - es.tip_depth_m) + burn  # holder moves down to compensate burn-off
    es.tip_depth_m = new_h
    es.unslipped_burn_m = es.unslipped_burn_m + burn
    for k in range(3):
        if es.unslipped_burn_m[k] >= slip_len_m:
            es.unslipped_burn_m[k] -= slip_len_m
            es.holder_m[k] -= slip_len_m
            es.slips[k] += 1
            events.append(f"electrode {k + 1} slip #{int(es.slips[k])}")
        es.holder_m[k] = float(np.clip(es.holder_m[k], 0.0, 2.4))
    return events
