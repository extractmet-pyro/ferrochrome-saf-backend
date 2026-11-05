"""Dynamic first-principles SAF ferrochrome twin (lumped multi-zone, fixed-step with substeps).

Fidelity statement: a lumped 5-zone (+gas, +refractory FV) advisory-grade model. It does NOT
resolve 3-D current paths, burden flow fields or coke-bed geometry. A high-fidelity
FEM/DEM/CFD model may supply surrogate corrections offline (see ML/surrogate hooks); it is
never the real-time core.
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..domain import electrical as el
from ..domain import kinetics as kn
from ..domain import metal as mt
from ..domain import offgas as og
from ..domain import plant_config as pc
from ..domain import slag as sg
from ..domain import species as sp
from ..domain.dynamics import Zone, make_zone, react, remove_gas, transfer
from ..domain.material_streams import DEFAULT_RECIPE, default_materials, recipe_species_per_t_ore, RawMaterial
from ..domain.phase_inventory import Ledger
from ..domain.reactions import RXN
from ..domain.refractory import build_walls
from ..domain.thermodynamics import h_species, invert_temperature, zone_enthalpy, TemperatureInversionError

T_FEED = 303.15
ZONES = ["B1", "B2", "CB", "SL", "MB"]


def _nu(*keys: str) -> np.ndarray:
    return np.array([RXN[k].nu_vec for k in keys])


@dataclass
class Setpoints:
    """Supervisory setpoints (what the DCS/PLC would hold). The twin never writes them to a plant."""
    tap: int = 11
    r_setpoint_mohm: float = 0.88
    recipe: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_RECIPE))
    feed_mode: str = "level"  # level | fixed
    fixed_feed_t_per_h: float = 25.0
    tap_interval_h: float = 2.0


@dataclass
class Disturbances:
    rho_cb_phase_factor: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    coke_reactivity_factor: float = 1.0
    fines_extra_pct: float = 0.0
    moisture_extra_pct: float = 0.0
    feed_temperature_k: float = T_FEED
    analyzer_stuck: bool = False
    analyzer_co_bias: float = 0.0
    thermocouple_bias_k: float = 0.0
    missing_tags: list = field(default_factory=list)
    lab_extra_delay_min: float = 0.0
    regulator_gain_factor: float = 1.0
    stuck_electrode: int = -1  # hoist fault on electrode index (0..2); -1 = none
    ood_flag: bool = False


class SimulationDiverged(RuntimeError):
    pass


class FeCrSAFSimulator:
    def __init__(self, cfg: dict | None = None, materials: dict[str, RawMaterial] | None = None,
                 setpoints: Setpoints | None = None, seed: int = 42, dt_s: float = 60.0, substeps: int = 4,
                 spinup_h: float = 30.0, mva: float = 45.0, enclosure: str = "closed"):
        self.cfg = cfg or pc.build_default_config(mva, enclosure)
        self.f = pc.flat(self.cfg)
        self.mats = materials or default_materials()
        self.sp = setpoints or Setpoints()
        self.dist = Disturbances()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.dt = float(dt_s)
        self.nsub = int(substeps)
        self.t = 0.0
        self.k = copy.deepcopy(kn.K_DEFAULT)
        self.params = dict(grain_radius_m=2.5e-4, droplet_radius_m=6.0e-4, L_S=12.0, active_burden_fraction=0.35,
                           coke_bed_height_m=0.8, eps_hx_B2=0.88, eps_hx_B1=0.82, dust_frac_base=0.008,
                           combustion_eff={"closed": 0.95, "semi-closed": 0.85, "open": 0.95}[self.f["enclosure"]],
                           metal_heel_depth_m=0.15, slag_heel_depth_m=0.30)
        self._geometry()
        self.design = el.design_from_config(self.f)
        el.calibrate(self.design, self.f, T_cb=2050.0, fill=1.0)
        self.sp.r_setpoint_mohm = el.nominal_resistance(self.f) * 1e3
        self.sp.tap = int(round((self.f["nominal_line_voltage_v"] - self.f["tap_min_v"]) / self.f["tap_step_v"])) + 1
        self.sp.tap_interval_h = self.f["tap_interval_h"]
        self.electrodes = el.ElectrodeState(tip_depth_m=np.full(3, self.f["electrode_tip_depth_nominal_m"]))
        self.walls = build_walls(self.f)
        self.analyzer = og.Analyzer(tau_s=self.f["analyzer_lag_s"])
        self.events: list[dict] = []
        self.taps: list[dict] = []
        self.lab_queue: list[dict] = []
        self.lab_results: list[dict] = []
        self.history: deque = deque(maxlen=20000)
        self.last_tap_t = 0.0
        self.last_tap_change_t = -1e9
        self.E_elec_since_tap = 0.0
        self.scheduled: list[dict] = []
        self.status = "ok"
        self.warnings: deque = deque(maxlen=200)
        self.last_good = None
        self.recipe_change_t = -1e9
        self.diag: dict = {}
        self.fb_T = 900.0
        self._init_zones()
        self.z["B1"].set_T(530.0)
        self.z["B2"].set_T(1220.0)
        self._init_walls()
        self.ledger = None
        if spinup_h > 0:
            self.spinup(spinup_h)
        self.reset_ledger()

    # ------------------------------------------------------------------ geometry & init
    def _geometry(self):
        f = self.f
        r = f["hearth_inner_diameter_m"] / 2
        self.A_hearth = math.pi * r * r
        self.A_cb = math.pi * (f["pitch_circle_diameter_m"] / 2 + f["electrode_diameter_m"]) ** 2
        self.A_cb = min(self.A_cb, 0.8 * self.A_hearth)
        V_burden = self.A_hearth * f["burden_height_m"] * self.params["active_burden_fraction"]
        self.M_burden_ref = V_burden * 1700.0  # kg (bulk density of mixed charge ~1.7 t/m3)
        self.C_cb_ref = self.A_cb * self.params["coke_bed_height_m"] * 0.55 * 1000 / 0.012011 * 0.85  # mol C
        rho_s, rho_m = 2900.0, 6900.0
        self.M_slag_heel = self.A_hearth * self.params["slag_heel_depth_m"] * rho_s
        self.M_metal_heel = self.A_hearth * self.params["metal_heel_depth_m"] * rho_m
        self.UA = {"CB_SL": 350.0 * 2 * self.A_cb, "SL_MB": 250.0 * self.A_hearth, "CB_B2": 120.0 * self.A_cb,
                   "B2_B1": 15.0 * self.A_hearth, "B1_FB": 25.0 * self.A_hearth}
        P_nom = f["nominal_power_mw"]
        self.feed_design_kg_s = P_nom / 3.9 * 3.2 * 1000 / 3600  # charge (t/t FeCr ~3.2) per SEC estimate
        self.tau_B1 = 0.5 * self.M_burden_ref / self.feed_design_kg_s
        self.tau_B2 = 0.5 * self.M_burden_ref / (0.85 * self.feed_design_kg_s)
        self.M_cb_solids_ref = 0.9 * self.feed_design_kg_s * 1500.0

    def _charge_vector_per_kg(self) -> tuple[np.ndarray, dict]:
        rec = self.sp.recipe
        v, masses = recipe_species_per_t_ore(rec, self._mats_with_dist())
        tot = sum(masses.values())
        return v / tot, {k: m / tot for k, m in masses.items()}

    def _mats_with_dist(self) -> dict:
        if self.dist.moisture_extra_pct == 0 and self.dist.fines_extra_pct == 0:
            return self.mats
        m = copy.deepcopy(self.mats)
        for k, mat in m.items():
            if mat.category == "ore":
                mat.moisture_wet_wt_pct = min(mat.moisture_wet_wt_pct + self.dist.moisture_extra_pct, 25)
                mat.fines_pct = min(mat.fines_pct + self.dist.fines_extra_pct, 60)
        return m

    def _init_zones(self):
        v_kg, _ = self._charge_vector_per_kg()
        Mb = 0.5 * self.M_burden_ref
        n_b1 = v_kg * Mb
        n_b2 = v_kg * Mb
        n_b2[sp.IDX["H2O_l"]] = 0.0
        n_b2[sp.IDX["VOL"]] = 0.0
        caco3 = n_b2[sp.IDX["CaCO3"]]
        n_b2[sp.IDX["CaCO3"]] = 0
        n_b2[sp.IDX["CaO"]] += caco3
        mgco3 = n_b2[sp.IDX["MgCO3"]]
        n_b2[sp.IDX["MgCO3"]] = 0
        n_b2[sp.IDX["MgO"]] += mgco3
        n_cb = v_kg * self.M_cb_solids_ref
        n_cb[sp.IDX["H2O_l"]] = 0
        n_cb[sp.IDX["VOL"]] = 0
        n_cb[sp.IDX["CaCO3"]] = 0
        n_cb[sp.IDX["MgCO3"]] = 0
        n_cb[sp.IDX["C"]] = self.C_cb_ref
        Ms = self.M_slag_heel + 25000.0
        slag_wt = {"SiO2": 34.0, "Al2O3": 24.0, "MgO": 22.0, "CaO": 1.7, "Cr2O3": 0.4, "CrO": 0.4, "FeO": 0.5,
                   "MnO": 0.05, "TiO2": 1.0, "FeCr2O4": 4.0, "MgCr2O4": 8.0, "MgAl2O4": 3.5, "Cr": 0.25, "Fe": 0.1,
                   "C_m": 0.03, "Cr7C3": 0.07, "Si": 0.02}
        n_sl = sp.from_mass_kg({k: Ms * w / 100 for k, w in slag_wt.items()})
        Mm = self.M_metal_heel + 15000.0
        met = {"Cr": 54.0, "Fe": 24.5, "Si": 4.0, "Mn": 0.45, "P": 0.02, "S": 0.25, "C_m": 0.8, "Cr7C3": 13.5}
        n_mb = sp.from_mass_kg({k: Mm * w / sum(met.values()) for k, w in met.items()})
        self.z = {"B1": make_zone("B1", n_b1, 650.0), "B2": make_zone("B2", n_b2, 1250.0),
                  "CB": make_zone("CB", n_cb, 1950.0), "SL": make_zone("SL", n_sl, 1900.0),
                  "MB": make_zone("MB", n_mb, 1900.0)}
        m_sp = float(np.sum((n_sl * sp.MW)[sp.SPINEL_MASK]))
        self.N_particles = m_sp / (4500.0 * 4 / 3 * math.pi * self.params["grain_radius_m"] ** 3)
        self._update_liquidus()

    def _init_walls(self):
        for w in self.walls.values():
            Tz = self.z[w.zone].T if w.zone in self.z else 900.0
            w.init_steady(Tz, self._slag_liq if w.freeze_lining else None)

    def reset_ledger(self):
        self.ledger = Ledger(self.total_inventory(), self.total_H(), self.total_wall_E())
        self.ledger_t0 = self.t
        self.E_elec_cum = 0.0

    def total_inventory(self) -> np.ndarray:
        return sum((z.n for z in self.z.values()), np.zeros(sp.NS))

    def total_H(self) -> float:
        return sum(z.H for z in self.z.values())

    def total_wall_E(self) -> float:
        return sum(w.stored_J() for w in self.walls.values())

    # ------------------------------------------------------------------ chemistry helpers
    def slag_liquid_wt(self) -> dict:
        n = self.z["SL"].n
        m = n * sp.MW
        d = {sp.KEYS[i]: m[i] for i in range(sp.NS) if sp.SLAG_OXIDE_MASK[i] and m[i] > 0}
        tot = sum(d.values())
        return {k: 100 * v / tot for k, v in d.items()} if tot > 0 else {}

    def slag_total_wt(self, n: np.ndarray | None = None) -> dict:
        """Tap-slag style analysis: oxides + spinel (as oxides) + entrained metal (as metal), wt%."""
        n = self.z["SL"].n if n is None else n
        el_m = sp.element_mass_kg(n)
        oxide = {}
        # spinels re-expressed as constituent oxides for an XRF-like analysis
        conv = {"FeCr2O4": {"FeO": 1, "Cr2O3": 1}, "MgCr2O4": {"MgO": 1, "Cr2O3": 1}, "MgAl2O4": {"MgO": 1, "Al2O3": 1}}
        for i, k in enumerate(sp.KEYS):
            if n[i] <= 0:
                continue
            if sp.SLAG_OXIDE_MASK[i]:
                oxide[k] = oxide.get(k, 0) + n[i] * sp.MW[i]
            elif k in conv:
                for ox, c in conv[k].items():
                    oxide[ox] = oxide.get(ox, 0) + c * n[i] * sp.SPECIES[ox].molar_mass
        metal_kg = float(np.sum((n * sp.MW)[sp.METAL_MASK]))
        tot = sum(oxide.values()) + metal_kg
        res = {k: 100 * v / tot for k, v in oxide.items()} if tot > 0 else {}
        res["metal_droplets"] = 100 * metal_kg / tot if tot > 0 else 0
        _ = el_m
        return res

    def metal_wt(self, n: np.ndarray | None = None) -> dict:
        n = self.z["MB"].n if n is None else n
        nm = np.where(sp.METAL_MASK, n, 0.0)
        em = sp.element_mass_kg(nm)
        d = {e: float(em[i]) for i, e in enumerate(sp.ELEMENTS) if em[i] > 0}
        return mt.element_wt(d)

    def _update_liquidus(self):
        wt = self.slag_liquid_wt()
        self._slag_liq = (sg.liquidus_C(wt)[0] + 273.15) if wt else 1990.0
        mw = self.metal_wt()
        self._metal_liq = mt.liquidus_C(mw.get("Cr", 55), mw.get("C", 7), mw.get("Si", 2)) + 273.15 if mw else 1800.0
        for z in self.z.values():
            z.set_liquidus("slag", self._slag_liq, 180.0)
            z.set_liquidus("metal", self._metal_liq, 120.0)
            z.refresh_T()

    def _slag_activities(self) -> tuple[dict, dict]:
        n = self.z["SL"].n
        x = kn.mole_frac(n, sp.SLAG_OXIDE_MASK)
        a = {k: kn.GAMMA.get(k, 1.0) * x.get(k, 0.0) for k in kn.GAMMA}
        xm = kn.mole_frac(self.z["MB"].n, sp.kind_mask("metal"))
        am = {k: kn.GAMMA_METAL.get(k, 1.0) * xm.get(k, 0.0) for k in kn.GAMMA_METAL}
        return a, am

    # ------------------------------------------------------------------ electrical
    def electrical(self) -> dict:
        Tcb = self.z["CB"].T
        fill = self.z["CB"].n[sp.IDX["C"]] / self.C_cb_ref
        Rl, parts = [], []
        for k in range(3):
            rho = el.rho_coke_bed(self.design.rho_cb0, Tcb, fill) * self.dist.rho_cb_phase_factor[k]
            p = el.phase_load(self.design, float(self.electrodes.tip_depth_m[k]), rho, self.design.rho_bur)
            Rl.append(p["R_load"])
            parts.append(p)
        circ = el.solve_circuit(self.design, self.sp.tap, np.array(Rl))
        I = np.array(circ["i_phase_A"])
        P_cb = P_sl = P_bur = 0.0
        for k in range(3):
            p = parts[k]
            I_tip = I[k] * p["R_bur"] / (p["R_cb"] + p["R_bur"])
            I_side = I[k] - I_tip
            P_tip = I_tip ** 2 * p["R_cb"]
            P_sl += P_tip * self.design.slag_path_frac
            P_cb += P_tip * (1 - self.design.slag_path_frac)
            P_bur += I_side ** 2 * p["R_bur"]
        circ.update(P_cb_W=P_cb, P_slag_W=P_sl, P_burden_W=P_bur, coke_bed_fill=fill,
                    R_cb_mohm=[p["R_cb"] * 1e3 for p in parts], R_bur_mohm=[p["R_bur"] * 1e3 for p in parts],
                    tip_depth_m=self.electrodes.tip_depth_m.tolist(), holder_m=self.electrodes.holder_m.tolist(),
                    slips=self.electrodes.slips.tolist(), r_setpoint_mohm=self.sp.r_setpoint_mohm)
        return circ

    # ------------------------------------------------------------------ main step
    def step(self, n_steps: int = 1) -> dict:
        snap = None
        for _ in range(n_steps):
            snap = self._step_once()
        return snap

    def _step_once(self) -> dict:
        backup = self._light_state()
        try:
            snap = self._advance()
            self._check_validity()
            self.status = "ok"
            self.last_good = snap
            return snap
        except (SimulationDiverged, TemperatureInversionError, FloatingPointError, ValueError) as exc:
            self._restore(backup)
            self.status = "frozen_last_good"
            msg = f"t={self.t / 3600:.2f} h solver rejected step: {exc}; holding last-good prediction"
            self.warnings.append(msg)
            self.events.append({"t_h": self.t / 3600, "type": "solver_fallback", "msg": msg})
            self.t += self.dt
            if self.last_good is not None:
                lg = dict(self.last_good)
                lg["status"] = self.status
                lg["warnings"] = list(self.warnings)[-5:]
                return lg
            raise

    def _light_state(self):
        return copy.deepcopy((self.z, self.walls, self.electrodes, self.ledger, self.N_particles, self.t,
                              self.E_elec_since_tap, self.E_elec_cum if hasattr(self, "E_elec_cum") else 0.0))

    def _restore(self, b):
        (self.z, self.walls, self.electrodes, self.ledger, self.N_particles, self.t, self.E_elec_since_tap,
         self.E_elec_cum) = b

    def _check_validity(self):
        for z in self.z.values():
            if not np.all(np.isfinite(z.n)) or not math.isfinite(z.H) or not math.isfinite(z.T):
                raise SimulationDiverged(f"non-finite state in {z.name}")
            if np.any(z.n < -1e-6):
                bad = sp.KEYS[int(np.argmin(z.n))]
                raise SimulationDiverged(f"negative inventory {bad} in {z.name}")
            z.n = np.maximum(z.n, 0.0)
            if not (250 < z.T < 3200):
                raise SimulationDiverged(f"temperature out of range in {z.name}: {z.T:.0f} K")

    def _advance(self) -> dict:
        dt = self.dt
        hs = dt / self.nsub
        self._apply_schedule()
        self._update_liquidus()
        ledger = self.ledger
        # --- walls (implicit, zone T held over the step); heat removed from zones over the step
        q_zone = {zn: 0.0 for zn in ZONES}
        wall_out = 0.0
        wall_parts = {"convection": 0.0, "radiation": 0.0, "forced_or_water": 0.0}
        wall_diag = {}
        for name, w in self.walls.items():
            if name == "roof":
                continue
            Tz = self.z[w.zone].T
            res = w.step(Tz, dt, self._slag_liq if w.freeze_lining else None)
            wall_diag[name] = res
            if w.zone in q_zone:
                q_zone[w.zone] += res["q_in_W"]
            wall_out += res["q_out_W"] * dt
            for k in wall_parts:
                wall_parts[k] += res["parts_W"][k]
        ledger.energy_out("wall_losses", wall_out)
        # --- electrical + regulator
        ecirc = self.electrical()
        P_cb, P_sl, P_bur = ecirc["P_cb_W"], ecirc["P_slag_W"], ecirc["P_burden_W"]
        P_furn = ecirc["p_furnace_W"]
        cons_kg_s = self.f["electrode_paste_kg_per_mwh"] * ecirc["p_total_W"] / 1e6 / 3600.0  # all phases
        A_e = math.pi * self.f["electrode_diameter_m"] ** 2 / 4
        burn = np.array(ecirc["p_load_W"]) / max(P_furn, 1) * cons_kg_s / (1550.0 * A_e)
        R_meas = np.array(ecirc["r_load_mohm"]) * 1e-3
        held = self.dist.stuck_electrode
        h_before = self.electrodes.tip_depth_m.copy()
        evs = el.regulator_step(self.design, self.electrodes, R_meas, np.full(3, self.sp.r_setpoint_mohm * 1e-3),
                                np.array(ecirc["i_phase_A"]), dt, self.f["electrode_max_speed_m_per_min"] / 60.0,
                                self.f["electrode_tip_depth_min_m"], self.f["electrode_tip_depth_max_m"],
                                gain=0.6 * self.dist.regulator_gain_factor, consumption_m_s=burn,
                                slip_len_m=self.f["slip_length_mm"] / 1000)
        if 0 <= held <= 2:
            self.electrodes.tip_depth_m[held] = h_before[held] - burn[held] * dt  # hoist fault: only burn-off moves the tip
        for e in evs:
            self.events.append({"t_h": self.t / 3600, "type": "electrode_slip", "msg": e})
        ledger.energy_in("electric_furnace", P_furn * dt)
        ledger.info("electric_network_loss", sum(ecirc["p_network_W"]) * dt)
        self.E_elec_since_tap += ecirc["p_total_W"] * dt
        self.E_elec_cum += ecirc["p_total_W"] * dt
        # --- feed (level control on upper burden)
        v_kg, mass_frac = self._charge_vector_per_kg()
        M_B1 = self.z["B1"].mass(~sp.GAS_MASK)
        M_B2 = self.z["B2"].mass(~sp.GAS_MASK)
        phi2 = float(np.clip(2.0 - M_B2 / (0.5 * self.M_burden_ref), 0.0, 1.5))
        desc1 = M_B1 / self.tau_B1 * phi2
        if self.sp.feed_mode == "level":
            feed = desc1 + 0.002 * (0.5 * self.M_burden_ref - M_B1)
        else:
            feed = self.sp.fixed_feed_t_per_h / 3.6
        feed = float(np.clip(feed, 0.0, 3.0 * self.feed_design_kg_s))
        n_feed = v_kg * feed * dt
        T_feed = self.dist.feed_temperature_k
        H_feed = float(n_feed @ h_species(T_feed))
        self.z["B1"].n += n_feed
        self.z["B1"].H += H_feed
        ledger.book_in("charge", n_feed, H_feed)
        n_el = sp.vec(C_el=cons_kg_s * dt / sp.SPECIES["C_el"].molar_mass)
        H_el = float(n_el @ h_species(T_FEED))
        self.z["CB"].n += n_el
        self.z["CB"].H += H_el
        ledger.book_in("electrode_carbon", n_el, H_el)
        # --- substeps
        gas_top = np.zeros(sp.NS)
        H_gas_top = 0.0
        dust_n = np.zeros(sp.NS)
        H_dust = 0.0
        xi_tot: dict[str, float] = {}
        diag_rates: dict[str, float] = {}
        q_b1_fb = 0.0
        for _ in range(self.nsub):
            g, Hg, dn_dust, Hd, xi, rates, qbf = self._substep(hs, P_cb, P_sl, P_bur, q_zone)
            gas_top += g
            H_gas_top += Hg
            dust_n += dn_dust
            H_dust += Hd
            q_b1_fb += qbf * hs
            for k, v in xi.items():
                xi_tot[k] = xi_tot.get(k, 0.0) + v
            diag_rates = rates
        # --- freeboard and off-gas
        air = self.f["air_infiltration_nm3_per_nm3"]
        roof = self.walls["roof"]
        G_roof = roof.inner_conductance(roof.T)
        fb = og.freeboard(gas_top / dt, H_gas_top / dt, dust_n / dt, H_dust / dt, air, self.f["ambient_temperature_k"],
                          G_roof, float(roof.T[0]), q_b1_fb / dt, self.params["combustion_eff"], self.fb_T)
        self.fb_T = fb["T"]
        rres = roof.step(self.fb_T, dt, None, q_in_prescribed=fb["q_roof_W"])
        wall_diag["roof"] = rres
        ledger.energy_out("wall_losses", rres["q_out_W"] * dt)
        for k in wall_parts:
            wall_parts[k] += rres["parts_W"][k]
        n_air = fb["n_air_rate"] * dt
        air_vec = sp.vec(O2=og.AIR_O2 * n_air, N2=(1 - og.AIR_O2) * n_air)
        ledger.book_in("infiltrated_air", air_vec, fb["H_air_rate"] * dt)
        H_exit = fb["H_rate"] * dt  # gas + dust leaving freeboard (incl. chemical enthalpy)
        ledger.book_out("offgas", fb["n_rate"] * dt + dust_n, H_exit)  # dust leaves with the gas stream
        ledger.n_out["dust"] = ledger.n_out["dust"] + dust_n  # reporting only (not double counted below)
        ledger.n_out["offgas"] = ledger.n_out["offgas"] - dust_n
        # --- tapping
        tap = None
        if self.t + dt - self.last_tap_t >= self.sp.tap_interval_h * 3600 - 1e-6:
            tap = self._tap()
        self.t += dt
        self._process_lab()
        return self._snapshot(ecirc, fb, gas_top / dt, dust_n / dt, wall_diag, wall_parts, xi_tot, diag_rates, tap,
                              mass_frac, feed, q_zone, q_b1_fb / dt)

    # ------------------------------------------------------------------ substep physics
    def _substep(self, hs: float, P_cb: float, P_sl: float, P_bur: float, q_zone: dict):
        z = self.z
        B1, B2, CB, SL, MB = z["B1"], z["B2"], z["CB"], z["SL"], z["MB"]
        xi_rec: dict[str, float] = {}
        rates: dict[str, float] = {}

        def rec(keys, xi):
            for k, v in zip(keys, xi):
                xi_rec[k] = xi_rec.get(k, 0.0) + float(v)
                rates[k] = float(v) / hs

        # 1) burden descent (consumption-driven)
        M_B2 = B2.mass(~sp.GAS_MASK)
        M_cbs = CB.mass(~(sp.GAS_MASK | sp.CARBON_MASK))
        phi_cb = float(np.clip(2.0 - M_cbs / self.M_cb_solids_ref, 0.0, 1.5))
        phi2 = float(np.clip(2.0 - M_B2 / (0.5 * self.M_burden_ref), 0.0, 1.5))
        transfer(B2, CB, np.where(sp.GAS_MASK, 0.0, B2.n) * min(hs / self.tau_B2 * phi_cb, 0.5))
        transfer(B1, B2, np.where(sp.GAS_MASK, 0.0, B1.n) * min(hs / self.tau_B1 * phi2, 0.5))

        # 2) coke bed: electrode carbon, solid-state reduction, melting
        T = CB.T
        fill = CB.n[sp.IDX["C"]] / self.C_cb_ref
        keys = ["R28_electrode", "R17_FeCr2O4_C", "R18_MgCr2O4_C", "R03_Cr2O3_C", "R02_FeO_C", "R01_Fe2O3_C",
                "R14_devol", "R13_drying", "R11_calcite", "R12_magnesite"]
        n = CB.n
        cC = min(1.0, n[sp.IDX["C"]] / max(4 * (n[sp.IDX["FeCr2O4"]] + n[sp.IDX["MgCr2O4"]]) + 1e-9, 1e-9))
        xi = np.array([
            kn.arrh(self.k["electrode"], T) * n[sp.IDX["C_el"]],
            kn.arrh(self.k["ss_FeCr2O4"], T) * n[sp.IDX["FeCr2O4"]] * cC * kn.driving_force("R17_FeCr2O4_C", T, 1, 1e-3),
            kn.arrh(self.k["ss_MgCr2O4"], T) * n[sp.IDX["MgCr2O4"]] * cC * kn.driving_force("R18_MgCr2O4_C", T, 1, 1e-3),
            kn.arrh(self.k["ss_Cr2O3"], T) * n[sp.IDX["Cr2O3"]] * cC * kn.driving_force("R03_Cr2O3_C", T, 1, 1e-3),
            kn.arrh(self.k["dir_FeO"], T) * n[sp.IDX["FeO"]],
            kn.arrh(self.k["dir_FeO"], T) * 3 * n[sp.IDX["Fe2O3"]],
            kn.arrh(self.k["devol"], T) * n[sp.IDX["VOL"]],
            n[sp.IDX["H2O_l"]] / 30.0,
            kn.arrh(self.k["calcite"], T) * n[sp.IDX["CaCO3"]],
            kn.arrh(self.k["magnesite"], T) * n[sp.IDX["MgCO3"]],
        ]) * hs
        nu = _nu(*keys)
        xi = kn.limit_extents(CB.n, nu, xi)
        react(CB, nu, xi)
        rec(keys, xi)
        CB.refresh_T()
        # melting / sinking of non-carbon solids into slag, metal trickles to slag as droplets
        melt_k = self.k["melt"].A * kn.sig((CB.T - (self._slag_liq + 60.0)) / 25.0)
        solids = np.where(sp.GAS_MASK | sp.CARBON_MASK | sp.METAL_MASK, 0.0, CB.n)
        dn_melt = solids * min(melt_k * hs, 0.5)
        m_sp_in = float(np.sum((dn_melt * sp.MW)[sp.SPINEL_MASK]))
        transfer(CB, SL, dn_melt)
        self.N_particles += m_sp_in / (4500.0 * 4 / 3 * math.pi * self.params["grain_radius_m"] ** 3)
        transfer(CB, SL, np.where(sp.METAL_MASK, CB.n, 0.0))

        # 3) slag: dissolution, interfacial reduction by coke-bed carbon, carburisation, exchange
        self._slag_reactions(hs, rec)
        # droplet settling
        sprops = self._slag_props_fast()
        v_st = kn.stokes_velocity(self.params["droplet_radius_m"], 6900.0 - sprops["rho"], sprops["eta_eff"])
        h_slag = max(SL.mass() / sprops["rho"] / self.A_hearth, 0.05)
        frac = min(v_st / h_slag * hs, 0.9)
        transfer(SL, MB, np.where(sp.METAL_MASK, SL.n, 0.0) * frac)
        rates["droplet_settling_frac_per_s"] = v_st / h_slag
        # 4) metal bath carbide equilibrium relaxation
        self._carbide(MB, hs, rec)
        for zz in (CB, SL, MB):
            zz.refresh_T()
        # 5) gas collection -> lower burden
        g = np.zeros(sp.NS)
        Hg = 0.0
        for zz in (CB, SL, MB):
            gi, Hi = remove_gas(zz)
            g += gi
            Hg += Hi
        n_g_tot = float(g.sum())
        T_gin = invert_temperature(Hg, g, None, CB.T) if n_g_tot > 1e-9 else CB.T
        # 6) lower burden
        T2_old = B2.T
        B2.n += g
        B2.H += Hg
        self._burden_reactions(B2, hs, rec, lower=True)
        B2.refresh_T()
        eps2 = self.params["eps_hx_B2"] * self._perm_factor()
        T_out2 = T2_old + (1 - eps2) * (T_gin - T2_old)
        g2 = np.where(sp.GAS_MASK, B2.n, 0.0)
        H2out = float(g2 @ h_species(T_out2))
        B2.n -= g2
        B2.H -= H2out
        # 7) upper burden
        T1_old = B1.T
        B1.n += g2
        B1.H += H2out
        self._burden_reactions(B1, hs, rec, lower=False)
        B1.refresh_T()
        eps1 = self.params["eps_hx_B1"] * self._perm_factor()
        T_out1 = T1_old + (1 - eps1) * (T_out2 - T1_old)
        g1 = np.where(sp.GAS_MASK, B1.n, 0.0)
        H1out = float(g1 @ h_species(T_out1))
        B1.n -= g1
        B1.H -= H1out
        # dust carry-over from upper burden
        fines = self._fines_pct()
        u_ratio = (g1.sum() / hs) / max(self._gas_ref_mol_s(), 1e-6)
        dust_kg_s = self.params["dust_frac_base"] * self.feed_design_kg_s * (fines / 8.0) * u_ratio ** 2
        solids1 = np.where(sp.GAS_MASK, 0.0, B1.n)
        m1 = float(np.sum(solids1 * sp.MW))
        dn_dust = solids1 * min(dust_kg_s * hs / max(m1, 1.0), 0.01)
        Hd = B1.carried_enthalpy(dn_dust)
        B1.n -= dn_dust
        B1.H -= Hd
        # 8) heat: electric, inter-zone, walls
        CB.H += P_cb * hs
        SL.H += P_sl * hs
        B2.H += P_bur * hs
        for (a, b), UA in ((("CB", "SL"), self.UA["CB_SL"]), (("SL", "MB"), self.UA["SL_MB"]),
                           (("CB", "B2"), self.UA["CB_B2"]), (("B2", "B1"), self.UA["B2_B1"])):
            q = UA * (z[a].T - z[b].T) * hs
            z[a].H -= q
            z[b].H += q
        q_bf = self.UA["B1_FB"] * (B1.T - self.fb_T)
        B1.H -= q_bf * hs
        for zn in ZONES:
            z[zn].H -= q_zone[zn] * hs
        for zz in z.values():
            zz.refresh_T()
        return g1, H1out, dn_dust, Hd, xi_rec, rates, q_bf

    def _perm_factor(self) -> float:
        fines = self._fines_pct()
        return float(np.clip(1.0 - 0.012 * max(fines - 8.0, 0.0), 0.55, 1.0))

    def _fines_pct(self) -> float:
        mats = self._mats_with_dist()
        blend = self.sp.recipe["ore_blend"]
        s = sum(blend.values())
        return sum(mats[k].fines_pct * v / s for k, v in blend.items())

    def _gas_ref_mol_s(self) -> float:
        return self.f["nominal_power_mw"] / 3.9 * 32.0 * 1000 / 3600 * 1.15

    def _slag_props_fast(self) -> dict:
        wt = self.slag_liquid_wt()
        SL = self.z["SL"]
        m_tot = SL.mass()
        m_sp = SL.mass(sp.SPINEL_MASK)
        fs = m_sp / max(m_tot, 1.0) * (2900 / 4500)
        eta = sg.urbain_viscosity_Pa_s(wt, SL.T) if wt else 0.5
        return {"rho": sg.density_kg_m3(wt, SL.T) if wt else 2900.0, "eta": eta,
                "eta_eff": sg.einstein_roscoe(eta, fs), "solid_frac": fs}

    def _slag_reactions(self, hs: float, rec):
        CB, SL, MB = self.z["CB"], self.z["SL"], self.z["MB"]
        T_int = 0.5 * (CB.T + SL.T)
        wt = self.slag_liquid_wt()
        if not wt:
            return
        b = sg.basicities(wt)
        w_sat = sg.cr2o3_solubility_wt(SL.T, b["B2_(CaO+MgO)/(SiO2+Al2O3)"])
        w_cr = sg.cr2o3_equivalent(wt)
        props = self._slag_props_fast()
        # chromite dissolution (shrinking particles, mass-transfer control)
        n = SL.n
        m_sp_cr = n[sp.IDX["FeCr2O4"]] * sp.MW[sp.IDX["FeCr2O4"]] + n[sp.IDX["MgCr2O4"]] * sp.MW[sp.IDX["MgCr2O4"]]
        m_ma = n[sp.IDX["MgAl2O4"]] * sp.MW[sp.IDX["MgAl2O4"]]
        M_sp = m_sp_cr + m_ma
        if M_sp > 1e-6 and self.N_particles > 0:
            r_mean = min((3 * M_sp / (4 * math.pi * 4500.0 * self.N_particles)) ** (1 / 3), 5 * self.params["grain_radius_m"])
            A_p = 3 * M_sp / (4500.0 * r_mean)
            km = kn.arrh_ref(self.k["km_chromite"], SL.T)
            dm_cr = km * (A_p * m_sp_cr / M_sp) * props["rho"] * max(w_sat - w_cr, 0.0) / 100 * hs
            Ksp = 35.0 * 30.0
            dm_ma = km * (A_p * m_ma / M_sp) * props["rho"] * 5.0 * max(1 - wt.get("MgO", 0) * wt.get("Al2O3", 0) / Ksp, 0) / 100 * hs
            fe_share = n[sp.IDX["FeCr2O4"]] * sp.MW[sp.IDX["FeCr2O4"]] / max(m_sp_cr, 1e-12)
            keys = ["R19_dis_FeCr2O4", "R20_dis_MgCr2O4", "R21_dis_MgAl2O4"]
            xi = np.array([dm_cr * fe_share / sp.MW[sp.IDX["FeCr2O4"]], dm_cr * (1 - fe_share) / sp.MW[sp.IDX["MgCr2O4"]],
                           dm_ma / sp.MW[sp.IDX["MgAl2O4"]]])
            nu = _nu(*keys)
            xi = kn.limit_extents(SL.n, nu, xi)
            react(SL, nu, xi)
            rec(keys, xi)
            self._diss_diag = dict(r_mean_um=r_mean * 1e6, A_m2=A_p, km_m_s=km, w_sat=w_sat, w_cr=w_cr)
        # interfacial reduction of dissolved oxides by coke-bed carbon
        a, am = self._slag_activities()
        fill = CB.n[sp.IDX["C"]] / self.C_cb_ref
        phiC = min(max(fill, 0.0), 1.5) ** (2 / 3)
        n = SL.n
        kk = self.k
        keys = ["R01_Fe2O3_C", "R02_FeO_C", "R03_Cr2O3_C", "R04_Cr2O3_CrO", "R05_CrO_C", "R06_SiO2_C", "R07_MnO_C",
                "R08_P2O5_C"]
        T = T_int
        xi = np.array([
            kn.arrh(kk["sl_FeO"], T) * 3 * n[sp.IDX["Fe2O3"]] * phiC,
            kn.arrh(kk["sl_FeO"], T) * n[sp.IDX["FeO"]] * phiC * kn.driving_force("R02_FeO_C", T, a["FeO"], am["Fe"]),
            kn.arrh(kk["sl_Cr2O3"], T) * n[sp.IDX["Cr2O3"]] * phiC * kn.driving_force("R03_Cr2O3_C", T, a["Cr2O3"], am["Cr"] ** 2),
            kn.arrh(kk["sl_Cr2O3_CrO"], T) * n[sp.IDX["Cr2O3"]] * phiC * kn.driving_force("R04_Cr2O3_CrO", T, a["Cr2O3"], a["CrO"] ** 2),
            kn.arrh(kk["sl_CrO"], T) * n[sp.IDX["CrO"]] * phiC * kn.driving_force("R05_CrO_C", T, a["CrO"], am["Cr"]),
            kn.arrh(kk["sl_SiO2"], T) * n[sp.IDX["SiO2"]] * phiC * kn.driving_force("R06_SiO2_C", T, a["SiO2"], am["Si"]),
            kn.arrh(kk["sl_MnO"], T) * n[sp.IDX["MnO"]] * phiC * kn.driving_force("R07_MnO_C", T, a["MnO"], am["Mn"]),
            kn.arrh(kk["sl_P2O5"], T) * n[sp.IDX["P2O5"]] * phiC,
        ]) * hs
        nu = _nu(*keys)
        # carburisation of droplets (Nernst-Brunner toward saturation)
        mwd = self.metal_wt(SL.n)
        m_drop = SL.mass(sp.METAL_MASK)
        csat = mt.c_saturation_wt(T, mwd.get("Cr", 55), mwd.get("Si", 2)) if mwd else 7.0
        c_now = mwd.get("C", 0.0) if mwd else 0.0
        xi_carb = kn.arrh(kk["carburise"], T) * max(csat - c_now, 0) / 100 * m_drop / 0.012011 * hs
        # bath carburisation through the slag/coke contact is slower (1/10)
        mwb = self.metal_wt(MB.n)
        csat_b = mt.c_saturation_wt(MB.T, mwb.get("Cr", 55), mwb.get("Si", 2))
        xi_carb_b = 0.3 * kn.arrh(kk["carburise"], T) * max(csat_b - mwb.get("C", 0), 0) / 100 * MB.mass(sp.METAL_MASK) / 0.012011 * hs
        C_need = float(-(nu[:, sp.IDX["C"]] @ xi)) + xi_carb + xi_carb_b
        C_avail = 0.9 * CB.n[sp.IDX["C"]]
        scale = min(1.0, C_avail / max(C_need, 1e-12))
        xi *= scale
        xi_carb *= scale
        xi_carb_b *= scale
        # sulfur follows consumed coke from the coke bed into the slag/droplet system
        c_cb = CB.n[sp.IDX["C"]]
        s_frac = CB.n[sp.IDX["S"]] / max(c_cb, 1e-9)
        dC = sp.vec(C=C_need * scale)
        dC[sp.IDX["S"]] = min(C_need * scale * s_frac, CB.n[sp.IDX["S"]])
        transfer(CB, SL, dC)
        xi = kn.limit_extents(SL.n, nu, xi)
        react(SL, nu, xi)
        rec(keys, xi)
        nu16 = _nu("R16_C_dissolution")
        x16 = kn.limit_extents(SL.n, nu16, np.array([xi_carb]))
        react(SL, nu16, x16)
        rec(["R16_C_dissolution"], x16)
        # bath carburisation: move C to bath then dissolve
        if xi_carb_b > 0:
            cmove = min(xi_carb_b, SL.n[sp.IDX["C"]])
            transfer(SL, MB, sp.vec(C=cmove))
            x16b = kn.limit_extents(MB.n, nu16, np.array([cmove]))
            react(MB, nu16, x16b)
        # silicothermic exchange on droplets and desulphurisation
        a, am = self._slag_activities()
        keys2 = ["R29_silicothermic", "R24_desulf"]
        nS_drop = SL.n[sp.IDX["S"]]
        nCaS = SL.n[sp.IDX["CaS"]]
        L = self.params["L_S"]
        target_CaS = (nS_drop + nCaS) * L / (1 + L)
        xi2 = np.array([
            kn.arrh(kk["silicothermic"], SL.T) * SL.n[sp.IDX["Si"]] * a["Cr2O3"] *
            kn.driving_force("R29_silicothermic", SL.T, max(am["Si"], 1e-12) ** 3 * max(a["Cr2O3"], 1e-12) ** 2,
                             max(a["SiO2"], 1e-12) ** 3 * max(am["Cr"], 1e-12) ** 4) / 3,
            kn.arrh(kk["desulf"], SL.T) * max(target_CaS - nCaS, 0.0),
        ]) * hs
        nu2 = _nu(*keys2)
        xi2 = kn.limit_extents(SL.n, nu2, xi2)
        react(SL, nu2, xi2)
        rec(keys2, xi2)
        # carbide relaxation inside droplets
        self._carbide(SL, hs, rec)
        # unused carbon returns to the coke bed
        left = SL.n[sp.IDX["C"]]
        if left > 0:
            transfer(SL, CB, sp.vec(C=left))

    def _carbide(self, Z: Zone, hs: float, rec):
        n = Z.n
        nCr, nFe = n[sp.IDX["Cr"]], n[sp.IDX["Fe"]]
        Ctot = n[sp.IDX["C_m"]] + 3 * n[sp.IDX["Cr7C3"]]
        if Ctot <= 0:
            return
        phi = min(0.9, 1.3 * nCr / max(nCr + nFe, 1e-12))
        target = phi * Ctot / 3
        d = (target - n[sp.IDX["Cr7C3"]]) * min(kn.arrh(self.k["carbide"], Z.T) * hs, 1.0)
        nu = _nu("R15_carbide")
        if d >= 0:
            x = kn.limit_extents(Z.n, nu, np.array([d]))
            react(Z, nu, x)
            rec(["R15_carbide"], x)
        else:
            x = kn.limit_extents(Z.n, -nu, np.array([-d]))
            react(Z, -nu, x)
            rec(["R15_carbide"], -x)

    def _burden_reactions(self, Z: Zone, hs: float, rec, lower: bool):
        T = Z.T
        n = Z.n
        gas = float(n[sp.GAS_MASK].sum())
        y = {k: n[sp.IDX[k]] / gas if gas > 0 else 0.0 for k in ("CO", "CO2", "H2", "H2O")}
        kk = self.k
        cr = self.dist.coke_reactivity_factor
        if lower:
            # pass 1: release of moisture and volatiles so the gas-phase steps below can act on them
            k0 = ["R13_drying", "R14_devol"]
            x0 = np.array([n[sp.IDX["H2O_l"]] / 60.0, kn.arrh(kk["devol"], T) * n[sp.IDX["VOL"]]]) * hs
            nu0 = _nu(*k0)
            x0 = kn.limit_extents(Z.n, nu0, x0)
            react(Z, nu0, x0)
            rec([k + "_B2" for k in k0], x0)
            n = Z.n
            gas = float(n[sp.GAS_MASK].sum())
            y = {k: n[sp.IDX[k]] / gas if gas > 0 else 0.0 for k in ("CO", "CO2", "H2", "H2O")}
            keys = ["R11_calcite", "R12_magnesite", "R09_Boudouard", "R22_Fe2O3_CO", "R23_FeO_CO", "R17_FeCr2O4_C",
                    "R18_MgCr2O4_C", "R02_FeO_C", "R01_Fe2O3_C", "R10_WGS", "R30_CH4_crack", "R31_C_H2O"]
            cC = min(1.0, n[sp.IDX["C"]] / max(4 * (n[sp.IDX["FeCr2O4"]] + n[sp.IDX["MgCr2O4"]]), 1e-9))
            xi = np.array([
                kn.arrh(kk["calcite"], T) * n[sp.IDX["CaCO3"]] * kn.driving_force("R11_calcite", T, 1, max(y["CO2"], 1e-6)),
                kn.arrh(kk["magnesite"], T) * n[sp.IDX["MgCO3"]],
                cr * kn.arrh(kk["boudouard"], T) * n[sp.IDX["C"]] * y["CO2"] *
                kn.driving_force("R09_Boudouard", T, max(y["CO2"], 1e-9), y["CO"] ** 2),
                kn.arrh(kk["ind_Fe2O3"], T) * n[sp.IDX["Fe2O3"]] * y["CO"],
                kn.arrh(kk["ind_FeO"], T) * n[sp.IDX["FeO"]] * y["CO"] * kn.driving_force("R23_FeO_CO", T, max(y["CO"], 1e-9), y["CO2"]),
                kn.arrh(kk["ss_FeCr2O4"], T) * n[sp.IDX["FeCr2O4"]] * cC * kn.driving_force("R17_FeCr2O4_C", T, 1, max(y["CO"], 1e-3) ** 4),
                kn.arrh(kk["ss_MgCr2O4"], T) * n[sp.IDX["MgCr2O4"]] * cC * kn.driving_force("R18_MgCr2O4_C", T, 1, max(y["CO"], 1e-3) ** 3),
                kn.arrh(kk["dir_FeO"], T) * n[sp.IDX["FeO"]] * kn.driving_force("R02_FeO_C", T, 1, max(y["CO"], 1e-3)),
                kn.arrh(kk["dir_FeO"], T) * n[sp.IDX["Fe2O3"]],
                0.0,
                kn.arrh(kk["ch4_crack"], T) * n[sp.IDX["CH4"]] * kn.driving_force("R30_CH4_crack", T, max(1 - y["H2"], 1e-6), y["H2"] ** 2),
                cr * kn.arrh(kk["c_h2o"], T) * n[sp.IDX["C"]] * y["H2O"] * kn.driving_force("R31_C_H2O", T, max(y["H2O"], 1e-9), y["CO"] * y["H2"]),
            ]) * hs
        else:
            keys = ["R13_drying", "R14_devol", "R12_magnesite", "R11_calcite", "R10_WGS", "R22_Fe2O3_CO"]
            xi = np.array([
                kn.arrh(kk["drying"], T) * n[sp.IDX["H2O_l"]] * kn.sig((T - 350.0) / 12.0),
                kn.arrh(kk["devol"], T) * n[sp.IDX["VOL"]],
                kn.arrh(kk["magnesite"], T) * n[sp.IDX["MgCO3"]],
                kn.arrh(kk["calcite"], T) * n[sp.IDX["CaCO3"]] * kn.driving_force("R11_calcite", T, 1, max(y["CO2"], 1e-6)),
                0.0,
                kn.arrh(kk["ind_Fe2O3"], T) * n[sp.IDX["Fe2O3"]] * y["CO"],
            ]) * hs
            # water-gas shift: approach to equilibrium on the gas passing through
            nCO, nH2O, nCO2, nH2 = (n[sp.IDX[k]] for k in ("CO", "H2O", "CO2", "H2"))
            K = RXN["R10_WGS"].K(T)
            phi = 1 - math.exp(-kn.arrh(kk["wgs"], T) * 30.0)
            a_ = 1 - K
            b_ = -(nCO2 + nH2) - K * (nCO + nH2O)
            c_ = nCO2 * nH2 - K * nCO * nH2O
            # (CO2+x)(H2+x) = K (CO-x)(H2O-x)  ->  (1-K)x^2 + (CO2+H2+K(CO+H2O))x + CO2 H2 - K CO H2O = 0
            b2 = (nCO2 + nH2) + K * (nCO + nH2O)
            if abs(a_) < 1e-12:
                x_eq = -c_ / b2 if b2 != 0 else 0.0
            else:
                disc = b2 * b2 - 4 * a_ * c_
                roots = [(-b2 + s * math.sqrt(max(disc, 0))) / (2 * a_) for s in (1, -1)]
                feas = [r for r in roots if -min(nCO2, nH2) <= r <= min(nCO, nH2O)]
                x_eq = feas[0] if feas else 0.0
            _ = b_
            xi[4] = max(x_eq, 0.0) * phi
        nu = _nu(*keys)
        xi = kn.limit_extents(Z.n, nu, np.maximum(xi, 0.0))
        react(Z, nu, xi)
        rec([k + ("_B2" if lower else "_B1") for k in keys], xi)

    # ------------------------------------------------------------------ tapping & lab
    def _tap(self) -> dict:
        MB, SL = self.z["MB"], self.z["SL"]
        Mm = MB.mass()
        Ms = SL.mass()
        fm = self.f["tap_fraction_metal"]
        fs = self.f["tap_fraction_slag"]
        out_m = max(Mm - self.M_metal_heel, 0.0) * fm
        out_s = max(Ms - self.M_slag_heel, 0.0) * fs
        dn_m = MB.n * (out_m / Mm if Mm > 0 else 0)
        dn_s = SL.n * (out_s / Ms if Ms > 0 else 0)
        Tm, Ts = MB.T, SL.T
        Hm = transfer_out(MB, dn_m)
        Hs = transfer_out(SL, dn_s)
        self.N_particles *= (1 - (out_s / Ms if Ms > 0 else 0))
        self.ledger.book_out("tapped_metal", dn_m, Hm)
        self.ledger.book_out("tapped_slag", dn_s, Hs)
        mw = self.metal_wt(dn_m)
        sw = self.slag_total_wt(dn_s)
        cr_eq = sg.cr2o3_equivalent(sw)
        el_s = sp.element_mass_kg(dn_s)
        el_m = sp.element_mass_kg(dn_m)
        ci = sp.ELEMENTS.index("Cr")
        drops_cr = float(sp.element_mass_kg(np.where(sp.METAL_MASK, dn_s, 0))[ci])
        spinel_cr = float(sp.element_mass_kg(np.where(sp.SPINEL_MASK, dn_s, 0))[ci])
        E = self.E_elec_since_tap
        tap = {"tap_no": len(self.taps) + 1, "t_h": (self.t + self.dt) / 3600, "metal_t": out_m / 1000, "slag_t": out_s / 1000,
               "metal_T_K": Tm, "slag_T_K": Ts, "metal_wt_pct": mw, "slag_wt_pct": sw, "slag_cr2o3_equiv_wt_pct": cr_eq,
               "slag_to_metal": out_s / out_m if out_m > 0 else None,
               "cr_in_slag_kg": float(el_s[ci]), "cr_in_metal_kg": float(el_m[ci]),
               "cr_as_droplets_in_slag_kg": drops_cr, "cr_as_undissolved_chromite_kg": spinel_cr,
               "electric_MWh_since_last_tap": E / 3.6e9,
               "sec_kwh_per_t": (E / 3.6e6) / (out_m / 1000) if out_m > 0 else None,
               "synthetic": True}
        self.taps.append(tap)
        self.E_elec_since_tap = 0.0
        self.last_tap_t = self.t + self.dt
        delay = (self.f["lab_delay_min"] + self.dist.lab_extra_delay_min) * 60
        noise = lambda v, s: v + self.rng.normal(0, s)
        self.lab_queue.append({"due_t": self.t + self.dt + delay, "sample_t_h": tap["t_h"], "tap_no": tap["tap_no"],
                               "metal": {k: noise(v, 0.25 if k in ("Cr", "Fe") else 0.05) for k, v in mw.items()},
                               "slag": {k: noise(v, 0.2) for k, v in sw.items()}})
        self.events.append({"t_h": tap["t_h"], "type": "tap", "msg": f"Tap #{tap['tap_no']}: {tap['metal_t']:.1f} t metal, {tap['slag_t']:.1f} t slag"})
        return tap

    def _process_lab(self):
        due = [r for r in self.lab_queue if r["due_t"] <= self.t]
        for r in due:
            self.lab_queue.remove(r)
            r["reported_t_h"] = self.t / 3600
            self.lab_results.append(r)
            self.events.append({"t_h": self.t / 3600, "type": "lab", "msg": f"Lab assay for tap #{r['tap_no']} reported"})

    # ------------------------------------------------------------------ schedule & disturbances
    def schedule(self, events: list[dict]):
        self.scheduled.extend(sorted(events, key=lambda e: e["t_h"]))

    def _apply_schedule(self):
        now_h = self.t / 3600
        for ev in [e for e in self.scheduled if e["t_h"] <= now_h + 1e-9]:
            self.scheduled.remove(ev)
            self.apply_action(ev["action"], ev.get("params", {}), source="schedule")

    def apply_action(self, action: str, params: dict, source: str = "operator"):
        d = self.dist
        msg = f"{action} {params}"
        if action == "set_tap":
            tap = int(np.clip(params["tap"], 1, self.design.tap_count))
            if self.t - self.last_tap_change_t < self.f["tap_min_dwell_s"] and source != "schedule":
                raise ValueError("tap change locked out (minimum dwell time)")
            self.sp.tap = tap
            self.last_tap_change_t = self.t
        elif action == "set_resistance":
            self.sp.r_setpoint_mohm = float(params["r_mohm"])
        elif action == "set_recipe":
            for k, v in params.items():
                if k == "ore_blend":
                    self.sp.recipe["ore_blend"] = dict(v)
                else:
                    self.sp.recipe[k] = float(v)
            self.recipe_change_t = self.t
        elif action == "set_tap_interval":
            self.sp.tap_interval_h = float(params["h"])
        elif action == "disturb":
            for k, v in params.items():
                if not hasattr(d, k):
                    raise ValueError(f"unknown disturbance {k}")
                setattr(d, k, v)
            if "analyzer_stuck" in params:
                self.analyzer.stuck = bool(params["analyzer_stuck"])
            if "analyzer_co_bias" in params:
                self.analyzer.bias = {"CO": float(params["analyzer_co_bias"])}
        elif action == "tap_now":
            self.last_tap_t = -1e12
        else:
            raise ValueError(f"unknown action {action}")
        self.events.append({"t_h": self.t / 3600, "type": "action", "source": source, "msg": msg})

    # ------------------------------------------------------------------ snapshot
    def operating_state(self, ecirc: dict, perm: float, tap: dict | None) -> str:
        if self.status != "ok":
            return "emergency_fallback"
        if tap is not None:
            return "tapping"
        if self.t < 3600 * 2 and self.history and False:
            return "warm_startup"
        if ecirc["p_total_W"] < 0.85 * self.f["nominal_power_mw"] * 1e6:
            return "power_curtailment"
        if ecirc["current_unbalance_pct"] > 12 or max(ecirc["i_phase_A"]) > self.design.i_max:
            return "transformer_electrode_abnormality"
        if perm < 0.8:
            return "burden_instability"
        if self.t - self.recipe_change_t < 4 * 3600:
            return "ore_recipe_transition"
        return "normal_steady_smelting"

    def _snapshot(self, ec, fb, gas_rate, dust_rate, wall_diag, wall_parts, xi, rates, tap, mass_frac, feed_kg_s,
                  q_zone, q_bf) -> dict:
        z = self.z
        gas_dry = og.composition(fb["n_rate"], dry=True)
        gas_wet = og.composition(fb["n_rate"], dry=False)
        furnace_gas_dry = og.composition(gas_rate, dry=True)
        flows = og.flows_nm3_per_h(fb["n_rate"])
        an = self.analyzer.update(furnace_gas_dry, self.dt)
        mw = self.metal_wt()
        sl_liq = self.slag_liquid_wt()
        sprops = sg.slag_properties(sl_liq, z["SL"].T, self._slag_props_fast()["solid_frac"])
        mprops = mt.metal_properties(mw, z["MB"].T)
        el_tot = self.ledger.mass_residual(self.total_inventory(), self.t - self.ledger_t0)
        e_res = self.ledger.energy_residual(self.total_H(), self.total_wall_E(), self.t - self.ledger_t0)
        perm = self._perm_factor()
        u = float(gas_rate[sp.GAS_MASK].sum()) * og.NM3_PER_MOL * (self.fb_T / 273.15) / self.A_hearth
        voidage = og.voidage_from_fines(self._fines_pct())
        dp = og.ergun_dp_pa(u * 3.0, 0.02 * perm, voidage, self.f["burden_height_m"])
        taps_window = [t for t in self.taps if t["t_h"] > self.t / 3600 - 24]
        metal_24 = sum(t["metal_t"] for t in taps_window)
        E_24 = sum(t["electric_MWh_since_last_tap"] for t in taps_window)
        cr_m = sum(t["cr_in_metal_kg"] for t in taps_window)
        cr_s = sum(t["cr_in_slag_kg"] for t in taps_window)
        ci = sp.ELEMENTS.index("Cr")
        cr_dust = float(sp.element_mass_kg(dust_rate)[ci]) * 3600 * 24
        prod_rate = (xi.get("R03_Cr2O3_C", 0) * 2 + xi.get("R05_CrO_C", 0) + 2 * xi.get("R17_FeCr2O4_C", 0) +
                     2 * xi.get("R18_MgCr2O4_C", 0) + 2 * xi.get("R17_FeCr2O4_C_B2", 0) + 2 * xi.get("R18_MgCr2O4_C_B2", 0) +
                     4 * xi.get("R29_silicothermic", 0)) * 0.051996 / self.dt
        cooling_W = wall_parts["forced_or_water"]
        cw_flow = 120.0  # kg/s total cooling water (example)
        H_gas_sens = float(fb["n_rate"] @ (h_species(self.fb_T) - h_species(298.15)))
        alarms = []
        co_risk = flows["dry_nm3_per_h"] > 0 and self.f["enclosure"] != "closed" and gas_dry.get("CO", 0) > 1.0
        if co_risk:
            alarms.append({"code": "CO_ADVISORY", "severity": "warning", "msg": "Residual CO in hood gas - check ventilation/CO detectors (advisory only)"})
        if ec["current_unbalance_pct"] > 10:
            alarms.append({"code": "PHASE_UNBALANCE", "severity": "warning", "msg": f"Current unbalance {ec['current_unbalance_pct']:.1f} %"})
        if max(ec["i_phase_A"]) > self.design.i_max:
            alarms.append({"code": "OVERCURRENT", "severity": "critical", "msg": "Electrode current above limit (regulator limiting)"})
        if sprops.get("superheat_K", 0) < 20:
            alarms.append({"code": "LOW_SLAG_SUPERHEAT", "severity": "warning", "msg": "Slag superheat < 20 K - tapping risk"})
        if perm < 0.85:
            alarms.append({"code": "BURDEN_PERMEABILITY", "severity": "warning", "msg": "Burden permeability low - blow/eruption risk"})
        for w, d in wall_diag.items():
            if d["over_service"]:
                alarms.append({"code": "REFRACTORY_T", "severity": "critical", "msg": f"{w}: refractory above service temperature"})
            if d["shell_T_K"] > 573:
                alarms.append({"code": "SHELL_HOTSPOT", "severity": "critical", "msg": f"{w}: shell {d['shell_T_K'] - 273.15:.0f} C"})
        if mw.get("Si", 0) > self.f["max_si_wt_pct"]:
            alarms.append({"code": "SI_HIGH", "severity": "warning", "msg": f"Predicted metal Si {mw['Si']:.2f} wt% above spec"})
        state = self.operating_state(ec, perm, tap)
        Pfurn = ec["p_furnace_W"]
        snap = {
            "t_s": self.t, "t_h": self.t / 3600, "synthetic": True, "mode_label": "SYNTHETIC / NOT PLANT-VALIDATED",
            "status": self.status, "operating_state": state, "model_quality": "physics-only" if self.status == "ok" else "frozen",
            "zone_temperature_k": {k: z[k].T for k in ZONES} | {"FB": self.fb_T},
            "zone_mass_t": {k: z[k].mass() / 1000 for k in ZONES},
            "metal_inventory_t": z["MB"].mass() / 1000, "slag_inventory_t": z["SL"].mass() / 1000,
            "coke_bed_carbon_t": z["CB"].n[sp.IDX["C"]] * 0.012011 / 1000, "coke_bed_fill": ec["coke_bed_fill"],
            "burden_upper_t": z["B1"].mass() / 1000, "burden_lower_t": z["B2"].mass() / 1000,
            "fecr_wt_pct": mw, "fecr_cr_wt_pct": mw.get("Cr", 0), "fecr_c_wt_pct": mw.get("C", 0), "fecr_si_wt_pct": mw.get("Si", 0),
            "fecr_fe_wt_pct": mw.get("Fe", 0), "fecr_p_wt_pct": mw.get("P", 0), "fecr_s_wt_pct": mw.get("S", 0),
            "metal_props": mprops, "slag_liquid_wt_pct": sl_liq, "slag_total_wt_pct": self.slag_total_wt(),
            "slag_props": sprops, "slag_cr2o3_equiv_wt_pct": sg.cr2o3_equivalent(self.slag_total_wt()),
            "undissolved_chromite_t": z["SL"].mass(sp.SPINEL_MASK) / 1000, "droplets_in_slag_t": z["SL"].mass(sp.METAL_MASK) / 1000,
            "dissolution": getattr(self, "_diss_diag", {}),
            "power_mw": ec["p_total_W"] / 1e6, "furnace_power_mw": Pfurn / 1e6, "electrical": ec,
            "power_split_mw": {"coke_bed": ec["P_cb_W"] / 1e6, "slag": ec["P_slag_W"] / 1e6, "burden": ec["P_burden_W"] / 1e6,
                               "network_loss": sum(ec["p_network_W"]) / 1e6},
            "electrode_depth_m": self.electrodes.tip_depth_m.tolist(),
            "feed_t_per_h": feed_kg_s * 3.6, "charge_mass_fraction": mass_frac,
            "production_rate_t_per_h_est": prod_rate * 3.6 / 0.52 if prod_rate > 0 else 0.0,
            "offgas_co_dry_vol_pct": gas_dry.get("CO", 0), "offgas_dry_vol_pct": gas_dry, "offgas_wet_vol_pct": gas_wet,
            "furnace_gas_dry_vol_pct": furnace_gas_dry, "analyzer_dry_vol_pct": an,
            "dry_offgas_nm3_per_h": flows["dry_nm3_per_h"], "wet_offgas_nm3_per_h": flows["wet_nm3_per_h"],
            "offgas_T_K": self.fb_T, "offgas_chemical_MW": og.chemical_energy_W(fb["n_rate"]) / 1e6,
            "offgas_sensible_MW": H_gas_sens / 1e6, "dust_kg_per_h": float(np.sum(dust_rate * sp.MW)) * 3600,
            "cr_in_dust_kg_per_h": float(sp.element_mass_kg(dust_rate)[ci]) * 3600, "freeboard_burned_mol_s": fb["burned_mol_s"],
            "burden_dp_kpa": dp / 1000, "permeability_index": perm, "gas_superficial_velocity_m_s": u,
            "walls": {k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in d.items()} for k, d in wall_diag.items()},
            "wall_loss_parts_MW": {k: v / 1e6 for k, v in wall_parts.items()},
            "heat_loss_total_MW": sum(d["q_out_W"] for d in wall_diag.values()) / 1e6,
            "cooling_water_dT_K": cooling_W / (cw_flow * 4186.0), "cooling_water_heat_MW": cooling_W / 1e6,
            "zone_wall_heat_MW": {k: v / 1e6 for k, v in q_zone.items()},
            "reaction_extents_mol": xi, "reaction_rates_mol_s": rates,
            "mass_balance": el_tot,
            "mass_balance_cr_residual_kg_per_h": el_tot["Cr"]["residual_kg_per_h"],
            "energy_balance": e_res, "energy_balance_residual_mw": e_res["residual_MW"],
            "kpi_24h": {"metal_t": metal_24, "electric_MWh": E_24,
                        "sec_kwh_per_t": E_24 * 1000 / metal_24 if metal_24 > 0 else None,
                        "cr_recovery_outputs_pct": 100 * cr_m / (cr_m + cr_s + cr_dust) if (cr_m + cr_s) > 0 else None,
                        "denominators": {"sec": "electric MWh (transformer secondary, incl. network) / t tapped metal",
                                         "cr_recovery_outputs": "Cr in tapped metal / (Cr in tapped metal + tapped slag + dust), 24 h window"}},
            "specific_electric_energy_kwh_per_t_fecr": (E_24 * 1000 / metal_24) if metal_24 > 0 else None,
            "cr_recovery_pct": (100 * cr_m / (cr_m + cr_s + cr_dust)) if (cr_m + cr_s) > 0 else None,
            "last_tap": self.taps[-1] if self.taps else None, "tap_now": tap,
            "next_tap_in_h": max(self.sp.tap_interval_h - (self.t - self.last_tap_t) / 3600, 0),
            "lab_pending": len(self.lab_queue), "latest_lab": self.lab_results[-1] if self.lab_results else None,
            "alarms": alarms, "warnings": list(self.warnings)[-5:],
            "setpoints": {"tap": self.sp.tap, "r_setpoint_mohm": self.sp.r_setpoint_mohm, "recipe": self.sp.recipe,
                          "tap_interval_h": self.sp.tap_interval_h, "feed_mode": self.sp.feed_mode},
            "disturbances": self.dist.__dict__.copy(), "freeze_lining_m": wall_diag["slag_band"]["freeze_m"],
            "liquidus_K": {"slag": self._slag_liq, "metal": self._metal_liq},
            "last_updated_at": self.t,
        }
        self.history.append(_compact(snap))
        return snap

    # ------------------------------------------------------------------ utilities
    def spinup(self, hours: float):
        dt0, n0 = self.dt, self.nsub
        self.dt, self.nsub = 120.0, 4
        steps = int(hours * 3600 / self.dt)
        self.reset_ledger()
        for _ in range(steps):
            self._step_once()
        self.dt, self.nsub = dt0, n0
        self.history.clear()
        self.taps.clear()
        self.events.clear()
        self.lab_results.clear()
        self.lab_queue.clear()
        self.t = 0.0
        self.last_tap_t = 0.0
        self.E_elec_since_tap = 0.0

    def run(self, hours: float) -> list[dict]:
        out = []
        for _ in range(int(round(hours * 3600 / self.dt))):
            out.append(self._step_once())
        return out

    def clone(self) -> "FeCrSAFSimulator":
        c = copy.copy(self)
        c.z = copy.deepcopy(self.z)
        c.walls = copy.deepcopy(self.walls)
        c.electrodes = copy.deepcopy(self.electrodes)
        c.sp = copy.deepcopy(self.sp)
        c.dist = copy.deepcopy(self.dist)
        c.ledger = copy.deepcopy(self.ledger)
        c.analyzer = copy.deepcopy(self.analyzer)
        c.history = deque(maxlen=200)
        c.events, c.taps, c.lab_queue, c.lab_results = [], list(self.taps), [], []
        c.scheduled = []
        c.warnings = deque(maxlen=50)
        c.rng = np.random.default_rng(self.seed + 1)
        return c


def transfer_out(z: Zone, dn: np.ndarray) -> float:
    dn = np.minimum(np.maximum(dn, 0), z.n)
    H = z.carried_enthalpy(dn)
    z.n = z.n - dn
    z.H -= H
    return H


TREND_KEYS = ["t_h", "power_mw", "furnace_power_mw", "metal_inventory_t", "slag_inventory_t", "fecr_cr_wt_pct", "fecr_c_wt_pct",
              "fecr_si_wt_pct", "slag_cr2o3_equiv_wt_pct", "offgas_co_dry_vol_pct", "dry_offgas_nm3_per_h", "coke_bed_fill",
              "coke_bed_carbon_t", "feed_t_per_h", "burden_dp_kpa", "permeability_index", "heat_loss_total_MW",
              "energy_balance_residual_mw", "mass_balance_cr_residual_kg_per_h", "undissolved_chromite_t", "droplets_in_slag_t",
              "offgas_T_K", "dust_kg_per_h", "freeze_lining_m", "cooling_water_heat_MW", "production_rate_t_per_h_est",
              "offgas_chemical_MW", "specific_electric_energy_kwh_per_t_fecr", "cr_recovery_pct"]


def _compact(s: dict) -> dict:
    d = {k: s.get(k) for k in TREND_KEYS}
    d.update({f"T_{k}": v for k, v in s["zone_temperature_k"].items()})
    e = s["electrical"]
    for i in range(3):
        d[f"I{i + 1}_kA"] = e["i_phase_A"][i] / 1000
        d[f"R{i + 1}_mohm"] = e["r_load_mohm"][i]
        d[f"h{i + 1}_m"] = e["tip_depth_m"][i]
        d[f"P{i + 1}_MW"] = e["p_load_W"][i] / 1e6
    d["pf"] = e["power_factor"]
    d["unbalance_pct"] = e["current_unbalance_pct"]
    d["tap"] = e["tap"]
    d["slag_liquidus_K"] = s["liquidus_K"]["slag"]
    d["slag_superheat_K"] = s["slag_props"].get("superheat_K")
    d["slag_viscosity_Pa_s"] = s["slag_props"].get("viscosity_effective_Pa_s")
    d["B1"] = s["slag_props"].get("basicity", {}).get("B1_(CaO+MgO)/SiO2")
    d["B2"] = s["slag_props"].get("basicity", {}).get("B2_(CaO+MgO)/(SiO2+Al2O3)")
    for k in ("CO", "CO2", "H2", "N2", "CH4", "O2"):
        d[f"gas_{k}"] = s["offgas_dry_vol_pct"].get(k, 0)
    d["analyzer_CO"] = s["analyzer_dry_vol_pct"].get("CO")
    d["shell_T_slag_K"] = s["walls"]["slag_band"]["shell_T_K"]
    d["shell_T_metal_K"] = s["walls"]["metal_band"]["shell_T_K"]
    d["hearth_shell_T_K"] = s["walls"]["hearth"]["shell_T_K"]
    d["state"] = s["operating_state"]
    return d
