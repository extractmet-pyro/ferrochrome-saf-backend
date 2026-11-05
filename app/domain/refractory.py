"""Transient multi-layer refractory heat conduction with coupled convection + radiation losses.

Each wall region (hearth, metal/slag/burden sidewall bands, roof) is discretised by finite
volumes (cylindrical shells for sidewalls, planar slabs for hearth/roof). Conductivity may be
temperature dependent, k(T) = k0 + k1 (T - 300). Time integration is backward Euler (implicit,
unconditionally stable), Picard-iterated for k(T) and the linearised radiation coefficient.

Outer boundary (shell): q = h_conv (Ts - Tamb) + eps*sigma*(Ts^4 - Tamb^4) [+ h_spray (Ts - Tcw)]
with natural convection from a simplified turbulent vertical-plate correlation
h = 1.31 * dT^(1/3) W/m2/K (McAdams), or forced air / water cooling where configured.
Inner boundary: convective from the zone (h_in), optionally through a freeze lining whose
quasi-steady thickness relaxes with time constant tau_f.

The discrete energy balance of every wall closes exactly:
  q_in*dt = dE_stored + q_out*dt   (q_out evaluated with the same linearised coefficients).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .plant_config import REFRACTORY_MATERIALS

SIGMA = 5.670374419e-8


@dataclass
class Wall:
    name: str
    zone: str
    layers: list[tuple[str, float]]
    geometry: str  # "cyl" | "slab"
    r_inner: float = 1.0  # cyl
    height: float = 1.0  # cyl band height
    area: float = 1.0  # slab area
    h_in: float = 300.0
    outer_mode: str = "natural"  # natural | spray | air | water
    h_outer_forced: float = 0.0
    T_coolant: float = 305.15
    T_amb: float = 303.15
    emissivity: float = 0.8
    freeze_lining: bool = False
    nodes_per_layer: int = 4
    # state
    T: np.ndarray = field(default=None, repr=False)
    delta_f: float = 0.05
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        xs, mats = [], []
        r = self.r_inner if self.geometry == "cyl" else 0.0
        edges = [r]
        for mat, th in self.layers:
            n = max(2, self.nodes_per_layer if th > 0.05 else 2)
            for _ in range(n):
                edges.append(edges[-1] + th / n)
                mats.append(mat)
        self.edges = np.array(edges)
        self.mats = mats
        self.N = len(mats)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        rho = np.array([REFRACTORY_MATERIALS[m]["rho"] for m in mats])
        cp = np.array([REFRACTORY_MATERIALS[m]["cp"] for m in mats])
        if self.geometry == "cyl":
            vol = math.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2) * self.height
            self.A_in = 2 * math.pi * self.edges[0] * self.height
            self.A_out = 2 * math.pi * self.edges[-1] * self.height
        else:
            vol = np.diff(self.edges) * self.area
            self.A_in = self.A_out = self.area
        self.C = rho * cp * vol  # J/K
        self.t_max = np.array([REFRACTORY_MATERIALS[m]["t_max"] for m in mats])
        if self.T is None:
            self.T = np.full(self.N, 600.0)

    def thickness(self) -> float:
        return float(self.edges[-1] - self.edges[0])

    def _k(self, T: np.ndarray) -> np.ndarray:
        k0 = np.array([REFRACTORY_MATERIALS[m]["k0"] for m in self.mats])
        k1 = np.array([REFRACTORY_MATERIALS[m]["k1"] for m in self.mats])
        return np.maximum(k0 + k1 * (T - 300.0), 0.05)

    def _conductances(self, T: np.ndarray) -> np.ndarray:
        """Conductance between node j and j+1 (W/K), series half-cells."""
        k = self._k(T)
        G = np.zeros(self.N - 1)
        for j in range(self.N - 1):
            if self.geometry == "cyl":
                Ra = math.log(self.edges[j + 1] / self.centers[j]) / (2 * math.pi * k[j] * self.height)
                Rb = math.log(self.centers[j + 1] / self.edges[j + 1]) / (2 * math.pi * k[j + 1] * self.height)
            else:
                Ra = (self.edges[j + 1] - self.centers[j]) / (k[j] * self.area)
                Rb = (self.centers[j + 1] - self.edges[j + 1]) / (k[j + 1] * self.area)
            G[j] = 1.0 / (Ra + Rb)
        return G

    def _half_res(self, T: np.ndarray, j: int, side: str) -> float:
        k = self._k(T)[j]
        if self.geometry == "cyl":
            if side == "in":
                return math.log(self.centers[j] / self.edges[j]) / (2 * math.pi * k * self.height)
            return math.log(self.edges[j + 1] / self.centers[j]) / (2 * math.pi * k * self.height)
        if side == "in":
            return (self.centers[j] - self.edges[j]) / (k * self.area)
        return (self.edges[j + 1] - self.centers[j]) / (k * self.area)

    def inner_conductance(self, T: np.ndarray) -> float:
        h_eff = self.h_in
        r_extra = 0.0
        if self.freeze_lining:
            r_extra = self.delta_f / (REFRACTORY_MATERIALS["frozen_slag"]["k0"] * self.A_in)
        return 1.0 / (1.0 / (h_eff * self.A_in) + r_extra + self._half_res(T, 0, "in"))

    def _outer_coeffs(self, Ts: float) -> dict:
        dT = max(Ts - self.T_amb, 0.1)
        h_nat = 1.31 * dT ** (1 / 3) if self.outer_mode in ("natural", "spray") else 0.0
        h_rad = self.emissivity * SIGMA * (Ts ** 2 + self.T_amb ** 2) * (Ts + self.T_amb)
        if self.outer_mode in ("water",):
            h_rad = 0.0
        h_f = self.h_outer_forced if self.outer_mode in ("spray", "air", "water") else 0.0
        T_f = self.T_coolant if self.outer_mode in ("spray", "water") else self.T_amb
        return dict(h_nat=h_nat, h_rad=h_rad, h_f=h_f, T_f=T_f)

    def step(self, T_zone: float, dt: float, T_liq_slag: float | None = None, q_in_prescribed: float | None = None) -> dict:
        """Advance wall by dt with zone temperature held (or a prescribed inner heat flow, W)."""
        Told = self.T.copy()
        Tn = self.T.copy()
        if self.freeze_lining and T_liq_slag is not None:
            q_s = 150.0 * max(T_zone - T_liq_slag, 1.0)  # slag->freeze hot face, W/m2
            hot = Tn[0]
            d_ss = REFRACTORY_MATERIALS["frozen_slag"]["k0"] * max(T_liq_slag - hot, 0.0) / q_s
            d_ss = min(max(d_ss, 0.002), 0.35)
            tau = 6 * 3600.0
            self.delta_f += (d_ss - self.delta_f) * min(dt / tau, 1.0)
        for _ in range(3):
            G = self._conductances(Tn)
            Gin = self.inner_conductance(Tn)
            Ts_guess = Tn[-1]
            oc = self._outer_coeffs(Ts_guess)
            R_half_out = self._half_res(Tn, self.N - 1, "out")
            h_tot = oc["h_nat"] + oc["h_rad"] + oc["h_f"]
            # equivalent ambient temperature of the parallel outer paths
            T_eq = ((oc["h_nat"] + oc["h_rad"]) * self.T_amb + oc["h_f"] * oc["T_f"]) / h_tot
            Gout = 1.0 / (R_half_out + 1.0 / (h_tot * self.A_out))
            ab = np.zeros((3, self.N))
            rhs = self.C / dt * Told
            diag = self.C / dt
            diag = diag.copy()
            diag[:-1] += G
            diag[1:] += G
            if q_in_prescribed is None:
                diag[0] += Gin
            diag[-1] += Gout
            rhs = rhs.copy()
            if q_in_prescribed is None:
                rhs[0] += Gin * T_zone
            else:
                rhs[0] += q_in_prescribed
            rhs[-1] += Gout * T_eq
            ab[1] = diag
            ab[0, 1:] = -G
            ab[2, :-1] = -G
            Tn = solve_banded((1, 1), ab, rhs)
        self.T = Tn
        q_in = Gin * (T_zone - Tn[0]) if q_in_prescribed is None else q_in_prescribed
        q_out = Gout * (Tn[-1] - T_eq)
        Ts = Tn[-1] - q_out * R_half_out
        parts = {"convection": oc["h_nat"] * self.A_out * (Ts - self.T_amb),
                 "radiation": oc["h_rad"] * self.A_out * (Ts - self.T_amb),
                 "forced_or_water": oc["h_f"] * self.A_out * (Ts - oc["T_f"])}
        hot_face = Tn[0] + (q_in * self._half_res(Tn, 0, "in"))
        self.last = dict(q_in_W=q_in, q_out_W=q_out, shell_T_K=Ts, hot_face_T_K=hot_face, parts_W=parts,
                         dE_J=float(np.sum(self.C * (Tn - Told))), freeze_m=self.delta_f if self.freeze_lining else 0.0,
                         over_service=bool(np.any(Tn > self.t_max)))
        return self.last

    def stored_J(self) -> float:
        return float(np.sum(self.C * (self.T - 298.15)))

    def init_steady(self, T_zone: float, T_liq_slag: float | None = None):
        for _ in range(400):
            self.step(T_zone, 3.0e5, T_liq_slag)

    def profile(self) -> dict:
        return {"x_m": (self.centers - self.edges[0]).tolist(), "T_K": self.T.tolist(),
                "materials": self.mats, "thickness_m": self.thickness(),
                "layers": [{"material": m, "label": REFRACTORY_MATERIALS[m]["label"], "thickness_m": t} for m, t in self.layers]}


def build_walls(cfg_flat: dict) -> dict[str, Wall]:
    lin = cfg_flat["refractory_lining"]
    D = cfg_flat["hearth_inner_diameter_m"]
    r = D / 2
    A_hearth = math.pi * r * r
    Tamb = cfg_flat["ambient_temperature_k"]
    Tcw = cfg_flat["cooling_water_inlet_k"]
    eps = cfg_flat["shell_emissivity"]
    spray = cfg_flat["sidewall_spray_cooling"]
    hs = cfg_flat["spray_htc_w_m2k"]
    hb = cfg_flat["burden_height_m"]
    mode_side = "spray" if spray else "natural"
    W = {
        "hearth": Wall("hearth", "MB", lin["hearth"], "slab", area=A_hearth, h_in=400.0, outer_mode="air",
                       h_outer_forced=cfg_flat["hearth_air_htc_w_m2k"], T_amb=Tamb, T_coolant=Tamb, emissivity=eps),
        "metal_band": Wall("metal_band", "MB", lin["metal_band"], "cyl", r_inner=r, height=cfg_flat["metal_bath_design_depth_m"] + 0.4,
                           h_in=400.0, outer_mode=mode_side, h_outer_forced=hs, T_amb=Tamb, T_coolant=Tcw, emissivity=eps),
        "slag_band": Wall("slag_band", "SL", lin["slag_band"], "cyl", r_inner=r, height=cfg_flat["slag_layer_design_depth_m"] + 0.3,
                          h_in=150.0, outer_mode=mode_side, h_outer_forced=hs, T_amb=Tamb, T_coolant=Tcw, emissivity=eps,
                          freeze_lining=True),
        "burden_lower": Wall("burden_lower", "B2", lin["burden_band"], "cyl", r_inner=r, height=0.5 * hb, h_in=40.0,
                             outer_mode="natural", T_amb=Tamb, emissivity=eps),
        "burden_upper": Wall("burden_upper", "B1", lin["burden_band"], "cyl", r_inner=r, height=0.5 * hb, h_in=25.0,
                             outer_mode="natural", T_amb=Tamb, emissivity=eps),
        "roof": Wall("roof", "FB", lin["roof"], "slab", area=A_hearth, h_in=60.0, outer_mode="water",
                     h_outer_forced=cfg_flat["roof_water_htc_w_m2k"], T_amb=Tamb, T_coolant=Tcw, emissivity=eps),
    }
    return W
