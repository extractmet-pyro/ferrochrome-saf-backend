"""Race-safe simulation sessions: twin + gateway + estimator + MPC per session, warm-start
cache, background ticking for live-feeling demos, WebSocket fan-out."""
from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid

import numpy as np

from ..control.mpc import SupervisoryMPC
from ..control.safety_gate import check as gate_check
from ..core.config import settings
from ..core.errors import NotFound, QuotaExceeded
from ..domain import plant_config as pc
from ..domain import species as sp
from ..domain.material_streams import default_materials
from ..domain.static_balance import static_balance
from ..integrations.synthetic_gateway import SyntheticGateway
from ..twin.events import SCENARIOS, resolve_events
from ..twin.simulator import FeCrSAFSimulator, Setpoints
from ..twin.state_estimator import StateEstimator
from .jsonable import jsonable

_cache: dict[tuple, FeCrSAFSimulator] = {}
_cache_lock = threading.Lock()


def _key(mva, enclosure, scenario, overrides):
    return (round(float(mva), 2), enclosure, scenario, tuple(sorted((overrides or {}).items())))


def base_sim(mva=45.0, enclosure="closed", scenario="A", overrides=None, spinup_h=None) -> FeCrSAFSimulator:
    k = _key(mva, enclosure, scenario, overrides)
    with _cache_lock:
        if k not in _cache:
            cfg = pc.build_default_config(mva, enclosure, overrides=overrides)
            sc = SCENARIOS.get(scenario, {})
            spt = Setpoints()
            if sc.get("initial_recipe"):
                spt.recipe = copy.deepcopy(sc["initial_recipe"])
            sim = FeCrSAFSimulator(cfg, setpoints=spt, spinup_h=0)
            for kk, vv in sc.get("initial_disturb", {}).items():
                setattr(sim.dist, kk, vv)
            sim.spinup(settings.spinup_h if spinup_h is None else spinup_h)
            sim.reset_ledger()
            _cache[k] = sim
        return copy.deepcopy(_cache[k])


class Session:
    def __init__(self, owner: str, mva: float, enclosure: str, scenario: str, seed: int, overrides: dict | None = None):
        self.id = f"sim-{uuid.uuid4().hex[:10]}"
        self.owner = owner
        self.created = time.time()
        self.scenario = scenario
        self.lock = threading.RLock()
        self.sim = base_sim(mva, enclosure, scenario, overrides)
        self.sim.rng = np.random.default_rng(seed)
        ev = resolve_events(SCENARIOS.get(scenario, {"events": []})["events"], self.sim.sp.tap)
        self.sim.schedule(ev)
        self.gw = SyntheticGateway(seed + 1)
        self.static = static_balance(self.sim.sp.recipe, default_materials())
        self.est = StateEstimator(self.sim, self.static)
        self.mpc = SupervisoryMPC(self.sim)
        self.mpc.secret = settings.secret_key.encode()
        self.running = False
        self.speed_steps = 2
        self.tick_s = 0.5
        self.snap = None
        self.tags: dict = {}
        self.est_state = None
        self.n_lab = 0
        self.recs: dict[str, dict] = {}
        self.quality_events: list[dict] = []
        self.subscribers: set = set()
        self.step(1)

    def step(self, n: int = 1) -> dict:
        with self.lock:
            for _ in range(n):
                s = self.sim._step_once()
                tags = self.gw.read(s, self.sim.dist.__dict__)
                labs = self.sim.lab_results[self.n_lab:]
                self.n_lab = len(self.sim.lab_results)
                self.est_state = self.est.step(s["t_h"], tags, s["tap_now"], labs, self.sim.dt / 3600)
                for t, v in tags.items():
                    if v["quality"] != "GOOD":
                        self.quality_events.append({"t_h": s["t_h"], "tag": t, "quality": v["quality"]})
                self.quality_events = self.quality_events[-500:]
                self.snap, self.tags = s, tags
            return self.snap

    def public_state(self, light: bool = False) -> dict:
        s = dict(self.snap)
        if light:
            for k in ("reaction_extents_mol", "mass_balance", "energy_balance"):
                s.pop(k, None)
        s["session_id"] = self.id
        s["scenario"] = self.scenario
        s["running"] = self.running
        s["estimator"] = self.est_state
        s["geometry"] = geometry(self.sim)
        s["data_quality"] = {"bad_or_stale": [t for t, v in self.tags.items() if v["quality"] != "GOOD"]}
        s["trend_row"] = self.sim.history[-1] if self.sim.history else None
        s["measured_tags"] = {k: {"value": v["value"], "quality": v["quality"], "units": v["units"]} for k, v in self.tags.items()}
        return jsonable(s)


def geometry(sim) -> dict:
    f = sim.f
    return {"hearth_diameter_m": f["hearth_inner_diameter_m"], "burden_height_m": f["burden_height_m"],
            "electrode_diameter_m": f["electrode_diameter_m"], "pcd_m": f["pitch_circle_diameter_m"],
            "freeboard_m": f["freeboard_height_m"], "A_hearth_m2": sim.A_hearth, "A_coke_bed_m2": sim.A_cb,
            "metal_depth_m": sim.z["MB"].mass() / 6900 / sim.A_hearth, "slag_depth_m": sim.z["SL"].mass() / 2800 / sim.A_hearth,
            "lining": {k: [{"material": m, "thickness_m": t} for m, t in v] for k, v in f["refractory_lining"].items()},
            "enclosure": f["enclosure"], "mva": f["transformer_mva"]}


class SimulationService:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.loop_task = None

    def create(self, owner: str, mva=45.0, enclosure="closed", scenario="A", seed=42, overrides=None) -> Session:
        with self.lock:
            mine = [s for s in self.sessions.values() if s.owner == owner]
            if len(self.sessions) >= settings.max_sessions:
                old = sorted(self.sessions.values(), key=lambda s: s.created)[0]
                del self.sessions[old.id]
            if len(mine) >= 4:
                raise QuotaExceeded("max 4 concurrent sessions per user")
        s = Session(owner, mva, enclosure, scenario, seed, overrides)
        with self.lock:
            self.sessions[s.id] = s
        return s

    def get(self, sid: str, user: dict | None = None) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise NotFound(f"simulation session {sid} not found (sessions are in-memory; recreate after restart)")
        return s

    async def ticker(self):
        """Background loop advancing running sessions and broadcasting to WebSocket subscribers."""
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0.25)
            for s in list(self.sessions.values()):
                if not s.running or time.time() - getattr(s, "_last_tick", 0) < s.tick_s:
                    continue
                s._last_tick = time.time()
                try:
                    await loop.run_in_executor(None, s.step, s.speed_steps)
                except Exception as exc:  # noqa: BLE001
                    s.running = False
                    s.sim.warnings.append(f"ticker stopped: {exc}")
                payload = s.public_state(light=True)
                for q in list(s.subscribers):
                    if q.qsize() < 5:
                        q.put_nowait(payload)

    def recommend(self, s: Session, hybrid: bool, registry=None) -> dict:
        with s.lock:
            offsets = None
            hyb_info = {}
            if hybrid and registry is not None and not s.sim.dist.ood_flag:
                from ..ml.predict import hybrid_predict
                feats = features_now(s)
                sec = hybrid_predict(registry, "sec_residual", feats, s.static["energy"]["sec_kwh_per_t_fecr"])
                crr = hybrid_predict(registry, "cr_rec_residual", feats, s.static["cr_recovery_pct"])
                offsets = {"sec": sec.get("correction", 0.0), "cr": crr.get("correction", 0.0)}
                hyb_info = {"sec": sec, "cr_recovery": crr}
                if sec["mode"] != "hybrid" and crr["mode"] != "hybrid":
                    offsets = None
            rec = self.mpc_rec(s, offsets)
            rec["hybrid"] = hyb_info
            rec["provenance"] = {"session": s.id, "snapshot_t_h": s.snap["t_h"], "config_revision": s.sim.cfg["revision"],
                                 "model_versions": {k: v.get("model_id") for k, v in hyb_info.items()},
                                 "parameter_set": "kinetics.K_DEFAULT (CALIBRATION-REQUIRED)",
                                 "estimator_mode": s.est_state["mode"] if s.est_state else None}
            rec["gate"] = gate_check(s.snap, s.tags, s.est_state, rec, s.mpc.cfg, s.sim.dist.__dict__)
            rec["decision"] = "pending"
            s.recs[rec["id"]] = rec
            return jsonable(rec)

    def mpc_rec(self, s: Session, offsets):
        return s.mpc.recommend(s.sim, offsets)


def features_now(s: Session) -> dict:
    sim = s.sim
    r = sim.sp.recipe
    bl = r["ore_blend"]
    sb = sum(bl.values())
    last = sim.taps[-1] if sim.taps else {}
    return {"power_mw": s.snap["power_mw"], "coke_kg_t": r.get("COKE_MET", 0), "coal_kg_t": r.get("COAL_LA", 0),
            "quartz_kg_t": r.get("QUARTZITE", 0), "frac_ore_A": bl.get("ORE_A_LUMPY", 0) / sb, "frac_ore_B": bl.get("ORE_B_PELLET", 0) / sb,
            "frac_ore_C": bl.get("ORE_C_FRIABLE", 0) / sb, "fines_pct": sim._fines_pct(), "moisture_extra_pct": sim.dist.moisture_extra_pct,
            "feed_T_K": sim.dist.feed_temperature_k, "coke_bed_fill": s.snap["coke_bed_fill"],
            "electrode_depth_m": float(np.mean(s.snap["electrode_depth_m"])), "burden_dp_kpa": s.snap["burden_dp_kpa"],
            "offgas_co_pct": s.snap["offgas_co_dry_vol_pct"], "tap_interval_h": sim.sp.tap_interval_h,
            "lag1_slag_cr2o3": last.get("slag_cr2o3_equiv_wt_pct", 9.0), "lag1_metal_si": last.get("metal_wt_pct", {}).get("Si", 3.6),
            "lag1_tap_T_K": last.get("metal_T_K", 1910.0), "physics_sec": s.static["energy"]["sec_kwh_per_t_fecr"],
            "physics_cr_rec": s.static["cr_recovery_pct"], "physics_slag_cr2o3": s.static["slag_cr2o3_equiv_wt_pct"]}


class _SimView:
    def __init__(self, sim):
        self.sim = sim
        self.snap = sim.last_good
        self.static = static_balance(sim.sp.recipe, sim._mats_with_dist())


def hybrid_offsets_for(sim, registry) -> dict | None:
    """ML residual corrections for the current simulated state (None if OOD / no active model)."""
    from ..ml.predict import hybrid_predict
    if sim.last_good is None or sim.dist.ood_flag:
        return None
    v = _SimView(sim)
    f = features_now(v)
    a = hybrid_predict(registry, "sec_residual", f, v.static["energy"]["sec_kwh_per_t_fecr"])
    b = hybrid_predict(registry, "cr_rec_residual", f, v.static["cr_recovery_pct"])
    if a["mode"] != "hybrid" and b["mode"] != "hybrid":
        return None
    return {"sec": a.get("correction", 0.0), "cr": b.get("correction", 0.0)}


service = SimulationService()
_ = sp
