import copy

import numpy as np

from app.control.constraints import default_mpc_config
from app.control.mpc import SupervisoryMPC
from app.control.safety_gate import check
from app.integrations.synthetic_gateway import SyntheticGateway


def _fast(sim):
    cfg = default_mpc_config(sim.f, sim.sp)
    cfg.horizon_h, cfg.sim_dt_s, cfg.sim_substeps, cfg.sample_h = 2.0, 600.0, 8, 0.5
    return cfg


def test_mpc_respects_bounds_and_slew(sim_factory):
    s = sim_factory()
    s.step(10)
    m = SupervisoryMPC(s, _fast(s))
    r = m.recommend(s)
    for mv in r["moves"]:
        assert mv["lo"] - 1e-9 <= mv["new"] <= mv["hi"] + 1e-9
        assert abs(mv["delta"]) <= mv["max_move"] + 1e-9
        if mv["discrete"]:
            assert float(mv["new"]).is_integer()
    assert r["advisory_only"] is True and r["audit_signature"]


def test_tap_lockout_respected_by_mpc(sim_factory):
    s = sim_factory()
    s.apply_action("set_tap", {"tap": s.sp.tap})
    m = SupervisoryMPC(s, _fast(s))
    r = m.recommend(s)
    assert r["solver"]["tap_locked"] and [x for x in r["moves"] if x["name"] == "tap"][0]["delta"] == 0


def test_timeout_and_infeasible_give_safe_fallback(sim_factory):
    s = sim_factory()
    snap = s.step()
    cfg = _fast(s)
    cfg.timeout_s = 0.0
    m = SupervisoryMPC(s, cfg)
    r = m.recommend(s)
    tags = SyntheticGateway(0).read(snap, {})
    g = check(snap, tags, {"mode": "EKF"}, r, cfg, {})
    assert g["status"] == "BLOCKED_SAFE_FALLBACK" and g["fallback_action"]


def test_stale_telemetry_blocks_recommendation(sim_factory):
    s = sim_factory()
    snap = s.step()
    cfg = _fast(s)
    rec = {"solver": {"success": True}, "moves": [], "verification": {"hard_violations": []}}
    gw = SyntheticGateway(0)
    tags = gw.read(snap, {"missing_tags": ["EL_I1"]})
    g = check(snap, tags, {"mode": "EKF"}, rec, cfg, {})
    assert g["status"] == "BLOCKED_SAFE_FALLBACK"
    tags_ok = SyntheticGateway(0).read(snap, {})
    g2 = check(snap, tags_ok, {"mode": "EKF"}, rec, cfg, {})
    assert g2["status"] == "ISSUED_FOR_REVIEW"


def test_unsupported_state_blocks(sim_factory):
    s = sim_factory()
    snap = dict(s.step())
    snap["operating_state"] = "transformer_electrode_abnormality"
    rec = {"solver": {"success": True}, "moves": [], "verification": {"hard_violations": []}}
    g = check(snap, SyntheticGateway(0).read(snap, {}), {"mode": "EKF"}, rec, _fast(s), {})
    assert g["status"] == "BLOCKED_SAFE_FALLBACK"


def test_no_command_interface():
    import app.control.mpc as mpc_mod
    from app.integrations.opcua_readonly_adapter import OPCUAReadOnlyAdapter, WriteForbidden
    assert not any(n for n in dir(mpc_mod.SupervisoryMPC) if "write" in n.lower() or "command" in n.lower())
    try:
        OPCUAReadOnlyAdapter("opc.tcp://x", {}).write("ns=2;s=x", 1)
        assert False
    except WriteForbidden:
        pass
