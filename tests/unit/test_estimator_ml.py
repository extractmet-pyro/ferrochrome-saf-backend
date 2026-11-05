import numpy as np
import pandas as pd

from app.domain.material_streams import DEFAULT_RECIPE, default_materials
from app.domain.static_balance import static_balance
from app.integrations.synthetic_gateway import TAGS, SyntheticGateway
from app.ml import features as F
from app.ml.drift import OODDetector, psi
from app.ml.train import train_target
from app.services.model_service import DATA
from app.twin.state_estimator import StateEstimator


def test_at_least_50_tags():
    assert len(TAGS) >= 50


def test_estimator_tracks_and_no_future_leakage(sim_factory):
    s = sim_factory()
    est = StateEstimator(s, static_balance(DEFAULT_RECIPE, default_materials()))
    gw = SyntheticGateway(3)
    nl = 0
    used = []
    for _ in range(int(7 * 3600 / s.dt)):
        snap = s.step()
        tags = gw.read(snap, s.dist.__dict__)
        labs = s.lab_results[nl:]
        nl = len(s.lab_results)
        for lab in labs:
            assert lab["reported_t_h"] <= snap["t_h"] + 1e-9  # only reported results
            used.append(lab)
        st = est.step(snap["t_h"], tags, snap["tap_now"], labs, s.dt / 3600)
    assert used, "at least one delayed assay fused"
    v = {x["name"]: x for x in st["states"]}
    assert abs(v["metal_cr_wt_pct"]["value"] - snap["fecr_cr_wt_pct"]) < 1.5
    assert abs(v["metal_inventory_t"]["value"] - snap["metal_inventory_t"]) < 6
    assert st["mode"] == "EKF"


def test_gateway_flags_missing_and_stuck(sim_factory):
    s = sim_factory()
    snap = s.step()
    gw = SyntheticGateway(1)
    t = gw.read(snap, {"missing_tags": ["GAS_FLOW_DRY"], "analyzer_stuck": True})
    assert t["GAS_FLOW_DRY"]["quality"] in ("STALE", "BAD")
    assert t["GAS_CO"]["quality"] == "SUSPECT_STUCK"


def test_ml_chronological_split_no_lookahead():
    df = F.add_lags(pd.read_csv(DATA))
    tr, va, te = F.chrono_split(df)
    assert tr.t_h.max() < va.t_h.min() < te.t_h.min()
    # lag features use only the previous tap
    assert np.allclose(df["lag1_slag_cr2o3"].iloc[1:].to_numpy(), df["slag_cr2o3_actual"].iloc[:-1].to_numpy())


def test_hybrid_improves_on_physics_for_synthetic_mismatch():
    df = pd.read_csv(DATA)
    r = train_target(df, "slag_cr2o3_residual", "ridge")
    assert r["metrics"]["test_hybrid"]["RMSE"] < r["metrics"]["test_physics_only"]["RMSE"]


def test_ood_and_psi():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    o = OODDetector(X)
    assert not o.is_ood(np.zeros(5))
    assert o.is_ood(np.full(5, 8.0))
    assert psi(X[:, 0], X[:, 0]) < 0.05
    assert psi(X[:, 0], X[:, 0] + 2) > 0.25
