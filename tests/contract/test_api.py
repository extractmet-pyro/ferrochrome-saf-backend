import time

from tests.conftest import token

REQUIRED = ["/api/v1/health", "/api/v1/ready", "/api/v1/version", "/api/v1/capabilities", "/api/v1/config/plant",
            "/api/v1/config/tags", "/api/v1/validate/assay", "/api/v1/charges/optimize", "/api/v1/balance/static",
            "/api/v1/balance/check", "/api/v1/simulation/run", "/api/v1/simulation/{sid}", "/api/v1/simulation/{sid}/step",
            "/api/v1/simulation/{sid}/pause", "/api/v1/simulation/{sid}/reset", "/api/v1/simulation/{sid}/timeseries",
            "/api/v1/scenario/compare", "/api/v1/scenarios", "/api/v1/scenarios/{scid}", "/api/v1/telemetry/import",
            "/api/v1/telemetry/latest", "/api/v1/telemetry/history", "/api/v1/quality/events", "/api/v1/estimator/update",
            "/api/v1/estimator/state", "/api/v1/models/train", "/api/v1/models", "/api/v1/models/{mid}/metrics",
            "/api/v1/models/{mid}/activate", "/api/v1/mpc/recommend", "/api/v1/mpc/diagnostics",
            "/api/v1/mpc/recommendations/{rid_}/decision", "/api/v1/reports/kpis", "/api/v1/reports/export", "/api/v1/audit"]


def test_openapi_has_required_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    missing = [p for p in REQUIRED if p not in paths]
    assert not missing, missing
    ops = [o.get("operationId") for p in paths.values() for o in p.values() if isinstance(o, dict)]
    assert len(ops) == len(set(ops))


def test_auth_required_and_rbac(client):
    assert client.get("/api/v1/config/plant").status_code == 401
    v = token(client, "viewer")
    r = client.put("/api/v1/config/plant", json={"changes": {"tap_interval_h": 2.5}, "reason": "test"}, headers=v)
    assert r.status_code == 403
    e = token(client, "engineer")
    r = client.put("/api/v1/config/plant", json={"changes": {"tap_interval_h": 2.5}, "reason": "test"}, headers=e)
    assert r.status_code == 200 and r.json()["params"]["tap_interval_h"]["revision"] == 2
    r = client.put("/api/v1/config/plant", json={"changes": {"tap_interval_h": 99}, "reason": "bad"}, headers=e)
    assert r.status_code == 422


def test_bad_token_rejected(client):
    assert client.get("/api/v1/models", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401


def test_session_flow_contract_fields(client):
    h = token(client, "operator")
    r = client.post("/api/v1/simulation/run", json={"scenario": "A", "run": False}, headers=h | {"Idempotency-Key": "k1"})
    assert r.status_code == 200
    r2 = client.post("/api/v1/simulation/run", json={"scenario": "A", "run": False}, headers=h | {"Idempotency-Key": "k1"})
    assert r2.json()["session_id"] == r.json()["session_id"]
    sid = r.json()["session_id"]
    s = client.post(f"/api/v1/simulation/{sid}/step", json={"steps": 30}, headers=h).json()
    for k in ("metal_inventory_t", "slag_inventory_t", "fecr_cr_wt_pct", "fecr_c_wt_pct", "slag_cr2o3_equiv_wt_pct", "power_mw",
              "electrode_depth_m", "zone_temperature_k", "offgas_co_dry_vol_pct", "dry_offgas_nm3_per_h",
              "mass_balance_cr_residual_kg_per_h", "energy_balance_residual_mw", "model_quality", "synthetic", "last_updated_at"):
        assert k in s, k
    assert s["synthetic"] is True and len(s["electrode_depth_m"]) == 3
    assert client.post(f"/api/v1/simulation/{sid}/step", json={"steps": 5000}, headers=h).status_code == 422
    other = token(client, "metallurgist")
    assert client.get(f"/api/v1/simulation/{sid}", headers=other).status_code == 200
    ts = client.get(f"/api/v1/simulation/{sid}/timeseries", headers=h).json()
    assert ts["n"] >= 30
    assert client.get(f"/api/v1/simulation/{sid}/details", headers=h).status_code == 200
    assert client.post(f"/api/v1/reports/export?session_id={sid}&fmt=pdf", headers=h).headers["content-type"] == "application/pdf"


def test_assay_validation_endpoint(client):
    h = token(client, "viewer")
    r = client.post("/api/v1/validate/assay", json={"assay_wt_pct_dry": {"Cr2O3": 0.5, "SiO2": 0.5}}, headers=h).json()
    assert not r["valid"]


def test_upload_limits(client):
    h = token(client, "metallurgist")
    r = client.post("/api/v1/telemetry/import", files={"file": ("x.exe", b"abc", "application/octet-stream")}, headers=h)
    assert r.status_code == 422
    csv = b"timestamp,EL_P_TOTAL\n2026-01-01T00:00:00Z,35\n2026-01-01T00:01:00Z,36\nbad,1\n2026-01-01T00:01:00Z,36\n"
    r = client.post("/api/v1/telemetry/import", files={"file": ("h.csv", csv, "text/csv")}, headers=h).json()
    assert r["report"]["invalid_timestamps_rejected"] == 1 and r["report"]["duplicate_timestamps_rejected"] == 1


def test_websocket_stream_and_reconnect(client):
    h = token(client, "operator")
    sid = client.post("/api/v1/simulation/run", json={"scenario": "A", "run": True, "speed_steps": 1}, headers=h).json()["session_id"]
    tok = h["Authorization"].split()[1]
    for _ in range(2):  # connect, disconnect, reconnect
        with client.websocket_connect(f"/api/v1/ws/telemetry?session_id={sid}&token={tok}") as ws:
            m = ws.receive_json()
            assert "power_mw" in m
    try:
        with client.websocket_connect(f"/api/v1/ws/telemetry?session_id={sid}&token=bad") as ws:
            ws.receive_json()
        assert False
    except Exception:
        pass
