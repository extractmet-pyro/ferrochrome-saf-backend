import os
import re

from tests.conftest import token

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_plant_write_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in paths:
        assert not re.search(r"(plc|write|command|actuat|opc)", p, re.I), p


def test_capabilities_advisory_only(client):
    c = client.get("/api/v1/capabilities").json()
    assert c["advisory_only"] and not c["closed_loop_control"] and not c["plc_write"]


def test_security_headers(client):
    r = client.get("/health")
    for h in ("X-Content-Type-Options", "X-Frame-Options", "X-Request-ID"):
        assert h in r.headers


def test_no_embedded_secrets():
    pat = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN (RSA )?PRIVATE KEY|sk-[A-Za-z0-9]{20,})")
    for d, _, fs in os.walk(os.path.join(ROOT, "app")):
        for f in fs:
            if f.endswith(".py"):
                assert not pat.search(open(os.path.join(d, f), encoding="utf-8").read()), f


def test_blocked_recommendation_cannot_be_accepted(client):
    from app.db import models as M
    from app.db.session import SessionLocal
    with SessionLocal() as db:
        db.add(M.Recommendation(id="rec-blocked", session_id="x", payload={}, gate_status="BLOCKED_SAFE_FALLBACK", signature="s"))
        db.commit()
    h = token(client, "operator")
    r = client.post("/api/v1/mpc/recommendations/rec-blocked/decision", json={"decision": "accept", "reason": "try"}, headers=h)
    assert r.status_code == 422


def test_model_activation_requires_engineer(client):
    h = token(client, "operator")
    assert client.post("/api/v1/models/x/activate", json={"reason": "promote"}, headers=h).status_code == 403
