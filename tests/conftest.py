import os
import sys

os.environ.setdefault("MODEL_DIR", "/tmp/fecr_test_models")
os.environ.setdefault("SPINUP_H", "4")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/fecr_test.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def sim_factory():
    from app.twin.simulator import FeCrSAFSimulator
    base = FeCrSAFSimulator(spinup_h=6)

    def make():
        import copy
        return copy.deepcopy(base)
    return make


@pytest.fixture(scope="session")
def client():
    if os.path.exists("/tmp/fecr_test.db"):
        os.remove("/tmp/fecr_test.db")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def token(client, user="engineer"):
    r = client.post("/api/v1/auth/login", json={"username": user, "password": "extractmet-demo"})
    return {"Authorization": "Bearer " + r.json()["access_token"]}
