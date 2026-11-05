from fastapi import APIRouter

from . import auth, charge, estimator, health, ml, mpc, plant, reports, simulation, telemetry

api = APIRouter(prefix="/api/v1")
for m in (auth, health, plant, charge, simulation, telemetry, estimator, ml, mpc, reports):
    api.include_router(m.router)
api.include_router(simulation.ws_router)
