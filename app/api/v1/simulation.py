import asyncio

from fastapi import APIRouter, Body, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from ...core.errors import DomainError
from ...core.security import current_user, decode, require_role
from ...db import models as M
from ...db.repositories import audit
from ...db.session import get_db
from ...schemas.simulation import ActionIn, ScenarioCompareIn, ScenarioSave, SimCreate, SpeedIn, StepIn
from ...services import scenario_service
from ...services.jsonable import jsonable
from ...services.report_service import history_csv
from ...services.simulation_service import service
from ...twin.events import scenario_table
from ...twin.reconciler import cr_balance_window
from ...workers.tasks import jobs
from .deps import idem_key, idempotent, rid

router = APIRouter(tags=["simulation"])


@router.post("/simulation/run", operation_id="simulation_create")
def create(body: SimCreate, request: Request, key=Depends(idem_key), user=Depends(current_user), db=Depends(get_db)):
    def go():
        s = service.create(user["sub"], body.mva, body.enclosure, body.scenario, body.seed, body.overrides or None)
        s.running, s.speed_steps = body.run, body.speed_steps
        db.add(M.SimulationRun(id=s.id, owner=user["sub"], params=body.model_dump(), status="created"))
        db.commit()
        return {"session_id": s.id, "state": s.public_state(light=True)}
    return idempotent(request, key, go)


@router.get("/simulation/sessions", operation_id="simulation_list")
def list_sessions(user=Depends(current_user)):
    return [{"session_id": s.id, "owner": s.owner, "scenario": s.scenario, "t_h": s.snap["t_h"], "running": s.running,
             "mva": s.sim.f["transformer_mva"]} for s in service.sessions.values()]


@router.get("/simulation/{sid}", operation_id="simulation_state")
def state(sid: str, light: bool = False, user=Depends(current_user)):
    return service.get(sid).public_state(light)


@router.post("/simulation/{sid}/step", operation_id="simulation_step")
def step(sid: str, body: StepIn, user=Depends(require_role("operator"))):
    s = service.get(sid)
    s.step(body.steps)
    return s.public_state(light=True)


@router.post("/simulation/{sid}/pause", operation_id="simulation_pause")
def pause(sid: str, body: SpeedIn | None = Body(default=None), user=Depends(require_role("operator"))):
    s = service.get(sid)
    if body is None:
        s.running = False
    else:
        s.running, s.speed_steps = body.running, body.speed_steps
    return {"session_id": sid, "running": s.running, "speed_steps": s.speed_steps}


@router.post("/simulation/{sid}/reset", operation_id="simulation_reset")
def reset(sid: str, user=Depends(require_role("operator"))):
    old = service.get(sid)
    f = old.sim.f
    s = service.create(old.owner, f["transformer_mva"], f["enclosure"], old.scenario, 42)
    service.sessions.pop(sid, None)
    s.id = sid
    service.sessions[sid] = s
    return s.public_state(light=True)


@router.post("/simulation/{sid}/action", operation_id="simulation_action")
def action(sid: str, body: ActionIn, request: Request, user=Depends(require_role("operator")), db=Depends(get_db)):
    """What-if action on the SIMULATED furnace only (never a plant command)."""
    s = service.get(sid)
    try:
        with s.lock:
            s.sim.apply_action(body.action, body.params, source=f"user:{user['sub']}")
    except (ValueError, KeyError) as exc:
        raise DomainError(str(exc)) from exc
    audit(db, user, "simulation.what_if", sid, body.model_dump(), rid(request))
    return {"ok": True, "applied_to": "simulation only", "setpoints": jsonable(s.snap["setpoints"])}


@router.get("/simulation/{sid}/timeseries", operation_id="simulation_timeseries")
def timeseries(sid: str, fmt: str = "json", t_from: float | None = None, t_to: float | None = None, limit: int = 3000,
               user=Depends(current_user)):
    s = service.get(sid)
    rows = [r for r in s.sim.history if (t_from is None or r["t_h"] >= t_from) and (t_to is None or r["t_h"] <= t_to)]
    rows = rows[-min(limit, 20000):]
    if fmt == "csv":
        return PlainTextResponse(history_csv(rows), media_type="text/csv")
    return jsonable({"rows": rows, "n": len(rows)})


@router.get("/simulation/{sid}/details", operation_id="simulation_details")
def details(sid: str, user=Depends(current_user)):
    """Heavy payloads: wall temperature profiles, mass/energy ledgers, taps, events, labs."""
    s = service.get(sid)
    sim = s.sim
    with s.lock:
        led = sim.ledger
        from ...domain import species as sp
        streams_in = {k: {"kg": float((v * sp.MW).sum()), "elements_kg": dict(zip(sp.ELEMENTS, sp.element_mass_kg(v).tolist())),
                          "species_kg": {sp.KEYS[i]: float(v[i] * sp.MW[i]) for i in range(sp.NS) if v[i] > 1e-9}} for k, v in led.n_in.items()}
        streams_out = {k: {"kg": float((v * sp.MW).sum()), "elements_kg": dict(zip(sp.ELEMENTS, sp.element_mass_kg(v).tolist())),
                           "species_kg": {sp.KEYS[i]: float(v[i] * sp.MW[i]) for i in range(sp.NS) if v[i] > 1e-9}} for k, v in led.n_out.items()}
        zones = {k: {"T_K": z.T, "mass_t": z.mass() / 1000,
                     "species_kg": {sp.KEYS[i]: float(z.n[i] * sp.MW[i]) for i in range(sp.NS) if z.n[i] > 1e-6}} for k, z in sim.z.items()}
        out = {"walls": {k: w.profile() | {"last": w.last} for k, w in sim.walls.items()},
               "mass_balance": s.snap["mass_balance"], "energy_balance": s.snap["energy_balance"],
               "streams_in": streams_in, "streams_out": streams_out, "zones": zones, "taps": sim.taps[-60:],
               "events": sim.events[-300:], "labs": sim.lab_results[-30:], "quality_events": s.quality_events[-100:],
               "reaction_extents_mol": s.snap["reaction_extents_mol"], "reaction_rates_mol_s": s.snap["reaction_rates_mol_s"],
               "elapsed_h": (sim.t - sim.ledger_t0) / 3600, "cr_reconciliation": cr_balance_window(sim)}
    return jsonable(out)


@router.get("/scenarios/catalog", operation_id="scenario_catalog")
def catalog(user=Depends(current_user)):
    return scenario_table()


@router.post("/scenario/compare", operation_id="scenario_compare")
def compare(body: ScenarioCompareIn, user=Depends(current_user)):
    bad = [x for x in body.scenarios if x not in "ABCDEFGHIJKL"]
    if bad:
        raise DomainError(f"unknown scenarios {bad}")
    jid = jobs.submit("scenario_compare", lambda progress, **_: scenario_service.compare(body.scenarios, body.hours, progress), user["sub"])
    return {"job_id": jid, "status": "queued"}


@router.get("/scenarios", operation_id="scenarios_list")
def scenarios(user=Depends(current_user), db=Depends(get_db)):
    return [{"id": r.id, "name": r.name, "author": r.author, "created_at": r.created_at.isoformat()} for r in db.query(M.Scenario).all()]


@router.post("/scenarios", operation_id="scenarios_save")
def save(body: ScenarioSave, user=Depends(require_role("operator")), db=Depends(get_db)):
    r = M.Scenario(name=body.name, definition=body.definition, author=user["sub"])
    db.add(r)
    db.commit()
    return {"id": r.id, "name": r.name}


@router.get("/scenarios/{scid}", operation_id="scenario_get")
def get_scenario(scid: int, user=Depends(current_user), db=Depends(get_db)):
    r = db.get(M.Scenario, scid)
    if not r:
        raise DomainError("scenario not found")
    return {"id": r.id, "name": r.name, "definition": r.definition, "result_summary": r.result_summary}


@router.get("/jobs/{jid}", operation_id="job_get")
def job(jid: str, user=Depends(current_user)):
    return jobs.get(jid)


@router.post("/jobs/{jid}/cancel", operation_id="job_cancel")
def job_cancel(jid: str, user=Depends(current_user)):
    return jobs.cancel(jid)


ws_router = APIRouter()


@ws_router.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket, session_id: str, token: str):
    try:
        decode(token)
        s = service.get(session_id)
    except Exception:  # noqa: BLE001
        await ws.close(code=4401)
        return
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    s.subscribers.add(q)
    try:
        await ws.send_json(s.public_state(light=True))
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=15)
                await ws.send_json(payload)
            except asyncio.TimeoutError:
                await ws.send_json({"heartbeat": True, "running": s.running})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        s.subscribers.discard(q)
