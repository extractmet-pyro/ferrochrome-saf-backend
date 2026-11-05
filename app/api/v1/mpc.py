from fastapi import APIRouter, Depends, Request

from ...core.errors import DomainError, NotFound
from ...core.security import current_user, require_role
from ...db import models as M
from ...db.repositories import audit
from ...db.session import get_db
from ...schemas.mpc import CompareIn, DecisionIn, RecommendIn
from ...services import model_service as ms
from ...services.jsonable import jsonable
from ...services.simulation_service import base_sim, service
from ...workers.tasks import jobs
from .deps import rid

router = APIRouter(tags=["supervisory MPC (advisory)"])


@router.post("/mpc/recommend", operation_id="mpc_recommend")
def recommend(body: RecommendIn, request: Request, user=Depends(require_role("operator")), db=Depends(get_db)):
    s = service.get(body.session_id)
    rec = service.recommend(s, body.hybrid, ms.registry)
    db.add(M.Recommendation(id=rec["id"], session_id=s.id, payload=rec, gate_status=rec["gate"]["status"], signature=rec["audit_signature"]))
    db.commit()
    audit(db, user, "mpc.recommend", rec["id"], {"gate": rec["gate"]["status"], "hybrid": body.hybrid}, rid(request))
    return rec


@router.get("/mpc/diagnostics", operation_id="mpc_diagnostics")
def diagnostics(session_id: str, user=Depends(current_user)):
    s = service.get(session_id)
    m = s.mpc
    return jsonable({"config": m.cfg.to_dict(), "identified_model": {k: v for k, v in (m.model or {}).items() if k != "free"},
                     "free_response": (m.model or {}).get("free"), "recommendations": [
                         {"id": r["id"], "t_h": r["created_t_h"], "gate": r["gate"]["status"], "decision": r["decision"]} for r in s.recs.values()],
                     "hierarchy": "MPC -> (operator approval) -> existing PLC/DCS loops; no write path implemented"})


@router.post("/mpc/recommendations/{rid_}/decision", operation_id="mpc_decision")
def decision(rid_: str, body: DecisionIn, request: Request, user=Depends(require_role("operator")), db=Depends(get_db)):
    r = db.get(M.Recommendation, rid_)
    if not r:
        raise NotFound("recommendation not found")
    if r.decision != "pending":
        raise DomainError(f"already decided: {r.decision}")
    if body.decision == "accept" and r.gate_status != "ISSUED_FOR_REVIEW":
        raise DomainError("blocked recommendations cannot be accepted")
    r.decision, r.decision_reason, r.decided_by = body.decision, body.reason, user["sub"]
    db.commit()
    for s in service.sessions.values():
        if rid_ in s.recs:
            s.recs[rid_]["decision"] = body.decision
    audit(db, user, f"mpc.{body.decision}", rid_, {"reason": body.reason}, rid(request))
    return {"id": rid_, "decision": body.decision,
            "note": "Recorded for audit. Acceptance does NOT transmit anything to plant systems; operators implement changes via approved procedures."}


@router.post("/mpc/recommendations/{rid_}/apply-to-simulation", operation_id="mpc_apply_sim")
def apply_sim(rid_: str, session_id: str, request: Request, user=Depends(require_role("operator")), db=Depends(get_db)):
    """Apply an ACCEPTED recommendation to the SIMULATED furnace (what-if). Never reaches a plant."""
    s = service.get(session_id)
    rec = s.recs.get(rid_)
    if not rec or rec["decision"] != "accept":
        raise DomainError("recommendation must exist in this session and be accepted")
    from ...control.baselines import _apply
    with s.lock:
        _apply(s.sim, rec)
    audit(db, user, "mpc.apply_to_simulation", rid_, {}, rid(request))
    return {"applied_to": "simulation only", "setpoints": jsonable(s.sim.sp.__dict__)}


@router.post("/mpc/rollback", operation_id="mpc_rollback")
def rollback(session_id: str, request: Request, user=Depends(require_role("operator")), db=Depends(get_db)):
    s = service.get(session_id)
    applied = [r for r in s.recs.values() if r["decision"] == "accept"]
    if not applied:
        raise DomainError("no accepted recommendation to roll back")
    last = applied[-1]
    with s.lock:
        for m in last["moves"]:
            if m["name"] == "tap":
                s.sim.sp.tap = int(m["current"])
            elif m["name"] == "r_setpoint_mohm":
                s.sim.sp.r_setpoint_mohm = m["current"]
            elif m["name"] == "coke_kg_t":
                from ...control.mpc import _coke_key
                s.sim.sp.recipe[_coke_key(s.sim)] = m["current"]
            elif m["name"] == "quartz_kg_t":
                s.sim.sp.recipe["QUARTZITE"] = m["current"]
    last["decision"] = "rolled_back"
    audit(db, user, "mpc.rollback", last["id"], {}, rid(request))
    return {"rolled_back": last["id"], "applied_to": "simulation only"}


@router.post("/mpc/compare", operation_id="mpc_compare")
def compare(body: CompareIn, user=Depends(require_role("metallurgist"))):
    from ...control.baselines import compare as cmp
    from ...services.simulation_service import hybrid_offsets_for
    offsets = (lambda sim: hybrid_offsets_for(sim, ms.registry)) if body.hybrid else None
    jid = jobs.submit("mpc_compare", lambda progress, **_: jsonable(cmp(lambda: base_sim(45.0, "closed", body.scenario), body.scenario,
                                                                         body.hours, 3.0, offsets, True, progress)), user["sub"])
    return {"job_id": jid, "status": "queued"}
