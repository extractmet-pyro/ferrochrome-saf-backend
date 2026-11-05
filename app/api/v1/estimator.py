from fastapi import APIRouter, Depends

from ...core.security import current_user, require_role
from ...services.jsonable import jsonable
from ...services.simulation_service import service

router = APIRouter(tags=["estimator"])


@router.post("/estimator/update", operation_id="estimator_update")
def update(session_id: str, steps: int = 1, user=Depends(require_role("operator"))):
    s = service.get(session_id)
    s.step(min(max(steps, 1), 120))
    return jsonable(s.est_state)


@router.get("/estimator/state", operation_id="estimator_state")
def state(session_id: str, user=Depends(current_user)):
    s = service.get(session_id)
    snap = s.snap
    truth = {"metal_inventory_t": snap["metal_inventory_t"], "slag_inventory_t": snap["slag_inventory_t"],
             "metal_cr_wt_pct": snap["fecr_cr_wt_pct"], "metal_c_wt_pct": snap["fecr_c_wt_pct"], "metal_si_wt_pct": snap["fecr_si_wt_pct"],
             "metal_T_K": snap["zone_temperature_k"]["MB"], "slag_cr2o3eq_wt_pct": snap["slag_cr2o3_equiv_wt_pct"],
             "coke_bed_fill": snap["coke_bed_fill"]}
    return jsonable({"estimate": s.est_state, "synthetic_truth_for_validation_only": truth})
