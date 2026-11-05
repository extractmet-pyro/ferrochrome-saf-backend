from fastapi import APIRouter, Depends, Request

from ...core.errors import NotFound
from ...core.security import current_user, require_role
from ...db import models as M
from ...db.repositories import audit
from ...db.session import get_db
from ...ml.drift import drift_report
from ...ml import features as F
from ...schemas.ml import ActivateIn, TrainIn
from ...services import model_service as ms
from ...services.jsonable import jsonable
from ...workers.tasks import jobs
from .deps import rid

router = APIRouter(tags=["machine learning"])


@router.post("/models/train", operation_id="models_train")
def train(body: TrainIn, user=Depends(require_role("metallurgist"))):
    jid = jobs.submit("train", lambda progress, **_: jsonable(ms.train(body.target, body.kind, body.seed, progress)), user["sub"])
    return {"job_id": jid, "status": "queued", "note": "new models register as SHADOW; activation is a separate authorised step"}


@router.get("/models", operation_id="models_list")
def models(user=Depends(current_user)):
    return jsonable(ms.registry.list())


@router.get("/models/{mid}/metrics", operation_id="models_metrics")
def metrics(mid: str, user=Depends(current_user)):
    m = ms.registry.meta.get(mid)
    if not m:
        raise NotFound("model not found")
    return jsonable({k: m[k] for k in ("id", "target", "kind", "status", "metrics", "split", "hyperparameters", "importance",
                                         "drift_test_vs_train", "test_series", "manifest") if k in m})


@router.post("/models/{mid}/activate", operation_id="models_activate")
def activate(mid: str, body: ActivateIn, request: Request, user=Depends(require_role("engineer")), db=Depends(get_db)):
    try:
        m = ms.registry.activate(mid, user["sub"])
    except KeyError as exc:
        raise NotFound("model not found") from exc
    db.merge(M.ModelVersion(id=mid, target=m["target"], kind=m["kind"], status="active", metrics=jsonable(m["metrics"]),
                            manifest=jsonable(m["manifest"])))
    db.commit()
    audit(db, user, "model.activate", mid, {"reason": body.reason}, rid(request))
    return {"id": mid, "status": "active"}


@router.get("/models/campaign/summary", operation_id="campaign_summary")
def campaign(user=Depends(current_user)):
    df = F.add_lags(ms.load_campaign())
    tr, va, te = F.chrono_split(df)
    return jsonable({"rows": len(df), "columns": list(df.columns), "head": df.head(5).to_dict("records"),
                     "series": df[["t_h", "sec_actual", "physics_sec", "cr_rec_actual", "physics_cr_rec", "slag_cr2o3_actual",
                                   "physics_slag_cr2o3", "fines_pct", "coke_bed_fill"]].to_dict("list"),
                     "drift_recent_vs_train": drift_report(F.matrix(tr), F.matrix(te), F.FEATURES), "synthetic": True})
