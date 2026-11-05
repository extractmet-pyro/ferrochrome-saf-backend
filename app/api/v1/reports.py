from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy import select

from ...core.security import current_user, require_role
from ...db import models as M
from ...db.repositories import audit
from ...db.session import get_db
from ...services.jsonable import jsonable
from ...services.report_service import history_csv, kpi_report, pdf_report
from ...services.simulation_service import service
from .deps import rid

router = APIRouter(tags=["reports & audit"])


@router.get("/reports/kpis", operation_id="reports_kpis")
def kpis(session_id: str, user=Depends(current_user)):
    return jsonable(kpi_report(service.get(session_id)))


@router.post("/reports/export", operation_id="reports_export")
def export(session_id: str, fmt: str = "json", request: Request = None, user=Depends(current_user), db=Depends(get_db)):
    s = service.get(session_id)
    audit(db, user, "report.export", session_id, {"fmt": fmt}, rid(request))
    if fmt == "csv":
        return PlainTextResponse(history_csv(s.sim.history), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename={session_id}.csv"})
    if fmt == "pdf":
        return Response(pdf_report(jsonable(kpi_report(s))), media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename={session_id}.pdf"})
    return jsonable(kpi_report(s))


@router.get("/audit", operation_id="audit_list")
def audit_list(limit: int = 200, offset: int = 0, user=Depends(require_role("viewer")), db=Depends(get_db)):
    q = db.scalars(select(M.AuditRecord).order_by(M.AuditRecord.id.desc()).offset(offset).limit(min(limit, 1000))).all()
    return [{"id": a.id, "ts": a.ts.isoformat(), "user": a.user, "role": a.role, "action": a.action, "target": a.target,
             "detail": a.detail, "request_id": a.request_id} for a in q]
