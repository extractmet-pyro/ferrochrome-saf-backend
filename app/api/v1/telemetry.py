import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile

from ...core.config import settings
from ...core.errors import DomainError, NotFound
from ...core.security import current_user, require_role
from ...integrations import tag_mapper
from ...integrations.file_import import ImportError_, read_table, validate
from ...schemas.telemetry import TagMapIn
from ...services.jsonable import jsonable
from ...services.simulation_service import service

router = APIRouter(tags=["telemetry & replay"])
DATASETS: dict[str, dict] = {}
MAPPING: dict = {}


@router.post("/telemetry/import", operation_id="telemetry_import")
async def import_file(file: UploadFile = File(...), user=Depends(require_role("metallurgist"))):
    data = await file.read()
    try:
        df = read_table(file.filename or "upload.csv", data, settings.max_upload_mb)
        df = tag_mapper.apply(df, MAPPING) if MAPPING else df
        df, rep = validate(df)
    except (ImportError_, ValueError) as exc:
        raise DomainError(str(exc)) from exc
    did = f"ds-{uuid.uuid4().hex[:8]}"
    DATASETS[did] = {"df": df, "report": rep, "name": file.filename, "owner": user["sub"]}
    return {"dataset_id": did, "report": rep, "columns": list(df.columns)}


@router.post("/telemetry/mapping", operation_id="telemetry_mapping")
def set_mapping(body: TagMapIn, user=Depends(require_role("engineer"))):
    try:
        m = tag_mapper.parse_mapping(body.mapping_csv)
    except (ValueError, KeyError) as exc:
        raise DomainError(str(exc)) from exc
    MAPPING.clear()
    MAPPING.update(m)
    return {"mapped": len(m)}


@router.get("/telemetry/latest", operation_id="telemetry_latest")
def latest(session_id: str, user=Depends(current_user)):
    s = service.get(session_id)
    return jsonable({"t_h": s.snap["t_h"], "tags": s.tags, "synthetic": True})


@router.get("/telemetry/history", operation_id="telemetry_history")
def history(dataset_id: str | None = None, session_id: str | None = None, limit: int = 2000, offset: int = 0,
            user=Depends(current_user)):
    limit = min(limit, 10000)
    if dataset_id:
        if dataset_id not in DATASETS:
            raise NotFound("dataset not found")
        df = DATASETS[dataset_id]["df"].iloc[offset:offset + limit]
        return jsonable({"rows": df.assign(timestamp=df["timestamp"].astype(str)).to_dict("records"), "total": len(DATASETS[dataset_id]["df"])})
    if session_id:
        rows = list(service.get(session_id).sim.history)
        return jsonable({"rows": rows[offset:offset + limit], "total": len(rows)})
    raise DomainError("provide dataset_id or session_id")


@router.get("/telemetry/datasets", operation_id="telemetry_datasets")
def datasets(user=Depends(current_user)):
    return [{"dataset_id": k, "name": v["name"], "report": v["report"]} for k, v in DATASETS.items()]


@router.post("/replay/{dataset_id}", operation_id="replay_dataset")
def replay(dataset_id: str, session_id: str, user=Depends(current_user)):
    """Compare imported measurements with the twin (driven by imported EL_TAP / recipe columns where present)."""
    if dataset_id not in DATASETS:
        raise NotFound("dataset not found")
    df = DATASETS[dataset_id]["df"]
    base = service.get(session_id)
    sim = base.sim.clone()
    t0 = df["timestamp"].iloc[0]
    out = {"t_h": [], "measured": {}, "simulated": {}, "residual": {}}
    cols = [c for c in ("EL_P_TOTAL", "GAS_CO", "TC_OFFGAS", "LAB_METAL_CR", "LAB_METAL_SI", "LAB_SLAG_CR2O3") if c in df.columns]
    path = {"EL_P_TOTAL": lambda s: s["power_mw"], "GAS_CO": lambda s: s["offgas_co_dry_vol_pct"], "TC_OFFGAS": lambda s: s["offgas_T_K"],
            "LAB_METAL_CR": lambda s: s["fecr_cr_wt_pct"], "LAB_METAL_SI": lambda s: s["fecr_si_wt_pct"],
            "LAB_SLAG_CR2O3": lambda s: s["slag_cr2o3_equiv_wt_pct"]}
    for c in cols:
        out["measured"][c], out["simulated"][c], out["residual"][c] = [], [], []
    for _, row in df.head(1500).iterrows():
        th = (row["timestamp"] - t0).total_seconds() / 3600
        while sim.t / 3600 < th:
            if "EL_TAP" in df.columns and np.isfinite(row.get("EL_TAP", np.nan)):
                sim.sp.tap = int(row["EL_TAP"])
            snap = sim.step()
        snap = sim.last_good or sim.step()
        out["t_h"].append(th)
        for c in cols:
            m = float(row[c]) if np.isfinite(row[c]) else None
            v = float(path[c](snap))
            out["measured"][c].append(m)
            out["simulated"][c].append(v)
            out["residual"][c].append(None if m is None else m - v)
    out["rmse"] = {c: float(np.sqrt(np.nanmean([r ** 2 for r in out["residual"][c] if r is not None]))) if any(r is not None for r in out["residual"][c]) else None for c in cols}
    out["lineage"] = {"dataset": DATASETS[dataset_id]["name"], "report": DATASETS[dataset_id]["report"], "twin_session": session_id}
    return jsonable(out)


@router.get("/quality/events", operation_id="quality_events")
def quality(session_id: str, user=Depends(current_user)):
    s = service.get(session_id)
    return {"events": s.quality_events[-300:], "current_bad": [t for t, v in s.tags.items() if v["quality"] != "GOOD"]}
