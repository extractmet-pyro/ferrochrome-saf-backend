from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from ...core.errors import DomainError
from ...core.security import current_user, require_role
from ...db import models as M
from ...db.repositories import audit
from ...db.session import get_db
from ...domain import plant_config as pc
from ...domain import reactions as rx
from ...domain import species as sp
from ...domain import kinetics as kn
from ...domain.material_streams import default_materials, validate_assay, cr_fe_ratio, allocate_mineralogy
from ...domain.refractory import build_walls
from ...domain.plant_config import REFRACTORY_MATERIALS
from ...integrations.synthetic_gateway import tag_table, tags_csv
from ...schemas.materials import AssayIn
from ...schemas.plant import PlantUpdate
from ...services.jsonable import jsonable
from .deps import rid

router = APIRouter(tags=["plant"])
_state = {"cfg": pc.build_default_config(45.0, "closed")}


@router.get("/config/plant", operation_id="get_plant_config")
def get_plant(mva: float | None = None, enclosure: str | None = None, user=Depends(current_user)):
    if mva is not None:
        return pc.build_default_config(mva, enclosure or "closed")
    return _state["cfg"]


@router.put("/config/plant", operation_id="put_plant_config")
def put_plant(body: PlantUpdate, request: Request, user=Depends(require_role("engineer")), db=Depends(get_db)):
    try:
        new = pc.update_config(_state["cfg"], body.changes, reviewer=user["sub"])
    except (ValueError, KeyError) as exc:
        raise DomainError(str(exc)) from exc
    _state["cfg"] = new
    db.add(M.PlantConfigRevision(revision=new["revision"], config=new, author=user["sub"]))
    db.commit()
    audit(db, user, "config.update", f"rev {new['revision']}", {"changes": body.changes, "reason": body.reason}, rid(request))
    return new


@router.get("/config/capacities", operation_id="capacity_table")
def capacities(user=Depends(current_user)):
    return {"presets": pc.capacity_table(), "note": "rule-of-thumb scaling (CALIBRATION-REQUIRED); edit via PUT /config/plant"}


@router.get("/config/tags", operation_id="get_tags")
def get_tags(fmt: str = "json", user=Depends(current_user)):
    if fmt == "csv":
        return PlainTextResponse(tags_csv(), media_type="text/csv")
    return {"count": len(tag_table()), "tags": tag_table()}


@router.get("/config/materials", operation_id="get_materials")
def materials(user=Depends(current_user)):
    out = []
    for m in default_materials().values():
        d = m.to_dict()
        d["cr_fe_ratio"] = cr_fe_ratio(m.assay_dry_wt_pct)
        if m.category == "ore":
            d["mineralogy_kg_per_100kg_dry"] = allocate_mineralogy(m.assay_dry_wt_pct, m.spinel_al_fraction)
        out.append(d)
    return jsonable(out)


@router.get("/config/refractory", operation_id="refractory_materials")
def refractory(user=Depends(current_user)):
    walls = build_walls(pc.flat(_state["cfg"]))
    return {"materials": REFRACTORY_MATERIALS,
            "walls": {k: {"zone": w.zone, "geometry": w.geometry, "thickness_m": w.thickness(), "layers": w.profile()["layers"]} for k, w in walls.items()}}


@router.post("/validate/assay", operation_id="validate_assay")
def validate(body: AssayIn, user=Depends(current_user)):
    errs = validate_assay(body.assay_wt_pct_dry, body.basis)
    return {"valid": not errs, "errors": errs, "total_wt_pct": sum(body.assay_wt_pct_dry.values()),
            "cr_fe_ratio": cr_fe_ratio(body.assay_wt_pct_dry) if not errs else None,
            "mineralogy_kg_per_100kg_dry": allocate_mineralogy(body.assay_wt_pct_dry) if not errs else None}


@router.get("/thermo/species", operation_id="species_table")
def species(user=Depends(current_user)):
    return sp.registry_table()


@router.get("/thermo/reactions", operation_id="reaction_table")
def reactions(T_K: float = 1973.15, user=Depends(current_user)):
    return {"T_K": T_K, "reactions": jsonable(rx.table(T_K)), "ellingham": rx.ellingham_series(),
            "kinetic_parameters": kn.params_table(), "unbalanced": rx.check_all_balanced()}
