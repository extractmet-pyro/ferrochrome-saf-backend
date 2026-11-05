from fastapi import APIRouter, Depends

from ...core.errors import DomainError
from ...core.security import current_user
from ...domain.economics import breakeven_tariff, kpis, pareto
from ...domain.material_streams import default_materials
from ...domain.static_balance import DEFAULT_PRICES, optimise_charge, static_balance
from ...schemas.materials import BalanceIn, OptimiseIn
from ...services.jsonable import jsonable
from ...twin.uncertainty import monte_carlo, sensitivity
from ...workers.tasks import jobs

router = APIRouter(tags=["charge & balance"])


def _recipe(r):
    d = r.model_dump()
    if not d["ore_blend"] or sum(d["ore_blend"].values()) <= 0:
        raise DomainError("ore_blend must contain at least one ore with positive fraction")
    mats = default_materials()
    unknown = [k for k in d["ore_blend"] if k not in mats or mats[k].category != "ore"]
    if unknown:
        raise DomainError(f"unknown ore keys {unknown}")
    return d


@router.post("/balance/static", operation_id="static_balance")
def balance_static(body: BalanceIn, user=Depends(current_user)):
    return jsonable(static_balance(_recipe(body.recipe), default_materials(), body.conditions, DEFAULT_PRICES | body.prices))


@router.post("/balance/check", operation_id="balance_check")
def balance_check(body: BalanceIn, user=Depends(current_user)):
    """Element-closure audit of the static balance (inputs vs metal+slag+gas+dust+coke-bed surplus)."""
    b = static_balance(_recipe(body.recipe), default_materials(), body.conditions, DEFAULT_PRICES)
    issues = []
    if b["cr_recovery_pct"] > 100 or b["cr_recovery_pct"] < 0:
        issues.append("Cr recovery outside 0-100 %")
    if b["slag_props"]["superheat_K"] < 0:
        issues.append("slag below liquidus at stated tapping temperature")
    if b["carbon_ratio_supply_to_demand"] < 0.97:
        issues.append("carbon deficit: recovery limited, electrodes will run deep")
    if b["carbon_ratio_supply_to_demand"] > 1.08:
        issues.append("carbon surplus accumulates in coke bed: Si pick-up and shallow electrodes")
    return jsonable({"ok": not issues, "issues": issues, "carbon_ratio": b["carbon_ratio_supply_to_demand"],
                     "cr_distribution_pct": b["cr_distribution_pct"], "energy_breakdown": b["energy"]["breakdown_kwh_per_t_fecr"]})


@router.post("/charges/optimize", operation_id="optimise_charge")
def optimise(body: OptimiseIn, user=Depends(current_user)):
    mats = default_materials()
    for k in body.ores + body.reductants + body.fluxes:
        if k not in mats:
            raise DomainError(f"unknown material {k}")
    r = optimise_charge(mats, _recipe(body.base_recipe), body.ores, body.reductants, body.fluxes, body.constraints,
                        body.prices, body.conditions, {k: tuple(v) for k, v in body.bounds.items()})
    return jsonable(r)


@router.post("/balance/uncertainty", operation_id="balance_uncertainty")
def uncertainty(body: BalanceIn, n: int = 120, user=Depends(current_user)):
    jid = jobs.submit("monte_carlo", lambda progress, **_: jsonable(monte_carlo(_recipe(body.recipe), n=min(n, 400))), user["sub"])
    return {"job_id": jid, "status": "queued"}


@router.post("/balance/sensitivity", operation_id="balance_sensitivity")
def sens(body: BalanceIn, user=Depends(current_user)):
    return jsonable(sensitivity(_recipe(body.recipe)))


@router.post("/economics/kpis", operation_id="economics_kpis")
def econ(body: BalanceIn, user=Depends(current_user)):
    r = _recipe(body.recipe)
    return jsonable({"kpis": kpis(r, body.prices, None, body.conditions), "breakeven": breakeven_tariff(r), "pareto": pareto(r)})
