import copy

from app.domain.material_streams import DEFAULT_RECIPE, default_materials
from app.domain.static_balance import DEFAULT_PRICES, optimise_charge, static_balance


def test_nominal_static_balance_ranges():
    b = static_balance(DEFAULT_RECIPE, default_materials(), None, DEFAULT_PRICES)
    assert b["converged"]
    assert 0 < b["cr_recovery_pct"] <= 100
    assert 55 < b["metal_wt_pct"]["Cr"] < 70
    assert 2500 < b["energy"]["sec_kwh_per_t_fecr"] < 5000
    d = b["cr_distribution_pct"]
    assert abs(d["metal"] + d["slag"] + d["dust"] - 100) < 1e-6
    assert b["gas"]["wet_nm3_per_t_ore"] > b["gas"]["dry_nm3_per_t_ore"]
    assert abs(b["energy"]["sec_kwh_per_t_fecr"] - sum(b["energy"]["breakdown_kwh_per_t_fecr"].values())) < 1.0


def test_carbon_demand_accounting():
    b = static_balance(DEFAULT_RECIPE, default_materials())
    cd = b["carbon_demand_kg_per_t_ore"]
    s = sum(cd["to_CO_by_oxide"].values()) + cd["dissolved_in_metal"]
    assert abs(s - cd["total"]) / cd["total"] < 0.02


def test_more_carbon_raises_si():
    r = copy.deepcopy(DEFAULT_RECIPE)
    r["COKE_MET"] += 30
    a = static_balance(DEFAULT_RECIPE, default_materials())["metal_wt_pct"]["Si"]
    b = static_balance(r, default_materials())["metal_wt_pct"]["Si"]
    assert b > a


def test_optimiser_feasible_and_respects_constraints():
    res = optimise_charge(default_materials(), DEFAULT_RECIPE, ["ORE_A_LUMPY", "ORE_B_PELLET"], ["COKE_MET", "COAL_LA"], ["QUARTZITE"],
                          n_starts=2)
    assert res["feasible"]
    assert res["balance"]["metal_wt_pct"]["Cr"] >= res["constraint_limits"]["cr_min"] - 0.05
    assert res["balance"]["metal_wt_pct"]["Si"] <= res["constraint_limits"]["si_max"] + 0.05


def test_optimiser_reports_infeasibility():
    res = optimise_charge(default_materials(), DEFAULT_RECIPE, ["ORE_C_FRIABLE"], ["COKE_MET"], ["QUARTZITE"],
                          constraints={"cr_min": 75.0}, n_starts=1)
    assert not res["feasible"]
    assert "cr_min" in res["infeasibility_diagnosis"]
