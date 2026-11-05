"""Steady-state mass / heat balance of an HC FeCr SAF and constrained charge-mix optimiser.

Mass: exact element bookkeeping per tonne of ore blend (wet). Distribution of Cr, Fe, Si, Mn,
P, S between metal, slag, gas and dust uses calibrated surrogate recoveries (documented
below, CALIBRATION-REQUIRED; nominal values were fitted to the dynamic twin's synthetic
nominal case, not to a plant).
Energy: first-principles enthalpy balance on the formation-enthalpy basis
  E_furnace = H_out(metal@Tm, slag@Ts, gas@Tg, dust@Tg) - H_in(feed@Tf) + Q_losses
  SEC [kWh/t FeCr] = E_furnace / eta_electrical / t_metal   (saleable basis = tapped metal)
Optimiser: SLSQP, multi-start, constraint margins and finite-difference shadow prices;
infeasible problems return the minimum-violation solution and the violated constraints.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import minimize

from . import metal as mt
from . import slag as sg
from . import species as sp
from .material_streams import RawMaterial, recipe_species_per_t_ore
from .offgas import NM3_PER_MOL
from .thermodynamics import FusionGroup, h_species, zone_enthalpy

AM = {e: sp.ATOMIC_MASS[e] / 1000 for e in sp.ELEMENTS}

DEFAULT_COND = dict(T_metal_K=1910.0, T_slag_K=1905.0, T_gas_K=600.0, T_feed_K=303.15, heat_loss_MW=1.2,
                    furnace_power_MW=35.4, electrical_efficiency=0.98, electrode_paste_kg_per_mwh=3.0,
                    dust_frac_of_charge=0.007, air_infiltration_nm3_per_nm3=0.02,
                    co2_ef_grid_t_per_mwh=0.71, droplet_frac_in_slag=0.008, eta_fe=0.965, eta_mn=0.75, eta_p=0.85,
                    s_to_metal=0.30, carbon_retention_c_sat_frac=0.97, ind_frac_fe2o3=0.6)


def recovery_surrogates(c_ratio: float, T_slag: float, fines_pct: float, slag_superheat: float) -> dict:
    """CALIBRATION-REQUIRED surrogates (fitted to synthetic nominal twin):
    eta_Cr = 0.905 - 0.004(fines-7) + 0.0008(Ts-1905) - 0.6 max(0, 1-c_ratio)     (c_ratio clipped 0.6..1.15)
    Si_wt  = 3.7 exp(0.012 (Ts-1905)) c_ratio^4
    Slag superheat is reported (and constrained in the optimiser) but does not feed back into eta_Cr.
    """
    cr = float(np.clip(c_ratio, 0.6, 1.15))
    eta = 0.905 - 0.004 * (fines_pct - 7.0) + 0.0008 * (T_slag - 1905.0) - 0.6 * max(0.0, 1.0 - cr)
    si = 3.7 * math.exp(0.012 * (T_slag - 1905.0)) * cr ** 4
    return {"eta_cr": float(np.clip(eta, 0.4, 0.97)), "si_wt": float(np.clip(si, 0.2, 12.0))}


def _fines(recipe, mats):
    b = recipe["ore_blend"]
    s = sum(b.values())
    return sum(mats[k].fines_pct * v / s for k, v in b.items())


def static_balance(recipe: dict, mats: dict[str, RawMaterial], cond: dict | None = None, prices: dict | None = None) -> dict:
    c = dict(DEFAULT_COND)
    c.update(cond or {})
    n_feed, masses = recipe_species_per_t_ore(recipe, mats)
    Tm, Ts, Tg, Tf = c["T_metal_K"], c["T_slag_K"], c["T_gas_K"], c["T_feed_K"]
    fines = _fines(recipe, mats)
    el_feed = sp.E_MATRIX @ n_feed  # mol of each element
    E = {e: el_feed[i] for i, e in enumerate(sp.ELEMENTS)}
    charge_kg = sum(masses.values())
    # dust (charge composition, before reactions)
    n_dust = n_feed * c["dust_frac_of_charge"] * (fines / 8.0)
    el_dust = sp.E_MATRIX @ n_dust
    Ed = {e: el_dust[i] for i, e in enumerate(sp.ELEMENTS)}
    n_eff = n_feed - n_dust
    sec_guess = 3300.0
    res = None
    superheat = 50.0
    c_ratio = 1.0
    for _ in range(8):
        # electrode carbon from SEC guess and metal guess
        metal_guess = (res["metal_kg"] if res else 420.0)
        n_el_C = c["electrode_paste_kg_per_mwh"] * sec_guess / 1000 * metal_guess / 1000 / AM["C"]
        sur = recovery_surrogates(c_ratio, Ts, fines, superheat)
        eta_cr = sur["eta_cr"]
        g = lambda k: float(n_eff[sp.IDX[k]])
        # element pools available for the metal (excluding dust)
        Cr_t = E["Cr"] - Ed["Cr"]
        Fe_t = E["Fe"] - Ed["Fe"]
        Mn_t = E["Mn"] - Ed["Mn"]
        P_t = E["P"] - Ed["P"]
        S_t = E["S"] - Ed["S"]
        Si_t = E["Si"] - Ed["Si"]
        cr_met_prefeed = g("Cr")
        fe_met_prefeed = g("Fe")
        m_Cr = eta_cr * Cr_t
        m_Fe = c["eta_fe"] * Fe_t
        m_Mn = c["eta_mn"] * Mn_t
        m_P = c["eta_p"] * P_t
        m_S = c["s_to_metal"] * S_t
        base = m_Cr * AM["Cr"] + m_Fe * AM["Fe"] + m_Mn * AM["Mn"] + m_P * AM["P"] + m_S * AM["S"]
        si_wt = sur["si_wt"]
        cr_wt = 60.0
        for _i in range(20):
            c_wt = c["carbon_retention_c_sat_frac"] * mt.c_saturation_wt(Tm, cr_wt, si_wt)
            metal_kg = base / (1 - (c_wt + si_wt) / 100)
            cr_wt = 100 * m_Cr * AM["Cr"] / metal_kg
        m_Si = min(si_wt / 100 * metal_kg / AM["Si"], 0.9 * Si_t)
        m_C = c_wt / 100 * metal_kg / AM["C"]
        metal_kg = base + m_Si * AM["Si"] + m_C * AM["C"]
        # oxygen removed by carbon (to CO), indirect Fe2O3 share to CO2
        cr_red = m_Cr - cr_met_prefeed
        fe_red = m_Fe - fe_met_prefeed
        O_cr = 1.5 * max(cr_red, 0)
        O_fe = max(fe_red, 0)
        fe2o3 = g("Fe2O3")
        O_fe2o3 = fe2o3  # Fe2O3 -> 2 FeO releases 1 O per Fe2O3
        O_si = 2 * m_Si
        O_mn = m_Mn
        O_p = 2.5 * m_P
        C_to_CO = O_cr + O_fe + O_si + O_mn + O_p + (1 - c["ind_frac_fe2o3"]) * O_fe2o3
        CO_to_CO2 = c["ind_frac_fe2o3"] * O_fe2o3
        C_fixed = g("C") + g("C_m")
        C_need = C_to_CO + m_C - g("C_m")
        C_avail = C_fixed + n_el_C
        c_ratio_new = C_avail / max(C_need, 1e-9)
        # slag
        slag_mol = {
            "SiO2": g("SiO2") - m_Si, "Al2O3": g("Al2O3") + g("MgAl2O4"), "MgO": g("MgO") + g("MgAl2O4") + g("MgCr2O4") + g("MgCO3"),
            "CaO": g("CaO") + g("CaCO3"), "TiO2": g("TiO2"), "MnO": max(g("MnO") - m_Mn, 0), "P2O5": max(g("P2O5") - m_P / 2, 0),
            "Cr2O3": max(Cr_t - m_Cr, 0) / 2, "FeO": max(Fe_t - m_Fe, 0),
        }
        s_slag = S_t - m_S
        slag_mol["CaS"] = s_slag * 0.9
        slag_mol["CaO"] -= slag_mol["CaS"]
        slag_kg_ox = {k: v * sp.SPECIES[k].molar_mass for k, v in slag_mol.items() if v > 0}
        drop_kg = c["droplet_frac_in_slag"] * sum(slag_kg_ox.values())
        slag_kg = sum(slag_kg_ox.values()) + drop_kg
        wt = {k: 100 * v / slag_kg for k, v in slag_kg_ox.items()}
        # Cr2O3 above solubility is carried as undissolved (Mg)Cr2O4 spinel -> excluded from the liquid
        b2 = sg.basicities(wt)["B2_(CaO+MgO)/(SiO2+Al2O3)"]
        sol = sg.cr2o3_solubility_wt(Ts, b2)
        excess = max(wt.get("Cr2O3", 0.0) - sol, 0.0)
        liq = dict(wt)
        liq["Cr2O3"] = wt.get("Cr2O3", 0.0) - excess
        mgo_sp = excess * 40.30 / 151.99
        liq["MgO"] = max(liq.get("MgO", 0.0) - mgo_sp, 0.0)
        solid_frac = (excess + mgo_sp) / 100.0
        props = sg.slag_properties(liq, Ts, solid_frac)
        props["undissolved_spinel_wt_pct"] = 100 * solid_frac
        superheat = props["superheat_K"]
        res = {"metal_kg": metal_kg}
        converged = abs(c_ratio_new - c_ratio) < 1e-4
        c_ratio = 0.5 * c_ratio + 0.5 * min(c_ratio_new, 1.3)
        # energy (formation-enthalpy basis)
        n_metal = sp.vec(Cr=m_Cr, Fe=m_Fe, Mn=m_Mn, P=m_P, S=m_S, Si=m_Si, C_m=m_C)
        n_slag = sp.vec(**{k: max(v, 0) for k, v in slag_mol.items()})
        n_gas = sp.vec(CO=C_to_CO - CO_to_CO2 + max(C_avail - C_need, 0) * 0.0, CO2=CO_to_CO2 + g("CaCO3") + g("MgCO3"),
                       H2O=g("H2O_l"), CH4=g("VOL"))
        n_gas[sp.IDX["CO"]] -= 0  # carbon surplus stays in coke bed (reported separately)
        air = c["air_infiltration_nm3_per_nm3"] * float(n_gas.sum())
        n_gas[sp.IDX["N2"]] += 0.79 * air
        n_gas[sp.IDX["O2"]] += 0.21 * air
        fm = FusionGroup("metal", sp.METAL_MASK, 2.7e5, Tm - 120, Tm - 1)
        fsl = FusionGroup("slag", sp.SLAG_OXIDE_MASK, 4.2e5, props["liquidus_K"] - 180, props["liquidus_K"])
        H_metal = zone_enthalpy(n_metal, Tm, [fm]) * (1 + drop_kg / max(metal_kg, 1) * 0)
        H_slag = zone_enthalpy(n_slag, Ts, [fsl]) + drop_kg / max(metal_kg, 1e-9) * zone_enthalpy(n_metal, Ts, [fm])
        H_gas = float(n_gas @ h_species(Tg))
        H_dust = float(n_dust @ h_species(Tg))
        n_el = sp.vec(C_el=n_el_C)
        H_in = float(n_feed @ h_species(Tf)) + float(n_el @ h_species(298.15)) + 0.21 * air * float(h_species(Tf)[sp.IDX["O2"]]) \
            + 0.79 * air * float(h_species(Tf)[sp.IDX["N2"]])
        dH = H_metal + H_slag + H_gas + H_dust - H_in
        # carbon surplus to coke bed is a storage term (enthalpy at feed T already in H_in; heated to ~1950 K)
        C_surplus = max(C_avail - C_need, 0.0)
        dH += C_surplus * float(h_species(1950.0)[sp.IDX["C"]] - h_species(Tf)[sp.IDX["C"]])
        prod_t_h = c["furnace_power_MW"] / max(sec_guess / 1000 * c["electrical_efficiency"], 0.5)
        loss_J = c["heat_loss_MW"] * 1e6 * 3600 / max(prod_t_h, 0.1) * (metal_kg / 1000)
        E_furn = dH + loss_J
        sec = E_furn / 3.6e6 / c["electrical_efficiency"] / (metal_kg / 1000)
        sec_guess = 0.5 * sec_guess + 0.5 * sec
        if converged and abs(sec - sec_guess) < 1:
            break
    tm = metal_kg / 1000
    gas_dry = {k: float(n_gas[sp.IDX[k]]) for k in ("CO", "CO2", "N2", "O2", "CH4", "H2")}
    tot_dry = sum(gas_dry.values())
    nm3_dry = tot_dry * NM3_PER_MOL
    nm3_wet = (tot_dry + n_gas[sp.IDX["H2O"]]) * NM3_PER_MOL
    metal_wt = mt.element_wt({"Cr": m_Cr * AM["Cr"], "Fe": m_Fe * AM["Fe"], "C": m_C * AM["C"], "Si": m_Si * AM["Si"],
                              "Mn": m_Mn * AM["Mn"], "P": m_P * AM["P"], "S": m_S * AM["S"]})
    # carbon / CO2 accounting (scope 1 assumes all furnace-gas carbon is ultimately oxidised to CO2)
    C_in_kg = (C_avail + g("CaCO3") + g("MgCO3") + g("VOL")) * AM["C"]
    C_metal_kg = m_C * AM["C"]
    co2_scope1 = (C_in_kg - C_metal_kg - C_surplus * AM["C"]) * 44.01 / 12.011 / 1000
    E_MWh = sec * tm / 1000
    co2_scope2 = E_MWh * c["co2_ef_grid_t_per_mwh"]
    out = {
        "basis": "per t of ore blend (wet) unless stated; SYNTHETIC / NOT PLANT-VALIDATED",
        "conditions": c, "converged": bool(converged), "carbon_ratio_supply_to_demand": float(c_ratio_new),
        "fines_pct": fines, "masses_in_kg_per_t_ore": masses, "charge_kg_per_t_ore": charge_kg,
        "electrode_carbon_kg_per_t_ore": n_el_C * AM["C"],
        "metal_kg_per_t_ore": metal_kg, "ore_t_per_t_fecr": 1.0 / tm, "reductant_fixed_C_kg_per_t_fecr": C_fixed * AM["C"] / tm,
        "metal_wt_pct": metal_wt, "metal_props": mt.metal_properties(metal_wt, Tm),
        "slag_kg_per_t_ore": slag_kg, "slag_to_metal": slag_kg / metal_kg, "slag_wt_pct": wt, "slag_props": props,
        "slag_cr2o3_equiv_wt_pct": sg.cr2o3_equivalent(wt),
        "cr_recovery_pct": 100 * m_Cr / E["Cr"], "fe_recovery_pct": 100 * m_Fe / E["Fe"],
        "cr_distribution_pct": {"metal": 100 * m_Cr / E["Cr"], "slag": 100 * (Cr_t - m_Cr) / E["Cr"], "dust": 100 * Ed["Cr"] / E["Cr"]},
        "carbon_demand_kg_per_t_ore": {"to_CO_by_oxide": {"Cr": O_cr * AM["C"], "Fe": O_fe * AM["C"], "Si": O_si * AM["C"],
                                                           "Mn": O_mn * AM["C"], "P": O_p * AM["C"],
                                                           "Fe2O3_direct": (1 - c["ind_frac_fe2o3"]) * O_fe2o3 * AM["C"]},
                                       "dissolved_in_metal": m_C * AM["C"], "total": C_need * AM["C"],
                                       "available_fixed_plus_electrode": C_avail * AM["C"], "surplus_to_coke_bed": C_surplus * AM["C"]},
        "gas": {"dry_vol_pct": {k: 100 * v / tot_dry for k, v in gas_dry.items()} if tot_dry > 0 else {},
                "dry_nm3_per_t_ore": nm3_dry, "wet_nm3_per_t_ore": nm3_wet, "dry_nm3_per_t_fecr": nm3_dry / tm,
                "wet_nm3_per_t_fecr": nm3_wet / tm, "h2o_kg_per_t_ore": float(n_gas[sp.IDX["H2O"]]) * 0.018015,
                "chemical_energy_GJ_per_t_fecr": (n_gas[sp.IDX["CO"]] * 282.98e3 + n_gas[sp.IDX["CH4"]] * 802.3e3) / 1e9 / tm},
        "dust_kg_per_t_ore": float(np.sum(n_dust * sp.MW)),
        "energy": {"sec_kwh_per_t_fecr": sec, "electric_MWh_per_t_ore": E_MWh,
                   "breakdown_kwh_per_t_fecr": {
                       "metal_enthalpy_out": (H_metal - float(n_metal @ h_species(298.15))) / 3.6e6 / tm,
                       "slag_enthalpy_out": (H_slag - float(n_slag @ h_species(298.15))) / 3.6e6 / tm,
                       "gas_sensible_out": (H_gas - float(n_gas @ h_species(298.15))) / 3.6e6 / tm,
                       "reaction_enthalpy_net": (float(n_metal @ h_species(298.15)) + float(n_slag @ h_species(298.15)) +
                                                 float(n_gas @ h_species(298.15)) + float(n_dust @ h_species(298.15)) - H_in) / 3.6e6 / tm,
                       "dust_sensible_and_coke_bed_heating": ((H_dust - float(n_dust @ h_species(298.15))) +
                                                              C_surplus * float(h_species(1950.0)[sp.IDX["C"]] - h_species(Tf)[sp.IDX["C"]])) / 3.6e6 / tm,
                       "heat_losses": loss_J / 3.6e6 / tm,
                       "electrical_network_losses": sec * (1 - c["electrical_efficiency"])},
                   "basis": "formation-enthalpy basis, reference 298.15 K; SEC on tapped metal, transformer secondary"},
        "emissions": {"scope1_t_co2_per_t_fecr": co2_scope1 / tm, "scope2_t_co2_per_t_fecr": co2_scope2 / tm,
                      "grid_ef_t_per_mwh": c["co2_ef_grid_t_per_mwh"],
                      "note": "scope 1 assumes furnace-gas CO is flared/combusted to CO2; avoided emissions from gas use are reported separately in economics without double counting"},
        "surrogates": recovery_surrogates(c_ratio, Ts, fines, superheat),
    }
    if prices:
        out["cost"] = cost_per_t(out, recipe, mats, prices)
    return out


def cost_per_t(bal: dict, recipe: dict, mats: dict, prices: dict) -> dict:
    tm = bal["metal_kg_per_t_ore"] / 1000
    mat_cost = {k: m / 1000 * mats[k].price_inr_per_t / tm for k, m in bal["masses_in_kg_per_t_ore"].items()}
    elec = bal["energy"]["sec_kwh_per_t_fecr"] / 1000 * prices.get("electricity_inr_per_mwh", 6500.0)
    paste = bal["electrode_carbon_kg_per_t_ore"] / tm / 1000 * prices.get("paste_inr_per_t", 45000.0)
    co2 = (bal["emissions"]["scope1_t_co2_per_t_fecr"] + bal["emissions"]["scope2_t_co2_per_t_fecr"]) * prices.get("co2_inr_per_t", 0.0)
    other = prices.get("other_inr_per_t_fecr", 2500.0)
    total = sum(mat_cost.values()) + elec + paste + co2 + other
    return {"materials_inr_per_t": mat_cost, "electricity_inr_per_t": elec, "electrode_paste_inr_per_t": paste,
            "co2_inr_per_t": co2, "other_inr_per_t": other, "total_inr_per_t_fecr": total, "currency": "INR (DEMO prices)"}


DEFAULT_PRICES = {"electricity_inr_per_mwh": 6500.0, "paste_inr_per_t": 45000.0, "co2_inr_per_t": 0.0, "other_inr_per_t_fecr": 2500.0}

DEFAULT_CONSTRAINTS = {"cr_min": 52.0, "si_max": 4.0, "c_max": 8.5, "p_max": 0.035, "c_ratio_lo": 0.98, "c_ratio_hi": 1.06,
                       "superheat_min_K": 40.0, "b1_lo": 0.6, "b1_hi": 1.1, "viscosity_max_Pa_s": 1.2}


def optimise_charge(mats: dict[str, RawMaterial], base_recipe: dict, ores: list[str], reductants: list[str],
                    fluxes: list[str], constraints: dict | None = None, prices: dict | None = None,
                    cond: dict | None = None, bounds: dict | None = None, n_starts: int = 4, seed: int = 0) -> dict:
    cons_lim = dict(DEFAULT_CONSTRAINTS)
    cons_lim.update(constraints or {})
    prices = dict(DEFAULT_PRICES) | (prices or {})
    bounds = bounds or {}
    nO = len(ores)
    var = ores + reductants + fluxes
    lb = [bounds.get(k, (0.0, 1.0))[0] if k in ores else bounds.get(k, (0.0, 400.0))[0] for k in var]
    ub = [bounds.get(k, (0.0, 1.0))[1] if k in ores else bounds.get(k, (0.0, 400.0 if k in reductants else 250.0))[1] for k in var]

    def recipe_of(x):
        r = copy.deepcopy(base_recipe)
        s = sum(x[:nO])
        r["ore_blend"] = {o: max(x[i], 0) / s for i, o in enumerate(ores)} if s > 0 else {ores[0]: 1.0}
        for i, k in enumerate(reductants + fluxes):
            r[k] = float(max(x[nO + i], 0.0))
        return r

    cache = {}

    def bal(x):
        key = tuple(np.round(x, 6))
        if key not in cache:
            cache[key] = static_balance(recipe_of(x), mats, cond, prices)
        return cache[key]

    def g_list(x):
        b = bal(x)
        mw = b["metal_wt_pct"]
        sp_ = b["slag_props"]
        b1 = sp_["basicity"]["B1_(CaO+MgO)/SiO2"]
        return {
            "cr_min": mw.get("Cr", 0) - cons_lim["cr_min"], "si_max": cons_lim["si_max"] - mw.get("Si", 0),
            "c_max": cons_lim["c_max"] - mw.get("C", 0), "p_max": (cons_lim["p_max"] - mw.get("P", 0)) * 100,
            "c_ratio_lo": (b["carbon_ratio_supply_to_demand"] - cons_lim["c_ratio_lo"]) * 100,
            "c_ratio_hi": (cons_lim["c_ratio_hi"] - b["carbon_ratio_supply_to_demand"]) * 100,
            "superheat": (sp_["superheat_K"] - cons_lim["superheat_min_K"]) / 10,
            "b1_lo": (b1 - cons_lim["b1_lo"]) * 10, "b1_hi": (cons_lim["b1_hi"] - b1) * 10,
            "viscosity": (cons_lim["viscosity_max_Pa_s"] - sp_["viscosity_liquid_Pa_s"]) * 10,
        }

    names = list(g_list(np.array([(lb[i] + ub[i]) / 2 for i in range(len(var))])).keys())

    def obj(x):
        return bal(x)["cost"]["total_inr_per_t_fecr"] / 1000.0

    cons = [{"type": "ineq", "fun": (lambda x, n=n: g_list(x)[n])} for n in names]
    cons.append({"type": "eq", "fun": lambda x: sum(x[:nO]) - 1.0})
    rng = np.random.default_rng(seed)
    x0s = []
    base = []
    for k in var:
        if k in ores:
            base.append(base_recipe["ore_blend"].get(k, 0.0))
        else:
            base.append(float(base_recipe.get(k, 0.0)))
    base = np.clip(np.array(base), lb, ub)
    if base[:nO].sum() > 0:
        base[:nO] /= base[:nO].sum()
    x0s.append(base)
    for _ in range(n_starts - 1):
        x = np.array([rng.uniform(lb[i], ub[i]) for i in range(len(var))])
        x[nO:] = base[nO:] * rng.uniform(0.8, 1.2, len(var) - nO)
        x[:nO] = x[:nO] / max(x[:nO].sum(), 1e-9)
        x0s.append(np.clip(x, lb, ub))
    best = None
    for x0 in x0s:
        try:
            r = minimize(obj, x0, method="SLSQP", bounds=list(zip(lb, ub)), constraints=cons,
                         options={"maxiter": 150, "ftol": 1e-6, "eps": 1e-3})
        except Exception as exc:  # noqa: BLE001
            continue
        g = g_list(r.x)
        feas = all(v >= -1e-3 for v in g.values()) and abs(sum(r.x[:nO]) - 1) < 1e-4
        if best is None or (feas and (not best["feasible"] or r.fun < best["fun"])) or (not best["feasible"] and not feas and
                                                                                         sum(min(v, 0) for v in g.values()) > best["viol"]):
            best = {"x": r.x, "fun": float(r.fun), "feasible": feas, "viol": sum(min(v, 0) for v in g.values()), "msg": r.message}
    x = best["x"]
    g = g_list(x)
    b = bal(x)
    result = {"feasible": best["feasible"], "solver_message": str(best["msg"]), "recipe": recipe_of(x), "balance": b,
              "constraint_margins": g, "active_constraints": [k for k, v in g.items() if abs(v) < 0.05],
              "violated_constraints": [k for k, v in g.items() if v < -1e-3], "constraint_limits": cons_lim,
              "objective_inr_per_t_fecr": b["cost"]["total_inr_per_t_fecr"], "evaluations": len(cache)}
    # shadow prices: re-solve with each active constraint relaxed by a small step
    shadow = {}
    steps = {"cr_min": ("cr_min", -0.5), "si_max": ("si_max", 0.5), "superheat": ("superheat_min_K", -5.0),
             "c_ratio_lo": ("c_ratio_lo", -0.01), "b1_lo": ("b1_lo", -0.05), "b1_hi": ("b1_hi", 0.05)}
    if best["feasible"]:
        for a in result["active_constraints"]:
            if a in steps:
                key, d = steps[a]
                lim2 = dict(cons_lim)
                lim2[key] = lim2[key] + d
                # cheap local estimate: one SLSQP from current optimum with relaxed limit
                cons_lim_bak = dict(cons_lim)
                cons_lim[key] = lim2[key]
                try:
                    rr = minimize(obj, x, method="SLSQP", bounds=list(zip(lb, ub)), constraints=cons,
                                  options={"maxiter": 60, "ftol": 1e-6, "eps": 1e-3})
                    shadow[a] = {"delta_limit": d, "limit": key, "delta_cost_inr_per_t": float(rr.fun * 1000 - result["objective_inr_per_t_fecr"]),
                                 "per_unit": float((rr.fun * 1000 - result["objective_inr_per_t_fecr"]) / abs(d))}
                finally:
                    cons_lim.clear()
                    cons_lim.update(cons_lim_bak)
    result["shadow_prices"] = shadow
    if not best["feasible"]:
        result["infeasibility_diagnosis"] = {k: v for k, v in sorted(g.items(), key=lambda kv: kv[1]) if v < 0}
    return result
