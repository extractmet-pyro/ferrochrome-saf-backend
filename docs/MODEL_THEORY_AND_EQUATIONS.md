# FeCr-TWIN — Model theory and equations

*Ferrochrome submerged arc furnace digital twin, process simulator and supervisory advisory system.*
*Extractmet Private Limited · Version 1.0 · SYNTHETIC / NOT PLANT-VALIDATED*

This document states exactly what the software computes. Every equation below maps to a module in
`app/domain`, `app/twin`, `app/ml` or `app/control`. Labels: **SOURCE-VERIFIED** (standard data or
published model used as published), **CALIBRATION-REQUIRED** (structure sound, parameters must be fitted
to site data), **HYPOTHESIS** (modelling assumption to be tested), **DEMO-ONLY** (value chosen for the
synthetic plant).

---

## 1. Model fidelity statement

The real-time core is a **lumped, multi-zone, dynamic, element- and enthalpy-conserving model**:

| Zone | Symbol | Physical content | Main phenomena |
|---|---|---|---|
| Upper burden | B1 | charge, moisture, volatiles | drying, devolatilisation, MgCO3 decomposition, water–gas shift, counter-current gas preheat, dust pick-up |
| Lower burden | B2 | preheated charge, coke | CaCO3 calcination, indirect (CO) reduction of Fe oxides, Boudouard, onset of solid-state carbothermic chromite reduction, side-path Joule heat |
| Coke bed / reaction zone | CB | coke bed around electrode tips, incoming solids | solid-state reduction of chromite spinels, burden melting, electrode carbon, main Joule heating |
| Slag | SL | dissolved oxides, undissolved chromite spinel, metal droplets | chromite dissolution, interfacial reduction by coke-bed carbon, Cr(III)/Cr(II), Si reduction, carburisation, silicothermic exchange, desulphurisation, droplet settling |
| Metal bath | MB | HC FeCr (Fe–Cr–C–Si–Mn–P–S, M7C3 carbide) | carbide equilibrium, bath carburisation |
| Gas / freeboard | FB | CO-rich furnace gas, dust, infiltrated air | quasi-steady transit, combustion (open/semi-closed), roof heat transfer, analyzer lag |
| Refractory walls | — | hearth, metal/slag/burden sidewall bands, roof | transient multi-layer conduction, convection + radiation + water cooling, freeze lining |

It does **not** resolve 3-D current paths, burden flow fields, coke-bed shape or slag/metal flow. A
high-fidelity FEM/DEM/CFD model can supply validated surrogate corrections offline (the ML residual layer
and `ThermoProvider` adapter are the integration points); it is never the real-time core.

---

## 2. Species, element matrix and phase inventories (`domain/species.py`)

38 species: oxides (Cr2O3, CrO, FeO, Fe2O3, SiO2, Al2O3, MgO, CaO, MnO, TiO2, P2O5), spinel end-members
(FeCr2O4, MgCr2O4, MgAl2O4), carbonates, free moisture, coal volatiles (CH4-equivalent), carbon (coke,
electrode, dissolved), metals (Fe, Cr, Si, Mn, P, S), carbide Cr7C3, CaS and gases (CO, CO2, H2, H2O, N2, O2,
CH4). Elements: Cr, Fe, C, O, Si, Mg, Al, Ca, Mn, Ti, S, P, H, N.

Element matrix **E** (elements × species). For a species-mole vector **n**:

    element moles = E · n ,   element mass_e = (E · n)_e · M_e

Every reaction ν (species coefficients) satisfies **E · ν = 0** — verified for all 29 reactions by
`tests/unit/test_stoichiometry_units.py` (max imbalance 0 to machine precision).

**Chromite mineralogy** (`material_streams.allocate_mineralogy`, HYPOTHESIS): the dry bulk assay is
allocated to spinel end-members so that mineral structure (which drives dissolution and solid-state
kinetics) is distinct from bulk chemistry (which drives bookkeeping):

    FeCr2O4 = min(FeO, Cr2O3)
    MgAl2O4 = min(f_Al · Al2O3, MgO)        f_Al = 0.85 (CALIBRATION-REQUIRED, from XRD/QEMSCAN)
    MgCr2O4 = min(Cr2O3 remaining, MgO remaining)
    remaining Cr2O3, FeO, MgO, Al2O3 stay as free oxides; Fe2O3 is kept separate (Fe3+).

The allocation conserves every element exactly (unit test).

**Bases.** Assays are wt% dry; moisture is wt% of the wet (as-received) mass; LOI is mapped to bound
water. Mass fractions (sum ≈ 1) are rejected, totals outside 100 ± 2 wt% are rejected, unknown speciation is
rejected. Gas volumes are Nm³ at 273.15 K and 101.325 kPa (22.414 L/mol); dry and wet bases are always
labelled; time is stored in UTC.

---

## 3. Thermochemistry (`domain/thermodynamics.py`)

Molar enthalpy, referenced to the elements at 298.15 K:

    h_i(T) = ΔH°f,i + ∫298^T Cp,i dT ,   Cp,i = a + b·T + c/T²      (Maier–Kelley)
    ∫ = a(T−298.15) + (b/2)(T² − 298.15²) − c(1/T − 1/298.15)

Data: NIST-JANAF and Kubaschewski–Alcock–Spencer (SOURCE-VERIFIED, rounded); spinel, CrO, carbide and
dissolved-element data are CALIBRATION-REQUIRED or HYPOTHESIS (see `species.py`).

**Zone enthalpy with phase-change groups**

    H_zone(T, n) = Σ_i n_i h_i(T) + Σ_g L_g · m_g · f_g(T)
    f_g(T) = 0 below solidus, 1 above liquidus, linear in between

Groups: molten slag oxides (L ≈ 0.42 MJ/kg, liquidus from the slag surrogate, solidus = liquidus − 180 K)
and molten metal (L ≈ 0.27 MJ/kg, liquidus from the metal surrogate, solidus = liquidus − 120 K). Latent heats
are CALIBRATION-REQUIRED.

**Temperature** is never integrated directly: each zone stores H (J) and n (mol); T is obtained by solving
H_zone(T, n) = H with safeguarded Newton–bisection (`invert_temperature`). Consequences:

* reaction heats are implicit in formation enthalpies → **no double counting** of endothermic heat;
* latent heat is absorbed/released automatically when material crosses a liquidus;
* energy closure is exact by construction and is independently audited by the ledger (§11).

---

## 4. Reaction network: equilibrium vs rate (`domain/reactions.py`, `domain/kinetics.py`)

### 4.1 Reactions and equilibrium data (ΔG° = A + B·T, J per mole of reaction)

| Key | Reaction | A | B | Zone | Basis |
|---|---|---|---|---|---|
| R01 | Fe2O3 + C → 2FeO + CO | 172 600 | −207.6 | reaction | Hess (illustrative overall step) |
| R02 | FeO + C → Fe + CO | 149 600 | −150.4 | reaction/slag | Turkdogan |
| R03 | Cr2O3 + 3C → 2Cr + 3CO | 766 940 | −504.63 | slag/CB | **lumped overall net** |
| R04 | Cr2O3 + C → 2CrO + CO | 327 740 | −205.49 | slag | HYPOTHESIS |
| R05 | CrO + C → Cr + CO | 219 600 | −149.57 | slag | HYPOTHESIS |
| R06 | SiO2 + 2C → Si + 2CO | 678 300 | −346.5 | slag | Turkdogan |
| R07 | MnO + C → Mn + CO | 274 500 | −162.07 | slag | Turkdogan |
| R08 | P2O5 + 5C → 2P + 5CO | 920 000 | −908.85 | slag | CALIBRATION-REQUIRED |
| R09 | C + CO2 → 2CO | 170 700 | −174.5 | B2 | Boudouard |
| R10 | CO + H2O → CO2 + H2 | −36 000 | 32.0 | B1 | water–gas shift |
| R11 | CaCO3 → CaO + CO2 | 161 300 | −137.2 | B2 | calcination |
| R12 | MgCO3 → MgO + CO2 | 117 600 | −170.0 | B1 | CALIBRATION-REQUIRED |
| R13 | H2O(l) → H2O(g) | — | — | B1/B2 | heat-limited drying (rate only) |
| R14 | VOL → CH4 | — | — | B1/B2 | devolatilisation (rate only) |
| R15 | 7Cr + 3C[m] → Cr7C3 | −228 000 | −17.1 | metal | carbide surrogate |
| R16 | C → C[m] | — | — | slag/metal | carburisation (rate only) |
| R17 | FeCr2O4 + 4C → Fe + 2Cr + 4CO | 968 540 | −667.03 | B2/CB | spinel solid-state |
| R18 | MgCr2O4 + 3C → MgO + 2Cr + 3CO | 808 940 | −510.63 | B2/CB | spinel solid-state |
| R19–21 | spinel → constituent oxides (dissolution) | — | — | slag | mass-transfer controlled |
| R22 | Fe2O3 + CO → 2FeO + CO2 | 1 900 | −33.1 | B2 | indirect |
| R23 | FeO + CO → Fe + CO2 | −21 100 | 24.1 | B2 | indirect |
| R24 | CaO + S + C → CaS + CO | — | — | slag | sulphur partition L_S |
| R25–27 | CO, H2, CH4 combustion | — | — | freeboard | open / semi-closed |
| R28 | C(electrode) → C | — | — | CB | electrode consumption |
| R29 | 3Si + 2Cr2O3 → 3SiO2 + 4Cr | −501 020 | 30.36 | slag | silicothermic exchange |

R03 is explicitly a **lumped net reaction**, not a mechanism; the model also carries the parallel Cr(III) →
Cr(II) → Cr path (R04/R05) and spinel solid-state paths (R17/R18).

### 4.2 Driving force (feasibility) — separated from the rate

    DF_j = max(0, 1 − Q_j / K_j(T)) ,   K_j = exp(−ΔG°_j / RT)

Q uses activities: slag a_i = γ_i x_i (γ_FeO 2.0, γ_Cr2O3 1.2, γ_CrO 1.5, γ_SiO2 1.0, γ_MnO 1.5),
metal a_i = γ_i x_i (γ_Si 0.01 reflecting the strong negative deviation of Si in Fe–Cr–C melts), carbon at
unit activity, p_CO = 1 atm at the coke interface. All γ are CALIBRATION-REQUIRED; a licensed FactSage/ChemApp
installation can replace them through `FactSageAdapter`.

### 4.3 Rate laws

    r_j = k_j(T) · n_lim · S_j · DF_j ,     k_j = A_j exp(−E_j / RT)          [mol/s]

* Solid-state reduction: S = carbon-contact factor min(1, n_C / (4 n_spinel)); E ≈ 300–330 kJ/mol.
* Interfacial slag reduction: S = φ_C = min(fill, 1.5)^(2/3), fill = coke-bed carbon / reference
  (interfacial area scales with coke-bed size). Interface temperature T = (T_CB + T_SL)/2.
* SiO2 reduction carries a high activation energy (470 kJ/mol) so Si pick-up is strongly temperature- and
  carbon-sensitive, as observed in practice.
* Boudouard rate ∝ n_C · y_CO2 · DF, scaled by a coke-reactivity factor (scenario D).
* Water–gas shift: approach to equilibrium on the gas passing through B1, extent from the quadratic
  (CO2+x)(H2+x) = K(CO−x)(H2O−x), times an approach factor 1 − exp(−k(T)τ).

Extents within each sub-step are limited so that no reactant inventory becomes negative, including shared
reactants (`limit_extents`). Pre-exponential factors were calibrated so the synthetic nominal case lands in
published HC FeCr ranges; they are CALIBRATION-REQUIRED for any real furnace.

### 4.4 Mass transfer and transport

**Chromite dissolution** (shrinking particles, mass-transfer control):

    dm/dt = − k_m(T) · A_p · ρ_slag · (w_sat − w_Cr2O3eq) / 100
    A_p = 3 M_sp / (ρ_sp · r̄),  r̄ = (3 M_sp / (4π ρ_sp N_p))^(1/3)
    k_m(T) = k_m,ref · exp(−E/R (1/T − 1/1973 K)),  k_m,ref = 0.8 µm/s (CALIBRATION-REQUIRED)

N_p (particle number) is added with incoming spinel at the grain radius (0.25 mm) and removed by tapping.
**w_sat** (Cr2O3 solubility) = 3.0 + 0.025 (T°C − 1550) + 2.0 (B2 − 1) wt% (surrogate). Reported tap-slag
Cr2O3 above solubility is largely undissolved chromite — the main kinetic Cr loss, as reported in the
ferrochrome literature (e.g. Visser et al., *Minerals* 2023).

**Carburisation** (Nernst–Brunner, toward Fe–Cr–C saturation):

    dn_C/dt = k_c · (C_sat − C) / 100 · m_metal / M_C
    C_sat [wt%] = 1.34 + 2.54·10⁻³ T[°C] + 0.040 Cr − 0.30 Si       (CALIBRATION-REQUIRED)

**Carbide**: the carbide-bound share of carbon relaxes to φ_eq = min(0.9, 1.3 x_Cr / (x_Cr + x_Fe)).

**Droplet settling** (Stokes): v = 2 g Δρ r² / (9 η_eff); fraction settling per second = v / h_slag;
r = 0.6 mm (CALIBRATION-REQUIRED). Metal droplets left in tapped slag are reported as Cr loss.

**Slag viscosity**: Urbain model (Urbain et al. 1981; Mills, *Slag Atlas*) with Einstein–Roscoe correction
η_eff = η (1 − 1.35 f_s)^−2.5 for suspended spinel.

**Burden**: voidage ε = 0.42 − 0.006·fines%; Ergun pressure-drop proxy; gas–solid heat-exchange
effectiveness and a permeability index fall with fines (scenarios C, I).

---

## 5. Slag and metal chemistry (`domain/slag.py`, `domain/metal.py`)

Basicity indices are named and never interchanged:

    B1 = (CaO + MgO) / SiO2          B2 = (CaO + MgO) / (SiO2 + Al2O3)          MgO/Al2O3

Slag liquidus (surrogate, CALIBRATION-REQUIRED, domain SiO2 22–50, MgO 15–42, Al2O3 10–35, CaO 0–20,
Cr2O3-eq 1–20 wt%; flags raised outside):

    T_liq[°C] = 1340 + 10.5 MgO + 3.0 Al2O3 − 4.0 SiO2 − 6.0 CaO + 12.0 Cr2O3eq − 5.0 FeO

Metal liquidus (HC FeCr surrogate): T_liq[°C] = 1520 + 3.0 (Cr − 60) − 20 (C − 7) − 8 (Si − 2).
Cr2O3-equivalent = Cr2O3 + CrO · M(Cr2O3)/(2 M(CrO)).

---

## 6. Electrical model (`domain/electrical.py`)

Star-connected secondary windings feeding three electrodes; furnace star point floating. Phasors:

    V_k = (V_LL/√3)·e^{−j2πk/3},  Y_k = 1 / (R_net + R_load,k + jX_k)
    V_n = Σ V_k Y_k / Σ Y_k ,  I_k = (V_k − V_n) Y_k
    P_k = |I_k|² R_load,k   (into furnace),  P_net = |I_k|² R_net,  Q_k = |I_k|² X_k

Power factor, apparent power, current unbalance and negative-sequence ratio are derived from the phasors —
never from a DC formula. Load resistance per phase:

    R_cb  = ρ_cb / (2π r_e (1 + h/r_e))                    tip path, coke bed → slag
    R_bur = ρ_bur / (2π h) · ln(PCD / D_e)                   side path, burden
    R_load = R_cb ∥ R_bur
    ρ_cb = ρ_cb0 · exp(1500 (1/T_CB − 1/1950)) · fill^−0.8 · (phase factor)

ρ_cb0 is **calibrated at start-up** so that the nominal tip depth reproduces the design resistance
R = V_ph² PF² / P_ph − R_net. Current splits between tip and side paths; tip-path power is divided between
coke bed and slag (slag-path fraction 0.55, CALIBRATION-REQUIRED); side-path power heats the lower burden.

**Electrode regulator** (models the existing PLC loop; the twin never commands it): tip velocity
v = clip(K (R − R_sp)/R_sp, ±1)·v_max, current limiting raises the electrode when |I| > I_max, depth limits
h_min…h_max. Paste consumption (kg/MWh) burns the tip off; holder travel compensates; a slip event is logged
each time the burnt length reaches the slip length. Transformer taps are discrete with a minimum dwell
lock-out. Design scaling (rules of thumb, CALIBRATION-REQUIRED): P ≈ 0.8·MVA; V_LL ≈ 68·P^(1/3);
I_max from current density 6.5 A/cm²; PCD ≈ 2.6 D_e.

---

## 7. Energy balance and heat transfer

Per sub-step for each zone z:

    ΔH_z = P_elec,z Δt + Σ (enthalpy carried in) − Σ (enthalpy carried out) − Σ UA (T_z − T_z′) Δt − q_wall,z Δt

* transfers carry h_i(T_source) plus the latent share of the source fusion group;
* gas leaving a burden zone exits at T_out = T_zone + (1 − ε)(T_in − T_zone) (counter-current
  effectiveness ε ≈ 0.82–0.88 × permeability factor);
* inter-zone conductances UA: CB–SL 350 W/m²K × 2A_cb, SL–MB 250 W/m²K × A_hearth, CB–B2 120 W/m²K × A_cb,
  B2–B1 15 W/m²K, B1→freeboard 25 W/m²K (CALIBRATION-REQUIRED);
* the displayed form for each step:
  ΔU/Δt = electrical heating to process + inlet enthalpy − outlet enthalpy − heat losses,
  where reaction heat is contained in the formation-enthalpy terms (shown separately only as a diagnostic).

**Freeboard and off-gas**: furnace gas + dust + infiltrated air (λ Nm³ air per Nm³ gas; closed 0.02,
semi-closed 1.5, open 25) burn CO/H2/CH4 up to the available O2 × combustion efficiency. The freeboard
temperature solves H_mix(T) + G_roof (T − T_roof,face) = H_available (implicit roof coupling), so the
roof heat flow is exactly the heat the wall receives.

### 7.1 Refractory heat transfer (`domain/refractory.py`)

Each wall (hearth, metal band, slag band, lower/upper burden bands, roof) is a stack of layers
(e.g. slag band: magnesia 0.46 m | high-alumina 0.23 m | insulating 0.115 m | steel 0.032 m) discretised by
finite volumes (cylindrical shells or slabs). Conductivity k(T) = k0 + k1 (T − 300). Backward-Euler:

    C_j (T_j^{n+1} − T_j^n)/Δt = G_{j−1,j}(T_{j−1} − T_j) + G_{j,j+1}(T_{j+1} − T_j)
    cylindrical G = 2π k H / ln(r_o / r_i) (series half-cells), slab G = k A / Δx

Outer boundary: q = h_conv(T_s − T_amb) + εσ(T_s⁴ − T_amb⁴) + h_cool(T_s − T_water) with natural convection
h = 1.31 ΔT^{1/3} (McAdams turbulent vertical plate), linearised radiation h_rad = εσ(T_s² + T_a²)(T_s + T_a),
water-spray or water-panel h where configured, forced air under the hearth. Picard iterations update k(T) and
h_rad. The discrete wall balance q_in Δt = ΔE_stored + q_out Δt closes exactly (unit test).

**Freeze lining** (slag band): quasi-steady thickness δ_ss = k_f (T_liq − T_hotface) / (h_s (T_slag − T_liq)),
relaxed with τ = 6 h; δ adds an inner thermal resistance. Mass exchange with the slag inventory is neglected
(< 0.5 % of inventory; stated limitation).

---

## 8. Burden movement, feeding, tapping and lab

* Charge is fed to B1 by **level control** (DCS feeder loop modelled) at the recipe ratios; descent
  B1→B2→CB is consumption-driven (rates slow when the next zone is over its reference inventory), so
  production follows power, as in a real SAF.
* Coke accumulates in the coke bed and is consumed by reduction and carburisation; the **carbon balance
  sets coke-bed fill, which sets furnace resistance and electrode position** — the central SAF control loop.
* Tapping is a discrete event every `tap_interval_h`: φ_m of metal and φ_s of slag above the heels are
  removed with their full enthalpy; a lab assay (with noise) is reported `lab_delay_min` later.
* Dust: carry-over from B1 ∝ base fraction × (fines/8) × (u/u_ref)²; composition = B1 solids. Total Cr in dust
  is reported; Cr(VI) is not inferred.

## 9. Numerical integration and safeguards (`twin/simulator.py`)

Fixed master step Δt = 60 s (configurable) with 4 sub-steps for chemistry/heat exchange; implicit wall
conduction at the master step. After every step the twin rejects non-finite states, negative inventories
(< −1 µmol) and temperatures outside 250–3200 K; on rejection it restores the previous state, marks
`status = frozen_last_good`, logs the reason and returns the last good prediction (tested). Time-step
convergence: halving the master step changes slag temperature by < 15 K and metal Cr by < 0.5 wt% (test).

## 10. Operating states

cold/warm start-up, normal smelting, ore/recipe transition, power curtailment, transformer/electrode
abnormality, burden instability, tapping, recovery, maintenance, emergency fallback. Each state lists which
models are valid and whether MPC recommendations may be issued (`twin/events.py: LIFECYCLE_STATES`).

## 11. Independent closure audit (`domain/phase_inventory.py`)

    mass:   residual_e = Σ in_e − Σ out_e − (inventory_e(t) − inventory_e(t0))   for every element
    energy: residual  = E_in − E_out − ΔH_zones − ΔE_walls

E_in = feed enthalpy + electrode carbon + furnace electrical energy + infiltrated air; E_out = off-gas and dust
(sensible + chemical), tapped metal and slag, shell/roof/hearth losses. Network losses lie outside the furnace
boundary. Typical residuals: < 1e−9 % of input for every element; < 1e−9 MW for energy. No post-hoc
normalisation is ever applied. The engineering energy distribution (reaction enthalpy, product heat, losses,
storage) is an exact identity of the same ledger.

## 12. Steady-state balance and charge optimiser (`domain/static_balance.py`)

Element bookkeeping per tonne of ore blend with surrogate distributions:

    η_Cr = 0.905 − 0.004 (fines − 7) + 0.0008 (T_slag − 1905) − 0.6 max(0, 1 − c_ratio)
    Si [wt%] = 3.7 exp(0.012 (T_slag − 1905)) c_ratio⁴          (c_ratio clipped 0.6–1.15)
    η_Fe = 0.965, η_Mn = 0.75, η_P = 0.85, S to metal 0.30        (all CALIBRATION-REQUIRED)

Carbon demand = O removed by C (per oxide path) + C dissolved − C already in metallic feed; carbon supply =
fixed C + electrode carbon. Cr2O3 above solubility is treated as undissolved spinel (excluded from the
liquid for liquidus/viscosity). Energy: E_furnace = H_out(metal, slag, gas, dust) − H_in(feed) + losses;
SEC = E_furnace / η_el / t_metal. The optimiser minimises cost (materials + electricity + paste + CO2 price)
with SLSQP over ore fractions, reductants and fluxes subject to Cr min, Si max, C max, P max, carbon-ratio band,
slag superheat ≥ min, B1 band and viscosity max; it reports margins, active constraints, finite-difference
shadow prices and, if infeasible, the minimum-violation recipe with the violated constraints.

## 13. State estimation (`twin/state_estimator.py`)

EKF on a 9-state observer model (metal and slag inventory, metal Cr/C/Si, metal T, slag Cr2O3-eq,
productivity t/MWh, coke-bed fill). Measurements: electrode resistance and tip depth (→ fill), tapped metal
and slag masses (weigh-bridge / pot count), tap temperature, delayed lab assays. Joseph-form covariance update;
NIS checked against χ² 99.9 %. **Delayed assays** are fused at their sample time by rewinding to the stored
prior and re-propagating stored inputs and measurements (no future leakage: only assays whose reported time ≤
now are used). Fallback: complementary exponential observer when the EKF loses positive definiteness or NIS
is inconsistent 10 times in a row.

## 14. Hybrid machine learning (`ml/`)

Targets are residuals actual − physics for SEC, Cr recovery and slag Cr2O3 per tap. Features: interval-mean
signals, recipe, fines, moisture, feed temperature, coke-bed fill, electrode depth, burden ΔP, CO, tap
interval, lag-1 lab values, and the physics predictions. Models: ridge (baseline) and gradient-boosted trees.
Chronological 60/20/20 split, hyper-parameter selection on validation, split-conformal 90 % intervals,
permutation importance, Mahalanobis OOD detector (threshold 1.25 × 99th percentile), PSI drift. Corrections
are clipped (SEC ±800 kWh/t, Cr recovery ±15 pts, slag Cr2O3 ±6 wt%) and the final value is clipped to
physical bounds. New models register as **shadow**; activation is an audited engineer action; no automatic
retraining of online models.

## 15. Supervisory MPC (`control/`)

Linearised DMC-type MPC identified online from the nonlinear twin:

    y(k) = y_free(k) + Σ_i S_i(k) Δu_i           (move blocking M = 1, horizon 6 h, sample 0.5 h)
    J = Σ_k Σ_cv w ((y − y_ref)/s)² + w_move Σ (Δu/s_u)² + w_SEC SEC/1000 + w_Cr (100 − CrRec)/10 + w_CO2 CO2 + S·slack

MVs: transformer tap (discrete, dwell lock-out, enumerated −1/0/+1), electrode resistance setpoint, coke and
quartzite per t ore. Protected/procedural functions (slipping, breakers, gas safety, cooling, refractory
protection) are never MVs. Hard MV bounds and move limits are optimisation bounds; CV soft limits are
slack-penalised; hard CV limits (current limit, superheat ≥ 0) are checked on a **nonlinear verification
run** of the chosen moves. SLSQP with warm/multi-start and a wall-clock budget; deterministic fallback "hold
current setpoints" if unsolved, timed out, or blocked by the safety gate. Hybrid MPC adds in-distribution ML
offsets to SEC and Cr recovery.

## 16. Economics and emissions (`domain/economics.py`)

Cost per t saleable FeCr = Σ material cost + electricity + electrode paste + CO2 price + other. Scope 1 = all
process carbon leaving as CO/CO2 (CO assumed ultimately oxidised) minus carbon retained in metal and coke bed;
Scope 2 = MWh × configurable grid factor; optional Scope 3 from configured cradle-to-gate factors. Avoided
emissions from furnace-gas use are a separate memo item, never netted against Scope 1.

## References (model basis; verify applicability before site use)

1. Visser, H. et al. An overview of currently applied ferrochrome production processes and their waste management practices. *Minerals* 13 (2023) 809. https://www.mdpi.com/2075-163X/13/6/809
2. Energy consumption and greenhouse gas emissions of high-carbon ferrochrome production. *JOM* (2023). https://link.springer.com/article/10.1007/s11837-023-05707-8
3. Ferrochrome production in a submerged electric arc furnace: fundamental analysis based upon the FactSage-Macro program approach. *steel research int.* (2024). https://onlinelibrary.wiley.com/doi/10.1002/srin.202300840
4. Metamodeling of the electrical conditions in submerged arc furnaces. *Metall. Mater. Trans. B* 52 (2021). https://link.springer.com/article/10.1007/s11663-021-02089-7
5. Simulation of ferrochrome production in submerged arc furnaces for optimisation of specific energy consumption. NML repository. https://eprints.nmlindia.org/3684/
6. Turkdogan, E.T. *Physical Chemistry of High Temperature Technology.* Academic Press, 1980.
7. Kubaschewski, O., Alcock, C.B., Spencer, P.J. *Materials Thermochemistry*, 6th ed. Pergamon, 1993.
8. Chase, M.W. *NIST-JANAF Thermochemical Tables*, 4th ed. J. Phys. Chem. Ref. Data Monograph 9, 1998.
9. Urbain, G., Cambier, F., Deletter, M., Anseau, M.R. Viscosity of silicate melts. *Trans. J. Br. Ceram. Soc.* 80 (1981) 139–141.
10. Mills, K.C. (ed.) *Slag Atlas*, 2nd ed. Verlag Stahleisen, 1995.
11. McAdams, W.H. *Heat Transmission*, 3rd ed. McGraw-Hill, 1954 (natural-convection correlation).
12. ISA/IEC 62443 series. https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards
