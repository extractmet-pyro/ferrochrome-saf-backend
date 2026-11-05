# FeCr-TWIN

## Digital twin, process simulator and supervisory advisor for ferrochrome submerged arc furnaces

*Extractmet Private Limited · IIT Madras Research Park, Chennai · Product brief v1.0, October 2026*

---

### The problem a ferrochrome smelter lives with

A submerged arc furnace hides almost everything that decides its cost. Operators see power, electrode
positions, off-gas analysis and — hours late — the lab result for the last tap. They do not see the size of
the coke bed, how much chromite is still undissolved in the slag, how hot the reaction zone is, or which way
metal silicon is heading before the next tap. Decisions on coke rate, quartzite, resistance setpoint and
transformer tap are made on experience and lagging assays.

The consequences are familiar to every plant manager:

* **Electricity** is the largest single cost; a few percent of specific energy is a large number every year.
* **Chromium lost to slag** (undissolved chromite, entrained metal, dissolved CrOx) is ore that was paid for,
  transported and smelted, and then dumped.
* **Carbon imbalance** shows up slowly: rising electrodes, high silicon, a growing coke bed — or the reverse —
  and is corrected late.
* **Raw-material variability** (lower-grade ore, fines, wet charge, high-ash reductant) changes the furnace for
  days before its effect is fully visible.

### What FeCr-TWIN is

FeCr-TWIN runs a physics model of the furnace alongside the plant, in real time, and turns it into an operator
desk.

| Capability | What the plant gets |
|---|---|
| **Physics twin** | 5-zone dynamic model (upper and lower burden, coke bed, slag, metal) with gas, freeboard and refractory walls; 38 species, 31 reactions with thermodynamics and kinetics; chromite dissolution, carburisation, droplet settling; three-phase electrical circuit with regulator and slips; conduction–convection–radiation heat losses through the actual lining |
| **Exact balances** | elemental mass balance (Cr, Fe, Si, C, O, Ca, Mg, Al, Mn, P, S, H, N) and energy balance closed at every time step to machine precision; every input and output stream with composition and weight |
| **State estimator** | Kalman filter fusing live tags with delayed lab assays: coke-bed fill, metal and slag inventory, metal chemistry and temperature — with confidence intervals |
| **Hybrid ML** | residual models that learn what the physics misses, trained chronologically, with prediction intervals, drift and out-of-distribution detection, shadow → active governance |
| **Supervisory MPC** | recommended moves for coke rate, quartzite, resistance setpoint and tap, verified on the nonlinear twin and passed through a safety gate; advisory only |
| **Planning tools** | charge-mix optimiser with shadow prices, furnace design and capacity (9–81 MVA), refractory layup, what-if scenarios, economics and CO₂ |
| **Operator desk** | 21 screens: plant schematic, live 3-D furnace cutaway, trajectories, chemistry, kinetics, electrical, energy Sankey, refractory, gas, mass balance, scenarios, estimator, ML, MPC, replay, audit |

### Safe by design

* **Advisory only.** No code path writes to a PLC, DCS or OPC UA server. The read-only gateway raises an error on
  any write attempt; the service refuses to start if advisory mode is switched off.
* **Gated recommendations.** Operating state, data quality, solver status, bounds and move limits, and a
  nonlinear verification run must all pass; otherwise the advice is "hold current setpoints".
* **Human authority.** Accept or reject requires a role and a written reason; every recommendation is signed and
  audited.
* **Honest labelling.** Model, estimated and measured values are tagged on screen; the demonstration is labelled
  *Synthetic / not plant-validated* everywhere.

### Where it runs

Two GitHub repositories — a FastAPI backend (Python) and a static web frontend (HTML/CSS/JavaScript with
Three.js) — deploy to Render in under an hour for evaluation, or to the customer's own servers or private cloud
for production. Plant data enter through an on-premises read-only OPC UA gateway in the DMZ that pushes outbound
only. PostgreSQL stores scenarios, recommendations, audit and lab data.

### What the demonstration shows today (synthetic 45 MVA furnace)

| Indicator | Value on the synthetic plant |
|---|---|
| Specific energy | ~3 300 kWh/t HC FeCr |
| Cr recovery | ~90.7 % |
| Metal Cr / C / Si | 64.0 / 6.97 / 3.5 wt % |
| Slag Cr2O3-equivalent | ~9 wt % |
| Mass and energy closure | residuals below 10⁻¹² of input |
| Scenarios | 12 (ore grade, moisture/fines, reductant ash, quartzite, pre-reduction, curtailment, phase unbalance, permeability, delayed tap, sensor faults, out-of-distribution) |
| Tests | 101 automated backend tests, browser end-to-end test of all 21 screens |

These are **synthetic** results. They prove the software and the physics work together; they are not a claim
about any plant.

### Illustrative value — to be replaced by the plant's own numbers

The value case is built with the customer from their own data. As an illustration only, for a 45 MVA furnace
producing about 88 000 t/y of HC FeCr at about 3.3 MWh/t:

| Lever | Illustrative arithmetic | Annual value |
|---|---|---|
| 1 % lower specific energy | 88 000 t × 3.3 MWh/t × 1 % × [tariff ₹/MWh] | at ₹6 500/MWh ≈ ₹1.9 crore |
| 1 pt higher Cr recovery | ≈ 1.1 % more metal from the same ore ≈ 970 t × [HC FeCr margin ₹/t] | plant-specific |
| Fewer off-spec taps (Si, C) | taps avoided × downgrade penalty | plant-specific |
| Planned curtailment with less Cr loss | scenario G shows Cr recovery falls ~5 pts during an unmanaged power cut | plant-specific |

The synthetic MPC study (scenario D) showed a small gain (SEC −0.3 %) with a small Cr-recovery cost; real
benefit must be measured in a shadow-mode pilot before any figure is promised.

### How an engagement runs

| Phase | Duration | Deliverables |
|---|---|---|
| 1. Discovery and data audit | 3–4 weeks | tag list, historian access, assay and tap records, furnace drawings, gap report |
| 2. Configuration and calibration | 6–8 weeks | furnace geometry and lining, material library, electrical fit, kinetic and heat-transfer calibration, validation report against ≥ 30 taps |
| 3. Shadow-mode pilot | 3 months | read-only gateway, live desk, logged recommendations compared with operator practice, benefit study |
| 4. Production | ongoing | site licence, model maintenance, retraining, support, operator training |

### Commercial options

| Option | Scope | Price |
|---|---|---|
| Evaluation | hosted demo with the synthetic plant, workshop with the metallurgy team | [to be quoted] |
| Pilot | phases 1–3 for one furnace, fixed fee | [to be quoted] |
| Site licence | per furnace per year, support and updates included | [to be quoted] |
| Enterprise | multi-furnace, on-premises, source escrow, custom models | [to be quoted] |

### Why Extractmet

Extractmet is a deep-tech pyrometallurgy company incubated at IIT Madras Research Park, founded by the head of
the Centre for Pyrometallurgy at IIT Madras. The team combines first-principles process modelling
(thermodynamics, kinetics, heat and mass transfer) with data-driven modelling and industrial software.
FeCr-TWIN belongs to a family of Extractmet twins for blast furnace, DRI, sinter, steelmaking, ladle, casting
and electric smelting, built on the same balance-first architecture.

### Honest limits

Lumped zones, not CFD; activity models and several thermodynamic data are approximate; DC furnaces and the
low-carbon FeCr route are not covered; kinetic parameters are tuned to published ranges and must be fitted to
plant data; closed-loop control is not offered. The full list is in *Assumptions and limitations*; the
evidence and its limits in the *Validation report*.

**Contact:** Extractmet Private Limited, IIT Madras Research Park, Chennai — [email] · [phone]
