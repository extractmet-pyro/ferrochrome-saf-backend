# FeCr-TWIN — Operator and metallurgist manual

*Advisory digital twin of a ferrochrome submerged arc furnace. Extractmet Private Limited.*
*All values in the demonstration are SYNTHETIC / NOT PLANT-VALIDATED.*

## 1. What the desk is for

FeCr-TWIN runs a physics model of the furnace next to the plant (or, in the demonstration, next to a synthetic
plant). It shows what instruments cannot measure — coke-bed size, zone temperatures, undissolved chromite,
metal chemistry before the lab result — and proposes setpoint changes with their predicted effect. It never
changes anything in the plant; people do.

## 2. First start (two clicks)

1. Sign in with your role account and the backend URL (supplied by your administrator).
2. In **Start the synthetic plant**, keep 45 MVA / closed / scenario A and press **Start simulation**.
   The furnace is warm-started from a converged state; the first start of a new size can take up to a minute.
3. Use **15×** or **60×** to watch hours of plant time in minutes; **Pause** freezes the twin.

## 3. Reading the screens

| Tag | Meaning |
|---|---|
| live | value from an instrument tag (synthetic gateway in the demo) |
| model | twin prediction |
| estimated | Kalman-filter estimate (with confidence interval on the State estimator screen) |
| stale | no update for more than 8 s |
| invalid | twin solver rejected a step and is holding the last good prediction |

Alarms always combine a shape and text: triangle = critical, diamond = warning, circle = information.

### Screen guide

* **Plant overview** — key indicators, advisories, next/last tap, equipment and data health, power and
  temperatures, chemistry, inventories, closure residuals.
* **Plant schematic** — whole flowsheet: stockhouse bins (t/h per material), batching, charging chutes, the furnace
  section with zone colours from model temperatures, electrodes with currents, transformer, short network, off-gas
  train (venturi/gas holder for closed, hood/baghouse for open roofs), cooling water, tap-hole, ladle, casting
  beds, slag handling and the laboratory queue.
* **3D furnace** — rotate, zoom, cutaway, layer toggles, camera presets; hover any part for its values. Phase
  levels follow inventories, electrode tips follow the regulator, coke-bed thickness follows the coke-bed fill.
* **Process trajectories** — every model trajectory; CSV and PDF export.
* **Chemistry and thermodynamics** — Ellingham-type ΔG° curves, reaction table (ΔG, ΔH, equilibrium temperature,
  basis), slag basicities B1 and B2 (kept separate), liquidus, superheat, viscosity, metal C saturation.
* **Kinetics and transport** — reaction rates, chromite dissolution (particle size, mass-transfer coefficient,
  driving force), droplet settling, production and dust.
* **Electrical and electrodes** — phasor diagram, per-phase table, resistance vs setpoint, tip depth and holder,
  slips, power split between coke bed, slag path, burden and network.
* **Temperatures and energy** — zone temperatures, energy ledger, Sankey of where the electric energy goes,
  loss mechanisms (convection, radiation, water), wall heat flows, shell temperatures, closure residual.
* **Refractory and shell** — through-wall temperature profiles of each wall, layups, freeze-lining thickness.
* **Gas and emissions** — dry/wet compositions, analyzer vs model, flows, temperatures, dust, chemical energy.
* **Mass balance and products** — elemental closure, every input and output stream with species and final
  weights, zone inventories, tap log with metal and slag compositions, Cr reconciliation, Cr losses by form.
* **Furnace design and capacity** — choose 9–81 MVA, edit dimensions, see the section to scale and start a twin
  of that design; what-if tap, resistance and tap interval on the running twin.
* **Charge-mix planner** — steady-state balance per t ore and per t FeCr, carbon demand by path, energy use,
  optimiser with constraints, shadow prices and infeasibility diagnosis; assay validation.
* **Scenarios A–L**, **Economics and ESG**, **State estimator**, **Machine learning**, **Supervisory MPC**,
  **Historical replay**, **Data, tags and audit**, **Help and glossary**.

## 4. Working with recommendations

1. Open **Supervisory MPC**, tick *Use hybrid ML corrections* if the models are in-distribution, press **Compute
   recommendation**.
2. Read the gate result. *Blocked* means: hold current setpoints; the reasons are listed (data quality, operating
   state, solver, verification).
3. Review suggested moves (each within its limits and move size), predicted trajectories (suggested, no change,
   nonlinear verification) and constraint margins.
4. Accept or reject **with a reason**. Acceptance is recorded; it does not send anything to the plant. Implement an
   accepted change through your normal procedure. In the demo, **Try in simulation** applies it to the twin.

## 5. Audit guide for metallurgists

**Chromium balance.** Mass balance screen → elemental closure row *Cr*: input (charge) = metal + slag + dust +
accumulation, residual in kg/h and % of input. The Cr distribution chart splits slag Cr into undissolved chromite,
metal droplets and dissolved CrOx; reconcile with the weighted-least-squares table (χ² global test).

**Carbon balance.** Charge planner → *Carbon demand by path*: oxygen removed from Cr, Fe, Si, Mn, P oxides, carbon
dissolved in metal, and surplus that accumulates in the coke bed. In the dynamic twin, a persistent surplus shows
as rising coke-bed fill, shallower electrodes and higher Si; a deficit shows the opposite and lower Cr recovery.

**Slag basicity.** Two indices are always shown separately: B1 = (CaO+MgO)/SiO2 and B2 = (CaO+MgO)/(SiO2+Al2O3),
plus MgO/Al2O3. Targets must be set per site practice (acid or basic); do not compare a B1 target with a B2 value.

**Reaction energetics.** Chemistry screen → reaction table gives ΔG and ΔH at 1700 °C and the equilibrium
temperature at unit activities. The Sankey shows how much of the electric energy goes to reduction reactions
(typically ~ 60–65 % in the nominal case), metal and slag heat, gas and losses.

**MPC constraints.** MPC screen → configuration table lists every manipulated variable (bounds, move limit) and
controlled variable (target, soft and hard limits). Margins in the recommendation are computed on the nonlinear
verification run.

## 6. Commissioning checklist (before any plant use)

1. Site assay campaign (≥ 4 weeks) for ores, reductants, fluxes, metal, slag, dust — with reconciliation.
2. Calibration records: thermocouples, flowmeters, weigh feeders, ladle and slag-pot weighing.
3. Electrode instrumentation: currents, voltages to hearth, holder positions, slip logs; electrical model fit.
4. Gas analyzer accuracy, sampling location and measured lag.
5. Slag liquidus and viscosity validation (lab measurement or licensed database) for the site slag window.
6. Cr recovery closure within agreed tolerance over ≥ 30 taps; SEC within agreed tolerance.
7. Kinetic and heat-transfer parameter fit with identifiability report; uncertainty bands agreed.
8. Cybersecurity review (IEC 62443 zones and conduits, read-only gateway, identity provider, backups).
9. Operator training and site acceptance test signed by the plant owner.
