# FeCr-TWIN — Supervisory MPC safety case (advisory release)

## 1. Claim

In this release the supervisory MPC **cannot change the state of any plant equipment**. It produces
recommendations for human review only. Plant protection remains entirely with the existing, independent
PLC/DCS loops, safety-instrumented systems, hard interlocks and operating procedures.

## 2. Arguments and evidence

| # | Argument | Evidence |
|---|---|---|
| A1 | No write path exists in the code | No endpoint, adapter or client function writes to PLC/DCS/OPC UA. `OPCUAReadOnlyAdapter.write()` raises `WriteForbidden`. Test `tests/security/test_security.py::test_no_plant_write_routes` scans the OpenAPI paths; `tests/unit/test_mpc_gate.py::test_no_command_interface` checks the MPC class. |
| A2 | Advisory-only cannot be switched off by configuration | `Settings.__post_init__` refuses to start when `ADVISORY_ONLY=false`; `closed_loop_enabled` is hard-coded `False`; `/capabilities` reports `plc_write: false`. |
| A3 | 3-D/GUI interactions cannot issue commands | GUI actions call `/simulation/{id}/action` which changes the **simulated** furnace only and is audited; the 3-D view has no action handlers. |
| A4 | Recommendations are gated | `control/safety_gate.py`: operating-state permission, required-tag data quality (stale/missing/suspect analyzer), solver status and time budget, MV bounds and move limits, estimator mode, nonlinear verification against hard CV limits. Blocked recommendations cannot be accepted (API returns 422; tested). |
| A5 | Deterministic fallback | Any optimiser failure, timeout, infeasibility or gate block ⇒ "hold current operator-approved setpoints". Tested: `test_timeout_and_infeasible_give_safe_fallback`. |
| A6 | Human authority and traceability | Accept/reject requires an operator role and a written reason; each recommendation carries provenance (snapshot time, config revision, model versions, solver status, margins) and an HMAC audit signature; decisions are written to the audit table. |
| A7 | Protected functions are never MVs | Electrode slipping/baking, breakers, gas safety and ventilation, cooling water and refractory protection are excluded (`control/constraints.py`). |
| A8 | Model limitations are visible | Synthetic label on every screen and payload; confidence and model mismatch reported per recommendation; OOD detection disables hybrid corrections. |

## 3. Hazards considered (examples)

| Hazard | Twin role | Independent protection that must remain |
|---|---|---|
| Electrode over-current / breakage | advisory current-margin display, hard-limit check in verification | electrode regulator current limit, breaker protection |
| Furnace eruption / blow (permeability loss) | permeability index advisory, burden-instability state blocks MPC | operating procedures, pressure relief, gas system interlocks |
| CO exposure | CO advisory only | fixed CO detectors, ventilation interlocks, permits |
| Water leak into furnace | not modelled | cooling-water leak detection and trips |
| Tap-hole break-out / hot-spot | shell temperature advisory | shell thermocouple alarms, cooling, procedures |
| Low slag superheat (freeze, tapping failure) | superheat constraint & alarm | tapping procedures |

## 4. Conditions for any future closed-loop commissioning (not implemented)

1. Site-specific HAZOP and LOPA; SIL determination for any function the MPC would influence.
2. Model validation against ≥ 3 months of plant data with documented error bounds per CV.
3. Factory and site acceptance tests (FAT/SAT) of the controller, the write gateway and fallbacks.
4. Independent SIS and interlocks untouched; MPC writes only to bounded setpoints of existing loops through a
   rate-limited, audited on-prem gateway with watchdog and bumpless fallback to operator setpoints.
5. Management of change, operator training, written authorisation by the plant owner.
6. The cloud (Render) deployment must never be in the control path.
