# FeCr-TWIN — Test report

*Extractmet Private Limited · FeCr-TWIN v1.0 · executed 2026-10-29 in the build environment
(Linux x86-64, 2 vCPU, Python 3.13.18, Node 22.23.4, Chromium 1243 headless with SwiftShader WebGL).*

Results are reported exactly as executed. Anything not run is marked **NOT EXECUTED**.

## 1. Summary

| Suite | Command | Result |
|---|---|---|
| Backend — unit, integration, contract, security, performance | `python -m pytest tests --cov=app -q` | **101 passed**, 0 failed, 213.9 s; line coverage **82 %** |
| Balance gate | `python scripts/check_balances.py` | **passed** — 31 reactions balanced; worst element residual 4.57 × 10⁻¹³ % of input; energy residual −1.7 × 10⁻¹⁴ MW |
| OpenAPI export | `python scripts/export_openapi.py` | **passed** — 58 paths written to `docs/API_CONTRACT.yaml` |
| Frontend — unit + contract | `npm test` (`node --test`) | **4 passed**, 0 failed |
| Frontend — production build | `npm run build` (Vite 7) | **passed** — JS bundles 135 kB app, 208 kB Chart.js, 480 kB Three.js (gzip 46/71/121 kB) |
| Browser end-to-end | `FECR_WEB=http://localhost:4173 npx playwright test` against a live backend on :8000 | **1 passed** (34.5 s): sign-in, two-click start, all 21 screens render, 3-D canvas + cutaway, zero console errors |
| PostgreSQL path (psycopg3 + Alembic on Postgres) | — | **NOT EXECUTED** (no Postgres server in the build environment; SQLite path tested). Verify on first Render deploy (`/ready` reports `database: true`). |
| Render deployment | — | **NOT EXECUTED** from the build environment (no outbound access to Render); configuration files `render.yaml` provided for both repos. |
| Dependency vulnerability audit (`pip-audit`, `npm audit`) | configured in CI | **NOT EXECUTED** here (no access to advisory databases). Runs in GitHub Actions. |
| Load test (concurrent users) | — | **NOT EXECUTED**. Single-instance design; see ASSUMPTIONS_AND_LIMITATIONS §5. |

## 2. Backend tests by area (60 test functions, 101 cases after parametrisation)

| Area | File | What is asserted |
|---|---|---|
| Stoichiometry & units | `unit/test_stoichiometry_units.py` | every reaction atom-balanced; molar masses; oxide↔element conversion; chromite mineralogy preserves elements; wet/dry basis; assay validation rejects fractions/unknown species; gas Nm³ basis |
| Thermo, slag, electrical, walls | `unit/test_thermo_slag_electrical.py` | enthalpy→temperature inversion round-trip; B1 ≠ B2 basicity; slag correlation extrapolation flagged; viscosity falls with T and rises with solids; balanced three-phase power; unbalance detection; wall energy closure and losses |
| Static balance & optimiser | `unit/test_static_balance.py` | nominal ranges; carbon-demand accounting; more carbon ⇒ more Si; optimiser feasible within constraints; infeasibility reported |
| Dynamic closure | `unit/test_dynamics_closure.py` | mass & energy closure every step; non-negative finite inventories; recovery and SEC denominators; seed reproducibility; time-step convergence; invalid step rejected and state frozen; tap-hole lockout; species-mask consistency |
| Estimator, gateway, ML | `unit/test_estimator_ml.py` | ≥ 50 tags; EKF tracks hidden truth with delayed assays and no future leakage; missing/stuck tags flagged; chronological split without look-ahead; hybrid beats physics on synthetic mismatch; OOD and PSI |
| MPC & safety gate | `unit/test_mpc_gate.py` | bounds and slew limits; tap lockout respected; timeout/infeasible ⇒ hold setpoints; stale telemetry blocks; unsupported operating state blocks; no command interface on the MPC class |
| Scenarios | `integration/test_scenarios.py` | scenario runs close mass and energy; G curtailment lowers production; H raises unbalance alarm; I raises Δp; C raises dust; E quartz lowers liquidus |
| API contract | `contract/test_api.py` | required routes; auth + RBAC; bad token rejected; session payload fields; assay endpoint; upload limits; WebSocket stream and reconnect |
| Security | `security/test_security.py` | no plant-write routes in OpenAPI; `/capabilities` advisory-only; security headers; no embedded secrets; blocked recommendation cannot be accepted (422); model activation needs engineer role |
| Performance | `performance/test_perf.py` | 24 h simulation under 30 s; static optimiser under 1 s |

## 3. Measured performance (build environment, 2 vCPU)

| Item | Value |
|---|---|
| Twin step (dt 60 s) | 52 steps/s ≈ 3 100 × real time |
| Scenario run, 24 h plant time | 24–29 s incl. spin-up and output |
| Static balance | 3.1 ms per evaluation |
| MPC recommendation (fast config, comparison runs) | 2.0–2.1 s mean |
| MPC recommendation (full 6 h horizon, GUI) | 5–15 s observed |
| 16-day synthetic campaign generation (180 taps) | 115 s |

## 4. Coverage notes

Total 82 % line coverage. Lowest-covered modules: `twin/uncertainty.py` (13 %, Monte Carlo and sensitivity are
exercised through the API manually, not in the automated suite), `services/scenario_service.py` (20 %),
`workers/*` (background job runner), `schemas/thermo.py`. These are candidates for the next test increment.

## 5. Issues found and fixed during this test cycle

| Issue | Fix |
|---|---|
| Nominal recipe accumulated carbon after adding CH4 cracking and steam-carbon reactions (fill 1.56, Si 4.5 % after 84 h) | coke 172 → 163 kg/t ore; 84 h re-check stable |
| Browser e2e against `vite preview` (port 4173) failed CORS with default origins | default `ALLOWED_ORIGINS` now includes 4173 for local preview; production still requires explicit origins |
| MPC bound error when the current MV lay outside its range | bounds clipped to include the current value; active coke grade selected automatically |
| WebSocket 404 under uvicorn | `websockets` pinned in requirements |
| Scenario H produced no unbalance (regulator compensated) | stuck-electrode disturbance added |

## 6. Reproduce

```bash
cd ferrochrome-saf-backend && pip install -r requirements-dev.txt
python -m pytest tests --cov=app -q && python scripts/check_balances.py
uvicorn app.main:app --port 8000 &
cd ../ferrochrome-saf-frontend && npm ci && npm test && npm run build && npx vite preview --port 4173 &
npx playwright install chromium && FECR_WEB=http://localhost:4173 npx playwright test
```
