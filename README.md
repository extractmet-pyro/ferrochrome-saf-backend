# FeCr-TWIN backend — Ferrochrome SAF digital twin (Extractmet Private Limited)

FastAPI service for an **industry-deployable, advisory-only** digital twin, process simulator and supervisory
process-control advisor for high-carbon ferrochrome production in a submerged arc furnace (SAF).

> **SYNTHETIC / NOT PLANT-VALIDATED.** The demonstration plant ("Industry-X") is simulated. No code path writes to
> plant controllers. See `docs/ASSUMPTIONS_AND_LIMITATIONS.md` and `docs/MPC_SAFETY_CASE.md`.

## What is inside

| Area | Module |
|---|---|
| Species, element matrix, mineralogy, assay validation | `app/domain/species.py`, `material_streams.py` |
| Thermochemistry, enthalpy-based temperatures | `app/domain/thermodynamics.py` |
| 31 balanced reactions, ΔG° data, rate laws, dissolution, transport | `app/domain/reactions.py`, `kinetics.py` |
| Slag (B1/B2, Urbain viscosity, liquidus) and metal (C saturation) | `app/domain/slag.py`, `metal.py` |
| Three-phase phasor circuit, electrode regulator, slipping | `app/domain/electrical.py` |
| Multi-layer refractory FV conduction + convection/radiation/water + freeze lining | `app/domain/refractory.py` |
| Off-gas, freeboard combustion, dust, analyzer | `app/domain/offgas.py` |
| Dynamic 5-zone twin with tapping, lab delay, events | `app/twin/simulator.py`, `events.py` |
| Independent mass/energy ledgers | `app/domain/phase_inventory.py` |
| Static balance + constrained charge optimiser | `app/domain/static_balance.py` |
| EKF with delayed-assay re-filtering, reconciliation, uncertainty | `app/twin/state_estimator.py`, `reconciler.py`, `uncertainty.py` |
| Hybrid ML (ridge / GBR residuals, OOD, drift, registry) | `app/ml/` |
| Supervisory MPC + safety gate + baseline comparison | `app/control/` |
| Economics & ESG | `app/domain/economics.py` |
| API, auth (JWT/RBAC), DB (SQLAlchemy/Alembic), WebSocket | `app/api/v1`, `app/core`, `app/db` |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload --port 8000                 # API docs at http://localhost:8000/docs
python -m pytest -q tests                                 # 99 tests
python scripts/check_balances.py                          # reaction + closure gate
python scripts/run_demo.py 6 A,G,H                        # headless scenarios
```

Seeded role accounts: `viewer`, `operator`, `metallurgist`, `engineer`, `admin` with password `DEMO_PASSWORD`.

## Deploy

Render web service + Postgres via `render.yaml` (Blueprint). Full procedure: `docs/DEPLOYMENT_GUIDE.md`.
The operator desk is the separate repository `ferrochrome-saf-frontend`.

## Documentation

`docs/MODEL_THEORY_AND_EQUATIONS.md` · `ASSUMPTIONS_AND_LIMITATIONS.md` · `VALIDATION_REPORT.md` ·
`DATA_DICTIONARY.csv` · `API_CONTRACT.yaml` · `OPERATOR_MANUAL.md` · `MPC_SAFETY_CASE.md` · `TEST_REPORT.md` · `PRODUCT_BRIEF.md` ·
`DEPLOYMENT_GUIDE.md`

© Extractmet Private Limited — proprietary licence (see LICENSE).
