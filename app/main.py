"""FeCr-TWIN backend entrypoint (FastAPI). Advisory-only digital twin of an HC FeCr SAF."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.v1 import health
from .api.v1.router import api
from .core.config import settings
from .core.errors import DomainError, domain_handler
from .core.logging import log
from .db.repositories import init_db
from .services import model_service
from .services.simulation_service import base_sim, service


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings.demo_password)
    threading.Thread(target=_warm, daemon=True).start()
    task = asyncio.create_task(service.ticker())
    log.info("FeCr-TWIN started (advisory_only=%s, env=%s)", settings.advisory_only, settings.app_env)
    yield
    task.cancel()


def _warm():
    try:
        model_service.bootstrap()
        base_sim(45.0, "closed", "A")
    except Exception as exc:  # noqa: BLE001
        log.error("warm-up failed: %s", exc)


app = FastAPI(title="FeCr-TWIN API - Ferrochrome SAF Digital Twin (Extractmet Private Limited)", version=settings.version,
              description="SYNTHETIC / NOT PLANT-VALIDATED. Advisory-only digital twin; no plant write path exists.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])
app.add_exception_handler(DomainError, domain_handler)

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    ip = request.client.host if request.client else "?"
    now = time.time()
    dq = _hits[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= settings.rate_limit_per_min:
        return JSONResponse({"error": {"code": "RATE_LIMITED", "message": "too many requests"}}, status_code=429)
    dq.append(now)
    cl = request.headers.get("content-length")
    if cl and int(cl) > settings.max_upload_mb * 1e6 + 1e5:
        return JSONResponse({"error": {"code": "PAYLOAD_TOO_LARGE", "message": "request body too large"}}, status_code=413)
    t0 = time.time()
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Strict-Transport-Security"] = "max-age=31536000"
    resp.headers["Cache-Control"] = "no-store"
    log.info("request", extra={"request_id": rid, "path": request.url.path, "status": resp.status_code,
                               "duration_ms": round(1000 * (time.time() - t0), 1)})
    return resp


app.include_router(health.router, include_in_schema=False)  # root-level probes (documented under /api/v1)
app.include_router(api)


@app.get("/", include_in_schema=False)
def root():
    return {"service": settings.app_name, "docs": "/docs", "api": "/api/v1", "label": "SYNTHETIC / NOT PLANT-VALIDATED"}
