"""Background job manager: bounded thread pool, task IDs, progress, cooperative cancellation.

Long simulations, training and MPC comparisons never run inside request handlers. On Render a
single web instance runs this in-process pool; for scale, run `python -m app.workers.worker`
as a separate Background Worker sharing the database (see DEPLOYMENT_GUIDE.md)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..core.config import settings
from ..core.errors import NotFound, QuotaExceeded


class Cancelled(Exception):
    pass


class JobManager:
    def __init__(self, workers: int):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.limit = workers * 4

    def submit(self, kind: str, fn, owner: str, **kw) -> str:
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
            if len(active) >= self.limit:
                raise QuotaExceeded("job queue full - retry later")
            jid = f"job-{uuid.uuid4().hex[:10]}"
            job = {"id": jid, "kind": kind, "owner": owner, "status": "queued", "progress": 0.0, "result": None, "error": None,
                   "created": time.time(), "cancel": False}
            self.jobs[jid] = job

        def progress(f):
            if job["cancel"]:
                raise Cancelled()
            job["progress"] = float(min(max(f, 0), 1))

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn(progress=progress, **kw)
                job["status"] = "done"
                job["progress"] = 1.0
            except Cancelled:
                job["status"] = "cancelled"
            except Exception as exc:  # noqa: BLE001
                job["status"] = "failed"
                job["error"] = f"{exc}"
                job["trace"] = traceback.format_exc()[-2000:]
        self.pool.submit(run)
        return jid

    def get(self, jid: str) -> dict:
        if jid not in self.jobs:
            raise NotFound(f"job {jid} not found")
        j = self.jobs[jid]
        return {k: v for k, v in j.items() if k not in ("cancel", "trace")}

    def cancel(self, jid: str):
        if jid not in self.jobs:
            raise NotFound(f"job {jid} not found")
        self.jobs[jid]["cancel"] = True
        return self.get(jid)


jobs = JobManager(settings.max_jobs)
