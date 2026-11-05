"""Structured JSON logging with request IDs."""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "path", "status", "user", "duration_ms"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        return json.dumps(d)


def setup_logging():
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger("fecr")
    root.handlers[:] = [h]
    root.setLevel(logging.INFO)
    return root


log = setup_logging()
