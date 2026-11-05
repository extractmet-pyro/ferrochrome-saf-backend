from __future__ import annotations

import time

from fastapi import Header, Request

_idem: dict[str, tuple[float, object]] = {}


def idempotent(request: Request, key: str | None, fn):
    """Return the cached response for a repeated Idempotency-Key (24 h), else run fn()."""
    if not key:
        return fn()
    user = getattr(request.state, "user", {}).get("sub", "anon")
    k = f"{user}:{request.url.path}:{key}"
    now = time.time()
    for kk in [kk for kk, (t, _) in _idem.items() if now - t > 86400]:
        _idem.pop(kk, None)
    if k in _idem:
        return _idem[k][1]
    r = fn()
    _idem[k] = (now, r)
    return r


def idem_key(idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return idempotency_key


def rid(request: Request) -> str:
    return getattr(request.state, "request_id", "")
