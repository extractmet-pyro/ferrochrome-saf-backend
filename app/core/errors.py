"""Error types and uniform error response bodies."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class DomainError(Exception):
    code = "DOMAIN_ERROR"
    status = 422

    def __init__(self, msg: str, details: dict | None = None):
        super().__init__(msg)
        self.details = details or {}


class NotFound(DomainError):
    code, status = "NOT_FOUND", 404


class Conflict(DomainError):
    code, status = "CONFLICT", 409


class QuotaExceeded(DomainError):
    code, status = "QUOTA_EXCEEDED", 429


async def domain_handler(request: Request, exc: DomainError):
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": str(exc), "details": exc.details,
                                                                   "request_id": getattr(request.state, "request_id", None)}})
