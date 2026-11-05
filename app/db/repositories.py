from __future__ import annotations

from sqlalchemy import select

from ..core.security import hash_password
from . import models as M
from .base import Base
from .session import SessionLocal, engine


def init_db(demo_password: str):
    """Dev convenience: create tables if missing (production runs `alembic upgrade head`)."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for u, r in (("viewer", "viewer"), ("operator", "operator"), ("metallurgist", "metallurgist"),
                     ("engineer", "engineer"), ("admin", "administrator")):
            if not db.scalar(select(M.User).where(M.User.username == u)):
                db.add(M.User(username=u, password_hash=hash_password(demo_password), role=r))
        db.commit()


def audit(db, user: dict, action: str, target: str = "", detail: dict | None = None, request_id: str = ""):
    db.add(M.AuditRecord(user=user.get("sub", "?"), role=user.get("role", "?"), action=action, target=target,
                         detail=detail or {}, request_id=request_id))
    db.commit()
