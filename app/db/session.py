from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import settings

_kw = {"connect_args": {"check_same_thread": False}} if settings.sqlite else {"pool_pre_ping": True, "pool_size": 5}
engine = create_engine(settings.database_url, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_ok() -> bool:
    try:
        with engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False
