"""Transactional persistence: users, config revisions, scenarios, runs, recommendations,
audit, model versions, lab assays, quality events. High-rate telemetry is kept in the
session ring buffer / exported files (a time-series store is recommended for plants)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32))
    plant: Mapped[str] = mapped_column(String(64), default="industry-x")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class PlantConfigRevision(Base):
    __tablename__ = "plant_config_revisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plant: Mapped[str] = mapped_column(String(64), index=True, default="industry-x")
    revision: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSON)
    author: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Scenario(Base):
    __tablename__ = "scenarios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    plant: Mapped[str] = mapped_column(String(64), default="industry-x")
    definition: Mapped[dict] = mapped_column(JSON)
    result_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    author: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    plant: Mapped[str] = mapped_column(String(64), default="industry-x")
    owner: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(40), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    gate_status: Mapped[str] = mapped_column(String(40))
    decision: Mapped[str] = mapped_column(String(16), default="pending")
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    decided_by: Mapped[str] = mapped_column(String(64), default="")
    signature: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditRecord(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    user: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    request_id: Mapped[str] = mapped_column(String(40), default="")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    metrics: Mapped[dict] = mapped_column(JSON)
    manifest: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class LabAssay(Base):
    __tablename__ = "lab_assays"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plant: Mapped[str] = mapped_column(String(64), default="industry-x")
    sample_type: Mapped[str] = mapped_column(String(16))
    sample_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reported_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    values: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(64), default="upload")


class QualityEvent(Base):
    __tablename__ = "quality_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    session_id: Mapped[str] = mapped_column(String(40), default="")
    tag: Mapped[str] = mapped_column(String(64))
    quality: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text, default="")
    sim_t_h: Mapped[float] = mapped_column(Float, default=0.0)
