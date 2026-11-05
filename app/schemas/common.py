from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}
    request_id: str | None = None


class JobRef(BaseModel):
    job_id: str = Field(description="background job id; poll GET /api/v1/jobs/{job_id}")
    status: str


class Health(BaseModel):
    status: str
    advisory_only: bool = True


class Ready(BaseModel):
    ready: bool
    database: bool
    models_active: dict
    warm_start_cached: bool


class Version(BaseModel):
    name: str
    version: str
    api: str = "v1"
    synthetic_demo: bool = True
