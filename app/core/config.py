"""Runtime configuration from environment variables (12-factor). No secrets in code."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field


def _b(v: str | None, d: bool) -> bool:
    return d if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_name: str = "FeCr-TWIN SAF Digital Twin (Extractmet)"
    version: str = "1.0.0"
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./fecr_twin_dev.db"))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", ""))
    jwt_ttl_min: int = field(default_factory=lambda: int(os.getenv("JWT_TTL_MIN", "480")))
    allowed_origins: list = field(default_factory=lambda: [o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173").split(",") if o.strip()])
    demo_password: str = field(default_factory=lambda: os.getenv("DEMO_PASSWORD", "extractmet-demo"))
    advisory_only: bool = field(default_factory=lambda: _b(os.getenv("ADVISORY_ONLY"), True))
    closed_loop_enabled: bool = False  # hard-coded: no command path exists in this codebase
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "./model_store"))
    max_sessions: int = field(default_factory=lambda: int(os.getenv("MAX_SESSIONS", "8")))
    max_jobs: int = field(default_factory=lambda: int(os.getenv("MAX_JOBS", "2")))
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "10")))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "600")))
    spinup_h: float = field(default_factory=lambda: float(os.getenv("SPINUP_H", "20")))
    thermo_provider: str = field(default_factory=lambda: os.getenv("THERMO_PROVIDER", "builtin"))

    def __post_init__(self):
        # Render/Heroku style URLs -> SQLAlchemy psycopg3 driver
        for pre in ("postgres://", "postgresql://"):
            if self.database_url.startswith(pre):
                self.database_url = "postgresql+psycopg://" + self.database_url[len(pre):]
        if not self.secret_key:
            if self.app_env == "production":
                raise RuntimeError("SECRET_KEY must be set in production")
            self.secret_key = secrets.token_urlsafe(32)
        if self.app_env == "production" and self.demo_password == "extractmet-demo":
            raise RuntimeError("DEMO_PASSWORD must be changed in production")
        if not self.advisory_only:
            raise RuntimeError("ADVISORY_ONLY=false is not supported: closed-loop control is not implemented in this release")

    @property
    def sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
