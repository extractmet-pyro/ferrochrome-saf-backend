from fastapi import APIRouter

from ...core.config import settings
from ...db.session import db_ok
from ...schemas.common import Health, Ready, Version
from ...services import simulation_service as ss
from ...services.model_service import TARGETS, registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, operation_id="health")
def health():
    return Health(status="ok", advisory_only=settings.advisory_only)


@router.get("/ready", response_model=Ready, operation_id="ready")
def ready():
    d = db_ok()
    act = {t: (registry.active(t) or [None])[0] for t in TARGETS}
    return Ready(ready=d, database=d, models_active=act, warm_start_cached=len(ss._cache) > 0)


@router.get("/version", response_model=Version, operation_id="version")
def version():
    return Version(name=settings.app_name, version=settings.version)


@router.get("/capabilities", operation_id="capabilities")
def capabilities():
    return {"advisory_only": True, "closed_loop_control": False, "plc_write": False,
            "modes": ["synthetic_demonstration", "offline_planning", "historical_replay", "online_shadow_readonly (gateway required)",
                      "advisory_optimisation"],
            "closed_loop_commissioning": "NOT IMPLEMENTED - future feature flag requiring HAZOP/LOPA, FAT/SAT, SIS independence and written authorisation",
            "thermo_provider": settings.thermo_provider, "capacities_mva": [9, 16.5, 30, 45, 60, 81],
            "enclosures": ["open", "semi-closed", "closed"], "furnace_types": ["AC-3-electrode-SAF"],
            "not_supported": ["DC SAF (separate configuration, not implemented)", "LC FeCr route"],
            "auth": "JWT HS256 (OIDC adapter point in core/security.py)", "synthetic_label": "SYNTHETIC / NOT PLANT-VALIDATED"}
