from fastapi import APIRouter
from app.api.schemas.response_models import HealthResponse
from app.core.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns app name and status. Used for monitoring and readiness checks.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version="0.1.0",
    )
