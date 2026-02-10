"""
Planning Models API endpoints.

Provides endpoints for listing available planning models
and switching the active model.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.planning.registry import get_planning_registry


router = APIRouter()


# ── Response schemas ──────────────────────────────────────────────

class PlanningModelResponse(BaseModel):
    """Description of a single planning model."""
    id: str
    name: str
    provider: str
    description: str
    supports_vision: bool
    supports_streaming: bool


class PlanningModelListResponse(BaseModel):
    """Response for listing available planning models."""
    models: List[PlanningModelResponse]
    active_model_id: Optional[str]


class SetActiveModelRequest(BaseModel):
    """Request to switch the active planning model."""
    model_id: str = Field(..., min_length=1, description="ID of the model to activate")


class SetActiveModelResponse(BaseModel):
    """Response after switching the active model."""
    active_model_id: str
    model: PlanningModelResponse


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("", response_model=PlanningModelListResponse)
async def list_planning_models() -> PlanningModelListResponse:
    """List all available planning models and the currently active one."""
    registry = get_planning_registry()
    models = registry.list_models()
    return PlanningModelListResponse(
        models=[
            PlanningModelResponse(
                id=m.id,
                name=m.name,
                provider=m.provider,
                description=m.description,
                supports_vision=m.supports_vision,
                supports_streaming=m.supports_streaming,
            )
            for m in models
        ],
        active_model_id=registry.get_active_model_id(),
    )


@router.get("/active", response_model=PlanningModelResponse)
async def get_active_planning_model() -> PlanningModelResponse:
    """Get the currently active planning model."""
    registry = get_planning_registry()
    model = registry.get_active_model()
    info = model.info()
    return PlanningModelResponse(
        id=info.id,
        name=info.name,
        provider=info.provider,
        description=info.description,
        supports_vision=info.supports_vision,
        supports_streaming=info.supports_streaming,
    )


@router.put("/active", response_model=SetActiveModelResponse)
async def set_active_planning_model(
    request: SetActiveModelRequest,
) -> SetActiveModelResponse:
    """Switch the active planning model."""
    registry = get_planning_registry()
    try:
        model = registry.set_active_model(request.model_id)
    except KeyError:
        available = [m.id for m in registry.list_models()]
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{request.model_id}'. Available: {available}",
        )
    info = model.info()
    return SetActiveModelResponse(
        active_model_id=info.id,
        model=PlanningModelResponse(
            id=info.id,
            name=info.name,
            provider=info.provider,
            description=info.description,
            supports_vision=info.supports_vision,
            supports_streaming=info.supports_streaming,
        ),
    )
