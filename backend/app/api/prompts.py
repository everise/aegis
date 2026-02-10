"""
Prompts API endpoints.

Handles CRUD operations for system prompt templates.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Prompt as DBPrompt, get_db_session


router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PromptCreate(BaseModel):
    """Request to create a new prompt."""
    name: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    is_active: bool = False


class PromptUpdate(BaseModel):
    """Request to update an existing prompt."""
    name: Optional[str] = Field(default=None, max_length=200)
    content: Optional[str] = None
    is_active: Optional[bool] = None


class PromptResponse(BaseModel):
    """Single prompt response."""
    id: int
    name: str
    content: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PromptListResponse(BaseModel):
    """Paginated list of prompts."""
    prompts: List[PromptResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=PromptResponse)
async def create_prompt(
    request: PromptCreate,
    db: AsyncSession = Depends(get_db_session),
) -> PromptResponse:
    """Create a new prompt template."""
    prompt = DBPrompt(
        name=request.name,
        content=request.content,
        is_active=request.is_active,
    )

    # If this prompt should be active, deactivate all others first
    if request.is_active:
        await db.execute(
            update(DBPrompt).where(DBPrompt.is_active == True).values(is_active=False)  # noqa: E712
        )

    db.add(prompt)
    await db.flush()
    await db.refresh(prompt)

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


@router.get("", response_model=PromptListResponse)
async def list_prompts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
) -> PromptListResponse:
    """List prompts with pagination."""
    # Total count
    count_result = await db.execute(select(DBPrompt.id))
    total = len(count_result.all())

    # Paginated results
    offset = (page - 1) * page_size
    query = (
        select(DBPrompt)
        .order_by(DBPrompt.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    prompts = result.scalars().all()

    return PromptListResponse(
        prompts=[
            PromptResponse(
                id=p.id,
                name=p.name,
                content=p.content,
                is_active=p.is_active,
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            for p in prompts
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
    prompt_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> PromptResponse:
    """Get a single prompt by ID."""
    result = await db.execute(select(DBPrompt).where(DBPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


@router.patch("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
    prompt_id: int,
    request: PromptUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> PromptResponse:
    """Update an existing prompt."""
    result = await db.execute(select(DBPrompt).where(DBPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

    if request.name is not None:
        prompt.name = request.name
    if request.content is not None:
        prompt.content = request.content
    if request.is_active is not None:
        # If activating, deactivate all others
        if request.is_active:
            await db.execute(
                update(DBPrompt)
                .where(DBPrompt.is_active == True, DBPrompt.id != prompt_id)  # noqa: E712
                .values(is_active=False)
            )
        prompt.is_active = request.is_active

    prompt.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(prompt)

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a prompt."""
    result = await db.execute(select(DBPrompt).where(DBPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

    await db.delete(prompt)
    return {"message": f"Prompt {prompt_id} deleted"}


@router.put("/{prompt_id}/activate", response_model=PromptResponse)
async def activate_prompt(
    prompt_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> PromptResponse:
    """Set a prompt as the active prompt (deactivates all others)."""
    result = await db.execute(select(DBPrompt).where(DBPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

    # Deactivate all
    await db.execute(
        update(DBPrompt).where(DBPrompt.is_active == True).values(is_active=False)  # noqa: E712
    )
    prompt.is_active = True
    prompt.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(prompt)

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        is_active=prompt.is_active,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )
