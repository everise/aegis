"""
Sessions API endpoints.

Handles session CRUD operations and session-level actions.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import (
    Session as DBSession,
    SessionStatus,
    get_db_session,
)


router = APIRouter()


# Request/Response Models
class SessionCreate(BaseModel):
    """Request to create a new session."""
    task_type: Optional[str] = Field(
        default="text_to_image",
        description="Type of task for this session"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional session metadata"
    )


class SessionUpdate(BaseModel):
    """Request to update a session."""
    status: Optional[SessionStatus] = None
    task_type: Optional[str] = None
    metadata: Optional[dict] = None


class SessionResponse(BaseModel):
    """Session response model."""
    id: int
    status: str
    task_type: Optional[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[SessionResponse]
    total: int
    page: int
    page_size: int


# API Endpoints
@router.post("", response_model=SessionResponse)
async def create_session(
    request: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Create a new chat session."""
    session = DBSession(
        status=SessionStatus.ACTIVE,
        task_type=request.task_type,
        metadata_json=request.metadata or {},
    )
    
    db.add(session)
    await db.flush()
    await db.refresh(session)
    
    return SessionResponse(
        id=session.id,
        status=session.status.value,
        task_type=session.task_type,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[SessionStatus] = None,
    task_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
) -> SessionListResponse:
    """List sessions with pagination and filtering."""
    # Build query
    query = select(DBSession)
    
    if status:
        query = query.where(DBSession.status == status)
    if task_type:
        query = query.where(DBSession.task_type == task_type)
    
    # Get total count
    count_query = select(DBSession.id)
    if status:
        count_query = count_query.where(DBSession.status == status)
    if task_type:
        count_query = count_query.where(DBSession.task_type == task_type)
    
    count_result = await db.execute(count_query)
    total = len(count_result.all())
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(DBSession.created_at.desc()).offset(offset).limit(page_size)
    query = query.options(selectinload(DBSession.messages))
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                status=s.status.value,
                task_type=s.task_type,
                metadata=s.metadata_json,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=len(s.messages),
            )
            for s in sessions
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Get a specific session by ID."""
    query = (
        select(DBSession)
        .where(DBSession.id == session_id)
        .options(selectinload(DBSession.messages))
    )
    
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return SessionResponse(
        id=session.id,
        status=session.status.value,
        task_type=session.task_type,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
    )


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: int,
    request: SessionUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    """Update a session."""
    query = select(DBSession).where(DBSession.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Update fields
    if request.status is not None:
        session.status = request.status
    if request.task_type is not None:
        session.task_type = request.task_type
    if request.metadata is not None:
        session.metadata_json = request.metadata
    
    session.updated_at = datetime.utcnow()
    
    await db.flush()
    await db.refresh(session)
    
    return SessionResponse(
        id=session.id,
        status=session.status.value,
        task_type=session.task_type,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a session and all associated data."""
    query = select(DBSession).where(DBSession.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    await db.delete(session)
    
    return {"message": f"Session {session_id} deleted"}


@router.post("/{session_id}/complete")
async def complete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Mark a session as completed."""
    query = select(DBSession).where(DBSession.id == session_id)
    result = await db.execute(query)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session.status = SessionStatus.COMPLETED
    session.updated_at = datetime.utcnow()
    
    return {"message": f"Session {session_id} marked as completed"}
