"""
Messages API endpoints.

Handles message operations within sessions, including
sending messages and triggering agent planning.
"""

from datetime import datetime
from typing import List, Optional, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import (
    Session as DBSession,
    Message as DBMessage,
    MessageRole,
    SessionStatus,
    get_db_session,
    get_db_session_context,
)
from app.services.react_planner import ReActPlanner
from app.services.sse_manager import (
    SSEConnection,
    get_sse_manager,
    stream_plan_execution,
)


router = APIRouter()


# Request/Response Models
class MessageCreate(BaseModel):
    """Request to create a new message."""
    content: str = Field(..., min_length=1, description="Message content")
    role: Optional[MessageRole] = Field(
        default=MessageRole.USER,
        description="Message role"
    )


class MessageResponse(BaseModel):
    """Message response model."""
    id: int
    session_id: int
    role: str
    content: str
    plan_json: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """Response for listing messages."""
    messages: List[MessageResponse]
    total: int


class PlanResponse(BaseModel):
    """Response containing execution plan."""
    message_id: int
    plan: dict
    status: str


# API Endpoints
@router.post("/{session_id}/messages", response_model=MessageResponse)
async def create_message(
    session_id: int,
    request: MessageCreate,
    db: AsyncSession = Depends(get_db_session),
) -> MessageResponse:
    """
    Create a new message in a session.
    
    For user messages, this also triggers the agent planning process.
    """
    # Verify session exists
    session_query = select(DBSession).where(DBSession.id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is not active"
        )
    
    # Create the message
    message = DBMessage(
        session_id=session_id,
        role=request.role,
        content=request.content,
    )
    
    db.add(message)
    await db.flush()
    await db.refresh(message)
    
    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role.value,
        content=message.content,
        plan_json=message.plan_json,
        created_at=message.created_at,
    )


@router.get("/{session_id}/messages", response_model=MessageListResponse)
async def list_messages(
    session_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> MessageListResponse:
    """List messages in a session."""
    # Verify session exists
    session_query = select(DBSession).where(DBSession.id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get messages
    query = (
        select(DBMessage)
        .where(DBMessage.session_id == session_id)
        .order_by(DBMessage.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # Get total count
    count_query = select(DBMessage.id).where(DBMessage.session_id == session_id)
    count_result = await db.execute(count_query)
    total = len(count_result.all())
    
    return MessageListResponse(
        messages=[
            MessageResponse(
                id=m.id,
                session_id=m.session_id,
                role=m.role.value,
                content=m.content,
                plan_json=m.plan_json,
                created_at=m.created_at,
            )
            for m in messages
        ],
        total=total,
    )


@router.get("/{session_id}/messages/{message_id}", response_model=MessageResponse)
async def get_message(
    session_id: int,
    message_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> MessageResponse:
    """Get a specific message."""
    query = (
        select(DBMessage)
        .where(DBMessage.id == message_id)
        .where(DBMessage.session_id == session_id)
    )
    
    result = await db.execute(query)
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(
            status_code=404,
            detail=f"Message {message_id} not found in session {session_id}"
        )
    
    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role.value,
        content=message.content,
        plan_json=message.plan_json,
        created_at=message.created_at,
    )


@router.post("/{session_id}/chat", response_model=PlanResponse)
async def chat(
    session_id: int,
    request: MessageCreate,
    db: AsyncSession = Depends(get_db_session),
) -> PlanResponse:
    """
    Send a chat message and get agent response.
    
    This endpoint:
    1. Creates the user message
    2. Triggers ReAct planning
    3. Creates the assistant response
    4. Returns the execution plan
    """
    # Verify session exists and is active
    session_query = select(DBSession).where(DBSession.id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is not active"
        )
    
    # Create user message
    user_message = DBMessage(
        session_id=session_id,
        role=MessageRole.USER,
        content=request.content,
    )
    db.add(user_message)
    await db.flush()
    
    # Execute ReAct planning
    planner = ReActPlanner(max_steps=10)
    plan = await planner.execute(request.content, session_id=session_id)
    await planner.close()
    
    # Create assistant message with plan
    assistant_content = plan.final_result.get("image_url", "") if plan.final_result else ""
    if not assistant_content and plan.steps:
        # Use last observation as content
        last_step = plan.steps[-1]
        if last_step.observation:
            assistant_content = str(last_step.observation)
    
    assistant_message = DBMessage(
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=assistant_content or "Task completed",
        plan_json=plan.to_dict(),
    )
    db.add(assistant_message)
    await db.flush()
    await db.refresh(assistant_message)
    
    return PlanResponse(
        message_id=assistant_message.id,
        plan=plan.to_dict(),
        status=plan.status.value,
    )


@router.get("/{session_id}/chat/stream")
async def chat_stream(
    session_id: int,
    message: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = None,
):
    """
    Stream chat response via Server-Sent Events.
    
    Returns a stream of planning events for real-time UI updates.
    Messages are saved to database for persistence.
    """
    import json
    from starlette.background import BackgroundTasks as BT
    
    # Verify session
    session_query = select(DBSession).where(DBSession.id == session_id)
    session_result = await db.execute(session_query)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Save user message to database
    user_message = DBMessage(
        session_id=session_id,
        role=MessageRole.USER,
        content=message,
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)
    
    # Shared state for collecting events
    collected_steps: list = []
    final_result_holder: dict = {"result": None}
    stream_completed = {"value": False}
    
    async def save_assistant_message():
        """Save assistant message to database."""
        # Wait a bit for stream to complete
        import asyncio
        await asyncio.sleep(0.5)
        
        try:
            async with get_db_session_context() as save_db:
                final_result = final_result_holder["result"]
                assistant_content = ""
                if final_result:
                    if isinstance(final_result, dict):
                        assistant_content = final_result.get("image_url", "") or final_result.get("message", "Task completed")
                    else:
                        assistant_content = str(final_result)
                
                assistant_message = DBMessage(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=assistant_content or "Task completed",
                    plan_json={"steps": collected_steps, "final_result": final_result},
                )
                save_db.add(assistant_message)
                await save_db.commit()
                print(f"Saved assistant message for session {session_id}")
        except Exception as save_error:
            print(f"Error saving assistant message: {save_error}")
    
    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        nonlocal collected_steps, final_result_holder, stream_completed
        
        # Send connected event
        yield f"data: {json.dumps({'type': 'connected', 'data': {'session_id': session_id}})}\n\n"
        
        # Execute planning with streaming
        planner = ReActPlanner(max_steps=10)
        
        try:
            async for event in planner.execute_stream(message, session_id):
                yield f"data: {json.dumps(event)}\n\n"
                
                # Collect events for saving
                event_type = event.get("type")
                if event_type in ("thought", "observation", "finished"):
                    collected_steps.append(event)
                
                # Capture final result for saving
                if event_type == "finished":
                    final_result_holder["result"] = event.get("data", {}).get("result", {})
            
            # Send completed event
            yield f"data: {json.dumps({'type': 'completed', 'data': {'message': 'Task completed'}})}\n\n"
            stream_completed["value"] = True
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
        finally:
            await planner.close()
            # Schedule save in background
            import asyncio
            asyncio.create_task(save_assistant_message())
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
