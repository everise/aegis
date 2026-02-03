"""
Skills API endpoints.

Handles skill listing, execution, and status tracking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    SkillExecution as DBSkillExecution,
    SkillExecutionStatus,
    get_db_session,
)
from app.services.skill_executor import (
    SkillExecutor,
    SkillResult,
    SkillStatus,
    SKILL_REGISTRY,
)


router = APIRouter()


# Request/Response Models
class SkillInfo(BaseModel):
    """Information about an available skill."""
    name: str
    description: str
    submit_endpoint: str
    parameters: Optional[Dict[str, Any]] = None


class SkillListResponse(BaseModel):
    """Response for listing skills."""
    skills: List[SkillInfo]
    total: int


class SkillExecuteRequest(BaseModel):
    """Request to execute a skill."""
    skill_name: str = Field(..., description="Name of the skill to execute")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the skill"
    )
    message_id: Optional[int] = Field(
        default=None,
        description="Associated message ID for tracking"
    )


class SkillExecuteResponse(BaseModel):
    """Response from skill execution."""
    execution_id: Optional[int] = None
    skill_name: str
    status: str
    task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    poll_count: int = 0


class SkillExecutionRecord(BaseModel):
    """Record of a skill execution."""
    id: int
    message_id: int
    skill_name: str
    status: str
    request_params: Optional[Dict[str, Any]]
    remote_task_id: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    poll_count: int
    submitted_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# API Endpoints
@router.get("", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    """List all available skills."""
    executor = SkillExecutor()
    skills_list = executor.list_skills()
    
    skills = []
    for skill_info in skills_list:
        skill_class = SKILL_REGISTRY.get(skill_info["name"])
        if skill_class:
            skills.append(SkillInfo(
                name=skill_info["name"],
                description=skill_info["description"],
                submit_endpoint=skill_class.submit_endpoint,
            ))
    
    return SkillListResponse(
        skills=skills,
        total=len(skills),
    )


@router.get("/{skill_name}", response_model=SkillInfo)
async def get_skill(skill_name: str) -> SkillInfo:
    """Get information about a specific skill."""
    skill_class = SKILL_REGISTRY.get(skill_name)
    
    if not skill_class:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{skill_name}' not found"
        )
    
    return SkillInfo(
        name=skill_class.name,
        description=skill_class.description,
        submit_endpoint=skill_class.submit_endpoint,
    )


@router.post("/execute", response_model=SkillExecuteResponse)
async def execute_skill(
    request: SkillExecuteRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SkillExecuteResponse:
    """
    Execute a skill and return the result.
    
    This is a synchronous endpoint that waits for skill completion.
    For long-running skills, consider using the async endpoint.
    """
    # Validate skill exists
    if request.skill_name not in SKILL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{request.skill_name}' not found"
        )
    
    # Create execution record if message_id provided
    execution_id = None
    if request.message_id:
        execution = DBSkillExecution(
            message_id=request.message_id,
            skill_name=request.skill_name,
            status=SkillExecutionStatus.PENDING,
            request_params=request.params,
        )
        db.add(execution)
        await db.flush()
        execution_id = execution.id
    
    # Execute skill
    executor = SkillExecutor()
    try:
        result = await executor.execute(request.skill_name, request.params)
        
        # Update execution record
        if execution_id:
            execution.status = _map_status(result.status)
            execution.remote_task_id = result.task_id
            execution.result_json = result.result
            execution.error_message = result.error
            execution.poll_count = result.poll_count
            execution.submitted_at = result.submitted_at
            execution.completed_at = result.completed_at
            await db.flush()
        
        return SkillExecuteResponse(
            execution_id=execution_id,
            skill_name=result.skill_name,
            status=result.status.value,
            task_id=result.task_id,
            result=result.result,
            error=result.error,
            poll_count=result.poll_count,
        )
    
    finally:
        await executor.close()


@router.post("/execute/async", response_model=SkillExecuteResponse)
async def execute_skill_async(
    request: SkillExecuteRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SkillExecuteResponse:
    """
    Submit a skill for async execution.
    
    Returns immediately with a task ID for polling status.
    """
    # Validate skill exists
    if request.skill_name not in SKILL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{request.skill_name}' not found"
        )
    
    # Create execution record
    execution = None
    if request.message_id:
        execution = DBSkillExecution(
            message_id=request.message_id,
            skill_name=request.skill_name,
            status=SkillExecutionStatus.PENDING,
            request_params=request.params,
        )
        db.add(execution)
        await db.flush()
    
    # Submit to remote API (just submit, don't poll)
    executor = SkillExecutor()
    try:
        skill = executor.get_skill(request.skill_name)
        validated_params = skill.validate_params(request.params)
        task_id = await skill.submit(validated_params)
        
        # Update execution record
        if execution:
            execution.status = SkillExecutionStatus.SUBMITTED
            execution.remote_task_id = task_id
            execution.submitted_at = datetime.utcnow()
            await db.flush()
        
        return SkillExecuteResponse(
            execution_id=execution.id if execution else None,
            skill_name=request.skill_name,
            status="submitted",
            task_id=task_id,
        )
    
    except Exception as e:
        if execution:
            execution.status = SkillExecutionStatus.FAILED
            execution.error_message = str(e)
            await db.flush()
        
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        await executor.close()


@router.get("/executions/{execution_id}", response_model=SkillExecutionRecord)
async def get_execution(
    execution_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> SkillExecutionRecord:
    """Get status of a skill execution."""
    query = select(DBSkillExecution).where(DBSkillExecution.id == execution_id)
    result = await db.execute(query)
    execution = result.scalar_one_or_none()
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found"
        )
    
    return SkillExecutionRecord(
        id=execution.id,
        message_id=execution.message_id,
        skill_name=execution.skill_name,
        status=execution.status.value,
        request_params=execution.request_params,
        remote_task_id=execution.remote_task_id,
        result=execution.result_json,
        error=execution.error_message,
        poll_count=execution.poll_count,
        submitted_at=execution.submitted_at,
        completed_at=execution.completed_at,
        created_at=execution.created_at,
    )


@router.get("/executions", response_model=List[SkillExecutionRecord])
async def list_executions(
    message_id: Optional[int] = None,
    skill_name: Optional[str] = None,
    status: Optional[SkillExecutionStatus] = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session),
) -> List[SkillExecutionRecord]:
    """List skill executions with optional filters."""
    query = select(DBSkillExecution)
    
    if message_id:
        query = query.where(DBSkillExecution.message_id == message_id)
    if skill_name:
        query = query.where(DBSkillExecution.skill_name == skill_name)
    if status:
        query = query.where(DBSkillExecution.status == status)
    
    query = query.order_by(DBSkillExecution.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    executions = result.scalars().all()
    
    return [
        SkillExecutionRecord(
            id=e.id,
            message_id=e.message_id,
            skill_name=e.skill_name,
            status=e.status.value,
            request_params=e.request_params,
            remote_task_id=e.remote_task_id,
            result=e.result_json,
            error=e.error_message,
            poll_count=e.poll_count,
            submitted_at=e.submitted_at,
            completed_at=e.completed_at,
            created_at=e.created_at,
        )
        for e in executions
    ]


def _map_status(skill_status: SkillStatus) -> SkillExecutionStatus:
    """Map SkillStatus to SkillExecutionStatus."""
    mapping = {
        SkillStatus.PENDING: SkillExecutionStatus.PENDING,
        SkillStatus.SUBMITTED: SkillExecutionStatus.SUBMITTED,
        SkillStatus.POLLING: SkillExecutionStatus.POLLING,
        SkillStatus.COMPLETED: SkillExecutionStatus.COMPLETED,
        SkillStatus.FAILED: SkillExecutionStatus.FAILED,
        SkillStatus.TIMEOUT: SkillExecutionStatus.TIMEOUT,
    }
    return mapping.get(skill_status, SkillExecutionStatus.FAILED)
