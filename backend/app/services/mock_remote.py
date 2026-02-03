"""
Mock remote API service for development and testing.

Simulates the submit-poll pattern for async skill executions.
Provides endpoints that mimic real image generation APIs.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()


# In-memory task storage for mock
_mock_tasks: Dict[str, dict] = {}


class TaskStatus(str, Enum):
    """Status of a mock task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Request/Response Models
class TextToImageRequest(BaseModel):
    """Request for text-to-image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=100)
    seed: Optional[int] = Field(None)


class EvaluateImageRequest(BaseModel):
    """Request for image evaluation."""
    image_url: str = Field(..., description="URL of image to evaluate")
    criteria: list[str] = Field(
        default=["quality", "aesthetics", "prompt_alignment"],
        description="Evaluation criteria"
    )


class RepairImageRequest(BaseModel):
    """Request for image repair/inpainting."""
    image_url: str = Field(..., description="URL of image to repair")
    mask_url: Optional[str] = Field(None, description="URL of mask image")
    prompt: str = Field(..., description="Repair instruction prompt")
    strength: float = Field(default=0.75, ge=0.0, le=1.0)


class SubmitResponse(BaseModel):
    """Response after submitting a task."""
    task_id: str
    status: TaskStatus
    message: str


class PollResponse(BaseModel):
    """Response when polling task status."""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# Mock processing functions
async def _process_text_to_image(task_id: str, request: dict):
    """Simulate text-to-image processing."""
    await asyncio.sleep(2)  # Simulate processing time
    
    _mock_tasks[task_id]["status"] = TaskStatus.PROCESSING
    _mock_tasks[task_id]["progress"] = 0.5
    
    await asyncio.sleep(1)
    
    # Generate mock result
    _mock_tasks[task_id]["status"] = TaskStatus.COMPLETED
    _mock_tasks[task_id]["progress"] = 1.0
    _mock_tasks[task_id]["result"] = {
        "image_url": f"http://mock-cdn.example.com/images/{task_id}.png",
        "width": request.get("width", 512),
        "height": request.get("height", 512),
        "seed": request.get("seed") or 12345,
        "generated_at": datetime.utcnow().isoformat(),
    }


async def _process_evaluate_image(task_id: str, request: dict):
    """Simulate image evaluation processing."""
    await asyncio.sleep(1)
    
    _mock_tasks[task_id]["status"] = TaskStatus.COMPLETED
    _mock_tasks[task_id]["progress"] = 1.0
    
    # Generate mock evaluation scores
    criteria = request.get("criteria", ["quality"])
    scores = {criterion: round(0.7 + 0.3 * (hash(criterion + task_id) % 100) / 100, 2) 
              for criterion in criteria}
    
    _mock_tasks[task_id]["result"] = {
        "scores": scores,
        "overall_score": sum(scores.values()) / len(scores),
        "feedback": "Image meets quality standards." if sum(scores.values()) / len(scores) > 0.7 
                   else "Image quality could be improved.",
        "evaluated_at": datetime.utcnow().isoformat(),
    }


async def _process_repair_image(task_id: str, request: dict):
    """Simulate image repair processing."""
    await asyncio.sleep(2)
    
    _mock_tasks[task_id]["status"] = TaskStatus.PROCESSING
    _mock_tasks[task_id]["progress"] = 0.5
    
    await asyncio.sleep(1)
    
    _mock_tasks[task_id]["status"] = TaskStatus.COMPLETED
    _mock_tasks[task_id]["progress"] = 1.0
    _mock_tasks[task_id]["result"] = {
        "image_url": f"http://mock-cdn.example.com/repaired/{task_id}.png",
        "original_url": request.get("image_url"),
        "repaired_at": datetime.utcnow().isoformat(),
    }


# API Endpoints
@router.post("/text-to-image/submit", response_model=SubmitResponse)
async def submit_text_to_image(request: TextToImageRequest) -> SubmitResponse:
    """Submit a text-to-image generation task."""
    task_id = str(uuid.uuid4())
    
    _mock_tasks[task_id] = {
        "type": "text_to_image",
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "request": request.model_dump(),
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    # Start async processing
    asyncio.create_task(_process_text_to_image(task_id, request.model_dump()))
    
    return SubmitResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task submitted successfully",
    )


@router.post("/evaluate-image/submit", response_model=SubmitResponse)
async def submit_evaluate_image(request: EvaluateImageRequest) -> SubmitResponse:
    """Submit an image evaluation task."""
    task_id = str(uuid.uuid4())
    
    _mock_tasks[task_id] = {
        "type": "evaluate_image",
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "request": request.model_dump(),
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    asyncio.create_task(_process_evaluate_image(task_id, request.model_dump()))
    
    return SubmitResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task submitted successfully",
    )


@router.post("/repair-image/submit", response_model=SubmitResponse)
async def submit_repair_image(request: RepairImageRequest) -> SubmitResponse:
    """Submit an image repair task."""
    task_id = str(uuid.uuid4())
    
    _mock_tasks[task_id] = {
        "type": "repair_image",
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "request": request.model_dump(),
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    asyncio.create_task(_process_repair_image(task_id, request.model_dump()))
    
    return SubmitResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task submitted successfully",
    )


@router.get("/tasks/{task_id}/poll", response_model=PollResponse)
async def poll_task_status(task_id: str) -> PollResponse:
    """Poll the status of a task."""
    if task_id not in _mock_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = _mock_tasks[task_id]
    
    return PollResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task["result"],
        error=task["error"],
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or processing task."""
    if task_id not in _mock_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = _mock_tasks[task_id]
    if task["status"] == TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")
    
    task["status"] = TaskStatus.FAILED
    task["error"] = "Task cancelled by user"
    
    return {"message": f"Task {task_id} cancelled"}


# Utility functions for testing
def clear_mock_tasks():
    """Clear all mock tasks. For testing only."""
    _mock_tasks.clear()


def get_mock_task(task_id: str) -> Optional[dict]:
    """Get a mock task by ID. For testing only."""
    return _mock_tasks.get(task_id)


def set_mock_task_status(task_id: str, status: TaskStatus, result: dict = None):
    """Set mock task status. For testing only."""
    if task_id in _mock_tasks:
        _mock_tasks[task_id]["status"] = status
        if result:
            _mock_tasks[task_id]["result"] = result
