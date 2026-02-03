"""
Training API endpoints.

Handles RL training job management and monitoring.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    TrainingJob as DBTrainingJob,
    TrainingSample as DBTrainingSample,
    Trajectory as DBTrajectory,
    TrainingJobStatus,
    get_db_session,
)
from app.rl.trainer import RLTrainer, TrainingConfig, TrainerStatus


router = APIRouter()

# Global trainer instance (in production, use proper state management)
_trainer: Optional[RLTrainer] = None


def get_trainer() -> RLTrainer:
    """Get or create trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = RLTrainer()
    return _trainer


# Request/Response Models
class TrainingConfigRequest(BaseModel):
    """Request to configure training."""
    total_epochs: int = Field(default=100, ge=1, le=10000)
    steps_per_epoch: int = Field(default=1000, ge=1, le=100000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=3e-4, gt=0, lt=1)
    discount_factor: float = Field(default=0.99, ge=0, le=1)
    buffer_size: int = Field(default=10000, ge=100, le=1000000)
    use_cross_policy: bool = Field(default=True)
    use_task_normalization: bool = Field(default=True)


class TrainingJobCreate(BaseModel):
    """Request to create a training job."""
    config: Optional[TrainingConfigRequest] = None
    policy_version: Optional[str] = Field(default="v1.0")


class TrainingJobResponse(BaseModel):
    """Response for a training job."""
    id: int
    status: str
    policy_version: Optional[str]
    config: Optional[Dict[str, Any]]
    total_epochs: int
    current_epoch: int
    total_steps: int
    current_step: int
    metrics: Dict[str, Any]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingMetricsResponse(BaseModel):
    """Response containing training metrics."""
    job_id: int
    epoch: int
    step: int
    policy_loss: float
    value_loss: float
    mean_return: float
    mean_reward: float
    buffer_size: int


class TrainingStatusResponse(BaseModel):
    """Response for trainer status."""
    status: str
    current_epoch: int
    current_step: int
    policy_version: str
    best_return: float
    buffer_size: int
    config: Dict[str, Any]


# Background task for training
async def run_training_background(job_id: int, config: TrainingConfig):
    """Background task to run training."""
    trainer = get_trainer()
    trainer.config = config
    
    # Note: In a real implementation, you would:
    # 1. Update job status in DB
    # 2. Run training epochs
    # 3. Save checkpoints
    # 4. Update metrics in DB
    
    async for metrics in trainer.train():
        # Update job metrics (would need DB session here)
        pass


# API Endpoints
@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    request: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> TrainingJobResponse:
    """Create a new training job."""
    # Create config
    config_dict = {}
    if request.config:
        config = TrainingConfig(
            total_epochs=request.config.total_epochs,
            steps_per_epoch=request.config.steps_per_epoch,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            discount_factor=request.config.discount_factor,
            buffer_size=request.config.buffer_size,
            use_cross_policy=request.config.use_cross_policy,
            use_task_normalization=request.config.use_task_normalization,
        )
        config_dict = config.to_dict()
    else:
        config = TrainingConfig()
        config_dict = config.to_dict()
    
    # Create job record
    job = DBTrainingJob(
        status=TrainingJobStatus.PENDING,
        config_json=config_dict,
        policy_version=request.policy_version,
        total_epochs=config.total_epochs,
    )
    
    db.add(job)
    await db.flush()
    await db.refresh(job)
    
    return TrainingJobResponse(
        id=job.id,
        status=job.status.value,
        policy_version=job.policy_version,
        config=job.config_json,
        total_epochs=job.total_epochs,
        current_epoch=job.current_epoch,
        total_steps=job.total_steps,
        current_step=job.current_step,
        metrics=job.metrics_json,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[TrainingJobStatus] = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
) -> List[TrainingJobResponse]:
    """List training jobs."""
    query = select(DBTrainingJob)
    
    if status:
        query = query.where(DBTrainingJob.status == status)
    
    query = query.order_by(DBTrainingJob.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return [
        TrainingJobResponse(
            id=j.id,
            status=j.status.value,
            policy_version=j.policy_version,
            config=j.config_json,
            total_epochs=j.total_epochs,
            current_epoch=j.current_epoch,
            total_steps=j.total_steps,
            current_step=j.current_step,
            metrics=j.metrics_json,
            started_at=j.started_at,
            completed_at=j.completed_at,
            created_at=j.created_at,
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> TrainingJobResponse:
    """Get a specific training job."""
    query = select(DBTrainingJob).where(DBTrainingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return TrainingJobResponse(
        id=job.id,
        status=job.status.value,
        policy_version=job.policy_version,
        config=job.config_json,
        total_epochs=job.total_epochs,
        current_epoch=job.current_epoch,
        total_steps=job.total_steps,
        current_step=job.current_step,
        metrics=job.metrics_json,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


@router.post("/jobs/{job_id}/start")
async def start_training_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
):
    """Start a pending training job."""
    query = select(DBTrainingJob).where(DBTrainingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    if job.status != TrainingJobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not in pending status"
        )
    
    # Update status
    job.status = TrainingJobStatus.RUNNING
    job.started_at = datetime.utcnow()
    await db.flush()
    
    # Start training in background
    config = TrainingConfig(**(job.config_json or {}))
    background_tasks.add_task(run_training_background, job_id, config)
    
    return {"message": f"Training job {job_id} started"}


@router.post("/jobs/{job_id}/pause")
async def pause_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Pause a running training job."""
    trainer = get_trainer()
    trainer.pause()
    
    query = select(DBTrainingJob).where(DBTrainingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if job and job.status == TrainingJobStatus.RUNNING:
        # Note: Would need proper status tracking
        pass
    
    return {"message": f"Training job {job_id} paused"}


@router.post("/jobs/{job_id}/resume")
async def resume_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Resume a paused training job."""
    trainer = get_trainer()
    trainer.resume()
    
    return {"message": f"Training job {job_id} resumed"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel a training job."""
    query = select(DBTrainingJob).where(DBTrainingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job.status = TrainingJobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    
    # Reset trainer
    trainer = get_trainer()
    trainer.reset()
    
    return {"message": f"Training job {job_id} cancelled"}


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current trainer status."""
    trainer = get_trainer()
    summary = trainer.get_training_summary()
    
    return TrainingStatusResponse(
        status=summary["status"],
        current_epoch=summary["current_epoch"],
        current_step=summary["current_step"],
        policy_version=summary["policy_version"],
        best_return=summary["best_return"],
        buffer_size=summary["buffer_size"],
        config=summary["config"],
    )


@router.get("/metrics")
async def get_training_metrics(
    job_id: Optional[int] = None,
    last_n: int = Query(default=100, ge=1, le=1000),
):
    """Get training metrics."""
    trainer = get_trainer()
    
    # Return recent metrics from trainer
    metrics = trainer._metrics_history[-last_n:]
    
    return {
        "metrics": [m.to_dict() for m in metrics],
        "count": len(metrics),
    }


@router.get("/buffer/stats")
async def get_buffer_statistics():
    """Get replay buffer statistics."""
    trainer = get_trainer()
    
    return {
        "replay_buffer": trainer.replay_buffer.get_statistics(),
        "task_buffer": trainer.task_buffer.get_statistics(),
    }


@router.post("/buffer/add")
async def add_trajectory_to_buffer(
    trajectory_data: Dict[str, Any],
):
    """
    Manually add a trajectory to the replay buffer.
    
    Used for populating buffer with demonstration data.
    """
    from app.rl.trajectory import Trajectory
    
    trainer = get_trainer()
    
    try:
        trajectory = Trajectory.from_dict(trajectory_data)
        trainer.add_trajectory(trajectory)
        
        return {
            "message": "Trajectory added to buffer",
            "buffer_size": trainer.replay_buffer.size,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid trajectory data: {str(e)}")
