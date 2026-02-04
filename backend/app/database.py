"""
Database models and session management for Aegis.

Uses SQLAlchemy async with aiosqlite for SQLite database operations.
"""

from datetime import datetime
from typing import AsyncGenerator, Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Enum,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Enums for status fields
class SessionStatus(str, PyEnum):
    """Status of a chat session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, PyEnum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SkillExecutionStatus(str, PyEnum):
    """Status of a skill execution."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    POLLING = "polling"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TrainingJobStatus(str, PyEnum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ORM Models
class Session(Base):
    """
    Represents a chat session with the agent.
    
    A session contains multiple messages and can have associated
    trajectories for RL training.
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    status = Column(Enum(SessionStatus), default=SessionStatus.ACTIVE, nullable=False)
    task_type = Column(String(100), nullable=True)  # e.g., "text_to_image", "image_edit"
    metadata_json = Column(JSON, default=dict)  # Additional session metadata

    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    trajectories = relationship("Trajectory", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """
    Represents a single message in a session.
    
    Messages form the conversation history and are used to construct
    trajectories for RL training.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    plan_json = Column(JSON, nullable=True)  # ReAct plan if assistant message
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="messages")
    skill_executions = relationship("SkillExecution", back_populates="message", cascade="all, delete-orphan")


class Trajectory(Base):
    """
    Represents a trajectory for RL training.
    
    A trajectory is a sequence of state-action-reward tuples from
    a conversation interaction. Used for Multi-Turn RL training.
    """
    __tablename__ = "trajectories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    steps_json = Column(JSON, nullable=False)  # List of {state, action, reward, next_state}
    total_reward = Column(Float, default=0.0)
    discount_factor = Column(Float, default=0.99)
    policy_version = Column(String(50), nullable=True)  # For Cross-Policy Sampling
    task_type = Column(String(100), nullable=True)  # For Task Advantage Normalization
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="trajectories")


class SkillExecution(Base):
    """
    Represents an execution of a skill (HTTP API call).
    
    Tracks the submit-poll pattern for async skill executions.
    """
    __tablename__ = "skill_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    skill_name = Column(String(100), nullable=False)  # e.g., "text_to_image"
    status = Column(Enum(SkillExecutionStatus), default=SkillExecutionStatus.PENDING, nullable=False)
    
    # Request/Response
    request_params = Column(JSON, nullable=True)  # Input parameters
    remote_task_id = Column(String(200), nullable=True)  # Task ID from remote API
    result_json = Column(JSON, nullable=True)  # Result from remote API
    error_message = Column(Text, nullable=True)
    
    # Timing
    submitted_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    poll_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    message = relationship("Message", back_populates="skill_executions")


class TrainingJob(Base):
    """
    Represents a training job for the RL model.
    
    Tracks the progress and metrics of a training run.
    """
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(Enum(TrainingJobStatus), default=TrainingJobStatus.PENDING, nullable=False)
    
    # Configuration
    config_json = Column(JSON, nullable=True)  # Training configuration
    policy_version = Column(String(50), nullable=True)
    
    # Progress
    total_epochs = Column(Integer, default=1)
    current_epoch = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    
    # Metrics
    metrics_json = Column(JSON, default=dict)  # Loss, rewards, etc.
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    samples = relationship("TrainingSample", back_populates="training_job", cascade="all, delete-orphan")


class TrainingSample(Base):
    """
    Represents a training sample used in a training job.
    
    Links trajectories to training jobs for batch processing.
    """
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    trajectory_id = Column(Integer, ForeignKey("trajectories.id", ondelete="SET NULL"), nullable=True)
    
    # Processed data
    state_embedding = Column(JSON, nullable=True)  # Encoded state
    action_embedding = Column(JSON, nullable=True)  # Encoded action
    advantage = Column(Float, nullable=True)  # Computed advantage
    normalized_advantage = Column(Float, nullable=True)  # After Task Advantage Normalization
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="samples")


# Database engine and session management
_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Get or create the async database engine."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            future=True,
        )
    return _async_engine


def get_async_session_factory():
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting a database session.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Initialize the database, creating all tables."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close the database connection."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None


def reset_db():
    """Reset database state for testing."""
    global _async_engine, _async_session_factory
    _async_engine = None
    _async_session_factory = None
