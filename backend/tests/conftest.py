"""
Pytest fixtures for Aegis backend tests.

Provides database sessions, mock clients, and test data factories.
"""

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import Settings, reset_settings
from app.database import (
    Base,
    Session,
    Message,
    Trajectory,
    SkillExecution,
    TrainingJob,
    TrainingSample,
    SessionStatus,
    MessageRole,
    SkillExecutionStatus,
    TrainingJobStatus,
    reset_db,
)


# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with overridden values (no YAML file needed)."""
    reset_settings()
    settings = Settings(
        database_url=TEST_DATABASE_URL,
        debug=True,
        remote_api_base_url="http://mock-api:8000/api/v1/tasks",
    )
    # Inject into the global singleton so get_settings() returns it.
    import app.config as _cfg
    _cfg._settings = settings
    
    yield settings
    
    reset_settings()


@pytest_asyncio.fixture
async def async_engine(test_settings):
    """Create an async engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def app_client(test_settings, async_engine) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing the FastAPI app."""
    # Import here to avoid circular imports
    from app.main import create_app
    from app.database import get_db_session
    
    app = create_app()
    
    # Override database dependency
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async def override_get_db():
        async with session_factory() as session:
            yield session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# Test Data Factories
class SessionFactory:
    """Factory for creating test Session objects."""
    
    @staticmethod
    def create(
        status: SessionStatus = SessionStatus.ACTIVE,
        task_type: str = "text_to_image",
        metadata: dict = None,
    ) -> Session:
        return Session(
            status=status,
            task_type=task_type,
            metadata_json=metadata or {},
        )


class MessageFactory:
    """Factory for creating test Message objects."""
    
    @staticmethod
    def create(
        session_id: int = 1,
        role: MessageRole = MessageRole.USER,
        content: str = "Test message content",
        plan_json: dict = None,
    ) -> Message:
        return Message(
            session_id=session_id,
            role=role,
            content=content,
            plan_json=plan_json,
        )


class TrajectoryFactory:
    """Factory for creating test Trajectory objects."""
    
    @staticmethod
    def create(
        session_id: int = 1,
        steps: list = None,
        total_reward: float = 0.0,
        policy_version: str = "v1.0",
        task_type: str = "text_to_image",
    ) -> Trajectory:
        default_steps = [
            {
                "state": {"prompt": "test prompt"},
                "action": {"type": "generate", "params": {}},
                "reward": 1.0,
                "next_state": {"image_url": "http://example.com/img.png"},
            }
        ]
        return Trajectory(
            session_id=session_id,
            steps_json=steps or default_steps,
            total_reward=total_reward,
            policy_version=policy_version,
            task_type=task_type,
        )


class SkillExecutionFactory:
    """Factory for creating test SkillExecution objects."""
    
    @staticmethod
    def create(
        message_id: int = 1,
        skill_name: str = "text_to_image",
        status: SkillExecutionStatus = SkillExecutionStatus.PENDING,
        request_params: dict = None,
        remote_task_id: str = None,
        result_json: dict = None,
    ) -> SkillExecution:
        return SkillExecution(
            message_id=message_id,
            skill_name=skill_name,
            status=status,
            request_params=request_params or {"prompt": "test"},
            remote_task_id=remote_task_id,
            result_json=result_json,
        )


class TrainingJobFactory:
    """Factory for creating test TrainingJob objects."""
    
    @staticmethod
    def create(
        status: TrainingJobStatus = TrainingJobStatus.PENDING,
        config: dict = None,
        policy_version: str = "v1.0",
        total_epochs: int = 10,
    ) -> TrainingJob:
        default_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "discount_factor": 0.99,
        }
        return TrainingJob(
            status=status,
            config_json=config or default_config,
            policy_version=policy_version,
            total_epochs=total_epochs,
        )


class TrainingSampleFactory:
    """Factory for creating test TrainingSample objects."""
    
    @staticmethod
    def create(
        training_job_id: int = 1,
        trajectory_id: int = None,
        advantage: float = 0.5,
        normalized_advantage: float = None,
    ) -> TrainingSample:
        return TrainingSample(
            training_job_id=training_job_id,
            trajectory_id=trajectory_id,
            advantage=advantage,
            normalized_advantage=normalized_advantage,
        )


# Fixture for factories
@pytest.fixture
def session_factory():
    """Get SessionFactory for tests."""
    return SessionFactory


@pytest.fixture
def message_factory():
    """Get MessageFactory for tests."""
    return MessageFactory


@pytest.fixture
def trajectory_factory():
    """Get TrajectoryFactory for tests."""
    return TrajectoryFactory


@pytest.fixture
def skill_execution_factory():
    """Get SkillExecutionFactory for tests."""
    return SkillExecutionFactory


@pytest.fixture
def training_job_factory():
    """Get TrainingJobFactory for tests."""
    return TrainingJobFactory


@pytest.fixture
def training_sample_factory():
    """Get TrainingSampleFactory for tests."""
    return TrainingSampleFactory


# Mock HTTP responses
@pytest.fixture
def mock_skill_submit_response():
    """Mock response for skill submit endpoint."""
    return {
        "task_id": "mock-task-123",
        "status": "pending",
        "message": "Task submitted successfully",
    }


@pytest.fixture
def mock_skill_poll_response():
    """Mock response for skill poll endpoint (completed)."""
    return {
        "task_id": "mock-task-123",
        "status": "completed",
        "result": {
            "image_url": "http://example.com/generated-image.png",
            "width": 512,
            "height": 512,
        },
    }


@pytest.fixture
def mock_skill_poll_pending_response():
    """Mock response for skill poll endpoint (still pending)."""
    return {
        "task_id": "mock-task-123",
        "status": "pending",
        "progress": 0.5,
    }
