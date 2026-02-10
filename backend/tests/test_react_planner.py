"""
Unit tests for ReAct planner and related components.

Tests planning models, ReAct planner, and SSE manager.
"""

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.planning.base import BasePlanningModel, PlanningStep, ActionType, PlanningModelInfo
from app.services.planning.gemini import GeminiPlanningModel
from app.services.planning.kimi import KimiPlanningModel
from app.services.planning.qwen_vl import QwenVLPlanningModel
from app.services.planning.registry import PlanningModelRegistry, get_planning_registry, reset_planning_registry
from app.services.react_planner import (
    ReActPlanner,
    ExecutionPlan,
    PlanStep,
    PlanStatus,
)
from app.services.sse_manager import (
    SSEManager,
    SSEConnection,
    SSEEvent,
    SSEEventType,
    get_sse_manager,
    reset_sse_manager,
    stream_plan_execution,
)
from app.services.skill_executor import SkillResult, SkillStatus


class TestPlanningModelInterface:
    """Tests for the planning model abstract interface and implementations."""

    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    def test_model_info(self, ModelClass):
        """Each model returns valid info."""
        model = ModelClass()
        info = model.info()
        assert isinstance(info, PlanningModelInfo)
        assert info.id
        assert info.name
        assert info.provider

    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    def test_reset(self, ModelClass):
        """Reset clears internal state."""
        model = ModelClass()
        model._step = 5
        model._repairs = 2
        model.reset()
        assert model._step == 0
        assert model._repairs == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    async def test_first_step_generates(self, ModelClass):
        """First step should always be a generate action."""
        model = ModelClass()
        step = await model.get_next_step("Create a sunset image")
        assert step.action == ActionType.GENERATE
        assert "text_to_image" in str(step.action_input)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    async def test_second_step_evaluates(self, ModelClass):
        """Second step should evaluate the generated image."""
        model = ModelClass()
        await model.get_next_step("Create an image")
        observation = {
            "result": {"image_url": "http://example.com/img.png"},
            "status": "completed",
        }
        step = await model.get_next_step("Create an image", observation)
        assert step.action == ActionType.EVALUATE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    async def test_high_quality_finishes(self, ModelClass):
        """High quality score should lead to a finish action."""
        model = ModelClass()
        model.quality_threshold = 0.7
        await model.get_next_step("Create an image")
        await model.get_next_step("Create an image", {"result": {"image_url": "http://example.com/img.png"}})
        step = await model.get_next_step("Create an image", {"result": {"overall_score": 0.9}})
        assert step.action == ActionType.FINISH
        assert step.action_input.get("result") == "success"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    async def test_low_quality_triggers_repair(self, ModelClass):
        """Low quality score should trigger repair."""
        model = ModelClass()
        model.quality_threshold = 0.7
        await model.get_next_step("Create an image")
        await model.get_next_step("Create an image", {"result": {"image_url": "http://example.com/img.png"}})
        step = await model.get_next_step("Create an image", {"result": {"overall_score": 0.5}})
        assert step.action == ActionType.REPAIR

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ModelClass", [GeminiPlanningModel, KimiPlanningModel, QwenVLPlanningModel])
    async def test_chat_completion_wrapper(self, ModelClass):
        """chat_completion returns valid OpenAI-like response."""
        model = ModelClass()
        response = await model.chat_completion([
            {"role": "user", "content": "Generate an image of a cat"}
        ])
        assert "choices" in response
        content = json.loads(response["choices"][0]["message"]["content"])
        assert "thought" in content
        assert "action" in content

    def test_format_step_as_dict(self):
        """PlanningStep to dict conversion."""
        model = GeminiPlanningModel()
        step = PlanningStep(
            thought="Test thought",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image"},
        )
        result = model.format_step_as_dict(step)
        assert result["thought"] == "Test thought"
        assert result["action"] == "generate"


class TestPlanningModelRegistry:
    """Tests for planning model registry."""

    def test_register_and_list(self):
        """Registered models appear in list."""
        reg = PlanningModelRegistry()
        reg.register(GeminiPlanningModel())
        reg.register(KimiPlanningModel())
        assert len(reg.list_models()) == 2

    def test_first_registered_is_default(self):
        """First registered model becomes the default active model."""
        reg = PlanningModelRegistry()
        reg.register(GeminiPlanningModel())
        reg.register(KimiPlanningModel())
        assert reg.get_active_model_id() == "gemini"

    def test_set_active_model(self):
        """set_active_model switches the active model."""
        reg = PlanningModelRegistry()
        reg.register(GeminiPlanningModel())
        reg.register(KimiPlanningModel())
        reg.set_active_model("kimi")
        assert reg.get_active_model_id() == "kimi"

    def test_set_unknown_model_raises(self):
        """Setting an unknown model id raises KeyError."""
        reg = PlanningModelRegistry()
        reg.register(GeminiPlanningModel())
        with pytest.raises(KeyError):
            reg.set_active_model("nonexistent")

    def test_global_registry(self):
        """get_planning_registry returns a populated singleton."""
        reset_planning_registry()
        reg = get_planning_registry()
        ids = [m.id for m in reg.list_models()]
        assert "gemini" in ids
        assert "kimi" in ids
        assert "qwen-vl" in ids


class TestReActPlanner:
    """Tests for ReActPlanner."""
    
    @pytest_asyncio.fixture
    async def mock_skill_executor(self):
        """Create mock skill executor."""
        executor = AsyncMock()
        executor.execute.return_value = SkillResult(
            skill_name="text_to_image",
            status=SkillStatus.COMPLETED,
            result={"image_url": "http://example.com/img.png"},
        )
        return executor
    
    @pytest.mark.asyncio
    async def test_execute_basic_flow(self, mock_skill_executor):
        """Test basic execution flow."""
        # Configure mock to return high quality on evaluation
        def mock_execute(skill_name, params):
            if skill_name == "text_to_image":
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.COMPLETED,
                    result={"image_url": "http://example.com/img.png"},
                )
            elif skill_name == "evaluate_image":
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.COMPLETED,
                    result={"overall_score": 0.9, "scores": {}},
                )
            return SkillResult(skill_name=skill_name, status=SkillStatus.COMPLETED, result={})
        
        mock_skill_executor.execute.side_effect = mock_execute
        
        planner = ReActPlanner(
            skill_executor=mock_skill_executor,
            planning_model=GeminiPlanningModel(),
            max_steps=10,
        )
        
        plan = await planner.execute("Generate a beautiful sunset")
        
        assert plan.status == PlanStatus.COMPLETED
        assert len(plan.steps) >= 2  # At least generate and evaluate
        assert plan.final_result is not None
    
    @pytest.mark.asyncio
    async def test_execute_respects_max_steps(self, mock_skill_executor):
        """Test that execution stops at max steps."""
        # Make evaluation always return low score to force repairs
        mock_skill_executor.execute.return_value = SkillResult(
            skill_name="evaluate_image",
            status=SkillStatus.COMPLETED,
            result={"overall_score": 0.3},
        )
        
        planner = ReActPlanner(
            skill_executor=mock_skill_executor,
            planning_model=GeminiPlanningModel(),
            max_steps=3,
        )
        
        plan = await planner.execute("Generate an image")
        
        assert len(plan.steps) <= 3
    
    @pytest.mark.asyncio
    async def test_execute_handles_skill_failure(self, mock_skill_executor):
        """Test handling of skill execution failure."""
        mock_skill_executor.execute.return_value = SkillResult(
            skill_name="text_to_image",
            status=SkillStatus.FAILED,
            error="API error",
        )
        
        planner = ReActPlanner(
            skill_executor=mock_skill_executor,
            planning_model=GeminiPlanningModel(),
            max_steps=5,
        )
        
        plan = await planner.execute("Generate an image")
        
        # Should have at least one step that recorded the failure
        assert any(s.observation and s.observation.get("error") for s in plan.steps)
    
    @pytest.mark.asyncio
    async def test_execute_stream_yields_events(self, mock_skill_executor):
        """Test streaming execution yields events."""
        mock_skill_executor.execute.return_value = SkillResult(
            skill_name="text_to_image",
            status=SkillStatus.COMPLETED,
            result={"image_url": "http://example.com/img.png", "overall_score": 0.9},
        )
        
        planner = ReActPlanner(
            skill_executor=mock_skill_executor,
            planning_model=GeminiPlanningModel(),
            max_steps=5,
        )
        
        events = []
        async for event in planner.execute_stream("Generate an image"):
            events.append(event)
        
        assert len(events) > 0
        event_types = [e["type"] for e in events]
        assert "plan_started" in event_types
    
    @pytest.mark.asyncio
    async def test_plan_to_dict(self, mock_skill_executor):
        """Test plan serialization to dict."""
        mock_skill_executor.execute.return_value = SkillResult(
            skill_name="text_to_image",
            status=SkillStatus.COMPLETED,
            result={"image_url": "http://example.com/img.png", "overall_score": 0.9},
        )
        
        planner = ReActPlanner(
            skill_executor=mock_skill_executor,
            planning_model=GeminiPlanningModel(),
            max_steps=5,
        )
        
        plan = await planner.execute("Generate an image")
        plan_dict = plan.to_dict()
        
        assert "steps" in plan_dict
        assert "status" in plan_dict
        assert plan_dict["user_message"] == "Generate an image"


class TestSSEManager:
    """Tests for SSEManager and SSEConnection."""
    
    def setup_method(self):
        """Reset SSE manager before each test."""
        reset_sse_manager()
    
    def test_create_connection(self):
        """Test creating a new connection."""
        manager = SSEManager()
        
        connection = manager.create_connection()
        
        assert connection is not None
        assert connection.connection_id is not None
        assert connection.is_connected
    
    def test_create_connection_with_id(self):
        """Test creating connection with specific ID."""
        manager = SSEManager()
        
        connection = manager.create_connection(connection_id="test-conn-1")
        
        assert connection.connection_id == "test-conn-1"
    
    def test_get_connection(self):
        """Test retrieving existing connection."""
        manager = SSEManager()
        connection = manager.create_connection(connection_id="test-conn")
        
        retrieved = manager.get_connection("test-conn")
        
        assert retrieved is connection
    
    def test_get_nonexistent_connection(self):
        """Test retrieving non-existent connection returns None."""
        manager = SSEManager()
        
        result = manager.get_connection("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_remove_connection(self):
        """Test removing a connection."""
        manager = SSEManager()
        manager.create_connection(connection_id="test-conn")
        
        await manager.remove_connection("test-conn")
        
        assert manager.get_connection("test-conn") is None
    
    def test_active_connections_count(self):
        """Test counting active connections."""
        manager = SSEManager()
        
        manager.create_connection()
        manager.create_connection()
        
        assert manager.active_connections == 2
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to all connections."""
        manager = SSEManager()
        conn1 = manager.create_connection()
        conn2 = manager.create_connection()
        
        await manager.broadcast(
            SSEEventType.HEARTBEAT,
            {"message": "test"},
        )
        
        # Both connections should have received the event
        assert not conn1._queue.empty()
        assert not conn2._queue.empty()
    
    @pytest.mark.asyncio
    async def test_broadcast_with_filter(self):
        """Test broadcasting with filter function."""
        manager = SSEManager()
        conn1 = manager.create_connection(connection_id="allowed")
        conn2 = manager.create_connection(connection_id="blocked")
        
        await manager.broadcast(
            SSEEventType.HEARTBEAT,
            {"message": "test"},
            filter_fn=lambda conn_id: conn_id == "allowed",
        )
        
        assert not conn1._queue.empty()
        assert conn2._queue.empty()


class TestSSEEvent:
    """Tests for SSEEvent."""
    
    def test_format_basic(self):
        """Test basic event formatting."""
        event = SSEEvent(
            event_type=SSEEventType.HEARTBEAT,
            data={"test": "value"},
        )
        
        formatted = event.format()
        
        assert "event: heartbeat" in formatted
        assert "data:" in formatted
        assert '"test"' in formatted
    
    def test_format_with_id(self):
        """Test event formatting with ID."""
        event = SSEEvent(
            event_type=SSEEventType.THOUGHT,
            data={"thought": "reasoning"},
            id="event-123",
        )
        
        formatted = event.format()
        
        assert "id: event-123" in formatted
        assert "event: thought" in formatted


class TestSSEConnection:
    """Tests for SSEConnection."""
    
    @pytest.mark.asyncio
    async def test_send_queues_event(self):
        """Test sending event queues it."""
        connection = SSEConnection(connection_id="test")
        
        await connection.send(SSEEventType.HEARTBEAT, {"test": True})
        
        assert not connection._queue.empty()
    
    @pytest.mark.asyncio
    async def test_close_sets_disconnected(self):
        """Test closing connection sets disconnected state."""
        connection = SSEConnection(connection_id="test")
        
        await connection.close()
        
        assert not connection.is_connected
    
    @pytest.mark.asyncio
    async def test_event_stream_yields_connected(self):
        """Test event stream starts with connected event."""
        connection = SSEConnection(connection_id="test", heartbeat_interval=0.1)
        
        # Get first event
        async for event in connection.event_stream():
            assert "connected" in event
            break
        
        await connection.close()


class TestStreamPlanExecution:
    """Tests for stream_plan_execution helper."""
    
    @pytest.mark.asyncio
    async def test_streams_plan_events(self):
        """Test streaming plan events to connection."""
        connection = SSEConnection(connection_id="test")
        
        async def mock_plan_stream():
            yield {"type": "plan_started", "data": {"message": "test"}}
            yield {"type": "thinking", "data": {"step": 1}}
            yield {"type": "finished", "data": {"result": "done"}}
        
        await stream_plan_execution(connection, mock_plan_stream())
        
        # Should have 3 events queued
        events = []
        while not connection._queue.empty():
            events.append(await connection._queue.get())
        
        assert len(events) == 3
