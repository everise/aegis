"""
Unit tests for ReAct planner and related components.

Tests planning models, ReAct planner, and SSE manager.
"""

import pytest
import asyncio
import json

from app.providers.base import BaseProvider, PlanningStep, ActionType, ProviderInfo
from app.providers.mock import MockProvider
from app.providers.registry import ProviderRegistry, get_provider_registry, reset_provider_registry
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

# Backward-compat aliases used in some tests
PlanningModelInfo = ProviderInfo
MockPlanningModel = MockProvider


class TestPlanningModelInterface:
    """Tests for the planning model abstract interface and implementations."""

    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    def test_model_info(self, ProviderClass):
        """Each provider returns valid info."""
        provider = ProviderClass()
        info = provider.info()
        assert isinstance(info, ProviderInfo)
        assert info.id
        assert info.name
        assert info.provider

    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    def test_reset(self, ProviderClass):
        """Reset clears internal state."""
        provider = ProviderClass()
        provider._step = 5
        provider._repairs = 2
        provider.reset()
        assert provider._step == 0
        assert provider._repairs == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    async def test_first_step_generates(self, ProviderClass):
        """First step should always be a generate action."""
        provider = ProviderClass()
        step = await provider.get_next_step("Create a sunset image")
        assert step.action == ActionType.GENERATE
        assert "text_to_image" in str(step.action_input)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    async def test_second_step_evaluates(self, ProviderClass):
        """Second step should evaluate the generated image."""
        provider = ProviderClass()
        await provider.get_next_step("Create an image")
        observation = {
            "result": {"image_url": "http://example.com/img.png"},
            "status": "completed",
        }
        step = await provider.get_next_step("Create an image", observation)
        assert step.action == ActionType.EVALUATE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    async def test_high_quality_finishes(self, ProviderClass):
        """High quality score should lead to a finish action."""
        provider = ProviderClass()
        provider.quality_threshold = 0.7
        await provider.get_next_step("Create an image")
        await provider.get_next_step("Create an image", {"result": {"image_url": "http://example.com/img.png"}})
        step = await provider.get_next_step("Create an image", {"result": {"overall_score": 0.9}})
        assert step.action == ActionType.FINISH
        assert step.action_input.get("result") == "success"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    async def test_low_quality_triggers_repair(self, ProviderClass):
        """Low quality score should trigger repair."""
        provider = ProviderClass()
        provider.quality_threshold = 0.7
        await provider.get_next_step("Create an image")
        await provider.get_next_step("Create an image", {"result": {"image_url": "http://example.com/img.png"}})
        step = await provider.get_next_step("Create an image", {"result": {"overall_score": 0.5}})
        assert step.action == ActionType.REPAIR

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ProviderClass", [MockProvider])
    async def test_first_step_produces_valid_step(self, ProviderClass):
        """Provider returns a valid PlanningStep on first call."""
        provider = ProviderClass()
        step = await provider.get_next_step("Generate an image of a cat")
        assert isinstance(step, PlanningStep)
        assert step.thought
        assert step.action in ActionType

    def test_format_step_as_dict(self):
        """PlanningStep to dict conversion."""
        provider = MockProvider()
        step = PlanningStep(
            thought="Test thought",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image"},
        )
        result = provider.format_step_as_dict(step)
        assert result["thought"] == "Test thought"
        assert result["action"] == "generate"


class TestPlanningModelRegistry:
    """Tests for provider registry."""

    def test_register_and_list(self):
        """Registered providers appear in list."""
        reg = ProviderRegistry()
        reg.register(MockProvider())
        assert len(reg.list_providers()) == 1

    def test_first_registered_is_default(self):
        """First registered provider becomes the default active one."""
        reg = ProviderRegistry()
        from app.providers.openrouter import OpenRouterProvider
        reg.register(OpenRouterProvider())
        reg.register(MockProvider())
        assert reg.get_active_provider_id() == "openrouter"

    def test_set_active_model(self):
        """set_active_provider switches the active provider."""
        reg = ProviderRegistry()
        from app.providers.openrouter import OpenRouterProvider
        reg.register(OpenRouterProvider())
        reg.register(MockProvider())
        reg.set_active_provider("mock")
        assert reg.get_active_provider_id() == "mock"

    def test_set_unknown_model_raises(self):
        """Setting an unknown provider id raises KeyError."""
        reg = ProviderRegistry()
        reg.register(MockProvider())
        with pytest.raises(KeyError):
            reg.set_active_provider("nonexistent")

    def test_global_registry(self):
        """get_provider_registry returns a populated singleton."""
        reset_provider_registry()
        reg = get_provider_registry()
        ids = [m.id for m in reg.list_providers()]
        assert "openrouter" in ids
        assert "mock" in ids


class TestReActPlanner:
    """Tests for ReActPlanner."""
    
    @pytest.mark.asyncio
    async def test_execute_basic_flow(self):
        """Test basic execution flow with MockProvider."""
        planner = ReActPlanner(
            provider=MockProvider(),
            max_steps=10,
        )
        
        plan = await planner.execute("Generate a beautiful sunset")
        
        assert plan.status == PlanStatus.COMPLETED
        assert len(plan.steps) >= 2  # At least generate and evaluate
        assert plan.final_result is not None
    
    @pytest.mark.asyncio
    async def test_execute_respects_max_steps(self):
        """Test that execution stops at max steps."""
        planner = ReActPlanner(
            provider=MockProvider(),
            max_steps=3,
        )
        
        plan = await planner.execute("Generate an image")
        
        assert len(plan.steps) <= 3
    
    @pytest.mark.asyncio
    async def test_execute_handles_skill_failure(self):
        """Test handling of skill execution failure."""
        # Create a mock provider where execute_skill always fails
        provider = MockProvider()
        original_execute = provider.execute_skill

        async def failing_execute(skill_name, params):
            return {
                "skill_name": skill_name,
                "status": "failed",
                "result": None,
                "error": "API error",
            }

        provider.execute_skill = failing_execute  # type: ignore[assignment]

        planner = ReActPlanner(
            provider=provider,
            max_steps=5,
        )
        
        plan = await planner.execute("Generate an image")
        
        # Should have at least one step that recorded the failure
        assert any(s.observation and s.observation.get("error") for s in plan.steps)
    
    @pytest.mark.asyncio
    async def test_execute_stream_yields_events(self):
        """Test streaming execution yields events."""
        planner = ReActPlanner(
            provider=MockProvider(),
            max_steps=5,
        )
        
        events = []
        async for event in planner.execute_stream("Generate an image"):
            events.append(event)
        
        assert len(events) > 0
        event_types = [e["type"] for e in events]
        assert "plan_started" in event_types
    
    @pytest.mark.asyncio
    async def test_plan_to_dict(self):
        """Test plan serialization to dict."""
        planner = ReActPlanner(
            provider=MockProvider(),
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
