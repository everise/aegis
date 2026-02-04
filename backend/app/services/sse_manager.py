"""
Server-Sent Events (SSE) manager for streaming responses.

Handles real-time updates to the frontend during agent execution.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class SSEEventType(str, Enum):
    """Types of SSE events."""
    # Connection events
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"
    DISCONNECTED = "disconnected"
    
    # Planning events
    PLAN_STARTED = "plan_started"
    THINKING = "thinking"
    THOUGHT = "thought"
    EXECUTING = "executing"
    OBSERVATION = "observation"
    FINISHED = "finished"
    
    # Error events
    ERROR = "error"
    MAX_STEPS_REACHED = "max_steps_reached"
    
    # Skill events
    SKILL_SUBMITTED = "skill_submitted"
    SKILL_POLLING = "skill_polling"
    SKILL_COMPLETED = "skill_completed"
    SKILL_FAILED = "skill_failed"


@dataclass
class SSEEvent:
    """Represents a single SSE event."""
    event_type: SSEEventType
    data: Dict[str, Any]
    id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def format(self) -> str:
        """Format event as SSE message string."""
        lines = []
        
        if self.id:
            lines.append(f"id: {self.id}")
        
        lines.append(f"event: {self.event_type.value}")
        
        # Add timestamp to data
        data_with_timestamp = {
            **self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        lines.append(f"data: {json.dumps(data_with_timestamp)}")
        
        # Empty line marks end of event
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)


class SSEConnection:
    """
    Manages a single SSE connection to a client.
    
    Handles event queuing, heartbeats, and connection lifecycle.
    """
    
    def __init__(
        self,
        connection_id: str,
        heartbeat_interval: float = 15.0,
    ):
        self.connection_id = connection_id
        self.heartbeat_interval = heartbeat_interval
        self._queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
        self._connected = True
        self._event_counter = 0
    
    def _next_event_id(self) -> str:
        """Generate next event ID."""
        self._event_counter += 1
        return f"{self.connection_id}-{self._event_counter}"
    
    async def send(self, event_type: SSEEventType, data: Dict[str, Any]) -> None:
        """Queue an event to send to the client."""
        if not self._connected:
            return
        
        event = SSEEvent(
            event_type=event_type,
            data=data,
            id=self._next_event_id(),
        )
        await self._queue.put(event)
    
    async def send_event(self, event: SSEEvent) -> None:
        """Queue a pre-built event."""
        if not self._connected:
            return
        
        if not event.id:
            event.id = self._next_event_id()
        await self._queue.put(event)
    
    async def close(self) -> None:
        """Close the connection."""
        self._connected = False
        await self.send(SSEEventType.DISCONNECTED, {"reason": "closed"})
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is still active."""
        return self._connected
    
    async def event_stream(self) -> AsyncIterator[str]:
        """
        Generate SSE event stream.
        
        Yields formatted SSE messages including heartbeats.
        """
        # Send connected event
        connected_event = SSEEvent(
            event_type=SSEEventType.CONNECTED,
            data={"connection_id": self.connection_id},
            id=self._next_event_id(),
        )
        yield connected_event.format()
        
        while self._connected:
            try:
                # Wait for event with timeout for heartbeat
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.heartbeat_interval,
                )
                yield event.format()
                
                # Check for disconnect event
                if event.event_type == SSEEventType.DISCONNECTED:
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = SSEEvent(
                    event_type=SSEEventType.HEARTBEAT,
                    data={"connection_id": self.connection_id},
                    id=self._next_event_id(),
                )
                yield heartbeat.format()


class SSEManager:
    """
    Manages multiple SSE connections.
    
    Provides connection pooling and broadcast capabilities.
    """
    
    def __init__(self):
        self._connections: Dict[str, SSEConnection] = {}
        self._connection_counter = 0
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        self._connection_counter += 1
        return f"conn-{self._connection_counter}-{datetime.utcnow().timestamp()}"
    
    def create_connection(
        self,
        connection_id: Optional[str] = None,
        heartbeat_interval: float = 15.0,
    ) -> SSEConnection:
        """Create a new SSE connection."""
        if connection_id is None:
            connection_id = self._generate_connection_id()
        
        connection = SSEConnection(
            connection_id=connection_id,
            heartbeat_interval=heartbeat_interval,
        )
        self._connections[connection_id] = connection
        
        return connection
    
    def get_connection(self, connection_id: str) -> Optional[SSEConnection]:
        """Get an existing connection by ID."""
        return self._connections.get(connection_id)
    
    async def remove_connection(self, connection_id: str) -> None:
        """Remove and close a connection."""
        if connection_id in self._connections:
            connection = self._connections.pop(connection_id)
            await connection.close()
    
    async def broadcast(
        self,
        event_type: SSEEventType,
        data: Dict[str, Any],
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """
        Broadcast an event to multiple connections.
        
        Args:
            event_type: Type of event to broadcast.
            data: Event data.
            filter_fn: Optional function to filter which connections receive the event.
        """
        for conn_id, connection in list(self._connections.items()):
            if not connection.is_connected:
                del self._connections[conn_id]
                continue
            
            if filter_fn and not filter_fn(conn_id):
                continue
            
            await connection.send(event_type, data)
    
    @property
    def active_connections(self) -> int:
        """Get count of active connections."""
        # Clean up disconnected connections
        self._connections = {
            k: v for k, v in self._connections.items()
            if v.is_connected
        }
        return len(self._connections)
    
    async def close_all(self) -> None:
        """Close all connections."""
        for connection in list(self._connections.values()):
            await connection.close()
        self._connections.clear()


# Global SSE manager instance
_sse_manager: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """Get the global SSE manager instance."""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEManager()
    return _sse_manager


def reset_sse_manager() -> None:
    """Reset the global SSE manager. For testing only."""
    global _sse_manager
    _sse_manager = None


async def stream_plan_execution(
    connection: SSEConnection,
    plan_stream: AsyncIterator[Dict[str, Any]],
) -> None:
    """
    Stream plan execution events to an SSE connection.
    
    Converts planner stream events to SSE events.
    """
    event_type_mapping = {
        "plan_started": SSEEventType.PLAN_STARTED,
        "thinking": SSEEventType.THINKING,
        "thought": SSEEventType.THOUGHT,
        "executing": SSEEventType.EXECUTING,
        "observation": SSEEventType.OBSERVATION,
        "finished": SSEEventType.FINISHED,
        "error": SSEEventType.ERROR,
        "max_steps_reached": SSEEventType.MAX_STEPS_REACHED,
    }
    
    async for event_data in plan_stream:
        event_type_str = event_data.get("type", "")
        sse_event_type = event_type_mapping.get(
            event_type_str,
            SSEEventType.OBSERVATION,  # Default
        )
        
        await connection.send(sse_event_type, event_data.get("data", {}))
