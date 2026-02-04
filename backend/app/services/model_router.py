"""
Multi-Model Router for load balancing and model selection.

Routes requests to appropriate models based on task type,
load, cost, and quality requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import random
import time
from collections import defaultdict


class ModelCapability(str, Enum):
    """Capabilities a model can have."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    UPSCALING = "upscaling"
    EVALUATION = "evaluation"
    CHAT = "chat"


class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    RANDOM = "random"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


@dataclass
class ModelEndpoint:
    """Represents a model endpoint."""
    endpoint_id: str
    name: str
    base_url: str
    capabilities: List[ModelCapability]
    
    # Performance characteristics
    avg_latency_ms: float = 1000.0
    quality_score: float = 0.8  # 0-1 scale
    cost_per_request: float = 0.01
    
    # Capacity
    max_concurrent: int = 10
    current_load: int = 0
    
    # Status
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    success_count: int = 0
    
    # Rate limiting
    requests_per_minute: int = 60
    request_count_this_minute: int = 0
    minute_start: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def load_factor(self) -> float:
        """Current load as a fraction of capacity."""
        return self.current_load / self.max_concurrent if self.max_concurrent > 0 else 1.0
    
    @property
    def success_rate(self) -> float:
        """Success rate of requests."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 1.0
    
    def can_handle(self, capability: ModelCapability) -> bool:
        """Check if endpoint supports the capability."""
        return capability in self.capabilities and self.is_healthy
    
    def is_rate_limited(self) -> bool:
        """Check if endpoint is rate limited."""
        now = datetime.utcnow()
        # Reset counter if new minute
        if (now - self.minute_start).seconds >= 60:
            self.request_count_this_minute = 0
            self.minute_start = now
        
        return self.request_count_this_minute >= self.requests_per_minute
    
    def record_request(self, success: bool, latency_ms: float) -> None:
        """Record request outcome for statistics."""
        if success:
            self.success_count += 1
            # Update average latency with exponential moving average
            alpha = 0.1
            self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
        else:
            self.error_count += 1
        
        self.request_count_this_minute += 1


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    endpoint: ModelEndpoint
    strategy_used: RoutingStrategy
    reason: str
    alternatives: List[ModelEndpoint] = field(default_factory=list)


class BaseRouter(ABC):
    """Abstract base class for routing strategies."""
    
    @abstractmethod
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select an endpoint for the request."""
        pass


class RoundRobinRouter(BaseRouter):
    """Round-robin routing strategy."""
    
    def __init__(self):
        self._counters: Dict[ModelCapability, int] = defaultdict(int)
    
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select endpoint in round-robin fashion."""
        available = [e for e in endpoints if e.can_handle(capability) and not e.is_rate_limited()]
        
        if not available:
            return None
        
        idx = self._counters[capability] % len(available)
        self._counters[capability] += 1
        
        return available[idx]


class LeastLoadRouter(BaseRouter):
    """Route to least loaded endpoint."""
    
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select endpoint with lowest load."""
        available = [e for e in endpoints if e.can_handle(capability) and not e.is_rate_limited()]
        
        if not available:
            return None
        
        return min(available, key=lambda e: e.load_factor)


class CostOptimizedRouter(BaseRouter):
    """Route to minimize cost."""
    
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select cheapest available endpoint."""
        available = [e for e in endpoints if e.can_handle(capability) and not e.is_rate_limited()]
        
        if not available:
            return None
        
        return min(available, key=lambda e: e.cost_per_request)


class QualityOptimizedRouter(BaseRouter):
    """Route to maximize quality."""
    
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select highest quality endpoint."""
        available = [e for e in endpoints if e.can_handle(capability) and not e.is_rate_limited()]
        
        if not available:
            return None
        
        return max(available, key=lambda e: e.quality_score)


class LatencyOptimizedRouter(BaseRouter):
    """Route to minimize latency."""
    
    def select_endpoint(
        self,
        endpoints: List[ModelEndpoint],
        capability: ModelCapability,
    ) -> Optional[ModelEndpoint]:
        """Select lowest latency endpoint."""
        available = [e for e in endpoints if e.can_handle(capability) and not e.is_rate_limited()]
        
        if not available:
            return None
        
        return min(available, key=lambda e: e.avg_latency_ms)


class ModelRouter:
    """
    Main model router that manages endpoints and routing decisions.
    """
    
    ROUTERS = {
        RoutingStrategy.ROUND_ROBIN: RoundRobinRouter,
        RoutingStrategy.LEAST_LOAD: LeastLoadRouter,
        RoutingStrategy.COST_OPTIMIZED: CostOptimizedRouter,
        RoutingStrategy.QUALITY_OPTIMIZED: QualityOptimizedRouter,
        RoutingStrategy.LATENCY_OPTIMIZED: LatencyOptimizedRouter,
    }
    
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.LEAST_LOAD,
        health_check_interval: float = 60.0,
    ):
        self.default_strategy = default_strategy
        self.health_check_interval = health_check_interval
        
        self._endpoints: Dict[str, ModelEndpoint] = {}
        self._routers: Dict[RoutingStrategy, BaseRouter] = {
            strategy: router_class()
            for strategy, router_class in self.ROUTERS.items()
        }
        
        # Statistics
        self._routing_history: List[Dict[str, Any]] = []
    
    def register_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Register a model endpoint."""
        self._endpoints[endpoint.endpoint_id] = endpoint
    
    def unregister_endpoint(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
    
    def get_endpoint(self, endpoint_id: str) -> Optional[ModelEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)
    
    def list_endpoints(
        self,
        capability: Optional[ModelCapability] = None,
    ) -> List[ModelEndpoint]:
        """List endpoints, optionally filtered by capability."""
        endpoints = list(self._endpoints.values())
        
        if capability:
            endpoints = [e for e in endpoints if capability in e.capabilities]
        
        return endpoints
    
    def route(
        self,
        capability: ModelCapability,
        strategy: Optional[RoutingStrategy] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a request to an appropriate endpoint.
        
        Args:
            capability: Required model capability
            strategy: Routing strategy (uses default if not specified)
            preferences: Additional routing preferences
            
        Returns:
            RoutingDecision with selected endpoint
        """
        strategy = strategy or self.default_strategy
        router = self._routers.get(strategy) or self._routers[self.default_strategy]
        
        # Get available endpoints
        available = [
            e for e in self._endpoints.values()
            if e.can_handle(capability)
        ]
        
        if not available:
            raise ValueError(f"No endpoints available for capability: {capability}")
        
        # Select endpoint
        selected = router.select_endpoint(available, capability)
        
        if selected is None:
            # All endpoints rate limited, use fallback
            selected = random.choice(available)
        
        # Record decision
        alternatives = [e for e in available if e.endpoint_id != selected.endpoint_id]
        
        decision = RoutingDecision(
            endpoint=selected,
            strategy_used=strategy,
            reason=f"Selected via {strategy.value}",
            alternatives=alternatives[:3],  # Top 3 alternatives
        )
        
        # Log for analytics
        self._routing_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "capability": capability.value,
            "strategy": strategy.value,
            "selected": selected.endpoint_id,
            "load_factor": selected.load_factor,
        })
        
        return decision
    
    def acquire_endpoint(
        self,
        capability: ModelCapability,
        strategy: Optional[RoutingStrategy] = None,
    ) -> ModelEndpoint:
        """
        Acquire an endpoint for use (increments load).
        
        Call release_endpoint when done.
        """
        decision = self.route(capability, strategy)
        endpoint = decision.endpoint
        endpoint.current_load += 1
        return endpoint
    
    def release_endpoint(
        self,
        endpoint: ModelEndpoint,
        success: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """Release an endpoint after use."""
        endpoint.current_load = max(0, endpoint.current_load - 1)
        endpoint.record_request(success, latency_ms)
    
    def mark_unhealthy(self, endpoint_id: str) -> None:
        """Mark an endpoint as unhealthy."""
        endpoint = self._endpoints.get(endpoint_id)
        if endpoint:
            endpoint.is_healthy = False
            endpoint.last_health_check = datetime.utcnow()
    
    def mark_healthy(self, endpoint_id: str) -> None:
        """Mark an endpoint as healthy."""
        endpoint = self._endpoints.get(endpoint_id)
        if endpoint:
            endpoint.is_healthy = True
            endpoint.last_health_check = datetime.utcnow()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            "total_endpoints": len(self._endpoints),
            "healthy_endpoints": sum(1 for e in self._endpoints.values() if e.is_healthy),
            "total_routes": len(self._routing_history),
            "endpoints": {},
        }
        
        for endpoint in self._endpoints.values():
            stats["endpoints"][endpoint.endpoint_id] = {
                "name": endpoint.name,
                "healthy": endpoint.is_healthy,
                "load_factor": endpoint.load_factor,
                "success_rate": endpoint.success_rate,
                "avg_latency_ms": endpoint.avg_latency_ms,
            }
        
        return stats


# Pre-configured router for image generation
def create_image_generation_router() -> ModelRouter:
    """Create a pre-configured router for image generation tasks."""
    router = ModelRouter(default_strategy=RoutingStrategy.LEAST_LOAD)
    
    # Register mock endpoints (in production, these would be real endpoints)
    endpoints = [
        ModelEndpoint(
            endpoint_id="sdxl-primary",
            name="SDXL Primary",
            base_url="http://sdxl-primary:8000",
            capabilities=[ModelCapability.TEXT_TO_IMAGE, ModelCapability.IMAGE_TO_IMAGE],
            quality_score=0.9,
            cost_per_request=0.02,
            avg_latency_ms=3000,
        ),
        ModelEndpoint(
            endpoint_id="sdxl-backup",
            name="SDXL Backup",
            base_url="http://sdxl-backup:8000",
            capabilities=[ModelCapability.TEXT_TO_IMAGE, ModelCapability.IMAGE_TO_IMAGE],
            quality_score=0.9,
            cost_per_request=0.02,
            avg_latency_ms=3500,
        ),
        ModelEndpoint(
            endpoint_id="inpaint-service",
            name="Inpainting Service",
            base_url="http://inpaint:8000",
            capabilities=[ModelCapability.INPAINTING],
            quality_score=0.85,
            cost_per_request=0.03,
            avg_latency_ms=4000,
        ),
        ModelEndpoint(
            endpoint_id="evaluator",
            name="Image Evaluator",
            base_url="http://evaluator:8000",
            capabilities=[ModelCapability.EVALUATION],
            quality_score=0.95,
            cost_per_request=0.005,
            avg_latency_ms=500,
        ),
    ]
    
    for endpoint in endpoints:
        router.register_endpoint(endpoint)
    
    return router
