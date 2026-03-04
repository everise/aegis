"""
Provider abstraction layer.

A *provider* encapsulates a specific LLM / AI service (e.g. OpenRouter,
Mock, Vertex AI).  Both the ReAct planning loop and skill execution go
through the unified ``BaseProvider`` interface, so the rest of the
application is decoupled from any concrete service.

Adding a new provider
─────────────────────
1. Create ``backend/app/providers/<provider>.py``
   implementing ``BaseProvider``.
2. Add config entries under ``<provider>:`` in ``aegis.yaml``.
3. Register the new class in ``registry._register_defaults()``.
4. (Frontend) Add a provider icon in ``model-icons.tsx``.
"""

from app.providers.base import (
    ActionType,
    BaseProvider,
    PlanningStep,
    ProviderInfo,
)
from app.providers.registry import (
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry,
)

__all__ = [
    "ActionType",
    "BaseProvider",
    "PlanningStep",
    "ProviderInfo",
    "ProviderRegistry",
    "get_provider_registry",
    "reset_provider_registry",
]
