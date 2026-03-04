"""
Planning model registry.

.. deprecated::
    Use ``app.providers.registry`` instead.  This module provides
    backward-compatible aliases that delegate to the provider registry.
"""

from typing import Dict, List, Optional

from app.providers.registry import (
    ProviderRegistry as PlanningModelRegistry,
    get_provider_registry as get_planning_registry,
    reset_provider_registry as reset_planning_registry,
)

# Re-export so old ``from app.services.planning.registry import ...``
# statements keep working.
__all__ = [
    "PlanningModelRegistry",
    "get_planning_registry",
    "reset_planning_registry",
]
