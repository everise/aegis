"""
OpenRouter planning model implementation.

.. deprecated::
    The canonical implementation has moved to
    ``app.providers.openrouter.OpenRouterProvider``.
    This module re-exports the provider class under the old name
    for backward compatibility.
"""

from app.providers.openrouter import OpenRouterProvider as OpenRouterPlanningModel  # noqa: F401

__all__ = ["OpenRouterPlanningModel"]
