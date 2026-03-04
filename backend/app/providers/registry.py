"""
Provider registry — singleton that manages available providers.

Mirrors the pattern previously used by ``PlanningModelRegistry`` but
at the *provider* level, so the rest of the application only deals
with the abstract ``BaseProvider`` interface.
"""

import logging
from typing import Dict, List, Optional

from app.providers.base import BaseProvider, ProviderInfo

logger = logging.getLogger("aegis.providers.registry")


class ProviderRegistry:
    """
    Singleton-style registry for provider implementations.

    All known providers are registered at startup.  The *active*
    provider is the one used by the planner for new conversations.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        self._active_id: Optional[str] = None

    # ── Registration ──────────────────────────────────────────────

    def register(self, provider: BaseProvider) -> None:
        """Register a provider."""
        pid = provider.info().id
        self._providers[pid] = provider
        # First registered provider becomes the default
        if self._active_id is None:
            self._active_id = pid
        logger.info("Registered provider: %s", pid)

    # ── Queries ───────────────────────────────────────────────────

    def list_providers(self) -> List[ProviderInfo]:
        """Return metadata for every registered provider."""
        return [p.info() for p in self._providers.values()]

    def get_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """Return a provider by id, or *None* if not found."""
        return self._providers.get(provider_id)

    def get_active_provider(self) -> BaseProvider:
        """Return the currently-active provider."""
        if self._active_id is None or self._active_id not in self._providers:
            raise RuntimeError("No active provider configured")
        return self._providers[self._active_id]

    def get_active_provider_id(self) -> Optional[str]:
        """Return the id of the currently-active provider."""
        return self._active_id

    # ── Mutation ──────────────────────────────────────────────────

    def set_active_provider(self, provider_id: str) -> BaseProvider:
        """
        Switch the active provider.

        Raises ``KeyError`` if the provider_id is not registered.
        Returns the newly-active provider instance.
        """
        if provider_id not in self._providers:
            raise KeyError(f"Unknown provider: {provider_id}")
        self._active_id = provider_id
        logger.info("Active provider set to: %s", provider_id)
        return self._providers[provider_id]

    # ── Backward-compat aliases  (planning registry API) ──────────
    # These allow existing code that used PlanningModelRegistry to
    # work without changes during the migration period.

    def list_models(self) -> List[ProviderInfo]:
        return self.list_providers()

    def get_model(self, model_id: str) -> Optional[BaseProvider]:
        return self.get_provider(model_id)

    def get_active_model(self) -> BaseProvider:
        return self.get_active_provider()

    def get_active_model_id(self) -> Optional[str]:
        return self.get_active_provider_id()

    def set_active_model(self, model_id: str) -> BaseProvider:
        return self.set_active_provider(model_id)


# ── Module-level singleton ────────────────────────────────────────

_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """
    Return the global provider registry.

    On first call the registry is created and all built-in providers
    are registered automatically.
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        _register_defaults(_registry)
    return _registry


def _register_defaults(registry: ProviderRegistry) -> None:
    """Register the built-in providers."""
    from app.providers.openrouter import OpenRouterProvider
    from app.providers.mock import MockProvider

    # OpenRouter is the default provider (registered first)
    registry.register(OpenRouterProvider())
    registry.register(MockProvider())


def reset_provider_registry() -> None:
    """Reset the registry (useful for testing)."""
    global _registry
    _registry = None
