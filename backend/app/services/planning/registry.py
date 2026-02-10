"""
Planning model registry.

Central registry that manages available planning models and the
currently-active model.  Provides factory methods so the rest of
the application never instantiates planning models directly.
"""

from typing import Dict, List, Optional

from app.services.planning.base import BasePlanningModel, PlanningModelInfo


class PlanningModelRegistry:
    """
    Singleton-style registry for planning model implementations.

    All known models are registered at startup. The *active* model is
    the one used by the planner for new conversations.
    """

    def __init__(self) -> None:
        self._models: Dict[str, BasePlanningModel] = {}
        self._active_model_id: Optional[str] = None

    # ── Registration ──────────────────────────────────────────────

    def register(self, model: BasePlanningModel) -> None:
        """Register a planning model."""
        model_id = model.info().id
        self._models[model_id] = model
        # First registered model becomes the default active model
        if self._active_model_id is None:
            self._active_model_id = model_id

    # ── Queries ───────────────────────────────────────────────────

    def list_models(self) -> List[PlanningModelInfo]:
        """Return metadata for every registered model."""
        return [m.info() for m in self._models.values()]

    def get_model(self, model_id: str) -> Optional[BasePlanningModel]:
        """Return a model by id, or *None* if not found."""
        return self._models.get(model_id)

    def get_active_model(self) -> BasePlanningModel:
        """Return the currently-active planning model."""
        if self._active_model_id is None or self._active_model_id not in self._models:
            raise RuntimeError("No active planning model configured")
        return self._models[self._active_model_id]

    def get_active_model_id(self) -> Optional[str]:
        """Return the id of the currently-active model."""
        return self._active_model_id

    # ── Mutation ──────────────────────────────────────────────────

    def set_active_model(self, model_id: str) -> BasePlanningModel:
        """
        Switch the active planning model.

        Raises ``KeyError`` if the model_id is not registered.
        Returns the newly-active model instance.
        """
        if model_id not in self._models:
            raise KeyError(f"Unknown planning model: {model_id}")
        self._active_model_id = model_id
        return self._models[model_id]


# ── Module-level singleton ────────────────────────────────────────

_registry: Optional[PlanningModelRegistry] = None


def get_planning_registry() -> PlanningModelRegistry:
    """
    Return the global planning model registry.

    On first call the registry is created and all built-in models
    are registered automatically.
    """
    global _registry
    if _registry is None:
        from app.services.planning.gemini import GeminiPlanningModel
        from app.services.planning.kimi import KimiPlanningModel
        from app.services.planning.qwen_vl import QwenVLPlanningModel

        _registry = PlanningModelRegistry()
        _registry.register(GeminiPlanningModel())
        _registry.register(KimiPlanningModel())
        _registry.register(QwenVLPlanningModel())

    return _registry


def reset_planning_registry() -> None:
    """Reset the registry (useful for testing)."""
    global _registry
    _registry = None
