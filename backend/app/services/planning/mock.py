"""
Mock planning model provider.

.. deprecated::
    The canonical implementation has moved to
    ``app.providers.mock.MockProvider``.
    This module re-exports the provider class under the old name
    for backward compatibility.
"""

from app.providers.mock import MockProvider as MockPlanningModel  # noqa: F401

__all__ = ["MockPlanningModel"]
