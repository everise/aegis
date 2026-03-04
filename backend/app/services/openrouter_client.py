"""
OpenRouter API client — backward-compatibility re-export.

.. deprecated::
    The canonical module has moved to
    ``app.providers.openrouter_client``.
    Import from there instead.
"""

from app.providers.openrouter_client import (  # noqa: F401
    OpenRouterClient,
    OPENROUTER_API_BASE,
    image_url_to_base64,
    file_to_base64,
    ensure_base64,
)
