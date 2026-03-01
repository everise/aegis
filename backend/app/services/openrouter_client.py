"""
OpenRouter API client  (system-level shared infrastructure).

Provides a reusable async HTTP wrapper around OpenRouter's
``/api/v1/chat/completions`` endpoint — used by both the planning
layer (``backend/app/services/planning/openrouter.py``) **and** the
skill implementations that live under ``skills/*/scripts/openrouter.py``.

Responsibilities
────────────────
* HTTP transport  (non-streaming + SSE streaming)
* Image ↔ base64  conversion helpers
* Structured logging for every outbound request / response
"""

import asyncio
import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.config import get_settings

# Transient errors that warrant an automatic retry
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.5  # seconds; actual delay = base * 2^attempt

logger = logging.getLogger("aegis.openrouter")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


# ── Image helpers ─────────────────────────────────────────────────


async def image_url_to_base64(
    image_url: str,
    http_client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Download an image from *image_url* and return a ``data:`` URI."""
    owns_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30.0)
        owns_client = True
    try:
        logger.debug("Downloading image for base64 conversion: %s", image_url)
        resp = await http_client.get(image_url)
        resp.raise_for_status()
        content_type = (
            resp.headers.get("content-type", "image/png").split(";")[0].strip()
        )
        b64 = base64.b64encode(resp.content).decode("utf-8")
        logger.debug(
            "Image converted: %s → %s (%.1f KB)",
            image_url, content_type, len(resp.content) / 1024,
        )
        return f"data:{content_type};base64,{b64}"
    finally:
        if owns_client:
            await http_client.aclose()


def file_to_base64(file_path: str) -> str:
    """Read a local file and return a ``data:`` URI."""
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    logger.debug("Local file encoded: %s (%.1f KB)", file_path, len(raw) / 1024)
    return f"data:{mime_type};base64,{b64}"


async def ensure_base64(image_ref: str) -> str:
    """Normalise any image reference to a ``data:`` URI.

    Accepts a data-URI, an HTTP(S) URL, or a local file path.
    URL paths like ``/data/images/...`` are resolved relative to CWD.
    """
    if image_ref.startswith("data:"):
        return image_ref
    if image_ref.startswith(("http://", "https://")):
        return await image_url_to_base64(image_ref)
    # Strip leading slash from URL paths so they resolve relative to CWD
    file_path = image_ref.lstrip("/")
    return file_to_base64(file_path)


# ── Client ────────────────────────────────────────────────────────


class OpenRouterClient:
    """Async HTTP client wrapping OpenRouter chat-completions."""

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            logger.warning("OpenRouter API key is empty — requests will fail")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=OPENROUTER_API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://aegis.app",
                    "X-OpenRouter-Title": "Aegis",
                },
                timeout=120.0,
            )
        return self._client

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _sanitise_messages_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip base64 payloads from messages so logs stay readable."""
        sanitised = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if (
                        part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), dict)
                    ):
                        url = part["image_url"].get("url", "")
                        if url.startswith("data:"):
                            parts.append({"type": "image_url", "image_url": {"url": f"{url[:40]}...(base64)"}})
                        else:
                            parts.append(part)
                    else:
                        parts.append(part)
                sanitised.append({**msg, "content": parts})
            else:
                sanitised.append(msg)
        return sanitised

    # ── Non-streaming completion ──────────────────────────────────

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        modalities: Optional[List[str]] = None,
        image_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a non-streaming chat completion.

        Returns the full JSON body from OpenRouter.
        """
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if modalities is not None:
            payload["modalities"] = modalities
        if image_config is not None:
            payload["image_config"] = image_config

        # ── Log request ──
        log_payload = {**payload, "messages": self._sanitise_messages_for_log(messages)}
        logger.info(
            "[OpenRouter] POST /chat/completions  model=%s  temperature=%.2f  max_tokens=%s  modalities=%s",
            model, temperature, max_tokens, modalities,
        )
        logger.debug("[OpenRouter] Request payload:\n%s", json.dumps(log_payload, ensure_ascii=False, indent=2))

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is not configured. "
                "Set 'openrouter.api_key' in aegis.yaml or the "
                "OPENROUTER_API_KEY environment variable."
            )

        t0 = time.monotonic()
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.post("/chat/completions", json=payload)
                break
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                elapsed = time.monotonic() - t0
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "[OpenRouter] Transient error on attempt %d/%d after %.2fs "
                        "(%s: %s) — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, elapsed,
                        type(exc).__name__, exc, delay,
                    )
                    # Dispose the broken connection pool and create a fresh client
                    try:
                        await self._client.aclose()  # type: ignore[union-attr]
                    except Exception:
                        pass
                    self._client = None
                    client = await self._get_client()
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[OpenRouter] Request FAILED after %d attempts (%.2fs): %s",
                        _MAX_RETRIES, elapsed, exc,
                    )
                    raise
            except httpx.HTTPError as exc:
                elapsed = time.monotonic() - t0
                logger.error(
                    "[OpenRouter] Request FAILED after %.2fs: %s", elapsed, exc,
                )
                raise
        elapsed = time.monotonic() - t0

        # ── Log response ──
        logger.info(
            "[OpenRouter] Response %d  elapsed=%.2fs  content-length=%s",
            resp.status_code, elapsed, resp.headers.get("content-length", "?"),
        )
        if resp.status_code >= 400:
            logger.error(
                "[OpenRouter] Error response body:\n%s", resp.text[:2000],
            )
        resp.raise_for_status()

        body = resp.json()
        # Log a trimmed version of the response (no base64 images)
        try:
            choices = body.get("choices", [])
            for c in choices:
                msg = c.get("message", {})
                if msg.get("content"):
                    preview = msg["content"][:300]
                    logger.debug("[OpenRouter] Response content preview: %s", preview)
                if msg.get("images"):
                    logger.debug("[OpenRouter] Response contains %d image(s)", len(msg["images"]))
            usage = body.get("usage", {})
            if usage:
                logger.info(
                    "[OpenRouter] Token usage: prompt=%s  completion=%s  total=%s",
                    usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"),
                )
        except Exception:
            pass

        return body

    # ── Streaming completion ──────────────────────────────────────

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming chat completion — yields parsed SSE chunks.

        The final chunk before ``[DONE]`` often contains a ``usage``
        dict.  It is yielded like any other chunk so callers can
        inspect it (check ``chunk.get("usage")``).
        """
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.info(
            "[OpenRouter] POST /chat/completions (stream)  model=%s", model,
        )

        t0 = time.monotonic()
        for attempt in range(_MAX_RETRIES):
            try:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        logger.error(
                            "[OpenRouter] Stream error %d:\n%s",
                            resp.status_code, body.decode("utf-8", errors="replace")[:2000],
                        )
                    resp.raise_for_status()
                    chunk_count = 0
                    last_usage = None
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk_count += 1
                            parsed = json.loads(data)
                            # Capture usage from any chunk that has it
                            if parsed.get("usage"):
                                last_usage = parsed["usage"]
                            yield parsed
                        except json.JSONDecodeError:
                            logger.warning("[OpenRouter] Skipped unparseable SSE chunk: %s", data[:200])
                            continue
                    elapsed = time.monotonic() - t0
                    if last_usage:
                        logger.info(
                            "[OpenRouter] Stream token usage: prompt=%s  completion=%s  total=%s",
                            last_usage.get("prompt_tokens"),
                            last_usage.get("completion_tokens"),
                            last_usage.get("total_tokens"),
                        )
                    logger.info(
                        "[OpenRouter] Stream finished: %d chunks in %.2fs", chunk_count, elapsed,
                    )
                    return  # success — exit retry loop
            except _RETRYABLE_EXCEPTIONS as exc:
                elapsed = time.monotonic() - t0
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "[OpenRouter] Stream transient error on attempt %d/%d "
                        "after %.2fs (%s) — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, elapsed, exc, delay,
                    )
                    try:
                        await self._client.aclose()  # type: ignore[union-attr]
                    except Exception:
                        pass
                    self._client = None
                    client = await self._get_client()
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[OpenRouter] Stream FAILED after %d attempts (%.2fs): %s",
                        _MAX_RETRIES, elapsed, exc,
                    )
                    raise

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

