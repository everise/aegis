"""
Abstract base class for planning models.

.. deprecated::
    Use ``app.providers.base`` instead.  This module re-exports the
    canonical types (``ActionType``, ``PlanningStep``) from the
    providers package and keeps ``BasePlanningModel`` / ``PlanningModelInfo``
    as backward-compatible aliases.
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from enum import Enum

# ── Canonical re-exports from providers package ──────────────────
from app.providers.base import (          # noqa: F401 – re-export
    ActionType,
    PlanningStep,
    ProviderInfo as _ProviderInfo,
    maybe_pad_thought as _maybe_pad_thought,
    _VERBOSE_FRAGMENTS,
)


# Backward-compatible alias
PlanningModelInfo = _ProviderInfo


class BasePlanningModel(ABC):
    """
    Abstract base class for all planning models.

    .. deprecated::
        Prefer implementing ``app.providers.base.BaseProvider`` directly.
        This class is retained for backward compatibility.
    """

    # ── Verbose padding for memory-compression testing ──────────
    _VERBOSE_FRAGMENTS = _VERBOSE_FRAGMENTS

    def _maybe_pad_thought(
        self,
        thought: str,
        probability: float = 0.6,
        fragment_range: tuple = (2, 4),
    ) -> str:
        """Randomly pad *thought* with verbose analysis fragments."""
        return _maybe_pad_thought(thought, probability, fragment_range)

    @abstractmethod
    def info(self) -> PlanningModelInfo:
        """Return metadata about this planning model."""
        ...

    @abstractmethod
    async def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> PlanningStep:
        """
        Generate the next ReAct step.

        Args:
            user_message: The original user request.
            observation: Result from previous action execution, if any.

        Returns:
            A PlanningStep containing thought, action, and action_input.
        """
        ...

    @abstractmethod
    async def get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream the next ReAct step token-by-token.

        Used for SSE streaming to the frontend.  Implementations must
        yield raw content tokens (for typewriter display) and, as the
        **very last yield**, a sentinel string that starts with
        ``\\x00`` followed by the JSON-encoded parsed step::

            yield "\\x00" + json.dumps({
                "thought": ...,
                "action": ...,
                "action_input": ...,
            })
        """
        ...
        # Must be an async generator; yield is required for type-checking
        yield ""  # pragma: no cover

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new conversation."""
        ...

    # ── Helpers used by all subclasses ──────────────────────────────

    def format_step_as_dict(self, step: PlanningStep) -> Dict[str, Any]:
        """Convert a PlanningStep to a JSON-serializable dict."""
        return {
            "thought": step.thought,
            "action": step.action.value,
            "action_input": step.action_input,
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion interface.

        Extracts user message and observation from *messages*, delegates
        to ``get_next_step``, and wraps the result in the standard
        ``choices`` envelope expected by the planner.
        """
        user_message = ""
        observation = None

        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "system" and "observation" in msg.get("content", "").lower():
                try:
                    observation = json.loads(
                        msg["content"].split("Observation:")[-1].strip()
                    )
                except (json.JSONDecodeError, IndexError):
                    pass

        step = await self.get_next_step(user_message, observation)
        content = self.format_step_as_dict(step)

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        OpenAI-compatible streaming chat completion interface.
        """
        user_message = ""
        observation = None

        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "system" and "observation" in msg.get("content", "").lower():
                try:
                    observation = json.loads(
                        msg["content"].split("Observation:")[-1].strip()
                    )
                except (json.JSONDecodeError, IndexError):
                    pass

        async for chunk in self.get_next_step_stream(user_message, observation):
            yield {
                "choices": [
                    {
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }

        yield {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
