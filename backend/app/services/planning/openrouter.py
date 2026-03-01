"""
OpenRouter planning model implementation.

Connects to OpenRouter's chat-completions API to drive the
ReAct planning loop with a real LLM.  Uploaded / observed images
are automatically converted to base64 before being sent.

This is a *non-default* implementation — register it alongside
the existing mock models in the planning registry.
"""

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import get_settings
from app.services.openrouter_client import (
    OpenRouterClient,
    ensure_base64,
)
from app.services.planning.base import (
    ActionType,
    BasePlanningModel,
    PlanningModelInfo,
    PlanningStep,
)

logger = logging.getLogger("aegis.openrouter.planning")


# ── System prompt for ReAct-style output ──────────────────────────

_SYSTEM_PROMPT = """\
You are an AI planning agent that follows the ReAct (Reasoning + Acting) framework.
You help users generate, evaluate, and improve images through an iterative process.

Available actions:
- generate: Generate an image using the text_to_image skill.
- evaluate: Evaluate image quality using the evaluate_image skill.
- repair:   Repair / improve an image using the repair_image skill.
- finish:   Complete the task with a final result.

You MUST respond with **valid JSON only** in precisely this format:
{
  "thought": "<your detailed reasoning about what to do next>",
  "action": "generate|evaluate|repair|finish",
  "action_input": { ... }
}

action_input schemas per action:
  generate → {"skill": "text_to_image", "params": {"prompt": "..."}}
  evaluate → {"skill": "evaluate_image", "params": {"image_url": "..."}}
  repair   → {"skill": "repair_image",  "params": {"image_url": "...", "prompt": "..."}}
  finish   → {"result": "success"|"failure", "image_url": "...", "message": "..."}

Workflow guidelines:
1. Start by generating an image matching the user's request.
2. Evaluate the generated image's quality.
3. If the quality_score < 0.7, repair the image (max 2 repair attempts), then re-evaluate.
4. Finish with "success" when quality_score ≥ 0.7, or "failure" after exhausting repair attempts.

Respond with JSON only — no markdown fences, no extra text.
"""


class OpenRouterPlanningModel(BasePlanningModel):
    """Planning model powered by OpenRouter API.

    Configuration is read from ``aegis.yaml`` under the ``openrouter``
    section.  The ``planning_model`` key specifies which upstream model
    to use (default: ``google/gemini-2.5-pro-preview``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self._model = model or settings.openrouter_planning_model
        self._client = OpenRouterClient(api_key=api_key)
        self._history: List[Dict[str, Any]] = []
        logger.info("OpenRouterPlanningModel created  model=%s", self._model)

    # ── BasePlanningModel interface ───────────────────────────────

    def info(self) -> PlanningModelInfo:
        return PlanningModelInfo(
            id="openrouter",
            name="OpenRouter",
            provider="OpenRouter",
            description=(
                "Planning model via OpenRouter API — uses a configurable "
                "upstream LLM for ReAct reasoning"
            ),
            supports_vision=True,
            supports_streaming=True,
        )

    def reset(self) -> None:
        self._history.clear()

    async def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> PlanningStep:
        messages = await self._build_messages(user_message, observation)

        logger.info(
            "[Planning] get_next_step  model=%s  history_len=%d  has_observation=%s",
            self._model, len(self._history), observation is not None,
        )

        try:
            resp = await self._client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("[Planning] OpenRouter request failed: %s", exc, exc_info=True)
            raise

        content = resp["choices"][0]["message"]["content"]
        logger.debug("[Planning] Raw LLM response: %s", content[:500])

        # Keep history for multi-turn
        self._history.append({"role": "assistant", "content": content})

        step = self._parse_step(content)
        logger.info(
            "[Planning] Parsed step: action=%s  thought=%.80s…",
            step.action.value, step.thought,
        )
        return step

    async def get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Stream the next ReAct step.

        Internally calls the non-streaming API and then yields
        the result token-by-token so the SSE layer can push
        incremental updates to the frontend.
        """
        step = await self.get_next_step(user_message, observation)

        yield "thought:"
        for word in step.thought.split():
            yield f" {word}"
        yield "\n"
        yield f"action: {step.action.value}\n"
        yield f"action_input: {json.dumps(step.action_input, ensure_ascii=False)}\n"

    # ── Private helpers ───────────────────────────────────────────

    async def _build_messages(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble the message list for the API call."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

        # Replay conversation history
        messages.extend(self._history)

        # Current turn
        if observation is not None:
            obs_text = (
                "Observation from previous action:\n"
                + json.dumps(observation, ensure_ascii=False)
            )
            # If the observation contains an image URL, convert to base64
            image_url = self._extract_image_url(observation)
            if image_url:
                try:
                    logger.debug("[Planning] Converting observation image to base64: %s", image_url)
                    base64_uri = await ensure_base64(image_url)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": obs_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_uri},
                                },
                            ],
                        }
                    )
                except Exception as exc:
                    logger.warning("[Planning] Failed to encode image %s: %s", image_url, exc)
                    messages.append({"role": "user", "content": obs_text})
            else:
                messages.append({"role": "user", "content": obs_text})
            # Store a text-only copy in history to avoid base64 bloat
            self._history.append({"role": "user", "content": obs_text})
        else:
            messages.append({"role": "user", "content": user_message})
            self._history.append({"role": "user", "content": user_message})

        return messages

    # ── Image utilities ───────────────────────────────────────────

    @staticmethod
    def _extract_image_url(observation: Dict[str, Any]) -> Optional[str]:
        """Return an image URL buried inside an observation dict, if any."""
        if not observation:
            return None
        result = observation.get("result")
        if isinstance(result, dict):
            return result.get("image_url")
        return None

    # ── Response parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_step(content: str) -> PlanningStep:
        """Parse the LLM's JSON response into a ``PlanningStep``."""
        content = content.strip()

        # 1) Try direct JSON parse
        data: Optional[Dict[str, Any]] = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

        # 2) Try to extract the first JSON object from surrounding text
        if data is None:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # 3) Fallback — treat the whole response as a thought and finish
        if data is None:
            return PlanningStep(
                thought=content,
                action=ActionType.FINISH,
                action_input={
                    "result": "failure",
                    "message": "Failed to parse model response as JSON",
                },
            )

        # Map action string → ActionType
        action_str = data.get("action", "finish").lower()
        try:
            action = ActionType(action_str)
        except ValueError:
            action = ActionType.FINISH

        return PlanningStep(
            thought=data.get("thought", ""),
            action=action,
            action_input=data.get("action_input", {}),
        )
