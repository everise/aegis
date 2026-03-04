"""
OpenRouter provider implementation.

Connects to OpenRouter's chat-completions API for both ReAct planning
and skill execution (image generation, evaluation, repair).

Configuration is read from ``aegis.yaml`` under the ``openrouter``
section.  Key settings:

* ``api_key``           — OpenRouter API key
* ``planning_model``    — model for ReAct reasoning
* ``image_gen_model``   — model for image generation / repair
* ``vl_model``          — vision-language model for image evaluation
"""

import asyncio
import base64
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import get_settings
from app.providers.openrouter_client import OpenRouterClient, ensure_base64
from app.providers.base import (
    ActionType,
    BaseProvider,
    PlanningStep,
    ProviderInfo,
)

logger = logging.getLogger("aegis.provider.openrouter")


# ── ReAct system prompt ──────────────────────────────────────────

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

# ── Image evaluation prompt ──────────────────────────────────────

_EVALUATION_PROMPT = """\
You are an expert image quality evaluator.  Analyse the provided image
and return a structured JSON evaluation.

Evaluate the image on these criteria:
1. **quality** (0.0 – 1.0): Technical quality — sharpness, noise level,
   exposure, dynamic range, colour accuracy.
2. **aesthetics** (0.0 – 1.0): Artistic merit — composition, colour
   harmony, visual appeal, balance, originality.
3. **prompt_alignment** (0.0 – 1.0): How well the image matches the
   original generation prompt: "{prompt}"

Return **ONLY** valid JSON in this exact schema:
{{
  "scores": {{
    "quality": <float>,
    "aesthetics": <float>,
    "prompt_alignment": <float>
  }},
  "overall_score": <float>,
  "feedback": "<concise textual feedback>"
}}

``overall_score`` = 0.4 × quality + 0.3 × aesthetics + 0.3 × prompt_alignment.
Be objective and precise.  No markdown, no extra text — JSON only.
"""


class OpenRouterProvider(BaseProvider):
    """Provider backed by the OpenRouter API.

    Handles ReAct planning via configurable upstream LLM as well as
    image generation, evaluation, and repair.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        planning_model: Optional[str] = None,
        image_gen_model: Optional[str] = None,
        vl_model: Optional[str] = None,
        output_dir: str = "data/images",
    ):
        settings = get_settings()
        self._planning_model = planning_model or settings.openrouter_planning_model
        self._image_gen_model = image_gen_model or settings.openrouter_image_gen_model
        self._vl_model = vl_model or settings.openrouter_vl_model
        self._client = OpenRouterClient(api_key=api_key)
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        # Conversation history for multi-turn planning
        self._history: List[Dict[str, Any]] = []

        # Token usage accumulators
        self._planning_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._skill_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        logger.info(
            "OpenRouterProvider created  planning=%s  image_gen=%s  vl=%s",
            self._planning_model, self._image_gen_model, self._vl_model,
        )

    # ── BaseProvider interface ────────────────────────────────────

    def info(self) -> ProviderInfo:
        return ProviderInfo(
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
        self._planning_usage = {k: 0 for k in self._planning_usage}
        self._skill_usage = {k: 0 for k in self._skill_usage}

    @property
    def planning_token_usage(self) -> Dict[str, int]:
        return dict(self._planning_usage)

    @property
    def skill_token_usage(self) -> Dict[str, int]:
        return dict(self._skill_usage)

    # Backward-compat alias used by react_planner
    @property
    def token_usage(self) -> Dict[str, int]:
        return self.planning_token_usage

    # ── Planning ──────────────────────────────────────────────────

    async def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> PlanningStep:
        messages = await self._build_messages(user_message, observation)

        logger.info(
            "[Planning] get_next_step  model=%s  history_len=%d  has_observation=%s",
            self._planning_model, len(self._history), observation is not None,
        )

        try:
            resp = await self._client.chat_completion(
                model=self._planning_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("[Planning] OpenRouter request failed: %s", exc, exc_info=True)
            raise

        content = resp["choices"][0]["message"]["content"]
        logger.debug("[Planning] Raw LLM response: %s", content[:500])

        usage = resp.get("usage")
        if usage:
            self._accumulate_usage(self._planning_usage, usage)

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

        Yields raw content tokens.  The very last yield is a sentinel:
        ``\\x00`` + JSON-encoded ``PlanningStep`` dict.
        """
        messages = await self._build_messages(user_message, observation)

        logger.info(
            "[Planning] get_next_step_stream  model=%s  history_len=%d  has_observation=%s",
            self._planning_model, len(self._history), observation is not None,
        )

        accumulated = ""
        try:
            async for chunk in self._client.chat_completion_stream(
                model=self._planning_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            ):
                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    self._accumulate_usage(self._planning_usage, chunk_usage)
                delta = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta:
                    accumulated += delta
                    yield delta
        except Exception as exc:
            logger.error("[Planning] OpenRouter stream failed: %s", exc, exc_info=True)
            raise

        logger.debug(
            "[Planning] Streamed raw response (%d chars): %s",
            len(accumulated), accumulated[:500],
        )

        self._history.append({"role": "assistant", "content": accumulated})

        step = self._parse_step(accumulated)
        logger.info(
            "[Planning] Parsed streamed step: action=%s  thought=%.80s…",
            step.action.value, step.thought,
        )

        yield "\x00" + json.dumps(
            {
                "thought": step.thought,
                "action": step.action.value,
                "action_input": step.action_input,
            },
            ensure_ascii=False,
        )

    # ── Skill execution ───────────────────────────────────────────

    async def execute_skill(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if skill_name == "text_to_image":
                return await self._generate_image(params)
            elif skill_name == "evaluate_image":
                return await self._evaluate_image(params)
            elif skill_name == "repair_image":
                return await self._repair_image(params)
            else:
                return {
                    "skill_name": skill_name,
                    "status": "failed",
                    "result": None,
                    "error": f"Unknown skill for OpenRouter provider: {skill_name}",
                }
        except Exception as exc:
            logger.exception("OpenRouterProvider skill error on skill=%s", skill_name)
            return {
                "skill_name": skill_name,
                "status": "failed",
                "result": None,
                "error": f"OpenRouter skill error: {exc}",
            }

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.close()

    # ══════════════════════════════════════════════════════════════
    #  Private helpers — planning
    # ══════════════════════════════════════════════════════════════

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
            image_url = self._extract_image_url(observation)
            if image_url:
                try:
                    logger.debug(
                        "[Planning] Converting observation image to base64: %s",
                        image_url,
                    )
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
                    logger.warning(
                        "[Planning] Failed to encode image %s: %s", image_url, exc,
                    )
                    messages.append({"role": "user", "content": obs_text})
            else:
                messages.append({"role": "user", "content": obs_text})
            # Store a text-only copy in history to avoid base64 bloat
            self._history.append({"role": "user", "content": obs_text})
        else:
            messages.append({"role": "user", "content": user_message})
            self._history.append({"role": "user", "content": user_message})

        return messages

    @staticmethod
    def _extract_image_url(observation: Dict[str, Any]) -> Optional[str]:
        """Return an image URL buried inside an observation dict, if any."""
        if not observation:
            return None
        result = observation.get("result")
        if isinstance(result, dict):
            return result.get("image_url")
        return None

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

    # ══════════════════════════════════════════════════════════════
    #  Private helpers — skills
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _accumulate_usage(
        target: Dict[str, int],
        usage: Dict[str, Any],
    ) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = usage.get(key)
            if val is not None:
                target[key] += int(val)

    # ── text_to_image ─────────────────────────────────────────────

    async def _generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt", "")
        aspect_ratio = params.get("aspect_ratio", "1:1")
        image_size = params.get("image_size", "1K")
        reference_image_url = params.get("reference_image_url")

        logger.info(
            "[ImageGen] generate  prompt=%.80s…  aspect=%s  size=%s  ref=%s",
            prompt, aspect_ratio, image_size, reference_image_url is not None,
        )

        content: Any
        if reference_image_url:
            base64_uri = await ensure_base64(reference_image_url)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_uri}},
            ]
        else:
            content = prompt

        messages = [{"role": "user", "content": content}]

        resp = await self._client.chat_completion(
            model=self._image_gen_model,
            messages=messages,
            modalities=["image", "text"],
            image_config={
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
        )

        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        images = message.get("images", [])
        text_content = message.get("content", "")
        usage = resp.get("usage", {})

        if usage:
            self._accumulate_usage(self._skill_usage, usage)

        if not images:
            logger.warning("[ImageGen] Model returned no images")
            return {
                "skill_name": "text_to_image",
                "status": "failed",
                "result": None,
                "error": "No image returned by model",
            }

        image_data_url: str = images[0].get("image_url", {}).get("url", "")
        saved_path = None
        if image_data_url.startswith("data:"):
            saved_path = self._save_base64_image(image_data_url)
            logger.info("[ImageGen] Image saved → %s", saved_path)

        return {
            "skill_name": "text_to_image",
            "status": "completed",
            "result": {
                "image_url": saved_path or image_data_url,
                "width": None,
                "height": None,
                "seed": None,
            },
            "error": None,
        }

    # ── evaluate_image ────────────────────────────────────────────

    async def _evaluate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_url = params.get("image_url", "")
        prompt = params.get("prompt", "")

        logger.info(
            "[VLScore] evaluate  image=%s  prompt=%.60s…",
            image_url[:80], prompt,
        )

        base64_uri = await ensure_base64(image_url)
        eval_text = _EVALUATION_PROMPT.format(prompt=prompt or "N/A")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": eval_text},
                    {"type": "image_url", "image_url": {"url": base64_uri}},
                ],
            }
        ]

        resp = await self._client.chat_completion(
            model=self._vl_model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        logger.debug("[VLScore] Raw response: %s", content[:300])

        if usage:
            self._accumulate_usage(self._skill_usage, usage)

        result = self._parse_evaluation(content)

        logger.info(
            "[VLScore] Evaluation result: overall=%.2f  feedback=%.60s…",
            result.get("overall_score", -1), result.get("feedback", ""),
        )

        return {
            "skill_name": "evaluate_image",
            "status": "completed",
            "result": {
                "scores": result.get("scores", {}),
                "overall_score": result.get("overall_score"),
                "feedback": result.get("feedback", ""),
            },
            "error": None,
        }

    # ── repair_image ──────────────────────────────────────────────

    async def _repair_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_url = params.get("image_url", "")
        prompt = params.get("prompt", "")
        mask_url = params.get("mask_url")
        strength = float(params.get("strength", 0.75))

        logger.info(
            "[Repair] repair  image=%s  prompt=%.60s…  strength=%.2f",
            image_url[:80], prompt, strength,
        )

        base64_image = await ensure_base64(image_url)

        full_prompt = (
            f"{prompt}\n\n"
            f"(Apply modifications with strength ≈ {strength:.0%}. "
            f"Preserve the overall composition while improving details.)"
        )

        content = [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": base64_image}},
        ]

        if mask_url:
            base64_mask = await ensure_base64(mask_url)
            content.append(
                {"type": "image_url", "image_url": {"url": base64_mask}}
            )

        messages = [{"role": "user", "content": content}]

        resp = await self._client.chat_completion(
            model=self._image_gen_model,
            messages=messages,
            modalities=["image", "text"],
        )

        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        images = message.get("images", [])
        usage = resp.get("usage", {})

        if usage:
            self._accumulate_usage(self._skill_usage, usage)

        if not images:
            logger.warning("[Repair] Model returned no images")
            return {
                "skill_name": "repair_image",
                "status": "failed",
                "result": None,
                "error": "No image returned during repair",
            }

        image_data_url: str = images[0].get("image_url", {}).get("url", "")
        saved_path = None
        if image_data_url.startswith("data:"):
            saved_path = self._save_base64_image(image_data_url)
            logger.info("[Repair] Repaired image saved → %s", saved_path)

        return {
            "skill_name": "repair_image",
            "status": "completed",
            "result": {
                "image_url": saved_path or image_data_url,
                "original_url": image_url,
            },
            "error": None,
        }

    # ── Shared helpers ────────────────────────────────────────────

    def _save_base64_image(self, data_uri: str) -> str:
        """Decode a ``data:`` URI and save to disk.  Returns URL path."""
        header, b64_data = data_uri.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        ext = mime.split("/")[1]
        if ext == "jpeg":
            ext = "jpg"
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(self._output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return f"/data/images/{filename}"

    @staticmethod
    def _parse_evaluation(content: str) -> Dict[str, Any]:
        """Parse a VL model's evaluation JSON response."""
        content = content.strip()

        # Strip markdown code fences
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?\s*```\s*$", "", content)
        content = content.strip()

        data: Optional[Dict[str, Any]] = None

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

        if data is None:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            logger.warning(
                "[VLScore] Failed to parse evaluation JSON — defaulting to PASS",
            )
            return {
                "scores": {
                    "quality": 0.85,
                    "aesthetics": 0.85,
                    "prompt_alignment": 0.85,
                },
                "overall_score": 0.85,
                "feedback": f"Evaluation parse failed, defaulting to pass. Raw: {content[:200]}",
            }

        scores = data.get("scores", {})
        for key in ("quality", "aesthetics", "prompt_alignment"):
            if key not in scores:
                scores[key] = 0.5
            scores[key] = max(0.0, min(1.0, float(scores[key])))

        overall = data.get("overall_score")
        if overall is None:
            overall = (
                scores["quality"] * 0.4
                + scores["aesthetics"] * 0.3
                + scores["prompt_alignment"] * 0.3
            )
        overall = max(0.0, min(1.0, float(overall)))

        return {
            "scores": scores,
            "overall_score": overall,
            "feedback": data.get("feedback", ""),
        }
