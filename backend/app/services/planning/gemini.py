"""
Google Gemini planning model (mock implementation).

Simulates Gemini-style responses with detailed analytical reasoning.
"""

import asyncio
import json
import random
from typing import Any, AsyncIterator, Dict, Optional

from app.services.planning.base import (
    ActionType,
    BasePlanningModel,
    PlanningModelInfo,
    PlanningStep,
)


# Gemini-flavoured response templates
_TEMPLATES = {
    "text_to_image": [
        PlanningStep(
            thought=(
                "[Gemini Analysis] I've carefully decomposed the user's request into visual components. "
                "The scene involves {topic}. Based on my multimodal understanding, I'll craft a detailed "
                "prompt that emphasises composition, lighting, and stylistic coherence. Proceeding with "
                "text_to_image generation."
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        PlanningStep(
            thought=(
                "[Gemini Reasoning] Parsing the user's intent: they want an image depicting {topic}. "
                "I'll leverage my visual-language grounding to produce a prompt that maximises "
                "aesthetic quality and semantic fidelity. Initiating generation."
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        PlanningStep(
            thought=(
                "[Gemini Evaluation] Image generated. Now I need to perform a comprehensive "
                "quality assessment — checking colour harmony, structural coherence, and alignment "
                "with the original request. Running evaluate_image."
            ),
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        PlanningStep(
            thought=(
                "[Gemini Repair] Quality score is {score}%, below the acceptable threshold. "
                "My analysis suggests the image needs refinement in detail sharpness and colour "
                "balance. Applying repair_image with targeted enhancement instructions."
            ),
            action=ActionType.REPAIR,
            action_input={
                "skill": "repair_image",
                "params": {
                    "image_url": "{image_url}",
                    "prompt": "Enhance detail sharpness, improve colour balance, and refine composition",
                },
            },
        ),
    ],
    "finish_success": [
        PlanningStep(
            thought=(
                "[Gemini Result] The image achieved a quality score of {score}%. "
                "It meets the visual-language alignment criteria and passes all quality checks. "
                "Delivering the final result to the user."
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "success",
                "image_url": "{image_url}",
                "message": "Image generated successfully by Gemini planning model!",
            },
        ),
    ],
    "finish_failure": [
        PlanningStep(
            thought=(
                "[Gemini Conclusion] Despite multiple attempts, the best quality score "
                "achieved was {score}%. The image does not meet the required standard. "
                "Recommending the user try a revised prompt."
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "failure",
                "reason": "Quality threshold not met after multiple attempts",
                "message": "Gemini could not produce a satisfactory image. Please try a different prompt.",
            },
        ),
    ],
}


class GeminiPlanningModel(BasePlanningModel):
    """Mock Gemini planning model."""

    def __init__(
        self,
        quality_threshold: float = 0.7,
        max_repair_attempts: int = 2,
    ):
        self.quality_threshold = quality_threshold
        self.max_repair_attempts = max_repair_attempts
        self._step = 0
        self._repairs = 0
        self._image_url: Optional[str] = None
        self._last_score: Optional[float] = None

    # ── Interface implementation ──────────────────────────────────

    def info(self) -> PlanningModelInfo:
        return PlanningModelInfo(
            id="gemini",
            name="Gemini 2.5 Pro",
            provider="Google",
            description="Google's multimodal model with strong analytical reasoning",
            supports_vision=True,
            supports_streaming=True,
        )

    def reset(self) -> None:
        self._step = 0
        self._repairs = 0
        self._image_url = None
        self._last_score = None

    async def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> PlanningStep:
        self._step += 1
        topic = self._extract_topic(user_message)
        return self._decide(user_message, observation, topic)

    async def get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        step = await self.get_next_step(user_message, observation)
        yield "thought:"
        for word in step.thought.split():
            await asyncio.sleep(0.04)
            yield f" {word}"
        yield "\n"
        yield f"action: {step.action.value}\n"
        yield f"action_input: {json.dumps(step.action_input)}\n"

    # ── Private helpers ───────────────────────────────────────────

    def _decide(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]],
        topic: str,
    ) -> PlanningStep:
        # Step 1: generate
        if self._step == 1:
            return self._from_template("text_to_image", {"prompt": user_message, "topic": topic})

        # After generation → update image_url, then evaluate
        if observation and "image_url" in observation.get("result", {}):
            self._image_url = observation["result"]["image_url"]

        if self._step == 2 and self._image_url:
            return self._from_template("evaluate", {"image_url": self._image_url})

        # After evaluation → check score
        if observation and "overall_score" in observation.get("result", {}):
            self._last_score = observation["result"]["overall_score"]
            pct = int(self._last_score * 100)

            if self._last_score >= self.quality_threshold:
                return self._from_template("finish_success", {"image_url": self._image_url, "score": str(pct)})

            if self._repairs < self.max_repair_attempts:
                self._repairs += 1
                return self._from_template("repair", {"image_url": self._image_url, "score": str(pct)})

            return self._from_template("finish_failure", {"score": str(pct)})

        # After repair → re-evaluate
        if observation and observation.get("skill_name") == "repair_image":
            if "image_url" in observation.get("result", {}):
                self._image_url = observation["result"]["image_url"]
            return self._from_template("evaluate", {"image_url": self._image_url})

        # Fallback
        return self._from_template("finish_success", {"image_url": self._image_url or "", "score": "85"})

    def _from_template(self, key: str, ctx: Dict[str, str]) -> PlanningStep:
        tpl = random.choice(_TEMPLATES[key])
        thought = self._replace(tpl.thought, ctx)
        thought = self._maybe_pad_thought(thought)
        return PlanningStep(
            thought=thought,
            action=tpl.action,
            action_input=self._replace_dict(tpl.action_input, ctx),
        )

    @staticmethod
    def _replace(text: str, ctx: Dict[str, str]) -> str:
        for k, v in ctx.items():
            text = text.replace(f"{{{k}}}", v)
        return text

    @staticmethod
    def _replace_dict(obj: Any, ctx: Dict[str, str]) -> Any:
        if isinstance(obj, str):
            r = obj
            for k, v in ctx.items():
                r = r.replace(f"{{{k}}}", v)
            return r
        if isinstance(obj, dict):
            return {k: GeminiPlanningModel._replace_dict(v, ctx) for k, v in obj.items()}
        if isinstance(obj, list):
            return [GeminiPlanningModel._replace_dict(i, ctx) for i in obj]
        return obj

    @staticmethod
    def _extract_topic(text: str) -> str:
        stop = {"a", "an", "the", "create", "generate", "make", "draw", "paint", "image", "picture", "of", "with", "and", "or"}
        words = [w for w in text.lower().split() if w not in stop][:3]
        return " ".join(words) if words else "the requested subject"
