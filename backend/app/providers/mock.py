"""
Mock provider implementation.

A local, template-based provider that simulates the ReAct loop and
skill execution without any external API calls.  Useful for
development, testing, and offline demos.
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Optional

from app.providers.base import (
    ActionType,
    BaseProvider,
    PlanningStep,
    ProviderInfo,
    maybe_pad_thought,
)

# ── Response templates ────────────────────────────────────────────

_TEMPLATES = {
    "text_to_image": [
        PlanningStep(
            thought=(
                "[Mock Analysis] Carefully decomposing the user's request into "
                "visual components. The scene involves {topic}. I'll craft a "
                "detailed prompt emphasising composition, lighting, and stylistic "
                "coherence. Proceeding with text_to_image generation."
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        PlanningStep(
            thought=(
                "[Mock Reasoning] Parsing the user's intent: they want an image "
                "depicting {topic}. Leveraging visual-language grounding to "
                "produce a prompt that maximises aesthetic quality and semantic "
                "fidelity. Initiating generation."
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        PlanningStep(
            thought=(
                "[Mock Evaluation] Image generated. Performing a comprehensive "
                "quality assessment — checking colour harmony, structural "
                "coherence, and alignment with the original request. Running "
                "evaluate_image."
            ),
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        PlanningStep(
            thought=(
                "[Mock Repair] Quality score is {score}%, below the acceptable "
                "threshold. Analysis suggests the image needs refinement in "
                "detail sharpness and colour balance. Applying repair_image "
                "with targeted enhancement instructions."
            ),
            action=ActionType.REPAIR,
            action_input={
                "skill": "repair_image",
                "params": {
                    "image_url": "{image_url}",
                    "prompt": (
                        "Enhance detail sharpness, improve colour balance, "
                        "and refine composition"
                    ),
                },
            },
        ),
    ],
    "finish_success": [
        PlanningStep(
            thought=(
                "[Mock Result] The image achieved a quality score of {score}%. "
                "It meets the alignment criteria and passes all quality checks. "
                "Delivering the final result to the user."
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "success",
                "image_url": "{image_url}",
                "message": "Image generated successfully by mock planning provider!",
            },
        ),
    ],
    "finish_failure": [
        PlanningStep(
            thought=(
                "[Mock Conclusion] Despite multiple attempts, the best quality "
                "score achieved was {score}%. The image does not meet the "
                "required standard. Recommending the user try a revised prompt."
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "failure",
                "reason": "Quality threshold not met after multiple attempts",
                "message": (
                    "Could not produce a satisfactory image. "
                    "Please try a different prompt."
                ),
            },
        ),
    ],
}


class MockProvider(BaseProvider):
    """
    Mock provider — runs entirely locally with template-based responses
    and simulated skill results.  No external API calls required.

    Use this provider for development, testing, and offline demos.
    """

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
        self._step_count = 0  # for skill simulation

    # ── BaseProvider interface ────────────────────────────────────

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id="mock",
            name="Mock (Local)",
            provider="Mock",
            description=(
                "Template-based local provider — no API calls, "
                "useful for development and testing"
            ),
            supports_vision=False,
            supports_streaming=True,
        )

    def reset(self) -> None:
        self._step = 0
        self._repairs = 0
        self._image_url = None
        self._last_score = None

    @property
    def planning_token_usage(self) -> Dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def skill_token_usage(self) -> Dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Backward-compat alias
    @property
    def token_usage(self) -> Dict[str, int]:
        return self.planning_token_usage

    # ── Planning ──────────────────────────────────────────────────

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
        # Simulate word-by-word streaming for typewriter effect
        for word in step.thought.split():
            await asyncio.sleep(0.04)
            yield f"{word} "
        # Sentinel with parsed step data
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
        """Return simulated skill results."""
        self._step_count += 1

        # Simulate realistic processing time
        await asyncio.sleep(random.uniform(3.0, 6.0))

        if skill_name == "text_to_image":
            return {
                "skill_name": skill_name,
                "status": "completed",
                "result": {
                    "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/512/512",
                    "width": 512,
                    "height": 512,
                    "seed": random.randint(1, 999999),
                },
                "error": None,
            }

        elif skill_name == "evaluate_image":
            score = random.uniform(0.6, 0.95)
            return {
                "skill_name": skill_name,
                "status": "completed",
                "result": {
                    "scores": {
                        "quality": score,
                        "aesthetics": score + random.uniform(-0.1, 0.1),
                        "prompt_alignment": score + random.uniform(-0.1, 0.1),
                    },
                    "overall_score": score,
                    "feedback": (
                        "Image looks good"
                        if score >= 0.7
                        else "Image needs improvement"
                    ),
                },
                "error": None,
            }

        elif skill_name == "repair_image":
            return {
                "skill_name": skill_name,
                "status": "completed",
                "result": {
                    "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/512/512",
                    "original_url": params.get("image_url"),
                },
                "error": None,
            }

        else:
            return {
                "skill_name": skill_name,
                "status": "failed",
                "result": None,
                "error": f"Unknown skill: {skill_name}",
            }

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        pass

    # ── Private helpers — planning ────────────────────────────────

    def _decide(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]],
        topic: str,
    ) -> PlanningStep:
        # Step 1: generate
        if self._step == 1:
            return self._from_template(
                "text_to_image", {"prompt": user_message, "topic": topic}
            )

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
                return self._from_template(
                    "finish_success",
                    {"image_url": self._image_url, "score": str(pct)},
                )

            if self._repairs < self.max_repair_attempts:
                self._repairs += 1
                return self._from_template(
                    "repair",
                    {"image_url": self._image_url, "score": str(pct)},
                )

            return self._from_template("finish_failure", {"score": str(pct)})

        # After repair → re-evaluate
        if observation and observation.get("skill_name") == "repair_image":
            if "image_url" in observation.get("result", {}):
                self._image_url = observation["result"]["image_url"]
            return self._from_template("evaluate", {"image_url": self._image_url})

        # Fallback
        return self._from_template(
            "finish_success",
            {"image_url": self._image_url or "", "score": "85"},
        )

    def _from_template(self, key: str, ctx: Dict[str, str]) -> PlanningStep:
        tpl = random.choice(_TEMPLATES[key])
        thought = self._replace(tpl.thought, ctx)
        thought = maybe_pad_thought(thought)
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
            return {
                k: MockProvider._replace_dict(v, ctx)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [MockProvider._replace_dict(i, ctx) for i in obj]
        return obj

    @staticmethod
    def _extract_topic(text: str) -> str:
        stop = {
            "a", "an", "the", "create", "generate", "make", "draw",
            "paint", "image", "picture", "of", "with", "and", "or",
        }
        words = [w for w in text.lower().split() if w not in stop][:3]
        return " ".join(words) if words else "the requested subject"
