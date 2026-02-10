"""
Moonshot Kimi planning model (mock implementation).

Simulates Kimi-style responses with thoughtful, conversational reasoning.
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


_TEMPLATES = {
    "text_to_image": [
        PlanningStep(
            thought=(
                "[Kimi 思考] 用户希望生成一张关于{topic}的图片。让我仔细理解用户的需求，"
                "提取关键的视觉元素，包括构图、色彩和风格要求。我将使用 text_to_image 技能"
                "来生成匹配用户描述的高质量图片。"
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        PlanningStep(
            thought=(
                "[Kimi 分析] 收到了用户的图片生成请求，主题涉及{topic}。我会深入分析"
                "用户的意图，确保 prompt 能够完整传达画面细节。开始调用 text_to_image。"
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        PlanningStep(
            thought=(
                "[Kimi 评估] 图片已经生成完成。接下来我需要对图片进行全面的质量检查，"
                "包括画面清晰度、色彩协调性以及与用户原始需求的匹配程度。调用 evaluate_image。"
            ),
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        PlanningStep(
            thought=(
                "[Kimi 修复] 当前质量评分为 {score}%，还有提升空间。我分析了图片中可以"
                "改进的地方：细节清晰度和整体构图需要优化。使用 repair_image 进行增强。"
            ),
            action=ActionType.REPAIR,
            action_input={
                "skill": "repair_image",
                "params": {
                    "image_url": "{image_url}",
                    "prompt": "优化细节清晰度，增强色彩表现力，改善整体构图",
                },
            },
        ),
    ],
    "finish_success": [
        PlanningStep(
            thought=(
                "[Kimi 完成] 图片质量评分达到了 {score}%，符合质量标准。"
                "画面效果很好，与用户的需求高度匹配。将结果呈现给用户。"
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "success",
                "image_url": "{image_url}",
                "message": "Kimi 已成功为您生成图片！",
            },
        ),
    ],
    "finish_failure": [
        PlanningStep(
            thought=(
                "[Kimi 总结] 经过多次尝试，最佳质量评分为 {score}%，"
                "未能达到理想标准。建议用户尝试更具体的描述，我可以再次为您服务。"
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "failure",
                "reason": "多次尝试后质量仍未达标",
                "message": "抱歉，Kimi 未能生成满意的图片，请尝试调整您的描述。",
            },
        ),
    ],
}


class KimiPlanningModel(BasePlanningModel):
    """Mock Kimi (Moonshot) planning model."""

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

    def info(self) -> PlanningModelInfo:
        return PlanningModelInfo(
            id="kimi",
            name="Kimi k2",
            provider="Moonshot AI",
            description="Moonshot's conversational model with strong Chinese-language understanding",
            supports_vision=False,
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
        yield f"action_input: {json.dumps(step.action_input, ensure_ascii=False)}\n"

    # ── Private ───────────────────────────────────────────────────

    def _decide(self, user_message: str, observation: Optional[Dict[str, Any]], topic: str) -> PlanningStep:
        if self._step == 1:
            return self._from_template("text_to_image", {"prompt": user_message, "topic": topic})

        if observation and "image_url" in observation.get("result", {}):
            self._image_url = observation["result"]["image_url"]

        if self._step == 2 and self._image_url:
            return self._from_template("evaluate", {"image_url": self._image_url})

        if observation and "overall_score" in observation.get("result", {}):
            self._last_score = observation["result"]["overall_score"]
            pct = int(self._last_score * 100)

            if self._last_score >= self.quality_threshold:
                return self._from_template("finish_success", {"image_url": self._image_url, "score": str(pct)})

            if self._repairs < self.max_repair_attempts:
                self._repairs += 1
                return self._from_template("repair", {"image_url": self._image_url, "score": str(pct)})

            return self._from_template("finish_failure", {"score": str(pct)})

        if observation and observation.get("skill_name") == "repair_image":
            if "image_url" in observation.get("result", {}):
                self._image_url = observation["result"]["image_url"]
            return self._from_template("evaluate", {"image_url": self._image_url})

        return self._from_template("finish_success", {"image_url": self._image_url or "", "score": "85"})

    def _from_template(self, key: str, ctx: Dict[str, str]) -> PlanningStep:
        tpl = random.choice(_TEMPLATES[key])
        return PlanningStep(
            thought=self._replace(tpl.thought, ctx),
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
            return {k: KimiPlanningModel._replace_dict(v, ctx) for k, v in obj.items()}
        if isinstance(obj, list):
            return [KimiPlanningModel._replace_dict(i, ctx) for i in obj]
        return obj

    @staticmethod
    def _extract_topic(text: str) -> str:
        stop = {"a", "an", "the", "create", "generate", "make", "draw", "paint", "image", "picture", "of", "with", "and", "or",
                "请", "帮我", "生成", "一张", "一幅", "图片", "图像", "画"}
        words = [w for w in text.lower().split() if w not in stop][:3]
        return " ".join(words) if words else "用户请求的主题"
