"""
Alibaba Qwen-VL planning model (mock implementation).

Simulates Qwen-VL-style responses with vision-language strengths.
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
                "[Qwen-VL 规划] 通过视觉语言模型分析用户请求，核心主题为{topic}。"
                "利用我的多模态理解能力，我将构建一个精确的生成 prompt，"
                "确保色彩、光影和构图都符合用户的期望。启动 text_to_image 生成。"
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        PlanningStep(
            thought=(
                "[Qwen-VL 分析] 结合视觉推理能力解读用户意图：{topic}。"
                "我将充分利用视觉-语言对齐能力来生成高保真度的图像。"
                "调用 text_to_image 技能。"
            ),
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        PlanningStep(
            thought=(
                "[Qwen-VL 视觉评估] 利用视觉理解能力对生成图片进行全方位评估："
                "分析画面元素的空间关系、色彩搭配的合理性、以及与文本描述的语义一致性。"
                "调用 evaluate_image。"
            ),
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        PlanningStep(
            thought=(
                "[Qwen-VL 优化] 评估得分 {score}%，视觉分析发现以下可优化点："
                "细节纹理可以更加清晰，整体色调需要调和。使用 repair_image 进行"
                "基于视觉理解的精准修复。"
            ),
            action=ActionType.REPAIR,
            action_input={
                "skill": "repair_image",
                "params": {
                    "image_url": "{image_url}",
                    "prompt": "基于视觉分析精准优化：提升细节纹理、调和色调、增强视觉层次感",
                },
            },
        ),
    ],
    "finish_success": [
        PlanningStep(
            thought=(
                "[Qwen-VL 交付] 最终评估得分 {score}%，视觉质量检查全部通过。"
                "图像在语义对齐、视觉美感和技术质量三个维度均达标。交付最终结果。"
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "success",
                "image_url": "{image_url}",
                "message": "Qwen-VL 已成功生成高质量图片！",
            },
        ),
    ],
    "finish_failure": [
        PlanningStep(
            thought=(
                "[Qwen-VL 总结] 经过多轮视觉分析与修复，最佳质量得分为 {score}%，"
                "仍未达到期望标准。建议用户提供更详细的视觉描述信息，以便进行更精确的生成。"
            ),
            action=ActionType.FINISH,
            action_input={
                "result": "failure",
                "reason": "多轮优化后质量仍未达标",
                "message": "Qwen-VL 暂时未能达到满意效果，建议提供更详细的描述。",
            },
        ),
    ],
}


class QwenVLPlanningModel(BasePlanningModel):
    """Mock Qwen-VL planning model."""

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
            id="qwen-vl",
            name="Qwen2.5-VL",
            provider="Alibaba",
            description="Alibaba's vision-language model with strong multimodal understanding",
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
        # Simulate word-by-word streaming for typewriter effect
        for word in step.thought.split():
            await asyncio.sleep(0.04)
            yield f"{word} "
        # Sentinel with parsed step data
        yield "\x00" + json.dumps({
            "thought": step.thought,
            "action": step.action.value,
            "action_input": step.action_input,
        }, ensure_ascii=False)

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
            return {k: QwenVLPlanningModel._replace_dict(v, ctx) for k, v in obj.items()}
        if isinstance(obj, list):
            return [QwenVLPlanningModel._replace_dict(i, ctx) for i in obj]
        return obj

    @staticmethod
    def _extract_topic(text: str) -> str:
        stop = {"a", "an", "the", "create", "generate", "make", "draw", "paint", "image", "picture", "of", "with", "and", "or",
                "请", "帮我", "生成", "一张", "一幅", "图片", "图像", "画"}
        words = [w for w in text.lower().split() if w not in stop][:3]
        return " ".join(words) if words else "用户请求的主题"
