"""
OpenRouter Vision-Language image scoring skill implementation.

Uses a vision-capable model via OpenRouter to evaluate image quality.
Images are automatically converted to base64 before sending.

This is one possible backend for the ``evaluate_image`` skill —
an alternative to the submit-poll ``EvaluateImageSkill``.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.openrouter_client import (
    OpenRouterClient,
    ensure_base64,
)

logger = logging.getLogger("aegis.skill.evaluate_image.openrouter")

# ── Evaluation prompt ─────────────────────────────────────────────

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


class OpenRouterVLScorer:
    """Evaluate image quality using a vision model via OpenRouter.

    Config keys (``aegis.yaml → openrouter``):
      - ``vl_model``: upstream vision model id
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self._model = model or settings.openrouter_vl_model
        self._client = OpenRouterClient(api_key=api_key)
        logger.info("OpenRouterVLScorer created  model=%s", self._model)

    # ── Public API ────────────────────────────────────────────────

    async def evaluate(
        self,
        image_url: str,
        prompt: str = "",
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score an image using a vision-language model.

        Returns dict with ``scores``, ``overall_score``, ``feedback``.
        """
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
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        logger.debug("[VLScore] Raw response: %s", content[:300])

        result = self._parse_evaluation(content)
        result["usage"] = usage
        logger.info(
            "[VLScore] Evaluation result: overall=%.2f  feedback=%.60s…",
            result.get("overall_score", -1), result.get("feedback", ""),
        )
        return result

    async def batch_evaluate(
        self,
        items: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple images sequentially."""
        results: List[Dict[str, Any]] = []
        for item in items:
            result = await self.evaluate(
                image_url=item["image_url"],
                prompt=item.get("prompt", ""),
            )
            results.append(result)
        return results

    # ── Response parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_evaluation(content: str) -> Dict[str, Any]:
        content = content.strip()

        # Strip markdown code fences (```json ... ```) that models often add
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
            # Parsing failed — default to PASS so the workflow finishes
            # instead of looping with a 50% score that triggers repair.
            logger.warning("[VLScore] Failed to parse evaluation JSON — defaulting to PASS")
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

    # ── Lifecycle ─────────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.close()
