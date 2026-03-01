"""
OpenRouter image repair skill implementation.

Uses OpenRouter's image-capable models for image-to-image repair.
The original image is converted to base64 and sent alongside the
repair instruction prompt.

This is one possible backend for the ``repair_image`` skill —
an alternative to the submit-poll ``RepairImageSkill``.
"""

import base64
import logging
import os
import uuid
from typing import Any, Dict, Optional

from app.config import get_settings
from app.services.openrouter_client import (
    OpenRouterClient,
    ensure_base64,
)

logger = logging.getLogger("aegis.skill.repair_image.openrouter")


class OpenRouterImageRepairer:
    """Repair / inpaint images via OpenRouter's image-output models.

    Config keys (``aegis.yaml → openrouter``):
      - ``image_gen_model``: upstream model id (shared with generation)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: str = "data/images",
    ):
        settings = get_settings()
        self._model = model or settings.openrouter_image_gen_model
        self._client = OpenRouterClient(api_key=api_key)
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        logger.info("OpenRouterImageRepairer created  model=%s", self._model)

    async def repair(
        self,
        image_url: str,
        prompt: str,
        *,
        mask_url: Optional[str] = None,
        strength: float = 0.75,
    ) -> Dict[str, Any]:
        """Repair / modify an image.

        Args:
            image_url: URL, path, or data-URI of the source image.
            prompt: Natural-language repair instructions.
            mask_url: Optional mask (white = area to repair).
            strength: Repair intensity hint (0.0 – 1.0).

        Returns:
            Dict with ``image_url``, ``original_url``, etc.
        """
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

        # If a mask is provided, include it too
        if mask_url:
            base64_mask = await ensure_base64(mask_url)
            content.append(
                {"type": "image_url", "image_url": {"url": base64_mask}}
            )

        messages = [{"role": "user", "content": content}]

        resp = await self._client.chat_completion(
            model=self._model,
            messages=messages,
            modalities=["image", "text"],
        )

        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        images = message.get("images", [])
        text_content = message.get("content", "")

        if not images:
            logger.warning("[Repair] Model returned no images")
            return {
                "image_url": None,
                "original_url": image_url,
                "text": text_content,
                "error": "No image returned during repair",
            }

        image_data_url: str = images[0].get("image_url", {}).get("url", "")
        saved_path = None
        if image_data_url.startswith("data:"):
            saved_path = self._save_base64_image(image_data_url)
            logger.info("[Repair] Repaired image saved → %s", saved_path)

        return {
            "image_url": saved_path or image_data_url,
            "original_url": image_url,
            "text": text_content,
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _save_base64_image(self, data_uri: str) -> str:
        header, b64_data = data_uri.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        ext = mime.split("/")[1]
        if ext == "jpeg":
            ext = "jpg"
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(self._output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return filepath

    async def close(self) -> None:
        await self._client.close()
