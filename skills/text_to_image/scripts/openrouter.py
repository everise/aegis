"""
OpenRouter text-to-image skill implementation.

Uses OpenRouter's chat-completions API with ``modalities: ["image", "text"]``
to generate images via upstream models (e.g. Gemini, Flux).

Reference images are automatically converted to base64 before being sent.
Generated images (returned as base64 data URIs) are saved to disk.

This is one possible backend for the ``text_to_image`` skill —
an alternative to the submit-poll ``TextToImageSkill``.
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

logger = logging.getLogger("aegis.skill.text_to_image.openrouter")


class OpenRouterImageGenerator:
    """Generate images via OpenRouter's image-output models.

    Config keys (``aegis.yaml → openrouter``):
      - ``image_gen_model``: upstream model id
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
        logger.info("OpenRouterImageGenerator created  model=%s  output=%s", self._model, self._output_dir)

    # ── Public API ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        reference_image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt.

        Returns dict with ``image_url``, ``base64_data``, ``text``, etc.
        """
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
            model=self._model,
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

        if not images:
            logger.warning("[ImageGen] Model returned no images")
            return {
                "image_url": None,
                "base64_data": None,
                "text": text_content,
                "width": None,
                "height": None,
                "error": "No image returned by model",
                "usage": usage,
            }

        image_data_url: str = images[0].get("image_url", {}).get("url", "")
        saved_path = None
        if image_data_url.startswith("data:"):
            saved_path = self._save_base64_image(image_data_url)
            logger.info("[ImageGen] Image saved → %s", saved_path)

        return {
            "image_url": saved_path or image_data_url,
            "base64_data": image_data_url,
            "text": text_content,
            "width": None,
            "height": None,
            "usage": usage,
        }

    # ── Image-to-image (repair) ───────────────────────────────────

    async def repair(
        self,
        image_url: str,
        prompt: str,
        *,
        strength: float = 0.75,
    ) -> Dict[str, Any]:
        """Repair / modify an image by sending it with a repair prompt."""
        logger.info("[ImageGen] repair  image=%s  prompt=%.60s…  strength=%.2f", image_url, prompt, strength)
        base64_uri = await ensure_base64(image_url)
        full_prompt = (
            f"{prompt}\n\n"
            f"(Apply modifications with strength ≈ {strength:.0%}. "
            f"Preserve the overall composition while improving details.)"
        )
        content = [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": base64_uri}},
        ]
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

        usage = resp.get("usage", {})

        if not images:
            logger.warning("[ImageGen] Repair returned no images")
            return {
                "image_url": None,
                "base64_data": None,
                "text": text_content,
                "width": None,
                "height": None,
                "error": "No image returned during repair",
                "usage": usage,
            }

        image_data_url: str = images[0].get("image_url", {}).get("url", "")
        saved_path = None
        if image_data_url.startswith("data:"):
            saved_path = self._save_base64_image(image_data_url)
            logger.info("[ImageGen] Repaired image saved → %s", saved_path)

        return {
            "image_url": saved_path or image_data_url,
            "base64_data": image_data_url,
            "original_url": image_url,
            "text": text_content,
            "width": None,
            "height": None,
            "usage": usage,
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
        # Return a URL path (not filesystem path) so the frontend can fetch it
        return f"/data/images/{filename}"

    async def close(self) -> None:
        await self._client.close()
