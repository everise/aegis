"""
Abstract base class for planning models.

Defines the interface that all planning model implementations must follow.
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from enum import Enum


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    GENERATE = "generate"
    EVALUATE = "evaluate"
    REPAIR = "repair"
    FINISH = "finish"


@dataclass
class PlanningStep:
    """A single step produced by the planning model."""
    thought: str
    action: ActionType
    action_input: Dict[str, Any]


@dataclass
class PlanningModelInfo:
    """Metadata describing a planning model."""
    id: str
    name: str
    provider: str
    description: str
    supports_vision: bool = False
    supports_streaming: bool = True


class BasePlanningModel(ABC):
    """
    Abstract base class for all planning models.

    Each planning model implementation wraps a specific LLM provider
    (e.g., Gemini, Kimi, Qwen VL) and provides the ReAct-style
    reasoning interface used by the planner.
    """

    # ── Verbose padding for memory-compression testing ──────────
    _VERBOSE_FRAGMENTS = [
        (
            "Performing a detailed compositional analysis: the rule of thirds, leading lines, "
            "golden ratio (≈1.618:1), color harmony via the color wheel, warm-vs-cool depth cues, "
            "and figure-ground separation all need careful consideration before I proceed."
        ),
        (
            "Assessing technical quality dimensions: dynamic range preservation, histogram balance "
            "across luminance spectrum, edge sharpness at multiple frequency bands, anti-aliasing "
            "quality along diagonals, and noise-vs-detail trade-off in uniform-color regions."
        ),
        (
            "Evaluating perceptual quality: FID (Fréchet Inception Distance), Inception Score, "
            "CLIP-based text-image alignment in shared latent space, aesthetic predictor confidence, "
            "and subjective human-preference correlation all inform this assessment."
        ),
        (
            "Considering art-historical context: Renaissance chiaroscuro for light-shadow interplay, "
            "Impressionist optical mixing and broken color, Bauhaus design principles, and contemporary "
            "digital art aesthetics with procedural complexity and emergent organic patterns."
        ),
        (
            "Checking semantic alignment: presence of all explicitly mentioned objects, correct spatial "
            "relationships (above/below/beside/behind), mood-atmosphere coherence with prompt tone, "
            "accurate attribute rendering (color, size, quantity), and common-sense plausibility."
        ),
        (
            "Analyzing color science: CIE L*a*b* perceptual uniformity, ΔE thresholds (< 1 imperceptible, "
            "> 5 obvious), gamut mapping for sRGB/DCI-P3/Rec.2020 coverage, ICC profile conformance, "
            "and saturation headroom for vibrant outputs."
        ),
        (
            "Running cognitive-perceptual review: 13 ms pre-attentive image processing, facial recognition "
            "circuits, symmetry detection, gestalt grouping (proximity, similarity, continuity, closure), "
            "and visual saliency mapping to predict viewer attention distribution."
        ),
        (
            "Multi-modal reasoning pipeline check: text encoder → latent representation, cross-attention "
            "layer focus per word, denoising U-Net refinement schedule, classifier-free guidance scale "
            "trade-off between fidelity and diversity, and VAE decoder reconstruction quality."
        ),
    ]

    def _maybe_pad_thought(
        self,
        thought: str,
        probability: float = 0.6,
        fragment_range: tuple = (2, 4),
    ) -> str:
        """Randomly pad *thought* with verbose analysis fragments.

        Used by mock planning models to inflate token counts and
        trigger working-memory compression during development.
        """
        if random.random() >= probability:
            return thought
        lo, hi = fragment_range
        n = random.randint(lo, min(hi, len(self._VERBOSE_FRAGMENTS)))
        fragments = random.sample(self._VERBOSE_FRAGMENTS, n)
        return thought + "\n\n" + "\n\n".join(fragments)

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

        Used for SSE streaming to the frontend.
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
