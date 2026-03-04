"""
Abstract base provider interface.

A *provider* encapsulates a specific LLM / AI service (e.g. OpenRouter,
Mock, Vertex AI).  Planning and skill execution both go through this
unified interface, so the rest of the application never needs to know
which concrete service is being used.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional


# ── Domain types shared by all providers ──────────────────────────


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    GENERATE = "generate"
    EVALUATE = "evaluate"
    REPAIR = "repair"
    FINISH = "finish"


@dataclass
class PlanningStep:
    """A single step produced by the provider's planning logic."""
    thought: str
    action: ActionType
    action_input: Dict[str, Any]


@dataclass
class ProviderInfo:
    """Metadata describing a provider."""
    id: str
    name: str
    provider: str           # Human-readable provider label (for API compat)
    description: str
    supports_vision: bool = False
    supports_streaming: bool = True


# ── Verbose padding fragments (for mock / testing) ───────────────

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


def maybe_pad_thought(
    thought: str,
    probability: float = 0.6,
    fragment_range: tuple = (2, 4),
) -> str:
    """Randomly pad *thought* with verbose analysis fragments.

    Used by mock providers to inflate token counts and trigger
    working-memory compression during development.
    """
    if random.random() >= probability:
        return thought
    lo, hi = fragment_range
    n = random.randint(lo, min(hi, len(_VERBOSE_FRAGMENTS)))
    fragments = random.sample(_VERBOSE_FRAGMENTS, n)
    return thought + "\n\n" + "\n\n".join(fragments)


# ── Abstract provider ─────────────────────────────────────────────


class BaseProvider(ABC):
    """
    Abstract base class for all providers.

    A provider handles *both* planning (ReAct reasoning) and skill
    execution (image generation, evaluation, repair).  Concrete
    subclasses implement the specifics for a given service.
    """

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return metadata about this provider."""
        ...

    # ── Planning interface ────────────────────────────────────────

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

        Yields raw content tokens (for typewriter display) and, as the
        **very last yield**, a sentinel string that starts with
        ``\\x00`` followed by the JSON-encoded parsed step::

            yield "\\x00" + json.dumps({
                "thought": ...,
                "action": ...,
                "action_input": ...,
            })
        """
        ...
        # Must be an async generator; yield is required for type-checking
        yield ""  # pragma: no cover

    # ── Skill interface ───────────────────────────────────────────

    @abstractmethod
    async def execute_skill(
        self,
        skill_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a skill (text_to_image, evaluate_image, repair_image).

        Returns a dict with at least:
            - skill_name: str
            - status: "completed" | "failed"
            - result: Optional[Dict]  (skill-specific output)
            - error: Optional[str]
        """
        ...

    # ── Lifecycle ─────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new conversation."""
        ...

    @property
    def planning_token_usage(self) -> Dict[str, int]:
        """Token usage accumulated from planning (LLM reasoning) calls."""
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def skill_token_usage(self) -> Dict[str, int]:
        """Token usage accumulated from skill execution calls."""
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def close(self) -> None:
        """Clean up resources.  Override if needed."""
        pass

    # ── Convenience helpers ───────────────────────────────────────

    def format_step_as_dict(self, step: PlanningStep) -> Dict[str, Any]:
        """Convert a PlanningStep to a JSON-serializable dict."""
        return {
            "thought": step.thought,
            "action": step.action.value,
            "action_input": step.action_input,
        }
