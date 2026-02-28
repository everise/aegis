"""
LLM simulator for development and testing.

Provides mock LLM responses that follow the ReAct pattern
without requiring actual API calls.
"""

import json
import random
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    GENERATE = "generate"       # text_to_image
    EVALUATE = "evaluate"       # evaluate_image
    REPAIR = "repair"          # repair_image
    FINISH = "finish"          # Task complete


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning process."""
    thought: str
    action: ActionType
    action_input: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────
# Verbose thought fragments — randomly injected to inflate token count
# and trigger memory compression during mock conversations.
# ──────────────────────────────────────────────────────────────────
VERBOSE_THOUGHT_FRAGMENTS = [
    (
        "Let me conduct a thorough analysis of the compositional elements. "
        "The rule of thirds suggests placing key elements along invisible grid lines that "
        "divide the frame into nine equal parts. Leading lines can guide the viewer's eye "
        "through the image. The golden ratio (approximately 1.618:1) is another powerful "
        "compositional tool that creates naturally pleasing proportions. Color harmony follows "
        "the color wheel, with complementary colors providing contrast and analogous colors "
        "offering unity. Warm tones (reds, oranges, yellows) advance toward the viewer, while "
        "cool tones (blues, greens, purples) recede, creating depth."
    ),
    (
        "From a technical perspective, I should consider the dynamic range of the final image. "
        "High dynamic range (HDR) techniques can preserve detail in both shadows and highlights. "
        "The histogram should show a balanced distribution of tonal values across the luminance "
        "spectrum. Edge sharpness at various frequency bands (low, mid, high) determines the "
        "perceived detail level. Anti-aliasing helps smooth jagged edges, particularly along "
        "diagonal lines and curves. Noise reduction algorithms need to balance between preserving "
        "detail and removing unwanted grain, especially in areas of uniform color."
    ),
    (
        "The psychological impact of visual elements is significant. Research in visual perception "
        "shows that humans process images in approximately 13 milliseconds, long before conscious "
        "awareness. Facial recognition, symmetry detection, and color associations happen at a "
        "pre-attentive level. The gestalt principles of proximity, similarity, continuity, and "
        "closure all influence how viewers perceive and organize visual information. This understanding "
        "helps in creating images that communicate the intended message effectively."
    ),
    (
        "Considering the art historical context, this type of visual work draws on traditions "
        "ranging from Renaissance chiaroscuro (the interplay of light and shadow) to Impressionist "
        "color theory (optical mixing, broken color technique) to contemporary digital art "
        "aesthetics. Each tradition brings specific expectations about quality: accuracy of "
        "proportions, richness of palette, sophistication of lighting, and emotional resonance. "
        "Modern generative art adds new dimensions including procedural complexity, emergent patterns, "
        "and the fascinating interplay between algorithmic precision and organic imperfection."
    ),
    (
        "The semantic alignment between the textual prompt and the visual output requires careful "
        "evaluation. This involves checking whether all explicitly mentioned objects are present, "
        "whether spatial relationships described in the prompt (above, below, next to, behind) are "
        "correctly rendered, whether the mood and atmosphere match the implied emotional tone of "
        "the text, and whether any attributes (colors, sizes, quantities) are accurately represented. "
        "Implicit expectations from common sense must also be satisfied — a sunset scene should have "
        "warm lighting, a winter landscape should suggest cold."
    ),
    (
        "I need to evaluate the perceptual quality metrics more carefully. The Fréchet Inception "
        "Distance (FID) measures the distance between the distribution of generated images and real "
        "images in a feature space. The Inception Score (IS) evaluates both quality and diversity. "
        "CLIP-based metrics can assess text-image alignment by comparing embeddings in a shared "
        "latent space. The aesthetic predictor models trained on large datasets of human preference "
        "judgments provide another quality dimension. These quantitative metrics complement subjective "
        "human evaluation."
    ),
    (
        "From a multi-modal perspective, the relationship between text and image generation is "
        "bidirectional. The text encoder transforms the prompt into a latent representation that "
        "guides the diffusion process. Each word's influence is modulated by the attention mechanism, "
        "with some words having stronger impact on the final output. Cross-attention layers allow the "
        "model to focus on different parts of the text at different stages of generation. The denoising "
        "process gradually refines random noise into a coherent image, guided by this textual signal. "
        "Understanding this pipeline helps in crafting better prompts and evaluating outputs."
    ),
    (
        "Regarding color science, the CIE L*a*b* color space provides a perceptually uniform "
        "representation where equal distances correspond to equal perceived color differences. "
        "Delta E (ΔE) values below 1.0 are generally imperceptible, while values above 5.0 are "
        "obviously different. Gamut mapping ensures colors can be reproduced across different "
        "devices and media. ICC profiles define the color characteristics of input and output "
        "devices. Wide-gamut displays (such as DCI-P3 or Rec.2020) can represent more saturated "
        "colors than standard sRGB, which is important for vibrant, colorful images."
    ),
]


# Predefined response templates for different scenarios
# Each scenario has multiple variations for more realistic conversations
RESPONSE_TEMPLATES = {
    "text_to_image": [
        ReActStep(
            thought="I understand the user wants to create an image. Let me analyze their request to extract the key visual elements. The prompt seems focused on {topic}. I'll use the text_to_image skill to generate a high-quality image matching their description.",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        ReActStep(
            thought="The user has provided a creative image request. I need to carefully consider the composition, style, and details they want. This involves {topic}. Let me generate the image using text_to_image with their full prompt to capture all the nuances.",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
        ReActStep(
            thought="Analyzing the user's image generation request. They want something related to {topic}. I should use the text_to_image skill to create a visually appealing image that matches their vision. Let me proceed with the generation.",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        ReActStep(
            thought="The image has been generated successfully. Now I need to evaluate its quality to ensure it meets professional standards. I'll check for visual coherence, artistic quality, and how well it matches the original prompt. Let me run the quality assessment.",
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
        ReActStep(
            thought="Good, the image generation is complete. Before presenting it to the user, I should verify the quality. I'll evaluate the aesthetics, technical quality, and prompt alignment to make sure we're delivering the best result possible.",
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        ReActStep(
            thought="The quality evaluation shows the image could be improved. The score of {score}% is below our quality threshold. I'll use the repair_image skill to enhance the visual quality, fix any artifacts, and improve overall aesthetics. This should help meet our quality standards.",
            action=ActionType.REPAIR,
            action_input={"skill": "repair_image", "params": {"image_url": "{image_url}", "prompt": "Improve overall quality, enhance details, and fix any visual artifacts"}},
        ),
        ReActStep(
            thought="The evaluation indicates room for improvement with a score of {score}%. I should attempt to repair and enhance the image. The repair skill can fix common issues like artifacts, improve sharpness, and enhance the overall composition.",
            action=ActionType.REPAIR,
            action_input={"skill": "repair_image", "params": {"image_url": "{image_url}", "prompt": "Enhance visual clarity, improve composition, and refine details"}},
        ),
    ],
    "finish_success": [
        ReActStep(
            thought="Excellent! The image has passed our quality assessment with a score of {score}%. The visual quality is good, and it accurately represents what the user requested. I'm confident this meets their expectations, so I'll present the final result.",
            action=ActionType.FINISH,
            action_input={"result": "success", "image_url": "{image_url}", "message": "Your image has been generated successfully!"},
        ),
        ReActStep(
            thought="The quality evaluation confirms this is a high-quality result with {score}% score. The image captures the essence of the user's request and meets our quality standards. Time to deliver the completed work to the user.",
            action=ActionType.FINISH,
            action_input={"result": "success", "image_url": "{image_url}", "message": "Here's your generated image!"},
        ),
    ],
    "finish_failure": [
        ReActStep(
            thought="Unfortunately, after multiple attempts, I wasn't able to generate an image that meets our quality standards. The best score achieved was {score}%. I apologize to the user and suggest they try rephrasing their request or providing more specific details.",
            action=ActionType.FINISH,
            action_input={"result": "failure", "reason": "Quality threshold not met after multiple attempts", "message": "I apologize, but I couldn't generate an image that meets quality standards. Please try a different prompt."},
        ),
    ],
}


class LLMSimulator:
    """
    Simulates LLM responses for ReAct planning.
    
    Used for development and testing without requiring actual LLM API calls.
    Follows a predetermined flow: generate -> evaluate -> (repair if needed) -> finish
    
    When verbose_probability > 0, some steps randomly include lengthy analysis
    paragraphs that inflate the token count, which helps test memory compression.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        max_repair_attempts: int = 2,
        simulate_delay: bool = False,
        verbose_probability: float = 0.6,
        verbose_fragments_range: tuple = (2, 5),
    ):
        self.quality_threshold = quality_threshold
        self.max_repair_attempts = max_repair_attempts
        self.simulate_delay = simulate_delay
        self.verbose_probability = verbose_probability
        self.verbose_fragments_range = verbose_fragments_range
        
        # State tracking
        self._step_count = 0
        self._repair_count = 0
        self._current_image_url: Optional[str] = None
        self._last_quality_score: Optional[float] = None
    
    def reset(self):
        """Reset simulator state for a new conversation."""
        self._step_count = 0
        self._repair_count = 0
        self._current_image_url = None
        self._last_quality_score = None

    def _maybe_pad_thought(self, thought: str) -> str:
        """Randomly pad a thought with verbose analysis fragments.
        
        This inflates the token count so that memory compression
        is triggered after several rounds of conversation.
        """
        if random.random() >= self.verbose_probability:
            return thought
        
        lo, hi = self.verbose_fragments_range
        n = random.randint(lo, hi)
        fragments = random.sample(
            VERBOSE_THOUGHT_FRAGMENTS,
            min(n, len(VERBOSE_THOUGHT_FRAGMENTS)),
        )
        padded = thought + "\n\n" + "\n\n".join(fragments)
        return padded
    
    def _interpolate_template(
        self,
        step: ReActStep,
        context: Dict[str, Any],
    ) -> ReActStep:
        """Replace placeholders in template with actual values,
        and optionally pad the thought with verbose fragments."""
        thought = step.thought
        
        # Deep copy and replace placeholders manually
        def replace_placeholders(obj: Any, ctx: Dict[str, Any]) -> Any:
            if isinstance(obj, str):
                result = obj
                for key, value in ctx.items():
                    result = result.replace(f"{{{key}}}", str(value))
                return result
            elif isinstance(obj, dict):
                return {k: replace_placeholders(v, ctx) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item, ctx) for item in obj]
            return obj
        
        action_input = replace_placeholders(step.action_input, context) if context else step.action_input
        
        # Apply verbose padding with randomness
        thought = self._maybe_pad_thought(thought)
        
        return ReActStep(
            thought=thought,
            action=step.action,
            action_input=action_input,
        )
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from user message for thought generation."""
        words = text.lower().split()
        # Simple heuristic: use first few meaningful words
        stop_words = {"a", "an", "the", "create", "generate", "make", "draw", "paint", "image", "picture", "of", "with", "and", "or"}
        meaningful = [w for w in words if w not in stop_words][:3]
        return " ".join(meaningful) if meaningful else "the requested subject"
    
    def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> ReActStep:
        """
        Get the next ReAct step based on current state and observation.
        
        Args:
            user_message: The original user request.
            observation: Result from the previous action (skill execution).
            
        Returns:
            ReActStep with thought, action, and action_input.
        """
        self._step_count += 1
        topic = self._extract_topic(user_message)
        
        # First step: Generate image
        if self._step_count == 1:
            context = {"prompt": user_message, "topic": topic}
            templates = RESPONSE_TEMPLATES["text_to_image"]
            template = random.choice(templates)
            return self._interpolate_template(template, context)
        
        # After generation: Evaluate
        if observation and "image_url" in observation.get("result", {}):
            self._current_image_url = observation["result"]["image_url"]
        
        if self._step_count == 2 and self._current_image_url:
            context = {"image_url": self._current_image_url}
            templates = RESPONSE_TEMPLATES["evaluate"]
            template = random.choice(templates)
            return self._interpolate_template(template, context)
        
        # After evaluation: Check quality and decide
        if observation and "overall_score" in observation.get("result", {}):
            self._last_quality_score = observation["result"]["overall_score"]
            score_percent = int(self._last_quality_score * 100)
            
            # Good quality: Finish
            if self._last_quality_score >= self.quality_threshold:
                context = {"image_url": self._current_image_url, "score": str(score_percent)}
                templates = RESPONSE_TEMPLATES["finish_success"]
                template = random.choice(templates)
                return self._interpolate_template(template, context)
            
            # Poor quality: Try repair if attempts remaining
            if self._repair_count < self.max_repair_attempts:
                self._repair_count += 1
                context = {
                    "image_url": self._current_image_url,
                    "score": str(score_percent),
                }
                templates = RESPONSE_TEMPLATES["repair"]
                template = random.choice(templates)
                return self._interpolate_template(template, context)
            
            # Max repairs reached: Finish with failure
            context = {"score": str(score_percent)}
            templates = RESPONSE_TEMPLATES["finish_failure"]
            template = random.choice(templates)
            return self._interpolate_template(template, context)
        
        # After repair: Re-evaluate
        if observation and observation.get("skill_name") == "repair_image":
            if "image_url" in observation.get("result", {}):
                self._current_image_url = observation["result"]["image_url"]
            
            context = {"image_url": self._current_image_url}
            templates = RESPONSE_TEMPLATES["evaluate"]
            template = random.choice(templates)
            return self._interpolate_template(template, context)
        
        # Fallback: Finish
        context = {"image_url": self._current_image_url or "", "score": "85"}
        templates = RESPONSE_TEMPLATES["finish_success"]
        template = random.choice(templates)
        return self._interpolate_template(template, context)
    
    async def get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream the next ReAct step token by token.
        
        Used for SSE streaming to the frontend.
        """
        import asyncio
        
        step = self.get_next_step(user_message, observation)
        
        # Stream thought
        yield "thought:"
        for word in step.thought.split():
            if self.simulate_delay:
                await asyncio.sleep(0.05)
            yield f" {word}"
        yield "\n"
        
        # Stream action
        yield f"action: {step.action.value}\n"
        
        # Stream action_input
        yield f"action_input: {json.dumps(step.action_input)}\n"
    
    def format_step_as_dict(self, step: ReActStep) -> Dict[str, Any]:
        """Convert ReActStep to dictionary for JSON serialization."""
        return {
            "thought": step.thought,
            "action": step.action.value,
            "action_input": step.action_input,
        }


class MockLLMClient:
    """
    Mock LLM client that wraps the simulator.
    
    Provides an interface similar to real LLM clients (OpenAI, Anthropic).
    """
    
    def __init__(self, simulator: Optional[LLMSimulator] = None):
        self.simulator = simulator or LLMSimulator()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Returns:
            Response dict with 'choices' containing the generated content.
        """
        # Extract user message and observation from messages
        user_message = ""
        observation = None
        
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "system" and "observation" in msg.get("content", "").lower():
                try:
                    observation = json.loads(msg["content"].split("Observation:")[-1].strip())
                except (json.JSONDecodeError, IndexError):
                    pass
        
        step = self.simulator.get_next_step(user_message, observation)
        content = self.simulator.format_step_as_dict(step)
        
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
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming chat completion response.
        
        Yields chunks of the response for SSE streaming.
        """
        user_message = ""
        observation = None
        
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "system" and "observation" in msg.get("content", "").lower():
                try:
                    observation = json.loads(msg["content"].split("Observation:")[-1].strip())
                except (json.JSONDecodeError, IndexError):
                    pass
        
        async for chunk in self.simulator.get_next_step_stream(user_message, observation):
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
