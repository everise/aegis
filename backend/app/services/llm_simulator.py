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


# Predefined response templates for different scenarios
RESPONSE_TEMPLATES = {
    "text_to_image": [
        ReActStep(
            thought="The user wants to generate an image. I should use the text_to_image skill with their prompt.",
            action=ActionType.GENERATE,
            action_input={"skill": "text_to_image", "params": {"prompt": "{prompt}"}},
        ),
    ],
    "evaluate": [
        ReActStep(
            thought="I need to evaluate the quality of the generated image to ensure it meets standards.",
            action=ActionType.EVALUATE,
            action_input={"skill": "evaluate_image", "params": {"image_url": "{image_url}"}},
        ),
    ],
    "repair": [
        ReActStep(
            thought="The image quality is below threshold. I should repair it to improve quality.",
            action=ActionType.REPAIR,
            action_input={"skill": "repair_image", "params": {"image_url": "{image_url}", "prompt": "{repair_prompt}"}},
        ),
    ],
    "finish_success": [
        ReActStep(
            thought="The image generation is complete and meets quality standards. I will present the result to the user.",
            action=ActionType.FINISH,
            action_input={"result": "success", "image_url": "{image_url}"},
        ),
    ],
    "finish_failure": [
        ReActStep(
            thought="After multiple attempts, I was unable to generate an image that meets the quality threshold.",
            action=ActionType.FINISH,
            action_input={"result": "failure", "reason": "Quality threshold not met"},
        ),
    ],
}


class LLMSimulator:
    """
    Simulates LLM responses for ReAct planning.
    
    Used for development and testing without requiring actual LLM API calls.
    Follows a predetermined flow: generate -> evaluate -> (repair if needed) -> finish
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        max_repair_attempts: int = 2,
        simulate_delay: bool = False,
    ):
        self.quality_threshold = quality_threshold
        self.max_repair_attempts = max_repair_attempts
        self.simulate_delay = simulate_delay
        
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
    
    def _interpolate_template(
        self,
        step: ReActStep,
        context: Dict[str, Any],
    ) -> ReActStep:
        """Replace placeholders in template with actual values."""
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
        
        return ReActStep(
            thought=thought,
            action=step.action,
            action_input=action_input,
        )
    
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
        
        # First step: Generate image
        if self._step_count == 1:
            context = {"prompt": user_message}
            template = RESPONSE_TEMPLATES["text_to_image"][0]
            return self._interpolate_template(template, context)
        
        # After generation: Evaluate
        if observation and "image_url" in observation.get("result", {}):
            self._current_image_url = observation["result"]["image_url"]
        
        if self._step_count == 2 and self._current_image_url:
            context = {"image_url": self._current_image_url}
            template = RESPONSE_TEMPLATES["evaluate"][0]
            return self._interpolate_template(template, context)
        
        # After evaluation: Check quality and decide
        if observation and "overall_score" in observation.get("result", {}):
            self._last_quality_score = observation["result"]["overall_score"]
            
            # Good quality: Finish
            if self._last_quality_score >= self.quality_threshold:
                context = {"image_url": self._current_image_url}
                template = RESPONSE_TEMPLATES["finish_success"][0]
                return self._interpolate_template(template, context)
            
            # Poor quality: Try repair if attempts remaining
            if self._repair_count < self.max_repair_attempts:
                self._repair_count += 1
                context = {
                    "image_url": self._current_image_url,
                    "repair_prompt": "Improve overall quality and fix artifacts",
                }
                template = RESPONSE_TEMPLATES["repair"][0]
                return self._interpolate_template(template, context)
            
            # Max repairs reached: Finish with failure
            template = RESPONSE_TEMPLATES["finish_failure"][0]
            return self._interpolate_template(template, {})
        
        # After repair: Re-evaluate
        if observation and observation.get("skill_name") == "repair_image":
            if "image_url" in observation.get("result", {}):
                self._current_image_url = observation["result"]["image_url"]
            
            context = {"image_url": self._current_image_url}
            template = RESPONSE_TEMPLATES["evaluate"][0]
            return self._interpolate_template(template, context)
        
        # Fallback: Finish
        context = {"image_url": self._current_image_url or ""}
        template = RESPONSE_TEMPLATES["finish_success"][0]
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
