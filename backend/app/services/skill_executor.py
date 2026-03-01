"""
Skill executor for HTTP API-based skills.

Implements the submit-poll pattern for async skill executions.
Manages skill lifecycle and result handling.

Also provides ``OpenRouterSkillExecutor`` which delegates to the
real OpenRouter skill implementations (text_to_image, evaluate_image,
repair_image) instead of simulating via the mock remote API.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from enum import Enum

import httpx
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger("aegis.skill_executor")


class SkillStatus(str, Enum):
    """Status of a skill execution."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    POLLING = "polling"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SkillResult(BaseModel):
    """Result of a skill execution."""
    skill_name: str
    status: SkillStatus
    task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    poll_count: int = 0
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BaseSkill(ABC):
    """
    Abstract base class for all skills.
    
    Skills follow the submit-poll pattern:
    1. Submit task to remote API
    2. Poll for completion
    3. Return result
    """
    
    name: str = "base_skill"
    description: str = "Base skill"
    submit_endpoint: str = ""
    poll_endpoint: str = "/tasks/{task_id}/poll"
    
    # Polling configuration
    poll_interval: float = 1.0  # seconds
    max_poll_attempts: int = 60
    timeout: float = 300.0  # seconds
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.settings = get_settings()
        self._http_client = http_client
        self._owns_client = False
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.remote_api_base_url,
                timeout=self.timeout,
            )
            self._owns_client = True
        return self._http_client
    
    async def close(self):
        """Close HTTP client if owned."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize input parameters."""
        pass
    
    @abstractmethod
    def format_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the raw API result for output."""
        pass
    
    async def submit(self, params: Dict[str, Any]) -> str:
        """
        Submit task to remote API.
        
        Args:
            params: Validated parameters for the skill.
            
        Returns:
            Task ID from remote API.
        """
        client = await self.get_http_client()
        
        response = await client.post(
            self.submit_endpoint,
            json=params,
        )
        response.raise_for_status()
        
        data = response.json()
        return data["task_id"]
    
    async def poll(self, task_id: str) -> Dict[str, Any]:
        """
        Poll task status from remote API.
        
        Args:
            task_id: Task ID to poll.
            
        Returns:
            Task status and result if completed.
        """
        client = await self.get_http_client()
        
        endpoint = self.poll_endpoint.format(task_id=task_id)
        response = await client.get(endpoint)
        response.raise_for_status()
        
        return response.json()
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute the skill with submit-poll pattern.
        
        Args:
            params: Raw input parameters.
            
        Returns:
            SkillResult with status and result/error.
        """
        result = SkillResult(
            skill_name=self.name,
            status=SkillStatus.PENDING,
        )
        
        try:
            # Validate parameters
            validated_params = self.validate_params(params)
            
            # Submit task
            result.status = SkillStatus.SUBMITTED
            result.submitted_at = datetime.utcnow()
            task_id = await self.submit(validated_params)
            result.task_id = task_id
            
            # Poll for completion
            result.status = SkillStatus.POLLING
            poll_count = 0
            
            while poll_count < self.max_poll_attempts:
                poll_response = await self.poll(task_id)
                poll_count += 1
                result.poll_count = poll_count
                
                status = poll_response.get("status", "").lower()
                
                if status == "completed":
                    result.status = SkillStatus.COMPLETED
                    result.completed_at = datetime.utcnow()
                    result.result = self.format_result(poll_response.get("result", {}))
                    return result
                
                elif status == "failed":
                    result.status = SkillStatus.FAILED
                    result.completed_at = datetime.utcnow()
                    result.error = poll_response.get("error", "Task failed")
                    return result
                
                # Still processing, wait and retry
                await asyncio.sleep(self.poll_interval)
            
            # Max attempts reached
            result.status = SkillStatus.TIMEOUT
            result.error = f"Polling timeout after {self.max_poll_attempts} attempts"
            
        except httpx.HTTPStatusError as e:
            result.status = SkillStatus.FAILED
            result.error = f"HTTP error: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            result.status = SkillStatus.FAILED
            result.error = f"Request error: {str(e)}"
        except Exception as e:
            result.status = SkillStatus.FAILED
            result.error = f"Unexpected error: {str(e)}"
        
        return result


class TextToImageSkill(BaseSkill):
    """Skill for text-to-image generation."""
    
    name = "text_to_image"
    description = "Generate an image from a text prompt"
    submit_endpoint = "/text-to-image/submit"
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text-to-image parameters."""
        if "prompt" not in params or not params["prompt"]:
            raise ValueError("prompt is required")
        
        return {
            "prompt": str(params["prompt"]),
            "negative_prompt": params.get("negative_prompt"),
            "width": int(params.get("width", 512)),
            "height": int(params.get("height", 512)),
            "steps": int(params.get("steps", 20)),
            "seed": params.get("seed"),
        }
    
    def format_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format text-to-image result."""
        return {
            "image_url": raw_result.get("image_url"),
            "width": raw_result.get("width"),
            "height": raw_result.get("height"),
            "seed": raw_result.get("seed"),
        }


class EvaluateImageSkill(BaseSkill):
    """Skill for evaluating image quality."""
    
    name = "evaluate_image"
    description = "Evaluate the quality and aesthetics of an image"
    submit_endpoint = "/evaluate-image/submit"
    poll_interval = 0.5  # Faster polling for evaluation
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evaluate-image parameters."""
        if "image_url" not in params or not params["image_url"]:
            raise ValueError("image_url is required")
        
        return {
            "image_url": str(params["image_url"]),
            "criteria": params.get("criteria", ["quality", "aesthetics", "prompt_alignment"]),
        }
    
    def format_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format evaluation result."""
        return {
            "scores": raw_result.get("scores", {}),
            "overall_score": raw_result.get("overall_score"),
            "feedback": raw_result.get("feedback"),
        }


class RepairImageSkill(BaseSkill):
    """Skill for repairing/inpainting images."""
    
    name = "repair_image"
    description = "Repair or inpaint parts of an image"
    submit_endpoint = "/repair-image/submit"
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate repair-image parameters."""
        if "image_url" not in params or not params["image_url"]:
            raise ValueError("image_url is required")
        if "prompt" not in params or not params["prompt"]:
            raise ValueError("prompt is required")
        
        return {
            "image_url": str(params["image_url"]),
            "mask_url": params.get("mask_url"),
            "prompt": str(params["prompt"]),
            "strength": float(params.get("strength", 0.75)),
        }
    
    def format_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format repair result."""
        return {
            "image_url": raw_result.get("image_url"),
            "original_url": raw_result.get("original_url"),
        }


# Skill registry
SKILL_REGISTRY: Dict[str, Type[BaseSkill]] = {
    "text_to_image": TextToImageSkill,
    "evaluate_image": EvaluateImageSkill,
    "repair_image": RepairImageSkill,
}


class SkillExecutor:
    """
    Manages skill execution lifecycle.
    
    Provides a unified interface for executing any registered skill.
    """
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self._http_client = http_client
        self._skill_instances: Dict[str, BaseSkill] = {}
    
    def get_skill(self, skill_name: str) -> BaseSkill:
        """Get or create a skill instance."""
        if skill_name not in SKILL_REGISTRY:
            raise ValueError(f"Unknown skill: {skill_name}")
        
        if skill_name not in self._skill_instances:
            skill_class = SKILL_REGISTRY[skill_name]
            self._skill_instances[skill_name] = skill_class(self._http_client)
        
        return self._skill_instances[skill_name]
    
    async def execute(self, skill_name: str, params: Dict[str, Any]) -> SkillResult:
        """
        Execute a skill by name.
        
        Args:
            skill_name: Name of the skill to execute.
            params: Parameters for the skill.
            
        Returns:
            SkillResult with execution status and result.
        """
        skill = self.get_skill(skill_name)
        return await skill.execute(params)
    
    def list_skills(self) -> List[Dict[str, str]]:
        """List all available skills."""
        return [
            {"name": name, "description": cls.description}
            for name, cls in SKILL_REGISTRY.items()
        ]
    
    async def close(self):
        """Close all skill instances."""
        for skill in self._skill_instances.values():
            await skill.close()
        self._skill_instances.clear()


class MockSkillExecutor:
    """
    Mock skill executor for development and testing.
    
    Returns simulated results without making actual API calls.
    """
    
    def __init__(self):
        self._step_count = 0
    
    async def execute(self, skill_name: str, params: Dict[str, Any]) -> SkillResult:
        """
        Execute a mock skill.
        
        Returns simulated results based on skill type.
        Includes realistic delays to simulate actual processing time.
        """
        import random
        
        self._step_count += 1
        
        # Simulate realistic processing time (3-6 seconds per skill)
        await asyncio.sleep(random.uniform(3.0, 6.0))
        
        if skill_name == "text_to_image":
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.COMPLETED,
                task_id=f"mock-task-{self._step_count}",
                result={
                    "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/512/512",
                    "width": 512,
                    "height": 512,
                    "seed": random.randint(1, 999999),
                },
                completed_at=datetime.utcnow(),
            )
        
        elif skill_name == "evaluate_image":
            # Generate random quality score
            score = random.uniform(0.6, 0.95)
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.COMPLETED,
                task_id=f"mock-task-{self._step_count}",
                result={
                    "scores": {
                        "quality": score,
                        "aesthetics": score + random.uniform(-0.1, 0.1),
                        "prompt_alignment": score + random.uniform(-0.1, 0.1),
                    },
                    "overall_score": score,
                    "feedback": "Image looks good" if score >= 0.7 else "Image needs improvement",
                },
                completed_at=datetime.utcnow(),
            )
        
        elif skill_name == "repair_image":
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.COMPLETED,
                task_id=f"mock-task-{self._step_count}",
                result={
                    "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/512/512",
                    "original_url": params.get("image_url"),
                },
                completed_at=datetime.utcnow(),
            )
        
        else:
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                error=f"Unknown skill: {skill_name}",
            )
    
    def list_skills(self) -> List[Dict[str, str]]:
        """List all available skills."""
        return [
            {"name": name, "description": cls.description}
            for name, cls in SKILL_REGISTRY.items()
        ]
    
    async def close(self):
        """No resources to close."""
        pass


class OpenRouterSkillExecutor:
    """Skill executor that delegates to real OpenRouter skill implementations.

    When the active planning model is OpenRouter, this executor is used
    instead of ``MockSkillExecutor`` or ``SkillExecutor`` so that
    text-to-image generation, image evaluation, and image repair all
    go through the OpenRouter API with the models configured in
    ``aegis.yaml → openrouter``.
    """

    def __init__(self) -> None:
        # Lazily instantiated on first call to avoid import-time side effects
        self._generator: Optional[Any] = None
        self._scorer: Optional[Any] = None
        self._repairer: Optional[Any] = None
        self._initialised = False
        # Accumulated actual API token usage across all skill calls
        self._accumulated_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @property
    def token_usage(self) -> Dict[str, int]:
        """Return accumulated actual API token usage for skill calls."""
        return dict(self._accumulated_usage)

    def _accumulate_usage(self, usage: Dict[str, Any]) -> None:
        """Add usage from an OpenRouter response to the running total."""
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = usage.get(key)
            if val is not None:
                self._accumulated_usage[key] += int(val)

    def _ensure_initialised(self) -> None:
        """Import and create the three OpenRouter skill backends."""
        if self._initialised:
            return

        from skills.text_to_image.scripts.openrouter import OpenRouterImageGenerator
        from skills.evaluate_image.scripts.openrouter import OpenRouterVLScorer
        from skills.repair_image.scripts.openrouter import OpenRouterImageRepairer

        self._generator = OpenRouterImageGenerator()
        self._scorer = OpenRouterVLScorer()
        self._repairer = OpenRouterImageRepairer()
        self._initialised = True
        logger.info("OpenRouterSkillExecutor initialised (generator + scorer + repairer)")

    async def execute(self, skill_name: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a skill via OpenRouter.

        Maps the canonical skill names (``text_to_image``,
        ``evaluate_image``, ``repair_image``) to the corresponding
        OpenRouter skill class and calls its async method.
        """
        self._ensure_initialised()

        try:
            if skill_name == "text_to_image":
                raw = await self._generator.generate(
                    prompt=params.get("prompt", ""),
                    aspect_ratio=params.get("aspect_ratio", "1:1"),
                    image_size=params.get("image_size", "1K"),
                    reference_image_url=params.get("reference_image_url"),
                )
                # Accumulate actual API token usage
                if raw.get("usage"):
                    self._accumulate_usage(raw["usage"])
                # Normalise to the SkillResult format expected by the planner
                error = raw.get("error")
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.FAILED if error else SkillStatus.COMPLETED,
                    task_id=None,
                    result={
                        "image_url": raw.get("image_url"),
                        "width": raw.get("width"),
                        "height": raw.get("height"),
                        "seed": None,
                    } if not error else None,
                    error=error,
                    completed_at=datetime.utcnow(),
                )

            elif skill_name == "evaluate_image":
                raw = await self._scorer.evaluate(
                    image_url=params.get("image_url", ""),
                    prompt=params.get("prompt", ""),
                    criteria=params.get("criteria"),
                )
                # Accumulate actual API token usage
                if raw.get("usage"):
                    self._accumulate_usage(raw["usage"])
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.COMPLETED,
                    task_id=None,
                    result={
                        "scores": raw.get("scores", {}),
                        "overall_score": raw.get("overall_score"),
                        "feedback": raw.get("feedback", ""),
                    },
                    completed_at=datetime.utcnow(),
                )

            elif skill_name == "repair_image":
                raw = await self._repairer.repair(
                    image_url=params.get("image_url", ""),
                    prompt=params.get("prompt", ""),
                    mask_url=params.get("mask_url"),
                    strength=float(params.get("strength", 0.75)),
                )
                # Accumulate actual API token usage
                if raw.get("usage"):
                    self._accumulate_usage(raw["usage"])
                error = raw.get("error")
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.FAILED if error else SkillStatus.COMPLETED,
                    task_id=None,
                    result={
                        "image_url": raw.get("image_url"),
                        "original_url": raw.get("original_url"),
                    } if not error else None,
                    error=error,
                    completed_at=datetime.utcnow(),
                )

            else:
                return SkillResult(
                    skill_name=skill_name,
                    status=SkillStatus.FAILED,
                    error=f"Unknown skill for OpenRouter executor: {skill_name}",
                )

        except Exception as exc:
            logger.exception("OpenRouterSkillExecutor error on skill=%s", skill_name)
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                error=f"OpenRouter skill error: {exc}",
                completed_at=datetime.utcnow(),
            )

    def list_skills(self) -> List[Dict[str, str]]:
        """List all available skills."""
        return [
            {"name": "text_to_image", "description": "Generate an image via OpenRouter"},
            {"name": "evaluate_image", "description": "Evaluate image quality via OpenRouter VL model"},
            {"name": "repair_image", "description": "Repair / improve an image via OpenRouter"},
        ]

    async def close(self) -> None:
        """Release HTTP resources held by the underlying clients."""
        for backend in (self._generator, self._scorer, self._repairer):
            if backend is not None:
                try:
                    await backend.close()
                except Exception:
                    pass
        self._initialised = False
