"""
Skill executor for HTTP API-based skills.

Implements the submit-poll pattern for async skill executions.
Manages skill lifecycle and result handling.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Type
from enum import Enum

import httpx
from pydantic import BaseModel

from app.config import get_settings


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
    
    def list_skills(self) -> list[Dict[str, str]]:
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
