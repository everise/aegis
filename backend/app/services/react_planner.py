"""
ReAct planner engine for agent reasoning.

Implements the Thought -> Action -> Observation cycle for
iterative task completion with skill execution.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator
from enum import Enum

from app.providers.base import BaseProvider, ActionType, PlanningStep
from app.providers.registry import get_provider_registry
from app.services.dual_retrieval import get_knowledge_base


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning process."""
    thought: str
    action: ActionType
    action_input: Dict[str, Any]


class PlanStatus(str, Enum):
    """Status of the planning process."""
    THINKING = "thinking"
    EXECUTING = "executing"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Optional[Dict[str, Any]] = None
    status: PlanStatus = PlanStatus.THINKING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan with all steps."""
    session_id: Optional[int] = None
    user_message: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.THINKING
    final_result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.steps.append(step)
    
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current (last) step."""
        return self.steps[-1] if self.steps else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "user_message": self.user_message,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "status": s.status.value,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "status": self.status.value,
            "final_result": self.final_result,
        }


class ReActPlanner:
    """
    ReAct planning engine for iterative task execution.
    
    Follows the cycle:
    1. Think: LLM reasons about current state
    2. Act: Execute a skill based on reasoning
    3. Observe: Record skill result
    4. Repeat until task complete or max steps reached
    """
    
    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        max_steps: int = 10,
        enable_retrieval: bool = True,
        # ── Backward-compat kwargs (ignored if provider is given) ──
        skill_executor=None,
        planning_model=None,
        use_mock: bool = True,
    ):
        # New path: accept a unified provider
        if provider is not None:
            self.provider = provider
        elif planning_model is not None:
            # Backward compat: treat a planning model AS a provider
            # (works because the old planning models share the interface)
            self.provider = planning_model
        else:
            registry = get_provider_registry()
            self.provider = registry.get_active_provider()

        # Keep a reference under the old name for _build_token_usage_event
        self.planning_model = self.provider

        # Legacy skill_executor is no longer needed — the provider
        # handles skill execution.  We keep the attribute only so that
        # close() and _build_token_usage_event() don't break.
        self.skill_executor = skill_executor

        self.max_steps = max_steps
        self.enable_retrieval = enable_retrieval
        print(
            f"[DEBUG] ReActPlanner initialized with provider: "
            f"{self.provider.info().name}  "
            f"(id={self.provider.info().id})"
        )
    
    async def _retrieve_knowledge(self, user_message: str) -> str:
        """Retrieve relevant domain knowledge using Dual-Level Retrieval.

        Uses BM25 + ChromaDB semantic search with RRF fusion to find
        the most relevant prompting / style / quality knowledge.
        """
        if not self.enable_retrieval:
            return ""
        try:
            kb = get_knowledge_base()
            augmented = await kb.get_augmented_context(user_message, top_k=3)
            return augmented
        except Exception as e:
            print(f"[WARN] Dual-Level Retrieval failed: {e}")
            return ""

    async def _get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Stream the next reasoning step from the provider.

        Yields raw token deltas.  The final yield is a sentinel
        ``\\x00`` + JSON with the parsed ReAct step.
        """
        async for token in self.provider.get_next_step_stream(user_message, observation):
            yield token

    async def _get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> ReActStep:
        """Get the next reasoning step (non-streaming convenience wrapper).

        Internally consumes the streaming helper and parses the final
        sentinel to return a complete ``ReActStep``.
        """
        parsed: Optional[dict] = None
        async for token in self._get_next_step_stream(user_message, observation):
            if token.startswith("\x00"):
                parsed = json.loads(token[1:])
        if parsed is None:
            raise RuntimeError("Planning model stream ended without a parsed step")
        return ReActStep(
            thought=parsed.get("thought", ""),
            action=ActionType(parsed.get("action", "finish")),
            action_input=parsed.get("action_input", {}),
        )
    
    async def _execute_action(
        self,
        action: ActionType,
        action_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an action via the active provider and return the observation."""
        if action == ActionType.FINISH:
            return {"finished": True, **action_input}
        
        skill_name = action_input.get("skill")
        params = action_input.get("params", {})
        
        if not skill_name:
            return {"error": "No skill specified in action_input"}
        
        raw = await self.provider.execute_skill(skill_name, params)
        
        return {
            "skill_name": skill_name,
            "status": raw.get("status", "failed"),
            "result": raw.get("result"),
            "error": raw.get("error"),
        }
    
    async def execute(
        self,
        user_message: str,
        session_id: Optional[int] = None,
    ) -> ExecutionPlan:
        """
        Execute a complete planning cycle for the user message.
        
        Args:
            user_message: The user's request.
            session_id: Optional session ID for tracking.
            
        Returns:
            ExecutionPlan with all steps and final result.
        """
        plan = ExecutionPlan(
            session_id=session_id,
            user_message=user_message,
        )
        
        # Reset provider state for new conversation
        self.provider.reset()
        
        # Dual-Level Retrieval: augment with domain knowledge
        retrieved_knowledge = await self._retrieve_knowledge(user_message)
        if retrieved_knowledge:
            augmented_message = f"{retrieved_knowledge}\n\n{user_message}"
        else:
            augmented_message = user_message
        
        observation: Optional[Dict[str, Any]] = None
        step_number = 0
        
        while step_number < self.max_steps:
            step_number += 1
            
            # Think: Get next step from planning model (with retrieved knowledge)
            try:
                react_step = await self._get_next_step(augmented_message, observation)
            except Exception as e:
                plan.status = PlanStatus.FAILED
                plan.steps.append(PlanStep(
                    step_number=step_number,
                    thought="",
                    action="error",
                    action_input={},
                    error=f"LLM error: {str(e)}",
                    status=PlanStatus.FAILED,
                ))
                break
            
            # Create plan step
            plan_step = PlanStep(
                step_number=step_number,
                thought=react_step.thought,
                action=react_step.action.value,
                action_input=react_step.action_input,
                status=PlanStatus.EXECUTING,
                started_at=datetime.utcnow(),
            )
            plan.add_step(plan_step)
            
            # Check for finish action
            if react_step.action == ActionType.FINISH:
                plan_step.status = PlanStatus.COMPLETED
                plan_step.completed_at = datetime.utcnow()
                plan_step.observation = react_step.action_input
                
                plan.status = PlanStatus.COMPLETED
                plan.final_result = react_step.action_input
                plan.completed_at = datetime.utcnow()
                break
            
            # Act: Execute the skill
            try:
                observation = await self._execute_action(
                    react_step.action,
                    react_step.action_input,
                )
                plan_step.observation = observation
                plan_step.status = PlanStatus.OBSERVING
                plan_step.completed_at = datetime.utcnow()
                
                # Check for execution errors
                if observation.get("error"):
                    plan_step.error = observation["error"]
                    if observation.get("status") == "failed":
                        plan.status = PlanStatus.FAILED
                        break
                        
            except Exception as e:
                plan_step.status = PlanStatus.FAILED
                plan_step.error = f"Execution error: {str(e)}"
                plan_step.completed_at = datetime.utcnow()
                plan.status = PlanStatus.FAILED
                break
        
        # Check if max steps reached without completion
        if step_number >= self.max_steps and plan.status != PlanStatus.COMPLETED:
            plan.status = PlanStatus.FAILED
            if plan.steps:
                plan.steps[-1].error = "Max steps reached without completion"
        
        return plan
    
    async def execute_stream(
        self,
        user_message: str,
        session_id: Optional[int] = None,
        *,
        image_config: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute planning with streaming updates.
        
        Yields status updates for each step of the planning process.
        Used for real-time UI updates via SSE.

        *image_config* (optional): ``{"aspect_ratio": "...", "image_size": "..."}``
        overrides that are merged into ``text_to_image`` skill params so
        the user’s UI selections are always applied.
        """
        import random
        
        plan = ExecutionPlan(
            session_id=session_id,
            user_message=user_message,
        )
        
        if self.provider is not None:
            self.provider.reset()
        
        # Dual-Level Retrieval: augment with domain knowledge
        retrieved_knowledge = await self._retrieve_knowledge(user_message)
        if retrieved_knowledge:
            augmented_message = f"{retrieved_knowledge}\n\n{user_message}"
        else:
            augmented_message = user_message
        
        yield {
            "type": "plan_started",
            "data": {
                "user_message": user_message,
                "session_id": session_id,
                "knowledge_retrieved": bool(retrieved_knowledge),
            },
        }
        
        # If retrieval found something, emit it
        if retrieved_knowledge:
            yield {
                "type": "knowledge_retrieved",
                "data": {"context": retrieved_knowledge},
            }
        
        # Initial delay to simulate receiving and processing request
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        observation: Optional[Dict[str, Any]] = None
        step_number = 0
        
        while step_number < self.max_steps:
            step_number += 1
            
            # Thinking phase - simulate LLM thinking time
            yield {
                "type": "thinking",
                "data": {"step_number": step_number},
            }
            
            # Simulate thinking delay (3-5 seconds)
            await asyncio.sleep(random.uniform(3.0, 5.0))
            
            try:
                react_step = await self._get_next_step(augmented_message, observation)
            except Exception as e:
                yield {
                    "type": "error",
                    "data": {"step_number": step_number, "error": str(e)},
                }
                break
            
            # Small delay before showing thought result
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            yield {
                "type": "thought",
                "data": {
                    "step_number": step_number,
                    "thought": react_step.thought,
                    "action": react_step.action.value,
                    "action_input": react_step.action_input,
                },
            }
            
            # Check finish
            if react_step.action == ActionType.FINISH:
                await asyncio.sleep(random.uniform(0.5, 1.0))
                yield {
                    "type": "finished",
                    "data": {"result": react_step.action_input},
                }
                break
            
            # Executing phase - delay before starting execution
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            # Merge user image config into text_to_image params
            if image_config and react_step.action_input.get("skill") == "text_to_image":
                params = react_step.action_input.get("params", {})
                params.setdefault("aspect_ratio", image_config.get("aspect_ratio", "1:1"))
                params.setdefault("image_size", image_config.get("image_size", "1K"))
                # Also override if the LLM picked different values
                params["aspect_ratio"] = image_config.get("aspect_ratio", params.get("aspect_ratio", "1:1"))
                params["image_size"] = image_config.get("image_size", params.get("image_size", "1K"))
                react_step.action_input["params"] = params
            
            yield {
                "type": "executing",
                "data": {
                    "step_number": step_number,
                    "skill": react_step.action_input.get("skill"),
                },
            }
            
            try:
                observation = await self._execute_action(
                    react_step.action,
                    react_step.action_input,
                )
                
                # Small delay before showing observation
                await asyncio.sleep(random.uniform(0.3, 0.8))
                
                yield {
                    "type": "observation",
                    "data": {
                        "step_number": step_number,
                        "observation": observation,
                    },
                }

                # Emit actual API token usage after each action
                yield self._build_token_usage_event()
                
                if observation.get("error") and observation.get("status") == "failed":
                    yield {
                        "type": "error",
                        "data": {"step_number": step_number, "error": observation["error"]},
                    }
                    break
                    
            except Exception as e:
                yield {
                    "type": "error",
                    "data": {"step_number": step_number, "error": str(e)},
                }
                break
        
        if step_number >= self.max_steps:
            yield {
                "type": "max_steps_reached",
                "data": {"max_steps": self.max_steps},
            }
    
    async def close(self):
        """Clean up resources."""
        await self.provider.close()

    def _build_token_usage_event(self) -> Dict[str, Any]:
        """Build an SSE event with accumulated actual API token usage.

        Combines tokens from the provider's planning calls and skill
        execution calls.
        """
        planning_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        skill_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if hasattr(self.provider, "planning_token_usage"):
            planning_usage = self.provider.planning_token_usage

        if hasattr(self.provider, "skill_token_usage"):
            skill_usage = self.provider.skill_token_usage

        return {
            "type": "api_token_usage",
            "data": {
                "planning": planning_usage,
                "skills": skill_usage,
                "total": {
                    "prompt_tokens": planning_usage["prompt_tokens"] + skill_usage["prompt_tokens"],
                    "completion_tokens": planning_usage["completion_tokens"] + skill_usage["completion_tokens"],
                    "total_tokens": planning_usage["total_tokens"] + skill_usage["total_tokens"],
                },
            },
        }
