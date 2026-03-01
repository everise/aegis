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

from app.services.skill_executor import SkillExecutor, MockSkillExecutor, SkillResult, SkillStatus
from app.services.planning.base import BasePlanningModel, ActionType, PlanningStep
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
        skill_executor: Optional[SkillExecutor] = None,
        planning_model: Optional[BasePlanningModel] = None,
        max_steps: int = 10,
        use_mock: bool = True,
        enable_retrieval: bool = True,
    ):
        # Use mock executor by default for development
        if use_mock and skill_executor is None:
            self.skill_executor = MockSkillExecutor()
        elif skill_executor is not None:
            self.skill_executor = skill_executor
        else:
            self.skill_executor = SkillExecutor()
        
        self.planning_model = planning_model
        self.max_steps = max_steps
        self.enable_retrieval = enable_retrieval
        if planning_model:
            print(f"[DEBUG] ReActPlanner initialized with planning_model: {planning_model.info().name}")
        else:
            print(f"[DEBUG] ReActPlanner initialized with skill_executor: {type(self.skill_executor).__name__}")
    
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

    async def _get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> ReActStep:
        """Get the next reasoning step from the planning model."""
        if self.planning_model is None:
            raise RuntimeError("No planning model configured for ReActPlanner")

        step = await self.planning_model.get_next_step(user_message, observation)
        return ReActStep(
            thought=step.thought,
            action=ActionType(step.action.value),
            action_input=step.action_input,
        )
    
    async def _execute_action(
        self,
        action: ActionType,
        action_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an action and return the observation."""
        if action == ActionType.FINISH:
            return {"finished": True, **action_input}
        
        skill_name = action_input.get("skill")
        params = action_input.get("params", {})
        
        if not skill_name:
            return {"error": "No skill specified in action_input"}
        
        result = await self.skill_executor.execute(skill_name, params)
        
        return {
            "skill_name": skill_name,
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
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
        
        # Reset planning model state for new conversation
        if self.planning_model is not None:
            self.planning_model.reset()
        
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
                    if observation.get("status") == SkillStatus.FAILED.value:
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
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute planning with streaming updates.
        
        Yields status updates for each step of the planning process.
        Used for real-time UI updates via SSE.
        """
        import random
        
        plan = ExecutionPlan(
            session_id=session_id,
            user_message=user_message,
        )
        
        if self.planning_model is not None:
            self.planning_model.reset()
        
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
                
                if observation.get("error") and observation.get("status") == SkillStatus.FAILED.value:
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
        await self.skill_executor.close()
