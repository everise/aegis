"""
Trajectory management for Multi-Turn RL training.

Trajectories are sequences of state-action-reward tuples from
agent-environment interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json


class StateType(str, Enum):
    """Types of states in the trajectory."""
    INITIAL = "initial"       # Starting state with user prompt
    INTERMEDIATE = "intermediate"  # After skill execution
    TERMINAL = "terminal"     # Final state


@dataclass
class State:
    """
    Represents a state in the trajectory.
    
    Contains all information needed to make a decision at this point.
    """
    state_type: StateType
    user_prompt: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_image_url: Optional[str] = None
    quality_scores: Optional[Dict[str, float]] = None
    step_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "state_type": self.state_type.value,
            "user_prompt": self.user_prompt,
            "conversation_history": self.conversation_history,
            "current_image_url": self.current_image_url,
            "quality_scores": self.quality_scores,
            "step_number": self.step_number,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create state from dictionary."""
        return cls(
            state_type=StateType(data["state_type"]),
            user_prompt=data["user_prompt"],
            conversation_history=data.get("conversation_history", []),
            current_image_url=data.get("current_image_url"),
            quality_scores=data.get("quality_scores"),
            step_number=data.get("step_number", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Action:
    """
    Represents an action taken by the agent.
    
    Contains the thought process and skill execution details.
    """
    thought: str
    action_type: str  # generate, evaluate, repair, finish
    skill_name: Optional[str] = None
    skill_params: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "thought": self.thought,
            "action_type": self.action_type,
            "skill_name": self.skill_name,
            "skill_params": self.skill_params,
            "raw_output": self.raw_output,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create action from dictionary."""
        return cls(
            thought=data["thought"],
            action_type=data["action_type"],
            skill_name=data.get("skill_name"),
            skill_params=data.get("skill_params", {}),
            raw_output=data.get("raw_output"),
        )


@dataclass
class Transition:
    """
    A single transition (s, a, r, s') in the trajectory.
    """
    state: State
    action: Action
    reward: float
    next_state: Optional[State] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary."""
        return {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward,
            "next_state": self.next_state.to_dict() if self.next_state else None,
            "done": self.done,
            "info": self.info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transition":
        """Create transition from dictionary."""
        return cls(
            state=State.from_dict(data["state"]),
            action=Action.from_dict(data["action"]),
            reward=data["reward"],
            next_state=State.from_dict(data["next_state"]) if data.get("next_state") else None,
            done=data.get("done", False),
            info=data.get("info", {}),
        )


@dataclass
class Trajectory:
    """
    A complete trajectory from a multi-turn interaction.
    
    Contains all transitions and computed returns.
    """
    trajectory_id: Optional[str] = None
    session_id: Optional[int] = None
    task_type: str = "text_to_image"
    policy_version: str = "v1.0"
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    discounted_return: float = 0.0
    discount_factor: float = 0.99
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the trajectory."""
        self.transitions.append(transition)
        self.total_reward += transition.reward
    
    def compute_returns(self) -> List[float]:
        """
        Compute discounted returns for each step.
        
        Returns:
            List of discounted returns G_t for each step.
        """
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for transition in reversed(self.transitions):
            G = transition.reward + self.discount_factor * G
            returns.insert(0, G)
        
        if returns:
            self.discounted_return = returns[0]
        
        return returns
    
    def compute_advantages(
        self,
        value_estimates: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute advantages A_t = G_t - V(s_t).
        
        Args:
            value_estimates: Optional value function estimates.
                            If None, uses Monte Carlo returns as advantage.
        
        Returns:
            List of advantages for each step.
        """
        returns = self.compute_returns()
        
        if value_estimates is None:
            # Use returns directly as advantages (no baseline)
            return returns
        
        # A_t = G_t - V(s_t)
        advantages = [
            G - V for G, V in zip(returns, value_estimates)
        ]
        
        return advantages
    
    @property
    def length(self) -> int:
        """Get trajectory length."""
        return len(self.transitions)
    
    @property
    def is_complete(self) -> bool:
        """Check if trajectory is complete (ends in terminal state)."""
        if not self.transitions:
            return False
        return self.transitions[-1].done
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "task_type": self.task_type,
            "policy_version": self.policy_version,
            "transitions": [t.to_dict() for t in self.transitions],
            "total_reward": self.total_reward,
            "discounted_return": self.discounted_return,
            "discount_factor": self.discount_factor,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create trajectory from dictionary."""
        trajectory = cls(
            trajectory_id=data.get("trajectory_id"),
            session_id=data.get("session_id"),
            task_type=data.get("task_type", "text_to_image"),
            policy_version=data.get("policy_version", "v1.0"),
            total_reward=data.get("total_reward", 0.0),
            discounted_return=data.get("discounted_return", 0.0),
            discount_factor=data.get("discount_factor", 0.99),
            metadata=data.get("metadata", {}),
        )
        
        if "created_at" in data:
            trajectory.created_at = datetime.fromisoformat(data["created_at"])
        
        for t_data in data.get("transitions", []):
            trajectory.transitions.append(Transition.from_dict(t_data))
        
        return trajectory


class TrajectoryBuilder:
    """
    Builder for constructing trajectories from plan executions.
    """
    
    def __init__(
        self,
        session_id: Optional[int] = None,
        task_type: str = "text_to_image",
        policy_version: str = "v1.0",
        discount_factor: float = 0.99,
    ):
        self.trajectory = Trajectory(
            session_id=session_id,
            task_type=task_type,
            policy_version=policy_version,
            discount_factor=discount_factor,
        )
        self._current_state: Optional[State] = None
    
    def set_initial_state(self, user_prompt: str) -> "TrajectoryBuilder":
        """Set the initial state with user prompt."""
        self._current_state = State(
            state_type=StateType.INITIAL,
            user_prompt=user_prompt,
            step_number=0,
        )
        return self
    
    def add_step(
        self,
        thought: str,
        action_type: str,
        skill_name: Optional[str],
        skill_params: Dict[str, Any],
        reward: float,
        observation: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> "TrajectoryBuilder":
        """Add a step to the trajectory."""
        if self._current_state is None:
            raise ValueError("Must set initial state first")
        
        action = Action(
            thought=thought,
            action_type=action_type,
            skill_name=skill_name,
            skill_params=skill_params,
        )
        
        # Create next state from observation
        if done:
            next_state = State(
                state_type=StateType.TERMINAL,
                user_prompt=self._current_state.user_prompt,
                step_number=self._current_state.step_number + 1,
            )
        else:
            next_state = State(
                state_type=StateType.INTERMEDIATE,
                user_prompt=self._current_state.user_prompt,
                step_number=self._current_state.step_number + 1,
            )
        
        # Update next state with observation
        if observation:
            if "image_url" in observation.get("result", {}):
                next_state.current_image_url = observation["result"]["image_url"]
            if "scores" in observation.get("result", {}):
                next_state.quality_scores = observation["result"]["scores"]
        
        transition = Transition(
            state=self._current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=observation or {},
        )
        
        self.trajectory.add_transition(transition)
        self._current_state = next_state
        
        return self
    
    def build(self) -> Trajectory:
        """Build and return the complete trajectory."""
        self.trajectory.compute_returns()
        return self.trajectory
