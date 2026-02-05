"""
Reward functions for Multi-Turn RL training.

Defines various reward signals for image generation tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class RewardType(str, Enum):
    """Types of reward signals."""
    QUALITY = "quality"           # Image quality score
    EFFICIENCY = "efficiency"     # Step efficiency
    SUCCESS = "success"          # Task completion
    COMPOSITE = "composite"      # Combined rewards


@dataclass
class RewardSignal:
    """A single reward signal with type and value."""
    reward_type: RewardType
    value: float
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    
    def weighted_value(self) -> float:
        """Get weighted reward value."""
        return self.value * self.weight


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """
        Compute reward for a transition.
        
        Args:
            observation: Result from skill execution.
            action: Action taken by the agent.
            context: Additional context (step number, etc.)
            
        Returns:
            RewardSignal with computed reward.
        """
        pass


class QualityReward(RewardFunction):
    """
    Reward based on image quality scores.
    
    Higher quality scores result in higher rewards.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        max_reward: float = 1.0,
        penalty_below_threshold: float = -0.5,
    ):
        self.quality_threshold = quality_threshold
        self.max_reward = max_reward
        self.penalty_below_threshold = penalty_below_threshold
    
    def compute(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Compute quality-based reward."""
        result = observation.get("result", {})
        
        # Get overall score from evaluation
        overall_score = result.get("overall_score")
        
        if overall_score is None:
            # No quality score available
            return RewardSignal(
                reward_type=RewardType.QUALITY,
                value=0.0,
                metadata={"reason": "no_score"},
            )
        
        # Compute reward based on quality
        if overall_score >= self.quality_threshold:
            # Scale reward based on how much above threshold
            excess = overall_score - self.quality_threshold
            max_excess = 1.0 - self.quality_threshold
            reward = self.max_reward * (0.5 + 0.5 * (excess / max_excess if max_excess > 0 else 0))
        else:
            # Penalty for below threshold
            deficit = self.quality_threshold - overall_score
            reward = self.penalty_below_threshold * (deficit / self.quality_threshold)
        
        return RewardSignal(
            reward_type=RewardType.QUALITY,
            value=reward,
            metadata={
                "overall_score": overall_score,
                "threshold": self.quality_threshold,
            },
        )


class EfficiencyReward(RewardFunction):
    """
    Reward based on step efficiency.
    
    Encourages completing tasks in fewer steps.
    """
    
    def __init__(
        self,
        optimal_steps: int = 3,
        max_steps: int = 10,
        step_penalty: float = -0.1,
    ):
        self.optimal_steps = optimal_steps
        self.max_steps = max_steps
        self.step_penalty = step_penalty
    
    def compute(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Compute efficiency-based reward."""
        context = context or {}
        step_number = context.get("step_number", 1)
        
        if step_number <= self.optimal_steps:
            # No penalty for optimal steps
            reward = 0.0
        else:
            # Increasing penalty for extra steps
            extra_steps = step_number - self.optimal_steps
            reward = self.step_penalty * extra_steps
        
        return RewardSignal(
            reward_type=RewardType.EFFICIENCY,
            value=reward,
            metadata={
                "step_number": step_number,
                "optimal_steps": self.optimal_steps,
            },
        )


class SuccessReward(RewardFunction):
    """
    Reward based on task completion success.
    
    Gives bonus for successful completion, penalty for failure.
    """
    
    def __init__(
        self,
        success_bonus: float = 2.0,
        failure_penalty: float = -1.0,
    ):
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
    
    def compute(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Compute success-based reward."""
        # Check if this is a finish action
        action_type = action.get("action_type", "")
        
        if action_type != "finish":
            return RewardSignal(
                reward_type=RewardType.SUCCESS,
                value=0.0,
                metadata={"reason": "not_terminal"},
            )
        
        # Check result
        action_input = action.get("action_input", {})
        result = action_input.get("result", "")
        
        if result == "success":
            reward = self.success_bonus
        elif result == "failure":
            reward = self.failure_penalty
        else:
            reward = 0.0
        
        return RewardSignal(
            reward_type=RewardType.SUCCESS,
            value=reward,
            metadata={"result": result},
        )


class CompositeReward(RewardFunction):
    """
    Combines multiple reward functions with weights.
    """
    
    def __init__(
        self,
        reward_functions: List[Tuple[RewardFunction, float]] = None,
    ):
        """
        Initialize with list of (reward_function, weight) tuples.
        """
        self.reward_functions = reward_functions or []
        
        # Default composition if none provided
        if not self.reward_functions:
            self.reward_functions = [
                (QualityReward(), 0.5),
                (EfficiencyReward(), 0.2),
                (SuccessReward(), 0.3),
            ]
    
    def add_reward_function(
        self,
        reward_fn: RewardFunction,
        weight: float,
    ) -> None:
        """Add a reward function with weight."""
        self.reward_functions.append((reward_fn, weight))
    
    def compute(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Compute weighted combination of rewards."""
        total_reward = 0.0
        component_rewards = {}
        
        for reward_fn, weight in self.reward_functions:
            signal = reward_fn.compute(observation, action, context)
            weighted = signal.value * weight
            total_reward += weighted
            component_rewards[signal.reward_type.value] = {
                "raw": signal.value,
                "weight": weight,
                "weighted": weighted,
            }
        
        return RewardSignal(
            reward_type=RewardType.COMPOSITE,
            value=total_reward,
            metadata={"components": component_rewards},
        )


class RewardNormalizer:
    """
    Normalizes rewards using running statistics.
    
    Helps stabilize training by keeping rewards in a reasonable range.
    """
    
    def __init__(
        self,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        epsilon: float = 1e-8,
    ):
        self.clip_range = clip_range
        self.epsilon = epsilon
        
        # Running statistics
        self._count = 0
        self._mean = 0.0
        self._var = 1.0
        self._m2 = 0.0  # For Welford's algorithm
    
    def update(self, reward: float) -> None:
        """Update running statistics with new reward."""
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._m2 += delta * delta2
        
        if self._count > 1:
            self._var = self._m2 / (self._count - 1)
    
    def normalize(self, reward: float) -> float:
        """Normalize a reward value."""
        if self._count < 2:
            return reward
        
        std = (self._var + self.epsilon) ** 0.5
        normalized = (reward - self._mean) / std
        
        # Clip to range
        return max(self.clip_range[0], min(self.clip_range[1], normalized))
    
    def update_and_normalize(self, reward: float) -> float:
        """Update statistics and return normalized reward."""
        self.update(reward)
        return self.normalize(reward)
    
    @property
    def mean(self) -> float:
        """Get current mean."""
        return self._mean
    
    @property
    def std(self) -> float:
        """Get current standard deviation."""
        return (self._var + self.epsilon) ** 0.5
    
    def reset(self) -> None:
        """Reset running statistics."""
        self._count = 0
        self._mean = 0.0
        self._var = 1.0
        self._m2 = 0.0


# Default reward function factory
def create_default_reward() -> CompositeReward:
    """Create the default composite reward function."""
    return CompositeReward([
        (QualityReward(quality_threshold=0.7), 0.5),
        (EfficiencyReward(optimal_steps=3, step_penalty=-0.1), 0.2),
        (SuccessReward(success_bonus=2.0, failure_penalty=-1.0), 0.3),
    ])
