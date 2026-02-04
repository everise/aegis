"""
Experience Replay Buffer for Multi-Turn RL training.

Stores and samples trajectories for training with various
prioritization strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import random
import heapq
import math

from rl.trajectory import Trajectory, Transition


@dataclass
class BufferEntry:
    """Entry in the replay buffer."""
    trajectory: Trajectory
    priority: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
    
    def __lt__(self, other: "BufferEntry") -> bool:
        """For heap operations."""
        return self.priority > other.priority  # Max heap


class ReplayBuffer:
    """
    Basic replay buffer with uniform sampling.
    
    Stores trajectories and provides random sampling for training.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        min_size: int = 100,
    ):
        self.capacity = capacity
        self.min_size = min_size
        self._buffer: deque[BufferEntry] = deque(maxlen=capacity)
        self._total_transitions = 0
    
    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        entry = BufferEntry(trajectory=trajectory)
        self._buffer.append(entry)
        self._total_transitions += trajectory.length
    
    def sample(self, batch_size: int) -> List[Trajectory]:
        """Sample a batch of trajectories uniformly."""
        if len(self._buffer) < self.min_size:
            return []
        
        batch_size = min(batch_size, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), batch_size)
        
        samples = []
        for idx in indices:
            entry = self._buffer[idx]
            entry.sample_count += 1
            samples.append(entry.trajectory)
        
        return samples
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """Sample individual transitions from trajectories."""
        if self._total_transitions == 0:
            return []
        
        # Collect all transitions
        all_transitions = []
        for entry in self._buffer:
            all_transitions.extend(entry.trajectory.transitions)
        
        batch_size = min(batch_size, len(all_transitions))
        return random.sample(all_transitions, batch_size)
    
    @property
    def size(self) -> int:
        """Current buffer size in trajectories."""
        return len(self._buffer)
    
    @property
    def total_transitions(self) -> int:
        """Total number of transitions in buffer."""
        return self._total_transitions
    
    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= self.min_size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_transitions = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self._buffer:
            return {
                "size": 0,
                "total_transitions": 0,
                "avg_trajectory_length": 0,
                "avg_return": 0,
            }
        
        returns = [e.trajectory.discounted_return for e in self._buffer]
        lengths = [e.trajectory.length for e in self._buffer]
        
        return {
            "size": len(self._buffer),
            "total_transitions": self._total_transitions,
            "avg_trajectory_length": sum(lengths) / len(lengths),
            "avg_return": sum(returns) / len(returns),
            "min_return": min(returns),
            "max_return": max(returns),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer using TD-error or return-based priorities.
    
    Samples trajectories proportional to their priority, focusing
    training on more important experiences.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        min_size: int = 100,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity, min_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self._priorities: List[float] = []
        self._max_priority = 1.0
    
    def add(
        self,
        trajectory: Trajectory,
        priority: Optional[float] = None,
    ) -> None:
        """Add a trajectory with optional priority."""
        if priority is None:
            priority = self._max_priority
        
        entry = BufferEntry(trajectory=trajectory, priority=priority)
        
        if len(self._buffer) >= self.capacity:
            # Remove oldest
            self._total_transitions -= self._buffer[0].trajectory.length
            self._priorities.pop(0)
        
        self._buffer.append(entry)
        self._priorities.append(priority ** self.alpha)
        self._total_transitions += trajectory.length
        
        self._max_priority = max(self._max_priority, priority)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[Trajectory], List[int], List[float]]:
        """
        Sample trajectories with priorities.
        
        Returns:
            Tuple of (trajectories, indices, importance_weights)
        """
        if len(self._buffer) < self.min_size:
            return [], [], []
        
        batch_size = min(batch_size, len(self._buffer))
        
        # Compute sampling probabilities
        priorities_sum = sum(self._priorities)
        probs = [p / priorities_sum for p in self._priorities]
        
        # Sample indices
        indices = random.choices(
            range(len(self._buffer)),
            weights=probs,
            k=batch_size,
        )
        
        # Compute importance sampling weights
        n = len(self._buffer)
        weights = []
        max_weight = (n * min(probs)) ** (-self.beta)
        
        for idx in indices:
            weight = (n * probs[idx]) ** (-self.beta)
            weights.append(weight / max_weight)
        
        # Get trajectories
        trajectories = []
        for idx in indices:
            entry = self._buffer[idx]
            entry.sample_count += 1
            trajectories.append(entry.trajectory)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return trajectories, indices, weights
    
    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float],
    ) -> None:
        """Update priorities for sampled trajectories."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = priority ** self.alpha
                self._buffer[idx].priority = priority
                self._max_priority = max(self._max_priority, priority)


class HindsightReplayBuffer(ReplayBuffer):
    """
    Hindsight Experience Replay buffer.
    
    Relabels failed trajectories with achieved goals to learn
    from failures.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        min_size: int = 100,
        hindsight_ratio: float = 0.5,
    ):
        super().__init__(capacity, min_size)
        self.hindsight_ratio = hindsight_ratio
    
    def _relabel_trajectory(
        self,
        trajectory: Trajectory,
        achieved_goal: Dict[str, Any],
    ) -> Trajectory:
        """
        Relabel a trajectory with a different goal.
        
        Creates a modified trajectory where the achieved outcome
        is treated as the intended goal.
        """
        from copy import deepcopy
        
        relabeled = deepcopy(trajectory)
        relabeled.metadata["hindsight"] = True
        relabeled.metadata["original_goal"] = trajectory.transitions[0].state.user_prompt
        relabeled.metadata["relabeled_goal"] = achieved_goal
        
        # Update states with new goal
        for transition in relabeled.transitions:
            transition.state.metadata["relabeled"] = True
        
        # Recompute rewards based on achieved goal
        if relabeled.transitions and relabeled.transitions[-1].done:
            # Mark as successful since we achieved the relabeled goal
            relabeled.transitions[-1].reward = 1.0
        
        relabeled.compute_returns()
        return relabeled
    
    def sample_with_hindsight(
        self,
        batch_size: int,
    ) -> List[Trajectory]:
        """
        Sample trajectories with hindsight relabeling.
        
        A portion of failed trajectories are relabeled with their
        achieved outcomes as goals.
        """
        if len(self._buffer) < self.min_size:
            return []
        
        # Regular samples
        regular_size = int(batch_size * (1 - self.hindsight_ratio))
        regular_samples = super().sample(regular_size)
        
        # Hindsight samples from failed trajectories
        hindsight_size = batch_size - len(regular_samples)
        failed_entries = [
            e for e in self._buffer
            if not e.trajectory.is_complete or e.trajectory.discounted_return < 0
        ]
        
        hindsight_samples = []
        if failed_entries and hindsight_size > 0:
            selected = random.sample(
                failed_entries,
                min(hindsight_size, len(failed_entries)),
            )
            
            for entry in selected:
                # Get achieved outcome from last transition
                last_transition = entry.trajectory.transitions[-1] if entry.trajectory.transitions else None
                if last_transition and last_transition.next_state:
                    achieved = {
                        "image_url": last_transition.next_state.current_image_url,
                        "quality": last_transition.next_state.quality_scores,
                    }
                    relabeled = self._relabel_trajectory(entry.trajectory, achieved)
                    hindsight_samples.append(relabeled)
        
        return regular_samples + hindsight_samples


class TaskSpecificBuffer:
    """
    Manages separate buffers for different task types.
    
    Ensures balanced sampling across task types during training.
    """
    
    def __init__(
        self,
        capacity_per_task: int = 5000,
        min_size_per_task: int = 50,
    ):
        self.capacity_per_task = capacity_per_task
        self.min_size_per_task = min_size_per_task
        self._buffers: Dict[str, ReplayBuffer] = {}
    
    def _get_or_create_buffer(self, task_type: str) -> ReplayBuffer:
        """Get or create buffer for task type."""
        if task_type not in self._buffers:
            self._buffers[task_type] = ReplayBuffer(
                capacity=self.capacity_per_task,
                min_size=self.min_size_per_task,
            )
        return self._buffers[task_type]
    
    def add(self, trajectory: Trajectory) -> None:
        """Add trajectory to appropriate task buffer."""
        buffer = self._get_or_create_buffer(trajectory.task_type)
        buffer.add(trajectory)
    
    def sample(
        self,
        batch_size: int,
        task_type: Optional[str] = None,
    ) -> List[Trajectory]:
        """
        Sample trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            task_type: Specific task type, or None for balanced sampling
        """
        if task_type:
            buffer = self._buffers.get(task_type)
            if buffer:
                return buffer.sample(batch_size)
            return []
        
        # Balanced sampling across task types
        ready_buffers = [
            (task, buf) for task, buf in self._buffers.items()
            if buf.is_ready
        ]
        
        if not ready_buffers:
            return []
        
        # Sample equally from each buffer
        per_buffer = max(1, batch_size // len(ready_buffers))
        samples = []
        
        for task_type, buffer in ready_buffers:
            samples.extend(buffer.sample(per_buffer))
        
        # Shuffle to mix task types
        random.shuffle(samples)
        return samples[:batch_size]
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all task buffers."""
        return {
            task: buffer.get_statistics()
            for task, buffer in self._buffers.items()
        }
    
    @property
    def total_size(self) -> int:
        """Total trajectories across all buffers."""
        return sum(buf.size for buf in self._buffers.values())
    
    def clear(self, task_type: Optional[str] = None) -> None:
        """Clear buffer(s)."""
        if task_type:
            if task_type in self._buffers:
                self._buffers[task_type].clear()
        else:
            for buffer in self._buffers.values():
                buffer.clear()
