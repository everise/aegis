"""
Task Advantage Normalization for Multi-Task RL training.

Implements per-task normalization of advantages to handle
different reward scales across task types.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math

from app.rl.trajectory import Trajectory


@dataclass
class TaskStatistics:
    """Running statistics for a task type."""
    task_type: str
    count: int = 0
    mean: float = 0.0
    variance: float = 1.0
    m2: float = 0.0  # For Welford's algorithm
    min_value: float = float('inf')
    max_value: float = float('-inf')
    
    def update(self, value: float) -> None:
        """Update statistics with a new value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        if self.count > 1:
            self.variance = self.m2 / (self.count - 1)
        
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
    
    @property
    def std(self) -> float:
        """Get standard deviation."""
        return math.sqrt(self.variance + 1e-8)
    
    def normalize(self, value: float, epsilon: float = 1e-8) -> float:
        """Normalize a value using current statistics."""
        if self.count < 2:
            return value
        return (value - self.mean) / (self.std + epsilon)
    
    def denormalize(self, normalized: float) -> float:
        """Denormalize a value back to original scale."""
        return normalized * self.std + self.mean


class TaskAdvantageNormalizer:
    """
    Normalizes advantages per task type.
    
    Different task types may have different reward scales and
    advantage distributions. This normalizer ensures fair
    contribution from all task types during training.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
        warmup_samples: int = 10,
    ):
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.warmup_samples = warmup_samples
        
        self._task_stats: Dict[str, TaskStatistics] = {}
    
    def _get_or_create_stats(self, task_type: str) -> TaskStatistics:
        """Get or create statistics for a task type."""
        if task_type not in self._task_stats:
            self._task_stats[task_type] = TaskStatistics(task_type=task_type)
        return self._task_stats[task_type]
    
    def update_statistics(
        self,
        task_type: str,
        advantages: List[float],
    ) -> None:
        """Update statistics with new advantage values."""
        stats = self._get_or_create_stats(task_type)
        for adv in advantages:
            stats.update(adv)
    
    def normalize_advantage(
        self,
        task_type: str,
        advantage: float,
    ) -> float:
        """
        Normalize a single advantage value.
        
        Args:
            task_type: The task type for normalization
            advantage: Raw advantage value
            
        Returns:
            Normalized advantage
        """
        stats = self._get_or_create_stats(task_type)
        
        # Skip normalization during warmup
        if stats.count < self.warmup_samples:
            normalized = advantage
        else:
            normalized = stats.normalize(advantage, self.epsilon)
        
        # Clip if range specified
        if self.clip_range:
            normalized = max(self.clip_range[0], min(self.clip_range[1], normalized))
        
        return normalized
    
    def normalize_advantages(
        self,
        task_type: str,
        advantages: List[float],
        update_stats: bool = True,
    ) -> List[float]:
        """
        Normalize a list of advantages.
        
        Args:
            task_type: The task type for normalization
            advantages: List of raw advantage values
            update_stats: Whether to update running statistics
            
        Returns:
            List of normalized advantages
        """
        if update_stats:
            self.update_statistics(task_type, advantages)
        
        return [
            self.normalize_advantage(task_type, adv)
            for adv in advantages
        ]
    
    def normalize_trajectory(
        self,
        trajectory: Trajectory,
        update_stats: bool = True,
    ) -> List[float]:
        """
        Normalize advantages for a trajectory.
        
        Args:
            trajectory: Trajectory with computed advantages
            update_stats: Whether to update running statistics
            
        Returns:
            List of normalized advantages
        """
        advantages = trajectory.compute_advantages()
        return self.normalize_advantages(
            trajectory.task_type,
            advantages,
            update_stats=update_stats,
        )
    
    def normalize_batch(
        self,
        trajectories: List[Trajectory],
        update_stats: bool = True,
    ) -> Dict[str, List[Tuple[Trajectory, List[float]]]]:
        """
        Normalize advantages for a batch of trajectories.
        
        Groups by task type and normalizes within each group.
        
        Args:
            trajectories: List of trajectories
            update_stats: Whether to update running statistics
            
        Returns:
            Dict mapping task_type to list of (trajectory, normalized_advantages)
        """
        # Group by task type
        by_task: Dict[str, List[Trajectory]] = defaultdict(list)
        for traj in trajectories:
            by_task[traj.task_type].append(traj)
        
        results: Dict[str, List[Tuple[Trajectory, List[float]]]] = {}
        
        for task_type, task_trajs in by_task.items():
            # Collect all advantages for this task type
            all_advantages = []
            traj_advantages = []
            
            for traj in task_trajs:
                advs = traj.compute_advantages()
                traj_advantages.append((traj, advs))
                all_advantages.extend(advs)
            
            # Update statistics with batch
            if update_stats and all_advantages:
                self.update_statistics(task_type, all_advantages)
            
            # Normalize all advantages
            normalized_results = []
            for traj, advs in traj_advantages:
                normalized = [
                    self.normalize_advantage(task_type, a)
                    for a in advs
                ]
                normalized_results.append((traj, normalized))
            
            results[task_type] = normalized_results
        
        return results
    
    def get_task_statistics(self, task_type: str) -> Optional[TaskStatistics]:
        """Get statistics for a specific task type."""
        return self._task_stats.get(task_type)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all task types."""
        return {
            task_type: {
                "count": stats.count,
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min_value if stats.min_value != float('inf') else None,
                "max": stats.max_value if stats.max_value != float('-inf') else None,
            }
            for task_type, stats in self._task_stats.items()
        }
    
    def reset(self, task_type: Optional[str] = None) -> None:
        """
        Reset statistics.
        
        Args:
            task_type: Specific task type to reset, or None for all
        """
        if task_type:
            if task_type in self._task_stats:
                del self._task_stats[task_type]
        else:
            self._task_stats.clear()


class GlobalAdvantageNormalizer:
    """
    Normalizes advantages globally across all task types.
    
    Alternative to per-task normalization when task types
    should be weighted equally regardless of their scales.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ):
        self.epsilon = epsilon
        self.clip_range = clip_range
        self._global_stats = TaskStatistics(task_type="global")
    
    def update(self, advantage: float) -> None:
        """Update global statistics."""
        self._global_stats.update(advantage)
    
    def normalize(self, advantage: float) -> float:
        """Normalize using global statistics."""
        normalized = self._global_stats.normalize(advantage, self.epsilon)
        
        if self.clip_range:
            normalized = max(self.clip_range[0], min(self.clip_range[1], normalized))
        
        return normalized
    
    def normalize_batch(
        self,
        advantages: List[float],
        update_stats: bool = True,
    ) -> List[float]:
        """Normalize a batch of advantages."""
        if update_stats:
            for adv in advantages:
                self.update(adv)
        
        return [self.normalize(adv) for adv in advantages]
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "count": self._global_stats.count,
            "mean": self._global_stats.mean,
            "std": self._global_stats.std,
        }
    
    def reset(self) -> None:
        """Reset statistics."""
        self._global_stats = TaskStatistics(task_type="global")


def create_advantage_normalizer(
    per_task: bool = True,
    **kwargs,
) -> TaskAdvantageNormalizer | GlobalAdvantageNormalizer:
    """
    Factory function to create an advantage normalizer.
    
    Args:
        per_task: Whether to use per-task normalization
        **kwargs: Additional arguments for the normalizer
        
    Returns:
        Appropriate normalizer instance
    """
    if per_task:
        return TaskAdvantageNormalizer(**kwargs)
    else:
        return GlobalAdvantageNormalizer(**kwargs)
