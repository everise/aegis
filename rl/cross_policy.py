"""
Cross-Policy Sampling for Multi-Turn RL training.

Implements sampling from multiple policy versions to improve
exploration and training stability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import random
import math
from collections import defaultdict

from rl.trajectory import Trajectory


@dataclass
class PolicyVersion:
    """Represents a version of the policy."""
    version_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    performance_score: float = 0.0
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_performance(self, reward: float, alpha: float = 0.1) -> None:
        """Update performance score with exponential moving average."""
        self.performance_score = (1 - alpha) * self.performance_score + alpha * reward
        self.sample_count += 1


class PolicyPool:
    """
    Manages a pool of policy versions for cross-policy sampling.
    
    Maintains multiple policy versions and provides sampling strategies
    for selecting which policy to use for action generation.
    """
    
    def __init__(
        self,
        max_versions: int = 5,
        min_version_samples: int = 10,
    ):
        self.max_versions = max_versions
        self.min_version_samples = min_version_samples
        self._policies: Dict[str, PolicyVersion] = {}
        self._current_version: Optional[str] = None
    
    def add_policy(
        self,
        version_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PolicyVersion:
        """Add a new policy version to the pool."""
        policy = PolicyVersion(
            version_id=version_id,
            metadata=metadata or {},
        )
        self._policies[version_id] = policy
        self._current_version = version_id
        
        # Remove oldest if exceeds max
        if len(self._policies) > self.max_versions:
            self._remove_oldest()
        
        return policy
    
    def _remove_oldest(self) -> None:
        """Remove the oldest policy version."""
        if not self._policies:
            return
        
        oldest = min(
            self._policies.values(),
            key=lambda p: p.created_at,
        )
        
        # Don't remove current version
        if oldest.version_id != self._current_version:
            del self._policies[oldest.version_id]
    
    def get_policy(self, version_id: str) -> Optional[PolicyVersion]:
        """Get a specific policy version."""
        return self._policies.get(version_id)
    
    @property
    def current_policy(self) -> Optional[PolicyVersion]:
        """Get the current (latest) policy version."""
        if self._current_version:
            return self._policies.get(self._current_version)
        return None
    
    @property
    def all_versions(self) -> List[str]:
        """Get all policy version IDs."""
        return list(self._policies.keys())
    
    def sample_policy(
        self,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.2,
        temperature: float = 1.0,
    ) -> PolicyVersion:
        """
        Sample a policy version based on the sampling strategy.
        
        Args:
            strategy: Sampling strategy ("epsilon_greedy", "softmax", "uniform")
            epsilon: Exploration rate for epsilon-greedy
            temperature: Temperature for softmax sampling
            
        Returns:
            Selected PolicyVersion
        """
        if not self._policies:
            raise ValueError("No policies in pool")
        
        if len(self._policies) == 1:
            return list(self._policies.values())[0]
        
        if strategy == "uniform":
            return random.choice(list(self._policies.values()))
        
        elif strategy == "epsilon_greedy":
            if random.random() < epsilon:
                # Explore: random selection
                return random.choice(list(self._policies.values()))
            else:
                # Exploit: best performing
                return max(
                    self._policies.values(),
                    key=lambda p: p.performance_score,
                )
        
        elif strategy == "softmax":
            # Softmax selection based on performance
            policies = list(self._policies.values())
            scores = [p.performance_score / temperature for p in policies]
            
            # Numerical stability
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            total = sum(exp_scores)
            probs = [e / total for e in exp_scores]
            
            return random.choices(policies, weights=probs, k=1)[0]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def update_policy_performance(
        self,
        version_id: str,
        reward: float,
    ) -> None:
        """Update performance score for a policy version."""
        policy = self._policies.get(version_id)
        if policy:
            policy.update_performance(reward)


class CrossPolicySampler:
    """
    Implements Cross-Policy Sampling for trajectory collection.
    
    Samples trajectories from multiple policy versions to improve
    exploration and reduce distribution shift.
    """
    
    def __init__(
        self,
        policy_pool: Optional[PolicyPool] = None,
        sampling_strategy: str = "epsilon_greedy",
        epsilon: float = 0.2,
        temperature: float = 1.0,
    ):
        self.policy_pool = policy_pool or PolicyPool()
        self.sampling_strategy = sampling_strategy
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Statistics
        self._sample_counts: Dict[str, int] = defaultdict(int)
        self._total_samples = 0
    
    def sample_policy_for_trajectory(self) -> PolicyVersion:
        """Sample a policy version for generating a new trajectory."""
        policy = self.policy_pool.sample_policy(
            strategy=self.sampling_strategy,
            epsilon=self.epsilon,
            temperature=self.temperature,
        )
        
        self._sample_counts[policy.version_id] += 1
        self._total_samples += 1
        
        return policy
    
    def record_trajectory_result(
        self,
        trajectory: Trajectory,
    ) -> None:
        """Record the result of a trajectory for policy performance update."""
        if trajectory.policy_version:
            self.policy_pool.update_policy_performance(
                trajectory.policy_version,
                trajectory.discounted_return,
            )
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about cross-policy sampling."""
        stats = {
            "total_samples": self._total_samples,
            "policy_counts": dict(self._sample_counts),
            "policy_ratios": {},
        }
        
        if self._total_samples > 0:
            stats["policy_ratios"] = {
                k: v / self._total_samples
                for k, v in self._sample_counts.items()
            }
        
        return stats
    
    def compute_importance_weights(
        self,
        trajectories: List[Trajectory],
        target_version: str,
    ) -> List[float]:
        """
        Compute importance sampling weights for off-policy correction.
        
        Used when training on trajectories collected from different
        policy versions.
        
        Args:
            trajectories: List of trajectories with policy versions
            target_version: Version we're training towards
            
        Returns:
            List of importance weights for each trajectory
        """
        weights = []
        
        for traj in trajectories:
            if traj.policy_version == target_version:
                # On-policy, weight = 1
                weights.append(1.0)
            else:
                # Off-policy, compute importance weight
                # Simplified: use inverse of performance ratio
                source_policy = self.policy_pool.get_policy(traj.policy_version)
                target_policy = self.policy_pool.get_policy(target_version)
                
                if source_policy and target_policy:
                    # Avoid division by zero
                    source_perf = max(source_policy.performance_score, 0.01)
                    target_perf = max(target_policy.performance_score, 0.01)
                    weight = min(target_perf / source_perf, 2.0)  # Clip weight
                else:
                    weight = 1.0
                
                weights.append(weight)
        
        return weights
    
    def reset_statistics(self) -> None:
        """Reset sampling statistics."""
        self._sample_counts.clear()
        self._total_samples = 0


class MixedBatchSampler:
    """
    Samples mixed batches from trajectories of multiple policy versions.
    
    Ensures training batches contain diverse experiences from different
    policy versions for better generalization.
    """
    
    def __init__(
        self,
        trajectories: List[Trajectory] = None,
        batch_size: int = 32,
        mix_ratio: float = 0.5,  # Ratio of current policy samples
    ):
        self.trajectories = trajectories or []
        self.batch_size = batch_size
        self.mix_ratio = mix_ratio
        
        # Index trajectories by policy version
        self._by_version: Dict[str, List[Trajectory]] = defaultdict(list)
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild the version index."""
        self._by_version.clear()
        for traj in self.trajectories:
            version = traj.policy_version or "unknown"
            self._by_version[version].append(traj)
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the pool."""
        self.trajectories.append(trajectory)
        version = trajectory.policy_version or "unknown"
        self._by_version[version].append(trajectory)
    
    def sample_batch(
        self,
        current_version: str,
    ) -> List[Trajectory]:
        """
        Sample a mixed batch of trajectories.
        
        Args:
            current_version: The current policy version
            
        Returns:
            List of sampled trajectories
        """
        if not self.trajectories:
            return []
        
        batch = []
        
        # Sample from current version
        current_trajs = self._by_version.get(current_version, [])
        current_count = int(self.batch_size * self.mix_ratio)
        
        if current_trajs:
            current_samples = random.choices(
                current_trajs,
                k=min(current_count, len(current_trajs)),
            )
            batch.extend(current_samples)
        
        # Sample from other versions
        other_count = self.batch_size - len(batch)
        other_trajs = [
            t for v, trajs in self._by_version.items()
            if v != current_version
            for t in trajs
        ]
        
        if other_trajs and other_count > 0:
            other_samples = random.choices(
                other_trajs,
                k=min(other_count, len(other_trajs)),
            )
            batch.extend(other_samples)
        
        # Fill remaining with any trajectory if needed
        while len(batch) < self.batch_size and self.trajectories:
            batch.append(random.choice(self.trajectories))
        
        return batch[:self.batch_size]
    
    def get_version_statistics(self) -> Dict[str, int]:
        """Get count of trajectories per version."""
        return {v: len(trajs) for v, trajs in self._by_version.items()}
