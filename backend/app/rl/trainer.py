"""
RL Trainer for Multi-Turn policy optimization.

Implements training loop with trajectory collection, advantage
computation, and policy updates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, AsyncIterator
from enum import Enum
import asyncio
import json

from app.rl.trajectory import Trajectory, TrajectoryBuilder
from app.rl.reward import CompositeReward, RewardNormalizer, create_default_reward
from app.rl.cross_policy import CrossPolicySampler, PolicyPool
from app.rl.task_norm import TaskAdvantageNormalizer
from app.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, TaskSpecificBuffer


class TrainerStatus(str, Enum):
    """Status of the trainer."""
    IDLE = "idle"
    COLLECTING = "collecting"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for the trainer."""
    # Training loop
    total_epochs: int = 100
    steps_per_epoch: int = 1000
    batch_size: int = 32
    
    # Learning
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO specific
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Replay buffer
    buffer_size: int = 10000
    min_buffer_size: int = 100
    prioritized_replay: bool = True
    
    # Cross-policy
    use_cross_policy: bool = True
    cross_policy_epsilon: float = 0.2
    max_policy_versions: int = 5
    
    # Task normalization
    use_task_normalization: bool = True
    advantage_clip: float = 10.0
    
    # Checkpointing
    checkpoint_interval: int = 10
    eval_interval: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_epochs": self.total_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "buffer_size": self.buffer_size,
            "use_cross_policy": self.use_cross_policy,
            "use_task_normalization": self.use_task_normalization,
        }


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    step: int = 0
    
    # Losses
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0
    
    # Rewards
    mean_reward: float = 0.0
    mean_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    
    # Buffer stats
    buffer_size: int = 0
    trajectories_collected: int = 0
    
    # Training progress
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "total_loss": self.total_loss,
            "mean_reward": self.mean_reward,
            "mean_return": self.mean_return,
            "max_return": self.max_return,
            "min_return": self.min_return,
            "buffer_size": self.buffer_size,
            "trajectories_collected": self.trajectories_collected,
        }


class RLTrainer:
    """
    Main trainer class for Multi-Turn RL.
    
    Coordinates trajectory collection, advantage computation,
    and policy optimization.
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        reward_function: Optional[CompositeReward] = None,
    ):
        self.config = config or TrainingConfig()
        self.reward_function = reward_function or create_default_reward()
        
        # Status
        self.status = TrainerStatus.IDLE
        self._current_epoch = 0
        self._current_step = 0
        self._policy_version = "v1.0"
        
        # Components
        self._init_components()
        
        # Metrics history
        self._metrics_history: List[TrainingMetrics] = []
        self._best_return = float('-inf')
    
    def _init_components(self) -> None:
        """Initialize training components."""
        # Replay buffer
        if self.config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size,
                min_size=self.config.min_buffer_size,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.buffer_size,
                min_size=self.config.min_buffer_size,
            )
        
        # Task-specific buffer for multi-task
        self.task_buffer = TaskSpecificBuffer(
            capacity_per_task=self.config.buffer_size // 3,
        )
        
        # Cross-policy sampler
        if self.config.use_cross_policy:
            self.policy_pool = PolicyPool(
                max_versions=self.config.max_policy_versions,
            )
            self.policy_pool.add_policy(self._policy_version)
            
            self.cross_policy_sampler = CrossPolicySampler(
                policy_pool=self.policy_pool,
                epsilon=self.config.cross_policy_epsilon,
            )
        else:
            self.policy_pool = None
            self.cross_policy_sampler = None
        
        # Advantage normalizer
        if self.config.use_task_normalization:
            self.advantage_normalizer = TaskAdvantageNormalizer(
                clip_range=(-self.config.advantage_clip, self.config.advantage_clip),
            )
        else:
            self.advantage_normalizer = None
        
        # Reward normalizer
        self.reward_normalizer = RewardNormalizer()
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a collected trajectory to the buffer."""
        # Set policy version
        trajectory.policy_version = self._policy_version
        
        # Add to buffers
        self.replay_buffer.add(trajectory)
        self.task_buffer.add(trajectory)
        
        # Update cross-policy statistics
        if self.cross_policy_sampler:
            self.cross_policy_sampler.record_trajectory_result(trajectory)
    
    def _compute_advantages(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, List[float]]:
        """Compute normalized advantages for trajectories."""
        advantages_by_traj: Dict[str, List[float]] = {}
        
        if self.advantage_normalizer:
            # Per-task normalization
            normalized = self.advantage_normalizer.normalize_batch(
                trajectories,
                update_stats=True,
            )
            
            for task_type, traj_advs in normalized.items():
                for traj, advs in traj_advs:
                    advantages_by_traj[traj.trajectory_id or str(id(traj))] = advs
        else:
            # Simple normalization
            for traj in trajectories:
                advs = traj.compute_advantages()
                advantages_by_traj[traj.trajectory_id or str(id(traj))] = advs
        
        return advantages_by_traj
    
    def _sample_batch(self) -> List[Trajectory]:
        """Sample a training batch."""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            trajectories, indices, weights = self.replay_buffer.sample(
                self.config.batch_size
            )
            # Store for priority updates
            self._last_sample_indices = indices
            self._last_sample_weights = weights
            return trajectories
        else:
            return self.replay_buffer.sample(self.config.batch_size)
    
    def _compute_losses(
        self,
        trajectories: List[Trajectory],
        advantages: Dict[str, List[float]],
    ) -> TrainingMetrics:
        """
        Compute training losses.
        
        Note: This is a placeholder implementation.
        In practice, this would compute policy and value losses
        using the actual neural network.
        """
        metrics = TrainingMetrics(
            epoch=self._current_epoch,
            step=self._current_step,
        )
        
        # Placeholder loss computation
        # In real implementation, this would involve:
        # 1. Forward pass through policy network
        # 2. Compute log probabilities of actions
        # 3. Compute policy loss (PPO clipped objective)
        # 4. Compute value loss
        # 5. Compute entropy bonus
        
        all_returns = [t.discounted_return for t in trajectories]
        all_rewards = [t.total_reward for t in trajectories]
        
        if all_returns:
            metrics.mean_return = sum(all_returns) / len(all_returns)
            metrics.max_return = max(all_returns)
            metrics.min_return = min(all_returns)
        
        if all_rewards:
            metrics.mean_reward = sum(all_rewards) / len(all_rewards)
        
        # Simulated losses
        metrics.policy_loss = 0.1  # Placeholder
        metrics.value_loss = 0.05  # Placeholder
        metrics.entropy_loss = 0.01  # Placeholder
        metrics.total_loss = (
            metrics.policy_loss +
            self.config.value_loss_coef * metrics.value_loss -
            self.config.entropy_coef * metrics.entropy_loss
        )
        
        metrics.buffer_size = self.replay_buffer.size
        metrics.learning_rate = self.config.learning_rate
        
        return metrics
    
    def _update_policy(self, metrics: TrainingMetrics) -> None:
        """
        Update policy parameters.
        
        Note: Placeholder implementation.
        """
        # In real implementation:
        # 1. Backpropagate losses
        # 2. Clip gradients
        # 3. Optimizer step
        
        # Update priorities in prioritized replay
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            if hasattr(self, '_last_sample_indices'):
                # Use TD-error or advantage magnitude as priority
                priorities = [1.0] * len(self._last_sample_indices)
                self.replay_buffer.update_priorities(
                    self._last_sample_indices,
                    priorities,
                )
    
    def train_step(self) -> TrainingMetrics:
        """Execute a single training step."""
        if not self.replay_buffer.is_ready:
            return TrainingMetrics()
        
        self.status = TrainerStatus.TRAINING
        self._current_step += 1
        
        # Sample batch
        trajectories = self._sample_batch()
        
        if not trajectories:
            return TrainingMetrics()
        
        # Compute advantages
        advantages = self._compute_advantages(trajectories)
        
        # Compute losses
        metrics = self._compute_losses(trajectories, advantages)
        metrics.trajectories_collected = self.replay_buffer.size
        
        # Update policy
        self._update_policy(metrics)
        
        # Record metrics
        self._metrics_history.append(metrics)
        
        # Check for best return
        if metrics.mean_return > self._best_return:
            self._best_return = metrics.mean_return
        
        return metrics
    
    async def train_epoch(self) -> List[TrainingMetrics]:
        """Execute a training epoch."""
        self._current_epoch += 1
        epoch_metrics = []
        
        for _ in range(self.config.steps_per_epoch):
            metrics = self.train_step()
            epoch_metrics.append(metrics)
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        return epoch_metrics
    
    async def train(
        self,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> AsyncIterator[TrainingMetrics]:
        """
        Run the full training loop.
        
        Yields metrics after each epoch.
        """
        self.status = TrainerStatus.TRAINING
        
        try:
            for epoch in range(self.config.total_epochs):
                epoch_metrics = await self.train_epoch()
                
                # Aggregate epoch metrics
                if epoch_metrics:
                    avg_metrics = TrainingMetrics(
                        epoch=self._current_epoch,
                        step=self._current_step,
                        policy_loss=sum(m.policy_loss for m in epoch_metrics) / len(epoch_metrics),
                        value_loss=sum(m.value_loss for m in epoch_metrics) / len(epoch_metrics),
                        mean_return=sum(m.mean_return for m in epoch_metrics) / len(epoch_metrics),
                        mean_reward=sum(m.mean_reward for m in epoch_metrics) / len(epoch_metrics),
                        buffer_size=self.replay_buffer.size,
                    )
                    
                    if callback:
                        callback(avg_metrics)
                    
                    yield avg_metrics
                
                # Checkpoint
                if epoch % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Update policy version periodically
                if self.policy_pool and epoch % 10 == 0:
                    self._policy_version = f"v{self._current_epoch}.{self._current_step}"
                    self.policy_pool.add_policy(self._policy_version)
            
            self.status = TrainerStatus.COMPLETED
            
        except Exception as e:
            self.status = TrainerStatus.FAILED
            raise
    
    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        # Placeholder - would save model weights, optimizer state, etc.
        pass
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            "status": self.status.value,
            "current_epoch": self._current_epoch,
            "current_step": self._current_step,
            "policy_version": self._policy_version,
            "best_return": self._best_return,
            "buffer_size": self.replay_buffer.size,
            "config": self.config.to_dict(),
            "metrics_history_length": len(self._metrics_history),
        }
    
    def pause(self) -> None:
        """Pause training."""
        self.status = TrainerStatus.PAUSED
    
    def resume(self) -> None:
        """Resume training."""
        self.status = TrainerStatus.TRAINING
    
    def reset(self) -> None:
        """Reset trainer state."""
        self._current_epoch = 0
        self._current_step = 0
        self._policy_version = "v1.0"
        self._metrics_history.clear()
        self._best_return = float('-inf')
        self._init_components()
        self.status = TrainerStatus.IDLE
