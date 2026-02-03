"""
RL Training Module for Aegis.

This module provides reinforcement learning components for training
the image generation agent through multi-turn interactions.
"""

from rl.trajectory import (
    State,
    StateType,
    Action,
    Transition,
    Trajectory,
    TrajectoryBuilder,
)

from rl.reward import (
    BaseReward,
    QualityReward,
    EfficiencyReward,
    SuccessReward,
    CompositeReward,
    RewardNormalizer,
    create_default_reward,
)

from rl.cross_policy import (
    PolicyVersion,
    PolicyPool,
    CrossPolicySampler,
    MixedBatchSampler,
)

from rl.task_norm import (
    TaskStatistics,
    TaskAdvantageNormalizer,
    GlobalAdvantageNormalizer,
    create_advantage_normalizer,
)

from rl.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    HindsightReplayBuffer,
    TaskSpecificBuffer,
)

from rl.trainer import (
    TrainerStatus,
    TrainingConfig,
    TrainingMetrics,
    RLTrainer,
)

__all__ = [
    # Trajectory
    "State",
    "StateType",
    "Action",
    "Transition",
    "Trajectory",
    "TrajectoryBuilder",
    # Reward
    "BaseReward",
    "QualityReward",
    "EfficiencyReward",
    "SuccessReward",
    "CompositeReward",
    "RewardNormalizer",
    "create_default_reward",
    # Cross-Policy
    "PolicyVersion",
    "PolicyPool",
    "CrossPolicySampler",
    "MixedBatchSampler",
    # Task Norm
    "TaskStatistics",
    "TaskAdvantageNormalizer",
    "GlobalAdvantageNormalizer",
    "create_advantage_normalizer",
    # Replay Buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "HindsightReplayBuffer",
    "TaskSpecificBuffer",
    # Trainer
    "TrainerStatus",
    "TrainingConfig",
    "TrainingMetrics",
    "RLTrainer",
]
