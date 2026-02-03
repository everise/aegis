"""
Unit tests for RL module components.

Tests trajectory, reward, cross-policy sampling, task normalization,
replay buffer, and trainer.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from app.rl.trajectory import (
    Trajectory,
    Transition,
    State,
    Action,
    StateType,
    TrajectoryBuilder,
)
from app.rl.reward import (
    QualityReward,
    EfficiencyReward,
    SuccessReward,
    CompositeReward,
    RewardNormalizer,
    RewardType,
    create_default_reward,
)
from app.rl.cross_policy import (
    PolicyPool,
    PolicyVersion,
    CrossPolicySampler,
    MixedBatchSampler,
)
from app.rl.task_norm import (
    TaskAdvantageNormalizer,
    GlobalAdvantageNormalizer,
    TaskStatistics,
)
from app.rl.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    HindsightReplayBuffer,
    TaskSpecificBuffer,
)
from app.rl.trainer import (
    RLTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainerStatus,
)


# ============== Trajectory Tests ==============

class TestState:
    """Tests for State class."""
    
    def test_state_creation(self):
        """Test creating a state."""
        state = State(
            state_type=StateType.INITIAL,
            user_prompt="Generate a sunset image",
            step_number=0,
        )
        
        assert state.state_type == StateType.INITIAL
        assert state.user_prompt == "Generate a sunset image"
        assert state.step_number == 0
    
    def test_state_to_dict(self):
        """Test state serialization."""
        state = State(
            state_type=StateType.INTERMEDIATE,
            user_prompt="test",
            current_image_url="http://example.com/img.png",
        )
        
        data = state.to_dict()
        
        assert data["state_type"] == "intermediate"
        assert data["current_image_url"] == "http://example.com/img.png"
    
    def test_state_from_dict(self):
        """Test state deserialization."""
        data = {
            "state_type": "terminal",
            "user_prompt": "test",
            "step_number": 5,
        }
        
        state = State.from_dict(data)
        
        assert state.state_type == StateType.TERMINAL
        assert state.step_number == 5


class TestTrajectory:
    """Tests for Trajectory class."""
    
    def test_trajectory_creation(self):
        """Test creating an empty trajectory."""
        traj = Trajectory(
            session_id=1,
            task_type="text_to_image",
        )
        
        assert traj.session_id == 1
        assert traj.task_type == "text_to_image"
        assert traj.length == 0
    
    def test_add_transition(self):
        """Test adding transitions."""
        traj = Trajectory()
        
        state = State(state_type=StateType.INITIAL, user_prompt="test")
        action = Action(thought="test", action_type="generate")
        transition = Transition(state=state, action=action, reward=1.0)
        
        traj.add_transition(transition)
        
        assert traj.length == 1
        assert traj.total_reward == 1.0
    
    def test_compute_returns(self):
        """Test computing discounted returns."""
        traj = Trajectory(discount_factor=0.9)
        
        # Add 3 transitions with rewards [1, 2, 3]
        for reward in [1.0, 2.0, 3.0]:
            state = State(state_type=StateType.INTERMEDIATE, user_prompt="test")
            action = Action(thought="", action_type="generate")
            traj.add_transition(Transition(state=state, action=action, reward=reward))
        
        returns = traj.compute_returns()
        
        # G_2 = 3
        # G_1 = 2 + 0.9 * 3 = 4.7
        # G_0 = 1 + 0.9 * 4.7 = 5.23
        assert len(returns) == 3
        assert abs(returns[2] - 3.0) < 0.01
        assert abs(returns[1] - 4.7) < 0.01
        assert abs(returns[0] - 5.23) < 0.01
    
    def test_trajectory_serialization(self):
        """Test trajectory to/from dict."""
        traj = Trajectory(
            trajectory_id="traj-1",
            task_type="text_to_image",
            policy_version="v1.0",
        )
        
        data = traj.to_dict()
        restored = Trajectory.from_dict(data)
        
        assert restored.trajectory_id == "traj-1"
        assert restored.task_type == "text_to_image"


class TestTrajectoryBuilder:
    """Tests for TrajectoryBuilder."""
    
    def test_build_trajectory(self):
        """Test building a complete trajectory."""
        builder = TrajectoryBuilder(
            session_id=1,
            task_type="text_to_image",
        )
        
        traj = (
            builder
            .set_initial_state("Generate a cat image")
            .add_step(
                thought="I will generate the image",
                action_type="generate",
                skill_name="text_to_image",
                skill_params={"prompt": "cat"},
                reward=0.5,
            )
            .add_step(
                thought="Task complete",
                action_type="finish",
                skill_name=None,
                skill_params={},
                reward=1.0,
                done=True,
            )
            .build()
        )
        
        assert traj.length == 2
        assert traj.is_complete
        assert traj.total_reward == 1.5


# ============== Reward Tests ==============

class TestQualityReward:
    """Tests for QualityReward."""
    
    def test_high_quality_reward(self):
        """Test reward for high quality score."""
        reward_fn = QualityReward(quality_threshold=0.7)
        
        signal = reward_fn.compute(
            observation={"result": {"overall_score": 0.9}},
            action={},
        )
        
        assert signal.reward_type == RewardType.QUALITY
        assert signal.value > 0
    
    def test_low_quality_penalty(self):
        """Test penalty for low quality score."""
        reward_fn = QualityReward(quality_threshold=0.7)
        
        signal = reward_fn.compute(
            observation={"result": {"overall_score": 0.4}},
            action={},
        )
        
        assert signal.value < 0
    
    def test_no_score_zero_reward(self):
        """Test zero reward when no score available."""
        reward_fn = QualityReward()
        
        signal = reward_fn.compute(
            observation={"result": {}},
            action={},
        )
        
        assert signal.value == 0.0


class TestEfficiencyReward:
    """Tests for EfficiencyReward."""
    
    def test_optimal_steps_no_penalty(self):
        """Test no penalty for optimal step count."""
        reward_fn = EfficiencyReward(optimal_steps=3)
        
        signal = reward_fn.compute(
            observation={},
            action={},
            context={"step_number": 2},
        )
        
        assert signal.value == 0.0
    
    def test_extra_steps_penalty(self):
        """Test penalty for extra steps."""
        reward_fn = EfficiencyReward(optimal_steps=3, step_penalty=-0.1)
        
        signal = reward_fn.compute(
            observation={},
            action={},
            context={"step_number": 5},
        )
        
        # 2 extra steps * -0.1 = -0.2
        assert signal.value == -0.2


class TestCompositeReward:
    """Tests for CompositeReward."""
    
    def test_composite_combines_rewards(self):
        """Test composite reward combines multiple signals."""
        composite = CompositeReward([
            (QualityReward(), 0.5),
            (EfficiencyReward(), 0.3),
            (SuccessReward(), 0.2),
        ])
        
        signal = composite.compute(
            observation={"result": {"overall_score": 0.8}},
            action={"action_type": "evaluate"},
            context={"step_number": 2},
        )
        
        assert signal.reward_type == RewardType.COMPOSITE
        assert "components" in signal.metadata


class TestRewardNormalizer:
    """Tests for RewardNormalizer."""
    
    def test_normalize_updates_stats(self):
        """Test normalizer updates running statistics."""
        normalizer = RewardNormalizer()
        
        for reward in [1.0, 2.0, 3.0, 4.0, 5.0]:
            normalizer.update(reward)
        
        assert normalizer.mean == 3.0
        assert normalizer.std > 0
    
    def test_normalize_clips_values(self):
        """Test normalization clips extreme values."""
        normalizer = RewardNormalizer(clip_range=(-2.0, 2.0))
        
        for _ in range(10):
            normalizer.update(0.0)
        
        result = normalizer.normalize(100.0)
        assert result <= 2.0


# ============== Cross-Policy Tests ==============

class TestPolicyPool:
    """Tests for PolicyPool."""
    
    def test_add_policy(self):
        """Test adding policies to pool."""
        pool = PolicyPool(max_versions=3)
        
        pool.add_policy("v1.0")
        pool.add_policy("v2.0")
        
        assert len(pool.all_versions) == 2
        assert pool.current_policy.version_id == "v2.0"
    
    def test_max_versions_enforced(self):
        """Test max versions limit is enforced."""
        pool = PolicyPool(max_versions=2)
        
        pool.add_policy("v1.0")
        pool.add_policy("v2.0")
        pool.add_policy("v3.0")
        
        assert len(pool.all_versions) == 2
        assert "v1.0" not in pool.all_versions  # Oldest removed
    
    def test_sample_policy_uniform(self):
        """Test uniform sampling."""
        pool = PolicyPool()
        pool.add_policy("v1.0")
        pool.add_policy("v2.0")
        
        sampled = pool.sample_policy(strategy="uniform")
        
        assert sampled.version_id in ["v1.0", "v2.0"]


class TestCrossPolicySampler:
    """Tests for CrossPolicySampler."""
    
    def test_sample_records_statistics(self):
        """Test sampling records statistics."""
        pool = PolicyPool()
        pool.add_policy("v1.0")
        
        sampler = CrossPolicySampler(policy_pool=pool)
        
        sampler.sample_policy_for_trajectory()
        sampler.sample_policy_for_trajectory()
        
        stats = sampler.get_sampling_statistics()
        assert stats["total_samples"] == 2
    
    def test_record_trajectory_updates_performance(self):
        """Test recording trajectory updates policy performance."""
        pool = PolicyPool()
        pool.add_policy("v1.0")
        
        sampler = CrossPolicySampler(policy_pool=pool)
        
        traj = Trajectory(policy_version="v1.0", discounted_return=5.0)
        sampler.record_trajectory_result(traj)
        
        policy = pool.get_policy("v1.0")
        assert policy.performance_score > 0


# ============== Task Normalization Tests ==============

class TestTaskAdvantageNormalizer:
    """Tests for TaskAdvantageNormalizer."""
    
    def test_update_statistics(self):
        """Test updating task statistics."""
        normalizer = TaskAdvantageNormalizer()
        
        normalizer.update_statistics("text_to_image", [1.0, 2.0, 3.0])
        
        stats = normalizer.get_task_statistics("text_to_image")
        assert stats is not None
        assert stats.count == 3
    
    def test_normalize_per_task(self):
        """Test per-task normalization."""
        normalizer = TaskAdvantageNormalizer(warmup_samples=2)
        
        # Add enough samples for warmup
        normalizer.update_statistics("task_a", [10.0, 20.0, 30.0])
        normalizer.update_statistics("task_b", [0.1, 0.2, 0.3])
        
        # Normalize values from different tasks
        norm_a = normalizer.normalize_advantage("task_a", 20.0)
        norm_b = normalizer.normalize_advantage("task_b", 0.2)
        
        # Both should be close to 0 (near mean)
        assert abs(norm_a) < 1.0
        assert abs(norm_b) < 1.0


class TestGlobalAdvantageNormalizer:
    """Tests for GlobalAdvantageNormalizer."""
    
    def test_global_normalization(self):
        """Test global normalization across all tasks."""
        normalizer = GlobalAdvantageNormalizer()
        
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            normalizer.update(val)
        
        normalized = normalizer.normalize(3.0)  # Mean value
        assert abs(normalized) < 0.1  # Should be close to 0


# ============== Replay Buffer Tests ==============

class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_add_trajectory(self):
        """Test adding trajectories."""
        buffer = ReplayBuffer(capacity=100, min_size=1)
        
        traj = Trajectory()
        buffer.add(traj)
        
        assert buffer.size == 1
    
    def test_sample_uniform(self):
        """Test uniform sampling."""
        buffer = ReplayBuffer(capacity=100, min_size=1)
        
        for i in range(10):
            traj = Trajectory(trajectory_id=f"traj-{i}")
            buffer.add(traj)
        
        samples = buffer.sample(5)
        assert len(samples) == 5
    
    def test_capacity_limit(self):
        """Test buffer respects capacity."""
        buffer = ReplayBuffer(capacity=5, min_size=1)
        
        for i in range(10):
            buffer.add(Trajectory())
        
        assert buffer.size == 5


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""
    
    def test_sample_with_priorities(self):
        """Test prioritized sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, min_size=1)
        
        # Add with different priorities
        for i in range(10):
            buffer.add(Trajectory(), priority=float(i + 1))
        
        trajs, indices, weights = buffer.sample(5)
        
        assert len(trajs) == 5
        assert len(indices) == 5
        assert len(weights) == 5
    
    def test_update_priorities(self):
        """Test updating priorities."""
        buffer = PrioritizedReplayBuffer(capacity=100, min_size=1)
        
        for _ in range(5):
            buffer.add(Trajectory(), priority=1.0)
        
        buffer.update_priorities([0, 1], [10.0, 10.0])
        
        # High priority items should be sampled more
        # (statistical test would be better here)


class TestTaskSpecificBuffer:
    """Tests for TaskSpecificBuffer."""
    
    def test_separate_buffers_per_task(self):
        """Test separate buffers for each task."""
        buffer = TaskSpecificBuffer()
        
        buffer.add(Trajectory(task_type="task_a"))
        buffer.add(Trajectory(task_type="task_a"))
        buffer.add(Trajectory(task_type="task_b"))
        
        stats = buffer.get_statistics()
        assert stats["task_a"]["size"] == 2
        assert stats["task_b"]["size"] == 1
    
    def test_balanced_sampling(self):
        """Test balanced sampling across tasks."""
        buffer = TaskSpecificBuffer(min_size_per_task=1)
        
        for _ in range(5):
            buffer.add(Trajectory(task_type="task_a"))
        for _ in range(5):
            buffer.add(Trajectory(task_type="task_b"))
        
        samples = buffer.sample(4)
        assert len(samples) == 4


# ============== Trainer Tests ==============

class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.total_epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 3e-4
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = TrainingConfig(total_epochs=50)
        
        data = config.to_dict()
        assert data["total_epochs"] == 50


class TestRLTrainer:
    """Tests for RLTrainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        trainer = RLTrainer()
        
        assert trainer.status == TrainerStatus.IDLE
        assert trainer._current_epoch == 0
    
    def test_add_trajectory(self):
        """Test adding trajectory to trainer."""
        trainer = RLTrainer()
        
        traj = Trajectory(task_type="text_to_image")
        trainer.add_trajectory(traj)
        
        assert trainer.replay_buffer.size == 1
    
    def test_train_step_requires_min_buffer(self):
        """Test training requires minimum buffer size."""
        config = TrainingConfig(buffer_size=100, min_buffer_size=10)
        trainer = RLTrainer(config=config)
        
        # Only add 5 trajectories (below min)
        for _ in range(5):
            trainer.add_trajectory(Trajectory())
        
        metrics = trainer.train_step()
        
        # Should return empty metrics since buffer not ready
        assert metrics.epoch == 0
    
    def test_training_summary(self):
        """Test getting training summary."""
        trainer = RLTrainer()
        
        summary = trainer.get_training_summary()
        
        assert "status" in summary
        assert "current_epoch" in summary
        assert "config" in summary
    
    def test_pause_resume(self):
        """Test pause and resume functionality."""
        trainer = RLTrainer()
        
        trainer.pause()
        assert trainer.status == TrainerStatus.PAUSED
        
        trainer.resume()
        assert trainer.status == TrainerStatus.TRAINING
    
    def test_reset(self):
        """Test resetting trainer state."""
        trainer = RLTrainer()
        trainer._current_epoch = 10
        trainer.add_trajectory(Trajectory())
        
        trainer.reset()
        
        assert trainer._current_epoch == 0
        assert trainer.replay_buffer.size == 0
        assert trainer.status == TrainerStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_train_epoch(self):
        """Test running a training epoch."""
        config = TrainingConfig(
            steps_per_epoch=5,
            buffer_size=100,
            min_buffer_size=5,
        )
        trainer = RLTrainer(config=config)
        
        # Add enough trajectories
        for _ in range(10):
            traj = Trajectory()
            # Add a transition so trajectory has content
            state = State(state_type=StateType.INITIAL, user_prompt="test")
            action = Action(thought="test", action_type="generate")
            traj.add_transition(Transition(state=state, action=action, reward=1.0))
            trainer.add_trajectory(traj)
        
        metrics = await trainer.train_epoch()
        
        assert len(metrics) == 5
        assert trainer._current_epoch == 1
