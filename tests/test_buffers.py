"""
Tests for experience buffers.

This module tests the RolloutBuffer and ReplayBuffer implementations to ensure
they work correctly with different observation/action spaces and MLX integration.
"""

import pytest
import numpy as np
import gymnasium as gym
import mlx.core as mx

from mlx_baselines3.common.buffers import BaseBuffer, RolloutBuffer, ReplayBuffer


class TestBaseBuffer:
    """Test the base buffer functionality."""
    
    def test_box_observation_space(self):
        """Test buffer with Box observation space."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        # This is an abstract class, so we'll create a minimal concrete implementation
        class TestBuffer(BaseBuffer):
            def _setup_storage(self):
                super()._setup_storage()
                
        buffer = TestBuffer(10, obs_space, action_space, n_envs=2)
        
        assert buffer.buffer_size == 10
        assert buffer.n_envs == 2
        assert buffer.observations.shape == (10, 2, 4)
        assert buffer.actions.shape == (10, 2)
        assert buffer.size() == 0
        
    def test_dict_observation_space(self):
        """Test buffer with Dict observation space."""
        obs_space = gym.spaces.Dict({
            "position": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "velocity": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })
        action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        class TestBuffer(BaseBuffer):
            def _setup_storage(self):
                super()._setup_storage()
                
        buffer = TestBuffer(5, obs_space, action_space, n_envs=1)
        
        assert isinstance(buffer.observations, dict)
        assert "position" in buffer.observations
        assert "velocity" in buffer.observations
        assert buffer.observations["position"].shape == (5, 1, 2)
        assert buffer.observations["velocity"].shape == (5, 1, 2)
        assert buffer.actions.shape == (5, 1, 1)
        
    def test_reset(self):
        """Test buffer reset functionality."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        class TestBuffer(BaseBuffer):
            def _setup_storage(self):
                super()._setup_storage()
                
        buffer = TestBuffer(10, obs_space, action_space)
        
        buffer.pos = 5
        buffer.full = True
        buffer.reset()
        
        assert buffer.pos == 0
        assert buffer.full is False


class TestRolloutBuffer:
    """Test the RolloutBuffer implementation."""
    
    def test_initialization(self):
        """Test RolloutBuffer initialization."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        buffer = RolloutBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=2
        )
        
        assert buffer.buffer_size == 10
        assert buffer.n_envs == 2
        assert buffer.gae_lambda == 0.95
        assert buffer.gamma == 0.99
        assert buffer.rewards.shape == (10, 2)
        assert buffer.values.shape == (10, 2)
        assert buffer.log_probs.shape == (10, 2)
        assert buffer.advantages.shape == (10, 2)
        assert buffer.returns.shape == (10, 2)
        
    def test_add_data(self):
        """Test adding data to RolloutBuffer."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        n_envs = 2
        
        buffer = RolloutBuffer(5, obs_space, action_space, n_envs=n_envs)
        
        # Add one step of data
        obs = np.random.randn(n_envs, 4).astype(np.float32)
        action = np.array([0, 1])
        reward = np.array([1.0, 0.5], dtype=np.float32)
        episode_start = np.array([False, True])
        value = np.array([0.8, 0.3], dtype=np.float32)
        log_prob = np.array([-0.1, -0.7], dtype=np.float32)
        
        buffer.add(obs, action, reward, episode_start, value, log_prob)
        
        assert buffer.pos == 1
        assert buffer.size() == 1
        assert np.array_equal(buffer.observations[0], obs)
        assert np.array_equal(buffer.actions[0], action)
        assert np.array_equal(buffer.rewards[0], reward)
        assert np.array_equal(buffer.episode_starts[0], episode_start)
        assert np.array_equal(buffer.values[0], value)
        assert np.array_equal(buffer.log_probs[0], log_prob)
        
    def test_fill_buffer(self):
        """Test filling the buffer completely."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 3
        n_envs = 2
        
        buffer = RolloutBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)
        
        # Fill the buffer
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 2).astype(np.float32)
            action = np.random.randint(0, 2, size=n_envs)
            reward = np.random.rand(n_envs).astype(np.float32)
            episode_start = np.array([False, False])
            value = np.random.rand(n_envs).astype(np.float32)
            log_prob = np.random.randn(n_envs).astype(np.float32)
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
        assert buffer.full is True
        assert buffer.size() == buffer_size
        
        # Adding more should raise an error
        with pytest.raises(ValueError):
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
    def test_compute_returns_and_advantage(self):
        """Test GAE computation."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 3
        n_envs = 1
        gamma = 0.99
        gae_lambda = 0.95
        
        buffer = RolloutBuffer(
            buffer_size, obs_space, action_space, 
            gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs
        )
        
        # Add data with known values
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 2).astype(np.float32)
            action = np.array([0])
            reward = np.array([rewards[i]])
            episode_start = np.array([False])
            value = np.array([values[i]])
            log_prob = np.array([-0.1])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
        # Compute returns and advantages
        last_values = np.array([2.0])  # Value for the final state
        dones = np.array([True])  # Episode ends
        
        buffer.compute_returns_and_advantage(last_values, dones)
        
        # Check that advantages and returns were computed
        assert buffer.advantages.shape == (buffer_size, n_envs)
        assert buffer.returns.shape == (buffer_size, n_envs)
        
        # Returns should be advantages + values
        np.testing.assert_array_almost_equal(
            buffer.returns, buffer.advantages + buffer.values
        )
        
    def test_get_batches(self):
        """Test getting training batches."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 4
        n_envs = 2
        
        buffer = RolloutBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)
        
        # Fill buffer
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 2).astype(np.float32)
            action = np.random.randint(0, 2, size=n_envs)
            reward = np.random.rand(n_envs).astype(np.float32)
            episode_start = np.array([False, False])
            value = np.random.rand(n_envs).astype(np.float32)
            log_prob = np.random.randn(n_envs).astype(np.float32)
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
        # Compute advantages
        buffer.compute_returns_and_advantage(
            np.random.rand(n_envs).astype(np.float32),
            np.array([False, False])
        )
        
        # Test getting batches
        batch_size = 4
        batches = list(buffer.get(batch_size))
        
        assert len(batches) == (buffer_size * n_envs) // batch_size
        
        for batch in batches:
            assert isinstance(batch, dict)
            assert "observations" in batch
            assert "actions" in batch
            assert "values" in batch
            assert "log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch
            
            # Check that data is converted to MLX arrays
            assert isinstance(batch["observations"], mx.array)
            assert isinstance(batch["actions"], mx.array)
            assert isinstance(batch["values"], mx.array)
            
            # Check batch size
            assert batch["observations"].shape[0] == batch_size
            
    def test_get_all_data(self):
        """Test getting all data at once."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 3
        n_envs = 2
        
        buffer = RolloutBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)
        
        # Fill buffer
        for i in range(buffer_size):
            obs = np.random.randn(n_envs, 2).astype(np.float32)
            action = np.random.randint(0, 2, size=n_envs)
            reward = np.random.rand(n_envs).astype(np.float32)
            episode_start = np.array([False, False])
            value = np.random.rand(n_envs).astype(np.float32)
            log_prob = np.random.randn(n_envs).astype(np.float32)
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
        buffer.compute_returns_and_advantage(
            np.random.rand(n_envs).astype(np.float32),
            np.array([False, False])
        )
        
        # Get all data
        batches = list(buffer.get(batch_size=None))
        assert len(batches) == 1
        
        batch = batches[0]
        assert batch["observations"].shape[0] == buffer_size * n_envs
        
    def test_dict_observations(self):
        """Test RolloutBuffer with dictionary observations."""
        obs_space = gym.spaces.Dict({
            "position": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "velocity": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })
        action_space = gym.spaces.Discrete(2)
        
        buffer = RolloutBuffer(3, obs_space, action_space, n_envs=1)
        
        # Add data with dict observations
        for i in range(3):
            obs = {
                "position": np.random.randn(1, 2).astype(np.float32),
                "velocity": np.random.randn(1, 2).astype(np.float32),
            }
            action = np.array([0])
            reward = np.array([1.0])
            episode_start = np.array([False])
            value = np.array([0.5])
            log_prob = np.array([-0.1])
            
            buffer.add(obs, action, reward, episode_start, value, log_prob)
            
        buffer.compute_returns_and_advantage(np.array([0.0]), np.array([True]))
        
        # Test sampling
        batches = list(buffer.get(batch_size=2))
        batch = batches[0]
        
        assert isinstance(batch["observations"], dict)
        assert "position" in batch["observations"]
        assert "velocity" in batch["observations"]
        assert isinstance(batch["observations"]["position"], mx.array)


class TestReplayBuffer:
    """Test the ReplayBuffer implementation."""
    
    def test_initialization(self):
        """Test ReplayBuffer initialization."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        buffer = ReplayBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            n_envs=1
        )
        
        assert buffer.buffer_size == 100
        assert buffer.n_envs == 1
        assert buffer.rewards.shape == (100, 1)
        assert buffer.dones.shape == (100, 1)
        assert hasattr(buffer, 'next_observations')
        
    def test_memory_optimization(self):
        """Test memory optimization mode."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        buffer = ReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            optimize_memory_usage=True
        )
        
        assert buffer.optimize_memory_usage is True
        assert not hasattr(buffer, 'next_observations')
        
    def test_add_data(self):
        """Test adding data to ReplayBuffer."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        n_envs = 2
        
        buffer = ReplayBuffer(10, obs_space, action_space, n_envs=n_envs)
        
        # Add one transition
        obs = np.random.randn(n_envs, 3).astype(np.float32)
        next_obs = np.random.randn(n_envs, 3).astype(np.float32)
        action = np.array([0, 1])
        reward = np.array([1.0, -0.5])
        done = np.array([False, True])
        infos = [{}, {}]
        
        buffer.add(obs, next_obs, action, reward, done, infos)
        
        assert buffer.pos == 1
        assert buffer.size() == 1
        assert np.array_equal(buffer.observations[0], obs)
        assert np.array_equal(buffer.next_observations[0], next_obs)
        assert np.array_equal(buffer.actions[0], action)
        assert np.array_equal(buffer.rewards[0], reward)
        assert np.array_equal(buffer.dones[0], done)
        
    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 3
        n_envs = 1
        
        buffer = ReplayBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)
        
        # Fill buffer beyond capacity
        for i in range(buffer_size + 2):
            obs = np.array([[float(i)]])
            next_obs = np.array([[float(i + 1)]])
            action = np.array([i % 2])
            reward = np.array([float(i)])
            done = np.array([False])
            infos = [{}]
            
            buffer.add(obs, next_obs, action, reward, done, infos)
            
        assert buffer.full is True
        assert buffer.pos == 2  # Should wrap around
        assert buffer.size() == buffer_size
        
        # Check that old data was overwritten
        assert buffer.observations[0, 0, 0] == 3.0  # Should be overwritten
        assert buffer.observations[1, 0, 0] == 4.0  # Should be overwritten
        assert buffer.observations[2, 0, 0] == 2.0  # Should be original
        
    def test_sample(self):
        """Test sampling from ReplayBuffer."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        buffer_size = 10
        n_envs = 1
        
        buffer = ReplayBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)
        
        # Add some data
        for i in range(5):
            obs = np.random.randn(n_envs, 2).astype(np.float32)
            next_obs = np.random.randn(n_envs, 2).astype(np.float32)
            action = np.array([i % 2])
            reward = np.array([float(i)])
            done = np.array([False])
            infos = [{}]
            
            buffer.add(obs, next_obs, action, reward, done, infos)
            
        # Sample a batch
        batch_size = 3
        batch = buffer.sample(batch_size)
        
        assert isinstance(batch, dict)
        assert "observations" in batch
        assert "actions" in batch
        assert "next_observations" in batch
        assert "rewards" in batch
        assert "dones" in batch
        
        # Check batch dimensions
        assert batch["observations"].shape[0] == batch_size
        assert batch["actions"].shape[0] == batch_size
        assert batch["next_observations"].shape[0] == batch_size
        assert batch["rewards"].shape[0] == batch_size
        assert batch["dones"].shape[0] == batch_size
        
        # Check MLX array types
        assert isinstance(batch["observations"], mx.array)
        assert isinstance(batch["actions"], mx.array)
        assert isinstance(batch["next_observations"], mx.array)
        assert isinstance(batch["rewards"], mx.array)
        assert isinstance(batch["dones"], mx.array)
        
    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer raises error."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
            buffer.sample(1)
            
    def test_dict_observations_replay(self):
        """Test ReplayBuffer with dictionary observations."""
        obs_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8),
            "vector": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        })
        action_space = gym.spaces.Discrete(4)
        
        buffer = ReplayBuffer(5, obs_space, action_space, n_envs=1)
        
        # Add data with dict observations
        for i in range(3):
            obs = {
                "image": np.random.randint(0, 256, (1, 3, 32, 32), dtype=np.uint8),
                "vector": np.random.randn(1, 4).astype(np.float32),
            }
            next_obs = {
                "image": np.random.randint(0, 256, (1, 3, 32, 32), dtype=np.uint8),
                "vector": np.random.randn(1, 4).astype(np.float32),
            }
            action = np.array([i % 4])
            reward = np.array([1.0])
            done = np.array([False])
            infos = [{}]
            
            buffer.add(obs, next_obs, action, reward, done, infos)
            
        # Sample and test
        batch = buffer.sample(2)
        
        assert isinstance(batch["observations"], dict)
        assert isinstance(batch["next_observations"], dict)
        assert "image" in batch["observations"]
        assert "vector" in batch["observations"]
        assert isinstance(batch["observations"]["image"], mx.array)
        assert isinstance(batch["observations"]["vector"], mx.array)
        
    def test_memory_optimized_next_obs(self):
        """Test memory optimized next observation computation."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        
        buffer = ReplayBuffer(
            buffer_size=5,
            observation_space=obs_space,
            action_space=action_space,
            optimize_memory_usage=True,
            n_envs=1
        )
        
        # Add some transitions
        for i in range(3):
            obs = np.array([[float(i), float(i)]])
            next_obs = np.array([[float(i+1), float(i+1)]])  # Not stored
            action = np.array([0])
            reward = np.array([1.0])
            done = np.array([i == 2])  # Last one is done
            infos = [{}]
            
            buffer.add(obs, next_obs, action, reward, done, infos)
            
        # Sample and check that next observations are computed correctly
        batch = buffer.sample(2)
        
        # Should have computed next observations from stored observations
        assert isinstance(batch["next_observations"], mx.array)
        assert batch["next_observations"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__])
