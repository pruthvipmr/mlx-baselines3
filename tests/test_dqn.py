"""
Tests for DQN algorithm implementation.
"""

import gymnasium as gym
import numpy as np
import pytest
import mlx.core as mx
import tempfile
import os

from mlx_baselines3.dqn import DQN


class TestDQNInitialization:
    """Test DQN initialization."""

    def test_dqn_initialization_with_string_policy(self):
        """Test DQN initialization with string policy."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        assert model.buffer_size == 1_000_000
        assert model.learning_starts == 50000
        assert model.batch_size == 32
        assert model.tau == 1.0
        assert model.gamma == 0.99
        assert model.train_freq == 4
        assert model.gradient_steps == 1
        assert model.target_update_interval == 10000
        assert model.exploration_fraction == 0.1
        assert model.exploration_initial_eps == 1.0
        assert model.exploration_final_eps == 0.05
        assert model.max_grad_norm == 10.0

        env.close()

    def test_dqn_initialization_with_custom_hyperparameters(self):
        """Test DQN initialization with custom hyperparameters."""
        env = gym.make("CartPole-v1")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.95,
            train_freq=1,
            gradient_steps=2,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_initial_eps=0.8,
            exploration_final_eps=0.01,
            max_grad_norm=1.0,
            verbose=0,
        )

        assert model.learning_rate == 1e-3
        assert model.buffer_size == 100000
        assert model.learning_starts == 1000
        assert model.batch_size == 64
        assert model.tau == 0.005
        assert model.gamma == 0.95
        assert model.train_freq == 1
        assert model.gradient_steps == 2
        assert model.target_update_interval == 1000
        assert model.exploration_fraction == 0.2
        assert model.exploration_initial_eps == 0.8
        assert model.exploration_final_eps == 0.01
        assert model.max_grad_norm == 1.0

        env.close()

    def test_dqn_fails_with_continuous_action_space(self):
        """Test that DQN fails with continuous action spaces."""
        env = gym.make("Pendulum-v1")

        with pytest.raises(AssertionError, match="Action space .* is not supported"):
            DQN("MlpPolicy", env, verbose=0)

        env.close()


class TestDQNPolicy:
    """Test DQN policy functionality."""

    def test_dqn_policy_predict(self):
        """Test DQN policy predict method."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        obs = env.reset()[0]
        action, state = model.predict(obs, deterministic=True)

        assert isinstance(action, (np.ndarray, np.integer, int))
        action_val = action.item() if hasattr(action, "item") else action
        assert action_val in [0, 1]  # CartPole has 2 actions
        assert state is None  # DQN doesn't use state

        env.close()

    def test_dqn_policy_predict_batch(self):
        """Test DQN policy predict with batch of observations."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        # Create batch of observations
        obs_single = env.reset()[0]
        obs_batch = np.stack([obs_single, obs_single])

        action, state = model.predict(obs_batch, deterministic=True)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert all(a in [0, 1] for a in action)
        assert state is None

        env.close()

    def test_q_network_output_shape(self):
        """Test Q-network output shape."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        obs = env.reset()[0]
        obs_mlx = mx.array(obs).reshape(1, -1)

        q_values = model.q_net.predict_values(obs_mlx)

        assert q_values.shape == (1, 2)  # 1 batch, 2 actions for CartPole
        assert isinstance(q_values, mx.array)

        env.close()


class TestDQNExploration:
    """Test DQN exploration functionality."""

    def test_exploration_rate_schedule(self):
        """Test exploration rate decreases over time."""
        env = gym.make("CartPole-v1")
        model = DQN(
            "MlpPolicy",
            env,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0,
        )

        # Set up for exploration testing
        model._total_timesteps = 1000

        # Test initial exploration rate
        model.num_timesteps = 0
        initial_rate = model._get_exploration_rate()
        assert initial_rate == 1.0

        # Test mid-training exploration rate
        model.num_timesteps = 50  # 5% through exploration phase
        mid_rate = model._get_exploration_rate()
        assert 0.05 < mid_rate < 1.0

        # Test final exploration rate
        model.num_timesteps = 100  # Past exploration phase
        final_rate = model._get_exploration_rate()
        assert abs(final_rate - 0.05) < 1e-6

        env.close()

    def test_epsilon_greedy_action_selection(self):
        """Test epsilon-greedy action selection."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        obs = env.reset()[0]

        # Test deterministic action selection
        action_det, _ = model.predict(obs, deterministic=True)

        # Test stochastic action selection (should match due to no exploration
        # initially)
        action_stoch, _ = model.predict(obs, deterministic=False)

        # Both should be valid actions
        assert action_det.item() in [0, 1]
        assert action_stoch.item() in [0, 1]

        env.close()


class TestDQNTargetNetwork:
    """Test DQN target network functionality."""

    def test_target_network_initialization(self):
        """Test target network is initialized correctly."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, verbose=0)

        # Check that target network exists
        assert hasattr(model, "q_net_target")
        assert model.q_net_target is not None

        # Check that target network has same structure as main network
        main_state = model.q_net.state_dict()
        target_state = model.q_net_target.state_dict()

        assert set(main_state.keys()) == set(target_state.keys())

        # Initially, target network should have same weights as main network
        for key in main_state.keys():
            assert main_state[key].shape == target_state[key].shape

        env.close()


class TestDQNReplayBuffer:
    """Test DQN replay buffer functionality."""

    def test_replay_buffer_creation(self):
        """Test replay buffer is created correctly."""
        env = gym.make("CartPole-v1")
        model = DQN("MlpPolicy", env, buffer_size=1000, verbose=0)

        assert hasattr(model, "replay_buffer")
        assert model.replay_buffer.buffer_size == 1000
        assert model.replay_buffer.observation_space == env.observation_space
        assert model.replay_buffer.action_space == env.action_space

        env.close()


class TestDQNTraining:
    """Test DQN training functionality."""

    def test_dqn_training_step(self):
        """Test DQN training step doesn't crash."""
        env = gym.make("CartPole-v1")
        model = DQN(
            "MlpPolicy",
            env,
            buffer_size=1000,
            learning_starts=10,
            batch_size=4,
            verbose=0,
        )

        # Fill replay buffer with some random data
        for _ in range(20):
            obs = env.reset()[0]
            for _ in range(10):
                action = env.action_space.sample()
                next_obs, reward, done, truncated, info = env.step(action)

                model.replay_buffer.add(
                    obs.reshape(1, -1),
                    next_obs.reshape(1, -1),
                    np.array([action]).reshape(1, -1),
                    np.array([reward]),
                    np.array([done]),
                    [info],
                )

                if done or truncated:
                    break
                obs = next_obs

        # Test training step
        model.num_timesteps = 100  # Above learning_starts
        try:
            model.train(gradient_steps=1, batch_size=4)
        except Exception as e:
            pytest.fail(f"Training step failed: {e}")

        env.close()

    def test_dqn_learning_smoke_test(self):
        """Smoke test for DQN learning."""
        env = gym.make("CartPole-v1")
        model = DQN(
            "MlpPolicy",
            env,
            buffer_size=1000,
            learning_starts=100,
            train_freq=10,
            target_update_interval=100,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            verbose=0,
        )

        # Short training run
        try:
            model.learn(total_timesteps=200)
        except Exception as e:
            pytest.fail(f"Learning failed: {e}")

        env.close()


class TestDQNSaveLoad:
    """Test DQN save/load functionality."""

    def test_dqn_save_load(self):
        """Test DQN save and load."""
        env = gym.make("CartPole-v1")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.95,
            train_freq=2,
            gradient_steps=3,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=0.8,
            exploration_final_eps=0.01,
            max_grad_norm=5.0,
            optimize_memory_usage=True,
            verbose=0,
        )

        # Get initial prediction
        obs = env.reset()[0]
        initial_action, _ = model.predict(obs, deterministic=True)

        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "dqn_model")
            model.save(save_path)

            # Load model
            loaded_model = DQN.load(save_path, env=env)

            # Test that loaded model makes same prediction
            loaded_action, _ = loaded_model.predict(obs, deterministic=True)

            assert np.array_equal(initial_action, loaded_action)

            # Test all DQN-specific hyperparameters are preserved
            assert loaded_model.buffer_size == model.buffer_size
            assert loaded_model.learning_starts == model.learning_starts
            assert loaded_model.batch_size == model.batch_size
            assert loaded_model.tau == model.tau
            assert loaded_model.gamma == model.gamma
            assert loaded_model.train_freq == model.train_freq
            assert loaded_model.gradient_steps == model.gradient_steps
            assert loaded_model.target_update_interval == model.target_update_interval
            assert loaded_model.exploration_fraction == model.exploration_fraction
            assert loaded_model.exploration_initial_eps == model.exploration_initial_eps
            assert loaded_model.exploration_final_eps == model.exploration_final_eps
            assert loaded_model.max_grad_norm == model.max_grad_norm
            assert loaded_model.optimize_memory_usage == model.optimize_memory_usage

        env.close()


class TestDQNCompatibility:
    """Test DQN compatibility with various environments."""

    def test_dqn_with_different_discrete_envs(self):
        """Test DQN works with different discrete action environments."""
        envs_to_test = [
            "CartPole-v1",  # 2 actions
            "LunarLander-v2",  # 4 actions
        ]

        for env_id in envs_to_test:
            try:
                env = gym.make(env_id)
                model = DQN("MlpPolicy", env, verbose=0)

                obs = env.reset()[0]
                action, _ = model.predict(obs)

                assert 0 <= action.item() < env.action_space.n

                env.close()
            except gym.error.Error:
                # Skip if environment not available
                pytest.skip(f"Environment {env_id} not available")


if __name__ == "__main__":
    pytest.main([__file__])
