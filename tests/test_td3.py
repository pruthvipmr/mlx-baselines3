"""Tests for TD3 algorithm."""

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from mlx_baselines3 import TD3
from mlx_baselines3.common.buffers import ReplayBuffer
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.td3.policies import TD3Policy, MlpPolicy


class TestTD3:
    """Test suite for TD3 algorithm."""

    def test_td3_init(self):
        """Test TD3 initialization."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        assert model.policy is not None
        assert hasattr(model, "policy_delay")
        assert hasattr(model, "target_policy_noise")
        assert hasattr(model, "target_noise_clip")
        assert model.policy_delay == 2  # Default value
        assert model.target_policy_noise == 0.2  # Default value
        assert model.target_noise_clip == 0.5  # Default value

    def test_td3_with_continuous_action_space(self):
        """Test TD3 with continuous action space (should work)."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        assert isinstance(model.action_space, gym.spaces.Box)

    def test_td3_with_discrete_action_space_raises_error(self):
        """Test TD3 with discrete action space (should raise error)."""
        env = gym.make("CartPole-v1")
        with pytest.raises(AssertionError, match="Action space .* is not supported"):
            TD3("MlpPolicy", env, verbose=0)

    def test_td3_custom_hyperparams(self):
        """Test TD3 with custom hyperparameters."""
        env = gym.make("Pendulum-v1")
        model = TD3(
            "MlpPolicy", 
            env, 
            learning_rate=1e-4,
            buffer_size=50000,
            batch_size=128,
            tau=0.01,
            gamma=0.95,
            policy_delay=3,
            target_policy_noise=0.1,
            target_noise_clip=0.3,
            verbose=0
        )
        
        assert model.learning_rate == 1e-4
        assert model.buffer_size == 50000
        assert model.batch_size == 128
        assert model.tau == 0.01
        assert model.gamma == 0.95
        assert model.policy_delay == 3
        assert model.target_policy_noise == 0.1
        assert model.target_noise_clip == 0.3

    def test_td3_policy_creation(self):
        """Test TD3 policy creation and basic functionality."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        assert isinstance(model.policy, TD3Policy)
        assert model.policy.n_critics == 2  # Twin critics
        assert hasattr(model.policy, "actor_net")
        assert hasattr(model.policy, "q_networks")
        assert hasattr(model.policy, "actor_target_net")
        assert hasattr(model.policy, "actor_target_output")
        assert hasattr(model.policy, "q_networks_target")
        assert len(model.policy.q_networks) == 2
        assert len(model.policy.q_networks_target) == 2

    def test_td3_predict(self):
        """Test TD3 prediction."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        assert action.shape == env.action_space.shape
        assert env.action_space.contains(action)

    def test_td3_predict_batch(self):
        """Test TD3 prediction with batch of observations."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        batch_size = 5
        obs_batch = np.random.random((batch_size,) + env.observation_space.shape).astype(np.float32)
        
        # For single env, predict should handle batch by processing each observation
        actions = []
        for obs in obs_batch:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        
        actions = np.array(actions)
        assert actions.shape == (batch_size,) + env.action_space.shape

    def test_td3_actor_forward(self):
        """Test TD3 actor forward pass."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        obs_tensor = mx.array(obs).reshape(1, -1)
        features = model.policy.extract_features(obs_tensor)
        
        actions = model.policy.actor_forward(features)
        
        assert isinstance(actions, mx.array)
        assert actions.shape == (1, env.action_space.shape[0])
        
        # Convert to numpy to check bounds
        actions_np = np.array(actions)
        assert np.all(actions_np >= env.action_space.low)
        assert np.all(actions_np <= env.action_space.high)

    def test_td3_actor_target_forward(self):
        """Test TD3 target actor forward pass."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        obs_tensor = mx.array(obs).reshape(1, -1)
        features = model.policy.extract_features(obs_tensor)
        
        # Test without noise
        actions = model.policy.actor_target_forward(features)
        assert isinstance(actions, mx.array)
        assert actions.shape == (1, env.action_space.shape[0])
        
        # Test with noise
        noise = mx.random.normal(shape=actions.shape) * 0.1
        actions_with_noise = model.policy.actor_target_forward(features, noise=noise)
        assert isinstance(actions_with_noise, mx.array)
        assert actions_with_noise.shape == (1, env.action_space.shape[0])

    def test_td3_critic_forward(self):
        """Test TD3 critic forward pass."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        obs_tensor = mx.array(obs).reshape(1, -1)
        features = model.policy.extract_features(obs_tensor)
        actions = mx.random.uniform(-1, 1, shape=(1, env.action_space.shape[0]))
        
        q_values = model.policy.critic_forward(features, actions)
        
        assert isinstance(q_values, list)
        assert len(q_values) == 2  # Twin critics
        for q_val in q_values:
            assert isinstance(q_val, mx.array)
            assert q_val.shape == (1, 1)

    def test_td3_critic_target_forward(self):
        """Test TD3 target critic forward pass."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        obs_tensor = mx.array(obs).reshape(1, -1)
        features = model.policy.extract_features(obs_tensor)
        actions = mx.random.uniform(-1, 1, shape=(1, env.action_space.shape[0]))
        
        q_values = model.policy.critic_target_forward(features, actions)
        
        assert isinstance(q_values, list)
        assert len(q_values) == 2  # Twin target critics
        for q_val in q_values:
            assert isinstance(q_val, mx.array)
            assert q_val.shape == (1, 1)

    def test_td3_replay_buffer_setup(self):
        """Test TD3 replay buffer setup."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, buffer_size=1000, verbose=0)
        
        assert model.replay_buffer is not None
        assert isinstance(model.replay_buffer, ReplayBuffer)
        assert model.replay_buffer.buffer_size == 1000

    def test_td3_short_training(self):
        """Test TD3 short training run to ensure no errors."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, learning_starts=10, verbose=0)
        
        # Should run without errors
        model.learn(total_timesteps=50)
        
        # Check that some training occurred
        assert model.num_timesteps >= 50
        assert hasattr(model, "_n_updates")

    def test_td3_save_load(self):
        """Test TD3 save and load functionality."""
        import tempfile
        import os
        
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        # Get initial prediction
        obs, _ = env.reset()
        action_before, _ = model.predict(obs, deterministic=True)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "td3_model")
            model.save(save_path)
            
            # Load model
            loaded_model = TD3.load(save_path, env=env)
            
            # Compare predictions
            action_after, _ = loaded_model.predict(obs, deterministic=True)
            np.testing.assert_array_almost_equal(action_before, action_after, decimal=5)

    def test_td3_deterministic_prediction(self):
        """Test that TD3 predictions are deterministic when specified."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        obs, _ = env.reset()
        
        # Multiple predictions should be identical for deterministic=True
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = model.predict(obs, deterministic=True)
        
        np.testing.assert_array_equal(action1, action2)

    def test_td3_parameter_count(self):
        """Test TD3 parameter counting and access."""
        env = gym.make("Pendulum-v1")
        model = TD3("MlpPolicy", env, verbose=0)
        
        params = model.policy.parameters()
        
        # Should have parameters for actor, critics, and targets
        actor_params = [name for name in params.keys() if "actor_net" in name and "target" not in name]
        critic_params = [name for name in params.keys() if "q_net_" in name and "target" not in name]
        target_actor_params = [name for name in params.keys() if "actor_target" in name]
        target_critic_params = [name for name in params.keys() if "q_net_target_" in name]
        
        assert len(actor_params) > 0
        assert len(critic_params) > 0
        assert len(target_actor_params) > 0
        assert len(target_critic_params) > 0
        
        # Should have twin critics
        critic_0_params = [name for name in critic_params if "q_net_0" in name]
        critic_1_params = [name for name in critic_params if "q_net_1" in name]
        assert len(critic_0_params) > 0
        assert len(critic_1_params) > 0


class TestTD3Policy:
    """Test suite for TD3 policy components."""

    def test_td3_policy_init(self):
        """Test TD3Policy initialization."""
        env = gym.make("Pendulum-v1")
        policy = TD3Policy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: 1e-3
        )
        
        assert policy.action_dim == env.action_space.shape[0]
        assert policy.n_critics == 2

    def test_td3_policy_deterministic_forward(self):
        """Test that TD3 policy forward pass is deterministic."""
        env = gym.make("Pendulum-v1")
        policy = TD3Policy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: 1e-3
        )
        
        obs, _ = env.reset()
        obs_tensor = mx.array(obs).reshape(1, -1)
        
        # Multiple forward passes should give identical results
        actions1, _, log_probs1 = policy.forward(obs_tensor, deterministic=True)
        actions2, _, log_probs2 = policy.forward(obs_tensor, deterministic=True)
        
        np.testing.assert_array_equal(np.array(actions1), np.array(actions2))
        np.testing.assert_array_equal(np.array(log_probs1), np.array(log_probs2))

    def test_mlp_policy_creation(self):
        """Test MlpPolicy creation."""
        env = gym.make("Pendulum-v1")
        policy = MlpPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: 1e-3
        )
        
        assert isinstance(policy, TD3Policy)
        assert policy.action_dim == env.action_space.shape[0]

    def test_cnn_policy_not_implemented(self):
        """Test that CnnPolicy raises NotImplementedError."""
        env = gym.make("Pendulum-v1")
        
        from mlx_baselines3.td3.policies import CnnPolicy
        with pytest.raises(NotImplementedError):
            CnnPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda x: 1e-3
            )

    def test_multi_input_policy_not_implemented(self):
        """Test that MultiInputPolicy raises NotImplementedError."""
        env = gym.make("Pendulum-v1")
        
        from mlx_baselines3.td3.policies import MultiInputPolicy
        with pytest.raises(NotImplementedError):
            MultiInputPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda x: 1e-3
            )


@pytest.mark.slow
class TestTD3Integration:
    """Integration tests for TD3 that may take longer to run."""

    def test_td3_pendulum_learning(self):
        """Test TD3 learning on Pendulum environment (longer test)."""
        env = gym.make("Pendulum-v1")
        model = TD3(
            "MlpPolicy", 
            env, 
            learning_starts=100,
            batch_size=64,
            verbose=0
        )
        
        # Train for a short period
        model.learn(total_timesteps=1000)
        
        # Test that model can generate valid actions
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        assert env.action_space.contains(action)
        assert model.num_timesteps >= 1000
        assert model._n_updates > 0

    def test_td3_with_vec_env(self):
        """Test TD3 with vectorized environment."""
        def make_env():
            return gym.make("Pendulum-v1")
        
        env = DummyVecEnv([make_env for _ in range(2)])
        model = TD3("MlpPolicy", env, learning_starts=50, verbose=0)
        
        # Should work with vectorized env
        model.learn(total_timesteps=200)
        
        assert model.num_timesteps >= 200
        assert env.num_envs == 2


if __name__ == "__main__":
    pytest.main([__file__])
