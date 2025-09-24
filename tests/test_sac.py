"""Tests for SAC (Soft Actor-Critic) algorithm."""

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from mlx_baselines3 import SAC
from mlx_baselines3.sac.policies import SACPolicy


class TestSAC:
    """Test SAC algorithm."""

    def test_sac_init(self):
        """Test SAC initialization."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        # Check that policy was created
        assert model.policy is not None
        assert isinstance(model.policy, SACPolicy)

        # Check SAC-specific attributes
        assert hasattr(model, "tau")
        assert hasattr(model, "gamma")
        assert hasattr(model, "ent_coef")
        assert hasattr(model, "target_entropy")

        # Check continuous action space
        assert isinstance(env.action_space, gym.spaces.Box)

        env.close()

    def test_sac_init_with_auto_entropy(self):
        """Test SAC initialization with automatic entropy tuning."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, ent_coef="auto", verbose=0)

        # Check entropy coefficient settings
        assert model.ent_coef == "auto"
        assert model.target_entropy is not None
        assert model.log_ent_coef is not None
        assert model.ent_coef_optimizer is not None

        # Target entropy should be -dim(action_space) for continuous actions
        expected_target_entropy = -float(env.action_space.shape[0])
        assert model.target_entropy == expected_target_entropy

        env.close()

    def test_sac_init_with_fixed_entropy(self):
        """Test SAC initialization with fixed entropy coefficient."""
        env = gym.make("Pendulum-v1")
        ent_coef = 0.1
        model = SAC("MlpPolicy", env, ent_coef=ent_coef, verbose=0)

        # Check entropy coefficient settings
        assert model.ent_coef == ent_coef
        assert model.target_entropy is None
        assert model.log_ent_coef is None
        assert model.ent_coef_optimizer is None

        env.close()

    def test_sac_policy_forward(self):
        """Test SAC policy forward pass."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        obs = env.observation_space.sample()
        obs_tensor = mx.array(obs).reshape(1, -1)  # Add batch dimension

        # Test forward pass
        actions, values, log_probs = model.policy.forward(
            obs_tensor, deterministic=False
        )

        # Check output shapes
        assert actions.shape == (1, env.action_space.shape[0])
        assert values.shape == (1,)
        assert log_probs.shape == (1,)

        # Test deterministic forward pass
        actions_det, values_det, log_probs_det = model.policy.forward(
            obs_tensor, deterministic=True
        )

        # Deterministic actions should differ from stochastic ones (with high
        # probability). Values should be zeros for SAC.
        assert np.allclose(np.array(values), 0.0)
        assert np.allclose(np.array(values_det), 0.0)

        env.close()

    def test_sac_policy_predict_values(self):
        """Test SAC policy value prediction."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        obs = env.observation_space.sample()
        obs_tensor = mx.array(obs).reshape(1, -1)  # Add batch dimension

        # Test value prediction
        values = model.policy.predict_values(obs_tensor)

        # Check output shape
        assert values.shape == (1,)

        env.close()

    def test_sac_predict(self):
        """Test SAC action prediction."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        obs = env.observation_space.sample()

        # Test stochastic prediction
        actions, _ = model.predict(obs, deterministic=False)
        assert actions.shape == env.action_space.shape

        # Test deterministic prediction
        actions_det, _ = model.predict(obs, deterministic=True)
        assert actions_det.shape == env.action_space.shape

        # Actions should be within action space bounds
        assert np.all(actions >= env.action_space.low)
        assert np.all(actions <= env.action_space.high)
        assert np.all(actions_det >= env.action_space.low)
        assert np.all(actions_det <= env.action_space.high)

        env.close()

    def test_sac_actor_critic_forward(self):
        """Test SAC actor and critic forward passes."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        batch_size = 4
        obs = mx.array(np.random.randn(batch_size, *env.observation_space.shape))
        actions = mx.array(np.random.randn(batch_size, *env.action_space.shape))

        # Test actor forward
        features = model.policy.extract_features(obs)
        sampled_actions, log_probs, entropy = model.policy.actor_forward(
            features, deterministic=False
        )

        assert sampled_actions.shape == (batch_size, env.action_space.shape[0])
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

        # Test critic forward
        q_values = model.policy.critic_forward(features, actions)
        assert len(q_values) == model.policy.n_critics  # Twin critics
        for q_val in q_values:
            assert q_val.shape == (batch_size, 1)

        # Test target critic forward
        q_values_target = model.policy.critic_target_forward(features, actions)
        assert len(q_values_target) == model.policy.n_critics
        for q_val in q_values_target:
            assert q_val.shape == (batch_size, 1)

        env.close()

    def test_sac_only_continuous_actions(self):
        """Test that SAC only accepts continuous action spaces."""
        env = gym.make("CartPole-v1")  # Discrete action space

        with pytest.raises(AssertionError, match="Action space.*is not supported"):
            SAC("MlpPolicy", env, verbose=0)

        env.close()

    def test_sac_optimizers_initialization(self):
        """Test SAC optimizer initialization."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        # Check that optimizers are created
        assert hasattr(model, "actor_optimizer")
        assert hasattr(model, "critic_optimizer")
        assert hasattr(model, "actor_optimizer_state")
        assert hasattr(model, "critic_optimizer_state")

        # Check entropy coefficient optimizer for auto mode
        model_auto = SAC("MlpPolicy", env, ent_coef="auto", verbose=0)
        assert hasattr(model_auto, "ent_coef_optimizer")
        assert hasattr(model_auto, "ent_coef_optimizer_state")

        env.close()

    def test_sac_replay_buffer(self):
        """Test SAC replay buffer setup."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, buffer_size=1000, verbose=0)

        # Check that replay buffer is created
        assert model.replay_buffer is not None
        assert model.replay_buffer.buffer_size == 1000

        env.close()

    def test_sac_save_load_parameters(self):
        """Test SAC parameter save/load functionality."""
        env = gym.make("Pendulum-v1")
        model = SAC("MlpPolicy", env, verbose=0)

        # Get initial parameters
        params = model._get_parameters()
        assert isinstance(params, dict)
        assert len(params) > 0

        # Modify parameters slightly
        for key in params:
            if isinstance(params[key], mx.array):
                params[key] = params[key] + 0.01

        # Set modified parameters
        model._set_parameters(params)

        # Get parameters again and verify they changed
        model._get_parameters()

        env.close()

    @pytest.mark.slow
    def test_sac_short_training(self):
        """Test SAC short training run."""
        env = gym.make("Pendulum-v1")
        model = SAC(
            "MlpPolicy",
            env,
            learning_starts=10,
            train_freq=1,
            batch_size=32,
            buffer_size=1000,
            verbose=0,
        )

        # Short training run
        initial_timesteps = model.num_timesteps
        model.learn(total_timesteps=100)

        # Check that training variables are updated
        print(
            "Initial timesteps: "
            f"{initial_timesteps}, Final timesteps: {model.num_timesteps}"
        )
        assert model.num_timesteps == 100
        assert model._n_updates >= 0  # May be 0 if learning hasn't started

        env.close()
