"""
Tests for PPO algorithm implementation.
"""

import gymnasium as gym
import numpy as np
import pytest
import mlx.core as mx

from mlx_baselines3.ppo import PPO, PPOPolicy, MlpPolicy
from mlx_baselines3.common.vec_env import DummyVecEnv, make_vec_env


class TestPPOInitialization:
    """Test PPO initialization."""
    
    def test_ppo_initialization_with_string_policy(self):
        """Test PPO initialization with string policy."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        assert model.n_steps == 2048
        assert model.batch_size == 64
        assert model.n_epochs == 10
        assert model.gamma == 0.99
        assert model.gae_lambda == 0.95
        assert model.clip_range == 0.2
        assert model.ent_coef == 0.0
        assert model.vf_coef == 0.5
        assert model.max_grad_norm == 0.5
        assert model.target_kl is None
        
        env.close()
        
    def test_ppo_initialization_with_policy_class(self):
        """Test PPO initialization with policy class."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO(PPOPolicy, env, verbose=0)
        
        assert model.policy is not None
        assert isinstance(model.policy, PPOPolicy)
        
        env.close()
        
    def test_ppo_initialization_with_custom_hyperparameters(self):
        """Test PPO initialization with custom hyperparameters."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=1.0,
            target_kl=0.02,
            verbose=0,
        )
        
        assert model.n_steps == 128
        assert model.batch_size == 32
        assert model.n_epochs == 5
        assert model.gamma == 0.95
        assert model.gae_lambda == 0.9
        assert model.clip_range == 0.1
        assert model.ent_coef == 0.01
        assert model.vf_coef == 0.25
        assert model.max_grad_norm == 1.0
        assert model.target_kl == 0.02
        
        env.close()
        
    def test_ppo_initialization_continuous_action_space(self):
        """Test PPO with continuous action space."""
        env = make_vec_env("Pendulum-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        assert isinstance(model.action_space, gym.spaces.Box)
        assert model.policy is not None
        
        env.close()
        
    def test_ppo_rollout_buffer_initialization(self):
        """Test rollout buffer is properly initialized."""
        env = make_vec_env("CartPole-v1", n_envs=2)
        model = PPO("MlpPolicy", env, n_steps=64, verbose=0)
        
        assert model.rollout_buffer is not None
        assert model.rollout_buffer.buffer_size == 64
        assert model.rollout_buffer.n_envs == 2
        
        env.close()


class TestPPOPrediction:
    """Test PPO prediction functionality."""
    
    def test_predict_single_observation_discrete(self):
        """Test prediction with single observation for discrete action space."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        obs = env.observation_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        
        # For discrete actions, action should be a scalar (int)
        assert isinstance(action, (int, np.integer)) or (hasattr(action, 'shape') and action.shape == ())
        if hasattr(action, 'shape'):
            assert action.shape == env.action_space.shape or action.shape == ()
        
        # Test stochastic prediction
        action_stoch, _states = model.predict(obs, deterministic=False)
        assert isinstance(action_stoch, (int, np.integer)) or (hasattr(action_stoch, 'shape') and action_stoch.shape == ())
        
        env.close()
        
    def test_predict_batch_observations_discrete(self):
        """Test prediction with batch of observations for discrete action space."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        obs_batch = np.array([env.observation_space.sample() for _ in range(3)])
        actions, _states = model.predict(obs_batch, deterministic=True)
        
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (3,)
        
        env.close()
        
    def test_predict_continuous_action_space(self):
        """Test prediction with continuous action space."""
        env = make_vec_env("Pendulum-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        obs = env.observation_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        
        assert action.shape == env.action_space.shape
        assert isinstance(action, np.ndarray)
        
        # Check action is within bounds
        assert np.all(action >= env.action_space.low)
        assert np.all(action <= env.action_space.high)
        
        env.close()


class TestPPOTraining:
    """Test PPO training functionality."""
    
    def test_collect_rollouts(self):
        """Test rollout collection."""
        env = make_vec_env("CartPole-v1", n_envs=2)
        model = PPO("MlpPolicy", env, n_steps=16, verbose=0)
        
        # Initialize for rollout collection
        model._last_obs = env.reset()
        model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        # Test rollout collection
        success = model.collect_rollouts(
            env, None, model.rollout_buffer, n_rollout_steps=16
        )
        
        assert success is True
        assert model.rollout_buffer.full
        assert model.rollout_buffer.pos == 16  # Position after filling buffer
        
        env.close()
        
    def test_train_step(self):
        """Test single training step."""
        env = make_vec_env("CartPole-v1", n_envs=2)
        model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, verbose=0)
        
        # Fill buffer with dummy data
        model._last_obs = env.reset()
        model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        # Collect rollouts
        model.collect_rollouts(env, None, model.rollout_buffer, n_rollout_steps=16)
        
        # Train
        initial_n_updates = model._n_updates
        model.train()
        
        # Check that training occurred
        assert model._n_updates > initial_n_updates
        
        env.close()
        
    def test_learn_short_training(self):
        """Test short training run."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, verbose=0)
        
        # Train for a small number of timesteps
        initial_timesteps = model.num_timesteps
        model.learn(total_timesteps=32, log_interval=None)
        
        assert model.num_timesteps > initial_timesteps
        assert model.num_timesteps >= 32
        
        env.close()


class TestPPOSaveLoad:
    """Test PPO save and load functionality."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading PPO model."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Train briefly to have some state
        model.learn(total_timesteps=64, log_interval=None)
        
        # Get prediction before saving
        obs = env.observation_space.sample()
        action_before, _ = model.predict(obs, deterministic=True)
        
        # Save model
        save_path = tmp_path / "test_ppo_model.zip"
        model.save(str(save_path))
        
        # Load model
        loaded_model = PPO.load(str(save_path), env=env)
        
        # Check that prediction is the same
        action_after, _ = loaded_model.predict(obs, deterministic=True)
        
        assert np.array_equal(action_before, action_after)
        assert loaded_model.n_steps == model.n_steps
        assert loaded_model.batch_size == model.batch_size
        
        env.close()
        
    def test_get_and_set_parameters(self):
        """Test parameter getting and setting."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Get parameters
        params = model._get_parameters()
        assert isinstance(params, dict)
        assert "policy_parameters" in params
        
        # Create another model and set parameters
        env2 = make_vec_env("CartPole-v1", n_envs=1)
        model2 = PPO("MlpPolicy", env2, verbose=0)
        model2._set_parameters(params)
        
        # Check that predictions are similar (not exactly equal due to randomness)
        obs = env.observation_space.sample()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = model2.predict(obs, deterministic=True)
        
        # They should be the same with deterministic=True and same parameters
        assert np.array_equal(action1, action2)
        
        env.close()
        env2.close()


class TestPPOEdgeCases:
    """Test PPO edge cases and error handling."""
    
    def test_ppo_requires_vectorized_env(self):
        """Test that PPO requires vectorized environment."""
        env = gym.make("CartPole-v1")
        
        with pytest.raises(AssertionError, match="PPO requires a vectorized environment"):
            PPO("MlpPolicy", env, verbose=0)
            
        env.close()
        
    def test_invalid_policy_name(self):
        """Test error handling for invalid policy name."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        
        with pytest.raises(ValueError, match="Unknown policy"):
            PPO("InvalidPolicy", env, verbose=0)
            
        env.close()
        
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        # Create dummy gradients
        import mlx.core as mx
        
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, max_grad_norm=1.0, verbose=0)
        
        # Test gradient clipping method
        grads = {
            "param1": mx.array([2.0, 3.0]),
            "param2": mx.array([1.0, 4.0]),
        }
        
        # Test the new gradient clipping function
        from mlx_baselines3.common.optimizers import clip_grad_norm
        
        clipped_grads, original_norm = clip_grad_norm(grads, max_norm=1.0)
        
        # Calculate total norm of clipped gradients
        total_norm = 0.0
        for grad in clipped_grads.values():
            if grad is not None:
                total_norm += mx.sum(grad ** 2)
        total_norm = float(mx.sqrt(total_norm))
        
        # Should be close to 1.0 (or less)
        assert total_norm <= 1.1  # Small tolerance for numerical precision
        
        # Original norm should be larger than clipped norm
        assert original_norm > total_norm
        
        env.close()


class TestPPOCompatibility:
    """Test PPO compatibility with different environments and configurations."""
    
    def test_ppo_with_different_action_spaces(self):
        """Test PPO with different action space types."""
        # Discrete action space
        env_discrete = make_vec_env("CartPole-v1", n_envs=1)
        model_discrete = PPO("MlpPolicy", env_discrete, verbose=0)
        assert isinstance(model_discrete.action_space, gym.spaces.Discrete)
        env_discrete.close()
        
        # Box action space
        env_box = make_vec_env("Pendulum-v1", n_envs=1)
        model_box = PPO("MlpPolicy", env_box, verbose=0)
        assert isinstance(model_box.action_space, gym.spaces.Box)
        env_box.close()
        
    def test_ppo_with_multiple_environments(self):
        """Test PPO with multiple parallel environments."""
        env = make_vec_env("CartPole-v1", n_envs=4)
        model = PPO("MlpPolicy", env, n_steps=32, verbose=0)
        
        assert model.rollout_buffer.n_envs == 4
        
        # Test rollout collection with multiple envs
        model._last_obs = env.reset()
        model._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        
        success = model.collect_rollouts(
            env, None, model.rollout_buffer, n_rollout_steps=32
        )
        assert success is True
        
        env.close()
        
    def test_schedule_functions(self):
        """Test schedule function handling."""
        env = make_vec_env("CartPole-v1", n_envs=1)
        
        # Test with constant value
        model = PPO("MlpPolicy", env, clip_range=0.2, verbose=0)
        assert model._get_schedule_value(0.2) == 0.2
        
        # Test with callable schedule
        def schedule_fn(progress):
            return 0.2 * progress
            
        model.clip_range = schedule_fn
        model._current_progress_remaining = 0.5
        assert model._get_schedule_value(schedule_fn) == 0.1
        
        env.close()
