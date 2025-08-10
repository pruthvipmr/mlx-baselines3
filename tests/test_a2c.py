"""Tests for A2C algorithm implementation."""

import pytest
import gymnasium as gym
import numpy as np
import tempfile
import os

from mlx_baselines3 import A2C
from mlx_baselines3.a2c.policies import A2CPolicy, MlpPolicy, CnnPolicy, MultiInputPolicy
from mlx_baselines3.common.vec_env import DummyVecEnv


class TestA2CBasic:
    """Test basic A2C functionality."""
    
    def test_import(self):
        """Test that A2C can be imported."""
        from mlx_baselines3 import A2C
        assert A2C is not None
    
    def test_policy_import(self):
        """Test that A2C policies can be imported."""
        from mlx_baselines3.a2c import MlpPolicy, CnnPolicy, MultiInputPolicy
        assert MlpPolicy is not None
        assert CnnPolicy is not None
        assert MultiInputPolicy is not None
    
    def test_string_policy_instantiation(self):
        """Test A2C instantiation with string policy names."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        # Test MlpPolicy
        model = A2C("MlpPolicy", env, n_steps=10)
        assert isinstance(model.policy, A2CPolicy)
        
        # Test different policy names
        for policy_name in ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]:
            model = A2C(policy_name, env, n_steps=10)
            assert model is not None
    
    def test_class_policy_instantiation(self):
        """Test A2C instantiation with policy classes."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        model = A2C(MlpPolicy, env, n_steps=10)
        assert isinstance(model.policy, A2CPolicy)
    
    def test_hyperparameters(self):
        """Test A2C hyperparameter setting."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            n_steps=32,
            gamma=0.98,
            gae_lambda=0.9,
            ent_coef=0.02,
            vf_coef=0.25,
            max_grad_norm=1.0,
            normalize_advantage=True,
            use_rms_prop=False,  # Use Adam instead
            rms_prop_eps=1e-6,
            verbose=0
        )
        
        assert model.learning_rate == 1e-3
        assert model.n_steps == 32
        assert model.gamma == 0.98
        assert model.gae_lambda == 0.9
        assert model.ent_coef == 0.02
        assert model.vf_coef == 0.25
        assert model.max_grad_norm == 1.0
        assert model.normalize_advantage == True
        assert model.use_rms_prop == False
        assert model.rms_prop_eps == 1e-6


class TestA2CTraining:
    """Test A2C training functionality."""
    
    def test_short_training(self):
        """Test A2C can run a short training session."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        # Should not raise an exception
        model.learn(total_timesteps=50)
        
        # Check that model has been updated
        assert model._n_updates > 0
        assert model.num_timesteps > 0
    
    def test_prediction(self):
        """Test A2C prediction functionality."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Test prediction
        obs = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        
        assert action is not None
        assert action.shape == (1,)  # Single environment
        assert isinstance(action[0], (int, np.integer))  # Discrete action
    
    def test_rmsprop_optimizer(self):
        """Test A2C with RMSProp optimizer (default)."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, use_rms_prop=True, verbose=0)
        
        # Should not raise an exception
        model.learn(total_timesteps=50)
        assert model._n_updates > 0
    
    def test_adam_optimizer(self):
        """Test A2C with Adam optimizer."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, use_rms_prop=False, verbose=0)
        
        # Should not raise an exception
        model.learn(total_timesteps=50)
        assert model._n_updates > 0
    
    def test_advantage_normalization(self):
        """Test A2C with advantage normalization."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, normalize_advantage=True, verbose=0)
        
        # Should not raise an exception
        model.learn(total_timesteps=50)
        assert model._n_updates > 0


class TestA2CEnvironments:
    """Test A2C with different environment types."""
    
    def test_discrete_action_space(self):
        """Test A2C with discrete action space."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        model.learn(total_timesteps=50)
        
        obs = env.reset()
        action, _ = model.predict(obs)
        assert isinstance(action[0], (int, np.integer))
    
    def test_continuous_action_space(self):
        """Test A2C with continuous action space."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        model.learn(total_timesteps=50)
        
        obs = env.reset()
        action, _ = model.predict(obs)
        # Action might be MLX array, check if it's numeric and has the right shape
        assert hasattr(action[0], '__iter__') or isinstance(action[0], (float, np.floating, np.ndarray))
        assert len(action[0]) == 1  # Pendulum has 1D action space
    
    @pytest.mark.skipif(not hasattr(gym.spaces, 'MultiDiscrete'), 
                       reason="MultiDiscrete not available")
    def test_multi_discrete_action_space(self):
        """Test A2C with MultiDiscrete action space (currently not supported)."""
        # Create a simple environment with MultiDiscrete action space
        class MultiDiscreteEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
                self.action_space = gym.spaces.MultiDiscrete([2, 3])
                
            def reset(self, **kwargs):
                return np.random.uniform(-1, 1, 4), {}
                
            def step(self, action):
                obs = np.random.uniform(-1, 1, 4)
                reward = np.random.random()
                terminated = False
                truncated = False
                return obs, reward, terminated, truncated, {}
        
        env = DummyVecEnv([lambda: MultiDiscreteEnv()])
        
        # MultiDiscrete is not yet supported, should raise AssertionError
        with pytest.raises(AssertionError, match="Action space.*is not supported"):
            model = A2C("MlpPolicy", env, n_steps=10, verbose=0)


class TestA2CSaveLoad:
    """Test A2C save/load functionality."""
    
    def test_save_load(self):
        """Test A2C save and load functionality."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Get prediction before saving
        obs = env.reset()
        action_before, _ = model.predict(obs, deterministic=True)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "a2c_test.zip")
            model.save(save_path)
            
            # Load model
            loaded_model = A2C.load(save_path, env=env)
            
            # Get prediction after loading
            action_after, _ = loaded_model.predict(obs, deterministic=True)
            
            # Predictions should be identical
            np.testing.assert_array_equal(action_before, action_after)
    
    def test_save_load_without_env(self):
        """Test A2C save and load without providing environment."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=10, verbose=0)
        
        # Train briefly
        model.learn(total_timesteps=50)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "a2c_test.zip")
            model.save(save_path)
            
            # Load model without providing env (should create from saved env_id)
            loaded_model = A2C.load(save_path)
            
            assert loaded_model is not None
            assert loaded_model.env is not None


class TestA2CPerformance:
    """Test A2C performance on simple environments."""
    
    @pytest.mark.slow
    def test_cartpole_learning(self):
        """Test that A2C can learn CartPole-v1 to reasonable performance."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = A2C("MlpPolicy", env, n_steps=32, verbose=0)
        
        # Train for longer to see learning
        model.learn(total_timesteps=10000)
        
        # Test performance
        total_reward = 0
        n_episodes = 10
        
        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
                if done[0]:
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        
        # A2C should achieve at least some reasonable performance
        # (CartPole-v1 random baseline is around 20-30)
        assert avg_reward > 50, f"Average reward {avg_reward} is too low for A2C on CartPole"


if __name__ == "__main__":
    pytest.main([__file__])
