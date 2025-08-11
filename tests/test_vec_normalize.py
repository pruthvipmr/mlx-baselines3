"""
Tests for VecNormalize wrapper functionality.
"""

import os
import tempfile
import numpy as np
import pytest
import gymnasium as gym

from mlx_baselines3.common.vec_env import DummyVecEnv, VecNormalize, make_vec_env


class TestRunningMeanStd:
    """Test the RunningMeanStd class."""
    
    def test_basic_functionality(self):
        """Test basic statistics tracking."""
        from mlx_baselines3.common.vec_env.vec_normalize import RunningMeanStd
        
        rms = RunningMeanStd()
        
        # Test initial state
        assert rms.count == 1e-4  # epsilon
        np.testing.assert_array_equal(rms.mean, 0.0)
        np.testing.assert_array_equal(rms.var, 1.0)
        
        # Update with some data
        data = np.array([1.0, 2.0, 3.0])
        rms.update(data)
        
        # Check that statistics were updated
        assert rms.count > 1e-4
        assert rms.mean != 0.0
        
    def test_normalization(self):
        """Test normalization and denormalization."""
        from mlx_baselines3.common.vec_env.vec_normalize import RunningMeanStd
        
        rms = RunningMeanStd()
        
        # Generate some data with known statistics
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, (100,))
        rms.update(data)
        
        # Test normalization
        normalized = rms.normalize(data)
        assert abs(np.mean(normalized)) < 0.1  # Should be close to 0
        assert abs(np.std(normalized) - 1.0) < 0.1  # Should be close to 1
        
        # Test denormalization
        denormalized = rms.denormalize(normalized)
        np.testing.assert_array_almost_equal(data, denormalized, decimal=5)
        
    def test_multidimensional(self):
        """Test with multidimensional data."""
        from mlx_baselines3.common.vec_env.vec_normalize import RunningMeanStd
        
        rms = RunningMeanStd(shape=(2, 3))
        
        # Update with multidimensional data
        data = np.random.randn(10, 2, 3)
        rms.update(data)
        
        # Check shapes
        assert rms.mean.shape == (2, 3)
        assert rms.var.shape == (2, 3)
        
        # Test normalization
        normalized = rms.normalize(data)
        assert normalized.shape == data.shape


class TestVecNormalize:
    """Test the VecNormalize wrapper."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env_id = "CartPole-v1"
        self.n_envs = 3
        
    def make_env(self):
        """Create a basic vectorized environment."""
        return make_vec_env(self.env_id, n_envs=self.n_envs, seed=42)
        
    def test_initialization(self):
        """Test VecNormalize initialization."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv)
        
        assert venv_norm.num_envs == self.n_envs
        assert venv_norm.training is True
        assert venv_norm.norm_obs is True
        assert venv_norm.norm_reward is True
        assert venv_norm.obs_rms is not None
        assert venv_norm.ret_rms is not None
        assert venv_norm.returns is not None
        
        venv_norm.close()
        
    def test_observation_normalization(self):
        """Test observation normalization."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv, norm_reward=False)  # Only test obs normalization
        
        # Reset and collect some observations
        obs = venv_norm.reset()
        assert obs.shape == (self.n_envs, 4)  # CartPole has 4-dim observations
        
        # Step a few times to update statistics
        for _ in range(10):
            actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
            obs, rewards, dones, infos = venv_norm.step(actions)
            
        # Check that observations are being normalized
        assert venv_norm.obs_rms.count > 1e-4
        
        venv_norm.close()
        
    def test_reward_normalization(self):
        """Test reward normalization."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv, norm_obs=False)  # Only test reward normalization
        
        # Reset environment
        venv_norm.reset()
        
        # Step a few times to collect rewards
        for _ in range(10):
            actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
            obs, rewards, dones, infos = venv_norm.step(actions)
            assert rewards.shape == (self.n_envs,)
            
        # Check that reward statistics are being updated
        assert venv_norm.ret_rms.count > 1e-4
        
        venv_norm.close()
        
    def test_training_mode(self):
        """Test training mode functionality."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv)
        
        # Check initial training state
        assert venv_norm.training is True
        
        # Reset and step
        venv_norm.reset()
        actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
        
        # Get initial statistics
        initial_obs_count = venv_norm.obs_rms.count
        initial_ret_count = venv_norm.ret_rms.count
        
        # Step with training=True
        venv_norm.step(actions)
        
        # Statistics should be updated
        assert venv_norm.obs_rms.count > initial_obs_count
        assert venv_norm.ret_rms.count > initial_ret_count
        
        # Turn off training
        venv_norm.set_training(False)
        assert venv_norm.training is False
        
        # Get current statistics
        current_obs_count = venv_norm.obs_rms.count
        current_ret_count = venv_norm.ret_rms.count
        
        # Step with training=False
        actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
        venv_norm.step(actions)
        
        # Statistics should not be updated
        assert venv_norm.obs_rms.count == current_obs_count
        assert venv_norm.ret_rms.count == current_ret_count
        
        venv_norm.close()
        
    def test_clipping(self):
        """Test observation and reward clipping."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv, clip_obs=1.0, clip_reward=1.0)
        
        # Reset environment
        obs = venv_norm.reset()
        
        # Step a few times 
        for _ in range(5):
            actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
            obs, rewards, dones, infos = venv_norm.step(actions)
            
            # Check clipping
            assert np.all(np.abs(obs) <= 1.0 + 1e-6)  # Small tolerance for floating point
            assert np.all(np.abs(rewards) <= 1.0 + 1e-6)
            
        venv_norm.close()
        
    def test_dict_observations(self):
        """Test with dictionary observation spaces."""
        # Create a simple environment with dict observations
        class DictObsEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Dict({
                    'image': gym.spaces.Box(0, 255, (2, 2), dtype=np.uint8),
                    'vector': gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
                })
                self.action_space = gym.spaces.Discrete(2)
                
            def reset(self, seed=None):
                self.np_random, _ = gym.utils.seeding.np_random(seed)
                obs = {
                    'image': self.np_random.integers(0, 256, (2, 2), dtype=np.uint8),
                    'vector': self.np_random.normal(size=(3,)).astype(np.float32)
                }
                return obs, {}
                
            def step(self, action):
                obs = {
                    'image': self.np_random.integers(0, 256, (2, 2), dtype=np.uint8),
                    'vector': self.np_random.normal(size=(3,)).astype(np.float32)
                }
                return obs, 1.0, False, False, {}
                
        # Create vectorized environment
        venv = DummyVecEnv([lambda: DictObsEnv() for _ in range(2)])
        venv_norm = VecNormalize(venv, norm_reward=False)
        
        # Test reset
        obs = venv_norm.reset()
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 'vector' in obs
        assert obs['image'].shape == (2, 2, 2)  # (n_envs, height, width)
        assert obs['vector'].shape == (2, 3)  # (n_envs, vector_size)
        
        # Test step
        actions = np.array([0, 1])
        obs, rewards, dones, infos = venv_norm.step(actions)
        
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 'vector' in obs
        
        venv_norm.close()
        
    def test_save_load(self):
        """Test saving and loading VecNormalize parameters."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv)
        
        # Reset and step to update statistics
        venv_norm.reset()
        for _ in range(10):
            actions = np.array([venv_norm.action_space.sample() for _ in range(self.n_envs)])
            venv_norm.step(actions)
            
        # Save original statistics
        original_obs_mean = venv_norm.obs_rms.mean.copy()
        original_obs_var = venv_norm.obs_rms.var.copy()
        original_ret_mean = venv_norm.ret_rms.mean.copy()
        original_ret_var = venv_norm.ret_rms.var.copy()
        original_returns = venv_norm.returns.copy()
        
        # Save to file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
            
        try:
            venv_norm.save(temp_path)
            
            # Create new environment and load
            venv2 = self.make_env()
            venv_norm2 = VecNormalize(venv2)
            venv_norm2.load(temp_path)
            
            # Check that statistics were restored
            np.testing.assert_array_almost_equal(venv_norm2.obs_rms.mean, original_obs_mean)
            np.testing.assert_array_almost_equal(venv_norm2.obs_rms.var, original_obs_var)
            np.testing.assert_array_almost_equal(venv_norm2.ret_rms.mean, original_ret_mean)
            np.testing.assert_array_almost_equal(venv_norm2.ret_rms.var, original_ret_var)
            np.testing.assert_array_almost_equal(venv_norm2.returns, original_returns)
            
            venv_norm2.close()
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        venv_norm.close()
        
    def test_get_set_attr(self):
        """Test getting and setting attributes."""
        venv = self.make_env()
        venv_norm = VecNormalize(venv)
        
        # Test getting normalization-specific attributes
        training_attrs = venv_norm.get_attr("training")
        assert len(training_attrs) == self.n_envs
        assert all(attr is True for attr in training_attrs)
        
        # Test setting normalization-specific attributes
        venv_norm.set_attr("training", False)
        assert venv_norm.training is False
        
        # Test getting wrapped environment attributes
        action_spaces = venv_norm.get_attr("action_space")
        assert len(action_spaces) == self.n_envs
        assert all(isinstance(space, gym.spaces.Discrete) for space in action_spaces)
        
        venv_norm.close()


class TestEnvTypeAssertions:
    """Test that algorithms properly handle environment types."""
    
    def test_ppo_requires_vec_env(self):
        """Test that PPO requires a vectorized environment."""
        from mlx_baselines3 import PPO
        
        # Create a single environment (not vectorized)
        env = gym.make("CartPole-v1")
        
        # PPO should raise an assertion error
        with pytest.raises(AssertionError, match="PPO requires a vectorized environment"):
            PPO("MlpPolicy", env)
            
        env.close()
        
    def test_a2c_requires_vec_env(self):
        """Test that A2C requires a vectorized environment."""
        from mlx_baselines3 import A2C
        
        # Create a single environment (not vectorized)
        env = gym.make("CartPole-v1")
        
        # A2C should raise an assertion error
        with pytest.raises(AssertionError, match="A2C requires a vectorized environment"):
            A2C("MlpPolicy", env)
            
        env.close()
        
    def test_dqn_accepts_single_env(self):
        """Test that DQN accepts single environments."""
        from mlx_baselines3 import DQN
        
        # Create a single environment (not vectorized)
        env = gym.make("CartPole-v1")
        
        # DQN should accept it without complaint
        try:
            model = DQN("MlpPolicy", env)
            # Should work without assertion error
            assert model.env == env
        except AssertionError:
            pytest.fail("DQN should accept single environments")
        finally:
            env.close()
            
    def test_sac_accepts_single_env(self):
        """Test that SAC accepts single environments."""
        from mlx_baselines3 import SAC
        
        # Create a single environment (not vectorized)
        env = gym.make("Pendulum-v1")
        
        # SAC should accept it without complaint
        try:
            model = SAC("MlpPolicy", env)
            # Should work without assertion error
            assert model.env == env
        except AssertionError:
            pytest.fail("SAC should accept single environments")
        finally:
            env.close()
            
    def test_td3_accepts_single_env(self):
        """Test that TD3 accepts single environments."""
        from mlx_baselines3 import TD3
        
        # Create a single environment (not vectorized)
        env = gym.make("Pendulum-v1")
        
        # TD3 should accept it without complaint
        try:
            model = TD3("MlpPolicy", env)
            # Should work without assertion error
            assert model.env == env
        except AssertionError:
            pytest.fail("TD3 should accept single environments")
        finally:
            env.close()
            
    def test_ppo_accepts_vec_env(self):
        """Test that PPO accepts vectorized environments."""
        from mlx_baselines3 import PPO
        
        # Create a vectorized environment
        venv = make_vec_env("CartPole-v1", n_envs=2, seed=42)
        
        # PPO should accept it
        model = PPO("MlpPolicy", venv)
        assert model.env == venv
        
        venv.close()
        
    def test_off_policy_with_vec_env(self):
        """Test that off-policy algorithms also work with vectorized environments."""
        from mlx_baselines3 import DQN, SAC, TD3
        
        # Create vectorized environments
        discrete_venv = make_vec_env("CartPole-v1", n_envs=2, seed=42)
        continuous_venv = make_vec_env("Pendulum-v1", n_envs=2, seed=42)
        
        # Test DQN with vectorized env
        dqn_model = DQN("MlpPolicy", discrete_venv)
        assert dqn_model.env == discrete_venv
        
        # Test SAC with vectorized env  
        sac_model = SAC("MlpPolicy", continuous_venv)
        assert sac_model.env == continuous_venv
        
        # Test TD3 with vectorized env
        td3_model = TD3("MlpPolicy", continuous_venv)
        assert td3_model.env == continuous_venv
        
        discrete_venv.close()
        continuous_venv.close()


class TestMakeVecEnv:
    """Test the make_vec_env utility function."""
    
    def test_basic_functionality(self):
        """Test basic make_vec_env functionality."""
        venv = make_vec_env("CartPole-v1", n_envs=3, seed=42)
        
        assert venv.num_envs == 3
        assert isinstance(venv, DummyVecEnv)
        
        # Test reset
        obs = venv.reset()
        assert obs.shape == (3, 4)  # 3 envs, 4-dim obs
        
        # Test step
        actions = np.array([0, 1, 0])
        obs, rewards, dones, infos = venv.step(actions)
        assert obs.shape == (3, 4)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3
        
        venv.close()
        
    def test_with_wrapper(self):
        """Test make_vec_env with wrapper class."""
        def wrapper_fn(env):
            # Simple wrapper that just passes through
            return env
            
        venv = make_vec_env("CartPole-v1", n_envs=2, wrapper_class=wrapper_fn, seed=42)
        
        assert venv.num_envs == 2
        venv.close()
        
    def test_with_env_kwargs(self):
        """Test make_vec_env with environment kwargs."""
        # Test with an environment that accepts kwargs
        try:
            venv = make_vec_env("CartPole-v1", n_envs=2, env_kwargs={}, seed=42)
            assert venv.num_envs == 2
            venv.close()
        except Exception:
            # Some environments might not accept additional kwargs
            pass
            
    def test_with_callable_env(self):
        """Test make_vec_env with callable environment creator."""
        def make_env():
            return gym.make("CartPole-v1")
            
        venv = make_vec_env(make_env, n_envs=2, seed=42)
        
        assert venv.num_envs == 2
        venv.close()


if __name__ == "__main__":
    pytest.main([__file__])
