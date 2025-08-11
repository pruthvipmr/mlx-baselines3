"""
Tests for reproducibility and seeding functionality.
"""
import numpy as np
import pytest
import mlx.core as mx
import gymnasium as gym

from mlx_baselines3 import PPO, A2C, DQN, SAC, TD3
from mlx_baselines3.common.vec_env import make_vec_env


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_ppo_reproducibility(self):
        """Test PPO reproducibility with same seed."""
        seed = 42
        
        # Create first model
        env1 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model1 = PPO("MlpPolicy", env1, seed=seed, verbose=0)
        
        # Create second model with same seed
        env2 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model2 = PPO("MlpPolicy", env2, seed=seed, verbose=0)
        
        # Get initial predictions
        obs1 = env1.reset()[0]
        obs2 = env2.reset()[0]
        
        # Both environments should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Predictions should be identical
        action1, _ = model1.predict(obs1, deterministic=True)
        action2, _ = model2.predict(obs2, deterministic=True)
        
        np.testing.assert_array_equal(action1, action2)
        
        env1.close()
        env2.close()

    def test_a2c_reproducibility(self):
        """Test A2C reproducibility with same seed."""
        seed = 123
        
        # Create first model
        env1 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model1 = A2C("MlpPolicy", env1, seed=seed, verbose=0)
        
        # Create second model with same seed
        env2 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model2 = A2C("MlpPolicy", env2, seed=seed, verbose=0)
        
        # Get initial predictions
        obs1 = env1.reset()[0]
        obs2 = env2.reset()[0]
        
        # Both environments should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Predictions should be identical
        action1, _ = model1.predict(obs1, deterministic=True)
        action2, _ = model2.predict(obs2, deterministic=True)
        
        np.testing.assert_array_equal(action1, action2)
        
        env1.close()
        env2.close()

    def test_dqn_reproducibility(self):
        """Test DQN reproducibility with same seed."""
        seed = 456
        
        # Create first model
        env1 = gym.make("CartPole-v1")
        env1.action_space.seed(seed)
        env1.observation_space.seed(seed)
        model1 = DQN("MlpPolicy", env1, seed=seed, verbose=0)
        
        # Create second model with same seed
        env2 = gym.make("CartPole-v1")
        env2.action_space.seed(seed)
        env2.observation_space.seed(seed)
        model2 = DQN("MlpPolicy", env2, seed=seed, verbose=0)
        
        # Get initial observations
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Both environments should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Predictions should be identical
        action1, _ = model1.predict(obs1, deterministic=True)
        action2, _ = model2.predict(obs2, deterministic=True)
        
        np.testing.assert_array_equal(action1, action2)
        
        env1.close()
        env2.close()

    def test_sac_reproducibility(self):
        """Test SAC reproducibility with same seed."""
        seed = 789
        
        # Create first model
        env1 = gym.make("Pendulum-v1")
        env1.action_space.seed(seed)
        env1.observation_space.seed(seed)
        model1 = SAC("MlpPolicy", env1, seed=seed, verbose=0)
        
        # Create second model with same seed
        env2 = gym.make("Pendulum-v1")
        env2.action_space.seed(seed)
        env2.observation_space.seed(seed)
        model2 = SAC("MlpPolicy", env2, seed=seed, verbose=0)
        
        # Get initial observations
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Both environments should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Predictions should be identical (deterministic mode)
        action1, _ = model1.predict(obs1, deterministic=True)
        action2, _ = model2.predict(obs2, deterministic=True)
        
        np.testing.assert_allclose(action1, action2, rtol=1e-6)
        
        env1.close()
        env2.close()

    def test_td3_reproducibility(self):
        """Test TD3 reproducibility with same seed."""
        seed = 321
        
        # Create first model
        env1 = gym.make("Pendulum-v1")
        env1.action_space.seed(seed)
        env1.observation_space.seed(seed)
        model1 = TD3("MlpPolicy", env1, seed=seed, verbose=0)
        
        # Create second model with same seed
        env2 = gym.make("Pendulum-v1")
        env2.action_space.seed(seed)
        env2.observation_space.seed(seed)
        model2 = TD3("MlpPolicy", env2, seed=seed, verbose=0)
        
        # Get initial observations
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Both environments should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Predictions should be identical (TD3 is always deterministic)
        action1, _ = model1.predict(obs1, deterministic=True)
        action2, _ = model2.predict(obs2, deterministic=True)
        
        np.testing.assert_allclose(action1, action2, rtol=1e-6)
        
        env1.close()
        env2.close()

    def test_mlx_random_seeding(self):
        """Test MLX random number generation seeding."""
        seed = 42
        
        # Set seed and generate random numbers
        mx.random.seed(seed)
        random1 = mx.random.normal((10,))
        
        # Reset seed and generate again
        mx.random.seed(seed)
        random2 = mx.random.normal((10,))
        
        # Should be identical
        np.testing.assert_array_equal(np.array(random1), np.array(random2))

    def test_numpy_random_seeding(self):
        """Test NumPy random number generation seeding."""
        seed = 42
        
        # Set seed and generate random numbers
        np.random.seed(seed)
        random1 = np.random.normal(size=(10,))
        
        # Reset seed and generate again
        np.random.seed(seed)
        random2 = np.random.normal(size=(10,))
        
        # Should be identical
        np.testing.assert_array_equal(random1, random2)

    def test_environment_seeding(self):
        """Test environment seeding consistency."""
        seed = 42
        
        # Create and seed environment
        env1 = gym.make("CartPole-v1")
        obs1, _ = env1.reset(seed=seed)
        
        # Create and seed another environment
        env2 = gym.make("CartPole-v1")
        obs2, _ = env2.reset(seed=seed)
        
        # Observations should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Take same actions
        action = env1.action_space.sample()
        
        next_obs1, reward1, done1, truncated1, _ = env1.step(action)
        next_obs2, reward2, done2, truncated2, _ = env2.step(action)
        
        # Results should be identical
        np.testing.assert_array_equal(next_obs1, next_obs2)
        assert reward1 == reward2
        assert done1 == done2
        assert truncated1 == truncated2
        
        env1.close()
        env2.close()


class TestTrainingReproducibility:
    """Test training reproducibility."""

    @pytest.mark.slow
    def test_ppo_training_reproducibility(self):
        """Test that PPO training is reproducible with same seed."""
        seed = 42
        total_timesteps = 100
        
        # First training run
        env1 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model1 = PPO("MlpPolicy", env1, seed=seed, verbose=0)
        model1.learn(total_timesteps=total_timesteps)
        
        # Get final parameters
        params1 = model1.policy.state_dict()
        
        # Second training run with same seed
        env2 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model2 = PPO("MlpPolicy", env2, seed=seed, verbose=0)
        model2.learn(total_timesteps=total_timesteps)
        
        # Get final parameters
        params2 = model2.policy.state_dict()
        
        # Parameters should be identical (or very close due to floating point)
        for key in params1.keys():
            np.testing.assert_allclose(
                np.array(params1[key]), 
                np.array(params2[key]), 
                rtol=1e-6, 
                err_msg=f"Parameter {key} differs between runs"
            )
        
        env1.close()
        env2.close()

    @pytest.mark.slow
    def test_a2c_training_reproducibility(self):
        """Test that A2C training is reproducible with same seed."""
        seed = 123
        total_timesteps = 100
        
        # First training run
        env1 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model1 = A2C("MlpPolicy", env1, seed=seed, verbose=0)
        model1.learn(total_timesteps=total_timesteps)
        
        # Get final parameters
        params1 = model1.policy.state_dict()
        
        # Second training run with same seed
        env2 = make_vec_env("CartPole-v1", n_envs=1, seed=seed)
        model2 = A2C("MlpPolicy", env2, seed=seed, verbose=0)
        model2.learn(total_timesteps=total_timesteps)
        
        # Get final parameters
        params2 = model2.policy.state_dict()
        
        # Parameters should be identical (or very close due to floating point)
        for key in params1.keys():
            np.testing.assert_allclose(
                np.array(params1[key]), 
                np.array(params2[key]), 
                rtol=1e-6, 
                err_msg=f"Parameter {key} differs between runs"
            )
        
        env1.close()
        env2.close()
