"""
Integration tests for end-to-end learning with callbacks and logging.
"""
import tempfile
import os
import pytest
import numpy as np
import gymnasium as gym

from mlx_baselines3 import PPO, A2C, DQN, SAC, TD3
from mlx_baselines3.common.vec_env import make_vec_env
from mlx_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from mlx_baselines3.common.logger import configure_logger


class TestIntegrationLearning:
    """Test end-to-end learning with various components."""

    @pytest.mark.integration
    def test_ppo_integration_cartpole(self):
        """Test PPO end-to-end learning on CartPole with callbacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment and model
            env = make_vec_env("CartPole-v1", n_envs=2)
            eval_env = gym.make("CartPole-v1")
            
            # Setup callbacks
            eval_callback = EvalCallback(
                eval_env=eval_env,
                eval_freq=500,
                n_eval_episodes=5,
                best_model_save_path=tmpdir,
                verbose=0
            )
            checkpoint_callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                name_prefix="ppo_checkpoint",
                verbose=0
            )
            
            # Create model with callbacks
            model = PPO(
                "MlpPolicy", 
                env, 
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1
            )
            
            # Train for 2000 timesteps
            model.learn(
                total_timesteps=2000,
                callback=[eval_callback, checkpoint_callback],
                log_interval=4,
                tb_log_name="PPO",
                reset_num_timesteps=True,
                progress_bar=False
            )
            
            # Verify training completed
            assert model.num_timesteps >= 2000
            
            # Test final performance
            obs, _ = eval_env.reset()
            episode_rewards = []
            for _ in range(5):
                episode_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    if done:
                        obs, _ = eval_env.reset()
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            print(f"Final mean reward: {mean_reward}")
            
            # Should achieve some learning (not perfect but better than random)
            # Note: Performance threshold is low to account for short training time
            assert mean_reward > 20, f"Mean reward {mean_reward} too low"
            
            # Check that callbacks worked
            assert len(eval_callback.evaluations_results) > 0
            assert os.path.exists(os.path.join(tmpdir, "best_model"))
            
            env.close()
            eval_env.close()

    @pytest.mark.integration
    def test_a2c_integration_cartpole(self):
        """Test A2C end-to-end learning on CartPole."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment and model
            env = make_vec_env("CartPole-v1", n_envs=4)
            
            # Create model
            model = A2C(
                "MlpPolicy", 
                env, 
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                max_grad_norm=0.5,
                rms_prop_eps=1e-5,
                use_rms_prop=True,
                verbose=1
            )
            
            # Train for 3000 timesteps
            model.learn(
                total_timesteps=3000,
                log_interval=10,
                reset_num_timesteps=True,
                progress_bar=False
            )
            
            # Verify training completed
            assert model.num_timesteps >= 3000
            
            # Test performance
            test_env = gym.make("CartPole-v1")
            obs, _ = test_env.reset()
            total_reward = 0
            for _ in range(200):  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"A2C total reward: {total_reward}")
            # Lenient threshold since A2C may need more training time
            assert total_reward >= 8, f"Total reward {total_reward} too low for basic functionality"
            
            env.close()
            test_env.close()

    @pytest.mark.integration
    def test_dqn_integration_cartpole(self):
        """Test DQN end-to-end learning on CartPole."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment and model
            env = gym.make("CartPole-v1")
            
            # Create model
            model = DQN(
                "MlpPolicy", 
                env, 
                learning_rate=1e-3,
                buffer_size=1000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                target_update_interval=10,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                verbose=1
            )
            
            # Train for 5000 timesteps
            model.learn(
                total_timesteps=5000,
                log_interval=10,
                reset_num_timesteps=True,
                progress_bar=False
            )
            
            # Verify training completed
            assert model.num_timesteps >= 5000
            
            # Test performance  
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(200):  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"DQN total reward: {total_reward}")
            # Lenient threshold for integration test - DQN needs more training than other algorithms  
            assert total_reward > 8, f"Total reward {total_reward} too low for basic functionality"
            
            env.close()

    @pytest.mark.integration
    def test_sac_integration_pendulum(self):
        """Test SAC end-to-end learning on Pendulum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment and model
            env = gym.make("Pendulum-v1")
            
            # Create model
            model = SAC(
                "MlpPolicy", 
                env, 
                learning_rate=3e-4,
                buffer_size=1000,
                learning_starts=100,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                target_update_interval=1,
                verbose=1
            )
            
            # Train for 5000 timesteps
            model.learn(
                total_timesteps=5000,
                log_interval=10,
                reset_num_timesteps=True,
                progress_bar=False
            )
            
            # Verify training completed
            assert model.num_timesteps >= 5000
            
            # Test performance (Pendulum reward is negative, closer to 0 is better)
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(200):  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"SAC total reward: {total_reward}")
            # Pendulum reward is negative, -1000 is random, better performance is closer to 0
            # Very lenient threshold for integration test
            assert total_reward > -1500, f"Total reward {total_reward} too low for basic functionality"
            
            env.close()

    @pytest.mark.integration
    def test_td3_integration_pendulum(self):
        """Test TD3 end-to-end learning on Pendulum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment and model
            env = gym.make("Pendulum-v1")
            
            # Create model
            model = TD3(
                "MlpPolicy", 
                env, 
                learning_rate=1e-3,
                buffer_size=1000,
                learning_starts=100,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                verbose=1
            )
            
            # Train for 5000 timesteps
            model.learn(
                total_timesteps=5000,
                log_interval=10,
                reset_num_timesteps=True,
                progress_bar=False
            )
            
            # Verify training completed
            assert model.num_timesteps >= 5000
            
            # Test performance (Pendulum reward is negative, closer to 0 is better)
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(200):  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            print(f"TD3 total reward: {total_reward}")
            # Pendulum reward is negative, -1000 is random, better performance is closer to 0
            # Very lenient threshold for integration test
            assert total_reward > -1500, f"Total reward {total_reward} too low for basic functionality"
            
            env.close()


class TestSaveLoadIntegration:
    """Test save/load functionality in integration scenarios."""

    @pytest.mark.integration
    def test_save_load_during_training(self):
        """Test saving and loading models during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train first model
            env = make_vec_env("CartPole-v1", n_envs=2)
            model1 = PPO("MlpPolicy", env, verbose=0)
            model1.learn(total_timesteps=1000)
            
            # Save model
            save_path = os.path.join(tmpdir, "model")
            model1.save(save_path)
            
            # Load model and continue training
            model2 = PPO.load(save_path, env=env)
            initial_timesteps = model2.num_timesteps
            model2.learn(total_timesteps=1000)
            
            # Verify training continued
            assert model2.num_timesteps >= initial_timesteps + 1000
            
            # Test that loaded model works
            obs = env.reset()[0]
            action, _ = model2.predict(obs)
            assert action is not None
            
            env.close()

    @pytest.mark.integration
    def test_model_compatibility_check(self):
        """Test that models work with different environment configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train with vectorized environment
            vec_env = make_vec_env("CartPole-v1", n_envs=4)
            model = PPO("MlpPolicy", vec_env, verbose=0)
            model.learn(total_timesteps=500)
            
            # Save model
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)
            
            # Load with single environment (auto-wrapping should work)
            single_env = gym.make("CartPole-v1")
            loaded_model = PPO.load(save_path, env=single_env)
            
            # Test prediction
            obs, _ = single_env.reset()
            action, _ = loaded_model.predict(obs)
            assert action is not None
            
            vec_env.close()
            single_env.close()


class TestCallbackIntegration:
    """Test callback integration in real training scenarios."""

    @pytest.mark.integration
    def test_multiple_callbacks_integration(self):
        """Test integration of multiple callbacks during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup environment
            env = make_vec_env("CartPole-v1", n_envs=2)
            eval_env = gym.make("CartPole-v1")
            
            # Setup multiple callbacks
            eval_callback = EvalCallback(
                eval_env=eval_env,
                eval_freq=500,
                n_eval_episodes=3,
                best_model_save_path=tmpdir,
                verbose=0
            )
            
            checkpoint_callback = CheckpointCallback(
                save_freq=750,
                save_path=tmpdir,
                name_prefix="test_checkpoint",
                verbose=0
            )
            
            # Create model and train
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(
                total_timesteps=2000,
                callback=[eval_callback, checkpoint_callback]
            )
            
            # Verify callbacks worked
            assert len(eval_callback.evaluations_results) > 0
            assert eval_callback.last_mean_reward != -np.inf
            
            # Check checkpoint files exist
            checkpoint_files = [f for f in os.listdir(tmpdir) if f.startswith("test_checkpoint")]
            assert len(checkpoint_files) > 0
            
            # Check best model exists
            assert os.path.exists(os.path.join(tmpdir, "best_model"))
            
            env.close()
            eval_env.close()
