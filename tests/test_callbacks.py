"""Tests for the callback system."""

import os
import tempfile
import numpy as np
import pytest
import gymnasium as gym

from mlx_baselines3.common.callbacks import (
    BaseCallback, 
    CallbackList, 
    CheckpointCallback, 
    EvalCallback,
    StopTrainingOnRewardThreshold,
    ProgressBarCallback,
    convert_callback
)
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.ppo import PPO


def make_vec_env(env_id="CartPole-v1", n_envs=1):
    """Helper to create vectorized environment for tests."""
    return DummyVecEnv([lambda: gym.make(env_id) for _ in range(n_envs)])


class DummyCallback(BaseCallback):
    """Dummy callback for testing."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_started = False
        self.steps_called = 0
        self.training_ended = False
        self.rollout_started = False
        self.rollout_ended = False

    def _init_callback(self):
        pass

    def _on_training_start(self):
        self.training_started = True

    def _on_step(self):
        self.steps_called += 1
        return True

    def _on_training_end(self):
        self.training_ended = True

    def _on_rollout_start(self):
        self.rollout_started = True

    def _on_rollout_end(self):
        self.rollout_ended = True


class StopAtStepCallback(BaseCallback):
    """Callback that stops training at a specific step."""
    
    def __init__(self, stop_at_step, verbose=0):
        super().__init__(verbose)
        self.stop_at_step = stop_at_step

    def _init_callback(self):
        pass

    def _on_training_start(self):
        pass

    def _on_step(self):
        return self.n_calls < self.stop_at_step

    def _on_training_end(self):
        pass


class TestBaseCallback:
    """Test base callback functionality."""

    def test_dummy_callback_init(self):
        """Test dummy callback initialization."""
        callback = DummyCallback()
        assert callback.verbose == 0
        assert callback.model is None
        assert callback.n_calls == 0
        assert not callback.training_started

    def test_dummy_callback_with_model(self):
        """Test dummy callback with model."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = DummyCallback()
        callback.init_callback(model)
        
        assert callback.model is model
        assert callback.training_env is model.env

    def test_callback_lifecycle(self):
        """Test callback lifecycle methods."""
        callback = DummyCallback()
        
        # Test lifecycle
        callback.on_training_start({}, {})
        assert callback.training_started
        
        # Simulate steps
        for _ in range(5):
            result = callback.on_step()
            assert result is True
        assert callback.steps_called == 5
        
        callback.on_training_end()
        assert callback.training_ended


class TestCallbackList:
    """Test callback list functionality."""

    def test_callback_list_init(self):
        """Test callback list initialization."""
        callbacks = [DummyCallback(), DummyCallback()]
        callback_list = CallbackList(callbacks)
        
        assert len(callback_list.callbacks) == 2
        assert all(isinstance(cb, DummyCallback) for cb in callback_list.callbacks)

    def test_callback_list_lifecycle(self):
        """Test callback list lifecycle."""
        dummy1 = DummyCallback()
        dummy2 = DummyCallback()
        callback_list = CallbackList([dummy1, dummy2])
        
        # Mock model
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        callback_list.init_callback(model)
        
        # Test lifecycle
        callback_list.on_training_start({}, {})
        assert dummy1.training_started and dummy2.training_started
        
        # Test steps
        for _ in range(3):
            result = callback_list.on_step()
            assert result is True
        
        assert dummy1.steps_called == 3 and dummy2.steps_called == 3
        
        callback_list.on_training_end()
        assert dummy1.training_ended and dummy2.training_ended

    def test_callback_list_early_stop(self):
        """Test callback list early stopping."""
        stop_callback = StopAtStepCallback(stop_at_step=2)
        dummy_callback = DummyCallback()
        callback_list = CallbackList([dummy_callback, stop_callback])
        
        # Test steps
        assert callback_list.on_step() is True  # step 1
        assert callback_list.on_step() is False  # step 2 (stops)


class TestCheckpointCallback:
    """Test checkpoint callback functionality."""

    def test_checkpoint_callback_init(self):
        """Test checkpoint callback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                name_prefix="test",
                verbose=1
            )
            assert callback.save_freq == 1000
            assert callback.save_path == tmpdir
            assert callback.name_prefix == "test"

    def test_checkpoint_callback_save(self):
        """Test checkpoint callback saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_vec_env("CartPole-v1")
            model = PPO("MlpPolicy", env, verbose=0)
            
            callback = CheckpointCallback(
                save_freq=100,
                save_path=tmpdir,
                name_prefix="test",
                verbose=0
            )
            callback.init_callback(model)
            
            # Simulate timesteps and trigger save
            model.num_timesteps = 100
            callback.n_calls = 99  # Will be incremented to 100 in on_step()
            result = callback.on_step()
            
            assert result is True
            # Check if checkpoint file was created
            expected_file = os.path.join(tmpdir, "test_100_steps")
            assert os.path.exists(expected_file)


class TestEvalCallback:
    """Test evaluation callback functionality."""

    def test_eval_callback_init(self):
        """Test eval callback initialization."""
        eval_env = gym.make("CartPole-v1")
        callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=3,
            eval_freq=1000,
            verbose=1
        )
        
        assert callback.eval_env is eval_env
        assert callback.n_eval_episodes == 3
        assert callback.eval_freq == 1000
        assert callback.best_mean_reward == -np.inf

    def test_eval_callback_no_eval(self):
        """Test eval callback when eval_freq not reached."""
        eval_env = gym.make("CartPole-v1")
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=1000,
            verbose=0
        )
        callback.init_callback(model)
        
        # Don't reach eval_freq
        callback.n_calls = 500
        result = callback.on_step()
        
        assert result is True
        assert callback.last_mean_reward == -np.inf

    def test_eval_callback_evaluation(self):
        """Test eval callback evaluation process."""
        eval_env = gym.make("CartPole-v1")
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = EvalCallback(
                eval_env=eval_env,
                eval_freq=10,  # Small freq for testing
                n_eval_episodes=2,
                best_model_save_path=tmpdir,
                verbose=0
            )
            callback.init_callback(model)
            
            # Trigger evaluation
            callback.n_calls = 9  # Will be incremented to 10 in on_step()
            result = callback.on_step()
            
            assert result is True
            assert callback.last_mean_reward != -np.inf
            assert len(callback.evaluations_results) > 0


class TestStopTrainingOnRewardThreshold:
    """Test stop training callback functionality."""

    def test_stop_training_callback_init(self):
        """Test stop training callback initialization."""
        callback = StopTrainingOnRewardThreshold(
            reward_threshold=200.0,
            verbose=1
        )
        assert callback.reward_threshold == 200.0

    def test_stop_training_no_episodes(self):
        """Test stop training when no episodes completed."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = StopTrainingOnRewardThreshold(
            reward_threshold=200.0,
            verbose=0
        )
        callback.init_callback(model)
        
        # No episodes in buffer
        result = callback.on_step()
        assert result is True

    def test_stop_training_threshold_reached(self):
        """Test stop training when threshold is reached."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = StopTrainingOnRewardThreshold(
            reward_threshold=150.0,
            verbose=0
        )
        callback.init_callback(model)
        
        # Add high reward episodes to buffer
        model.ep_info_buffer = [{'r': 200.0} for _ in range(100)]
        
        result = callback.on_step()
        assert result is False  # Should stop training


class TestProgressBarCallback:
    """Test progress bar callback functionality."""

    def test_progress_bar_callback_init(self):
        """Test progress bar callback initialization."""
        callback = ProgressBarCallback(verbose=1)
        assert callback.verbose == 1

    def test_progress_bar_no_tqdm(self):
        """Test progress bar callback without tqdm."""
        callback = ProgressBarCallback(verbose=0)
        callback._init_callback()
        
        # Should handle missing tqdm gracefully
        callback.on_training_start({'total_timesteps': 1000}, {})
        result = callback.on_step()
        assert result is True


class TestConvertCallback:
    """Test callback conversion utility."""

    def test_convert_none_callback(self):
        """Test converting None callback."""
        callback = convert_callback(None)
        
        # Should have required methods
        assert hasattr(callback, 'init_callback')
        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_step')
        assert hasattr(callback, 'on_training_end')
        
        # Test that methods work
        callback.init_callback(None)
        callback.on_training_start()
        assert callback.on_step() is True
        callback.on_training_end()

    def test_convert_single_callback(self):
        """Test converting single callback."""
        dummy = DummyCallback()
        callback = convert_callback(dummy)
        
        assert callback is dummy

    def test_convert_callback_list(self):
        """Test converting callback list."""
        dummy1 = DummyCallback()
        dummy2 = DummyCallback()
        callback = convert_callback([dummy1, dummy2])
        
        assert isinstance(callback, CallbackList)
        assert len(callback.callbacks) == 2


class TestCallbackIntegration:
    """Test callback integration with algorithms."""

    def test_ppo_with_dummy_callback(self):
        """Test PPO with dummy callback."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        callback = DummyCallback()
        
        # Train for a few steps
        model.learn(total_timesteps=100, callback=callback)
        
        assert callback.training_started
        assert callback.steps_called > 0
        assert callback.training_ended

    def test_ppo_with_early_stop(self):
        """Test PPO with early stopping callback."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Stop after 2 callback calls
        callback = StopAtStepCallback(stop_at_step=2)
        
        # Train should stop early
        model.learn(total_timesteps=10000, callback=callback)
        
        # Should have stopped early
        assert model.num_timesteps < 10000

    def test_ppo_with_checkpoint_callback(self):
        """Test PPO with checkpoint callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_vec_env("CartPole-v1")
            model = PPO("MlpPolicy", env, verbose=0)
            
            callback = CheckpointCallback(
                save_freq=50,
                save_path=tmpdir,
                name_prefix="ppo_test",
                verbose=0
            )
            
            # Train and trigger checkpoint
            model.learn(total_timesteps=100, callback=callback)
            
            # Check checkpoint was created
            checkpoint_files = [f for f in os.listdir(tmpdir) if f.startswith("ppo_test")]
            assert len(checkpoint_files) > 0

    def test_ppo_with_callback_list(self):
        """Test PPO with multiple callbacks."""
        env = make_vec_env("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        dummy1 = DummyCallback()
        dummy2 = DummyCallback()
        callbacks = [dummy1, dummy2]
        
        # Train with callback list
        model.learn(total_timesteps=100, callback=callbacks)
        
        # Both callbacks should have been called
        assert dummy1.training_started and dummy2.training_started
        assert dummy1.steps_called > 0 and dummy2.steps_called > 0
        assert dummy1.training_ended and dummy2.training_ended


if __name__ == "__main__":
    pytest.main([__file__])
