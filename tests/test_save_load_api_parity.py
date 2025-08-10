"""
Test save/load API parity as specified in the technical spec.

Tests:
1. Save persists model config, policy state dict, spaces, hyperparameters, optimizer state, and env_id
2. Load without env parameter works on standard envs (CartPole, Pendulum)
3. Loading twice yields identical predictions for same observation & deterministic policy seed
4. Backward compatibility - unknown keys warn and skip, don't crash
"""

import tempfile
import os
import warnings
import pytest
import mlx.core as mx
import gymnasium as gym
from mlx_baselines3.ppo import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv


def test_save_persists_all_required_data():
    """Test that save() persists all required data as per spec."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    # Create and train PPO model to ensure optimizer state exists
    model = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    model.learn(total_timesteps=100)  # Train briefly to create optimizer state
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model.save(save_path)
        
        # Load the raw data to verify contents
        import cloudpickle
        with open(save_path, "rb") as f:
            data = cloudpickle.load(f)
        
        # Check all required fields are saved
        required_fields = [
            "policy", "observation_space", "action_space", "n_envs", 
            "num_timesteps", "seed", "learning_rate", "lr_schedule",
            "device", "verbose", "_total_timesteps", "_episode_num",
            "_current_progress_remaining", "env_id", "optimizer_state"
        ]
        
        for field in required_fields:
            assert field in data, f"Required field '{field}' not found in saved data"
        
        # Verify env_id is correctly extracted
        assert data["env_id"] == "CartPole-v1", f"Expected env_id 'CartPole-v1', got {data['env_id']}"
        
        # Verify optimizer state structure
        opt_state = data["optimizer_state"]
        assert "step" in opt_state, "Optimizer state missing 'step'"
        assert "m" in opt_state, "Optimizer state missing 'm' (first moments)"
        assert "v" in opt_state, "Optimizer state missing 'v' (second moments)"
        assert opt_state["step"] > 0, "Step count should be > 0 after training"
        
        # Verify PPO-specific hyperparameters are saved
        ppo_fields = ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", 
                     "clip_range", "clip_range_vf", "ent_coef", "vf_coef", 
                     "max_grad_norm", "target_kl"]
        for field in ppo_fields:
            assert field in data, f"PPO field '{field}' not found in saved data"


def test_load_without_env_parameter():
    """Test that load works without env parameter on standard environments."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    # Create and save model
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    model1.learn(total_timesteps=100)  # Train briefly
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Load without providing env parameter
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model2 = PPO.load(save_path)  # No env parameter
            
            # Check if environment was recreated successfully
            assert model2.env is not None, "Environment should be recreated from env_id"
            assert hasattr(model2.env, "envs"), "Should create DummyVecEnv wrapper"
            
            # Verify the environment works
            obs = model2.env.reset()
            action, _ = model2.predict(obs)
            assert action is not None, "Model should be able to predict actions"


def test_pendulum_env_load_without_env():
    """Test load without env parameter works for Pendulum."""
    env = gym.make("Pendulum-v1")
    env = DummyVecEnv([lambda: env])
    
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    model1.learn(total_timesteps=100)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Load without env parameter
        model2 = PPO.load(save_path)
        
        # Verify environment properties match
        assert model2.env is not None
        obs = model2.env.reset()
        action, _ = model2.predict(obs)
        assert action is not None


def test_deterministic_predictions_after_load():
    """Test that loading twice yields identical predictions for same observation."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    # Create and train model
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0, seed=42)
    model1.learn(total_timesteps=500)  # Train more to ensure meaningful state
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Load the model twice
        model2 = PPO.load(save_path, seed=42)
        model3 = PPO.load(save_path, seed=42)
        
        # Test multiple observations to be thorough
        test_observations = [
            mx.array([[0.1, 0.2, 0.3, 0.4]]),  # Single observation
            mx.array([[0.0, 0.0, 0.0, 0.0]]),  # Zero observation
            mx.array([[-0.5, 1.0, -0.1, 0.8]]), # Mixed values
        ]
        
        for obs in test_observations:
            # Get predictions from both loaded models with deterministic=True
            action2, _ = model2.predict(obs, deterministic=True)
            action3, _ = model3.predict(obs, deterministic=True)
            
            assert mx.array_equal(action2, action3), \
                f"Predictions should be identical after loading. obs={obs}, action2={action2}, action3={action3}"
        
        # Also test that optimizer state is preserved
        assert model2.optimizer_state["step"] == model3.optimizer_state["step"], \
            "Optimizer step count should be identical"
        
        # Test a few parameter keys to ensure they're identical
        state2 = model2.policy.state_dict()
        state3 = model3.policy.state_dict()
        for key in list(state2.keys())[:3]:  # Check first 3 parameters
            assert mx.array_equal(state2[key], state3[key]), f"Parameter {key} differs between loads"


def test_backward_compatibility_unknown_keys():
    """Test that unknown keys in saved file warn and skip, don't crash."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Manually add unknown keys to the saved file
        import cloudpickle
        with open(save_path, "rb") as f:
            data = cloudpickle.load(f)
        
        # Add some future/unknown keys
        data["future_feature_v2"] = "some_value"
        data["unknown_hyperparameter"] = 42
        data["new_optimizer_config"] = {"fancy_setting": True}
        
        # Save the modified data
        with open(save_path, "wb") as f:
            cloudpickle.dump(data, f)
        
        # Load should work but generate warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model2 = PPO.load(save_path)
            
            # Check that warnings were generated
            warning_messages = [str(warning.message) for warning in w]
            unknown_key_warnings = [msg for msg in warning_messages if "Unknown keys" in msg]
            assert len(unknown_key_warnings) > 0, "Should generate warning about unknown keys"
            
            # Model should still work despite unknown keys
            assert model2 is not None
            obs = model2.env.reset()
            action, _ = model2.predict(obs)
            assert action is not None


def test_optimizer_state_restoration():
    """Test that optimizer state (Adam moments) is properly restored."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    # Create and train model to build up optimizer state
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    model1.learn(total_timesteps=1000)  # More training for meaningful optimizer state
    
    # Save optimizer state before saving
    original_opt_state = {
        "step": model1.optimizer_state["step"],
        "m": {k: mx.array(v) for k, v in model1.optimizer_state["m"].items()},
        "v": {k: mx.array(v) for k, v in model1.optimizer_state["v"].items()},
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Load model
        model2 = PPO.load(save_path)
        
        # Verify optimizer state is restored
        assert model2.optimizer_state["step"] == original_opt_state["step"], \
            "Optimizer step count not restored correctly"
        
        # Check that Adam moments are restored
        for key in original_opt_state["m"]:
            assert mx.array_equal(model2.optimizer_state["m"][key], original_opt_state["m"][key]), \
                f"First moment {key} not restored correctly"
                
        for key in original_opt_state["v"]:
            assert mx.array_equal(model2.optimizer_state["v"][key], original_opt_state["v"][key]), \
                f"Second moment {key} not restored correctly"


def test_env_creation_fallback():
    """Test fallback behavior when env_id is missing or invalid."""
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])
    
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        model1.save(save_path)
        
        # Test 1: Remove env_id completely
        import cloudpickle
        with open(save_path, "rb") as f:
            data = cloudpickle.load(f)
        
        del data["env_id"]
        
        with open(save_path, "wb") as f:
            cloudpickle.dump(data, f)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model2 = PPO.load(save_path)
            
            # Should warn about missing env_id and fallback to CartPole-v1
            warning_messages = [str(warning.message) for warning in w]
            fallback_warnings = [msg for msg in warning_messages if "No env_id found" in msg]
            assert len(fallback_warnings) > 0, "Should warn about missing env_id"
            
            # Model should still work with fallback environment
            assert model2.env is not None
        
        # Test 2: Invalid env_id
        data["env_id"] = "NonExistentEnv-v999"
        
        with open(save_path, "wb") as f:
            cloudpickle.dump(data, f)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model3 = PPO.load(save_path)
            
            # Should warn about failed env creation and fallback
            warning_messages = [str(warning.message) for warning in w]
            failed_warnings = [msg for msg in warning_messages if "Failed to recreate environment" in msg]
            assert len(failed_warnings) > 0, "Should warn about failed environment creation"
            
            # Model should still work with fallback
            assert model3.env is not None


if __name__ == "__main__":
    pytest.main([__file__])
