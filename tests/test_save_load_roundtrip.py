"""
Test save/load round-trip functionality for bit-identical parameters.
"""

import tempfile
import os
import pytest
import mlx.core as mx
import gymnasium as gym
from mlx_baselines3.ppo import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv


def test_ppo_save_load_roundtrip():
    """Test PPO save->load round-trip for bit-identical params."""

    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])

    # Create PPO model
    model1 = PPO("MlpPolicy", env, n_steps=32, verbose=0)

    # Get initial policy state
    initial_state = model1.policy.state_dict()

    # Create a temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")

        # Save the model
        model1.save(save_path)

        # Load the model
        model2 = PPO.load(save_path, env=env)

        # Get loaded policy state
        loaded_state = model2.policy.state_dict()

        # Verify all parameters are bit-identical
        assert set(initial_state.keys()) == set(loaded_state.keys()), (
            "Parameter names should match"
        )

        for name in initial_state:
            initial_param = initial_state[name]
            loaded_param = loaded_state[name]

            assert initial_param.shape == loaded_param.shape, (
                f"Shape mismatch for {name}"
            )
            assert mx.array_equal(initial_param, loaded_param), (
                f"Values not identical for {name}"
            )

        # Test that predictions are identical
        obs = mx.array([[0.1, 0.2, 0.3, 0.4]])  # Batch of 1
        action1, _ = model1.policy.predict(obs, deterministic=True)
        action2, _ = model2.policy.predict(obs, deterministic=True)

        assert mx.array_equal(action1, action2), (
            "Predictions should be identical after loading"
        )


def test_policy_direct_save_load_roundtrip():
    """Test direct policy save/load round-trip."""
    from mlx_baselines3.common.policies import ActorCriticPolicy

    env = gym.make("CartPole-v1")

    # Create two identical policies
    policy1 = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.001,
        net_arch=[32, 32],
    )

    policy2 = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.001,
        net_arch=[32, 32],
    )

    # Get initial states
    state1 = policy1.state_dict()
    state2 = policy2.state_dict()

    # Verify they start different
    params_different = False
    for name in state1:
        if not mx.array_equal(state1[name], state2[name]):
            params_different = True
            break
    assert params_different, "Policies should start with different parameters"

    # Copy state from policy1 to policy2
    policy2.load_state_dict(state1, strict=True)

    # Verify round-trip preserves exact values
    state1_copy = policy1.state_dict()
    state2_new = policy2.state_dict()

    for name in state1_copy:
        assert mx.array_equal(state1_copy[name], state2_new[name]), (
            f"Round-trip failed for {name}"
        )

    # Test predictions are identical
    obs = mx.array([[0.1, 0.2, 0.3, 0.4]])
    action1, _ = policy1.predict(obs, deterministic=True)
    action2, _ = policy2.predict(obs, deterministic=True)

    assert mx.array_equal(action1, action2), (
        "Predictions should be identical after state copy"
    )


def test_parameter_persistence_across_copies():
    """Test that parameter references are properly handled."""
    from mlx_baselines3.common.torch_layers import MlxLinear

    # Create a linear layer
    layer = MlxLinear(10, 5)
    original_state = layer.state_dict()

    # Create a copy
    state_copy = {name: mx.array(param) for name, param in original_state.items()}

    # Modify original parameters
    layer.add_parameter("weight", layer._parameters["weight"] + 1.0)

    # Copy should be unchanged
    for name in state_copy:
        original_modified = layer.state_dict()[name]
        if name == "weight":
            # Original should be different (we added 1.0)
            assert not mx.array_equal(original_modified, state_copy[name])
        else:
            # Other params should be the same
            assert mx.array_equal(original_modified, state_copy[name])

    # Load the copy back
    layer.load_state_dict(state_copy, strict=True)

    # Should now match the original copy
    restored_state = layer.state_dict()
    for name in state_copy:
        assert mx.array_equal(restored_state[name], state_copy[name])


if __name__ == "__main__":
    pytest.main([__file__])
