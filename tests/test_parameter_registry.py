"""
Tests for parameter registry functionality (state_dict, load_state_dict, etc.)
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import gymnasium as gym
from mlx_baselines3.common.torch_layers import (
    MlxModule,
    MlxLinear,
)
from mlx_baselines3.common.policies import ActorCriticPolicy
from mlx_baselines3.ppo import PPO


class SimpleTestModule(MlxModule):
    """Simple test module for parameter registry tests."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = MlxLinear(input_dim, 32)
        self.linear2 = MlxLinear(32, output_dim)

        # Register submodules
        self.add_module("linear1", self.linear1)
        self.add_module("linear2", self.linear2)

    def __call__(self, x):
        x = nn.relu(self.linear1(x))
        return self.linear2(x)


def test_mlx_module_parameters():
    """Test that MlxModule properly tracks parameters."""
    module = SimpleTestModule(10, 5)

    # Check parameters method
    params = module.parameters()
    assert len(params) == 4  # 2 weights + 2 biases

    # Check parameter names
    expected_names = {
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    }
    assert set(params.keys()) == expected_names

    # Check parameter shapes
    assert params["linear1.weight"].shape == (32, 10)
    assert params["linear1.bias"].shape == (32,)
    assert params["linear2.weight"].shape == (5, 32)
    assert params["linear2.bias"].shape == (5,)


def test_mlx_module_named_parameters():
    """Test that named_parameters returns same as parameters."""
    module = SimpleTestModule(10, 5)

    params = module.parameters()
    named_params = module.named_parameters()

    assert params.keys() == named_params.keys()
    for name in params:
        assert mx.array_equal(params[name], named_params[name])


def test_mlx_module_state_dict():
    """Test state_dict functionality."""
    module = SimpleTestModule(10, 5)

    state_dict = module.state_dict()
    params = module.parameters()

    # state_dict should return same as parameters
    assert state_dict.keys() == params.keys()
    for name in state_dict:
        assert mx.array_equal(state_dict[name], params[name])


def test_mlx_module_load_state_dict_strict():
    """Test load_state_dict with strict=True."""
    module1 = SimpleTestModule(10, 5)
    module2 = SimpleTestModule(10, 5)

    # Get initial state
    state1 = module1.state_dict()
    state2 = module2.state_dict()

    # Verify they start different (due to random initialization)
    params_different = False
    for name in state1:
        if not mx.array_equal(state1[name], state2[name]):
            params_different = True
            break
    assert params_different, "Modules should start with different parameters"

    # Load state from module1 into module2
    module2.load_state_dict(state1, strict=True)

    # Verify they're now identical
    state2_new = module2.state_dict()
    for name in state1:
        assert mx.array_equal(state1[name], state2_new[name])


def test_mlx_module_load_state_dict_strict_missing_keys():
    """Test load_state_dict strict mode with missing keys."""
    module = SimpleTestModule(10, 5)

    # Create incomplete state dict
    state_dict = module.state_dict()
    incomplete_state = {k: v for k, v in state_dict.items() if "linear1" in k}

    # Should raise KeyError for missing keys
    with pytest.raises(KeyError, match="Missing keys"):
        module.load_state_dict(incomplete_state, strict=True)


def test_mlx_module_load_state_dict_strict_unexpected_keys():
    """Test load_state_dict strict mode with unexpected keys."""
    module = SimpleTestModule(10, 5)

    # Create state dict with extra keys
    state_dict = module.state_dict()
    state_dict["extra_param"] = mx.array([1.0, 2.0])

    # Should raise KeyError for unexpected keys
    with pytest.raises(KeyError, match="Unexpected keys"):
        module.load_state_dict(state_dict, strict=True)


def test_mlx_module_load_state_dict_non_strict():
    """Test load_state_dict with strict=False."""
    module = SimpleTestModule(10, 5)

    # Create state dict with extra and missing keys
    state_dict = module.state_dict()
    state_dict["extra_param"] = mx.array([1.0, 2.0])
    del state_dict["linear2.bias"]

    # Should work with strict=False (may print warnings)
    module.load_state_dict(state_dict, strict=False)

    # Verify parameters that were loaded
    new_state = module.state_dict()
    for name in ["linear1.weight", "linear1.bias", "linear2.weight"]:
        assert mx.array_equal(state_dict[name], new_state[name])


def test_mlx_module_load_state_dict_shape_mismatch():
    """Test load_state_dict with shape mismatch."""
    module = SimpleTestModule(10, 5)

    # Create state dict with wrong shape
    state_dict = module.state_dict()
    state_dict["linear1.weight"] = mx.ones((16, 10))  # Wrong shape

    # Should raise ValueError for shape mismatch
    with pytest.raises(ValueError, match="shape mismatch"):
        module.load_state_dict(state_dict, strict=True)


def test_actor_critic_policy_parameters():
    """Test parameter registry for ActorCriticPolicy."""
    env = gym.make("CartPole-v1")
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.001,
        net_arch=[32, 32],
    )

    # Check that all expected parameters are present
    params = policy.parameters()

    # Should have parameters from features extractor, action net, and value net
    param_prefixes = {name.split(".")[0] for name in params.keys()}
    expected_prefixes = {"features_extractor", "action_net", "value_net"}

    # features_extractor might be None for simple spaces, so check what's actually there
    assert len(param_prefixes.intersection(expected_prefixes)) > 0

    # Check that action_net and value_net parameters are present
    action_params = [name for name in params.keys() if name.startswith("action_net")]
    value_params = [name for name in params.keys() if name.startswith("value_net")]

    assert len(action_params) > 0, "Should have action network parameters"
    assert len(value_params) > 0, "Should have value network parameters"


def test_actor_critic_policy_state_dict_roundtrip():
    """Test save->load round-trip for ActorCriticPolicy."""
    env = gym.make("CartPole-v1")
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

    # Verify they're different initially
    params_different = False
    for name in state1:
        if not mx.array_equal(state1[name], state2[name]):
            params_different = True
            break
    assert params_different, "Policies should start with different parameters"

    # Load state from policy1 into policy2
    policy2.load_state_dict(state1, strict=True)

    # Verify they're now identical
    state2_new = policy2.state_dict()
    for name in state1:
        assert mx.array_equal(state1[name], state2_new[name])

    # Test that predictions are now identical
    obs = mx.array([[0.1, 0.2, 0.3, 0.4]])  # Batch of 1
    action1, _ = policy1.predict(obs, deterministic=True)
    action2, _ = policy2.predict(obs, deterministic=True)

    assert mx.array_equal(action1, action2), (
        "Predictions should be identical after loading state"
    )


def test_ppo_state_dict_integration():
    """Test that PPO correctly uses state_dict functionality."""
    from mlx_baselines3.common.vec_env import DummyVecEnv

    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO("MlpPolicy", env, n_steps=32, verbose=0)

    # Get policy state dict
    policy_state = model.policy.state_dict()

    # Should have parameters
    assert len(policy_state) > 0

    # Test that we can save and load
    state_copy = {name: mx.array(param) for name, param in policy_state.items()}
    model.policy.load_state_dict(state_copy, strict=True)

    # Verify state is preserved
    new_state = model.policy.state_dict()
    for name in policy_state:
        assert mx.array_equal(policy_state[name], new_state[name])


if __name__ == "__main__":
    pytest.main([__file__])
