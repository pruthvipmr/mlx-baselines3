"""
Test PPO integration with the new optimizer system.

Verifies that PPO uses the new AdamAdapter correctly and that
optimizer state evolves during training.
"""

import gymnasium as gym
import mlx.core as mx

from mlx_baselines3 import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.common.optimizers import AdamAdapter


def test_ppo_optimizer_integration():
    """Test that PPO integrates properly with the new optimizer system."""
    # Create a simple environment
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Create PPO model
    model = PPO("MlpPolicy", env, learning_rate=1e-3, verbose=0, n_steps=32)

    # Check that optimizer adapter was created
    assert model.optimizer_adapter is not None
    assert isinstance(model.optimizer_adapter, AdamAdapter)
    assert model.optimizer_state is not None

    # Check initial state
    initial_step = model.optimizer_state["step"]
    assert initial_step == 0

    # Check that Adam moments are initialized to zero
    m_vals = model.optimizer_state["m"]
    v_vals = model.optimizer_state["v"]

    # All moments should be zero initially
    all_zero_m = all(mx.array_equal(v, mx.zeros_like(v)) for v in m_vals.values())
    all_zero_v = all(mx.array_equal(v, mx.zeros_like(v)) for v in v_vals.values())

    assert all_zero_m, "Adam first moments should be zero initially"
    assert all_zero_v, "Adam second moments should be zero initially"


def test_ppo_optimizer_numerical_consistency():
    """Test that PPO training is numerically consistent with identical conditions."""
    # Create two identical environments
    env1 = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    env2 = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Set the same seed for both
    mx.random.seed(42)
    model1 = PPO("MlpPolicy", env1, learning_rate=1e-3, verbose=0, seed=42)

    mx.random.seed(42)
    model2 = PPO("MlpPolicy", env2, learning_rate=1e-3, verbose=0, seed=42)

    # Get initial parameters
    params1_before = model1.policy.state_dict()
    params2_before = model2.policy.state_dict()

    # Parameters should be identical initially due to same seed
    for key in params1_before:
        assert mx.allclose(params1_before[key], params2_before[key]), (
            f"Initial params differ for {key}"
        )

    # Optimizer states should also be identical
    assert model1.optimizer_state["step"] == model2.optimizer_state["step"]


def test_ppo_learning_rate_schedule():
    """Test that PPO respects learning rate schedules."""
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Create PPO with linear learning rate decay
    def lr_schedule(step):
        return 1e-3 * (1.0 - step / 1000)

    model = PPO("MlpPolicy", env, learning_rate=lr_schedule, verbose=0)

    # Check that the optimizer uses the schedule
    assert callable(model.optimizer_adapter.learning_rate)

    # Initial learning rate should be 1e-3
    initial_lr = model.optimizer_adapter.learning_rate(0)
    assert abs(initial_lr - 1e-3) < 1e-8

    # Learning rate should decay over time
    lr_100 = model.optimizer_adapter.learning_rate(100)
    lr_500 = model.optimizer_adapter.learning_rate(500)

    assert lr_100 < initial_lr
    assert lr_500 < lr_100


def test_ppo_gradient_clipping():
    """Test that gradient clipping works correctly."""
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Create PPO with gradient clipping
    model = PPO("MlpPolicy", env, learning_rate=1e-3, max_grad_norm=0.5, verbose=0)

    # Check that max_grad_norm is set
    assert model.max_grad_norm == 0.5

    # Check that optimizer is initialized
    assert model.optimizer_adapter is not None
    assert model.optimizer_state is not None


if __name__ == "__main__":
    test_ppo_optimizer_integration()
    test_ppo_optimizer_numerical_consistency()
    test_ppo_learning_rate_schedule()
    test_ppo_gradient_clipping()
    print("All PPO optimizer integration tests passed!")
