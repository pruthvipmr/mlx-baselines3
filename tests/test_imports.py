"""Test basic imports and package structure."""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import mlx_baselines3
    assert mlx_baselines3.__version__ == "0.1.0"


def test_mlx_import():
    """Test that MLX is available."""
    import mlx.core as mx
    import mlx.nn
    
    # Basic tensor creation
    x = mx.array([1, 2, 3])
    assert x.shape == (3,)


def test_gymnasium_import():
    """Test that Gymnasium is available."""
    import gymnasium as gym
    
    # Create a simple environment
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    assert obs is not None
    env.close()


def test_common_package():
    """Test that common package exists."""
    import mlx_baselines3.common
    import mlx_baselines3.common.vec_env
