"""MLX Baselines3: Stable Baselines 3 implementation using Apple's MLX framework."""

__version__ = "0.1.0"

# Algorithms - implemented
from mlx_baselines3.ppo import PPO

# Policy aliases - available for implemented algorithms
from mlx_baselines3.ppo import MlpPolicy, CnnPolicy, MultiInputPolicy

# Algorithms - not yet implemented (placeholder imports for API compatibility)
class A2C:
    """A2C algorithm - not yet implemented"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("A2C is not yet implemented in MLX Baselines3")

class DQN:
    """DQN algorithm - not yet implemented"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DQN is not yet implemented in MLX Baselines3")

class SAC:
    """SAC algorithm - not yet implemented"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SAC is not yet implemented in MLX Baselines3")

class TD3:
    """TD3 algorithm - not yet implemented"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TD3 is not yet implemented in MLX Baselines3")


def show_versions():
    """Show versions of MLX Baselines3 and its dependencies."""
    import sys
    import platform
    
    try:
        import mlx.core
        # MLX doesn't have __version__ attribute, so we check if it's importable
        mlx_version = "available (version unknown)"
    except ImportError:
        mlx_version = "not available"
    
    try:
        import gymnasium
        gym_version = gymnasium.__version__
    except (ImportError, AttributeError):
        gym_version = "not available"
    
    try:
        import numpy
        numpy_version = numpy.__version__
    except (ImportError, AttributeError):
        numpy_version = "not available"

    print(f"MLX Baselines3: {__version__}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"MLX: {mlx_version}")
    print(f"Gymnasium: {gym_version}")
    print(f"NumPy: {numpy_version}")


__all__ = [
    # Algorithms
    "PPO", "A2C", "DQN", "SAC", "TD3",
    # Policy aliases
    "MlpPolicy", "CnnPolicy", "MultiInputPolicy",
    # Utilities
    "show_versions",
]
