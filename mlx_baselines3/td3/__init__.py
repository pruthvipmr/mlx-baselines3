"""TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm implementation."""

from mlx_baselines3.td3.policies import (
    TD3Policy,
    MlpPolicy,
    CnnPolicy,
    MultiInputPolicy,
)
from mlx_baselines3.td3.td3 import TD3

__all__ = ["TD3Policy", "TD3", "MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
