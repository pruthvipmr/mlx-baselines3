"""
Vectorized environment module for MLX Baselines3.

This module provides vectorized environment implementations that allow
running multiple environments in parallel or sequentially.
"""

from mlx_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from mlx_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv, make_vec_env

__all__ = [
    "VecEnv",
    "VecEnvWrapper", 
    "DummyVecEnv",
    "make_vec_env",
]
