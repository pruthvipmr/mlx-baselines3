"""MLX-specific utilities for device management, tensor operations, and helper functions."""

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def get_device() -> str:
    """
    Get the best available device for MLX operations.
    
    Returns:
        str: 'gpu' if MLX GPU is available, else 'cpu'
    """
    try:
        # Check if MLX GPU is available by attempting a simple operation
        test_array = mx.array([1.0])
        mx.eval(test_array)
        return "gpu"
    except Exception:
        return "cpu"


def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seeds for reproducible experiments across all libraries.
    
    Args:
        seed: Random seed value. If None, no seeding is performed.
    """
    if seed is None:
        return
        
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set MLX random seed
    mx.random.seed(seed)


def polyak_update(
    params: Dict[str, mx.array], 
    target_params: Dict[str, mx.array], 
    tau: float
) -> Dict[str, mx.array]:
    """
    Perform Polyak averaging (soft update) of target network parameters.
    
    target_params = tau * params + (1 - tau) * target_params
    
    Args:
        params: Source network parameters
        target_params: Target network parameters to update
        tau: Soft update coefficient (0 < tau <= 1)
        
    Returns:
        Updated target parameters
    """
    if not (0 < tau <= 1):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    
    updated_params = {}
    for key in params.keys():
        if key not in target_params:
            warnings.warn(f"Key '{key}' found in params but not in target_params")
            continue
        updated_params[key] = tau * params[key] + (1 - tau) * target_params[key]
    
    return updated_params


def explained_variance(y_pred: mx.array, y_true: mx.array) -> float:
    """
    Compute the explained variance between predictions and true values.
    
    explained_var = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance as a float between -inf and 1
        (1 = perfect prediction, 0 = as good as predicting the mean)
    """
    assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
    
    var_y = mx.var(y_true)
    
    # Avoid division by zero
    if mx.abs(var_y) < 1e-8:
        return np.nan
    
    return float(1 - mx.var(y_true - y_pred) / var_y)


def safe_mean(arr: mx.array, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> mx.array:
    """
    Compute mean with protection against empty arrays.
    
    Args:
        arr: Input array
        axis: Axis or axes along which to compute mean
        
    Returns:
        Mean of array, or 0.0 if array is empty
    """
    if arr.size == 0:
        return mx.array(0.0)
    return mx.mean(arr, axis=axis)


def numpy_to_mlx(arr: np.ndarray) -> mx.array:
    """
    Convert NumPy array to MLX array.
    
    Args:
        arr: NumPy array
        
    Returns:
        MLX array
    """
    return mx.array(arr)


def mlx_to_numpy(arr: mx.array) -> np.ndarray:
    """
    Convert MLX array to NumPy array.
    
    Args:
        arr: MLX array
        
    Returns:
        NumPy array
    """
    return np.array(arr)


def get_linear_fn(start: float, end: float, end_fraction: float) -> callable:
    """
    Create a linear schedule function.
    
    Args:
        start: Initial value
        end: Final value
        end_fraction: Fraction of training when end value is reached (0 to 1)
        
    Returns:
        Schedule function that takes progress (0 to 1) and returns current value
    """
    def func(progress_remaining: float) -> float:
        if end_fraction > 0:
            # Convert progress_remaining to progress_made
            progress_made = 1 - progress_remaining
            if progress_made >= end_fraction:
                return end
            else:
                # Linear interpolation
                return start + (end - start) * (progress_made / end_fraction)
        else:
            return end
    
    return func


def get_schedule_fn(value_schedule: Union[float, str, callable]) -> callable:
    """
    Transform schedule string/value to a schedule function.
    
    Args:
        value_schedule: Either a constant value, a string ('linear'), 
                       or a callable schedule function
                       
    Returns:
        Schedule function
    """
    if callable(value_schedule):
        return value_schedule
    elif isinstance(value_schedule, (int, float)):
        # Constant value
        return lambda _: float(value_schedule)
    elif isinstance(value_schedule, str):
        if value_schedule == "linear":
            # Default linear schedule from 1.0 to 0.0
            return get_linear_fn(1.0, 0.0, 1.0)
        else:
            raise ValueError(f"Invalid schedule string: {value_schedule}")
    else:
        raise ValueError(f"Invalid schedule type: {type(value_schedule)}")


def update_learning_rate(optimizer: Any, learning_rate: float) -> None:
    """
    Update the learning rate of an MLX optimizer.
    
    Args:
        optimizer: MLX optimizer instance
        learning_rate: New learning rate value
    """
    if hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate = learning_rate
    elif hasattr(optimizer, 'lr'):
        optimizer.lr = learning_rate
    else:
        warnings.warn("Could not update learning rate: optimizer has no 'learning_rate' or 'lr' attribute")


def clip_grad_norm(grads: Dict[str, mx.array], max_norm: float) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        grads: Dictionary of gradients
        max_norm: Maximum allowed gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    # Compute total gradient norm
    total_norm = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm += mx.sum(grad ** 2)
    
    total_norm = float(mx.sqrt(total_norm))
    
    # Clip gradients if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for key in grads:
            if grads[key] is not None:
                grads[key] = grads[key] * clip_coef
    
    return total_norm


def obs_as_mlx(obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[mx.array, Dict[str, mx.array]]:
    """
    Convert observation(s) to MLX arrays.
    
    Args:
        obs: Observation as numpy array or dict of numpy arrays
        
    Returns:
        Observation as MLX array or dict of MLX arrays
    """
    if isinstance(obs, np.ndarray):
        return mx.array(obs)
    elif isinstance(obs, dict):
        return {key: mx.array(val) for key, val in obs.items()}
    else:
        raise ValueError(f"Unsupported observation type: {type(obs)}")


def is_vectorized_observation(observation: Union[np.ndarray, Dict], observation_space) -> bool:
    """
    Check if observation is vectorized (from multiple environments).
    
    Args:
        observation: The observation to check
        observation_space: The observation space of a single environment
        
    Returns:
        True if observation is vectorized, False otherwise
    """
    if isinstance(observation, dict):
        if len(observation) == 0:
            return False
        # Check the first key
        first_key = next(iter(observation.keys()))
        return observation[first_key].shape[0] != observation_space[first_key].shape[0]
    else:
        return observation.shape[0] != observation_space.shape[0]


def constant_fn(val: float) -> callable:
    """
    Create a constant schedule function.
    
    Args:
        val: Constant value to return
        
    Returns:
        Function that always returns val
    """
    return lambda _: val
