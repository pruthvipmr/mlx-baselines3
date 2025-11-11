"""
MLX-specific utilities for device management, tensor operations, and helper
functions.
"""

import math
import random
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import mlx.core as mx
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
    params: Dict[str, mx.array], target_params: Dict[str, mx.array], tau: float
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

    updated_params: Dict[str, mx.array] = {}
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
    assert y_pred.shape == y_true.shape, (
        f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
    )

    var_y = mx.var(y_true)

    # Avoid division by zero
    if mx.abs(var_y) < 1e-8:
        return np.nan

    return float(1 - mx.var(y_true - y_pred) / var_y)


def safe_mean(
    arr: Union[mx.array, np.ndarray, list],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Union[mx.array, float]:
    """
    Compute mean with protection against empty arrays.

    Args:
        arr: Input array (MLX array, numpy array, or list)
        axis: Axis or axes along which to compute mean

    Returns:
        Mean of array, or 0.0 if array is empty
    """
    # Convert to numpy if it's a list or other type
    if isinstance(arr, list):
        arr = np.array(arr)

    # Handle empty arrays
    if hasattr(arr, "size") and arr.size == 0:
        return 0.0
    elif isinstance(arr, list) and len(arr) == 0:
        return 0.0

    # Use appropriate mean function based on type
    if isinstance(arr, mx.array):
        return mx.mean(arr, axis=axis)
    else:
        # numpy array or similar
        return float(np.mean(arr, axis=axis))


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


def get_linear_fn(
    start: float, end: float, end_fraction: float
) -> Callable[[float], float]:
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


def get_schedule_fn(
    value_schedule: Union[float, str, Callable[[float], float]]
) -> Callable[[float], float]:
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
    if isinstance(value_schedule, (int, float)):
        # Constant value
        return lambda _: float(value_schedule)
    if isinstance(value_schedule, str):
        if value_schedule == "linear":
            # Default linear schedule from 1.0 to 0.0
            return get_linear_fn(1.0, 0.0, 1.0)
        raise ValueError(f"Invalid schedule string: {value_schedule}")
    raise ValueError(f"Invalid schedule type: {type(value_schedule)}")


def update_learning_rate(optimizer: Any, learning_rate: float) -> None:
    """
    Update the learning rate of an MLX optimizer.

    Args:
        optimizer: MLX optimizer instance
        learning_rate: New learning rate value
    """
    if hasattr(optimizer, "learning_rate"):
        optimizer.learning_rate = learning_rate
    elif hasattr(optimizer, "lr"):
        optimizer.lr = learning_rate
    else:
        warnings.warn(
            "Could not update learning rate: optimizer has no "
            "'learning_rate' or 'lr' attribute"
        )


def obs_as_mlx(
    obs: Union[np.ndarray, mx.array, Dict[str, Union[np.ndarray, mx.array]]],
) -> Union[mx.array, Dict[str, mx.array]]:
    """
    Convert observation(s) to MLX arrays.

    Args:
        obs: Observation as numpy array, MLX array, or dict of numpy arrays

    Returns:
        Observation as MLX array or dict of MLX arrays
    """
    if isinstance(obs, mx.array):
        return obs  # Already an MLX array
    if isinstance(obs, np.ndarray):
        return mx.array(obs)
    if isinstance(obs, dict):
        converted: Dict[str, mx.array] = {}
        for key, val in obs.items():
            if isinstance(val, mx.array):
                converted[key] = val
            elif isinstance(val, np.ndarray):
                converted[key] = mx.array(val)
            else:
                raise ValueError(
                    f"Unsupported observation value type for key {key!r}: {type(val)}"
                )
        return converted
    raise ValueError(f"Unsupported observation type: {type(obs)}")


def is_vectorized_observation(
    observation: Union[np.ndarray, Mapping[str, np.ndarray]],
    observation_space: Union[np.ndarray, Mapping[str, np.ndarray]],
) -> bool:
    """
    Check if observation is vectorized (from multiple environments).

    Args:
        observation: The observation to check
        observation_space: The observation space of a single environment

    Returns:
        True if observation is vectorized, False otherwise
    """
    if isinstance(observation, Mapping):
        if len(observation) == 0:
            return False
        # Check the first key
        first_key = next(iter(observation.keys()))
        obs_array = observation[first_key]
        if not isinstance(observation_space, Mapping):
            return False
        ref_array = observation_space[first_key]
        return bool(obs_array.shape[0] != ref_array.shape[0])
    assert isinstance(observation, np.ndarray)
    assert isinstance(observation_space, np.ndarray)
    return bool(observation.shape[0] != observation_space.shape[0])


def constant_fn(val: float) -> Callable[[float], float]:
    """
    Create a constant schedule function.

    Args:
        val: Constant value to return

    Returns:
        Function that always returns val
    """
    return lambda _: val
