from __future__ import annotations

"""
Array utility functions for converting between NumPy and MLX arrays.

This module provides typed conversion functions to handle the boundary
between NumPy arrays (used in gym environments and buffers) and MLX arrays
(used in neural network computations).
"""

from typing import Any, Mapping, overload

import numpy as np
from numpy.typing import NDArray

from .types import ArrayLike, MlxArray, NumpyArray, Obs, is_dict_obs, is_mlx_array

mx: Any
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - runtime fallback
    mx = None

MLX_AVAILABLE = mx is not None


@overload
def as_numpy(x: NDArray[Any]) -> NDArray[Any]: ...

@overload  
def as_numpy(x: Any) -> NDArray[Any]: ...

def as_numpy(x: ArrayLike) -> NumpyArray:
    """Convert an array-like object to a NumPy array.
    
    Args:
        x: Array-like object (NumPy array, MLX array, or compatible)
        
    Returns:
        NumPy array
    """
    if isinstance(x, np.ndarray):
        return x
    
    if MLX_AVAILABLE and hasattr(x, '__array__'):
        # MLX arrays support __array__ protocol
        return np.asarray(x)
    
    if hasattr(x, 'numpy'):
        # Some frameworks have .numpy() method
        return x.numpy()
        
    return np.asarray(x)


def ensure_mlx(x: ArrayLike) -> MlxArray:
    """Convert an array-like object to an MLX array.
    
    Args:
        x: Array-like object (NumPy array, MLX array, or compatible)
        
    Returns:
        MLX array
        
    Raises:
        ImportError: If MLX is not available
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available. Cannot convert to MLX array.")

    assert mx is not None  # for type checkers

    if is_mlx_array(x):
        return x

    return mx.array(x)


# Overloaded obs_as_mlx functions
@overload
def obs_as_mlx(obs: ArrayLike) -> MlxArray: ...

@overload
def obs_as_mlx(obs: Mapping[str, ArrayLike]) -> dict[str, MlxArray]: ...

def obs_as_mlx(obs: Obs) -> MlxArray | dict[str, MlxArray]:
    """Convert observations to MLX arrays.
    
    Handles both single observations and dictionary observations.
    
    Args:
        obs: Observation(s) to convert
        
    Returns:
        MLX array or dict of MLX arrays
    """
    if is_dict_obs(obs):
        return {key: ensure_mlx(value) for key, value in obs.items()}
    else:
        return ensure_mlx(obs)


# Overloaded obs_as_numpy functions  
@overload
def obs_as_numpy(obs: ArrayLike) -> NumpyArray: ...

@overload
def obs_as_numpy(obs: Mapping[str, ArrayLike]) -> dict[str, NumpyArray]: ...

def obs_as_numpy(obs: Obs) -> NumpyArray | dict[str, NumpyArray]:
    """Convert observations to NumPy arrays.
    
    Handles both single observations and dictionary observations.
    
    Args:
        obs: Observation(s) to convert
        
    Returns:
        NumPy array or dict of NumPy arrays
    """
    if is_dict_obs(obs):
        return {key: as_numpy(value) for key, value in obs.items()}
    else:
        return as_numpy(obs)
