"""
Observation preprocessing utilities for MLX Baselines3.

This module provides functions for preprocessing observations before feeding them
to neural networks, with special handling for image observations and normalization.
"""

import warnings
from typing import Any, Dict, List, Tuple, TypeVar, Union, cast

import gymnasium as gym
import numpy as np
import mlx.core as mx

from mlx_baselines3.common.type_aliases import (
    MlxArray,
    ObsType,
    NumpyObsType,
    GymSpace,
)


def is_image_space(observation_space: GymSpace, check_channels: bool = True) -> bool:
    """
    Check if an observation space is an image space.

    This function determines whether an observation space represents image data
    based on its shape and data type.

    Args:
        observation_space: The observation space to check
        check_channels: If True, check that the last or first dimension
                       has a reasonable number of channels (1, 3, or 4)

    Returns:
        True if the observation space is likely an image space
    """
    if isinstance(observation_space, gym.spaces.Box):
        shape = observation_space.shape

        # Must be 2D or 3D
        if len(shape) < 2 or len(shape) > 3:
            return False

        # Check data type - images are typically uint8
        if observation_space.dtype != np.uint8:
            return False

        # For 3D, check if channels are in reasonable range
        if len(shape) == 3 and check_channels:
            # Check if first or last dimension could be channels
            channels_first = shape[0] in [1, 3, 4]  # (C, H, W)
            channels_last = shape[-1] in [1, 3, 4]  # (H, W, C)

            if not (channels_first or channels_last):
                return False

        # Check if spatial dimensions are reasonable for images
        if len(shape) >= 2:
            if len(shape) == 2:
                # Grayscale image (H, W)
                height, width = shape[0], shape[1]
            elif len(shape) == 3:
                if shape[0] in [1, 3, 4]:
                    # Channels first format (C, H, W)
                    height, width = shape[1], shape[2]
                else:
                    # Channels last format (H, W, C)
                    height, width = shape[0], shape[1]

            # Images should have reasonable spatial dimensions
            if height < 8 or width < 8 or height > 2048 or width > 2048:
                return False

        return True

    elif isinstance(observation_space, gym.spaces.Dict):
        # Check if any sub-space is an image space
        return any(
            is_image_space(subspace, check_channels)
            for subspace in observation_space.spaces.values()
        )

    return False


def is_image_space_channels_first(observation_space: gym.spaces.Box) -> bool:
    """
    Check if an image observation space uses channels-first format.

    Args:
        observation_space: Box observation space

    Returns:
        True if channels are first (C, H, W), False if channels are last (H, W, C)
    """
    if not isinstance(observation_space, gym.spaces.Box):
        return False

    shape = observation_space.shape
    if len(shape) != 3:
        return False

    # If first dimension is 1, 3, or 4 and others are larger, likely channels first
    if shape[0] in [1, 3, 4] and shape[1] >= 8 and shape[2] >= 8:
        return True

    return False


def maybe_transpose(
    observation: np.ndarray, observation_space: gym.spaces.Box
) -> np.ndarray:
    """
    Transpose image observations if needed to ensure channels-first format.

    Most deep learning frameworks (including MLX) expect images in channels-first
    format (C, H, W), but some environments provide channels-last (H, W, C).

    Args:
        observation: The observation array to potentially transpose
        observation_space: The observation space

    Returns:
        Observation in channels-first format if it's an image, unchanged otherwise
    """
    if not is_image_space(observation_space):
        return observation

    if len(observation.shape) == 3:
        # If observation space indicates channels-first, return as is
        if is_image_space_channels_first(observation_space):
            return observation
        else:
            # Convert from (H, W, C) to (C, H, W)
            return np.transpose(observation, (2, 0, 1))
    elif len(observation.shape) == 4:
        # Batch of images - check last observation space
        if is_image_space_channels_first(observation_space):
            return observation
        else:
            # Convert from (N, H, W, C) to (N, C, H, W)
            return np.transpose(observation, (0, 3, 1, 2))

    return observation


ArrayLikeT = TypeVar("ArrayLikeT", np.ndarray, MlxArray)


def normalize_image(observation: ArrayLikeT, dtype: Any = np.float32) -> ArrayLikeT:
    """
    Normalize image observations from [0, 255] to [0, 1].

    Args:
        observation: Image observation(s) to normalize
        dtype: Target data type for normalized observation

    Returns:
        Normalized observation in range [0, 1]
    """
    if isinstance(observation, mx.array):
        # MLX array
        return cast(ArrayLikeT, mx.array(observation, dtype=mx.float32) / 255.0)
    # NumPy array
    return cast(ArrayLikeT, observation.astype(dtype) / 255.0)


def preprocess_obs(
    obs: NumpyObsType,
    observation_space: GymSpace,
    normalize_images: bool = True,
    transpose_images: bool = True,
) -> NumpyObsType:
    """
    Preprocess observations for neural networks.

    This function handles common preprocessing steps:
    - Transposing image observations to channels-first format
    - Normalizing image pixel values from [0, 255] to [0, 1]
    - Handling dictionary observation spaces

    Args:
        obs: Observation(s) to preprocess
        observation_space: The observation space
        normalize_images: Whether to normalize image observations to [0, 1]
        transpose_images: Whether to transpose images to channels-first format

    Returns:
        Preprocessed observation(s)
    """
    if isinstance(observation_space, gym.spaces.Dict):
        # Handle dictionary observations
        if not isinstance(obs, dict):
            raise ValueError("Expected dict observation for Dict observation space")

        preprocessed_obs: Dict[str, NumpyObsType] = {}
        for key, subspace in observation_space.spaces.items():
            if key in obs:
                preprocessed_obs[key] = preprocess_obs(
                    obs[key], subspace, normalize_images, transpose_images
                )
            else:
                warnings.warn(f"Key '{key}' not found in observation dict")

        return cast(NumpyObsType, preprocessed_obs)

    elif isinstance(observation_space, gym.spaces.Box):
        processed_obs = np.array(obs, copy=True)

        # Handle image preprocessing
        if is_image_space(observation_space):
            # Transpose if needed
            if transpose_images:
                processed_obs = maybe_transpose(processed_obs, observation_space)

            # Normalize if needed
            if normalize_images:
                processed_obs = cast(np.ndarray, normalize_image(processed_obs))

        return processed_obs

    else:
        # For other spaces (Discrete, MultiBinary, etc.), return as is
        return obs


def get_obs_shape(
    observation_space: GymSpace,
) -> Union[Tuple[int, ...], Dict[str, Any]]:
    """
    Get the shape of observations after preprocessing.

    This is useful for determining input dimensions for neural networks.

    Args:
        observation_space: The observation space

    Returns:
        Shape tuple for Box spaces, or dict of shapes for Dict spaces
    """
    if isinstance(observation_space, gym.spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for key, subspace in observation_space.spaces.items()
        }

    elif isinstance(observation_space, gym.spaces.Box):
        shape = observation_space.shape

        # If it's an image space, account for potential transposing
        if is_image_space(observation_space) and len(shape) == 3:
            if not is_image_space_channels_first(observation_space):
                # Will be transposed from (H, W, C) to (C, H, W)
                return (shape[2], shape[0], shape[1])

        return shape

    elif isinstance(observation_space, gym.spaces.Discrete):
        return (1,)  # Single discrete value

    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return observation_space.shape

    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return observation_space.shape

    else:
        # Fallback for other space types
        shape_attr = getattr(observation_space, "shape", None)
        if shape_attr is None:
            raise ValueError(
                f"Unsupported observation space type: {type(observation_space)}"
            )
        if isinstance(shape_attr, tuple):
            return shape_attr
        if isinstance(shape_attr, list):
            return tuple(int(dim) for dim in shape_attr)
        if isinstance(shape_attr, int):
            return (shape_attr,)
        raise ValueError(
            f"Unsupported shape attribute {shape_attr!r} for {type(observation_space)}"
        )


def check_for_nested_spaces(observation_space: GymSpace) -> bool:
    """
    Check if observation space contains nested spaces (like Tuple or nested Dict).

    Args:
        observation_space: The observation space to check

    Returns:
        True if the space contains nested structures
    """
    if isinstance(observation_space, gym.spaces.Dict):
        return any(
            isinstance(subspace, (gym.spaces.Dict, gym.spaces.Tuple))
            for subspace in observation_space.spaces.values()
        )

    elif isinstance(observation_space, gym.spaces.Tuple):
        return True

    return False


def flatten_obs(
    obs: Union[np.ndarray, Dict[str, Any]], observation_space: GymSpace
) -> np.ndarray:
    """
    Flatten observations for algorithms that require flat input.

    Args:
        obs: Observation to flatten
        observation_space: The observation space

    Returns:
        Flattened observation as 1D array
    """
    if isinstance(observation_space, gym.spaces.Dict):
        if not isinstance(obs, dict):
            raise ValueError("Expected dict observation for Dict observation space")

        # Flatten each sub-observation and concatenate
        flattened_parts: List[np.ndarray] = []
        for key in sorted(observation_space.spaces.keys()):  # Sort for consistency
            if key in obs:
                sub_obs = obs[key]
                subspace = observation_space.spaces[key]
                if isinstance(sub_obs, dict):
                    flattened_parts.append(flatten_obs(sub_obs, subspace).flatten())
                elif isinstance(sub_obs, np.ndarray):
                    flattened_parts.append(sub_obs.flatten())
                else:
                    flattened_parts.append(np.array([sub_obs]).flatten())

        return np.concatenate(flattened_parts)

    elif isinstance(observation_space, gym.spaces.Box):
        return np.asarray(obs).flatten()

    else:
        # For discrete and other spaces
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        else:
            return np.array([obs]).flatten()


def get_flattened_obs_dim(observation_space: GymSpace) -> int:
    """
    Get the dimension of flattened observations.

    Args:
        observation_space: The observation space

    Returns:
        Total dimension after flattening
    """
    if isinstance(observation_space, gym.spaces.Dict):
        total_dim = 0
        for subspace in observation_space.spaces.values():
            total_dim += get_flattened_obs_dim(subspace)
        return total_dim

    elif isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape))

    elif isinstance(observation_space, gym.spaces.Discrete):
        return 1

    elif isinstance(
        observation_space, (gym.spaces.MultiBinary, gym.spaces.MultiDiscrete)
    ):
        return int(np.prod(np.asarray(observation_space.shape)))

    else:
        # Fallback
        shape_attr = getattr(observation_space, "shape", None)
        if shape_attr is None:
            return 1
        return int(np.prod(np.asarray(shape_attr)))


def convert_to_mlx(obs: NumpyObsType) -> ObsType:
    """
    Convert preprocessed observations to MLX arrays.

    Args:
        obs: Preprocessed observation(s) as numpy arrays

    Returns:
        Observation(s) as MLX arrays
    """
    if isinstance(obs, dict):
        converted: Dict[str, ObsType] = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                converted[key] = convert_to_mlx(value)
            else:
                converted[key] = mx.array(value)
        return cast(ObsType, converted)
    return mx.array(obs)
