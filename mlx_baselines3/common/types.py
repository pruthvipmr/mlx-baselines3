"""
Unified array types for MLX Baselines3.

This module provides consistent type aliases for arrays that work with both
NumPy and MLX arrays, avoiding the type mismatches that were causing many
mypy errors throughout the codebase.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Protocol,
    TypeGuard,
    TypedDict,
    Union,
    runtime_checkable,
)

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - fallback for older Python
    from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    MlxArray: TypeAlias = NDArray[Any]
    _MLX_ARRAY_TYPE: type[Any] | None = None
else:
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - runtime fallback
        mx = None

    if mx is not None:
        _MLX_ARRAY_TYPE = type(mx.array([0]))
        MlxArray = _MLX_ARRAY_TYPE
    else:
        _MLX_ARRAY_TYPE = None

        class MlxArray:
            def __init__(self) -> None:
                self._shape: tuple[int, ...] = ()
                self._dtype: Any = None

            @property
            def shape(self) -> tuple[int, ...]:
                return self._shape

            @property
            def dtype(self) -> Any:
                return self._dtype

            def astype(self, dtype: Any) -> "MlxArray":  # pragma: no cover - trivial
                self._dtype = dtype
                return self


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for array-like objects supporting basic array operations."""

    shape: tuple[int, ...]
    dtype: Any

    def astype(self, dtype: Any) -> ArrayProtocol: ...


# Type aliases for different use cases
ArrayLike: TypeAlias = Union[NDArray[Any], MlxArray]
"""Array-like type that can be either NumPy or MLX array."""

NumpyArray = NDArray[np.generic]
"""Specifically a NumPy array."""

# Single observation can be an array or a dict of arrays
Obs = Union[ArrayLike, Mapping[str, ArrayLike]]
Action = ArrayLike  # keep as array-like only

# Batched observation has the same structural shape but with leading batch dim
BatchObs = Union[ArrayLike, Mapping[str, ArrayLike]]
BatchAction = ArrayLike

# For backward compatibility
ObsType = Obs
"""Type for observations."""

ActionType = Action
"""Type for actions."""

# For policy outputs
PolicyOutputType = tuple[ArrayLike, ArrayLike | None]
"""Type for policy outputs: (actions, log_probs or states)."""

# For value function outputs
ValueType = ArrayLike
"""Type for value function outputs."""

# Common shape types
Shape = tuple[int, ...]
"""Type for array shapes."""


# Episode info TypedDict
class EpisodeInfo(TypedDict, total=False):
    """Episode information dictionary."""

    r: float  # reward
    l: float  # length  # noqa: E741
    t: float  # time


# Type guards for narrowing Union types
def is_numpy_array(x: Any) -> TypeGuard[NDArray[Any]]:
    """Check if x is a NumPy array."""
    return isinstance(x, np.ndarray)


def is_mlx_array(x: Any) -> TypeGuard[MlxArray]:
    """Check if x is an MLX array."""
    return _MLX_ARRAY_TYPE is not None and isinstance(x, _MLX_ARRAY_TYPE)


def is_dict_obs(x: Any) -> TypeGuard[Mapping[str, ArrayLike]]:
    """Check if x is a dictionary observation."""
    return isinstance(x, Mapping) and not isinstance(x, (str, bytes))


# Policy Protocol
@runtime_checkable
class PolicyProtocol(Protocol):
    """Protocol for policy objects."""

    def predict(
        self,
        observation: BatchObs,
        state: ArrayLike | None,
        episode_start: np.ndarray | None,
        deterministic: bool,
    ) -> tuple[ArrayLike, ArrayLike | None]: ...


# VecEnv Protocol
@runtime_checkable
class VecEnvProtocol(Protocol):
    """Protocol for vectorized environments."""

    num_envs: int

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict]: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[Obs, np.ndarray, np.ndarray, np.ndarray, list[dict]]: ...


# ReplayBuffer Protocol
@runtime_checkable
class ReplayBufferProtocol(Protocol):
    """Protocol for replay buffers."""

    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict],
    ) -> None: ...

    def sample(self, batch_size: int) -> dict[str, np.ndarray]: ...
