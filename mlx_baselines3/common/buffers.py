"""
Experience buffers for storing and sampling training data.

This module implements RolloutBuffer for on-policy algorithms (PPO, A2C) and
ReplayBuffer for off-policy algorithms (SAC, TD3, DQN) with MLX integration.
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import gymnasium as gym
import numpy as np
import mlx.core as mx

from mlx_baselines3.common.type_aliases import (
    GymSpace,
    MlxArray,
    NumpyActionType,
    NumpyObsType,
    ReplayBatch,
    RolloutBatch,
    TensorOrDict,
)


def _space_shape(space: GymSpace) -> Tuple[int, ...]:
    shape = getattr(space, "shape", None)
    if shape is None:
        return ()
    if isinstance(shape, tuple):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, list):
        return tuple(int(dim) for dim in shape)
    if isinstance(shape, int):
        return (int(shape),)
    return ()


class BaseBuffer:
    """
    Base class for all experience buffers.

    Provides common functionality for observation/action space handling
    and MLX tensor conversion.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: GymSpace,
        action_space: GymSpace,
        device: str = "cpu",
        n_envs: int = 1,
    ):
        """
        Initialize the base buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            observation_space: Observation space of the environment
            action_space: Action space of the environment
            device: Device to use for tensor operations ("cpu" or "gpu")
            n_envs: Number of parallel environments
        """
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs

        self.pos = 0
        self.full = False

        self.observations: Union[np.ndarray, Dict[str, np.ndarray]]
        self.actions: np.ndarray

        # Initialize storage arrays
        self._setup_storage()

    def _setup_storage(self) -> None:
        """Setup storage arrays based on observation and action spaces."""
        # Handle observation space
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observations = {}
            for key, subspace in self.observation_space.spaces.items():
                obs_shape = (self.buffer_size, self.n_envs) + _space_shape(subspace)
                self.observations[key] = np.zeros(obs_shape, dtype=subspace.dtype)
        else:
            obs_shape = (self.buffer_size, self.n_envs) + _space_shape(
                self.observation_space
            )
            dtype = getattr(self.observation_space, "dtype", np.float32)
            self.observations = np.zeros(obs_shape, dtype=dtype)

        # Handle action space
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (self.buffer_size, self.n_envs)
            self.actions = np.zeros(action_shape, dtype=np.int64)
        else:
            action_shape = (self.buffer_size, self.n_envs) + _space_shape(
                self.action_space
            )
            dtype = getattr(self.action_space, "dtype", np.float32)
            self.actions = np.zeros(action_shape, dtype=dtype)

    def size(self) -> int:
        """Get the current size of the buffer."""
        if self.full:
            return self.buffer_size
        return self.pos

    def reset(self) -> None:
        """Reset the buffer to empty state."""
        self.pos = 0
        self.full = False

    def _get_samples(self, batch_inds: np.ndarray) -> Dict[str, TensorOrDict]:
        """
        Get samples at specified indices and convert to MLX arrays.

        Args:
            batch_inds: Indices to sample

        Returns:
            Dictionary of MLX arrays
        """
        # Convert observations to MLX
        if isinstance(self.observations, dict):
            obs_batch: Dict[str, MlxArray] = {
                key: mx.array(obs[batch_inds]) for key, obs in self.observations.items()
            }
        else:
            obs_batch = mx.array(self.observations[batch_inds])

        # Convert actions to MLX
        actions_batch = mx.array(self.actions[batch_inds])

        samples: Dict[str, TensorOrDict] = {
            "observations": obs_batch,
            "actions": actions_batch,
        }
        return samples


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer for on-policy algorithms (PPO, A2C).

    Stores complete rollouts and computes advantages using GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: GymSpace,
        action_space: GymSpace,
        device: str = "cpu",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        normalize_advantage: bool = True,
    ):
        """
        Initialize the rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            observation_space: Observation space
            action_space: Action space
            device: Device for tensor operations
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            n_envs: Number of parallel environments
            normalize_advantage: Whether to normalize advantages
        """
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        # Additional storage for on-policy data
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Will be computed when buffer is full
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: NumpyObsType,
        action: NumpyActionType,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """
        Add one step of data to the buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            episode_start: Whether this step starts a new episode
            value: Value estimate from critic
            log_prob: Log probability of the action
        """
        if self.pos >= self.buffer_size:
            raise ValueError("RolloutBuffer is full")

        # Store observations
        if isinstance(obs, dict):
            obs_storage = cast(Dict[str, np.ndarray], self.observations)
            for key, obs_val in obs.items():
                obs_storage[key][self.pos] = np.asarray(obs_val).copy()
        else:
            obs_storage = cast(np.ndarray, self.observations)
            obs_storage[self.pos] = np.asarray(obs).copy()

        self.actions[self.pos] = np.asarray(action).copy()
        self.rewards[self.pos] = np.asarray(reward).copy()
        self.episode_starts[self.pos] = np.asarray(episode_start).copy()
        self.values[self.pos] = np.asarray(value).copy()
        self.log_probs[self.pos] = np.asarray(log_prob).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self, last_values: np.ndarray, dones: np.ndarray
    ) -> None:
        """
        Compute the returns and advantages using GAE.

        Args:
            last_values: Value estimates for the last observations
            dones: Done flags for the last observations
        """
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

            self.advantages[step] = last_gae_lam

        # Returns are advantages + values (computed before normalization)
        self.returns = self.advantages + self.values

        # Optionally normalize advantages across all environments
        # Note: This modifies advantages for training but keeps returns unchanged
        if self.normalize_advantage:
            advantages_flat = self.advantages.flatten()
            advantages_mean = np.mean(advantages_flat)
            advantages_std = np.std(advantages_flat)
            if advantages_std > 1e-8:  # Avoid division by zero
                self.advantages = (self.advantages - advantages_mean) / (
                    advantages_std + 1e-8
                )

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBatch, None, None]:
        """
        Get batches of data for training.

        Args:
            batch_size: Size of mini-batches. If None, return all data.

        Yields:
            Batches of training data as MLX arrays
        """
        if not self.full:
            raise ValueError("RolloutBuffer must be full before sampling")

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Flatten arrays for sampling
        flat_obs = self._flatten_obs()
        flat_actions = self.actions.reshape(-1, *self.actions.shape[2:])
        flat_values = self.values.flatten()
        flat_log_probs = self.log_probs.flatten()
        flat_advantages = self.advantages.flatten()
        flat_returns = self.returns.flatten()

        if batch_size is None:
            batch_size = len(indices)

        start_idx = 0
        while start_idx < len(indices):
            batch_inds = indices[start_idx : start_idx + batch_size]

            # Get observations
            if isinstance(flat_obs, dict):
                obs_batch: TensorOrDict = {
                    key: mx.array(obs[batch_inds]) for key, obs in flat_obs.items()
                }
            else:
                obs_batch = mx.array(flat_obs[batch_inds])

            yield {
                "observations": obs_batch,
                "actions": mx.array(flat_actions[batch_inds]),
                "values": mx.array(flat_values[batch_inds]),
                "log_probs": mx.array(flat_log_probs[batch_inds]),
                "advantages": mx.array(flat_advantages[batch_inds]),
                "returns": mx.array(flat_returns[batch_inds]),
            }

            start_idx += batch_size

    def _flatten_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Flatten observations for sampling."""
        if isinstance(self.observations, dict):
            obs_dict = cast(Dict[str, np.ndarray], self.observations)
            flat_obs: Dict[str, np.ndarray] = {
                key: obs.reshape(-1, *obs.shape[2:]) for key, obs in obs_dict.items()
            }
            return flat_obs
        else:
            obs_array = cast(np.ndarray, self.observations)
            return obs_array.reshape(-1, *obs_array.shape[2:])


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer for off-policy algorithms (SAC, TD3, DQN).

    Stores transitions and provides random sampling.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: GymSpace,
        action_space: GymSpace,
        device: str = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        """
        Initialize the replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            observation_space: Observation space
            action_space: Action space
            device: Device for tensor operations
            n_envs: Number of parallel environments
            optimize_memory_usage: If True, optimize memory by not storing
                next observations
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        self.optimize_memory_usage = optimize_memory_usage

        # Additional storage for off-policy data
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.terminated = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.truncated = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.has_final_obs = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)

        self.next_observations: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = (
            None
        )
        self.final_observations: Union[np.ndarray, Dict[str, np.ndarray]]

        # Store next observations unless optimizing memory
        if not self.optimize_memory_usage:
            if isinstance(self.observation_space, gym.spaces.Dict):
                next_obs: Dict[str, np.ndarray] = {}
                for key, subspace in self.observation_space.spaces.items():
                    obs_shape = (self.buffer_size, self.n_envs) + _space_shape(subspace)
                    next_obs[key] = np.zeros(obs_shape, dtype=subspace.dtype)
                self.next_observations = next_obs
            else:
                obs_shape = (
                    self.buffer_size,
                    self.n_envs,
                ) + _space_shape(self.observation_space)
                dtype = getattr(self.observation_space, "dtype", np.float32)
                self.next_observations = np.zeros(obs_shape, dtype=dtype)

        if isinstance(self.observation_space, gym.spaces.Dict):
            final_obs: Dict[str, np.ndarray] = {}
            for key, subspace in self.observation_space.spaces.items():
                obs_shape = (self.buffer_size, self.n_envs) + _space_shape(subspace)
                final_obs[key] = np.zeros(obs_shape, dtype=subspace.dtype)
            self.final_observations = final_obs
        else:
            obs_shape = (self.buffer_size, self.n_envs) + _space_shape(
                self.observation_space
            )
            dtype = getattr(self.observation_space, "dtype", np.float32)
            self.final_observations = np.zeros(obs_shape, dtype=dtype)

    def add(
        self,
        obs: NumpyObsType,
        next_obs: NumpyObsType,
        action: NumpyActionType,
        reward: np.ndarray,
        terminated: Union[np.ndarray, np.bool_, bool],
        truncated: Optional[Union[np.ndarray, np.bool_, bool]] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add one step of data to the buffer.

        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received
            terminated: Episode termination flags
            truncated: Optional truncation flags (e.g. TimeLimit)
            infos: Additional information (may contain TimeLimit.truncated,
                final_observation, and _timeout)
        """
        if infos is None and isinstance(truncated, list):
            if all(isinstance(item, dict) for item in truncated):
                infos = truncated
                truncated = None

        if infos is None:
            infos = [{} for _ in range(self.n_envs)]

        terminated_array = np.asarray(terminated, dtype=np.bool_).reshape(self.n_envs)
        if truncated is None:
            truncated_array = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos],
                dtype=np.bool_,
            )
        else:
            truncated_array = np.asarray(truncated, dtype=np.bool_).reshape(self.n_envs)

        # Handle dictionary observations
        if isinstance(obs, dict):
            obs_storage = cast(Dict[str, np.ndarray], self.observations)
            for key, obs_val in obs.items():
                obs_storage[key][self.pos] = np.asarray(obs_val).copy()
        else:
            obs_storage = cast(np.ndarray, self.observations)
            obs_storage[self.pos] = np.asarray(obs).copy()

        self.actions[self.pos] = np.asarray(action).copy()
        self.rewards[self.pos] = np.asarray(reward).copy()

        final_observations: List[Optional[NumpyObsType]] = [None] * self.n_envs
        timeouts = np.zeros(self.n_envs, dtype=np.bool_)

        for i, info in enumerate(infos):
            if info is None:
                continue
            if "TimeLimit.truncated" in info:
                truncated_array[i] = truncated_array[i] or bool(
                    info["TimeLimit.truncated"]
                )
            if "_timeout" in info:
                timeouts[i] = bool(info["_timeout"])
            final_obs = info.get("final_observation")
            if final_obs is None:
                final_obs = info.get("terminal_observation")
            if isinstance(final_obs, tuple):
                final_obs = final_obs[0]
            if final_obs is not None:
                final_observations[i] = final_obs

        self.terminated[self.pos] = terminated_array
        self.truncated[self.pos] = truncated_array
        self.timeouts[self.pos] = timeouts

        has_final_obs = np.zeros(self.n_envs, dtype=np.bool_)

        if isinstance(self.observation_space, gym.spaces.Dict):
            final_storage = cast(Dict[str, np.ndarray], self.final_observations)
            if not self.optimize_memory_usage:
                if self.next_observations is None:
                    raise RuntimeError("next_observations is unavailable")
                next_storage = cast(Dict[str, np.ndarray], self.next_observations)
                next_obs_dict = cast(Dict[str, np.ndarray], next_obs)
                for key, next_obs_val in next_obs_dict.items():
                    updated_next_obs = np.asarray(next_obs_val).copy()
                    for env_idx, final_obs in enumerate(final_observations):
                        if isinstance(final_obs, dict) and final_obs is not None:
                            updated_next_obs[env_idx] = np.asarray(final_obs[key])
                            has_final_obs[env_idx] = True
                    next_storage[key][self.pos] = updated_next_obs
            for key in final_storage:
                final_array = final_storage[key][self.pos]
                final_array[...] = 0
                for env_idx, final_obs in enumerate(final_observations):
                    if isinstance(final_obs, dict) and final_obs is not None:
                        final_array[env_idx] = np.asarray(final_obs[key])
                        has_final_obs[env_idx] = True
        else:
            next_obs_array = np.asarray(next_obs).copy()
            final_array = cast(np.ndarray, self.final_observations)[self.pos]
            final_array[...] = 0
            for env_idx, final_obs in enumerate(final_observations):
                if final_obs is not None:
                    final_array[env_idx] = np.asarray(final_obs)
                    next_obs_array[env_idx] = np.asarray(final_obs)
                    has_final_obs[env_idx] = True
            if not self.optimize_memory_usage:
                next_storage_arr = cast(np.ndarray, self.next_observations)
                next_storage_arr[self.pos] = next_obs_array

        self.has_final_obs[self.pos] = has_final_obs

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0  # Circular buffer

    def sample(self, batch_size: int, env: Optional[Any] = None) -> ReplayBatch:
        """
        Sample a batch of transitions randomly.

        Args:
            batch_size: Number of transitions to sample
            env: Environment (used for next obs when optimizing memory)

        Returns:
            Batch of transitions as MLX arrays
        """
        if self.size() == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample random indices
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        env_indices = np.random.randint(0, self.n_envs, size=batch_size)

        # Get observations
        if isinstance(self.observations, dict):
            obs_batch: TensorOrDict = {
                key: mx.array(obs[batch_inds, env_indices])
                for key, obs in self.observations.items()
            }
        else:
            obs_batch = mx.array(self.observations[batch_inds, env_indices])

        # Get next observations
        if self.optimize_memory_usage:
            # Compute next observations on the fly
            next_obs_batch = self._get_next_obs_optimized(batch_inds, env_indices, env)
        else:
            if self.next_observations is None:
                raise RuntimeError("next_observations is unavailable")
            if isinstance(self.next_observations, dict):
                next_obs_batch = {
                    key: mx.array(next_obs[batch_inds, env_indices])
                    for key, next_obs in self.next_observations.items()
                }
            else:
                next_obs_batch = mx.array(
                    self.next_observations[batch_inds, env_indices]
                )

        batch: ReplayBatch = {
            "observations": obs_batch,
            "actions": mx.array(self.actions[batch_inds, env_indices]),
            "next_observations": next_obs_batch,
            "terminated": mx.array(self.terminated[batch_inds, env_indices]),
            "rewards": mx.array(self.rewards[batch_inds, env_indices]),
            "truncated": mx.array(self.truncated[batch_inds, env_indices]),
            "timeouts": mx.array(self.timeouts[batch_inds, env_indices]),
        }
        return batch

    def _get_next_obs_optimized(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env
    ) -> Union[MlxArray, Dict[str, MlxArray]]:
        """
        Get next observations when optimizing memory usage.

        Args:
            batch_inds: Batch indices
            env_indices: Environment indices
            env: Environment to get observations from

        Returns:
            Next observations as MLX arrays
        """
        # For memory optimization, we need to handle the case where
        # next_obs is the current obs at the next timestep
        next_batch_inds = (batch_inds + 1) % self.buffer_size

        terminated = self.terminated[batch_inds, env_indices]
        truncated = self.truncated[batch_inds, env_indices]
        has_final = self.has_final_obs[batch_inds, env_indices]

        if isinstance(self.observations, dict):
            obs_dict = cast(Dict[str, np.ndarray], self.observations)
            final_dict = cast(Dict[str, np.ndarray], self.final_observations)
            next_obs_batch = {}
            for key, obs in obs_dict.items():
                next_obs = obs[next_batch_inds, env_indices].copy()
                # Zero-out terminal transitions
                next_obs[terminated] = 0
                if np.any(truncated & has_final):
                    final_obs = final_dict[key][batch_inds, env_indices]
                    mask = truncated & has_final
                    next_obs[mask] = final_obs[mask]
                next_obs_batch[key] = mx.array(next_obs)
        else:
            obs_array = cast(np.ndarray, self.observations)
            final_array = cast(np.ndarray, self.final_observations)
            next_obs = obs_array[next_batch_inds, env_indices].copy()
            next_obs[terminated] = 0
            if np.any(truncated & has_final):
                final_obs = final_array[batch_inds, env_indices]
                mask = truncated & has_final
                next_obs[mask] = final_obs[mask]
            next_obs_batch = mx.array(next_obs)

        return next_obs_batch
