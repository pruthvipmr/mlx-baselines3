"""
Observation and reward normalization wrapper for vectorized environments.

This module provides VecNormalize, which normalizes observations and rewards
using running statistics to improve training stability.
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from mlx_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from mlx_baselines3.common.type_aliases import ObsType


class RunningMeanStd:
    """
    Tracks the mean, std and count of values using Welford's algorithm.

    This implementation is numerically stable and memory efficient.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Initialize running statistics tracker.

        Args:
            epsilon: Small value to avoid division by zero
            shape: Shape of the data being tracked
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new data.

        Args:
            x: New data to incorporate into statistics
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """
        Update running statistics from batch moments.

        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        """Get the current standard deviation."""
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using current statistics.

        Args:
            x: Data to normalize

        Returns:
            Normalized data
        """
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize data using current statistics.

        Args:
            x: Normalized data to denormalize

        Returns:
            Denormalized data
        """
        return x * self.std + self.mean

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "epsilon": self.epsilon,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from unpickling."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes observations and returns.

    This wrapper maintains running statistics of observations and returns to
    normalize them, which can significantly improve training stability and
    performance.

    Args:
        venv: Vectorized environment to wrap
        training: Whether to update normalization statistics during
            environment interactions
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        clip_obs: Maximum absolute value for normalized observations
        clip_reward: Maximum absolute value for normalized rewards
        gamma: Discount factor for reward normalization
        epsilon: Small value to avoid division by zero
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__(venv)

        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize observation normalization
        if self.norm_obs:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.obs_rms = {
                    key: RunningMeanStd(shape=space.shape, epsilon=epsilon)
                    for key, space in self.observation_space.spaces.items()
                }
            else:
                self.obs_rms = RunningMeanStd(
                    shape=self.observation_space.shape, epsilon=epsilon
                )
        else:
            self.obs_rms = None

        # Initialize reward normalization
        if self.norm_reward:
            self.ret_rms = RunningMeanStd(shape=(), epsilon=epsilon)
            self.returns = np.zeros(self.num_envs)
        else:
            self.ret_rms = None
            self.returns = None

    def reset(self) -> ObsType:
        """
        Reset all environments and return normalized observations.

        Returns:
            Normalized initial observations
        """
        obs = self.venv.reset()

        if self.norm_reward:
            self.returns = np.zeros(self.num_envs)

        return self._normalize_obs(obs)

    def step_wait(self) -> Tuple[ObsType, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Wait for step to complete and return normalized results.

        Returns:
            Tuple of (normalized_obs, normalized_rewards, dones, infos)
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Update and normalize rewards
        if self.norm_reward:
            # Update return estimation
            self.returns = self.returns * self.gamma + rewards

            # Update reward normalization statistics during training
            if self.training:
                self.ret_rms.update(self.returns)

            # Normalize rewards
            rewards = self._normalize_reward(rewards)

            # Reset returns for done environments
            self.returns[dones] = 0.0

        # Normalize observations
        obs = self._normalize_obs(obs)

        return obs, rewards, dones, infos

    def _normalize_obs(self, obs: ObsType) -> ObsType:
        """
        Normalize observations using running statistics.

        Args:
            obs: Raw observations

        Returns:
            Normalized observations
        """
        if not self.norm_obs:
            return obs

        if isinstance(obs, dict):
            # Handle dictionary observations
            normalized_obs = {}
            for key, values in obs.items():
                if self.training:
                    self.obs_rms[key].update(values)
                normalized_obs[key] = np.clip(
                    self.obs_rms[key].normalize(values), -self.clip_obs, self.clip_obs
                )
            return normalized_obs
        else:
            # Handle array observations
            if self.training:
                self.obs_rms.update(obs)
            return np.clip(self.obs_rms.normalize(obs), -self.clip_obs, self.clip_obs)

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using running statistics.

        Args:
            rewards: Raw rewards

        Returns:
            Normalized rewards
        """
        if not self.norm_reward:
            return rewards

        return np.clip(rewards / self.ret_rms.std, -self.clip_reward, self.clip_reward)

    def get_original_obs(self) -> ObsType:
        """
        Get the original (unnormalized) observations from the wrapped
        environment.

        Returns:
            Original observations
        """
        return self.venv.get_attr("_observations")[0]

    def get_original_reward(self) -> np.ndarray:
        """
        Get the original (unnormalized) rewards from the wrapped environment.

        Returns:
            Original rewards
        """
        return self.venv.get_attr("_rewards")[0]

    def set_training(self, training: bool) -> None:
        """
        Set training mode. When training=False, normalization statistics are
        not updated.

        Args:
            training: Whether to update statistics during environment
                interactions
        """
        self.training = training

    def save(self, path: str) -> None:
        """
        Save normalization parameters to file.

        Args:
            path: Path where to save the parameters
        """
        state = {
            "obs_rms": self.obs_rms,
            "ret_rms": self.ret_rms,
            "training": self.training,
            "norm_obs": self.norm_obs,
            "norm_reward": self.norm_reward,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "returns": self.returns,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """
        Load normalization parameters from file.

        Args:
            path: Path to the saved parameters
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Restore state
        self.obs_rms = state["obs_rms"]
        self.ret_rms = state["ret_rms"]
        self.training = state["training"]
        self.norm_obs = state["norm_obs"]
        self.norm_reward = state["norm_reward"]
        self.clip_obs = state["clip_obs"]
        self.clip_reward = state["clip_reward"]
        self.gamma = state["gamma"]
        self.epsilon = state["epsilon"]
        self.returns = state.get("returns", np.zeros(self.num_envs))

    def load(self, path: str, venv: Optional[VecEnv] = None) -> "VecNormalize":
        """
        Load VecNormalize parameters.

        When called on an instance, loads normalization statistics into the
        existing wrapper. When called on the class, a ``VecEnv`` must be
        provided and a new ``VecNormalize`` instance is returned, matching the
        Stable Baselines 3 API.

        Args:
            path: Path to the saved parameters
            venv: Environment to wrap when called on the class

        Returns:
            The VecNormalize wrapper with loaded parameters
        """

        if isinstance(self, type):
            cls = self
            if venv is None:
                raise TypeError("VecNormalize.load() missing required argument: 'venv'")

            with open(path, "rb") as f:
                state = pickle.load(f)

            vec_normalize = cls(venv)
            vec_normalize.obs_rms = state["obs_rms"]
            vec_normalize.ret_rms = state["ret_rms"]
            vec_normalize.training = state["training"]
            vec_normalize.norm_obs = state["norm_obs"]
            vec_normalize.norm_reward = state["norm_reward"]
            vec_normalize.clip_obs = state["clip_obs"]
            vec_normalize.clip_reward = state["clip_reward"]
            vec_normalize.gamma = state["gamma"]
            vec_normalize.epsilon = state["epsilon"]
            vec_normalize.returns = state.get(
                "returns", np.zeros(vec_normalize.num_envs)
            )

            return vec_normalize

        # Instance method: load state into existing wrapper
        self.load_state(path)
        return self

    def get_attr(
        self, attr_name: str, indices: Optional[List[int]] = None
    ) -> List[Any]:
        """
        Get attribute from environments, handling normalization-specific attributes.

        Args:
            attr_name: Name of the attribute to get
            indices: Indices of environments to get attribute from

        Returns:
            List of attribute values
        """
        if attr_name in ["obs_rms", "ret_rms", "training", "norm_obs", "norm_reward"]:
            # Return normalization-specific attributes
            return [getattr(self, attr_name)] * (
                len(indices) if indices else self.num_envs
            )
        else:
            return self.venv.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: Optional[List[int]] = None
    ) -> None:
        """
        Set attribute in environments, handling normalization-specific attributes.

        Args:
            attr_name: Name of the attribute to set
            value: Value to set
            indices: Indices of environments to set attribute in
        """
        if attr_name in [
            "training",
            "norm_obs",
            "norm_reward",
            "clip_obs",
            "clip_reward",
        ]:
            # Set normalization-specific attributes
            setattr(self, attr_name, value)
        else:
            self.venv.set_attr(attr_name, value, indices)

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        state = self.__dict__.copy()
        # Remove the unwrapped env reference which may not be serializable
        if hasattr(self, "venv"):
            # Don't directly pickle the wrapped environment
            state["venv"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state from unpickling."""
        self.__dict__.update(state)
        # Note: venv needs to be restored separately when loading
