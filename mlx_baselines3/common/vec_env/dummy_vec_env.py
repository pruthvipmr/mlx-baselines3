"""
Vectorized environment that runs environments sequentially in a single process.

This is a simple implementation of vectorized environments that runs multiple
environments sequentially in the same process. It's the most basic form of
vectorization and is compatible with most environments.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from mlx_baselines3.common.vec_env.base_vec_env import VecEnv
from mlx_baselines3.common.type_aliases import ObsType


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments.

    Runs environments sequentially in a single process. This is useful for
    environments that are not thread-safe or when you want to avoid the
    overhead of multiprocessing.

    Args:
        env_fns: List of functions that create environments
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        Initialize the DummyVecEnv.

        Args:
            env_fns: List of functions that create gym environments
        """
        self.envs = [fn() for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You cannot reuse the same environment instance in a "
                "vectorized environment. Each environment function should "
                "create a new environment instance."
            )

        env = self.envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

        self._actions = None
        self._observations = None
        self._rewards = None
        self._dones = None
        self._infos = None

    def step_async(self, actions: np.ndarray) -> None:
        """
        Start stepping the environments with the given actions.

        Args:
            actions: Array of actions for each environment
        """
        self._actions = actions

    def step_wait(self) -> Tuple[ObsType, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Wait for the step to complete and return results.

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            if self._dones is not None and self._dones[i]:
                # Environment was done on previous step, reset it
                obs, info = env.reset()
                observations.append(obs)
                rewards.append(0.0)
                dones.append(False)
                infos.append(info)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Auto-reset if episode is done
                if done:
                    # Save terminal observation and info
                    infos.append({"terminal_observation": obs, **info})
                    # Reset environment and get new observation
                    obs, reset_info = env.reset()
                    # Add reset info to the terminal info
                    infos[-1].update(reset_info)
                else:
                    infos.append(info)

                observations.append(obs)
                rewards.append(reward)
                dones.append(done)

        # Convert to numpy arrays
        self._observations = self._obs_from_list(observations)
        self._rewards = np.array(rewards, dtype=np.float32)
        self._dones = np.array(dones, dtype=bool)
        self._infos = infos

        return self._observations, self._rewards, self._dones, self._infos

    def reset(self) -> ObsType:
        """
        Reset all environments and return initial observations.

        Returns:
            Initial observations from all environments
        """
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        self._observations = self._obs_from_list(observations)
        self._dones = np.array([False] * self.num_envs, dtype=bool)

        return self._observations

    def close(self) -> None:
        """
        Close all environments.
        """
        for env in self.envs:
            env.close()

    def get_attr(
        self, attr_name: str, indices: Optional[List[int]] = None
    ) -> List[Any]:
        """
        Get attribute from environments.

        Args:
            attr_name: Name of the attribute to get
            indices: Indices of environments to get attribute from

        Returns:
            List of attribute values
        """
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: Optional[List[int]] = None
    ) -> None:
        """
        Set attribute in environments.

        Args:
            attr_name: Name of the attribute to set
            value: Value to set
            indices: Indices of environments to set attribute in
        """
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """
        Call method on environments.

        Args:
            method_name: Name of the method to call
            *method_args: Arguments to pass to the method
            indices: Indices of environments to call method on
            **method_kwargs: Keyword arguments to pass to the method

        Returns:
            List of return values from each environment
        """
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Set seeds for all environments.

        Args:
            seed: Random seed

        Returns:
            List of seeds used for each environment
        """
        if seed is None:
            seeds = [None] * self.num_envs
        else:
            seeds = [seed + i for i in range(self.num_envs)]

        results = []
        for i, env_seed in enumerate(seeds):
            if env_seed is not None:
                results.append(self.envs[i].reset(seed=env_seed))
            else:
                results.append(None)

        return results

    def _get_target_envs(self, indices: Optional[List[int]]) -> List[gym.Env]:
        """
        Get target environments based on indices.

        Args:
            indices: Environment indices or None for all

        Returns:
            List of target environments
        """
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def _obs_from_list(self, obs_list: List[Any]) -> ObsType:
        """
        Convert list of observations to appropriate format.

        Args:
            obs_list: List of observations from environments

        Returns:
            Observations in appropriate format (numpy array or dict)
        """
        if isinstance(self.observation_space, gym.spaces.Dict):
            # Handle dictionary observations
            obs_dict = {}
            for key in self.observation_space.spaces.keys():
                obs_dict[key] = np.array([obs[key] for obs in obs_list])
            return obs_dict
        else:
            # Handle array observations
            return np.array(obs_list)

    def render(self, mode: str = "human", **kwargs) -> Optional[np.ndarray]:
        """
        Render the environments.

        Args:
            mode: Rendering mode
            **kwargs: Additional rendering arguments

        Returns:
            Rendered images if mode is 'rgb_array', otherwise None
        """
        if mode == "human":
            # Render the first environment only
            self.envs[0].render()
            return None
        elif mode == "rgb_array":
            # Return rendered images from all environments
            images = []
            for env in self.envs:
                images.append(env.render())
            return np.array(images)
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        Get rendered images from all environments.

        Returns:
            List of rendered images
        """
        return [env.render() for env in self.envs]

    @property
    def unwrapped(self) -> List[gym.Env]:
        """
        Get the unwrapped environments.

        Returns:
            List of unwrapped gym environments
        """
        return [env.unwrapped for env in self.envs]


def make_vec_env(
    env_id: Union[str, Callable[[], gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Callable[..., VecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a vectorized environment from multiple copies of an environment.

    Args:
        env_id: Either the environment ID or a callable that creates the environment
        n_envs: Number of environments to create
        seed: Random seed for the environments
        start_index: Start index for environment IDs (for logging purposes)
        monitor_dir: Directory to save Monitor logs to (not implemented)
        wrapper_class: Optional wrapper class to apply to each environment
        env_kwargs: Optional keyword arguments to pass to the environment
        vec_env_cls: Vectorized environment class to use (defaults to DummyVecEnv)
        vec_env_kwargs: Optional keyword arguments to pass to the vectorized environment
        monitor_kwargs: Optional keyword arguments for Monitor wrapper (not implemented)

    Returns:
        Vectorized environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id()

            if seed is not None:
                env.reset(seed=seed + rank)

            if wrapper_class is not None:
                env = wrapper_class(env)

            return env

        return _init

    # Create environment functions
    env_fns = [make_env(i + start_index) for i in range(n_envs)]

    return vec_env_cls(env_fns, **vec_env_kwargs)
