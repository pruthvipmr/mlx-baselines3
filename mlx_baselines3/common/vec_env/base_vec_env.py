"""
Base class for vectorized environments.

This module provides the abstract base class for all vectorized environments
in MLX Baselines3, maintaining API compatibility with Stable Baselines3.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import mlx.core as mx

from mlx_baselines3.common.type_aliases import ObsType, MlxArray


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    
    This is the base class for all vectorized environments. It provides a
    common interface for running multiple environments in parallel.
    """
    
    def __init__(
        self,
        num_envs: int,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        """
        Initialize the vectorized environment.
        
        Args:
            num_envs: Number of environments to run in parallel
            observation_space: Observation space of a single environment
            action_space: Action space of a single environment
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        
    @abstractmethod
    def reset(self) -> ObsType:
        """
        Reset all environments and return initial observations.
        
        Returns:
            Initial observations from all environments
        """
        
    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step.
        
        Args:
            actions: Actions to take in each environment
        """
        
    @abstractmethod
    def step_wait(self) -> Tuple[ObsType, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Wait for the step taken with step_async().
        
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        
    def step(self, actions: np.ndarray) -> Tuple[ObsType, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with the given actions.
        
        Args:
            actions: Actions to take in each environment
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        self.step_async(actions)
        return self.step_wait()
        
    @abstractmethod
    def close(self) -> None:
        """
        Close all environments.
        """
        
    @abstractmethod
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """
        Get attribute from each environment.
        
        Args:
            attr_name: Name of the attribute to get
            indices: Indices of environments to get attribute from.
                    If None, get from all environments.
                    
        Returns:
            List of attribute values from each environment
        """
        
    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        """
        Set attribute in each environment.
        
        Args:
            attr_name: Name of the attribute to set
            value: Value to set the attribute to
            indices: Indices of environments to set attribute in.
                    If None, set in all environments.
        """
        
    @abstractmethod
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs
    ) -> List[Any]:
        """
        Call method on each environment.
        
        Args:
            method_name: Name of the method to call
            *method_args: Arguments to pass to the method
            indices: Indices of environments to call method on.
                    If None, call on all environments.
            **method_kwargs: Keyword arguments to pass to the method
            
        Returns:
            List of return values from each environment
        """
        
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Set the random seeds for all environments.
        
        Args:
            seed: Random seed. If None, no seed is set.
            
        Returns:
            List of seeds used for each environment
        """
        if seed is None:
            seeds = [None] * self.num_envs
        else:
            seeds = [seed + i for i in range(self.num_envs)]
            
        return [self.env_method("seed", seed=env_seed, indices=[i])[0] 
                for i, env_seed in enumerate(seeds)]
    
    def render(self, mode: str = "human", **kwargs) -> Optional[np.ndarray]:
        """
        Render the environments.
        
        Args:
            mode: Rendering mode
            **kwargs: Additional arguments for rendering
            
        Returns:
            Rendered images if mode is 'rgb_array', otherwise None
        """
        if mode == "human":
            # Render first environment only for human mode
            self.env_method("render", mode=mode, indices=[0], **kwargs)
            return None
        elif mode == "rgb_array":
            # Return all rendered images
            images = self.env_method("render", mode=mode, **kwargs)
            return np.array(images)
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")
    
    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        Get rendered images from all environments.
        
        Returns:
            List of rendered images from each environment
        """
        return self.env_method("render", mode="rgb_array")
    
    @property
    def unwrapped(self) -> List[gym.Env]:
        """
        Get the unwrapped environments.
        
        Returns:
            List of unwrapped gym environments
        """
        return self.get_attr("unwrapped")
    
    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """
        Check if an attribute is found at a given depth in the environment stack.
        
        Args:
            name: Name of the attribute to check
            already_found: Whether the attribute was already found at a shallower depth
            
        Returns:
            Error message if there's an issue, otherwise None
        """
        if hasattr(self, name) and already_found:
            return (
                f"Found attribute {name} in both the wrapper and the wrapped environment. "
                f"Using the wrapper's version."
            )
        return None
        
    def _get_indices(self, indices: Optional[List[int]]) -> List[int]:
        """
        Get environment indices, defaulting to all environments if None.
        
        Args:
            indices: Environment indices or None for all
            
        Returns:
            List of environment indices
        """
        if indices is None:
            return list(range(self.num_envs))
        return indices


class VecEnvWrapper(VecEnv):
    """
    Base class for vectorized environment wrappers.
    
    Provides a base implementation that delegates all calls to the wrapped environment.
    """
    
    def __init__(self, venv: VecEnv):
        """
        Initialize the wrapper.
        
        Args:
            venv: Vectorized environment to wrap
        """
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)
        
    def step_wait(self) -> Tuple[ObsType, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        return self.venv.step_wait()
        
    def reset(self) -> ObsType:
        return self.venv.reset()
        
    def close(self) -> None:
        return self.venv.close()
        
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)
        
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)
        
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs
    ) -> List[Any]:
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
        
    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        return self.venv.getattr_depth_check(name, already_found)
        
    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from the wrapped environment if not found in wrapper.
        
        Args:
            name: Name of the attribute
            
        Returns:
            Attribute value
        """
        blocked_class = VecEnv
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited")
        if name in self.__dict__:
            return getattr(self, name)
        elif hasattr(blocked_class, name):
            raise AttributeError(
                f"Accessing '{name}' from {blocked_class} is prohibited. "
                f"Use the {name} property directly: vec_env.{name}"
            )
        else:
            return getattr(self.venv, name)
