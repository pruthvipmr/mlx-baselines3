"""
DQN-specific policy implementations using MLX.

This module provides policy classes tailored for the DQN algorithm,
including Q-networks for value-based control.
"""

from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_baselines3.common.torch_layers import MlxModule
from mlx_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from mlx_baselines3.common.type_aliases import Schedule
from mlx_baselines3.common.utils import obs_as_mlx


class QNetwork(MlxModule):
    """
    Q-network that maps observations to Q-values for each action.

    Args:
        observation_space: Observation space
        action_space: Action space
        net_arch: Architecture of the Q-network
        features_extractor: Features extractor to use
        features_dim: Number of features extracted by the features extractor
        activation_fn: Activation function
        normalize_images: Whether to normalize images or not
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int] = [64, 64],
        features_extractor: nn.Module = None,
        features_dim: int = 0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__()

        assert isinstance(action_space, gym.spaces.Discrete), (
            "DQN only supports Discrete action spaces"
        )

        action_dim = action_space.n

        # Build Q-network
        self.q_net = create_mlp(features_dim, action_dim, net_arch, activation_fn)

        # Register the Q-network as a submodule
        self.add_module("q_net", self.q_net)

    def __call__(self, features: mx.array) -> mx.array:
        """
        Forward pass through the Q-network.

        Args:
            features: Features extracted from observations

        Returns:
            Q-values for each action
        """
        return self.q_net(features)


class DQNPolicy(MlxModule):
    """
    Policy class for DQN algorithm using MLX.

    This class implements a Q-network that maps observations to Q-values
    for discrete action spaces.

    Args:
        observation_space: Observation space
        action_space: Action space (must be Discrete)
        lr_schedule: Learning rate schedule
        net_arch: The specification of the Q-network
        activation_fn: Activation function
        features_extractor_class: Features extractor to use
        features_extractor_kwargs: Keyword arguments for features extractor
        normalize_images: Whether to normalize images or not
        optimizer_class: The optimizer to use
        optimizer_kwargs: Additional keyword arguments for the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class=None,  # Will use optimizer adapters
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if net_arch is None:
            net_arch = [64, 64]

        super().__init__()

        assert isinstance(action_space, gym.spaces.Discrete), (
            "DQN only supports Discrete action spaces"
        )

        # Store spaces and configuration
        self.observation_space = observation_space
        self.action_space = action_space

        # Set up features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        self.features_extractor_kwargs = features_extractor_kwargs

        # Build features extractor
        self.features_extractor = features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.lr_schedule = lr_schedule
        self.normalize_images = normalize_images

        # Build Q-network
        self.q_net = QNetwork(
            self.observation_space,
            self.action_space,
            net_arch=self.net_arch,
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
        )

        # Register modules for state_dict
        self.add_module("features_extractor", self.features_extractor)
        self.add_module("q_net", self.q_net)

    def __call__(self, obs: mx.array, deterministic: bool = True) -> mx.array:
        """
        Forward pass through the policy.

        Args:
            obs: Observation
            deterministic: Whether to use deterministic action selection

        Returns:
            Q-values for each action
        """
        features = self.extract_features(obs)
        return self.q_net(features)

    def extract_features(self, obs: mx.array) -> mx.array:
        """
        Extract features from observations.

        Args:
            obs: Observations

        Returns:
            Extracted features
        """
        return self.features_extractor(obs)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Any] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Optional[Any]]:
        """
        Get the policy action from an observation.

        Args:
            observation: Observation
            state: Policy state (unused for DQN)
            episode_start: Episode start flags (unused for DQN)
            deterministic: Whether to use deterministic action selection

        Returns:
            Action and new state (None for DQN)
        """
        observation = obs_as_mlx(observation)

        # Get Q-values
        q_values = self(observation, deterministic=deterministic)

        # Select action with highest Q-value
        action = mx.argmax(q_values, axis=-1)

        # Convert back to numpy and ensure correct shape
        action = np.array(action)

        # For single observations, return scalar action
        if action.shape == (1,):
            action = action[0]

        return action, None

    def predict_values(self, obs: mx.array) -> mx.array:
        """
        Get the Q-values for the given observations.

        Args:
            obs: Observations

        Returns:
            Q-values for each action
        """
        return self(obs)


class MlpPolicy(DQNPolicy):
    """
    Policy class for DQN using Multi-Layer Perceptron networks.
    This policy is suitable for vector observation spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=FlattenExtractor)


class CnnPolicy(DQNPolicy):
    """
    Policy class for DQN using Convolutional Neural Networks.
    This policy is suitable for image observation spaces.
    """

    def __init__(self, *args, **kwargs):
        # TODO: Implement CNN features extractor
        # For now, fallback to MLP (will be enhanced when a CNN extractor is
        # implemented)
        super().__init__(*args, **kwargs, features_extractor_class=FlattenExtractor)


class MultiInputPolicy(DQNPolicy):
    """
    Policy class for DQN using Multi-input networks.
    This policy is suitable for dictionary observation spaces.
    """

    def __init__(self, *args, **kwargs):
        # TODO: Implement MultiInput features extractor
        # For now, fallback to MLP (will be enhanced when a MultiInput
        # extractor is implemented)
        super().__init__(*args, **kwargs, features_extractor_class=FlattenExtractor)
