"""
TD3 policies for continuous action spaces with deterministic actor and twin
critics.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_baselines3.common.policies import BasePolicy
from mlx_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from mlx_baselines3.common.type_aliases import Schedule


class TD3Policy(BasePolicy):
    """
    TD3 policy with deterministic actor and twin critics.

    TD3 (Twin Delayed Deep Deterministic Policy Gradient) uses:
    - A deterministic actor that outputs continuous actions
    - Twin critics (Q1 and Q2) to reduce overestimation bias
    - Separate target actor and critics for stable training
    - Target policy smoothing during training
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Optional[Type] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        # Set attributes before calling super().__init__() because _build() is
        # invoked there
        if net_arch is None:
            net_arch = [400, 300]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor

        # TD3 is only for continuous action spaces
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("TD3 only supports continuous action spaces (Box)")

        self.action_dim = int(action_space.shape[0])
        self.action_space = action_space

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                n_critics=self.n_critics,
                share_features_extractor=self.share_features_extractor,
                lr_schedule=self._dummy_schedule,  # Dummy schedule
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        """Build the MLP feature extractor for actor and critic."""
        if self.share_features_extractor:
            # Shared feature extractor
            self.mlp_extractor = None
            # Feature extractor is shared between actor and critic
        else:
            # Separate feature extractors for actor and critics
            self.mlp_extractor = None

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the actor and critic networks.
        """
        # Use features extractor created by BasePolicy
        features_dim = self.features_extractor.features_dim

        # Get network architecture
        if (
            isinstance(self.net_arch, list)
            and len(self.net_arch) > 0
            and isinstance(self.net_arch[0], dict)
        ):
            # Custom architecture with separate pi/qf networks
            actor_arch = self.net_arch[0].get("pi", [400, 300])
            critic_arch = self.net_arch[0].get("qf", [400, 300])
        else:
            # Shared architecture
            actor_arch = self.net_arch
            critic_arch = self.net_arch

        # Build actor network (feature processing layers)
        if actor_arch:
            self.actor_net = create_mlp(
                features_dim, actor_arch[-1], actor_arch[:-1], self.activation_fn
            )
        else:
            # No hidden layers - use identity
            self.actor_net = create_mlp(
                features_dim, features_dim, [], self.activation_fn
            )
        # Register actor_net as submodule
        self.add_module("actor_net", self.actor_net)

        # Final output layer for deterministic actions
        from mlx_baselines3.common.torch_layers import MlxLinear

        latent_dim = actor_arch[-1] if actor_arch else features_dim
        self.actor_output = MlxLinear(latent_dim, self.action_dim)
        self.add_module("actor_output", self.actor_output)

        # Build twin critics (Q1 and Q2)
        self.q_networks = []
        for i in range(self.n_critics):
            q_net = create_mlp(
                features_dim + self.action_dim, 1, critic_arch, self.activation_fn
            )
            self.add_module(f"q_net_{i}", q_net)
            self.q_networks.append(q_net)

        # Build target networks for both actor and critics
        if actor_arch:
            self.actor_target_net = create_mlp(
                features_dim, actor_arch[-1], actor_arch[:-1], self.activation_fn
            )
        else:
            self.actor_target_net = create_mlp(
                features_dim, features_dim, [], self.activation_fn
            )
        self.add_module("actor_target_net", self.actor_target_net)

        # Target actor output layer
        latent_dim = actor_arch[-1] if actor_arch else features_dim
        self.actor_target_output = MlxLinear(latent_dim, self.action_dim)
        self.add_module("actor_target_output", self.actor_target_output)

        self.q_networks_target = []
        for i in range(self.n_critics):
            q_net_target = create_mlp(
                features_dim + self.action_dim, 1, critic_arch, self.activation_fn
            )
            self.add_module(f"q_net_target_{i}", q_net_target)
            self.q_networks_target.append(q_net_target)

    def _get_data(self) -> Dict[str, Any]:
        """Get data to save."""
        data = super()._get_data()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                n_critics=self.n_critics,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def _build_target_networks(self) -> None:
        """
        Build target networks by copying the main networks.
        This should be called after the main networks are built.
        """
        # Initialize target actor with same weights as main actor
        # Copy actor_net to actor_target_net
        actor_net_params = dict(self.actor_net.parameters())
        target_actor_net_params = dict(self.actor_target_net.parameters())
        for name in target_actor_net_params:
            if name in actor_net_params:
                target_actor_net_params[name] = mx.array(actor_net_params[name])

        # Copy actor_output to actor_target_output
        actor_output_params = dict(self.actor_output.parameters())
        target_actor_output_params = dict(self.actor_target_output.parameters())
        for name in target_actor_output_params:
            if name in actor_output_params:
                target_actor_output_params[name] = mx.array(actor_output_params[name])

        # Initialize target critics with same weights as main critics
        for i in range(self.n_critics):
            target_params = dict(self.q_networks[i].parameters())
            main_params = dict(self.q_networks_target[i].parameters())
            for name in main_params:
                if name in target_params:
                    main_params[name] = mx.array(target_params[name])

    def make_actor(self, features_extractor: nn.Module) -> nn.Module:
        """
        Create the actor network.

        Returns:
            actor_net: Deterministic actor network
        """
        features_dim = features_extractor.features_dim

        # Get actor architecture
        if (
            isinstance(self.net_arch, list)
            and len(self.net_arch) > 0
            and isinstance(self.net_arch[0], dict)
        ):
            actor_arch = self.net_arch[0].get("pi", [400, 300])
        else:
            actor_arch = self.net_arch

        # Create actor network with tanh output
        actor_net = create_mlp(
            features_dim, self.action_dim, actor_arch, self.activation_fn
        )
        actor_net = nn.Sequential(*actor_net)

        return actor_net

    def make_critic(self, features_extractor: nn.Module) -> List[nn.Module]:
        """
        Create the critic networks.

        Returns:
            List of critic networks
        """
        features_dim = features_extractor.features_dim

        # Get critic architecture
        if (
            isinstance(self.net_arch, list)
            and len(self.net_arch) > 0
            and isinstance(self.net_arch[0], dict)
        ):
            critic_arch = self.net_arch[0].get("qf", [400, 300])
        else:
            critic_arch = self.net_arch

        critics = []
        for i in range(self.n_critics):
            q_net = create_mlp(
                features_dim + self.action_dim, 1, critic_arch, self.activation_fn
            )
            q_net = nn.Sequential(*q_net)
            critics.append(q_net)

        return critics

    def actor_forward(self, features: mx.array) -> mx.array:
        """
        Forward pass through the actor network.

        Args:
            features: Input features

        Returns:
            actions: Deterministic actions (tanh-bounded to [-1, 1])
        """
        # Forward through actor network
        actor_output = self.actor_net(features)
        actions = self.actor_output(actor_output)

        # Apply tanh to bound actions to [-1, 1]
        actions = mx.tanh(actions)

        # Scale actions to action space bounds
        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            low = mx.array(self.action_space.low)
            high = mx.array(self.action_space.high)
            actions = low + (actions + 1.0) * 0.5 * (high - low)

        return actions

    def actor_target_forward(
        self, features: mx.array, noise: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass through the target actor network.

        Args:
            features: Input features
            noise: Optional noise to add to target actions (for policy smoothing)

        Returns:
            actions: Target actions with optional noise
        """
        # Forward through target actor network
        actor_output = self.actor_target_net(features)
        actions = self.actor_target_output(actor_output)

        # Apply tanh to bound actions to [-1, 1]
        actions = mx.tanh(actions)

        # Add noise for target policy smoothing (if provided)
        if noise is not None:
            actions = actions + noise
            # Re-clip after adding noise
            actions = mx.clip(actions, -1.0, 1.0)

        # Scale actions to action space bounds
        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            low = mx.array(self.action_space.low)
            high = mx.array(self.action_space.high)
            actions = low + (actions + 1.0) * 0.5 * (high - low)

        return actions

    def critic_forward(self, features: mx.array, actions: mx.array) -> List[mx.array]:
        """
        Forward pass through the critic networks.

        Args:
            features: Observation features
            actions: Actions to evaluate

        Returns:
            List of Q-values from each critic
        """
        # Concatenate features and actions
        q_input = mx.concatenate([features, actions], axis=-1)

        # Forward through each critic
        q_values = []
        for q_net in self.q_networks:
            q_val = q_net(q_input)
            q_values.append(q_val)

        return q_values

    def critic_target_forward(
        self, features: mx.array, actions: mx.array
    ) -> List[mx.array]:
        """
        Forward pass through the target critic networks.

        Args:
            features: Observation features
            actions: Actions to evaluate

        Returns:
            List of Q-values from each target critic
        """
        # Concatenate features and actions
        q_input = mx.concatenate([features, actions], axis=-1)

        # Forward through each target critic
        q_values = []
        for q_net_target in self.q_networks_target:
            q_val = q_net_target(q_input)
            q_values.append(q_val)

        return q_values

    def forward(
        self,
        obs: mx.array,
        deterministic: bool = True,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass used by BasePolicy and the algorithms.
        For TD3 we return:
            actions: deterministic action from the actor
            values: dummy values (TD3 does not learn state values)
            log_prob: dummy log probabilities (deterministic policy)
        """
        features = self.extract_features(obs)
        actions = self.actor_forward(features)

        # TD3 has no separate state-value function; return zeros to keep the
        # interface consistent with other algorithms.
        values = mx.zeros(actions.shape[:-1])
        log_probs = mx.zeros(actions.shape[:-1])  # Deterministic policy

        return actions, values, log_probs

    def predict_values(self, obs: mx.array) -> mx.array:
        """
        Predict state values using Q-networks.
        We approximate state value by taking the min of the twin Q-functions
        for the current policy's action.
        """
        features = self.extract_features(obs)
        actions = self.actor_forward(features)
        q_values = self.critic_forward(features, actions)
        # Take minimum of twin critics (conservative estimate)
        values = mx.minimum(q_values[0], q_values[1]).squeeze(-1)
        return values

    def evaluate_actions_functional(
        self,
        params: Dict[str, mx.array],
        observations: mx.array,
        actions: mx.array,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """
        Evaluate Q-values for provided actions using explicit parameters.

        Returns concatenated critic outputs and placeholder log-probabilities.
        """

        def _evaluate() -> Tuple[mx.array, mx.array, Optional[mx.array]]:
            features = self.extract_features(observations)
            q_values = self.critic_forward(features, actions)
            q_values_concat = mx.concatenate(q_values, axis=-1)
            log_prob = mx.zeros(q_values_concat.shape[:-1])
            return q_values_concat, log_prob, None

        return self._with_temporary_params(params, _evaluate)

    def act_functional(
        self,
        params: Dict[str, mx.array],
        observations: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Select deterministic actions using explicit parameters.
        """

        def _act() -> Tuple[mx.array, mx.array, mx.array]:
            actions, values, log_prob = self.forward(observations, deterministic=True)
            return actions, log_prob, values

        actions, log_prob, values = self._with_temporary_params(params, _act)
        return actions, log_prob, values

    def __call__(
        self, obs: mx.array, deterministic: bool = True
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass through the policy.

        Args:
            obs: Observations
            deterministic: Always True for TD3 (deterministic policy)

        Returns:
            actions: Selected actions
            values: Not used in TD3 (returns zeros)
            log_probs: Not used in TD3 (returns zeros)
        """
        return self.forward(obs, deterministic=True)  # TD3 is always deterministic

    def predict(
        self,
        observation: Union[mx.array, Dict[str, mx.array]],
        state: Optional[Tuple[mx.array, ...]] = None,
        episode_start: Optional[mx.array] = None,
        deterministic: bool = True,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, ...]]]:
        """
        Predict actions for given observations.

        Args:
            observation: Input observations
            state: Not used in TD3
            episode_start: Not used in TD3
            deterministic: Always True for TD3 (deterministic policy)

        Returns:
            actions: Predicted actions
            state: Not used in TD3 (returns None)
        """
        self.set_training_mode(False)

        from ..common.utils import obs_as_mlx

        # Convert to MLX arrays while preserving shape
        obs_tensor = obs_as_mlx(observation)

        # Add batch dimension only if observation is not already batched
        if isinstance(obs_tensor, dict):
            # For dict observations, check if we need to add batch dimension
            obs_batch = {}
            for key, obs in obs_tensor.items():
                if obs.ndim == len(self.observation_space[key].shape):
                    obs_batch[key] = obs[None]  # Add batch dimension
                else:
                    obs_batch[key] = obs  # Already batched
            obs_tensor = obs_batch
        else:
            # For array observations, check if we need to add batch dimension
            if obs_tensor.ndim == len(self.observation_space.shape):
                obs_tensor = obs_tensor[None]  # Add batch dimension
            # else: already batched, use as-is

        actions, _, _ = self(obs_tensor, deterministic=True)

        # Convert back to numpy and remove batch dimension for single observations
        if isinstance(actions, mx.array):
            actions = np.array(actions)
            # Remove batch dimension only if original observation was not batched
            if isinstance(observation, np.ndarray) and observation.ndim == len(
                self.observation_space.shape
            ):
                actions = actions.squeeze(0)
            elif isinstance(observation, dict):
                # For dict observations, check if original was not batched
                first_key = next(iter(observation.keys()))
                if observation[first_key].ndim == len(
                    self.observation_space[first_key].shape
                ):
                    actions = actions.squeeze(0)

        return actions, None


class MlpPolicy(TD3Policy):
    """
    TD3 policy with MLP networks for both actor and critics.

    This is the standard TD3 policy for environments with flat observation spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=FlattenExtractor)


class CnnPolicy(TD3Policy):
    """
    TD3 policy with a CNN feature extractor for image observations and an MLP
    for actor/critics.

    This policy is suitable for environments with image observations.
    Note: CNN support is not yet implemented in MLX-Baselines3.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CNN feature extractor is not yet implemented")


class MultiInputPolicy(TD3Policy):
    """
    TD3 policy for environments with multiple input types (e.g., images plus
    vector observations).

    This policy uses a combined feature extractor that can handle dictionary
    observation spaces.
    Note: MultiInput support is not yet implemented in MLX-Baselines3.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MultiInput feature extractor is not yet implemented")


# Register policy aliases for easy access
TD3Policy.MlpPolicy = MlpPolicy
TD3Policy.CnnPolicy = CnnPolicy
TD3Policy.MultiInputPolicy = MultiInputPolicy
