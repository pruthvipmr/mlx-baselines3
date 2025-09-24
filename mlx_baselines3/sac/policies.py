"""SAC policies for continuous action spaces with stochastic actor and twin critics."""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_baselines3.common.distributions import SquashedDiagGaussianDistribution
from mlx_baselines3.common.policies import BasePolicy
from mlx_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from mlx_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    create_mlp,
)
from mlx_baselines3.common.type_aliases import Schedule


class SACPolicy(BasePolicy):
    """
    SAC policy with stochastic actor and twin critics.
    
    SAC uses:
    - A stochastic actor that outputs a squashed Gaussian distribution
    - Twin critics (Q1 and Q2) to reduce overestimation bias
    - Separate target critics for stable training
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Optional[Type] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        # Set attributes before calling super().__init__() because _build() is called from there
        if net_arch is None:
            net_arch = [256, 256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor
        
        if use_sde:
            raise NotImplementedError("State Dependent Exploration (SDE) is not yet supported")

        # SAC is only for continuous action spaces
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("SAC only supports continuous action spaces (Box)")

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
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
                n_critics=self.n_critics,
                share_features_extractor=self.share_features_extractor,
                lr_schedule=self._dummy_schedule,  # Dummy schedule
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """Reset noise for SDE (not implemented)."""
        if self.use_sde:
            raise NotImplementedError("SDE is not supported")

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
        if isinstance(self.net_arch, list) and len(self.net_arch) > 0 and isinstance(self.net_arch[0], dict):
            # Custom architecture with separate pi/qf networks
            actor_arch = self.net_arch[0].get("pi", [256, 256])
            critic_arch = self.net_arch[0].get("qf", [256, 256])
        else:
            # Shared architecture
            actor_arch = self.net_arch
            critic_arch = self.net_arch

        # Build actor network (feature processing layers only)
        if actor_arch:
            self.actor_net = create_mlp(features_dim, actor_arch[-1], actor_arch[:-1], self.activation_fn)
        else:
            # No hidden layers - use identity
            self.actor_net = create_mlp(features_dim, features_dim, [], self.activation_fn)
        # Register actor_net as submodule
        self.add_module("actor_net", self.actor_net)
        
        # Actor head: outputs mean and log_std using MLX-native linear layers
        from mlx_baselines3.common.torch_layers import MlxLinear
        latent_dim = actor_arch[-1] if actor_arch else features_dim
        mu_layer = MlxLinear(latent_dim, self.action_dim)
        log_std_layer = MlxLinear(latent_dim, self.action_dim)
        self.add_module("mu", mu_layer)
        self.add_module("log_std", log_std_layer)

        # Build twin critics (Q1 and Q2)
        self.q_networks = []
        for i in range(self.n_critics):
            q_net = create_mlp(
                features_dim + self.action_dim, 1, critic_arch, self.activation_fn
            )
            self.add_module(f"q_net_{i}", q_net)
            self.q_networks.append(q_net)

        # Build target networks for critics
        self.q_networks_target = []
        for i in range(self.n_critics):
            q_net_target = create_mlp(
                features_dim + self.action_dim, 1, critic_arch, self.activation_fn
            )
            self.add_module(f"q_net_target_{i}", q_net_target)
            self.q_networks_target.append(q_net_target)

        # Initialize log_std
        self.log_std_init_value = self.log_std_init

        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(self.action_dim)
        self.action_dist.action_space = self.action_space

    def _get_data(self) -> Dict[str, Any]:
        """Get data to save."""
        data = super()._get_data()
        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            use_expln=self.use_expln,
            clip_mean=self.clip_mean,
            n_critics=self.n_critics,
            share_features_extractor=self.share_features_extractor,
        ))
        return data

    def _build_target_networks(self) -> None:
        """
        Build target networks by copying the main networks.
        This should be called after the main networks are built.
        """
        # Initialize target networks with same weights as main networks
        for i in range(self.n_critics):
            target_params = dict(self.q_networks[i].parameters())
            main_params = dict(self.q_networks_target[i].parameters())
            for name in main_params:
                if name in target_params:
                    main_params[name] = mx.array(target_params[name])

    def make_actor(self, features_extractor: nn.Module) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Create the actor network components.
        
        Returns:
            actor_net: Feature processing network
            mu: Mean output layer
            log_std: Log standard deviation output layer
        """
        features_dim = features_extractor.features_dim
        
        # Get actor architecture
        if isinstance(self.net_arch, list) and len(self.net_arch) > 0 and isinstance(self.net_arch[0], dict):
            actor_arch = self.net_arch[0].get("pi", [256, 256])
        else:
            actor_arch = self.net_arch

        # Create actor network
        actor_net = create_mlp(features_dim, -1, actor_arch, self.activation_fn)
        actor_net = nn.Sequential(*actor_net)
        
        # Output layers
        mu = nn.Linear(actor_arch[-1] if actor_arch else features_dim, self.action_dim)
        log_std = nn.Linear(actor_arch[-1] if actor_arch else features_dim, self.action_dim)
        
        return actor_net, mu, log_std

    def make_critic(self, features_extractor: nn.Module) -> List[nn.Module]:
        """
        Create the critic networks.
        
        Returns:
            List of critic networks
        """
        features_dim = features_extractor.features_dim
        
        # Get critic architecture
        if isinstance(self.net_arch, list) and len(self.net_arch) > 0 and isinstance(self.net_arch[0], dict):
            critic_arch = self.net_arch[0].get("qf", [256, 256])
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

    def actor_forward(self, features: mx.array, deterministic: bool = False) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass through the actor network.
        
        Args:
            features: Input features
            deterministic: Whether to sample deterministically
            
        Returns:
            actions: Sampled actions (tanh-squashed)
            log_prob: Log probability of the actions
            entropy: Entropy of the action distribution
        """
        actor_output = self.actor_net(features)

        mean = self.mu(actor_output)
        log_std = self.log_std(actor_output)

        mean = mx.clip(mean, -self.clip_mean, self.clip_mean)
        log_std = mx.clip(log_std, -20, 2)

        dist = self.action_dist.proba_distribution(mean, log_std)

        if deterministic:
            squashed_actions = dist.mode()
            log_prob = mx.zeros(squashed_actions.shape[:-1])
        else:
            squashed_actions, log_prob = dist.sample_and_log_prob()

        # Approximate entropy with unsquashed Gaussian entropy
        entropy = mx.zeros_like(log_prob) if deterministic else dist.entropy()

        actions = squashed_actions
        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            low = mx.array(self.action_space.low, dtype=actions.dtype)
            high = mx.array(self.action_space.high, dtype=actions.dtype)
            scale = 0.5 * (high - low)
            actions = low + (actions + 1.0) * scale

            if not deterministic:
                log_scale = mx.sum(mx.log(scale + self.action_dist.epsilon))
                log_prob = log_prob - log_scale

        return actions, log_prob, entropy

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

    def critic_target_forward(self, features: mx.array, actions: mx.array) -> List[mx.array]:
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
        deterministic: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass used by BasePolicy and the algorithms.
        For SAC we return:
            actions: action sampled (or mean when deterministic)
            values: dummy values (SAC does not learn state values)
            log_prob: log π(a|s)
        """
        features = self.extract_features(obs)
        actions, log_probs, _ = self.actor_forward(features, deterministic=deterministic)

        # SAC has no separate state-value function; return zeros to keep the
        # interface consistent with other algorithms.
        values = mx.zeros(actions.shape[:-1])

        return actions, values, log_probs

    def predict_values(self, obs: mx.array) -> mx.array:
        """
        Predict state values using Q-networks.
        We approximate state value by taking the min of the twin Q-functions
        for the current policy's action.
        """
        features = self.extract_features(obs)
        # Use deterministic action for value prediction
        actions, _, _ = self.actor_forward(features, deterministic=True)
        q_values = self.critic_forward(features, actions)
        # Take minimum of twin critics (conservative estimate)
        values = mx.minimum(q_values[0], q_values[1]).squeeze(-1)
        return values

    def __call__(self, obs: mx.array, deterministic: bool = False) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass through the policy.
        
        Args:
            obs: Observations
            deterministic: Whether to sample deterministically
            
        Returns:
            actions: Selected actions
            values: Not used in SAC (returns zeros)
            log_probs: Log probabilities of actions
        """
        return self.forward(obs, deterministic)

    def get_distribution(self, obs: mx.array) -> SquashedDiagGaussianDistribution:
        """
        Get the action distribution for given observations.
        
        Args:
            obs: Observations
            
        Returns:
            Action distribution
        """
        features = self.extract_features(obs)
        actor_output = self.actor_net(features)
        mean = self.mu(actor_output)
        log_std = self.log_std(actor_output)
        
        # Clip mean and log_std
        mean = mx.clip(mean, -self.clip_mean, self.clip_mean)
        log_std = mx.clip(log_std, -20, 2)
        
        return self.action_dist.proba_distribution(mean, log_std)

    def predict(
        self,
        observation: Union[mx.array, Dict[str, mx.array]],
        state: Optional[Tuple[mx.array, ...]] = None,
        episode_start: Optional[mx.array] = None,
        deterministic: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, ...]]]:
        """
        Predict actions for given observations.
        
        Args:
            observation: Input observations
            state: Not used in SAC
            episode_start: Not used in SAC
            deterministic: Whether to sample deterministically
            
        Returns:
            actions: Predicted actions
            state: Not used in SAC (returns None)
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
        
        actions, _, _ = self(obs_tensor, deterministic=deterministic)
        
        # Convert back to numpy for compatibility and remove batch dimension if single observation
        if isinstance(actions, mx.array):
            actions = np.array(actions)
            # Remove batch dimension only if original observation was not batched
            if isinstance(observation, np.ndarray) and observation.ndim == len(self.observation_space.shape):
                actions = actions.squeeze(0)
            elif isinstance(observation, dict):
                # For dict observations, check if original was not batched
                first_key = next(iter(observation.keys()))
                if observation[first_key].ndim == len(self.observation_space[first_key].shape):
                    actions = actions.squeeze(0)
        
        return actions, None


class MlpPolicy(SACPolicy):
    """
    SAC policy with MLP networks for both actor and critics.
    
    This is the standard SAC policy for environments with flat observation spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=FlattenExtractor)


class CnnPolicy(SACPolicy):
    """
    SAC policy with CNN feature extractor for image observations and MLP for actor/critics.
    
    This policy is suitable for environments with image observations.
    Note: CNN support is not yet implemented in MLX-Baselines3.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CNN feature extractor is not yet implemented")


class MultiInputPolicy(SACPolicy):
    """
    SAC policy for environments with multiple input types (e.g., images + vector observations).
    
    This policy uses a combined feature extractor that can handle dictionary observation spaces.
    Note: MultiInput support is not yet implemented in MLX-Baselines3.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MultiInput feature extractor is not yet implemented")


# Register policy aliases for easy access
SACPolicy.MlpPolicy = MlpPolicy
SACPolicy.CnnPolicy = CnnPolicy
SACPolicy.MultiInputPolicy = MultiInputPolicy
