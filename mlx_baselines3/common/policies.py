"""
Policy Networks

Policy network implementations for reinforcement learning algorithms using MLX.
Includes base policy classes and specific implementations for different types of RL algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from mlx_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor, 
    MlpExtractor,
    MlxModule,
    create_mlp,
)
from mlx_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    make_proba_distribution,
)
from mlx_baselines3.common.type_aliases import MlxArray, Schedule
from mlx_baselines3.common.utils import get_device


class BasePolicy(ABC):
    """
    The base policy object: makes predictions, holds state.
    
    Args:
        observation_space: The observation space
        action_space: The action space
        lr_schedule: Learning rate schedule (could be constant)
        use_sde: Whether to use State Dependent Exploration
        sde_sample_freq: Sample a new noise matrix every n steps
        features_extractor_class: Features extractor to use
        features_extractor_kwargs: Keyword arguments for features extractor
        optimizer_class: The optimizer to use
        optimizer_kwargs: Additional keyword arguments for the optimizer
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
            
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        
        # Features extractor
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)
        
        # Optimizer settings
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        
        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)
        
        # Initialize training mode
        self.training = True
        
        self._build(lr_schedule)
    
    @abstractmethod
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy networks."""
        pass
    
    @abstractmethod 
    def forward(self, obs: MlxArray, deterministic: bool = False) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Forward pass in the policy network.
        
        Args:
            obs: Observations
            deterministic: Whether to sample deterministically
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        pass
    
    @abstractmethod
    def predict_values(self, obs: MlxArray) -> MlxArray:
        """
        Get the estimated values according to the current policy.
        
        Args:
            obs: Observations
            
        Returns:
            Estimated values
        """
        pass
    
    def predict(
        self, 
        observation: Union[MlxArray, Dict[str, MlxArray]], 
        state: Optional[Tuple[MlxArray, ...]] = None,
        episode_start: Optional[MlxArray] = None,
        deterministic: bool = False,
    ) -> Tuple[MlxArray, Optional[Tuple[MlxArray, ...]]]:
        """
        Get the policy action from an observation.
        
        Args:
            observation: The input observation
            state: The last states (used in recurrent policies)
            episode_start: Whether the observations correspond to new episodes
            deterministic: Whether to sample deterministically
            
        Returns:
            The model's action and the next state (used in recurrent policies)
        """
        # Preprocess observation
        vectorized_env = False
        if isinstance(observation, dict):
            # Handle dict observations
            for key, obs in observation.items():
                if len(obs.shape) > 1:
                    vectorized_env = True
                    break
        else:
            vectorized_env = len(observation.shape) > len(self.observation_space.shape)
        
        if not vectorized_env:
            # Add batch dimension
            if isinstance(observation, dict):
                observation = {key: mx.expand_dims(obs, 0) for key, obs in observation.items()}
            else:
                observation = mx.expand_dims(observation, 0)
        
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        
        if not vectorized_env:
            # Remove batch dimension
            actions = actions[0]
        
        return actions, state
    
    def extract_features(self, obs: MlxArray) -> MlxArray:
        """
        Preprocess the observation if needed and extract features.
        
        Args:
            obs: Observations
            
        Returns:
            Extracted features
        """
        return self.features_extractor(obs)
        
    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        
        Args:
            mode: If True, set to training mode, else set to evaluation mode
        """
        # In MLX, we don't have explicit training/eval modes like PyTorch
        # This is mainly for API compatibility
        self.training = mode
        
    def training_mode(self) -> bool:
        """Check if the policy is in training mode."""
        return getattr(self, 'training', True)
    
    def get_distribution(self, obs: MlxArray) -> Distribution:
        """
        Get the current policy distribution.
        
        Args:
            obs: Observations
            
        Returns:
            Action distribution
        """
        raise NotImplementedError()
    
    def evaluate_actions(self, obs: MlxArray, actions: MlxArray) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Evaluate actions according to the current policy.
        
        Args:
            obs: Observations
            actions: Actions
            
        Returns:
            Tuple of (estimated values, log prob of actions, entropy)
        """
        distribution = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        values = self.predict_values(obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    # ----- Parameter management (for save/load and optimization) -----
    def named_parameters(self) -> Dict[str, MlxArray]:
        """Return a flat dict of all trainable parameters for this policy.
        Keys are hierarchical names and values are MLX arrays."""
        params: Dict[str, MlxArray] = {}
        # Feature extractors (if present)
        if hasattr(self, "features_extractor") and hasattr(self.features_extractor, "parameters"):
            for k, v in self.features_extractor.parameters().items():
                params[f"features_extractor.{k}"] = v
        if hasattr(self, "pi_features_extractor") and self.pi_features_extractor is not self.features_extractor:
            for k, v in self.pi_features_extractor.parameters().items():
                params[f"pi_features_extractor.{k}"] = v
        if hasattr(self, "vf_features_extractor") and self.vf_features_extractor is not self.features_extractor:
            for k, v in self.vf_features_extractor.parameters().items():
                params[f"vf_features_extractor.{k}"] = v
        # Networks
        if hasattr(self, "action_net") and hasattr(self.action_net, "parameters"):
            for k, v in self.action_net.parameters().items():
                params[f"action_net.{k}"] = v
        if hasattr(self, "value_net") and hasattr(self.value_net, "parameters"):
            for k, v in self.value_net.parameters().items():
                params[f"value_net.{k}"] = v
        return params

    def parameters(self) -> Dict[str, MlxArray]:
        """Alias for named_parameters for optimizer compatibility."""
        return self.named_parameters()

    def state_dict(self) -> Dict[str, MlxArray]:
        """Export parameters to a dictionary suitable for serialization."""
        return {k: v for k, v in self.named_parameters().items()}

    def load_state_dict(self, state_dict: Dict[str, MlxArray], strict: bool = True) -> None:
        """Load parameters from a state dictionary.
        If strict is True, raise KeyError for missing parameters."""
        # Helper: set a parameter by traversing nested modules
        def set_param_by_path(root_module, path: str, value: MlxArray) -> bool:
            parts = path.split(".")
            module = root_module
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                if is_last:
                    # Set parameter on current module
                    if hasattr(module, "_parameters") and part in module._parameters:
                        module._parameters[part] = value
                        return True
                    return False
                # Traverse into submodule
                if hasattr(module, "_modules") and part in module._modules:
                    module = module._modules[part]
                else:
                    return False
            return False
        
        # Try to set parameters for each provided key
        applied_keys = set()
        for name, value in state_dict.items():
            if name.startswith("features_extractor.") and hasattr(self, "features_extractor"):
                if set_param_by_path(self.features_extractor, name.split(".", 1)[1], value):
                    applied_keys.add(name)
                    continue
            if name.startswith("pi_features_extractor.") and hasattr(self, "pi_features_extractor"):
                if set_param_by_path(self.pi_features_extractor, name.split(".", 1)[1], value):
                    applied_keys.add(name)
                    continue
            if name.startswith("vf_features_extractor.") and hasattr(self, "vf_features_extractor"):
                if set_param_by_path(self.vf_features_extractor, name.split(".", 1)[1], value):
                    applied_keys.add(name)
                    continue
            if name.startswith("action_net.") and hasattr(self, "action_net"):
                if set_param_by_path(self.action_net, name.split(".", 1)[1], value):
                    applied_keys.add(name)
                    continue
            if name.startswith("value_net.") and hasattr(self, "value_net"):
                if set_param_by_path(self.value_net, name.split(".", 1)[1], value):
                    applied_keys.add(name)
                    continue
        
        if strict and applied_keys != set(state_dict.keys()):
            unknown = set(state_dict.keys()) - applied_keys
            if unknown:
                raise KeyError(f"Unexpected parameter keys: {sorted(list(unknown))}")
        
        # Ensure arrays are evaluated on device
        try:
            mx.eval(list(self.named_parameters().values()))
        except Exception:
            pass
        return None


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (A2C, PPO, etc.).
    
    Args:
        observation_space: The observation space
        action_space: The action space  
        lr_schedule: Learning rate schedule
        net_arch: The specification of the policy and value networks architecture
        activation_fn: Activation function
        ortho_init: Whether to apply orthogonal initialization
        use_sde: Whether to use State Dependent Exploration
        log_std_init: Initial value for log standard deviation
        full_std: Whether to use full standard deviation matrix
        use_expln: Whether to use exponential linear for std
        squash_output: Whether to squash the output using tanh
        features_extractor_class: Features extractor to use
        features_extractor_kwargs: Keyword arguments for features extractor
        share_features_extractor: Whether to share features extractor between actor and critic
        normalize_images: Whether to normalize images
        optimizer_class: The optimizer to use
        optimizer_kwargs: Additional keyword arguments for the optimizer
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.share_features_extractor = share_features_extractor
        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        self.squash_output = squash_output
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy and value networks."""
        # Set default network architecture
        if self.net_arch is None:
            self.net_arch = dict(pi=[64, 64], vf=[64, 64])
        elif isinstance(self.net_arch, list):
            # Use same architecture for both actor and critic
            self.net_arch = dict(pi=self.net_arch, vf=self.net_arch)
        
        # Features extractor
        if not self.share_features_extractor:
            # Create separate features extractors for actor and critic
            self.pi_features_extractor = self.features_extractor_class(
                self.observation_space, **self.features_extractor_kwargs
            )
            self.vf_features_extractor = self.features_extractor_class(
                self.observation_space, **self.features_extractor_kwargs
            )
        else:
            # Shared features extractor
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        
        # Build actor (policy) network
        latent_dim_pi = self.pi_features_extractor.features_dim
        self.action_net = self._build_action_net(latent_dim_pi)
        
        # Build critic (value) network  
        latent_dim_vf = self.vf_features_extractor.features_dim
        self.value_net = self._build_value_net(latent_dim_vf)
        
        # Initialize weights
        if self.ortho_init:
            self._apply_init_weights()
            
        # Create optimizer
        self.optimizer = self.optimizer_class(learning_rate=self.lr_schedule(1))
    
    def _build_action_net(self, latent_dim: int) -> MlxModule:
        """Build the action network."""
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space
            return create_mlp(
                latent_dim,
                2 * self.action_space.shape[0],  # Mean and log_std
                self.net_arch["pi"],
                self.activation_fn,
                squash_output=self.squash_output,
            )
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Discrete action space
            return create_mlp(
                latent_dim,
                self.action_space.n,
                self.net_arch["pi"], 
                self.activation_fn,
            )
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")
    
    def _build_value_net(self, latent_dim: int) -> MlxModule:
        """Build the value network."""
        return create_mlp(
            latent_dim,
            1,  # Single value output
            self.net_arch["vf"],
            self.activation_fn,
        )
    
    def _apply_init_weights(self) -> None:
        """Apply orthogonal initialization to policy and value networks."""
        from mlx_baselines3.common.torch_layers import init_weights
        
        init_weights(self.action_net, gain=0.01 if self.squash_output else 1.0)
        init_weights(self.value_net, gain=1.0)
    
    def forward(self, obs: MlxArray, deterministic: bool = False) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Forward pass in both actor and critic networks.
        
        Args:
            obs: Observations
            deterministic: Whether to sample deterministically
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        latent_pi = self.pi_features_extractor(obs)
        latent_vf = self.vf_features_extractor(obs)
        
        # Get action distribution
        distribution = self.get_distribution(obs)
        actions = distribution.sample(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Get values
        values = self.value_net(latent_vf).squeeze(-1)
        
        return actions, values, log_prob
    
    def get_distribution(self, obs: MlxArray) -> Distribution:
        """Get the current policy distribution."""
        latent_pi = self.pi_features_extractor(obs)
        action_logits = self.action_net(latent_pi)
        
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous actions - split into mean and log_std
            mean, log_std = mx.split(action_logits, 2, axis=-1)
            
            # Clip log_std to reasonable range
            log_std = mx.clip(log_std, -20, 2)
            
            return self.action_dist.proba_distribution(mean, log_std)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Discrete actions
            return self.action_dist.proba_distribution(action_logits)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")
    
    def predict_values(self, obs: MlxArray) -> MlxArray:
        """Get the estimated values according to the current policy."""
        latent_vf = self.vf_features_extractor(obs)
        return self.value_net(latent_vf).squeeze(-1)
    
    def evaluate_actions(self, obs: MlxArray, actions: MlxArray) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Evaluate actions according to the current policy.
        
        Args:
            obs: Observations
            actions: Actions
            
        Returns:
            Tuple of (estimated values, log prob of actions, entropy)
        """
        distribution = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        values = self.predict_values(obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy


class MultiInputActorCriticPolicy(ActorCriticPolicy):
    """
    MultiInputActorCriticPolicy class for multi-input observations (Dict observations).
    
    This policy is designed to handle dictionary observation spaces where different
    parts of the observation might require different processing.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        **kwargs,
    ):
        # Use MlpExtractor as default for multi-input observations
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = MlpExtractor
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )


# Register policy names for easy access
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticPolicy  # Will be implemented later when CNN features extractor is added
MultiInputPolicy = MultiInputActorCriticPolicy
