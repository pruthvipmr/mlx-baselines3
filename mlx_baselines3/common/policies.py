"""
Policy Networks.

Policy network implementations for reinforcement learning algorithms using
MLX. Includes base policy classes and specific implementations for different
types of RL algorithms.
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    MlxModule,
    create_mlp,
)
from mlx_baselines3.common.distributions import (
    Distribution,
    make_proba_distribution,
)
from mlx_baselines3.common.type_aliases import MlxArray, Schedule


class BasePolicy(MlxModule):
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
        super().__init__()  # Initialize MlxModule

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
        self.features_extractor = features_extractor_class(
            observation_space, **features_extractor_kwargs
        )
        # Register as a module for parameter discovery
        self.add_module("features_extractor", self.features_extractor)

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
    def forward(
        self, obs: MlxArray, deterministic: bool = False
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
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

    @abstractmethod
    def evaluate_actions_functional(
        self,
        params: Dict[str, MlxArray],
        observations: MlxArray,
        actions: MlxArray,
    ) -> Tuple[MlxArray, MlxArray, Optional[MlxArray]]:
        """
        Functional evaluation that uses explicit parameters instead of module state.

        Args:
            params: Parameter dictionary, typically from ``state_dict()``
            observations: Batch of observations
            actions: Batch of actions that were taken

        Returns:
            Tuple of (values, log_prob, entropy) as pure MLX arrays. Entropy
            may be ``None`` when not defined by a distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def act_functional(
        self,
        params: Dict[str, MlxArray],
        observations: MlxArray,
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Functional action selection that samples using explicit parameters.

        Args:
            params: Parameter dictionary, typically from ``state_dict()``
            observations: Batch of observations to act on

        Returns:
            Tuple of (actions, log_prob, values) as pure MLX arrays.
        """
        raise NotImplementedError

    def _with_temporary_params(
        self, params: Dict[str, MlxArray], fn: Callable[[], Tuple[Any, ...]]
    ) -> Tuple[Any, ...]:
        """
        Execute ``fn`` after temporarily loading ``params`` into the module.

        Note: This helper keeps current implementations working until policies
        expose fully functional (side-effect free) evaluation paths.
        """
        original_params = self.state_dict()

        try:
            self.load_state_dict(params, strict=False)
            return fn()
        finally:
            self.load_state_dict(original_params, strict=False)

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
                observation = {
                    key: mx.expand_dims(obs, 0) for key, obs in observation.items()
                }
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
        return getattr(self, "training", True)

    def get_distribution(self, obs: MlxArray) -> Distribution:
        """
        Get the current policy distribution.

        Args:
            obs: Observations

        Returns:
            Action distribution
        """
        raise NotImplementedError()

    def evaluate_actions(
        self, obs: MlxArray, actions: MlxArray
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
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

    def __call__(
        self, obs: MlxArray, deterministic: bool = False
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Call the policy forward method.

        Args:
            obs: Observations
            deterministic: Whether to sample deterministically

        Returns:
            Tuple of (actions, values, log_probs)
        """
        return self.forward(obs, deterministic=deterministic)

    # Parameter management is now inherited from MlxModule


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (A2C, PPO, etc.).

    Args:
        observation_space: The observation space
        action_space: The action space
        lr_schedule: Learning rate schedule
        net_arch: The specification of the policy and value networks
            architecture
        activation_fn: Activation function
        ortho_init: Whether to apply orthogonal initialization
        use_sde: Whether to use State Dependent Exploration
        log_std_init: Initial value for log standard deviation
        full_std: Whether to use full standard deviation matrix
        use_expln: Whether to use exponential linear for std
        squash_output: Whether to squash the output using tanh
        features_extractor_class: Features extractor to use
        features_extractor_kwargs: Keyword arguments for features extractor
        share_features_extractor: Whether to share features extractor between
            actor and critic
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
            # Register them as modules
            self.add_module("pi_features_extractor", self.pi_features_extractor)
            self.add_module("vf_features_extractor", self.vf_features_extractor)
        else:
            # Shared features extractor
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor

        # Build actor (policy) network
        latent_dim_pi = self.pi_features_extractor.features_dim
        self.action_net = self._build_action_net(latent_dim_pi)
        # Register action network as a module
        self.add_module("action_net", self.action_net)

        # Build critic (value) network
        latent_dim_vf = self.vf_features_extractor.features_dim
        self.value_net = self._build_value_net(latent_dim_vf)
        # Register value network as a module
        self.add_module("value_net", self.value_net)

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

    def forward(
        self, obs: MlxArray, deterministic: bool = False
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Forward pass in both actor and critic networks.

        Args:
            obs: Observations
            deterministic: Whether to sample deterministically

        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
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

    def evaluate_actions(
        self, obs: MlxArray, actions: MlxArray
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
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

    def __call__(
        self, obs: MlxArray, deterministic: bool = False
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Call the policy forward method.

        Args:
            obs: Observations
            deterministic: Whether to sample deterministically

        Returns:
            Tuple of (actions, values, log_probs)
        """
        return self.forward(obs, deterministic=deterministic)

    def evaluate_actions_functional(
        self,
        params: Dict[str, MlxArray],
        observations: MlxArray,
        actions: MlxArray,
    ) -> Tuple[MlxArray, MlxArray, Optional[MlxArray]]:
        """
        Evaluate actions using explicit parameters.

        Note: This temporarily loads parameters and will be replaced by a pure
        functional implementation in a later phase.
        """

        def _evaluate() -> Tuple[MlxArray, MlxArray, MlxArray]:
            values, log_prob, entropy = self.evaluate_actions(observations, actions)
            return values, log_prob, entropy

        values, log_prob, entropy = self._with_temporary_params(params, _evaluate)
        return values, log_prob, entropy

    def act_functional(
        self,
        params: Dict[str, MlxArray],
        observations: MlxArray,
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Select actions using explicit parameters.

        Note: This temporarily loads parameters and will be replaced by a pure
        functional implementation in a later phase.
        """

        def _act() -> Tuple[MlxArray, MlxArray, MlxArray]:
            actions, values, log_prob = self.forward(observations, deterministic=False)
            return actions, log_prob, values

        actions, log_prob, values = self._with_temporary_params(params, _act)
        return actions, log_prob, values

    def functional_evaluate_actions(
        self, params: Dict[str, mx.array], obs: MlxArray, actions: MlxArray
    ) -> Tuple[MlxArray, MlxArray, MlxArray]:
        """
        Evaluate actions functionally without modifying policy state.

        This method applies the policy with given parameters without
        loading them into the module state, enabling efficient
        gradient computation.

        Args:
            params: Parameter dictionary
            obs: Observations
            actions: Actions

        Returns:
            Tuple of (estimated values, log prob of actions, entropy)
        """
        values, log_prob, entropy = self.evaluate_actions_functional(
            params, obs, actions
        )
        return values, log_prob, entropy

    def create_functional_apply_fn(self):
        """
        Create a functional application function for this policy.

        Returns:
            Function that takes (params, obs, actions) and returns
            (values, log_prob, entropy)
        """

        def apply_fn(params: Dict[str, mx.array], obs: MlxArray, actions: MlxArray):
            return self.functional_evaluate_actions(params, obs, actions)

        return apply_fn


class MultiInputActorCriticPolicy(ActorCriticPolicy):
    """
    MultiInputActorCriticPolicy for multi-input observations (dict observations).

    This policy handles dictionary observation spaces where different parts of
    the observation might require different processing.
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
CnnPolicy = (
    ActorCriticPolicy  # Will be implemented later when CNN features extractor is added
)
MultiInputPolicy = MultiInputActorCriticPolicy
