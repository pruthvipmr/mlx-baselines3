"""
Action Distributions

Probability distributions for action sampling and log probability computation
in reinforcement learning algorithms using MLX.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import math
import mlx.core as mx
import mlx.nn as nn
from mlx_baselines3.common.type_aliases import MlxArray


class Distribution(ABC):
    """Abstract base class for action distributions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample an action from the distribution."""
        pass

    @abstractmethod
    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        pass

    @abstractmethod
    def entropy(self) -> MlxArray:
        """Compute entropy of the distribution."""
        pass

    @abstractmethod
    def mode(self) -> MlxArray:
        """Return the mode (most likely value) of the distribution."""
        pass


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete action spaces.

    Used for environments with discrete actions (e.g., Atari games).
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.logits = None
        self.probs = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Linear:
        """
        Create the layer that outputs the logits for the categorical distribution.

        Args:
            latent_dim: Dimension of the last layer of the policy network

        Returns:
            Linear layer that outputs action logits
        """
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits: MlxArray) -> "CategoricalDistribution":
        """
        Create the distribution given the action logits.

        Args:
            action_logits: Logits for each action

        Returns:
            Self for chaining
        """
        self.logits = action_logits
        self.probs = nn.softmax(action_logits, axis=-1)
        return self

    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample an action from the categorical distribution."""
        if self.probs is None:
            raise ValueError("Must call proba_distribution() first")

        if deterministic:
            return self.mode()

        # Sample from categorical distribution
        # Use Gumbel-max trick for sampling
        gumbel = -mx.log(-mx.log(mx.random.uniform(shape=self.probs.shape)))
        return mx.argmax(self.logits + gumbel, axis=-1)

    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        if self.logits is None:
            raise ValueError("Must call proba_distribution() first")

        # Convert actions to int32 if needed
        actions = actions.astype(mx.int32)

        # Ensure actions are within valid range
        actions = mx.clip(actions, 0, self.action_dim - 1)

        # Use log_softmax for numerical stability
        log_probs = nn.log_softmax(self.logits, axis=-1)

        # Gather log probabilities for selected actions
        batch_size = actions.shape[0]
        indices = mx.arange(batch_size)

        if len(actions.shape) == 1:
            # Single action per sample
            return log_probs[indices, actions]
        else:
            # Multiple actions (should not happen for categorical)
            return log_probs[indices, actions.squeeze(-1)]

    def entropy(self) -> MlxArray:
        """Compute entropy of the categorical distribution."""
        if self.probs is None:
            raise ValueError("Must call proba_distribution() first")

        # Entropy = -sum(p * log(p))
        log_probs = nn.log_softmax(self.logits, axis=-1)
        return -mx.sum(self.probs * log_probs, axis=-1)

    def mode(self) -> MlxArray:
        """Return the mode (most likely action) of the distribution."""
        if self.logits is None:
            raise ValueError("Must call proba_distribution() first")

        return mx.argmax(self.logits, axis=-1)

    def actions_from_params(
        self, action_logits: MlxArray, deterministic: bool = False
    ) -> MlxArray:
        """
        Convenience method to sample actions from logits.

        Args:
            action_logits: Logits for each action
            deterministic: Whether to sample deterministically

        Returns:
            Sampled actions
        """
        self.proba_distribution(action_logits)
        return self.sample(deterministic)

    def log_prob_from_params(
        self, action_logits: MlxArray, actions: MlxArray
    ) -> MlxArray:
        """
        Convenience method to compute log probabilities from logits.

        Args:
            action_logits: Logits for each action
            actions: Actions to compute log probabilities for

        Returns:
            Log probabilities
        """
        self.proba_distribution(action_logits)
        return self.log_prob(actions)


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix for continuous action spaces.

    Used for environments with continuous actions (e.g., robotic control).
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean = None
        self.log_std = None
        self.std = None
        self.action_space = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Linear, MlxArray]:
        """
        Create the layers that output the mean and log std for the Gaussian
        distribution.

        Args:
            latent_dim: Dimension of the last layer of the policy network
            log_std_init: Initial value for log standard deviation

        Returns:
            Tuple of (mean_layer, log_std_parameter)
        """
        mean_layer = nn.Linear(latent_dim, self.action_dim)
        # Log std is a learnable parameter independent of the input
        log_std = mx.full((self.action_dim,), log_std_init)
        return mean_layer, log_std

    def proba_distribution(
        self, mean: MlxArray, log_std: MlxArray
    ) -> "DiagGaussianDistribution":
        """
        Create the distribution given the mean and log std.

        Args:
            mean: Mean of the Gaussian distribution
            log_std: Log standard deviation of the Gaussian distribution

        Returns:
            Self for chaining
        """
        self.mean = mean
        self.log_std = log_std
        self.std = mx.exp(log_std)
        return self

    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample an action from the Gaussian distribution."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")

        if deterministic:
            return self.mode()

        # Sample from standard normal and scale/shift
        noise = mx.random.normal(shape=self.mean.shape)
        actions = self.mean + noise * self.std

        # Apply action clipping if action space bounds are available
        if hasattr(self, "action_space") and self.action_space is not None:
            if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
                low = mx.array(self.action_space.low)
                high = mx.array(self.action_space.high)
                actions = mx.clip(actions, low, high)

        return actions

    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")

        # Log probability of multivariate Gaussian with diagonal covariance
        # log p(x) = -0.5 * sum((x - mu)^2 / sigma^2) - 0.5 * sum(log(2*pi*sigma^2))

        log_prob = -0.5 * mx.sum(
            ((actions - self.mean) / self.std) ** 2
            + 2 * self.log_std
            + math.log(2 * math.pi),
            axis=-1,
        )

        return log_prob

    def entropy(self) -> MlxArray:
        """Compute entropy of the Gaussian distribution."""
        if self.log_std is None:
            raise ValueError("Must call proba_distribution() first")

        # Entropy of multivariate Gaussian with diagonal covariance
        # H = 0.5 * sum(log(2*pi*e*sigma^2))
        return 0.5 * mx.sum(2 * self.log_std + math.log(2 * math.pi * math.e), axis=-1)

    def mode(self) -> MlxArray:
        """Return the mode (mean) of the distribution."""
        if self.mean is None:
            raise ValueError("Must call proba_distribution() first")

        actions = self.mean

        # Apply action clipping if action space bounds are available
        if hasattr(self, "action_space") and self.action_space is not None:
            if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
                low = mx.array(self.action_space.low)
                high = mx.array(self.action_space.high)
                actions = mx.clip(actions, low, high)

        return actions

    def actions_from_params(
        self, mean: MlxArray, log_std: MlxArray, deterministic: bool = False
    ) -> MlxArray:
        """
        Convenience method to sample actions from mean and log std.

        Args:
            mean: Mean of the Gaussian distribution
            log_std: Log standard deviation of the Gaussian distribution
            deterministic: Whether to sample deterministically (return mean)

        Returns:
            Sampled actions
        """
        self.proba_distribution(mean, log_std)
        return self.sample(deterministic)

    def log_prob_from_params(
        self, mean: MlxArray, log_std: MlxArray, actions: MlxArray
    ) -> MlxArray:
        """
        Convenience method to compute log probabilities from mean and log std.

        Args:
            mean: Mean of the Gaussian distribution
            log_std: Log standard deviation of the Gaussian distribution
            actions: Actions to compute log probabilities for

        Returns:
            Log probabilities
        """
        self.proba_distribution(mean, log_std)
        return self.log_prob(actions)


class MultiCategoricalDistribution(Distribution):
    """
    Multi-categorical distribution for multi-discrete action spaces.

    Used for environments where each action component is drawn from
    a separate categorical distribution (e.g., multi-agent scenarios).
    """

    def __init__(self, nvec):
        """
        Initialize multi-categorical distribution.

        Args:
            nvec: List/array of the number of categories for each action component
        """
        super().__init__()
        self.nvec = nvec if isinstance(nvec, list) else list(nvec)
        self.action_dims = len(self.nvec)
        self.total_action_dim = sum(self.nvec)
        self.logits = None
        self.split_logits = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Linear:
        """
        Create the layer that outputs the logits for all action components.

        Args:
            latent_dim: Dimension of the last layer of the policy network

        Returns:
            Linear layer that outputs concatenated action logits
        """
        return nn.Linear(latent_dim, self.total_action_dim)

    def proba_distribution(
        self, action_logits: MlxArray
    ) -> "MultiCategoricalDistribution":
        """
        Create the distribution given the action logits.

        Args:
            action_logits: Concatenated logits for all action components

        Returns:
            Self for chaining
        """
        self.logits = action_logits

        # Split logits into separate components
        self.split_logits = []
        start_idx = 0
        for nvec_i in self.nvec:
            end_idx = start_idx + nvec_i
            self.split_logits.append(action_logits[..., start_idx:end_idx])
            start_idx = end_idx

        return self

    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample actions from the multi-categorical distribution."""
        if self.split_logits is None:
            raise ValueError("Must call proba_distribution() first")

        if deterministic:
            return self.mode()

        # Sample from each categorical distribution
        actions = []
        for logits in self.split_logits:
            # Use Gumbel-max trick for sampling
            gumbel = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape)))
            action = mx.argmax(logits + gumbel, axis=-1)
            actions.append(action)

        # Stack actions along the last dimension
        return mx.stack(actions, axis=-1)

    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        if self.split_logits is None:
            raise ValueError("Must call proba_distribution() first")

        # Convert actions to int32 if needed
        actions = actions.astype(mx.int32)

        # Compute log probability for each action component
        total_log_prob = mx.zeros(
            actions.shape[:-1]
        )  # Remove last dimension (action components)

        for i, (logits, nvec_i) in enumerate(zip(self.split_logits, self.nvec)):
            # Extract action for this component
            action_i = actions[..., i]

            # Clip to valid range
            action_i = mx.clip(action_i, 0, nvec_i - 1)

            # Compute log probabilities using log_softmax for numerical stability
            log_probs_i = nn.log_softmax(logits, axis=-1)

            # Gather log probabilities for selected actions
            batch_indices = mx.arange(action_i.shape[0])
            if len(action_i.shape) == 1:
                log_prob_i = log_probs_i[batch_indices, action_i]
            else:
                # Handle batched case
                log_prob_i = log_probs_i[batch_indices, action_i.squeeze(-1)]

            total_log_prob = total_log_prob + log_prob_i

        return total_log_prob

    def entropy(self) -> MlxArray:
        """Compute entropy of the multi-categorical distribution."""
        if self.split_logits is None:
            raise ValueError("Must call proba_distribution() first")

        # Sum entropy of each categorical distribution
        total_entropy = mx.zeros(
            self.split_logits[0].shape[:-1]
        )  # Remove last dimension (actions)

        for logits in self.split_logits:
            # Compute entropy for this component
            probs = nn.softmax(logits, axis=-1)
            log_probs = nn.log_softmax(logits, axis=-1)
            entropy_i = -mx.sum(probs * log_probs, axis=-1)
            total_entropy = total_entropy + entropy_i

        return total_entropy

    def mode(self) -> MlxArray:
        """Return the mode (most likely action) of the distribution."""
        if self.split_logits is None:
            raise ValueError("Must call proba_distribution() first")

        # Get mode for each action component
        modes = []
        for logits in self.split_logits:
            mode_i = mx.argmax(logits, axis=-1)
            modes.append(mode_i)

        # Stack modes along the last dimension
        return mx.stack(modes, axis=-1)

    def actions_from_params(
        self, action_logits: MlxArray, deterministic: bool = False
    ) -> MlxArray:
        """
        Convenience method to sample actions from logits.

        Args:
            action_logits: Concatenated logits for all action components
            deterministic: Whether to sample deterministically

        Returns:
            Sampled actions
        """
        self.proba_distribution(action_logits)
        return self.sample(deterministic)

    def log_prob_from_params(
        self, action_logits: MlxArray, actions: MlxArray
    ) -> MlxArray:
        """
        Convenience method to compute log probabilities from logits.

        Args:
            action_logits: Concatenated logits for all action components
            actions: Actions to compute log probabilities for

        Returns:
            Log probabilities
        """
        self.proba_distribution(action_logits)
        return self.log_prob(actions)


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for multi-binary action spaces.

    Used for environments where each action component is a binary choice
    (e.g., multiple on/off switches).
    """

    def __init__(self, action_dim: int):
        """
        Initialize Bernoulli distribution.

        Args:
            action_dim: Number of binary action components
        """
        super().__init__()
        self.action_dim = action_dim
        self.logits = None
        self.probs = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Linear:
        """
        Create the layer that outputs the logits for the Bernoulli distribution.

        Args:
            latent_dim: Dimension of the last layer of the policy network

        Returns:
            Linear layer that outputs action logits
        """
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits: MlxArray) -> "BernoulliDistribution":
        """
        Create the distribution given the action logits.

        Args:
            action_logits: Logits for each binary action component

        Returns:
            Self for chaining
        """
        self.logits = action_logits
        self.probs = mx.sigmoid(action_logits)
        return self

    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample actions from the Bernoulli distribution."""
        if self.probs is None:
            raise ValueError("Must call proba_distribution() first")

        if deterministic:
            return self.mode()

        # Sample from Bernoulli distribution
        # Generate uniform random numbers and compare with probabilities
        uniform_samples = mx.random.uniform(shape=self.probs.shape)
        return (uniform_samples < self.probs).astype(mx.float32)

    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        if self.logits is None:
            raise ValueError("Must call proba_distribution() first")

        # Ensure actions are binary (0 or 1)
        actions = mx.clip(actions, 0.0, 1.0)

        # Log probability of Bernoulli distribution
        # log p(x) = x * log(p) + (1-x) * log(1-p)
        # Using logits for numerical stability:
        # log p(x) = x * logits - log(1 + exp(logits))

        # Use log_sigmoid for numerical stability
        # log_sigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
        # log_sigmoid(-x) = log(1 - sigmoid(x))
        log_prob_pos = nn.log_sigmoid(self.logits)  # log(sigmoid(logits))
        log_prob_neg = nn.log_sigmoid(-self.logits)  # log(1 - sigmoid(logits))

        log_probs = actions * log_prob_pos + (1 - actions) * log_prob_neg

        # Sum log probabilities across action dimensions
        return mx.sum(log_probs, axis=-1)

    def entropy(self) -> MlxArray:
        """Compute entropy of the Bernoulli distribution."""
        if self.probs is None:
            raise ValueError("Must call proba_distribution() first")

        # Entropy of Bernoulli distribution
        # H = -p*log(p) - (1-p)*log(1-p)
        # Use logits for numerical stability

        # entropy = -sigmoid(logits) * log_sigmoid(logits)
        # - sigmoid(-logits) * log_sigmoid(-logits)
        entropy_per_dim = -(
            self.probs * nn.log_sigmoid(self.logits)
            + (1 - self.probs) * nn.log_sigmoid(-self.logits)
        )

        # Sum entropy across action dimensions
        return mx.sum(entropy_per_dim, axis=-1)

    def mode(self) -> MlxArray:
        """Return the mode (most likely action) of the distribution."""
        if self.probs is None:
            raise ValueError("Must call proba_distribution() first")

        # Mode is 1 if p > 0.5, else 0
        return (self.probs > 0.5).astype(mx.float32)

    def actions_from_params(
        self, action_logits: MlxArray, deterministic: bool = False
    ) -> MlxArray:
        """
        Convenience method to sample actions from logits.

        Args:
            action_logits: Logits for each binary action component
            deterministic: Whether to sample deterministically

        Returns:
            Sampled actions
        """
        self.proba_distribution(action_logits)
        return self.sample(deterministic)

    def log_prob_from_params(
        self, action_logits: MlxArray, actions: MlxArray
    ) -> MlxArray:
        """
        Convenience method to compute log probabilities from logits.

        Args:
            action_logits: Logits for each binary action component
            actions: Actions to compute log probabilities for

        Returns:
            Log probabilities
        """
        self.proba_distribution(action_logits)
        return self.log_prob(actions)


# Alias for compatibility
MultiBinaryDistribution = BernoulliDistribution


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with tanh squashing for continuous action spaces.

    Provides reparameterized sampling and applies the Jacobian correction
    for the tanh transformation in log probability computations.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        self.epsilon = epsilon

    def _rsample(self) -> Tuple[MlxArray, MlxArray]:
        """Sample pre- and post-tanh actions via pathwise sampling."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")

        noise = mx.random.normal(shape=self.mean.shape)
        pre_tanh = self.mean + self.std * noise
        squashed = mx.tanh(pre_tanh)
        return pre_tanh, squashed

    def _log_prob_from_pre_tanh(self, pre_tanh: MlxArray) -> MlxArray:
        """Compute log π with tanh correction given pre-tanh actions."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")

        diff = (pre_tanh - self.mean) / self.std
        gaussian_log_prob = -0.5 * mx.sum(
            diff**2 + 2 * self.log_std + math.log(2 * math.pi),
            axis=-1,
        )
        correction = mx.sum(
            mx.log(1.0 - mx.tanh(pre_tanh) ** 2 + self.epsilon),
            axis=-1,
        )
        return gaussian_log_prob - correction

    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample squashed actions."""
        if deterministic:
            return self.mode()

        _, squashed = self._rsample()
        return squashed

    def sample_and_log_prob(self) -> Tuple[MlxArray, MlxArray]:
        """Sample squashed actions and return their log probability."""
        pre_tanh, squashed = self._rsample()
        log_prob = self._log_prob_from_pre_tanh(pre_tanh)
        return squashed, log_prob

    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of provided squashed actions."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")

        actions_clipped = mx.clip(actions, -1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh = mx.arctanh(actions_clipped)
        return self._log_prob_from_pre_tanh(pre_tanh)

    def mode(self) -> MlxArray:
        """Return the mode (tanh of mean) of the distribution."""
        if self.mean is None:
            raise ValueError("Must call proba_distribution() first")

        return mx.tanh(self.mean)


def make_proba_distribution(
    action_space, use_sde: bool = False, dist_kwargs: Optional[dict] = None
) -> Distribution:
    """
    Create a probability distribution from an action space.

    Args:
        action_space: The action space
        use_sde: Whether to use State Dependent Exploration (not implemented)
        dist_kwargs: Additional arguments for the distribution

    Returns:
        The appropriate distribution for the action space
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if use_sde:
        raise NotImplementedError(
            "State Dependent Exploration (SDE) is not supported yet"
        )

    import gymnasium as gym

    if isinstance(action_space, gym.spaces.Discrete):
        # Discrete action space
        return CategoricalDistribution(action_space.n)
    elif isinstance(action_space, gym.spaces.Box):
        # Continuous action space
        action_dim = action_space.shape[0]
        dist = DiagGaussianDistribution(action_dim)
        # Store action space for clipping
        dist.action_space = action_space
        return dist
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        # Multi-discrete action space
        return MultiCategoricalDistribution(action_space.nvec)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        # Multi-binary action space
        action_dim = action_space.n
        return BernoulliDistribution(action_dim)
    else:
        raise NotImplementedError(f"Action space {action_space} is not supported")
