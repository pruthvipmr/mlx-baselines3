"""
Action Distributions

Probability distributions for action sampling and log probability computation
in reinforcement learning algorithms using MLX.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
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
    
    def actions_from_params(self, action_logits: MlxArray, deterministic: bool = False) -> MlxArray:
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
    
    def log_prob_from_params(self, action_logits: MlxArray, actions: MlxArray) -> MlxArray:
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
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Linear, MlxArray]:
        """
        Create the layers that output the mean and log std for the Gaussian distribution.
        
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
    
    def proba_distribution(self, mean: MlxArray, log_std: MlxArray) -> "DiagGaussianDistribution":
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
        return self.mean + noise * self.std
    
    def log_prob(self, actions: MlxArray) -> MlxArray:
        """Compute log probability of actions."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")
        
        # Log probability of multivariate Gaussian with diagonal covariance
        # log p(x) = -0.5 * sum((x - mu)^2 / sigma^2) - 0.5 * sum(log(2*pi*sigma^2))
        
        log_prob = -0.5 * mx.sum(
            ((actions - self.mean) / self.std) ** 2 + 
            2 * self.log_std + 
            math.log(2 * math.pi), 
            axis=-1
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
        
        return self.mean
    
    def actions_from_params(
        self, 
        mean: MlxArray, 
        log_std: MlxArray, 
        deterministic: bool = False
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
        self, 
        mean: MlxArray, 
        log_std: MlxArray, 
        actions: MlxArray
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


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with tanh squashing for continuous action spaces.
    
    Used in SAC and other algorithms that require bounded action spaces.
    The actions are squashed to [-1, 1] using tanh.
    """
    
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        self.epsilon = epsilon
    
    def sample(self, deterministic: bool = False) -> MlxArray:
        """Sample an action and apply tanh squashing."""
        if deterministic:
            # For deterministic sampling, return tanh of the mean
            return mx.tanh(self.mean)
        
        # Sample from unsquashed distribution
        unsquashed_actions = super().sample(deterministic=False)
        
        # Apply tanh squashing
        return mx.tanh(unsquashed_actions)
    
    def log_prob(self, actions: MlxArray) -> MlxArray:
        """
        Compute log probability of squashed actions.
        
        This requires correcting for the Jacobian of the tanh transformation.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Must call proba_distribution() first")
        
        # Inverse tanh to get unsquashed actions
        # Use clipping to avoid numerical issues
        actions_clipped = mx.clip(actions, -1 + self.epsilon, 1 - self.epsilon)
        unsquashed_actions = mx.arctanh(actions_clipped)
        
        # Log probability of unsquashed actions
        log_prob_unsquashed = super().log_prob(unsquashed_actions)
        
        # Jacobian correction for tanh transformation
        # d/dx tanh(x) = 1 - tanh^2(x) = 1 - y^2 where y = tanh(x)
        log_jacobian = mx.sum(mx.log(1 - actions_clipped ** 2 + self.epsilon), axis=-1)
        
        return log_prob_unsquashed - log_jacobian
    
    def mode(self) -> MlxArray:
        """Return the mode (tanh of mean) of the distribution."""
        if self.mean is None:
            raise ValueError("Must call proba_distribution() first")
        
        return mx.tanh(self.mean)


def make_proba_distribution(
    action_space, 
    use_sde: bool = False, 
    dist_kwargs: Optional[dict] = None
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
    
    import gymnasium as gym
    
    if isinstance(action_space, gym.spaces.Discrete):
        # Discrete action space
        return CategoricalDistribution(action_space.n)
    elif isinstance(action_space, gym.spaces.Box):
        # Continuous action space
        action_dim = action_space.shape[0]
        return DiagGaussianDistribution(action_dim)
    elif isinstance(action_space, (gym.spaces.MultiBinary, gym.spaces.MultiDiscrete)):
        # MultiBinary, MultiDiscrete spaces not supported yet
        raise NotImplementedError(f"Action space {action_space} is not supported yet")
    else:
        raise NotImplementedError(f"Action space {action_space} is not supported")
