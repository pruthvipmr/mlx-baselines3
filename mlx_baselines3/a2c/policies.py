"""
A2C-specific policy implementations using MLX.

This module provides policy classes tailored for the A2C algorithm,
including the main A2CPolicy class and convenience aliases.
"""

from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import mlx.nn as nn

from mlx_baselines3.common.policies import ActorCriticPolicy
from mlx_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from mlx_baselines3.common.type_aliases import Schedule


class A2CPolicy(ActorCriticPolicy):
    """
    Policy class for A2C algorithm using MLX.
    
    This class extends the base ActorCriticPolicy with A2C-specific configurations
    and optimizations.
    
    Args:
        observation_space: Observation space
        action_space: Action space  
        lr_schedule: Learning rate schedule
        net_arch: The specification of the policy and value networks
        activation_fn: Activation function
        ortho_init: Whether to use orthogonal initialization
        use_sde: Whether to use State Dependent Exploration
        log_std_init: Initial value for the log standard deviation
        full_std: Whether to use (n_features x n_actions) parameters for the std 
                 instead of only (n_features,) when using gSDE
        use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
                  positive standard deviations (cf paper). It allows to keep variance
                  above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        squash_output: Whether to squash the output using a tanh function,
                      this allows to ensure boundaries when using continuous actions.
        features_extractor_class: Features extractor to use
        features_extractor_kwargs: Keyword arguments for features extractor
        share_features_extractor: Whether to share the features extractor between actor and critic
        normalize_images: Whether to normalize images or not,
                         dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use
        optimizer_kwargs: Additional keyword arguments for the optimizer
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
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
        optimizer_class: Type = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Set default optimizer if not provided
        if optimizer_class is None:
            import mlx.optimizers as optim
            optimizer_class = optim.Adam
            
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])
            
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


# Convenience aliases for easier import
MlpPolicy = A2CPolicy


class CnnPolicy(A2CPolicy):
    """
    CNN policy class for A2C when using image observations.
    
    This policy uses a convolutional neural network for feature extraction
    from image observations.
    """
    
    def __init__(self, *args, **kwargs):
        # For now, use MlpExtractor as fallback until NatureCNN is implemented
        from mlx_baselines3.common.torch_layers import MlpExtractor
        
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=MlpExtractor,
            normalize_images=True,
        )


class MultiInputPolicy(A2CPolicy):
    """
    MultiInput policy class for A2C when using dictionary observations.
    
    This policy handles multiple input types (e.g., images + vectors)
    through a multi-input features extractor.
    """
    
    def __init__(self, *args, **kwargs):
        # For now, use MlpExtractor as fallback until CombinedExtractor is implemented
        from mlx_baselines3.common.torch_layers import MlpExtractor
        
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=MlpExtractor,
        )


# Register policy classes for string-based instantiation
A2C_POLICY_CLASSES = {
    "MlpPolicy": MlpPolicy,
    "CnnPolicy": CnnPolicy, 
    "MultiInputPolicy": MultiInputPolicy,
}


def get_a2c_policy_class(policy_name: str) -> Type[A2CPolicy]:
    """
    Get A2C policy class by name.
    
    Args:
        policy_name: Name of the policy class
        
    Returns:
        Policy class
        
    Raises:
        ValueError: If policy name is not recognized
    """
    if policy_name in A2C_POLICY_CLASSES:
        return A2C_POLICY_CLASSES[policy_name]
    else:
        raise ValueError(f"Unknown policy: {policy_name}. Available policies: {list(A2C_POLICY_CLASSES.keys())}")
