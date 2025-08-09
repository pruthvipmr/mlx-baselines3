"""
Tests for policy networks.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym

from mlx_baselines3.common.policies import (
    BasePolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
    MlpPolicy,
)
from mlx_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
from mlx_baselines3.common.distributions import CategoricalDistribution, DiagGaussianDistribution


def constant_lr_schedule(progress_remaining: float) -> float:
    """Constant learning rate schedule for testing."""
    return 0.001


class TestBasePolicy:
    """Test BasePolicy abstract base class."""
    
    def test_abstract_policy(self):
        """Test that BasePolicy is abstract."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        with pytest.raises(TypeError):
            BasePolicy(observation_space, action_space, constant_lr_schedule)


class TestActorCriticPolicy:
    """Test ActorCriticPolicy implementation."""
    
    def test_initialization_discrete_actions(self):
        """Test policy initialization with discrete actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(3)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[32, 16],
        )
        
        assert policy.observation_space == observation_space
        assert policy.action_space == action_space
        assert isinstance(policy.action_dist, CategoricalDistribution)
        assert policy.action_dist.action_dim == 3
    
    def test_initialization_continuous_actions(self):
        """Test policy initialization with continuous actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Box(-2, 2, (2,))
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[32, 16],
        )
        
        assert policy.observation_space == observation_space
        assert policy.action_space == action_space
        assert isinstance(policy.action_dist, DiagGaussianDistribution)
        assert policy.action_dist.action_dim == 2
    
    def test_initialization_with_dict_net_arch(self):
        """Test policy initialization with dictionary network architecture."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        net_arch = dict(pi=[64, 32], vf=[64, 32])
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=net_arch,
        )
        
        assert policy.net_arch == net_arch
    
    def test_shared_features_extractor(self):
        """Test policy with shared features extractor."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            share_features_extractor=True,
        )
        
        assert policy.pi_features_extractor is policy.vf_features_extractor
    
    def test_separate_features_extractors(self):
        """Test policy with separate features extractors."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            share_features_extractor=False,
        )
        
        assert policy.pi_features_extractor is not policy.vf_features_extractor
    
    def test_forward_pass_discrete(self):
        """Test forward pass with discrete actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(3)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        actions, values, log_probs = policy.forward(obs, deterministic=False)
        
        assert actions.shape == (2,)
        assert values.shape == (2,)
        assert log_probs.shape == (2,)
        
        # Actions should be valid indices
        assert mx.all(actions >= 0)
        assert mx.all(actions < 3)
    
    def test_forward_pass_continuous(self):
        """Test forward pass with continuous actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Box(-2, 2, (2,))
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        actions, values, log_probs = policy.forward(obs, deterministic=False)
        
        assert actions.shape == (2, 2)
        assert values.shape == (2,)
        assert log_probs.shape == (2,)
    
    def test_predict_values(self):
        """Test value prediction."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((3, 4))
        values = policy.predict_values(obs)
        
        assert values.shape == (3,)
    
    def test_get_distribution_discrete(self):
        """Test getting action distribution for discrete actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(3)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        distribution = policy.get_distribution(obs)
        
        assert isinstance(distribution, CategoricalDistribution)
        assert distribution.logits is not None
        assert distribution.logits.shape == (2, 3)
    
    def test_get_distribution_continuous(self):
        """Test getting action distribution for continuous actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Box(-2, 2, (2,))
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        distribution = policy.get_distribution(obs)
        
        assert isinstance(distribution, DiagGaussianDistribution)
        assert distribution.mean is not None
        assert distribution.mean.shape == (2, 2)
        assert distribution.log_std is not None
    
    def test_evaluate_actions_discrete(self):
        """Test action evaluation for discrete actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(3)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        actions = mx.array([0, 2])
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        
        assert values.shape == (2,)
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
    
    def test_evaluate_actions_continuous(self):
        """Test action evaluation for continuous actions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Box(-2, 2, (2,))
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((2, 4))
        actions = mx.random.normal((2, 2))
        
        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        
        assert values.shape == (2,)
        assert log_probs.shape == (2,)
        assert entropy.shape == (2,)
    
    def test_predict_single_observation(self):
        """Test prediction with single observation."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((4,))  # Single observation
        actions, state = policy.predict(obs, deterministic=True)
        
        assert actions.shape == ()  # Scalar action
        assert state is None
    
    def test_predict_batch_observations(self):
        """Test prediction with batch of observations."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((3, 4))  # Batch of observations
        actions, state = policy.predict(obs, deterministic=True)
        
        assert actions.shape == (3,)
        assert state is None
    
    def test_deterministic_vs_stochastic_prediction(self):
        """Test deterministic vs stochastic prediction."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(3)
        
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
        )
        
        obs = mx.random.normal((1, 4))
        
        # Deterministic prediction should be consistent
        action1, _ = policy.predict(obs, deterministic=True)
        action2, _ = policy.predict(obs, deterministic=True)
        assert mx.array_equal(action1, action2)
    
    def test_different_activation_functions(self):
        """Test policy with different activation functions."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        # Test with ReLU activation
        policy_relu = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
            activation_fn=nn.ReLU,
        )
        
        # Test with Tanh activation
        policy_tanh = ActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[8],
            activation_fn=nn.Tanh,
        )
        
        obs = mx.random.normal((1, 4))
        
        # Both should work without errors
        actions1, _, _ = policy_relu.forward(obs)
        actions2, _, _ = policy_tanh.forward(obs)
        
        assert actions1.shape == (1,)
        assert actions2.shape == (1,)


class TestMultiInputActorCriticPolicy:
    """Test MultiInputActorCriticPolicy for dictionary observations."""
    
    def test_initialization(self):
        """Test multi-input policy initialization."""
        observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(-1, 1, (4,)),
            'achieved_goal': gym.spaces.Box(-1, 1, (2,)),
            'desired_goal': gym.spaces.Box(-1, 1, (2,)),
        })
        action_space = gym.spaces.Discrete(3)
        
        policy = MultiInputActorCriticPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
            net_arch=[16],
        )
        
        assert policy.observation_space == observation_space
        assert policy.action_space == action_space
        assert isinstance(policy.features_extractor, MlpExtractor)


class TestMlpPolicy:
    """Test MlpPolicy alias."""
    
    def test_mlp_policy_alias(self):
        """Test that MlpPolicy is an alias for ActorCriticPolicy."""
        assert MlpPolicy is ActorCriticPolicy
    
    def test_mlp_policy_creation(self):
        """Test creating MlpPolicy."""
        observation_space = gym.spaces.Box(-1, 1, (4,))
        action_space = gym.spaces.Discrete(2)
        
        policy = MlpPolicy(
            observation_space,
            action_space,
            constant_lr_schedule,
        )
        
        assert isinstance(policy, ActorCriticPolicy)
