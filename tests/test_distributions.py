"""
Tests for action distributions.
"""

import pytest
import mlx.core as mx
import gymnasium as gym
import math

from mlx_baselines3.common.distributions import (
    Distribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    make_proba_distribution,
)


class TestDistribution:
    """Test Distribution abstract base class."""
    
    def test_abstract_distribution(self):
        """Test that Distribution is abstract."""
        with pytest.raises(TypeError):
            Distribution()


class TestCategoricalDistribution:
    """Test CategoricalDistribution for discrete actions."""
    
    def test_initialization(self):
        """Test categorical distribution initialization."""
        dist = CategoricalDistribution(action_dim=4)
        
        assert dist.action_dim == 4
        assert dist.logits is None
        assert dist.probs is None
    
    def test_proba_distribution(self):
        """Test probability distribution setup."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 2.0, 0.5], [0.0, 1.0, -1.0]])
        
        dist.proba_distribution(logits)
        
        assert dist.logits is not None
        assert dist.probs is not None
        assert dist.logits.shape == (2, 3)
        assert dist.probs.shape == (2, 3)
        
        # Check probabilities sum to 1
        prob_sums = mx.sum(dist.probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones(2), atol=1e-6)
    
    def test_mode(self):
        """Test mode (most likely action)."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 3.0, 0.5], [2.0, 1.0, 0.0]])
        
        dist.proba_distribution(logits)
        mode = dist.mode()
        
        expected = mx.array([1, 0])  # Indices of max logits
        assert mx.array_equal(mode, expected)
    
    def test_sample_deterministic(self):
        """Test deterministic sampling (should return mode)."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 3.0, 0.5]])
        
        dist.proba_distribution(logits)
        action = dist.sample(deterministic=True)
        mode = dist.mode()
        
        assert mx.array_equal(action, mode)
    
    def test_sample_stochastic(self):
        """Test stochastic sampling."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 3.0, 0.5]])
        
        dist.proba_distribution(logits)
        action = dist.sample(deterministic=False)
        
        assert action.shape == (1,)
        assert 0 <= action[0] < 3
    
    def test_log_prob(self):
        """Test log probability computation."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 2.0, 0.5]])
        actions = mx.array([1])
        
        dist.proba_distribution(logits)
        log_prob = dist.log_prob(actions)
        
        assert log_prob.shape == (1,)
        
        # Verify log probability is reasonable (should be largest for action 1)
        log_probs_all = mx.log(dist.probs[0])
        assert mx.allclose(log_prob[0], log_probs_all[1], atol=1e-6)
    
    def test_entropy(self):
        """Test entropy computation."""
        dist = CategoricalDistribution(action_dim=3)
        
        # Uniform distribution should have maximum entropy
        logits = mx.array([[0.0, 0.0, 0.0]])
        dist.proba_distribution(logits)
        entropy_uniform = dist.entropy()
        
        # Peaked distribution should have lower entropy
        logits = mx.array([[10.0, 0.0, 0.0]])
        dist.proba_distribution(logits)
        entropy_peaked = dist.entropy()
        
        assert entropy_uniform[0] > entropy_peaked[0]
    
    def test_convenience_methods(self):
        """Test convenience methods."""
        dist = CategoricalDistribution(action_dim=3)
        logits = mx.array([[1.0, 2.0, 0.5]])
        actions = mx.array([1])
        
        # Test actions_from_params
        sampled_actions = dist.actions_from_params(logits, deterministic=True)
        assert sampled_actions.shape == (1,)
        
        # Test log_prob_from_params
        log_prob = dist.log_prob_from_params(logits, actions)
        assert log_prob.shape == (1,)
    
    def test_error_without_proba_distribution(self):
        """Test that methods raise error without calling proba_distribution first."""
        dist = CategoricalDistribution(action_dim=3)
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.sample()
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.log_prob(mx.array([1]))
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.entropy()
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.mode()


class TestDiagGaussianDistribution:
    """Test DiagGaussianDistribution for continuous actions."""
    
    def test_initialization(self):
        """Test Gaussian distribution initialization."""
        dist = DiagGaussianDistribution(action_dim=2)
        
        assert dist.action_dim == 2
        assert dist.mean is None
        assert dist.log_std is None
        assert dist.std is None
    
    def test_proba_distribution(self):
        """Test probability distribution setup."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[1.0, -0.5], [0.0, 2.0]])
        log_std = mx.array([0.0, -1.0])  # std = [1.0, ~0.37]
        
        dist.proba_distribution(mean, log_std)
        
        assert dist.mean is not None
        assert dist.log_std is not None
        assert dist.std is not None
        assert dist.mean.shape == (2, 2)
        assert dist.log_std.shape == (2,)
        assert dist.std.shape == (2,)
        
        expected_std = mx.exp(log_std)
        assert mx.allclose(dist.std, expected_std)
    
    def test_mode(self):
        """Test mode (mean) of Gaussian distribution."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[1.0, -0.5]])
        log_std = mx.array([0.0, 0.0])
        
        dist.proba_distribution(mean, log_std)
        mode = dist.mode()
        
        assert mx.allclose(mode, mean)
    
    def test_sample_deterministic(self):
        """Test deterministic sampling (should return mean)."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[1.0, -0.5]])
        log_std = mx.array([0.0, 0.0])
        
        dist.proba_distribution(mean, log_std)
        action = dist.sample(deterministic=True)
        
        assert mx.allclose(action, mean)
    
    def test_sample_stochastic(self):
        """Test stochastic sampling."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[0.0, 0.0]])
        log_std = mx.array([0.0, 0.0])  # std = 1.0
        
        dist.proba_distribution(mean, log_std)
        action = dist.sample(deterministic=False)
        
        assert action.shape == (1, 2)
    
    def test_log_prob(self):
        """Test log probability computation."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[0.0, 0.0]])
        log_std = mx.array([0.0, 0.0])  # std = 1.0
        actions = mx.array([[0.0, 0.0]])  # At mean
        
        dist.proba_distribution(mean, log_std)
        log_prob = dist.log_prob(actions)
        
        assert log_prob.shape == (1,)
        
        # Log prob at mean should be maximum
        actions_off = mx.array([[1.0, 1.0]])
        log_prob_off = dist.log_prob(actions_off)
        
        assert log_prob[0] > log_prob_off[0]
    
    def test_entropy(self):
        """Test entropy computation."""
        dist = DiagGaussianDistribution(action_dim=2)
        
        # Lower variance should have lower entropy
        mean = mx.array([[0.0, 0.0]])
        log_std_low = mx.array([-1.0, -1.0])  # Low std
        dist.proba_distribution(mean, log_std_low)
        entropy_low = dist.entropy()
        
        # Higher variance should have higher entropy
        log_std_high = mx.array([1.0, 1.0])  # High std
        dist.proba_distribution(mean, log_std_high)
        entropy_high = dist.entropy()
        
        assert entropy_high > entropy_low
    
    def test_convenience_methods(self):
        """Test convenience methods."""
        dist = DiagGaussianDistribution(action_dim=2)
        mean = mx.array([[1.0, -0.5]])
        log_std = mx.array([0.0, 0.0])
        actions = mx.array([[1.0, -0.5]])
        
        # Test actions_from_params
        sampled_actions = dist.actions_from_params(mean, log_std, deterministic=True)
        assert mx.allclose(sampled_actions, mean)
        
        # Test log_prob_from_params
        log_prob = dist.log_prob_from_params(mean, log_std, actions)
        assert log_prob.shape == (1,)
    
    def test_error_without_proba_distribution(self):
        """Test that methods raise error without calling proba_distribution first."""
        dist = DiagGaussianDistribution(action_dim=2)
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.sample()
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.log_prob(mx.array([[1.0, 1.0]]))
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.entropy()
        
        with pytest.raises(ValueError, match="Must call proba_distribution"):
            dist.mode()


class TestSquashedDiagGaussianDistribution:
    """Test SquashedDiagGaussianDistribution."""
    
    def test_initialization(self):
        """Test squashed Gaussian distribution initialization."""
        dist = SquashedDiagGaussianDistribution(action_dim=2)
        
        assert dist.action_dim == 2
        assert dist.epsilon == 1e-6
    
    def test_sample_bounds(self):
        """Test that sampled actions are bounded in [-1, 1]."""
        dist = SquashedDiagGaussianDistribution(action_dim=2)
        mean = mx.array([[5.0, -5.0]])  # Large values
        log_std = mx.array([1.0, 1.0])
        
        dist.proba_distribution(mean, log_std)
        actions = dist.sample(deterministic=False)
        
        assert mx.all(actions >= -1.0)
        assert mx.all(actions <= 1.0)
    
    def test_deterministic_sample(self):
        """Test deterministic sampling returns tanh of mean."""
        dist = SquashedDiagGaussianDistribution(action_dim=2)
        mean = mx.array([[2.0, -1.0]])
        log_std = mx.array([0.0, 0.0])
        
        dist.proba_distribution(mean, log_std)
        actions = dist.sample(deterministic=True)
        expected = mx.tanh(mean)
        
        assert mx.allclose(actions, expected)
    
    def test_mode(self):
        """Test mode returns tanh of mean."""
        dist = SquashedDiagGaussianDistribution(action_dim=2)
        mean = mx.array([[2.0, -1.0]])
        log_std = mx.array([0.0, 0.0])
        
        dist.proba_distribution(mean, log_std)
        mode = dist.mode()
        expected = mx.tanh(mean)
        
        assert mx.allclose(mode, expected)
    
    def test_log_prob_jacobian_correction(self):
        """Test that log prob includes Jacobian correction."""
        dist = SquashedDiagGaussianDistribution(action_dim=1)
        mean = mx.array([[0.0]])
        log_std = mx.array([0.0])
        actions = mx.array([[0.5]])  # Valid action in [-1, 1]
        
        dist.proba_distribution(mean, log_std)
        log_prob_squashed = dist.log_prob(actions)
        
        # Compare with unsquashed distribution
        unsquashed_dist = DiagGaussianDistribution(action_dim=1)
        unsquashed_actions = mx.arctanh(actions)
        unsquashed_dist.proba_distribution(mean, log_std)
        log_prob_unsquashed = unsquashed_dist.log_prob(unsquashed_actions)
        
        # Squashed log prob should be less due to Jacobian correction (in absolute value)
        # The Jacobian correction reduces the log probability
        # For this test, we just check that the correction is applied (result is different)
        assert not mx.allclose(log_prob_squashed, log_prob_unsquashed, atol=1e-3)


class TestMakeProbaDistribution:
    """Test make_proba_distribution factory function."""
    
    def test_discrete_action_space(self):
        """Test creating distribution for discrete action space."""
        action_space = gym.spaces.Discrete(5)
        dist = make_proba_distribution(action_space)
        
        assert isinstance(dist, CategoricalDistribution)
        assert dist.action_dim == 5
    
    def test_continuous_action_space(self):
        """Test creating distribution for continuous action space."""
        action_space = gym.spaces.Box(-1, 1, (3,))
        dist = make_proba_distribution(action_space)
        
        assert isinstance(dist, DiagGaussianDistribution)
        assert dist.action_dim == 3
    
    def test_unsupported_action_space(self):
        """Test error for unsupported action space."""
        action_space = gym.spaces.MultiBinary(3)
        
        with pytest.raises(NotImplementedError):
            make_proba_distribution(action_space)
