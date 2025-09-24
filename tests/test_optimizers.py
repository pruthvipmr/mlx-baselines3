"""
Tests for MLX optimizer adapters and gradient flow functionality.

Tests the core optimizer system required for section 2 of the spec:
- AdamAdapter and SGDAdapter functionality
- Gradient computation helpers
- Gradient clipping by global norm
- Learning rate schedules
- Optimizer state persistence
"""

import numpy as np
import pytest
import mlx.core as mx

from mlx_baselines3.common.optimizers import (
    AdamAdapter,
    SGDAdapter,
    create_optimizer_adapter,
    clip_grad_norm,
    compute_loss_and_grads,
)
from mlx_baselines3.common.schedules import (
    constant_schedule,
    linear_schedule,
    get_schedule_fn,
    exponential_schedule,
)


class TestAdamAdapter:
    """Test AdamAdapter functionality."""

    def test_adam_init_state(self):
        """Test that Adam initializes state correctly."""
        adapter = AdamAdapter(learning_rate=1e-3)

        # Create dummy parameters
        params = {
            "w1": mx.random.normal((3, 4)),
            "b1": mx.random.normal((4,)),
            "w2": mx.random.normal((4, 2)),
        }

        state = adapter.init_state(params)

        # Check state structure
        assert isinstance(state, dict)
        assert "step" in state
        assert "m" in state
        assert "v" in state
        assert state["step"] == 0

        # Check moment initialization
        for key in params:
            assert key in state["m"]
            assert key in state["v"]
            assert mx.array_equal(state["m"][key], mx.zeros_like(params[key]))
            assert mx.array_equal(state["v"][key], mx.zeros_like(params[key]))

    def test_adam_update_step(self):
        """Test that Adam updates parameters and state correctly."""
        adapter = AdamAdapter(learning_rate=1e-2)

        # Simple 1D parameter for easy verification
        params = {"w": mx.array([1.0, 2.0, 3.0])}
        grads = {"w": mx.array([0.1, 0.2, 0.3])}

        state = adapter.init_state(params)

        # Perform update
        new_params, new_state = adapter.update(params, grads, state)

        # Check that parameters changed
        assert not mx.array_equal(new_params["w"], params["w"])

        # Check that state updated
        assert new_state["step"] == 1
        assert not mx.array_equal(new_state["m"]["w"], state["m"]["w"])
        assert not mx.array_equal(new_state["v"]["w"], state["v"]["w"])

        # Check that moments are non-zero after update
        assert not mx.array_equal(new_state["m"]["w"], mx.zeros_like(params["w"]))
        assert not mx.array_equal(new_state["v"]["w"], mx.zeros_like(params["w"]))

    def test_adam_moments_evolution(self):
        """Test that Adam moments evolve correctly across multiple steps."""
        adapter = AdamAdapter(learning_rate=1e-2, betas=(0.9, 0.999))

        params = {"w": mx.array([1.0])}
        grads = {"w": mx.array([0.1])}

        state = adapter.init_state(params)

        # Perform multiple updates
        current_params = params
        current_state = state

        for step in range(5):
            current_params, current_state = adapter.update(
                current_params, grads, current_state
            )

            # Check step counter
            assert current_state["step"] == step + 1

            # Check that moments are evolving
            m_val = float(current_state["m"]["w"][0])
            v_val = float(current_state["v"]["w"][0])

            # Moments should be positive for positive gradients
            assert m_val > 0
            assert v_val > 0

    def test_adam_learning_rate_schedule(self):
        """Test Adam with learning rate schedule."""

        def lr_schedule(step):
            return 1e-2 * (0.9**step)

        adapter = AdamAdapter(learning_rate=lr_schedule)

        params = {"w": mx.array([1.0])}
        grads = {"w": mx.array([0.1])}
        state = adapter.init_state(params)

        # First update with initial LR
        new_params_1, new_state_1 = adapter.update(params, grads, state)

        # Second update with decayed LR
        new_params_2, new_state_2 = adapter.update(new_params_1, grads, new_state_1)

        # Parameter changes should be different due to LR decay
        delta_1 = float(new_params_1["w"][0] - params["w"][0])
        delta_2 = float(new_params_2["w"][0] - new_params_1["w"][0])

        # Second update should have smaller magnitude due to decayed LR
        assert abs(delta_2) < abs(delta_1)


class TestSGDAdapter:
    """Test SGDAdapter functionality."""

    def test_sgd_init_state(self):
        """Test SGD state initialization."""
        adapter = SGDAdapter(learning_rate=1e-2, momentum=0.9)

        params = {"w": mx.random.normal((2, 3))}
        state = adapter.init_state(params)

        assert isinstance(state, dict)
        assert "step" in state
        assert "m" in state
        assert state["step"] == 0

        # With momentum, should have velocity initialization
        assert "w" in state["m"]
        assert mx.array_equal(state["m"]["w"], mx.zeros_like(params["w"]))

    def test_sgd_update_simple(self):
        """Test simple SGD update without momentum."""
        adapter = SGDAdapter(learning_rate=1e-1, momentum=0.0)

        params = {"w": mx.array([1.0, 2.0])}
        grads = {"w": mx.array([0.1, 0.2])}
        state = adapter.init_state(params)

        new_params, new_state = adapter.update(params, grads, state)

        # Simple SGD: new_param = param - lr * grad
        expected = mx.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2])
        assert mx.allclose(new_params["w"], expected)
        assert new_state["step"] == 1

    def test_sgd_with_momentum(self):
        """Test SGD with momentum."""
        adapter = SGDAdapter(learning_rate=1e-1, momentum=0.9)

        params = {"w": mx.array([1.0])}
        grads = {"w": mx.array([0.1])}
        state = adapter.init_state(params)

        # First update (momentum starts at zero)
        new_params_1, new_state_1 = adapter.update(params, grads, state)

        # Second update (momentum should accumulate)
        new_params_2, new_state_2 = adapter.update(new_params_1, grads, new_state_1)

        # Check that momentum is accumulating
        momentum_1 = float(new_state_1["m"]["w"][0])
        momentum_2 = float(new_state_2["m"]["w"][0])

        assert momentum_1 > 0
        assert momentum_2 > momentum_1  # Momentum should accumulate


class TestOptimizerFactory:
    """Test optimizer factory function."""

    def test_create_adam_adapter(self):
        """Test creating Adam adapter via factory."""
        adapter = create_optimizer_adapter("adam", learning_rate=1e-3)
        assert isinstance(adapter, AdamAdapter)
        assert adapter.learning_rate == 1e-3

    def test_create_sgd_adapter(self):
        """Test creating SGD adapter via factory."""
        adapter = create_optimizer_adapter("sgd", learning_rate=1e-2, momentum=0.9)
        assert isinstance(adapter, SGDAdapter)
        assert adapter.learning_rate == 1e-2
        assert adapter.momentum == 0.9

    def test_unsupported_optimizer(self):
        """Test error for unsupported optimizer."""
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            create_optimizer_adapter("unsupported")


class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_clip_grad_norm_no_clipping(self):
        """Test gradient clipping when norm is below threshold."""
        grads = {
            "w1": mx.array([0.1, 0.2]),
            "w2": mx.array([0.3]),
        }

        clipped_grads, norm = clip_grad_norm(grads, max_norm=1.0)

        # Gradients should be unchanged
        for key in grads:
            assert mx.array_equal(clipped_grads[key], grads[key])

        # Norm should be computed correctly
        expected_norm = np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
        assert abs(norm - expected_norm) < 1e-6

    def test_clip_grad_norm_with_clipping(self):
        """Test gradient clipping when norm exceeds threshold."""
        grads = {
            "w1": mx.array([3.0, 4.0]),  # Norm = 5.0
        }

        clipped_grads, norm = clip_grad_norm(grads, max_norm=2.0)

        # Original norm should be 5.0
        assert abs(norm - 5.0) < 1e-6

        # Clipped gradients should have norm = 2.0
        clipped_norm = float(mx.sqrt(mx.sum(clipped_grads["w1"] ** 2)))
        assert abs(clipped_norm - 2.0) < 1e-6

        # Direction should be preserved
        original_direction = grads["w1"] / norm
        clipped_direction = clipped_grads["w1"] / clipped_norm
        assert mx.allclose(original_direction, clipped_direction)

    def test_clip_grad_norm_empty_grads(self):
        """Test gradient clipping with empty gradients."""
        grads = {}
        clipped_grads, norm = clip_grad_norm(grads, max_norm=1.0)

        assert clipped_grads == {}
        assert norm == 0.0

    def test_clip_grad_norm_none_values(self):
        """Test gradient clipping with None values."""
        grads = {
            "w1": mx.array([1.0, 1.0]),
            "w2": None,
        }

        clipped_grads, norm = clip_grad_norm(grads, max_norm=1.0)

        # None values should be preserved
        assert clipped_grads["w2"] is None

        # Norm should only include non-None gradients
        expected_norm = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
        assert abs(norm - expected_norm) < 1e-6


class TestGradientComputation:
    """Test centralized gradient computation."""

    def test_compute_loss_and_grads(self):
        """Test loss and gradient computation."""

        def simple_loss_fn(params):
            # Simple quadratic loss: sum((w - 1)^2)
            return mx.sum((params["w"] - 1.0) ** 2)

        params = {"w": mx.array([0.0, 2.0, 3.0])}

        loss, grads = compute_loss_and_grads(simple_loss_fn, params)

        # Check loss value
        expected_loss = (0 - 1) ** 2 + (2 - 1) ** 2 + (3 - 1) ** 2  # 1 + 1 + 4 = 6
        assert abs(float(loss) - expected_loss) < 1e-6

        # Check gradients: d/dw (w-1)^2 = 2(w-1)
        expected_grads = mx.array([2 * (0 - 1), 2 * (2 - 1), 2 * (3 - 1)])  # [-2, 2, 4]
        assert mx.allclose(grads["w"], expected_grads)

    def test_compute_loss_and_grads_multiple_params(self):
        """Test gradient computation with multiple parameters."""

        def loss_fn(params):
            return mx.sum(params["w1"] ** 2) + mx.sum(params["w2"] ** 2)

        params = {
            "w1": mx.array([1.0, 2.0]),
            "w2": mx.array([3.0]),
        }

        loss, grads = compute_loss_and_grads(loss_fn, params)

        # Check gradients
        assert mx.allclose(grads["w1"], 2 * params["w1"])  # d/dw1 w1^2 = 2*w1
        assert mx.allclose(grads["w2"], 2 * params["w2"])  # d/dw2 w2^2 = 2*w2


class TestSchedules:
    """Test learning rate schedules."""

    def test_constant_schedule(self):
        """Test constant schedule."""
        schedule = constant_schedule(0.01)

        # Should return constant value regardless of step
        assert schedule(0) == 0.01
        assert schedule(100) == 0.01
        assert schedule(1000) == 0.01

    def test_linear_schedule(self):
        """Test linear schedule."""
        schedule = linear_schedule(1.0, 0.0)

        # Should interpolate between initial and final values
        assert schedule(0.0) == 1.0
        assert schedule(1.0) == 0.0
        assert abs(schedule(0.5) - 0.5) < 1e-6

    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        schedule = exponential_schedule(1.0, decay_rate=0.9)

        # Should decay exponentially
        assert schedule(0) == 1.0
        assert abs(schedule(1) - 0.9) < 1e-6
        assert abs(schedule(2) - 0.81) < 1e-6

    def test_get_schedule_fn_float(self):
        """Test schedule creation from float."""
        schedule = get_schedule_fn(0.01)

        # Should create constant schedule
        assert schedule(0) == 0.01
        assert schedule(100) == 0.01

    def test_get_schedule_fn_callable(self):
        """Test schedule creation from callable."""

        def custom_schedule(step):
            return 0.01 * (0.95**step)

        schedule = get_schedule_fn(custom_schedule)

        # Should return the same function
        assert schedule is custom_schedule


class TestOptimizerIntegration:
    """Integration tests for optimizer system."""

    def test_adam_numerical_consistency(self):
        """Test numerical consistency of Adam updates."""
        # Set random seed for reproducibility
        mx.random.seed(42)

        adapter = AdamAdapter(learning_rate=1e-2, betas=(0.9, 0.999), eps=1e-8)

        # Create identical initial conditions
        params1 = {"w": mx.array([1.0, 2.0, 3.0])}
        params2 = {"w": mx.array([1.0, 2.0, 3.0])}

        grads = {"w": mx.array([0.1, 0.2, 0.3])}

        state1 = adapter.init_state(params1)
        state2 = adapter.init_state(params2)

        # Perform identical updates
        new_params1, new_state1 = adapter.update(params1, grads, state1)
        new_params2, new_state2 = adapter.update(params2, grads, state2)

        # Results should be identical
        assert mx.allclose(new_params1["w"], new_params2["w"])
        assert mx.allclose(new_state1["m"]["w"], new_state2["m"]["w"])
        assert mx.allclose(new_state1["v"]["w"], new_state2["v"]["w"])

    def test_optimizer_state_persistence(self):
        """Test that optimizer state can be saved and restored."""
        adapter = AdamAdapter(learning_rate=1e-2)

        params = {"w": mx.array([1.0, 2.0])}
        grads = {"w": mx.array([0.1, 0.2])}

        # Initial state
        state = adapter.init_state(params)

        # Perform several updates
        current_params = params
        current_state = state

        for _ in range(3):
            current_params, current_state = adapter.update(
                current_params, grads, current_state
            )

        # Save state
        saved_state = {
            "step": current_state["step"],
            "m": {k: v for k, v in current_state["m"].items()},
            "v": {k: v for k, v in current_state["v"].items()},
        }

        # Continue training
        next_params, next_state = adapter.update(current_params, grads, current_state)

        # Training from saved state should give same result
        restored_params, restored_state = adapter.update(
            current_params, grads, saved_state
        )

        assert mx.allclose(next_params["w"], restored_params["w"])
        assert next_state["step"] == restored_state["step"]
