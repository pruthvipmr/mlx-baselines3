"""
Test schedule functionality for MLX-Baselines3.

Tests hyperparameter schedules like learning rate, clip range, entropy coefficient, etc.
"""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES_PATH = PROJECT_ROOT / "mlx_baselines3" / "common" / "schedules.py"
spec = importlib.util.spec_from_file_location("mlx_schedules", SCHEDULES_PATH)
schedules = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(schedules)

constant_schedule = schedules.constant_schedule
linear_schedule = schedules.linear_schedule
piecewise_schedule = schedules.piecewise_schedule
exponential_schedule = schedules.exponential_schedule
cosine_annealing_schedule = schedules.cosine_annealing_schedule
get_schedule_fn = schedules.get_schedule_fn
schedule_from_string = schedules.schedule_from_string
apply_schedule_to_param = schedules.apply_schedule_to_param
make_progress_schedule = schedules.make_progress_schedule
get_linear_schedule = schedules.get_linear_schedule
get_constant_schedule = schedules.get_constant_schedule


class TestBasicSchedules:
    """Test basic schedule functions."""

    def test_constant_schedule(self):
        """Test constant schedule returns fixed value."""
        schedule = constant_schedule(0.01)

        # Should return constant value regardless of step
        assert schedule(0) == 0.01
        assert schedule(100) == 0.01
        assert schedule(1000) == 0.01

    def test_linear_schedule(self):
        """Test linear schedule interpolates correctly."""
        schedule = linear_schedule(1.0, 0.0)

        # Test SB3-style progress_remaining semantics
        assert schedule(1.0) == 1.0
        assert schedule(0.5) == 0.5
        assert schedule(0.0) == 0.0

    def test_linear_schedule_monotonic(self):
        """Test that linear schedule is monotonic."""
        schedule = linear_schedule(1.0, 0.0)

        # Should be monotonically decreasing as progress_remaining decreases
        prev_val = float("inf")
        for step in range(10, -1, -1):
            progress_remaining = step / 10.0
            val = schedule(progress_remaining)
            assert val <= prev_val
            prev_val = val

    def test_linear_schedule_bounds(self):
        """Test linear schedule handles out-of-bounds progress."""
        schedule = linear_schedule(1.0, 0.0)

        # Should clamp to bounds
        assert schedule(1.5) == pytest.approx(1.0)  # Clamped to start
        assert schedule(-0.5) == pytest.approx(0.0)  # Clamped to end

    def test_piecewise_schedule(self):
        """Test piecewise schedule."""
        endpoints = [0.0, 0.5, 1.0]
        values = [1.0, 0.5, 0.0]
        schedule = piecewise_schedule(endpoints, values)

        # Test exact endpoints
        assert schedule(1.0) == pytest.approx(1.0)
        assert schedule(0.5) == pytest.approx(0.5)
        assert schedule(0.0) == pytest.approx(0.0)

        # Test interpolation within segments
        assert schedule(0.75) == pytest.approx(0.75)
        assert schedule(0.25) == pytest.approx(0.25)

    def test_piecewise_schedule_constant(self):
        """Test piecewise schedule with constant interpolation."""
        endpoints = [0.0, 0.5, 1.0]
        values = [1.0, 0.5, 0.0]
        schedule = piecewise_schedule(endpoints, values, interpolation="constant")

        # Should use step function (constant returns left value of interval)
        assert schedule(1.0) == pytest.approx(1.0)
        assert schedule(0.6) == pytest.approx(1.0)  # Still in first segment
        assert schedule(0.5) == pytest.approx(1.0)  # Boundary uses left value
        assert schedule(0.4) == pytest.approx(0.5)  # In second segment
        assert schedule(0.1) == pytest.approx(0.5)  # Still in second segment
        assert schedule(0.0) == pytest.approx(0.0)  # Final endpoint

    def test_piecewise_schedule_validation(self):
        """Test piecewise schedule input validation."""
        # Length mismatch
        with pytest.raises(ValueError, match="same length"):
            piecewise_schedule([0.0, 1.0], [1.0, 0.5, 0.0])

        # Wrong endpoints
        with pytest.raises(ValueError, match="start at 0.0 and end at 1.0"):
            piecewise_schedule([0.1, 1.0], [1.0, 0.0])

        with pytest.raises(ValueError, match="start at 0.0 and end at 1.0"):
            piecewise_schedule([0.0, 0.9], [1.0, 0.0])

    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        schedule = exponential_schedule(1.0, decay_rate=0.9)

        # Should decay exponentially
        assert schedule(0) == 1.0
        assert abs(schedule(1) - 0.9) < 1e-6
        assert abs(schedule(2) - 0.81) < 1e-6

        # Should be monotonically decreasing
        prev_val = float("inf")
        for step in range(10):
            val = schedule(step)
            assert val <= prev_val
            prev_val = val

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule."""
        schedule = cosine_annealing_schedule(1.0, min_value=0.0, cycle_length=100)

        # Test cycle behavior
        assert abs(schedule(0) - 1.0) < 1e-6  # Maximum at start
        assert abs(schedule(50) - 0.5) < 1e-6  # Midpoint value
        assert abs(schedule(100) - 1.0) < 1e-6  # Back to maximum at cycle end


class TestScheduleFromString:
    """Test string-based schedule creation."""

    def test_constant_schedule_string(self):
        """Test creating constant schedule from string."""
        schedule = schedule_from_string("constant", default_value=0.01)

        assert schedule(1.0) == pytest.approx(0.01)
        assert schedule(0.5) == pytest.approx(0.01)
        assert schedule(0.0) == pytest.approx(0.01)

    def test_linear_schedule_string(self):
        """Test creating linear schedule from string."""
        schedule = schedule_from_string("linear", default_value=1.0)

        assert schedule(1.0) == pytest.approx(1.0)
        assert schedule(0.0) == pytest.approx(0.0)
        assert schedule(0.5) == pytest.approx(0.5)

    def test_get_schedule_fn_strings(self):
        """Test get_schedule_fn with string inputs."""
        # Linear with initial value
        schedule = get_schedule_fn("linear_0.001")
        assert schedule(1.0) == pytest.approx(0.001)
        assert schedule(0.0) == pytest.approx(0.0)

        # Linear with initial and final values
        schedule = get_schedule_fn("linear_0.001_0.0001")
        assert schedule(1.0) == pytest.approx(0.001)
        assert schedule(0.0) == pytest.approx(0.0001)

        # Piecewise schedule
        schedule = get_schedule_fn("piecewise_0.0:0.1_0.5:0.05_1.0:0.01")
        assert schedule(1.0) == pytest.approx(0.1)
        assert schedule(0.5) == pytest.approx(0.05)
        assert schedule(0.0) == pytest.approx(0.01)

    def test_get_schedule_fn_validation(self):
        """Test get_schedule_fn input validation."""
        # Invalid linear format
        with pytest.raises(ValueError, match="Invalid linear schedule"):
            get_schedule_fn("linear_invalid")

        # Invalid piecewise format
        with pytest.raises(
            ValueError, match="Piecewise schedule needs at least 2 points"
        ):
            get_schedule_fn("piecewise_invalid")

        # Unsupported string
        with pytest.raises(ValueError, match="Unsupported schedule string"):
            get_schedule_fn("unknown_schedule")

        # Unsupported type
        with pytest.raises(ValueError, match="Unsupported schedule type"):
            get_schedule_fn([1, 2, 3])


class TestApplyScheduleToParam:
    """Test apply_schedule_to_param utility function."""

    def test_float_param(self):
        """Test with float parameter."""
        result = apply_schedule_to_param(0.01, 0.5)
        assert result == 0.01

    def test_callable_param(self):
        """Test with callable parameter."""

        def schedule_fn(progress_remaining: float) -> float:
            return progress_remaining**2

        result = apply_schedule_to_param(schedule_fn, 0.3)
        assert result == pytest.approx(0.09)

    def test_string_param(self):
        """Test with string parameter."""
        result = apply_schedule_to_param("linear_0.001", 0.5)
        assert result == pytest.approx(0.0005)

    def test_string_param_with_default(self):
        """Test string parameter with default value."""
        result = apply_schedule_to_param("constant", 0.5, default_value=0.01)
        assert result == pytest.approx(0.01)

        result = apply_schedule_to_param("linear", 0.5, default_value=1.0)
        assert result == pytest.approx(0.5)


class TestProgressSchedule:
    """Test progress-based schedule conversion."""

    def test_make_progress_schedule(self):
        """Test converting progress schedule to step schedule."""
        progress_schedule = linear_schedule(1.0, 0.0)
        step_schedule = make_progress_schedule(progress_schedule)

        # Test with different total steps
        assert step_schedule(0, 100) == pytest.approx(1.0)
        assert step_schedule(50, 100) == pytest.approx(0.5)
        assert step_schedule(100, 100) == pytest.approx(0.0)

        # Test edge case: zero total steps
        assert step_schedule(10, 0) == pytest.approx(1.0)

        # Test edge case: more steps than total
        assert step_schedule(150, 100) == 0.0


class TestScheduleIntegration:
    """Test schedule integration with PPO."""

    def test_consistent_clip_ranges(self):
        """Test that clip ranges are consistent across epochs."""
        # Test linear decay
        schedule = linear_schedule(0.2, 0.1)

        values = [schedule(1.0 - i / 9.0) for i in range(10)]

        # Should be monotonically decreasing
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

        # Should respect bounds
        assert values[0] == 0.2
        assert values[-1] == 0.1

    def test_monotonic_lr_decay(self):
        """Test monotonic learning rate decay."""
        schedule = linear_schedule(1e-3, 1e-5)

        # Test over training progress
        prev_lr = float("inf")
        for i in range(20):
            progress_remaining = 1.0 - i / 19.0
            lr = schedule(progress_remaining)
            assert lr <= prev_lr
            prev_lr = lr

        # Check bounds
        assert schedule(1.0) == pytest.approx(1e-3)
        assert schedule(0.0) == pytest.approx(1e-5)


class TestConvenienceFunctions:
    """Test convenience functions for common schedules."""

    def test_get_linear_schedule(self):
        """Test get_linear_schedule convenience function."""
        schedule = get_linear_schedule(1.0, 0.0)

        assert schedule(1.0) == 1.0
        assert schedule(0.0) == 0.0
        assert schedule(0.5) == pytest.approx(0.5)

    def test_get_constant_schedule(self):
        """Test get_constant_schedule convenience function."""
        schedule = get_constant_schedule(0.01)

        assert schedule(0) == 0.01
        assert schedule(100) == 0.01


class TestScheduleNumericStability:
    """Test numeric stability of schedule functions."""

    def test_linear_schedule_precision(self):
        """Test linear schedule maintains precision with small values."""
        schedule = linear_schedule(1e-8, 1e-10)

        # Should maintain precision even with very small values
        result = schedule(0.5)
        expected = 5.05e-9  # (1e-8 + 1e-10) / 2
        assert abs(result - expected) < 1e-12

    def test_piecewise_schedule_edge_cases(self):
        """Test piecewise schedule handles edge cases."""
        endpoints = [0.0, 1.0]
        values = [1.0, 0.0]
        schedule = piecewise_schedule(endpoints, values)

        # Test very close to boundaries
        assert abs(schedule(1.0 - 1e-10) - 1.0) < 1e-6
        assert abs(schedule(1e-10) - 0.0) < 1e-6


# Example tests showing SB3 compatibility
class TestSB3Compatibility:
    """Test compatibility with SB3-style schedule usage."""

    def test_ppo_clip_range_schedule(self):
        """Test PPO-style clip range schedule."""
        # Common PPO usage: linear decay from 0.2 to 0.0
        clip_schedule = linear_schedule(0.2, 0.0)

        # Simulate training progress
        for i in range(10):
            progress_remaining = 1.0 - i / 9.0
            clip_value = clip_schedule(progress_remaining)
            assert 0.0 <= clip_value <= 0.2

    def test_entropy_coefficient_schedule(self):
        """Test entropy coefficient schedule."""
        # Common usage: decay entropy coefficient over time
        ent_schedule = linear_schedule(0.01, 0.0)

        # Should start high and decay to 0
        assert ent_schedule(1.0) == 0.01
        assert ent_schedule(0.0) == 0.0

        # Should be monotonic
        prev_val = float("inf")
        for i in range(11):
            progress_remaining = 1.0 - i / 10.0
            val = ent_schedule(progress_remaining)
            assert val <= prev_val
            prev_val = val
