"""
Learning rate and hyperparameter schedules for MLX-Baselines3.

Provides schedule functions compatible with SB3 API for learning rates,
clip ranges, and other hyperparameters that change during training.
"""

from typing import Union, Callable
import numpy as np


def constant_schedule(value: float) -> Callable[[int], float]:
    """
    Create a constant schedule function.
    
    Args:
        value: Constant value to return
        
    Returns:
        Schedule function that always returns the constant value
    """
    def schedule_fn(step: int) -> float:
        return value
    
    return schedule_fn


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Create a linear schedule function that interpolates between initial and final values.
    
    Args:
        initial_value: Starting value (at progress=0.0)
        final_value: Ending value (at progress=1.0)
        
    Returns:
        Schedule function that takes progress [0.0, 1.0] and returns interpolated value
    """
    def schedule_fn(progress: float) -> float:
        # Ensure progress is in [0, 1]
        progress = max(0.0, min(1.0, progress))
        return initial_value + progress * (final_value - initial_value)
    
    return schedule_fn


def piecewise_schedule(
    endpoints: list,
    values: list,
    interpolation: str = "linear"
) -> Callable[[float], float]:
    """
    Create a piecewise schedule function.
    
    Args:
        endpoints: List of progress points [0.0, ..., 1.0] where values change
        values: List of values at each endpoint (same length as endpoints)
        interpolation: Type of interpolation ("linear", "constant")
        
    Returns:
        Schedule function that interpolates between the specified points
    """
    if len(endpoints) != len(values):
        raise ValueError("endpoints and values must have the same length")
    
    if endpoints[0] != 0.0 or endpoints[-1] != 1.0:
        raise ValueError("endpoints must start at 0.0 and end at 1.0")
    
    def schedule_fn(progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        
        # Find the right interval
        for i in range(len(endpoints) - 1):
            if progress <= endpoints[i + 1]:
                if interpolation == "constant":
                    return values[i]
                elif interpolation == "linear":
                    # Linear interpolation between endpoints[i] and endpoints[i+1]
                    t = (progress - endpoints[i]) / (endpoints[i + 1] - endpoints[i])
                    return values[i] + t * (values[i + 1] - values[i])
        
        return values[-1]
    
    return schedule_fn


def exponential_schedule(
    initial_value: float,
    decay_rate: float = 0.95
) -> Callable[[int], float]:
    """
    Create an exponential decay schedule.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate per step (multiplied each step)
        
    Returns:
        Schedule function that applies exponential decay
    """
    def schedule_fn(step: int) -> float:
        return initial_value * (decay_rate ** step)
    
    return schedule_fn


def cosine_annealing_schedule(
    initial_value: float,
    min_value: float = 0.0,
    cycle_length: int = 1000
) -> Callable[[int], float]:
    """
    Create a cosine annealing schedule.
    
    Args:
        initial_value: Maximum value
        min_value: Minimum value
        cycle_length: Length of one complete cycle
        
    Returns:
        Schedule function that follows cosine annealing pattern
    """
    def schedule_fn(step: int) -> float:
        cycle_position = step % cycle_length
        progress = cycle_position / cycle_length
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return min_value + (initial_value - min_value) * cosine_factor
    
    return schedule_fn


def get_schedule_fn(value: Union[float, str, Callable]) -> Callable:
    """
    Convert various schedule specifications to callable schedule functions.
    
    This function provides SB3-compatible schedule creation, supporting:
    - Float values (converted to constant schedules)
    - String specifications (e.g., "linear", "constant")
    - Callable functions (returned as-is)
    
    Args:
        value: Schedule specification (float, string, or callable)
        
    Returns:
        Callable schedule function
        
    Raises:
        ValueError: If the schedule specification is not supported
    """
    if callable(value):
        return value
    elif isinstance(value, (int, float)):
        return constant_schedule(float(value))
    elif isinstance(value, str):
        if value == "constant":
            raise ValueError("Constant schedule requires a numeric value")
        elif value.startswith("linear"):
            # Parse linear schedule specification like "linear_0.001"
            parts = value.split("_")
            if len(parts) == 2:
                try:
                    initial_value = float(parts[1])
                    return linear_schedule(initial_value)
                except ValueError:
                    pass
            raise ValueError(f"Invalid linear schedule specification: {value}")
        else:
            raise ValueError(f"Unsupported schedule string: {value}")
    else:
        raise ValueError(f"Unsupported schedule type: {type(value)}")


def make_progress_schedule(
    schedule_fn: Callable[[float], float]
) -> Callable[[int, int], float]:
    """
    Convert a progress-based schedule (taking progress ∈ [0,1]) to a step-based schedule.
    
    Args:
        schedule_fn: Function that takes progress ∈ [0,1] and returns a value
        
    Returns:
        Function that takes (current_step, total_steps) and returns a value
    """
    def step_schedule_fn(current_step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return schedule_fn(1.0)
        progress = min(1.0, current_step / total_steps)
        return schedule_fn(progress)
    
    return step_schedule_fn


# Common schedule presets for convenience
def get_linear_schedule(initial_value: float, final_value: float = 0.0):
    """Convenience function for creating linear decay schedules."""
    return linear_schedule(initial_value, final_value)


def get_constant_schedule(value: float):
    """Convenience function for creating constant schedules."""
    return constant_schedule(value)
