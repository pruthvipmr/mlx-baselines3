"""Helpers for optionally using MLX JIT-compiled kernels."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import mlx.core as mx


def jit_loss_computation(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    JIT compile loss computation functions.

    Args:
        func: Function to JIT compile

    Returns:
        JIT compiled function
    """
    try:
        # Use MLX's JIT if available
        if hasattr(mx, "compile"):
            return mx.compile(func)
        else:
            # Fallback - just return the original function
            return func
    except Exception:
        # If JIT compilation fails, return original function
        return func


@jit_loss_computation
def jit_ppo_loss_core(
    values: mx.array,
    log_probs: mx.array,
    old_log_probs: mx.array,
    advantages: mx.array,
    returns: mx.array,
    old_values: mx.array,
    entropy: mx.array,
    clip_range: float,
    clip_range_vf: float,
    ent_coef: float,
    vf_coef: float,
) -> mx.array:
    """
    JIT-compiled core PPO loss computation.

    This function contains the mathematical operations for PPO loss
    that can benefit from JIT compilation.
    """
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

    # Policy loss
    ratio = mx.exp(log_probs - old_log_probs)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))

    # Value loss
    if clip_range_vf > 0:
        values_pred = old_values + mx.clip(
            values - old_values, -clip_range_vf, clip_range_vf
        )
        value_loss_1 = (returns - values) ** 2
        value_loss_2 = (returns - values_pred) ** 2
        value_loss = mx.mean(mx.maximum(value_loss_1, value_loss_2))
    else:
        value_loss = mx.mean((returns - values) ** 2)

    # Entropy loss
    entropy_loss = -mx.mean(entropy)

    # Total loss
    return cast(
        mx.array,
        policy_loss + ent_coef * entropy_loss + vf_coef * value_loss,
    )


@jit_loss_computation
def jit_gradient_clipping(
    grads_flat: mx.array, max_norm: float
) -> Tuple[mx.array, mx.array]:
    """
    JIT-compiled gradient clipping.

    Args:
        grads_flat: Flattened gradients
        max_norm: Maximum gradient norm

    Returns:
        Tuple of (clipped_grads, original_norm)
    """
    grad_norm = mx.sqrt(mx.sum(grads_flat**2))
    clip_coef = mx.minimum(max_norm / (grad_norm + 1e-8), 1.0)
    clipped_grads = grads_flat * clip_coef
    return clipped_grads, grad_norm


@jit_loss_computation
def jit_advantage_normalization(advantages: mx.array) -> mx.array:
    """
    JIT-compiled advantage normalization.

    Args:
        advantages: Raw advantages

    Returns:
        Normalized advantages
    """
    if len(advantages) > 1:
        return cast(
            mx.array,
            (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8),
        )
    return advantages


class JITOptimizedOperations:
    """
    Collection of JIT-optimized operations for training.
    """

    def __init__(self) -> None:
        """Initialize JIT operations."""
        self.jit_enabled = self._check_jit_support()
        self.compiled_ops: Dict[str, Callable[..., Any]] = {}
        if self.jit_enabled:
            self._compile_operations()

    def _check_jit_support(self) -> bool:
        """Check if JIT compilation is supported."""
        try:
            # Test if mx.compile is available and working
            if hasattr(mx, "compile"):

                @mx.compile
                def test_func(x: mx.array) -> mx.array:
                    return x * 2

                # Test the compiled function
                test_func(mx.array([1.0]))
                return True
            return False
        except Exception:
            return False

    def _compile_operations(self) -> None:
        """Compile frequently used operations."""
        if not self.jit_enabled:
            return

        # Pre-compile common operations
        self.compiled_ops = {
            "ppo_loss": jit_ppo_loss_core,
            "grad_clip": jit_gradient_clipping,
            "advantage_norm": jit_advantage_normalization,
        }

    def optimized_ppo_loss(
        self,
        values: mx.array,
        log_probs: mx.array,
        old_log_probs: mx.array,
        advantages: mx.array,
        returns: mx.array,
        old_values: mx.array,
        entropy: mx.array,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
    ) -> mx.array:
        """
        Compute PPO loss with JIT optimization if available.

        Args:
            values: Current value estimates
            log_probs: Current log probabilities
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Returns
            old_values: Old value estimates
            entropy: Entropy values
            clip_range: PPO clipping range
            clip_range_vf: Value function clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient

        Returns:
            Total loss
        """
        vf_clip = 0.0 if clip_range_vf is None else clip_range_vf

        if self.jit_enabled:
            return self.compiled_ops["ppo_loss"](
                values,
                log_probs,
                old_log_probs,
                advantages,
                returns,
                old_values,
                entropy,
                clip_range,
                vf_clip,
                ent_coef,
                vf_coef,
            )
        else:
            # Fallback to regular computation
            return jit_ppo_loss_core(
                values,
                log_probs,
                old_log_probs,
                advantages,
                returns,
                old_values,
                entropy,
                clip_range,
                vf_clip,
                ent_coef,
                vf_coef,
            )

    def optimized_grad_clipping(
        self, grads: Dict[str, Optional[mx.array]], max_norm: float
    ) -> Tuple[Dict[str, Optional[mx.array]], float]:
        """
        Clip gradients with JIT optimization.

        Args:
            grads: Gradient dictionary
            max_norm: Maximum gradient norm

        Returns:
            Tuple of (clipped_grads, original_norm)
        """
        # Flatten gradients for efficient norm computation
        grad_values: List[mx.array] = [g for g in grads.values() if g is not None]
        if not grad_values:
            return grads, 0.0

        grads_flat = mx.concatenate([mx.flatten(g) for g in grad_values])

        if self.jit_enabled:
            clipped_flat, grad_norm_array = self.compiled_ops["grad_clip"](
                grads_flat, max_norm
            )
            grad_norm = float(grad_norm_array)
        else:
            clipped_flat, grad_norm_array = jit_gradient_clipping(grads_flat, max_norm)
            grad_norm = float(grad_norm_array)

        # Reshape back to original structure
        clipped_grads: Dict[str, Optional[mx.array]] = {}
        start_idx = 0

        for key, grad in grads.items():
            if grad is None:
                clipped_grads[key] = None
                continue

            grad_size = grad.size
            end_idx = start_idx + grad_size
            clipped_grads[key] = clipped_flat[start_idx:end_idx].reshape(grad.shape)
            start_idx = end_idx

        return clipped_grads, grad_norm


def create_jit_optimizer() -> JITOptimizedOperations:
    """
    Factory function to create JIT optimizer.

    Returns:
        JITOptimizedOperations instance
    """
    return JITOptimizedOperations()


def benchmark_jit_performance() -> Dict[str, Any]:
    """
    Benchmark JIT compilation performance improvements.

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Create test data
    batch_size = 64
    values = mx.random.normal((batch_size,), dtype=mx.float32)
    log_probs = mx.random.normal((batch_size,), dtype=mx.float32)
    old_log_probs = mx.random.normal((batch_size,), dtype=mx.float32)
    advantages = mx.random.normal((batch_size,), dtype=mx.float32)
    returns = mx.random.normal((batch_size,), dtype=mx.float32)
    old_values = mx.random.normal((batch_size,), dtype=mx.float32)
    entropy = mx.random.normal((batch_size,), dtype=mx.float32)

    # Regular computation
    def regular_loss() -> mx.array:
        return jit_ppo_loss_core(
            values,
            log_probs,
            old_log_probs,
            advantages,
            returns,
            old_values,
            entropy,
            0.2,
            0.0,
            0.0,
            0.5,
        )

    # JIT optimized computation
    jit_ops = create_jit_optimizer()

    def jit_loss() -> mx.array:
        return jit_ops.optimized_ppo_loss(
            values,
            log_probs,
            old_log_probs,
            advantages,
            returns,
            old_values,
            entropy,
            0.2,
            0.0,
            0.0,
            0.5,
        )

    # Benchmark both approaches
    n_runs = 100

    # Regular timing
    times_regular = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = regular_loss()
        mx.eval(result)
        end = time.perf_counter()
        times_regular.append(end - start)

    # JIT timing (with warmup)
    # Warmup
    for _ in range(5):
        result = jit_loss()
        mx.eval(result)

    times_jit = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = jit_loss()
        mx.eval(result)
        end = time.perf_counter()
        times_jit.append(end - start)

    import numpy as np

    regular_mean = np.mean(times_regular)
    jit_mean = np.mean(times_jit)

    return {
        "jit_enabled": jit_ops.jit_enabled,
        "regular_time": regular_mean,
        "jit_time": jit_mean,
        "speedup": regular_mean / jit_mean if jit_mean > 0 else 1.0,
        "improvement_pct": ((regular_mean - jit_mean) / regular_mean * 100)
        if regular_mean > 0
        else 0.0,
    }


if __name__ == "__main__":
    results = benchmark_jit_performance()

    print("JIT Compilation Benchmark Results:")
    print(f"JIT Enabled: {results['jit_enabled']}")
    print(f"Regular computation time: {results['regular_time']:.6f}s")
    print(f"JIT computation time: {results['jit_time']:.6f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Improvement: {results['improvement_pct']:.1f}%")
