"""
Test gradient norm stability and dtype consistency over long training runs.

This test ensures that the optimizations don't introduce numerical instability
or unwanted dtype promotions during extended training.
"""

import mlx.core as mx
import numpy as np
import gymnasium as gym
from typing import List, Dict
import pytest

from mlx_baselines3.ppo.optimized_ppo import OptimizedPPO
from mlx_baselines3.common.vec_env import DummyVecEnv


class GradientStabilityTest:
    """Test gradient norm stability and dtype consistency."""
    
    def __init__(self):
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment and optimized PPO."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        self.ppo = OptimizedPPO(
            "MlpPolicy",
            env,
            n_steps=64,  # Smaller for faster testing
            batch_size=16,
            n_epochs=1,
            verbose=0,
            use_jit=True,
            enforce_float32=True
        )
        
        # Initialize
        self.ppo._last_obs = env.reset()
        self.ppo._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    def test_gradient_norm_stability(self, n_iterations: int = 20) -> Dict[str, float]:
        """
        Test gradient norm stability over multiple training iterations.
        
        Args:
            n_iterations: Number of training iterations to test
            
        Returns:
            Dictionary with stability metrics
        """
        grad_norms = []
        loss_values = []
        
        for i in range(n_iterations):
            # Collect rollout
            self.ppo.collect_rollouts(
                self.ppo.env, None, self.ppo.rollout_buffer, n_rollout_steps=self.ppo.n_steps
            )
            
            # Track gradient norms during training
            iteration_grad_norms = []
            iteration_losses = []
            
            # Get rollout data for loss computation
            for rollout_data in self.ppo.rollout_buffer.get(self.ppo.batch_size):
                # Get current parameters
                params = self.ppo.policy.state_dict()
                
                # Ensure float32
                if self.ppo.enforce_float32:
                    params = self.ppo._ensure_float32_arrays(params)
                    rollout_data = self.ppo._ensure_float32_arrays(rollout_data)
                
                # Define loss function
                def loss_fn(p):
                    self.ppo.policy.load_state_dict(p, strict=False)
                    return self.ppo._optimized_compute_loss(rollout_data, self.ppo.policy, 0.2, None, 0.0)
                
                # Compute loss and gradients
                from mlx_baselines3.common.optimizers import compute_loss_and_grads
                loss_val, grads = compute_loss_and_grads(loss_fn, params)
                
                # Compute gradient norm
                grad_norm = float(mx.sqrt(sum(mx.sum(g ** 2) for g in grads.values() if g is not None)))
                
                iteration_grad_norms.append(grad_norm)
                iteration_losses.append(float(loss_val))
                
                # Update parameters for next iteration
                grads, _ = self.ppo._optimized_gradient_clipping(grads)
                params, self.ppo.optimizer_state = self.ppo.optimizer_adapter.update(
                    params, grads, self.ppo.optimizer_state
                )
                self.ppo.policy.load_state_dict(params, strict=False)
                mx.eval(list(params.values()))
                
                break  # Only one batch per iteration for speed
            
            if iteration_grad_norms:
                grad_norms.extend(iteration_grad_norms)
                loss_values.extend(iteration_losses)
        
        # Compute stability metrics
        grad_norms = np.array(grad_norms)
        loss_values = np.array(loss_values)
        
        # Filter out infinite or NaN values
        valid_grad_norms = grad_norms[np.isfinite(grad_norms)]
        valid_loss_values = loss_values[np.isfinite(loss_values)]
        
        return {
            "grad_norm_mean": float(np.mean(valid_grad_norms)),
            "grad_norm_std": float(np.std(valid_grad_norms)),
            "grad_norm_max": float(np.max(valid_grad_norms)),
            "grad_norm_min": float(np.min(valid_grad_norms)),
            "grad_norm_cv": float(np.std(valid_grad_norms) / np.mean(valid_grad_norms)) if np.mean(valid_grad_norms) > 0 else float('inf'),
            "loss_mean": float(np.mean(valid_loss_values)),
            "loss_std": float(np.std(valid_loss_values)),
            "n_valid_grad_norms": len(valid_grad_norms),
            "n_total_iterations": len(grad_norms),
            "stability_score": 1.0 / (1.0 + float(np.std(valid_grad_norms) / np.mean(valid_grad_norms))) if np.mean(valid_grad_norms) > 0 else 0.0
        }
    
    def test_dtype_consistency(self, n_checks: int = 10) -> Dict[str, bool]:
        """
        Test that dtypes remain consistent throughout training.
        
        Args:
            n_checks: Number of training steps to check
            
        Returns:
            Dictionary with dtype consistency results
        """
        dtype_consistent = True
        unexpected_dtypes = []
        
        for i in range(n_checks):
            # Collect rollout
            self.ppo.collect_rollouts(
                self.ppo.env, None, self.ppo.rollout_buffer, n_rollout_steps=self.ppo.n_steps
            )
            
            # Check parameter dtypes
            params = self.ppo.policy.state_dict()
            for key, param in params.items():
                if self.ppo.enforce_float32 and param.dtype != mx.float32:
                    dtype_consistent = False
                    unexpected_dtypes.append(f"{key}: {param.dtype}")
            
            # Check rollout data dtypes
            for rollout_data in self.ppo.rollout_buffer.get(self.ppo.batch_size):
                if self.ppo.enforce_float32:
                    rollout_data = self.ppo._ensure_float32_arrays(rollout_data)
                
                for key, data in rollout_data.items():
                    if isinstance(data, mx.array) and self.ppo.enforce_float32 and data.dtype != mx.float32:
                        dtype_consistent = False
                        unexpected_dtypes.append(f"rollout_{key}: {data.dtype}")
                
                break  # Only check one batch
        
        return {
            "dtype_consistent": dtype_consistent,
            "unexpected_dtypes": unexpected_dtypes,
            "n_checks_performed": n_checks
        }
    
    def test_numerical_stability(self) -> Dict[str, bool]:
        """
        Test for numerical stability issues (NaN, inf).
        
        Returns:
            Dictionary with numerical stability results
        """
        has_nan = False
        has_inf = False
        nan_locations = []
        inf_locations = []
        
        # Run a few training steps and check for NaN/inf
        for i in range(5):
            self.ppo.collect_rollouts(
                self.ppo.env, None, self.ppo.rollout_buffer, n_rollout_steps=self.ppo.n_steps
            )
            
            # Check parameters
            params = self.ppo.policy.state_dict()
            for key, param in params.items():
                if mx.any(mx.isnan(param)):
                    has_nan = True
                    nan_locations.append(f"param_{key}")
                if mx.any(mx.isinf(param)):
                    has_inf = True
                    inf_locations.append(f"param_{key}")
            
            # Check gradients during training
            for rollout_data in self.ppo.rollout_buffer.get(self.ppo.batch_size):
                def loss_fn(p):
                    self.ppo.policy.load_state_dict(p, strict=False)
                    return self.ppo._optimized_compute_loss(rollout_data, self.ppo.policy, 0.2, None, 0.0)
                
                from mlx_baselines3.common.optimizers import compute_loss_and_grads
                loss_val, grads = compute_loss_and_grads(loss_fn, params)
                
                # Check loss
                if mx.isnan(loss_val):
                    has_nan = True
                    nan_locations.append("loss")
                if mx.isinf(loss_val):
                    has_inf = True
                    inf_locations.append("loss")
                
                # Check gradients
                for key, grad in grads.items():
                    if grad is not None:
                        if mx.any(mx.isnan(grad)):
                            has_nan = True
                            nan_locations.append(f"grad_{key}")
                        if mx.any(mx.isinf(grad)):
                            has_inf = True
                            inf_locations.append(f"grad_{key}")
                
                break  # Only check one batch
        
        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "nan_locations": nan_locations,
            "inf_locations": inf_locations,
            "numerically_stable": not (has_nan or has_inf)
        }


def run_gradient_stability_tests() -> Dict[str, Dict]:
    """Run all gradient stability tests."""
    results = {}
    
    print("Running Gradient Stability and Numerical Tests...")
    print("=" * 60)
    
    test = GradientStabilityTest()
    
    # Gradient norm stability
    print("1. Gradient Norm Stability Test")
    results["gradient_stability"] = test.test_gradient_norm_stability(n_iterations=10)
    
    print(f"   Mean gradient norm: {results['gradient_stability']['grad_norm_mean']:.4f}")
    print(f"   Gradient norm std: {results['gradient_stability']['grad_norm_std']:.4f}")
    print(f"   Coefficient of variation: {results['gradient_stability']['grad_norm_cv']:.4f}")
    print(f"   Stability score: {results['gradient_stability']['stability_score']:.4f}")
    
    # Dtype consistency
    print("\n2. Dtype Consistency Test")
    results["dtype_consistency"] = test.test_dtype_consistency(n_checks=5)
    
    print(f"   Dtype consistent: {results['dtype_consistency']['dtype_consistent']}")
    if results['dtype_consistency']['unexpected_dtypes']:
        print(f"   Unexpected dtypes: {results['dtype_consistency']['unexpected_dtypes']}")
    
    # Numerical stability
    print("\n3. Numerical Stability Test")
    results["numerical_stability"] = test.test_numerical_stability()
    
    print(f"   Numerically stable: {results['numerical_stability']['numerically_stable']}")
    print(f"   Has NaN: {results['numerical_stability']['has_nan']}")
    print(f"   Has Inf: {results['numerical_stability']['has_inf']}")
    
    if results['numerical_stability']['nan_locations']:
        print(f"   NaN locations: {results['numerical_stability']['nan_locations']}")
    if results['numerical_stability']['inf_locations']:
        print(f"   Inf locations: {results['numerical_stability']['inf_locations']}")
    
    print("=" * 60)
    
    # Overall assessment
    stable = (
        results['gradient_stability']['stability_score'] > 0.5 and
        results['dtype_consistency']['dtype_consistent'] and
        results['numerical_stability']['numerically_stable']
    )
    
    print(f"Overall Stability Assessment: {'PASS' if stable else 'NEEDS_ATTENTION'}")
    
    return results


@pytest.mark.slow
def test_gradient_stability():
    """Test that gradient norms remain stable and dtypes are consistent."""
    results = run_gradient_stability_tests()
    
    # Check gradient norm stability
    assert results['gradient_stability']['stability_score'] > 0.3, \
        f"Gradient stability score {results['gradient_stability']['stability_score']:.3f} is too low"
    
    # Check coefficient of variation is reasonable
    assert results['gradient_stability']['grad_norm_cv'] < 5.0, \
        f"Gradient norm coefficient of variation {results['gradient_stability']['grad_norm_cv']:.3f} is too high"
    
    # Check dtype consistency
    assert results['dtype_consistency']['dtype_consistent'], \
        f"Dtype consistency failed: {results['dtype_consistency']['unexpected_dtypes']}"
    
    # Check numerical stability
    assert results['numerical_stability']['numerically_stable'], \
        f"Numerical instability detected: NaN={results['numerical_stability']['has_nan']}, Inf={results['numerical_stability']['has_inf']}"


if __name__ == "__main__":
    results = run_gradient_stability_tests()
    
    # Save results
    import json
    with open("gradient_stability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGradient stability results saved to gradient_stability_results.json")
