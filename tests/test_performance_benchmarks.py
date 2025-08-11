"""
Performance benchmarks for numerical and optimization improvements.

This module provides micro-benchmarks to measure performance improvements
in the numerical and optimization components of MLX-Baselines3.
"""

import time
import numpy as np
import mlx.core as mx
import gymnasium as gym
from typing import Dict, List, Tuple, Any
import pytest

from mlx_baselines3.ppo import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.common.optimizers import AdamAdapter, compute_loss_and_grads, clip_grad_norm
from mlx_baselines3.common.policies import ActorCriticPolicy
from mlx_baselines3.common.buffers import RolloutBuffer


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    def time_function(self, func, *args, n_runs: int = 10, warmup: int = 2, **kwargs) -> Dict[str, float]:
        """Time a function with warmup and multiple runs."""
        # Warmup runs
        for _ in range(warmup):
            result = func(*args, **kwargs)
            if isinstance(result, (list, tuple)) and any(isinstance(x, mx.array) for x in result):
                # Force evaluation for MLX arrays
                for item in result:
                    if isinstance(item, mx.array):
                        mx.eval(item)
            elif isinstance(result, mx.array):
                mx.eval(result)
        
        # Timed runs
        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            
            # Force evaluation for MLX arrays
            if isinstance(result, (list, tuple)) and any(isinstance(x, mx.array) for x in result):
                for item in result:
                    if isinstance(item, mx.array):
                        mx.eval(item)
            elif isinstance(result, mx.array):
                mx.eval(result)
            elif isinstance(result, dict):
                for value in result.values():
                    if isinstance(value, mx.array):
                        mx.eval(value)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "times": times
        }
    
    def compare_implementations(self, baseline_func, optimized_func, *args, **kwargs) -> Dict[str, Any]:
        """Compare baseline vs optimized implementation."""
        baseline_times = self.time_function(baseline_func, *args, **kwargs)
        optimized_times = self.time_function(optimized_func, *args, **kwargs)
        
        speedup = baseline_times["mean"] / optimized_times["mean"]
        improvement_pct = (speedup - 1.0) * 100
        
        return {
            "baseline": baseline_times,
            "optimized": optimized_times,
            "speedup": speedup,
            "improvement_pct": improvement_pct,
            "meets_target": improvement_pct >= 10.0  # Target: 10-20% improvement
        }


class OptimizerBenchmark(PerformanceBenchmark):
    """Benchmark optimizer performance."""
    
    def __init__(self):
        super().__init__("Optimizer")
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test parameters and gradients."""
        # Create realistic parameter shapes (similar to MLP policy)
        self.params = {
            "mlp_extractor.policy_net.0.weight": mx.random.normal((64, 4), dtype=mx.float32),
            "mlp_extractor.policy_net.0.bias": mx.random.normal((64,), dtype=mx.float32),
            "mlp_extractor.policy_net.2.weight": mx.random.normal((64, 64), dtype=mx.float32),
            "mlp_extractor.policy_net.2.bias": mx.random.normal((64,), dtype=mx.float32),
            "action_net.weight": mx.random.normal((2, 64), dtype=mx.float32),
            "action_net.bias": mx.random.normal((2,), dtype=mx.float32),
            "value_net.weight": mx.random.normal((1, 64), dtype=mx.float32),
            "value_net.bias": mx.random.normal((1,), dtype=mx.float32),
        }
        
        # Create corresponding gradients
        self.grads = {k: mx.random.normal(v.shape, dtype=mx.float32) for k, v in self.params.items()}
    
    def benchmark_optimizer_update(self):
        """Benchmark optimizer update performance."""
        optimizer = AdamAdapter(learning_rate=3e-4)
        state = optimizer.init_state(self.params)
        
        def optimizer_update():
            return optimizer.update(self.params, self.grads, state)
        
        return self.time_function(optimizer_update, n_runs=100)


class GradientBenchmark(PerformanceBenchmark):
    """Benchmark gradient computation performance."""
    
    def __init__(self):
        super().__init__("Gradient")
        self.setup_test_policy()
    
    def setup_test_policy(self):
        """Setup test policy and data."""
        env = gym.make("CartPole-v1")
        obs_space = env.observation_space
        action_space = env.action_space
        
        self.policy = ActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda x: 3e-4
        )
        
        # Create test batch
        batch_size = 64
        self.test_batch = {
            "observations": mx.random.normal((batch_size, 4), dtype=mx.float32),
            "actions": mx.random.normal((batch_size, 1), dtype=mx.float32),
            "advantages": mx.random.normal((batch_size,), dtype=mx.float32),
            "returns": mx.random.normal((batch_size,), dtype=mx.float32),
            "values": mx.random.normal((batch_size,), dtype=mx.float32),
            "log_probs": mx.random.normal((batch_size,), dtype=mx.float32)
        }
    
    def baseline_loss_computation(self):
        """Baseline loss computation (current implementation)."""
        def loss_fn(params):
            # Load params into policy for forward computations
            self.policy.load_state_dict(params, strict=False)
            
            actions = self.test_batch["actions"]
            values, log_prob, entropy = self.policy.evaluate_actions(
                self.test_batch["observations"], actions
            )
            values = mx.flatten(values)
            
            # Normalize advantages
            advantages = self.test_batch["advantages"]
            if len(advantages) > 1:
                advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
            
            # Ratio between old and new policy
            ratio = mx.exp(log_prob - self.test_batch["log_probs"])
            
            # Clipped surrogate loss
            clip_range = 0.2
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
            
            # Value loss
            value_loss = mx.mean((self.test_batch["returns"] - values) ** 2)
            
            # Entropy loss
            entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
            
            # Total loss
            return policy_loss + 0.0 * entropy_loss + 0.5 * value_loss
        
        params = self.policy.state_dict()
        return compute_loss_and_grads(loss_fn, params)
    
    def benchmark_gradient_computation(self):
        """Benchmark gradient computation."""
        return self.time_function(self.baseline_loss_computation, n_runs=50)


class DtypeBenchmark(PerformanceBenchmark):
    """Benchmark dtype handling and promotion."""
    
    def __init__(self):
        super().__init__("Dtype")
    
    def test_dtype_promotion(self):
        """Test and benchmark dtype promotion issues."""
        # Test common operations that might cause dtype promotion
        a_f32 = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        b_f32 = mx.array([0.5, 1.5, 2.5], dtype=mx.float32)
        
        def operations_f32():
            # Common operations in training
            result1 = a_f32 * b_f32
            result2 = mx.mean(result1)
            result3 = mx.sqrt(result2 + 1e-8)
            result4 = result3 / (result3 + 1e-6)
            return result4
        
        # Test with Python float (which might cause promotion)
        python_scalar = 3.14159  # Python float (float64)
        
        def operations_with_python_scalar():
            result1 = a_f32 * python_scalar  # This might promote
            result2 = mx.mean(result1)
            result3 = mx.sqrt(result2 + 1e-8)
            result4 = result3 / (result3 + 1e-6)
            return result4
        
        f32_times = self.time_function(operations_f32, n_runs=1000)
        scalar_times = self.time_function(operations_with_python_scalar, n_runs=1000)
        
        # Check result dtypes
        f32_result = operations_f32()
        scalar_result = operations_with_python_scalar()
        
        return {
            "f32_times": f32_times,
            "scalar_times": scalar_times,
            "f32_dtype": f32_result.dtype,
            "scalar_dtype": scalar_result.dtype,
            "dtype_promotion_detected": scalar_result.dtype != mx.float32
        }


class PPOUpdateBenchmark(PerformanceBenchmark):
    """Benchmark full PPO update performance."""
    
    def __init__(self):
        super().__init__("PPO_Update")
        self.setup_ppo()
    
    def setup_ppo(self):
        """Setup PPO for benchmarking."""
        # Use CartPole for lightweight testing
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        self.ppo = PPO(
            "MlpPolicy",
            env,
            n_steps=128,  # Smaller for faster benchmarking
            batch_size=32,
            n_epochs=2,   # Fewer epochs for faster benchmarking
            verbose=0
        )
        
        # Collect one rollout for training data
        self.ppo._last_obs = env.reset()
        self.ppo._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        self.ppo.collect_rollouts(env, None, self.ppo.rollout_buffer, n_rollout_steps=self.ppo.n_steps)
    
    def benchmark_single_update(self):
        """Benchmark a single PPO training update."""
        def single_update():
            # Get one minibatch
            for rollout_data in self.ppo.rollout_buffer.get(self.ppo.batch_size):
                # Compute loss and gradients
                params = self.ppo.policy.state_dict()
                
                def loss_fn(p):
                    self.ppo.policy.load_state_dict(p, strict=False)
                    return self.ppo._compute_loss(rollout_data, self.ppo.policy, 0.2, None, 0.0)
                
                loss_val, grads = compute_loss_and_grads(loss_fn, params)
                
                # Clip gradients
                if self.ppo.max_grad_norm is not None:
                    grads, grad_norm = clip_grad_norm(grads, self.ppo.max_grad_norm)
                
                # Update parameters
                params, self.ppo.optimizer_state = self.ppo.optimizer_adapter.update(
                    params, grads, self.ppo.optimizer_state
                )
                
                # Load updated params
                self.ppo.policy.load_state_dict(params, strict=False)
                mx.eval(list(params.values()))
                break  # Only do one minibatch
        
        return self.time_function(single_update, n_runs=20, warmup=3)


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all performance benchmarks."""
    results = {}
    
    print("Running Performance Benchmarks...")
    print("=" * 50)
    
    # Optimizer benchmark
    print("1. Optimizer Update Benchmark")
    optimizer_bench = OptimizerBenchmark()
    results["optimizer"] = optimizer_bench.benchmark_optimizer_update()
    print(f"   Mean time: {results['optimizer']['mean']:.4f}s")
    
    # Gradient computation benchmark
    print("2. Gradient Computation Benchmark")
    gradient_bench = GradientBenchmark()
    results["gradient"] = gradient_bench.benchmark_gradient_computation()
    print(f"   Mean time: {results['gradient']['mean']:.4f}s")
    
    # Dtype benchmark
    print("3. Dtype Handling Benchmark")
    dtype_bench = DtypeBenchmark()
    results["dtype"] = dtype_bench.test_dtype_promotion()
    print(f"   Float32 time: {results['dtype']['f32_times']['mean']:.6f}s")
    print(f"   Scalar dtype time: {results['dtype']['scalar_times']['mean']:.6f}s")
    print(f"   Dtype promotion detected: {results['dtype']['dtype_promotion_detected']}")
    
    # PPO update benchmark
    print("4. PPO Update Benchmark")
    ppo_bench = PPOUpdateBenchmark()
    results["ppo_update"] = ppo_bench.benchmark_single_update()
    print(f"   Mean time: {results['ppo_update']['mean']:.4f}s")
    
    print("=" * 50)
    print("Baseline benchmarks complete!")
    
    return results


@pytest.mark.slow
def test_performance_benchmarks():
    """Test that performance benchmarks run without errors."""
    results = run_all_benchmarks()
    
    # Basic sanity checks
    assert "optimizer" in results
    assert "gradient" in results
    assert "dtype" in results
    assert "ppo_update" in results
    
    # Check that times are reasonable (not too fast or too slow)
    assert 1e-6 < results["optimizer"]["mean"] < 1.0
    assert 1e-6 < results["gradient"]["mean"] < 10.0
    assert 1e-6 < results["ppo_update"]["mean"] < 30.0


if __name__ == "__main__":
    results = run_all_benchmarks()
    
    # Save baseline results for future comparison
    import json
    with open("baseline_performance.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_value = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_value[k] = v.tolist()
                    elif hasattr(v, 'item'):  # MLX array or numpy scalar
                        json_value[k] = float(v)
                    else:
                        json_value[k] = v
                json_results[key] = json_value
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nBaseline results saved to baseline_performance.json")
