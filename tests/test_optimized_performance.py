"""
Test optimized performance implementations against baseline.

This module tests the performance improvements from the optimized
training loops and functional loss computations.
"""

import time
import numpy as np
import mlx.core as mx
import gymnasium as gym
from typing import Dict, Any
import pytest

from mlx_baselines3.ppo import PPO
from mlx_baselines3.common.vec_env import DummyVecEnv
from mlx_baselines3.common.optimized_training import OptimizedPPOTrainer, create_optimized_ppo_trainer
from mlx_baselines3.common.functional_losses import ppo_functional_loss
from tests.test_performance_benchmarks import PerformanceBenchmark


class OptimizedPPOPerformanceTest(PerformanceBenchmark):
    """Test optimized PPO performance against baseline."""
    
    def __init__(self):
        super().__init__("OptimizedPPO")
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment and PPO instances."""
        # Create test environment
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        
        # Create baseline PPO
        self.baseline_ppo = PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=32,
            n_epochs=2,
            verbose=0
        )
        
        # Create optimized PPO trainer
        self.optimized_trainer = create_optimized_ppo_trainer(
            policy=self.baseline_ppo.policy,
            optimizer_adapter=self.baseline_ppo.optimizer_adapter,
            max_grad_norm=self.baseline_ppo.max_grad_norm,
            enforce_float32=True
        )
        
        # Collect rollout data for testing
        self.baseline_ppo._last_obs = env.reset()
        self.baseline_ppo._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        self.baseline_ppo.collect_rollouts(
            env, None, self.baseline_ppo.rollout_buffer, n_rollout_steps=self.baseline_ppo.n_steps
        )
        
        # Get sample batch for benchmarking
        self.test_batch = next(iter(self.baseline_ppo.rollout_buffer.get(self.baseline_ppo.batch_size)))
        
        # Get initial parameters
        self.initial_params = self.baseline_ppo.policy.state_dict()
        self.initial_optimizer_state = self.baseline_ppo.optimizer_state
    
    def baseline_single_update(self):
        """Baseline PPO single update (current implementation)."""
        # Reset to initial state
        params = {k: mx.array(v) for k, v in self.initial_params.items()}
        optimizer_state = self.initial_optimizer_state
        
        # Simulate current training approach
        def loss_fn(p):
            self.baseline_ppo.policy.load_state_dict(p, strict=False)
            return self.baseline_ppo._compute_loss(
                self.test_batch, self.baseline_ppo.policy, 0.2, None, 0.0
            )
        
        # Compute loss and gradients
        from mlx_baselines3.common.optimizers import compute_loss_and_grads, clip_grad_norm
        loss_val, grads = compute_loss_and_grads(loss_fn, params)
        
        # Clip gradients
        if self.baseline_ppo.max_grad_norm is not None:
            grads, grad_norm = clip_grad_norm(grads, self.baseline_ppo.max_grad_norm)
        
        # Update parameters
        params, optimizer_state = self.baseline_ppo.optimizer_adapter.update(
            params, grads, optimizer_state
        )
        
        # Load updated params
        self.baseline_ppo.policy.load_state_dict(params, strict=False)
        mx.eval(list(params.values()))
        
        return params, optimizer_state
    
    def optimized_single_update(self):
        """Optimized PPO single update."""
        # Reset to initial state
        params = {k: mx.array(v) for k, v in self.initial_params.items()}
        optimizer_state = self.initial_optimizer_state
        
        # Use optimized trainer
        params, optimizer_state, step_stats = self.optimized_trainer.optimized_train_step(
            params=params,
            optimizer_state=optimizer_state,
            rollout_data=self.test_batch,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5
        )
        
        return params, optimizer_state, step_stats
    
    def benchmark_single_update_comparison(self):
        """Compare baseline vs optimized single update performance."""
        return self.compare_implementations(
            self.baseline_single_update,
            lambda: self.optimized_single_update()[:2],  # Only return params and state
            n_runs=20
        )
    
    def baseline_epoch_training(self):
        """Baseline epoch training."""
        params = {k: mx.array(v) for k, v in self.initial_params.items()}
        optimizer_state = self.initial_optimizer_state
        
        # Simulate baseline epoch training
        for rollout_data in self.baseline_ppo.rollout_buffer.get(self.baseline_ppo.batch_size):
            def loss_fn(p):
                self.baseline_ppo.policy.load_state_dict(p, strict=False)
                return self.baseline_ppo._compute_loss(rollout_data, self.baseline_ppo.policy, 0.2, None, 0.0)
            
            from mlx_baselines3.common.optimizers import compute_loss_and_grads, clip_grad_norm
            loss_val, grads = compute_loss_and_grads(loss_fn, params)
            
            if self.baseline_ppo.max_grad_norm is not None:
                grads, grad_norm = clip_grad_norm(grads, self.baseline_ppo.max_grad_norm)
            
            params, optimizer_state = self.baseline_ppo.optimizer_adapter.update(
                params, grads, optimizer_state
            )
            
            self.baseline_ppo.policy.load_state_dict(params, strict=False)
            mx.eval(list(params.values()))
        
        return params, optimizer_state
    
    def optimized_epoch_training(self):
        """Optimized epoch training."""
        params = {k: mx.array(v) for k, v in self.initial_params.items()}
        optimizer_state = self.initial_optimizer_state
        
        # Use optimized trainer
        params, optimizer_state, epoch_stats, continue_training = self.optimized_trainer.train_epoch(
            params=params,
            optimizer_state=optimizer_state,
            rollout_buffer=self.baseline_ppo.rollout_buffer,
            batch_size=self.baseline_ppo.batch_size,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5
        )
        
        return params, optimizer_state, epoch_stats
    
    def benchmark_epoch_comparison(self):
        """Compare baseline vs optimized epoch training."""
        return self.compare_implementations(
            self.baseline_epoch_training,
            lambda: self.optimized_epoch_training()[:2],  # Only return params and state
            n_runs=10
        )


class FunctionalLossPerformanceTest(PerformanceBenchmark):
    """Test functional loss computation performance."""
    
    def __init__(self):
        super().__init__("FunctionalLoss")
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test data for loss computation."""
        # Create test policy
        env = gym.make("CartPole-v1")
        from mlx_baselines3.common.policies import ActorCriticPolicy
        
        self.policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
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
        
        self.params = self.policy.state_dict()
        self.policy_apply_fn = self.policy.create_functional_apply_fn()
    
    def baseline_loss_computation(self):
        """Baseline loss computation (with parameter loading)."""
        def loss_fn(p):
            self.policy.load_state_dict(p, strict=False)
            values, log_prob, entropy = self.policy.evaluate_actions(
                self.test_batch["observations"], self.test_batch["actions"]
            )
            values = mx.flatten(values)
            
            # Simple loss computation
            advantages = self.test_batch["advantages"]
            if len(advantages) > 1:
                advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
            
            ratio = mx.exp(log_prob - self.test_batch["log_probs"])
            clip_range = 0.2
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
            
            value_loss = mx.mean((self.test_batch["returns"] - values) ** 2)
            entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
            
            return policy_loss + 0.0 * entropy_loss + 0.5 * value_loss
        
        from mlx_baselines3.common.optimizers import compute_loss_and_grads
        return compute_loss_and_grads(loss_fn, self.params)
    
    def optimized_functional_loss_computation(self):
        """Optimized functional loss computation."""
        loss_val = ppo_functional_loss(
            params=self.params,
            rollout_data=self.test_batch,
            policy_apply_fn=self.policy_apply_fn,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5
        )
        
        # For fair comparison, also compute gradients
        def pure_loss_fn(p):
            return ppo_functional_loss(
                params=p,
                rollout_data=self.test_batch,
                policy_apply_fn=self.policy_apply_fn,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.0,
                vf_coef=0.5
            )
        
        from mlx_baselines3.common.optimizers import compute_loss_and_grads
        return compute_loss_and_grads(pure_loss_fn, self.params)
    
    def benchmark_functional_loss_comparison(self):
        """Compare baseline vs functional loss computation."""
        return self.compare_implementations(
            self.baseline_loss_computation,
            self.optimized_functional_loss_computation,
            n_runs=50
        )


def run_optimization_benchmarks() -> Dict[str, Any]:
    """Run all optimization benchmarks."""
    results = {}
    
    print("Running Optimization Performance Benchmarks...")
    print("=" * 60)
    
    # PPO single update comparison
    print("1. PPO Single Update Comparison")
    ppo_test = OptimizedPPOPerformanceTest()
    results["ppo_single_update"] = ppo_test.benchmark_single_update_comparison()
    
    print(f"   Baseline: {results['ppo_single_update']['baseline']['mean']:.4f}s")
    print(f"   Optimized: {results['ppo_single_update']['optimized']['mean']:.4f}s") 
    print(f"   Speedup: {results['ppo_single_update']['speedup']:.2f}x")
    print(f"   Improvement: {results['ppo_single_update']['improvement_pct']:.1f}%")
    print(f"   Meets target: {results['ppo_single_update']['meets_target']}")
    
    # PPO epoch comparison  
    print("\n2. PPO Epoch Training Comparison")
    results["ppo_epoch"] = ppo_test.benchmark_epoch_comparison()
    
    print(f"   Baseline: {results['ppo_epoch']['baseline']['mean']:.4f}s")
    print(f"   Optimized: {results['ppo_epoch']['optimized']['mean']:.4f}s")
    print(f"   Speedup: {results['ppo_epoch']['speedup']:.2f}x")
    print(f"   Improvement: {results['ppo_epoch']['improvement_pct']:.1f}%")
    print(f"   Meets target: {results['ppo_epoch']['meets_target']}")
    
    # Functional loss comparison
    print("\n3. Functional Loss Computation Comparison")
    loss_test = FunctionalLossPerformanceTest()
    results["functional_loss"] = loss_test.benchmark_functional_loss_comparison()
    
    print(f"   Baseline: {results['functional_loss']['baseline']['mean']:.4f}s")
    print(f"   Optimized: {results['functional_loss']['optimized']['mean']:.4f}s")
    print(f"   Speedup: {results['functional_loss']['speedup']:.2f}x")
    print(f"   Improvement: {results['functional_loss']['improvement_pct']:.1f}%")
    print(f"   Meets target: {results['functional_loss']['meets_target']}")
    
    print("=" * 60)
    
    # Summary
    total_improvements = [
        results["ppo_single_update"]["meets_target"],
        results["ppo_epoch"]["meets_target"],
        results["functional_loss"]["meets_target"]
    ]
    
    avg_improvement = np.mean([
        results["ppo_single_update"]["improvement_pct"],
        results["ppo_epoch"]["improvement_pct"],
        results["functional_loss"]["improvement_pct"]
    ])
    
    print(f"Overall Performance Summary:")
    print(f"  Components meeting 10%+ improvement target: {sum(total_improvements)}/3")
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print(f"  Target achievement: {'PASS' if sum(total_improvements) >= 2 else 'NEEDS_WORK'}")
    
    return results


@pytest.mark.slow
def test_optimization_performance():
    """Test that optimization components can be run without errors."""
    # Performance optimizations are hardware-dependent, so we just test functionality
    results = run_optimization_benchmarks()
    
    # Verify that all benchmark components completed successfully
    assert "ppo_single_update" in results
    assert "ppo_epoch" in results
    assert "functional_loss" in results
    
    # Verify all results have required fields
    for component in ["ppo_single_update", "ppo_epoch", "functional_loss"]:
        assert "baseline" in results[component]
        assert "optimized" in results[component]
        assert "speedup" in results[component]
        assert "improvement_pct" in results[component]
        assert "meets_target" in results[component]
        
        # Verify times are positive
        assert results[component]["baseline"]["mean"] > 0
        assert results[component]["optimized"]["mean"] > 0
    
    # Log results for manual review (hardware-dependent performance)
    print("Performance optimization results logged for review")
    print(f"Results: {results}")


if __name__ == "__main__":
    results = run_optimization_benchmarks()
    
    # Save results
    import json
    with open("optimization_performance.json", "w") as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_value = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        json_value[k] = v if isinstance(v, list) else v.tolist()
                    elif hasattr(v, 'item'):
                        json_value[k] = float(v)
                    else:
                        json_value[k] = v
                json_results[key] = json_value
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nOptimization results saved to optimization_performance.json")
