"""
Optimized PPO implementation with performance improvements.

This module provides an enhanced PPO implementation that incorporates
JIT compilation, efficient batch processing, and minimal parameter reloading.
"""

import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_baselines3.common.base_class import OnPolicyAlgorithm
from mlx_baselines3.common.buffers import RolloutBuffer
from mlx_baselines3.common.policies import ActorCriticPolicy
from mlx_baselines3.common.type_aliases import GymEnv, MlxArray, Schedule
from mlx_baselines3.common.utils import explained_variance, obs_as_mlx
from mlx_baselines3.common.vec_env import VecEnv
from mlx_baselines3.common.optimizers import (
    AdamAdapter, 
    SGDAdapter, 
    create_optimizer_adapter,
    clip_grad_norm,
    compute_loss_and_grads,
    OptimizerState
)
from mlx_baselines3.common.schedules import apply_schedule_to_param
from mlx_baselines3.common.schedules import get_schedule_fn, make_progress_schedule
from mlx_baselines3.common.jit_optimizations import create_jit_optimizer


class OptimizedPPO(OnPolicyAlgorithm):
    """
    Optimized Proximal Policy Optimization algorithm (PPO) using MLX.
    
    This implementation includes performance optimizations:
    - JIT compilation of loss computations
    - Efficient batch processing
    - Minimal parameter reloading
    - Float32 enforcement
    - Optimized gradient clipping
    """
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        ent_coef: Union[float, str, Schedule] = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
        # Performance optimization flags
        use_jit: bool = True,
        enforce_float32: bool = True,
        **kwargs,
    ):
        # PPO hyperparameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Performance optimization settings
        self.use_jit = use_jit
        self.enforce_float32 = enforce_float32
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            device=device,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=[
                gym.spaces.Box,
                gym.spaces.Discrete,
            ],
            **kwargs,
        )
        
        # Initialize JIT optimizer
        if self.use_jit:
            self.jit_ops = create_jit_optimizer()
            if self.verbose >= 1 and self.jit_ops.jit_enabled:
                print("JIT compilation enabled for performance optimization")
        else:
            self.jit_ops = None
        
        # Initialize training counters
        self._n_updates = 0
        
    def _setup_model(self) -> None:
        """Setup model: create policy, networks, buffers, optimizers, etc."""
        if self.verbose >= 1:
            print("Setting up Optimized PPO model...")
            
        # Create policy
        self._make_policy()
        
        # Initialize rollout buffer
        assert isinstance(self.env, VecEnv), "PPO requires a vectorized environment"
        
        if self.verbose >= 1:
            print("Creating rollout buffer...")
            
        try:
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.n_steps,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.env.num_envs,
            )
            if self.verbose >= 1:
                print(f"Rollout buffer created: {self.rollout_buffer}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"Error creating rollout buffer: {e}")
            raise
        
        if self.verbose >= 1:
            print("Optimized PPO model setup complete")
    
    def _setup_optimizer(self) -> None:
        """Setup the optimizer adapter with proper schedule support."""
        if self.verbose >= 1:
            print("Setting up optimizer adapter...")
        
        # Create learning rate schedule function
        lr_schedule = get_schedule_fn(self.learning_rate)
        
        # Create optimizer adapter (default to Adam)
        self.optimizer_adapter = create_optimizer_adapter(
            optimizer_name="adam",
            learning_rate=lr_schedule,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
        
        # Get initial parameters from policy and initialize optimizer state
        initial_params = self.policy.state_dict()
        self.optimizer_state = self.optimizer_adapter.init_state(initial_params)
        
        if self.verbose >= 1:
            print(f"Optimizer adapter initialized: {type(self.optimizer_adapter).__name__}")
        
    def _make_policy(self) -> None:
        """Create policy instance."""
        from mlx_baselines3.ppo.policies import get_ppo_policy_class
        
        if isinstance(self.policy, str):
            policy_class = get_ppo_policy_class(self.policy)
            self.policy = policy_class(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
            )
        elif isinstance(self.policy, ActorCriticPolicy):
            # Already a constructed policy instance
            pass
        else:
            # Assume a policy class was provided
            policy_class = self.policy
            self.policy = policy_class(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
            )
        
        # Setup optimizer adapter after policy is created
        self._setup_optimizer()
        
    def _get_schedule_value(self, schedule: Union[float, str, Schedule]) -> float:
        """Get current value from schedule or return constant."""
        return apply_schedule_to_param(schedule, 1.0 - self._current_progress_remaining)
    
    def _ensure_float32_arrays(self, data: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Ensure all arrays are float32 for performance."""
        if not self.enforce_float32:
            return data
        
        result = {}
        for key, value in data.items():
            if isinstance(value, mx.array) and value.dtype != mx.float32:
                result[key] = value.astype(mx.float32)
            else:
                result[key] = value
        return result
    
    def _optimized_compute_loss(
        self, 
        rollout_data: Dict[str, MlxArray], 
        model: ActorCriticPolicy,
        clip_range: float, 
        clip_range_vf: Optional[float], 
        ent_coef: float
    ) -> MlxArray:
        """Compute PPO loss with optimizations."""
        actions = rollout_data["actions"]
        
        # Get current policy predictions
        values, log_prob, entropy = model.evaluate_actions(rollout_data["observations"], actions)
        values = mx.flatten(values)
        
        # Ensure float32 if requested
        if self.enforce_float32:
            values = values.astype(mx.float32)
            log_prob = log_prob.astype(mx.float32)
            if entropy is not None:
                entropy = entropy.astype(mx.float32)
        
        # Use JIT-optimized loss computation if available
        if self.use_jit and self.jit_ops and self.jit_ops.jit_enabled:
            return self.jit_ops.optimized_ppo_loss(
                values=values,
                log_probs=log_prob,
                old_log_probs=rollout_data["log_probs"],
                advantages=rollout_data["advantages"],
                returns=rollout_data["returns"],
                old_values=rollout_data["values"],
                entropy=entropy if entropy is not None else mx.zeros_like(values),
                clip_range=clip_range,
                clip_range_vf=clip_range_vf or 0.0,
                ent_coef=ent_coef,
                vf_coef=self.vf_coef
            )
        else:
            # Fallback to standard computation
            return self._compute_loss_standard(
                values, log_prob, entropy, rollout_data, clip_range, clip_range_vf, ent_coef
            )
    
    def _compute_loss_standard(
        self,
        values: mx.array,
        log_prob: mx.array,
        entropy: mx.array,
        rollout_data: Dict[str, MlxArray],
        clip_range: float,
        clip_range_vf: Optional[float],
        ent_coef: float
    ) -> mx.array:
        """Standard loss computation (fallback)."""
        # Normalize advantages
        advantages = rollout_data["advantages"]
        if len(advantages) > 1:
            advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
        
        # Ratio between old and new policy
        ratio = mx.exp(log_prob - rollout_data["log_probs"])
        
        # Clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
        
        # Value loss
        if clip_range_vf is None:
            value_loss = mx.mean((rollout_data["returns"] - values) ** 2)
        else:
            values_pred = rollout_data["values"] + mx.clip(
                values - rollout_data["values"], -clip_range_vf, clip_range_vf
            )
            value_loss_1 = (rollout_data["returns"] - values) ** 2
            value_loss_2 = (rollout_data["returns"] - values_pred) ** 2
            value_loss = mx.mean(mx.maximum(value_loss_1, value_loss_2))
        
        # Entropy loss
        entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
        
        # Total loss
        return policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss
    
    def _optimized_gradient_clipping(self, grads: Dict[str, mx.array]) -> tuple:
        """Optimized gradient clipping using JIT if available."""
        if self.max_grad_norm is None:
            # Just compute norm for logging
            grad_norm = float(mx.sqrt(sum(mx.sum(g ** 2) for g in grads.values() if g is not None)))
            return grads, grad_norm
        
        # Use JIT-optimized clipping if available
        if self.use_jit and self.jit_ops and self.jit_ops.jit_enabled:
            return self.jit_ops.optimized_grad_clipping(grads, self.max_grad_norm)
        else:
            # Fallback to standard clipping
            return clip_grad_norm(grads, self.max_grad_norm)
    
    def train(self) -> None:
        """
        Update policy using optimized PPO algorithm.
        """
        # Switch to training mode
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Get current clip ranges and entropy coefficient
        clip_range = self._get_schedule_value(self.clip_range)
        clip_range_vf = self._get_schedule_value(self.clip_range_vf) if self.clip_range_vf is not None else None
        ent_coef = self._get_schedule_value(self.ent_coef)
        
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        
        continue_training = True
        epochs_run = 0
        
        # Initialize parameters for functional updates
        params = self.policy.state_dict()
        
        # Ensure float32 if requested
        if self.enforce_float32:
            params = self._ensure_float32_arrays(params)
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            epochs_run = epoch + 1
            approx_kl_divs = []
            
            # Process all minibatches in the epoch
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Ensure float32 for rollout data
                if self.enforce_float32:
                    rollout_data = self._ensure_float32_arrays(rollout_data)
                
                # Define loss function for gradient computation
                def loss_fn(p):
                    # Load params temporarily for forward pass
                    self.policy.load_state_dict(p, strict=False)
                    return self._optimized_compute_loss(rollout_data, self.policy, clip_range, clip_range_vf, ent_coef)
                
                # Compute loss and gradients
                loss_val, grads = compute_loss_and_grads(loss_fn, params)
                
                # Optimized gradient clipping
                grads, grad_norm = self._optimized_gradient_clipping(grads)
                
                # Update parameters using optimizer adapter
                if self.optimizer_adapter is not None and self.optimizer_state is not None:
                    try:
                        params, self.optimizer_state = self.optimizer_adapter.update(
                            params, grads, self.optimizer_state
                        )
                    except Exception as e:
                        warnings.warn(
                            f"Optimizer update failed: {e}. Falling back to SGD.",
                            UserWarning
                        )
                        # Fallback to simple SGD
                        lr = 3e-4
                        params = {k: params[k] - lr * grads.get(k, 0) for k in params.keys()}
                else:
                    # Fallback to simple SGD if optimizer not initialized
                    lr = 3e-4
                    params = {k: params[k] - lr * grads.get(k, 0) for k in params.keys()}
                
                # Update policy state (minimal reloading)
                self.policy.load_state_dict(params, strict=False)
                mx.eval(list(params.values()))
                
                # Compute diagnostics for logging (reuse forward pass results)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data["observations"], rollout_data["actions"]
                )
                values = mx.flatten(values)
                
                # Policy loss diagnostics
                advantages = rollout_data["advantages"]
                if len(advantages) > 1:
                    advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)
                
                ratio = mx.exp(log_prob - rollout_data["log_probs"])
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
                
                # Calculate clip fraction for diagnostics
                clip_fraction = mx.mean((mx.abs(ratio - 1) > clip_range).astype(mx.float32))
                clip_fractions.append(float(clip_fraction))
                
                # Value loss
                if clip_range_vf is None:
                    value_loss = mx.mean((rollout_data["returns"] - values) ** 2)
                else:
                    values_pred = rollout_data["values"] + mx.clip(
                        values - rollout_data["values"], -clip_range_vf, clip_range_vf
                    )
                    value_loss_1 = (rollout_data["returns"] - values) ** 2
                    value_loss_2 = (rollout_data["returns"] - values_pred) ** 2
                    value_loss = mx.mean(mx.maximum(value_loss_1, value_loss_2))
                
                # Entropy loss
                entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
                
                # Store losses for logging
                pg_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                entropy_losses.append(float(entropy_loss))
                
                # Approximate KL divergence for early stopping
                log_ratio = log_prob - rollout_data["log_probs"]
                approx_kl_div = mx.mean((mx.exp(log_ratio) - 1) - log_ratio)
                approx_kl_divs.append(float(approx_kl_div))
                    
            # Early stopping based on KL divergence
            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                continue_training = False
                break
                
        self._n_updates += epochs_run
        explained_var = explained_variance(mx.array(self.rollout_buffer.values.flatten()), mx.array(self.rollout_buffer.returns.flatten()))
        
        # Log training metrics
        if self.verbose >= 1:
            print(f"Explained variance: {explained_var:.2f}")
            print(f"Policy loss: {np.mean(pg_losses):.3f}")
            print(f"Value loss: {np.mean(value_losses):.3f}")
            print(f"Entropy loss: {np.mean(entropy_losses):.3f}")
            print(f"Clip fraction: {np.mean(clip_fractions):.3f}")
            if approx_kl_divs:
                print(f"KL divergence: {np.mean(approx_kl_divs):.3f}")
    
    # Inherit other methods from the base PPO class
    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        """Use the same rollout collection as base PPO."""
        from mlx_baselines3.ppo.ppo import PPO
        # Temporarily create base PPO instance to reuse rollout method
        base_ppo = PPO.__new__(PPO)
        base_ppo.__dict__.update(self.__dict__)
        return base_ppo.collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
    
    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1, tb_log_name: str = "OptimizedPPO", reset_num_timesteps: bool = True, progress_bar: bool = False):
        """Use the same learning loop as base PPO."""
        from mlx_baselines3.ppo.ppo import PPO
        # Temporarily create base PPO instance to reuse learn method
        base_ppo = PPO.__new__(PPO)
        base_ppo.__dict__.update(self.__dict__)
        return base_ppo.learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving."""
        data = {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "use_jit": self.use_jit,
            "enforce_float32": self.enforce_float32,
        }
        return data
        
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load algorithm-specific data."""
        for key in ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", 
                   "clip_range", "clip_range_vf", "ent_coef", "vf_coef", 
                   "max_grad_norm", "target_kl", "use_jit", "enforce_float32"]:
            if key in data:
                setattr(self, key, data[key])
        
        # Reinitialize JIT optimizer if needed
        if getattr(self, 'use_jit', True):
            self.jit_ops = create_jit_optimizer()

    def _update_learning_rate(self, optimizer):
        """Update learning rate in the optimizer."""
        if hasattr(optimizer, 'learning_rate'):
            new_lr = self.lr_schedule(self._current_progress_remaining)
            optimizer.learning_rate = new_lr
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = {}
        if self.policy is not None:
            params["policy_parameters"] = dict(self.policy.named_parameters())
        return params
        
    def _set_parameters(self, params: Dict[str, Any], exact_match: bool = True) -> None:
        """Set algorithm parameters."""
        if "policy_parameters" in params and self.policy is not None:
            # Load policy parameters
            self.policy.load_state_dict(params["policy_parameters"], strict=exact_match)
