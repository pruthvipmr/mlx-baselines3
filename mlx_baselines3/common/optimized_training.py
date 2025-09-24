"""
Optimized training loops with reduced parameter reloads and improved performance.

This module provides optimized training implementations that minimize
parameter reloading and use functional computation patterns.
"""

import mlx.core as mx
from typing import Dict, Any, Tuple, Optional

from mlx_baselines3.common.type_aliases import MlxArray
from mlx_baselines3.common.optimizers import compute_loss_and_grads, clip_grad_norm
from mlx_baselines3.common.functional_losses import (
    ppo_functional_loss,
    ensure_float32_dtype,
    batch_efficient_collate,
)


class OptimizedPPOTrainer:
    """
    Optimized PPO trainer with minimal parameter reloading.

    This trainer uses functional loss computation to avoid repeatedly
    loading parameters into the policy, significantly improving performance.
    """

    def __init__(
        self,
        policy,
        optimizer_adapter,
        max_grad_norm: Optional[float] = None,
        enforce_float32: bool = True,
    ):
        """
        Initialize the optimized trainer.

        Args:
            policy: Policy instance
            optimizer_adapter: Optimizer adapter
            max_grad_norm: Maximum gradient norm for clipping
            enforce_float32: Whether to enforce float32 dtypes
        """
        self.policy = policy
        self.optimizer_adapter = optimizer_adapter
        self.max_grad_norm = max_grad_norm
        self.enforce_float32 = enforce_float32

        # Create functional apply function once
        self.policy_apply_fn = self.policy.create_functional_apply_fn()

        # Track training statistics
        self.training_stats = {
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "clip_fractions": [],
            "grad_norms": [],
            "approx_kl_divs": [],
        }

    def optimized_train_step(
        self,
        params: Dict[str, mx.array],
        optimizer_state: Dict[str, Any],
        rollout_data: Dict[str, MlxArray],
        clip_range: float,
        clip_range_vf: Optional[float],
        ent_coef: float,
        vf_coef: float,
    ) -> Tuple[Dict[str, mx.array], Dict[str, Any], Dict[str, float]]:
        """
        Perform one optimized training step with minimal parameter reloading.

        Args:
            params: Current parameters
            optimizer_state: Current optimizer state
            rollout_data: Batch of rollout data
            clip_range: PPO clipping range
            clip_range_vf: Value function clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient

        Returns:
            Tuple of (new_params, new_optimizer_state, step_stats)
        """
        # Ensure float32 dtypes if requested
        if self.enforce_float32:
            params = ensure_float32_dtype(params)
            rollout_data = ensure_float32_dtype(rollout_data)

        # Define pure functional loss
        def pure_loss_fn(p: Dict[str, mx.array]) -> mx.array:
            return ppo_functional_loss(
                params=p,
                rollout_data=rollout_data,
                policy_apply_fn=self.policy_apply_fn,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
            )

        # Compute loss and gradients functionally
        loss_val, grads = compute_loss_and_grads(pure_loss_fn, params)

        # Clip gradients if specified
        if self.max_grad_norm is not None:
            grads, grad_norm = clip_grad_norm(grads, self.max_grad_norm)
        else:
            grad_norm = float(
                mx.sqrt(sum(mx.sum(g**2) for g in grads.values() if g is not None))
            )

        # Update parameters using optimizer
        new_params, new_optimizer_state = self.optimizer_adapter.update(
            params, grads, optimizer_state
        )

        # Compute diagnostics (only load params once at the end)
        step_stats = self._compute_step_diagnostics(
            new_params, rollout_data, clip_range, loss_val, grad_norm
        )

        return new_params, new_optimizer_state, step_stats

    def _compute_step_diagnostics(
        self,
        params: Dict[str, mx.array],
        rollout_data: Dict[str, MlxArray],
        clip_range: float,
        loss_val: mx.array,
        grad_norm: float,
    ) -> Dict[str, float]:
        """
        Compute diagnostic statistics for the training step.

        Args:
            params: Current parameters
            rollout_data: Batch data
            clip_range: Clipping range
            loss_val: Total loss value
            grad_norm: Gradient norm

        Returns:
            Dictionary of diagnostic statistics
        """
        # Evaluate policy with current params (single parameter load)
        values, log_prob, entropy = self.policy_apply_fn(
            params, rollout_data["observations"], rollout_data["actions"]
        )
        values = mx.flatten(values)

        # Compute individual loss components for logging
        advantages = rollout_data["advantages"]
        if len(advantages) > 1:
            advantages = (advantages - mx.mean(advantages)) / (
                mx.std(advantages) + 1e-8
            )

        # Policy loss diagnostics
        ratio = mx.exp(log_prob - rollout_data["log_probs"])
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * mx.clip(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))

        # Value loss
        value_loss = mx.mean((rollout_data["returns"] - values) ** 2)

        # Entropy loss
        entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0

        # Clip fraction
        clip_fraction = mx.mean((mx.abs(ratio - 1) > clip_range).astype(mx.float32))

        # Approximate KL divergence
        log_ratio = log_prob - rollout_data["log_probs"]
        approx_kl_div = mx.mean((mx.exp(log_ratio) - 1) - log_ratio)

        return {
            "total_loss": float(loss_val),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_loss": float(entropy_loss),
            "clip_fraction": float(clip_fraction),
            "grad_norm": grad_norm,
            "approx_kl_div": float(approx_kl_div),
        }

    def train_epoch(
        self,
        params: Dict[str, mx.array],
        optimizer_state: Dict[str, Any],
        rollout_buffer,
        batch_size: int,
        clip_range: float,
        clip_range_vf: Optional[float],
        ent_coef: float,
        vf_coef: float,
        target_kl: Optional[float] = None,
        verbose: int = 0,
    ) -> Tuple[Dict[str, mx.array], Dict[str, Any], Dict[str, list], bool]:
        """
        Train for one epoch with optimized parameter handling.

        Args:
            params: Current parameters
            optimizer_state: Current optimizer state
            rollout_buffer: Rollout buffer with data
            batch_size: Batch size for training
            clip_range: PPO clipping range
            clip_range_vf: Value function clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            target_kl: Target KL divergence for early stopping
            verbose: Verbosity level

        Returns:
            Tuple of (new_params, new_optimizer_state, epoch_stats, continue_training)
        """
        epoch_stats = {
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "clip_fractions": [],
            "grad_norms": [],
            "approx_kl_divs": [],
        }

        continue_training = True

        # Process all minibatches
        for rollout_data in rollout_buffer.get(batch_size):
            # Perform optimized training step
            params, optimizer_state, step_stats = self.optimized_train_step(
                params=params,
                optimizer_state=optimizer_state,
                rollout_data=rollout_data,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
            )

            # Accumulate statistics
            for key, value in step_stats.items():
                if key in epoch_stats:
                    epoch_stats[key].append(value)

            # Check for early stopping
            if target_kl is not None and step_stats["approx_kl_div"] > 1.5 * target_kl:
                if verbose >= 1:
                    kl_div = step_stats["approx_kl_div"]
                    print(f"Early stopping due to KL divergence: {kl_div:.3f}")
                continue_training = False
                break

        # Update policy with final parameters (single load at epoch end)
        self.policy.load_state_dict(params, strict=False)
        mx.eval(list(params.values()))

        return params, optimizer_state, epoch_stats, continue_training


class BatchOptimizer:
    """
    Optimized batch processing utilities.
    """

    @staticmethod
    def efficient_batch_collation(
        samples: list, target_dtype: mx.Dtype = mx.float32
    ) -> Dict[str, mx.array]:
        """
        Efficiently collate batch samples with contiguous memory layout.

        Args:
            samples: List of sample dictionaries
            target_dtype: Target dtype for arrays

        Returns:
            Batched dictionary with contiguous arrays
        """
        return batch_efficient_collate(samples)

    @staticmethod
    def preprocess_batch(
        batch: Dict[str, mx.array],
        normalize_advantages: bool = True,
        advantage_key: str = "advantages",
    ) -> Dict[str, mx.array]:
        """
        Preprocess batch data for training.

        Args:
            batch: Batch dictionary
            normalize_advantages: Whether to normalize advantages
            advantage_key: Key for advantages in batch

        Returns:
            Preprocessed batch
        """
        if normalize_advantages and advantage_key in batch:
            advantages = batch[advantage_key]
            if len(advantages) > 1:
                batch[advantage_key] = (advantages - mx.mean(advantages)) / (
                    mx.std(advantages) + 1e-8
                )

        return batch


def create_optimized_ppo_trainer(
    policy,
    optimizer_adapter,
    max_grad_norm: Optional[float] = None,
    enforce_float32: bool = True,
) -> OptimizedPPOTrainer:
    """
    Factory function to create an optimized PPO trainer.

    Args:
        policy: Policy instance
        optimizer_adapter: Optimizer adapter
        max_grad_norm: Maximum gradient norm for clipping
        enforce_float32: Whether to enforce float32 dtypes

    Returns:
        Optimized PPO trainer instance
    """
    return OptimizedPPOTrainer(
        policy=policy,
        optimizer_adapter=optimizer_adapter,
        max_grad_norm=max_grad_norm,
        enforce_float32=enforce_float32,
    )
