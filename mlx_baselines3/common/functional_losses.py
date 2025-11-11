"""
Functional loss computation utilities for improved performance.

This module provides pure functional loss computations that avoid
parameter reloading and improve training efficiency.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set, Tuple, cast

import mlx.core as mx
from mlx_baselines3.common.type_aliases import MlxArray


def ppo_functional_loss(
    params: Dict[str, mx.array],
    rollout_data: Dict[str, MlxArray],
    policy_apply_fn: Callable,
    clip_range: float,
    clip_range_vf: Optional[float],
    ent_coef: float,
    vf_coef: float,
) -> mx.array:
    """
    Compute PPO loss functionally without modifying policy state.

    Args:
        params: Parameter dictionary
        rollout_data: Batch of rollout data
        policy_apply_fn: Function to apply policy with params
        clip_range: PPO clipping range
        clip_range_vf: Value function clipping range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient

    Returns:
        Total loss as MLX array
    """
    actions = rollout_data["actions"]

    # Get current policy predictions using functional approach
    values, log_prob, entropy = policy_apply_fn(
        params, rollout_data["observations"], actions
    )
    values = mx.flatten(values)

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
    entropy_loss = (
        -mx.mean(entropy) if entropy is not None else mx.array(0.0, dtype=mx.float32)
    )

    # Total loss
    return cast(
        mx.array,
        policy_loss + ent_coef * entropy_loss + vf_coef * value_loss,
    )


def a2c_functional_loss(
    params: Dict[str, mx.array],
    rollout_data: Dict[str, MlxArray],
    policy_apply_fn: Callable,
    ent_coef: float,
    vf_coef: float,
) -> mx.array:
    """
    Compute A2C loss functionally without modifying policy state.

    Args:
        params: Parameter dictionary
        rollout_data: Batch of rollout data
        policy_apply_fn: Function to apply policy with params
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient

    Returns:
        Total loss as MLX array
    """
    actions = rollout_data["actions"]

    # Get current policy predictions using functional approach
    values, log_prob, entropy = policy_apply_fn(
        params, rollout_data["observations"], actions
    )
    values = mx.flatten(values)

    # Normalize advantages
    advantages = rollout_data["advantages"]
    if len(advantages) > 1:
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

    # Policy gradient loss (no clipping for A2C)
    policy_loss = -mx.mean(advantages * log_prob)

    # Value loss
    value_loss = mx.mean((rollout_data["returns"] - values) ** 2)

    # Entropy loss
    entropy_loss = (
        -mx.mean(entropy) if entropy is not None else mx.array(0.0, dtype=mx.float32)
    )

    # Total loss
    return cast(
        mx.array,
        policy_loss + ent_coef * entropy_loss + vf_coef * value_loss,
    )


def dqn_functional_loss(
    params: Dict[str, mx.array],
    batch_data: Dict[str, MlxArray],
    q_network_apply_fn: Callable,
    target_params: Dict[str, mx.array],
    gamma: float,
    huber_loss: bool = True,
) -> mx.array:
    """
    Compute DQN loss functionally.

    Args:
        params: Q-network parameters
        batch_data: Batch of experience
        q_network_apply_fn: Function to apply Q-network with params
        target_params: Target network parameters
        gamma: Discount factor
        huber_loss: Whether to use Huber loss instead of MSE

    Returns:
        Q-learning loss
    """
    obs = batch_data["observations"]
    actions = batch_data["actions"]
    rewards = batch_data["rewards"]
    next_obs = batch_data["next_observations"]
    terminated = batch_data["terminated"]

    # Current Q-values
    q_values = q_network_apply_fn(params, obs)
    current_q_values = mx.take_along_axis(q_values, actions.astype(mx.int32), axis=-1)
    current_q_values = mx.squeeze(current_q_values, axis=-1)

    # Target Q-values (no gradient through target network)
    next_q_values = mx.stop_gradient(q_network_apply_fn(target_params, next_obs))
    max_next_q_values = mx.max(next_q_values, axis=-1)
    target_q_values = rewards + gamma * (1 - terminated) * max_next_q_values

    # Loss computation
    if huber_loss:
        # Huber loss for more stable training
        delta = current_q_values - target_q_values
        huber_delta = 1.0
        loss = mx.where(
            mx.abs(delta) < huber_delta,
            0.5 * delta**2,
            huber_delta * (mx.abs(delta) - 0.5 * huber_delta),
        )
        return mx.mean(loss)
    else:
        # MSE loss
        return mx.mean((current_q_values - target_q_values) ** 2)


def sac_functional_loss(
    params: Dict[str, mx.array],
    batch_data: Dict[str, MlxArray],
    policy_apply_fn: Callable,
    critic_apply_fn: Callable,
    target_critic_params: Dict[str, mx.array],
    log_alpha: mx.array,
    gamma: float,
    tau: float = 0.005,
    target_entropy: float = 0.0,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Compute SAC losses functionally.

    Args:
        params: Combined parameters dict with actor and critic params
        batch_data: Batch of experience
        policy_apply_fn: Function to apply policy
        critic_apply_fn: Function to apply critic
        target_critic_params: Target critic parameters
        log_alpha: Log of temperature parameter
        gamma: Discount factor
        tau: Soft update coefficient

    Returns:
        Tuple of (actor_loss, critic_loss, alpha_loss)
    """
    obs = batch_data["observations"]
    actions = batch_data["actions"]
    rewards = batch_data["rewards"]
    next_obs = batch_data["next_observations"]
    terminated = batch_data["terminated"]

    alpha = mx.exp(log_alpha)

    # Actor loss
    new_actions, new_log_probs = policy_apply_fn(params, obs)
    q1_new, q2_new = critic_apply_fn(params, obs, new_actions)
    min_q_new = mx.minimum(q1_new, q2_new)
    actor_loss = mx.mean(alpha * new_log_probs - min_q_new)

    # Critic loss
    next_actions, next_log_probs = policy_apply_fn(params, next_obs)
    next_actions = mx.stop_gradient(next_actions)
    next_log_probs = mx.stop_gradient(next_log_probs)
    target_q1, target_q2 = critic_apply_fn(target_critic_params, next_obs, next_actions)
    target_q = mx.minimum(target_q1, target_q2) - alpha * next_log_probs
    target_q_values = mx.stop_gradient(rewards + gamma * (1 - terminated) * target_q)

    current_q1, current_q2 = critic_apply_fn(params, obs, actions)
    critic1_loss = mx.mean((current_q1 - target_q_values) ** 2)
    critic2_loss = mx.mean((current_q2 - target_q_values) ** 2)
    critic_loss = critic1_loss + critic2_loss

    # Alpha loss (temperature)
    alpha_loss = -mx.mean(log_alpha * (new_log_probs + target_entropy))

    return actor_loss, critic_loss, alpha_loss


def batch_efficient_collate(
    batch_list: list[Dict[str, Any]], device: str = "cpu"
) -> Dict[str, mx.array]:
    """
    Efficiently collate a batch of samples into contiguous arrays.

    Avoids Python loops on inner dimensions and ensures contiguous memory layout.

    Args:
        batch_list: List of sample dictionaries
        device: Target device for arrays

    Returns:
        Dictionary with batched arrays
    """
    if not batch_list:
        return {}

    # Get keys from first sample
    keys = batch_list[0].keys()
    batch_dict: Dict[str, mx.array] = {}

    for key in keys:
        # Stack all samples for this key
        samples = [sample[key] for sample in batch_list]

        # Convert to MLX arrays if needed and stack
        if isinstance(samples[0], mx.array):
            # Already MLX arrays - stack directly
            batch_dict[key] = mx.stack(samples, axis=0)
        elif isinstance(samples[0], (int, float)):
            # Scalar values - convert to array then stack
            arrays = [mx.array([s], dtype=mx.float32) for s in samples]
            batch_dict[key] = mx.concatenate(arrays, axis=0)
        else:
            # Numpy or other - convert then stack
            arrays = [mx.array(s, dtype=mx.float32) for s in samples]
            batch_dict[key] = mx.stack(arrays, axis=0)

    return batch_dict


def ensure_float32_dtype(
    arrays: Dict[str, mx.array], excluded_keys: Optional[Set[str]] = None
) -> Dict[str, mx.array]:
    """
    Ensure all arrays in dictionary are float32 unless excluded.

    Args:
        arrays: Dictionary of arrays
        excluded_keys: Keys to exclude from dtype conversion

    Returns:
        Dictionary with float32 arrays
    """
    active_exclusions: Set[str] = excluded_keys or set()
    result: Dict[str, mx.array] = {}

    for key, array in arrays.items():
        if key in active_exclusions:
            result[key] = array
        elif array.dtype != mx.float32:
            result[key] = array.astype(mx.float32)
        else:
            result[key] = array

    return result


def compute_gradient_norm(grads: Dict[str, mx.array]) -> float:
    """
    Efficiently compute gradient norm without extra allocations.

    Args:
        grads: Gradient dictionary

    Returns:
        Gradient norm as float
    """
    norm_squared = mx.array(0.0, dtype=mx.float32)
    for grad in grads.values():
        if grad is not None:
            norm_squared = norm_squared + mx.sum(grad.astype(mx.float32) ** 2)

    return float(mx.sqrt(norm_squared))
