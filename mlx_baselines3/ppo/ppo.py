"""
Proximal Policy Optimization (PPO) algorithm implementation using MLX.

PPO is an on-policy algorithm that uses a clipped surrogate objective to prevent
large policy updates, leading to more stable training compared to vanilla policy gradients.
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


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) using MLX.
    
    Paper: https://arxiv.org/abs/1707.06347
    
    Args:
        policy: The policy model to use
        env: The environment to learn from
        learning_rate: The learning rate
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO surrogate loss
        clip_range_vf: Clipping parameter for value function loss
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: Maximum value for gradient clipping
        target_kl: Threshold for early stopping based on KL divergence
        device: Device to use for computation
        verbose: Verbosity level
        seed: Random generator seed
        **kwargs: Additional arguments
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
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # PPO hyperparameters (set before super init)
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
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ],
            **kwargs,
        )
        
        # Initialize training counters
        self._n_updates = 0
        
    def _setup_model(self) -> None:
        """Setup model: create policy, networks, buffers, optimizers, etc."""
        if self.verbose >= 1:
            print("Setting up PPO model...")
            
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
            print("PPO model setup complete")
        
    def _make_policy(self) -> None:
        """Create policy instance."""
        from mlx_baselines3.ppo.policies import get_ppo_policy_class
        
        if isinstance(self.policy, str):
            policy_class = get_ppo_policy_class(self.policy)
        else:
            policy_class = self.policy
            
        self.policy = policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
        )
        
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
        
    def _get_schedule_value(self, schedule: Union[float, Schedule]) -> float:
        """Get current value from schedule or return constant."""
        if callable(schedule):
            return schedule(self._current_progress_remaining)
        return schedule
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a rollout buffer.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        
        # Switch to evaluation mode
        self.policy.set_training_mode(False)
        
        n_steps = 0
        rollout_buffer.reset()
        
        while n_steps < n_rollout_steps:
            # Convert obs to MLX array
            obs_mlx = obs_as_mlx(self._last_obs)
            
            # Get action and value predictions
            actions, values, log_probs = self.policy.forward(obs_mlx)
                
            # Convert actions to numpy for environment
            actions_np = np.array(actions)
            
            # Clip actions for continuous action spaces
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions_np
                
            # Step environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            self.num_timesteps += env.num_envs
            
            # Give access to local variables
            if callback is not None:
                if not callback.on_step():
                    return False
                    
            self._update_info_buffer(infos)
            n_steps += 1
            
            # Store data in buffer (convert MLX arrays to numpy)
            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                np.array(values),
                np.array(log_probs),
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
        # Compute returns and advantages
        # Get value estimates for the last observations
        obs_mlx = obs_as_mlx(new_obs)
        values = self.policy.predict_values(obs_mlx)
            
        rollout_buffer.compute_returns_and_advantage(last_values=np.array(values), dones=dones)
        
        return True
        
    def train(self) -> None:
        """
        Update policy using PPO algorithm.
        """
        # Switch to training mode
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Get current clip and value function clip ranges
        clip_range = self._get_schedule_value(self.clip_range)
        clip_range_vf = self._get_schedule_value(self.clip_range_vf) if self.clip_range_vf is not None else None
        
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        
        continue_training = True
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data["actions"]
                
                # Get current policy predictions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data["observations"], actions
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
                
                # Calculate clip fraction for diagnostics
                clip_fraction = mx.mean((mx.abs(ratio - 1) > clip_range).astype(mx.float32))
                clip_fractions.append(float(clip_fraction))
                
                # Value loss
                if clip_range_vf is None:
                    # No clipping
                    value_loss = mx.mean((rollout_data["returns"] - values) ** 2)
                else:
                    # Clipped value loss
                    values_pred = rollout_data["values"] + mx.clip(
                        values - rollout_data["values"], -clip_range_vf, clip_range_vf
                    )
                    value_loss_1 = (rollout_data["returns"] - values) ** 2
                    value_loss_2 = (rollout_data["returns"] - values_pred) ** 2
                    value_loss = mx.mean(mx.maximum(value_loss_1, value_loss_2))
                
                # Entropy loss
                entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
                
                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Optimization step
                def compute_loss_fn(model):
                    return self._compute_loss(rollout_data, model, clip_range, clip_range_vf)
                
                loss_and_grad_fn = mx.value_and_grad(compute_loss_fn)
                loss_val, grads = loss_and_grad_fn(self.policy)
                
                # Clip gradients
                if self.max_grad_norm is not None:
                    grads = self._clip_gradients(grads, self.max_grad_norm)
                
                self.policy.optimizer.update(self.policy, grads)
                mx.eval(self.policy.parameters())
                
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
                
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        
        # Log training metrics
        if self.verbose >= 1:
            print(f"Explained variance: {explained_var:.2f}")
            print(f"Policy loss: {np.mean(pg_losses):.3f}")
            print(f"Value loss: {np.mean(value_losses):.3f}")
            print(f"Entropy loss: {np.mean(entropy_losses):.3f}")
            print(f"Clip fraction: {np.mean(clip_fractions):.3f}")
            if approx_kl_divs:
                print(f"KL divergence: {np.mean(approx_kl_divs):.3f}")
                
    def _compute_loss(self, rollout_data: Dict[str, MlxArray], model, clip_range: float, clip_range_vf: Optional[float]) -> MlxArray:
        """Compute the total loss for PPO."""
        actions = rollout_data["actions"]
        
        # Get current policy predictions
        values, log_prob, entropy = model.evaluate_actions(rollout_data["observations"], actions)
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
        entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0
        
        # Total loss
        return policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        
    def _clip_gradients(self, grads: Dict[str, MlxArray], max_norm: float) -> Dict[str, MlxArray]:
        """Clip gradients by global norm."""
        # Calculate global norm
        total_norm = 0.0
        for grad in grads.values():
            if grad is not None:
                total_norm += mx.sum(grad ** 2)
        total_norm = mx.sqrt(total_norm)
        
        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            grads = {k: grad * clip_coef if grad is not None else grad for k, grad in grads.items()}
            
        return grads
        
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Return a trained model.
        
        Args:
            total_timesteps: The total number of samples (env steps) to train on
            callback: Callback(s) called at every step
            log_interval: The number of episodes before logging
            tb_log_name: The name of the run for tensorboard log
            reset_num_timesteps: Whether to reset timesteps when learn restarts
            progress_bar: Whether to display a progress bar
            
        Returns:
            The trained model
        """
        iteration = 0
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        
        callback.on_training_start(locals(), globals())
        
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            
            if not continue_training:
                break
                
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time() - self.start_time), 1e-8)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                if self.verbose >= 1:
                    print(f"------------------------------------")
                    print(f"| rollout/              |         |")
                    print(f"|    ep_len_mean        | {np.mean([ep_info['l'] for ep_info in self.ep_info_buffer]):.1f}     |")
                    print(f"|    ep_rew_mean        | {np.mean([ep_info['r'] for ep_info in self.ep_info_buffer]):.1f}     |")
                    print(f"| time/                 |         |")
                    print(f"|    fps                | {fps}       |")
                    print(f"|    iterations         | {iteration}       |")
                    print(f"|    time_elapsed       | {int(time_elapsed)}       |")
                    print(f"|    total_timesteps    | {self.num_timesteps}       |")
                    print(f"------------------------------------")
            
            self.train()
            
        callback.on_training_end()
        
        return self
        
    def _setup_learn(self, total_timesteps: int, callback, reset_num_timesteps: bool, tb_log_name: str, progress_bar: bool):
        """Setup learning process."""
        import time
        
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        self.start_time = time.time()
        
        # Initialize callback (simple placeholder for now)
        if callback is None:
            from types import SimpleNamespace
            callback = SimpleNamespace()
            callback.on_training_start = lambda *args, **kwargs: None
            callback.on_step = lambda *args, **kwargs: True
            callback.on_training_end = lambda *args, **kwargs: None
        
        # Reset environment
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        
        return total_timesteps, callback
        
    def _update_info_buffer(self, infos):
        """Update the info buffer with episode information."""
        if not hasattr(self, 'ep_info_buffer'):
            self.ep_info_buffer = []
            
        for info in infos:
            if isinstance(info, dict):
                if 'episode' in info:
                    ep_info = info['episode']
                    if 'r' in ep_info and 'l' in ep_info:
                        self.ep_info_buffer.append({'r': ep_info['r'], 'l': ep_info['l']})
                        # Keep only last 100 episodes
                        if len(self.ep_info_buffer) > 100:
                            self.ep_info_buffer.pop(0)
                            
    def _update_learning_rate(self, optimizer):
        """Update learning rate in the optimizer."""
        if hasattr(optimizer, 'learning_rate'):
            new_lr = self.lr_schedule(self._current_progress_remaining)
            optimizer.learning_rate = new_lr
        
    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving."""
        return {
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
        }
        
    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load algorithm-specific data."""
        for key in ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", 
                   "clip_range", "clip_range_vf", "ent_coef", "vf_coef", 
                   "max_grad_norm", "target_kl"]:
            if key in data:
                setattr(self, key, data[key])
