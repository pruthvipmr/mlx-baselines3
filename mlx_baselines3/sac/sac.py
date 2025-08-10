"""
Soft Actor-Critic (SAC) algorithm implementation using MLX.

Based on the original SAC papers:
- "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
  https://arxiv.org/abs/1801.01290
- "Soft Actor-Critic Algorithms and Applications"
  https://arxiv.org/abs/1812.05905
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_baselines3.common.base_class import OffPolicyAlgorithm
from mlx_baselines3.common.buffers import ReplayBuffer
from mlx_baselines3.common.optimizers import AdamAdapter, compute_loss_and_grads, clip_grad_norm
from mlx_baselines3.common.schedules import get_schedule_fn
from mlx_baselines3.common.type_aliases import GymEnv, Schedule
from mlx_baselines3.common.utils import (
    get_schedule_fn,
    obs_as_mlx,
    polyak_update,
    safe_mean,
)
from mlx_baselines3.sac.policies import SACPolicy


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) algorithm.
    
    Paper: https://arxiv.org/abs/1801.01290 and https://arxiv.org/abs/1812.05905
    
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It uses:
    - A stochastic actor with tanh-squashed Gaussian policy
    - Twin critics to reduce overestimation bias
    - Automatic entropy coefficient tuning
    
    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from
        learning_rate: Learning rate for the optimizers
        buffer_size: Size of the replay buffer
        learning_starts: How many steps to collect before training starts
        batch_size: Batch size for training
        tau: The soft update coefficient for target networks
        gamma: Discount factor
        train_freq: Update the model every `train_freq` steps
        gradient_steps: How many gradient steps to do after each rollout
        action_noise: Action noise type for exploration
        replay_buffer_class: Replay buffer class to use
        replay_buffer_kwargs: Keyword arguments for replay buffer
        optimize_memory_usage: Enable memory efficient storage
        ent_coef: Entropy regularization coefficient ("auto" for automatic tuning)
        target_entropy: Target entropy for automatic entropy coefficient tuning
        use_sde: Whether to use State Dependent Exploration
        sde_sample_freq: Sample new noise every n steps when using SDE
        use_sde_at_warmup: Whether to use SDE during warmup phase
        stats_window_size: Window size for logging statistics
        tensorboard_log: Path to tensorboard log directory
        policy_kwargs: Additional arguments for the policy
        verbose: Verbosity level
        seed: Random seed
        device: Device to use for computation
        _init_setup_model: Whether to build the network immediately
    """
    
    policy_aliases: Dict[str, str] = {
        "MlpPolicy": "MlpPolicy",  # Will be resolved in _setup_model
    }

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        # Set SAC-specific parameters before calling super().__init__()
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.use_sde_at_warmup = use_sde_at_warmup
        self.optimize_memory_usage = optimize_memory_usage
        self.policy_kwargs = policy_kwargs or {}
        self.stats_window_size = stats_window_size
        self.tensorboard_log = tensorboard_log
        self.replay_buffer = None
        
        # Set entropy coefficient parameters before super().__init__() because _setup_model() is called from there
        if ent_coef == "auto":
            # Target entropy is -dim(A) for continuous action spaces (will be set after env is available)
            self.target_entropy = target_entropy  # Will be converted to float later
            # Learnable log entropy coefficient
            self.log_ent_coef = mx.array([0.0])  # Initialize to log(1) = 0
            self.ent_coef_optimizer = None  # Will be set in _setup_model
        else:
            self.ent_coef = float(ent_coef)
            self.target_entropy = None
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
        
        super().__init__(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
        seed=seed,
        supported_action_spaces=[gym.spaces.Box],
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        )
        
        # Training/update counters
        self._n_updates = 0
        
        # SAC is only for continuous action spaces
        if not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("SAC only supports continuous action spaces (Box)")

        # Finalize entropy coefficient setup now that action_space is available
        if ent_coef == "auto":
            if target_entropy == "auto":
                self.target_entropy = -float(self.action_space.shape[0])
            else:
                self.target_entropy = float(target_entropy)

        # Note: _setup_model() is already called by BaseAlgorithm.__init__()

    def _setup_model(self) -> None:
        """Set up the model and optimizers."""
        # Store original policy spec before it gets modified
        original_policy = self.policy
        
        # Create policy networks
        if isinstance(original_policy, str):
            if original_policy == "MlpPolicy":
                from mlx_baselines3.sac.policies import MlpPolicy as SACMlpPolicy
                policy_class = SACMlpPolicy
            else:
                raise ValueError(f"Unknown policy: {original_policy}")
        else:
            policy_class = original_policy

        # Create policy with SAC-specific network architecture
        policy_instance = policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = policy_instance
        
        self._create_aliases()
        
        # Set up replay buffer
        if self.replay_buffer_class is None:
            self.replay_buffer_class = ReplayBuffer
            
        if self.replay_buffer is None:
            # Default kwargs for replay buffer
            replay_buffer_kwargs = {} if self.replay_buffer_kwargs is None else self.replay_buffer_kwargs.copy()
            replay_buffer_kwargs.update({
                "optimize_memory_usage": self.optimize_memory_usage,
            })
            
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                **replay_buffer_kwargs,
            )

        # Create optimizers
        self.actor_optimizer = AdamAdapter(learning_rate=self.learning_rate)
        self.critic_optimizer = AdamAdapter(learning_rate=self.learning_rate)
        
        # Entropy coefficient optimizer (if auto-tuning)
        if self.ent_coef == "auto":
            self.ent_coef_optimizer = AdamAdapter(learning_rate=self.learning_rate)

        # Initialize optimizer states
        self._initialize_optimizers()

        # Copy policy networks to target networks
        self.policy._build_target_networks()

    def _initialize_optimizers(self) -> None:
        """Initialize optimizer states."""
        # Get actor parameters (features_extractor + actor_net + mu + log_std)
        actor_params = {}
        for name, param in self.policy.parameters().items():
            if any(component in name for component in ["features_extractor", "actor_net", "mu", "log_std"]):
                actor_params[name] = param
        
        # Get critic parameters (q_net_0 and q_net_1)
        critic_params = {}
        for name, param in self.policy.parameters().items():
            if any(f"q_net_{i}" in name for i in range(self.policy.n_critics)) and "target" not in name:
                critic_params[name] = param

        # Initialize optimizer states
        self.actor_optimizer_state = self.actor_optimizer.init_state(actor_params)
        self.critic_optimizer_state = self.critic_optimizer.init_state(critic_params)
        
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            ent_params = {"log_ent_coef": self.log_ent_coef}
            self.ent_coef_optimizer_state = self.ent_coef_optimizer.init_state(ent_params)

    def _create_aliases(self) -> None:
        """Create aliases for easy access to policy components."""
        self.actor = self.policy
        self.critic = self.policy
        self.critic_target = self.policy

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update policy using gradient steps on batches from the replay buffer.
        
        Args:
            gradient_steps: Number of gradient steps to take
            batch_size: Size of training batches
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update learning rate according to schedule
        self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])
        if self.ent_coef_optimizer is not None:
            self._update_learning_rate([self.ent_coef_optimizer])

        actor_losses = []
        critic_losses = []
        ent_coef_losses = []
        ent_coefs = []

        for _ in range(gradient_steps):
            # Sample a batch from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
            
            # Convert to MLX arrays
            observations = obs_as_mlx(replay_data["observations"])
            actions = mx.array(replay_data["actions"])
            next_observations = obs_as_mlx(replay_data["next_observations"])
            rewards = mx.array(replay_data["rewards"]).flatten()
            dones = mx.array(replay_data["dones"]).flatten()

            # Current entropy coefficient
            if self.ent_coef == "auto":
                ent_coef = mx.exp(self.log_ent_coef)
            else:
                ent_coef = mx.array([self.ent_coef])

            # Update critics
            critic_loss = self._update_critics(
                observations, actions, next_observations, rewards, dones, ent_coef
            )
            critic_losses.append(float(critic_loss))

            # Update actor
            actor_loss = self._update_actor(observations, ent_coef)
            actor_losses.append(float(actor_loss))

            # Update entropy coefficient (if auto-tuning)
            if self.ent_coef == "auto":
                ent_coef_loss = self._update_entropy_coef(observations)
                ent_coef_losses.append(float(ent_coef_loss))
                ent_coefs.append(float(ent_coef))

            # Update target networks
            self._update_target_networks()

        # Increment number of updates
        self._n_updates += gradient_steps

        # Store training stats (optional logger)
        if hasattr(self, "logger"):
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/actor_loss", safe_mean(actor_losses))
            self.logger.record("train/critic_loss", safe_mean(critic_losses))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", safe_mean(ent_coef_losses))
                self.logger.record("train/ent_coef", safe_mean(ent_coefs))
            elif self.ent_coef != "auto":
                self.logger.record("train/ent_coef", self.ent_coef)

    def _update_critics(
        self,
        observations: mx.array,
        actions: mx.array,
        next_observations: mx.array,
        rewards: mx.array,
        dones: mx.array,
        ent_coef: mx.array,
    ) -> mx.array:
        """Update critic networks."""
        
        def critic_loss_fn(critic_params: Dict[str, mx.array]) -> mx.array:
            """Compute critic loss."""
            # Update policy parameters for forward pass
            old_params = self.policy.parameters()
            temp_params = {**old_params, **critic_params}
            self.policy.load_state_dict(temp_params, strict=False)
            
            # Compute current Q-values
            features = self.policy.extract_features(observations)
            current_q_values = self.policy.critic_forward(features, actions)
            
            # Compute target Q-values (no explicit no_grad in MLX)
            next_features = self.policy.extract_features(next_observations)
            next_actions, next_log_probs, _ = self.policy.actor_forward(next_features)
            target_q_values = self.policy.critic_target_forward(next_features, next_actions)
            
            # Take minimum of target Q-values (clipped double Q-learning)
            target_q = mx.minimum(target_q_values[0], target_q_values[1])
            
            # Add entropy term to target Q-values
            target_q = target_q - ent_coef * next_log_probs.reshape(-1, 1)
            
            # Compute target values with Bellman backup
            target_q = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * self.gamma * target_q
            
            # Compute losses for both critics
            critic_losses = []
            for current_q in current_q_values:
                critic_loss = mx.mean((current_q - target_q) ** 2)
                critic_losses.append(critic_loss)
            
            total_critic_loss = sum(critic_losses)
            
            # Restore original parameters
            self.policy.load_state_dict(old_params, strict=False)
            
            return total_critic_loss

        # Get critic parameters
        critic_params = {}
        for name, param in self.policy.parameters().items():
            if any(f"q_net_{i}" in name for i in range(self.policy.n_critics)) and "target" not in name:
                critic_params[name] = param

        # Compute loss and gradients
        loss, gradients = compute_loss_and_grads(critic_loss_fn, critic_params)
        
        # Clip gradients
        gradients = clip_grad_norm(gradients, max_norm=10.0)
        
        # Update parameters
        updated_params, self.critic_optimizer_state = self.critic_optimizer.update(
            critic_params, gradients, self.critic_optimizer_state
        )
        
        # Update policy with new parameters
        all_params = self.policy.parameters()
        all_params.update(updated_params)
        self.policy.load_state_dict(all_params, strict=False)
        
        return loss

    def _update_actor(self, observations: mx.array, ent_coef: mx.array) -> mx.array:
        """Update actor network."""
        
        def actor_loss_fn(actor_params: Dict[str, mx.array]) -> mx.array:
            """Compute actor loss."""
            # Update policy parameters for forward pass
            old_params = self.policy.parameters()
            temp_params = {**old_params, **actor_params}
            self.policy.load_state_dict(temp_params, strict=False)
            
            # Forward pass through actor
            features = self.policy.extract_features(observations)
            actions, log_probs, _ = self.policy.actor_forward(features)
            
            # Compute Q-values for sampled actions
            q_values = self.policy.critic_forward(features, actions)
            
            # Take minimum of Q-values
            min_q = mx.minimum(q_values[0], q_values[1])
            
            # Actor loss: maximize Q(s, a) - alpha * log_prob(a)
            actor_loss = mx.mean(ent_coef * log_probs - min_q.flatten())
            
            # Restore original parameters
            self.policy.load_state_dict(old_params, strict=False)
            
            return actor_loss

        # Get actor parameters
        actor_params = {}
        for name, param in self.policy.parameters().items():
            if any(component in name for component in ["features_extractor", "actor_net", "mu", "log_std"]):
                actor_params[name] = param

        # Compute loss and gradients
        loss, gradients = compute_loss_and_grads(actor_loss_fn, actor_params)
        
        # Clip gradients
        gradients = clip_grad_norm(gradients, max_norm=10.0)
        
        # Update parameters
        updated_params, self.actor_optimizer_state = self.actor_optimizer.update(
            actor_params, gradients, self.actor_optimizer_state
        )
        
        # Update policy with new parameters
        all_params = self.policy.parameters()
        all_params.update(updated_params)
        self.policy.load_state_dict(all_params, strict=False)
        
        return loss

    def _update_entropy_coef(self, observations: mx.array) -> mx.array:
        """Update entropy coefficient (alpha) when using automatic tuning."""
        
        def ent_coef_loss_fn(ent_params: Dict[str, mx.array]) -> mx.array:
            """Compute entropy coefficient loss."""
            log_ent_coef = ent_params["log_ent_coef"]
            
            # Forward pass through actor to get log_probs (no explicit no_grad in MLX)
            features = self.policy.extract_features(observations)
            _, log_probs, _ = self.policy.actor_forward(features)
            
            # Entropy coefficient loss: -alpha * (log_prob + target_entropy)
            ent_coef_loss = -mx.mean(log_ent_coef * (log_probs + self.target_entropy))
            
            return ent_coef_loss

        # Parameters for entropy coefficient
        ent_params = {"log_ent_coef": self.log_ent_coef}

        # Compute loss and gradients
        loss, gradients = compute_loss_and_grads(ent_coef_loss_fn, ent_params)
        
        # Update parameters
        updated_params, self.ent_coef_optimizer_state = self.ent_coef_optimizer.update(
            ent_params, gradients, self.ent_coef_optimizer_state
        )
        
        # Update log_ent_coef
        self.log_ent_coef = updated_params["log_ent_coef"]
        
        return loss

    def _update_target_networks(self) -> None:
        """Update target networks with polyak averaging."""
        # Get main critic parameters
        main_critic_params = {}
        for name, param in self.policy.parameters().items():
            if any(f"q_net_{i}" in name for i in range(self.policy.n_critics)) and "target" not in name:
                main_critic_params[name] = param

        # Get target critic parameters
        target_critic_params = {}
        for name, param in self.policy.parameters().items():
            if any(f"q_net_target_{i}" in name for i in range(self.policy.n_critics)):
                target_critic_params[name] = param

        # Update target parameters
        updated_target_params = polyak_update(main_critic_params, target_critic_params, self.tau)
        
        # Update policy with new target parameters
        all_params = self.policy.parameters()
        all_params.update(updated_target_params)
        self.policy.load_state_dict(all_params, strict=False)

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1000,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Run the SAC off-policy training loop.
        This fills the replay buffer with environment interactions and performs
        gradient updates according to train_freq/gradient_steps.
        """
        # Reset counters
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        self._total_timesteps = total_timesteps
        
        # Simple callback shim
        if callback is None:
            from types import SimpleNamespace
            callback = SimpleNamespace()
            callback.on_training_start = lambda *args, **kwargs: None
            callback.on_step = lambda *args, **kwargs: True
            callback.on_training_end = lambda *args, **kwargs: None
        
        # Initial reset
        if not hasattr(self, "_last_obs") or self._last_obs is None:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                obs0, _ = reset_out
            else:
                obs0 = reset_out
            # Ensure batch dim for non-Vec envs
            if not hasattr(self.env, 'num_envs') or self.n_envs == 1:
                self._last_obs = np.expand_dims(obs0, 0)
            else:
                self._last_obs = obs0
        if not hasattr(self, "_last_episode_starts"):
            self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)
        
        callback.on_training_start(locals(), globals())
        
        timesteps_since_last_train = 0
        
        while self.num_timesteps < total_timesteps:
            # Sample action(s)
            actions = self._sample_action(self.learning_starts, self.action_noise, n_envs=self.n_envs)
            
            # Step the environment (handle non-Vec single env API)
            if self.n_envs == 1 and not hasattr(self.env, 'num_envs'):
                step_action = actions if np.asarray(actions).ndim == 1 else actions[0]
                obs_, reward, terminated, truncated, info = self.env.step(step_action)
                done = np.array([terminated or truncated])
                new_obs = np.expand_dims(obs_, 0)
                rewards = np.expand_dims(np.array([reward], dtype=np.float32), 0)
                infos = [info]
            else:
                new_obs, rewards, done, infos = self.env.step(actions)
                done = np.array(done)
            
            # Ensure batch dimensions for replay buffer
            if not isinstance(new_obs, dict) and new_obs.ndim == len(self.observation_space.shape):
                new_obs = np.expand_dims(new_obs, 0)
            if not isinstance(self._last_obs, dict) and self._last_obs is not None and self._last_obs.ndim == len(self.observation_space.shape):
                last_obs_batched = np.expand_dims(self._last_obs, 0)
            else:
                last_obs_batched = self._last_obs
            if np.asarray(rewards).ndim == 1:
                rewards = np.expand_dims(rewards, 0)
            if np.asarray(actions).ndim == len(self.action_space.shape) + 1:
                actions_batched = actions
            else:
                actions_batched = np.expand_dims(actions, 0)
            
            # Store transition in replay buffer
            self.replay_buffer.add(last_obs_batched, new_obs, actions_batched, rewards, done, infos)
            
            self._last_obs = new_obs
            self._last_episode_starts = done
            
            # Increment timesteps
            self.num_timesteps += self.n_envs
            timesteps_since_last_train += self.n_envs
            
            # Update learning progress
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            # Train according to frequency
            if self.num_timesteps >= self.learning_starts and timesteps_since_last_train >= int(self.train_freq):
                self.train(self.gradient_steps, batch_size=self.batch_size)
                timesteps_since_last_train = 0
            
            # Callback step
            if callback is not None and not callback.on_step():
                break
        
        callback.on_training_end()
        return self

    def _excluded_save_params(self) -> List[str]:
        """
        Parameters that should not be saved.
        
        Returns:
            List of parameter names to exclude from saving
        """
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data to save."""
        data = super()._get_save_data()
        
        # Save SAC-specific parameters
        data.update({
            "tau": self.tau,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "target_entropy": self.target_entropy,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
            "use_sde_at_warmup": self.use_sde_at_warmup,
        })
        
        # Save optimizer states
        if hasattr(self, "actor_optimizer_state"):
            data["actor_optimizer_state"] = self.actor_optimizer_state
        if hasattr(self, "critic_optimizer_state"):
            data["critic_optimizer_state"] = self.critic_optimizer_state
        if hasattr(self, "ent_coef_optimizer_state"):
            data["ent_coef_optimizer_state"] = self.ent_coef_optimizer_state
        
        # Save entropy coefficient
        if self.log_ent_coef is not None:
            data["log_ent_coef"] = self.log_ent_coef
        
        return data

    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load data from save."""
        super()._load_save_data(data)
        
        # Load SAC-specific parameters
        self.tau = data.get("tau", self.tau)
        self.gamma = data.get("gamma", self.gamma)
        self.ent_coef = data.get("ent_coef", self.ent_coef)
        self.target_entropy = data.get("target_entropy", self.target_entropy)
        self.use_sde = data.get("use_sde", self.use_sde)
        self.sde_sample_freq = data.get("sde_sample_freq", self.sde_sample_freq)
        self.use_sde_at_warmup = data.get("use_sde_at_warmup", self.use_sde_at_warmup)
        
        # Load optimizer states
        if "actor_optimizer_state" in data:
            self.actor_optimizer_state = data["actor_optimizer_state"]
        if "critic_optimizer_state" in data:
            self.critic_optimizer_state = data["critic_optimizer_state"]
        if "ent_coef_optimizer_state" in data:
            self.ent_coef_optimizer_state = data["ent_coef_optimizer_state"]
        
        # Load entropy coefficient
        if "log_ent_coef" in data:
            self.log_ent_coef = data["log_ent_coef"]

    def _get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self.policy.parameters()

    def _set_parameters(self, params: Dict[str, Any], exact_match: bool = True) -> None:
        """Set algorithm parameters."""
        self.policy.load_state_dict(params, strict=exact_match)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict actions for given observations.
        
        Args:
            observation: Input observations
            state: Not used in SAC
            episode_start: Not used in SAC
            deterministic: Whether to sample deterministically
            
        Returns:
            actions: Predicted actions
            state: Not used in SAC (returns None)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)
