"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm implementation
using MLX.

Based on the original TD3 paper, "Addressing Function Approximation Error in
Actor-Critic Methods": https://arxiv.org/abs/1802.09477
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np

from mlx_baselines3.common.base_class import OffPolicyAlgorithm
from mlx_baselines3.common.buffers import ReplayBuffer
from mlx_baselines3.common.optimizers import (
    AdamAdapter,
    compute_loss_and_grads,
    clip_grad_norm,
)
from mlx_baselines3.common.type_aliases import GymEnv, Schedule
from mlx_baselines3.common.utils import (
    obs_as_mlx,
    polyak_update,
    safe_mean,
)
from mlx_baselines3.td3.policies import TD3Policy


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    Paper: https://arxiv.org/abs/1802.09477

    TD3 is an off-policy actor-critic deep RL algorithm that extends DDPG with:
    - Twin critics to reduce overestimation bias
    - Delayed policy updates (update actor less frequently than critics)
    - Target policy smoothing (add noise to target actions)

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
        policy_delay: Policy will only be updated once every policy_delay steps
        target_policy_noise: Standard deviation of Gaussian noise added to target policy
        target_noise_clip: Range to clip target policy noise
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
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        # Set TD3-specific parameters before calling super().__init__()
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.optimize_memory_usage = optimize_memory_usage
        self.policy_kwargs = policy_kwargs or {}
        self.stats_window_size = stats_window_size
        self.tensorboard_log = tensorboard_log
        self.replay_buffer = None

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

        # TD3 is only for continuous action spaces
        if not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("TD3 only supports continuous action spaces (Box)")

        # Note: _setup_model() is already called by BaseAlgorithm.__init__()

    def _setup_model(self) -> None:
        """Set up the model and optimizers."""
        # Store original policy spec before it gets modified
        original_policy = self.policy

        # Create policy networks
        if isinstance(original_policy, str):
            if original_policy == "MlpPolicy":
                from mlx_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy

                policy_class = TD3MlpPolicy
            else:
                raise ValueError(f"Unknown policy: {original_policy}")
        else:
            policy_class = original_policy

        # Create policy with TD3-specific network architecture
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
            replay_buffer_kwargs = (
                {}
                if self.replay_buffer_kwargs is None
                else self.replay_buffer_kwargs.copy()
            )
            replay_buffer_kwargs.update(
                {
                    "optimize_memory_usage": self.optimize_memory_usage,
                }
            )

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

        # Initialize optimizer states
        self._initialize_optimizers()

        # Copy policy networks to target networks
        self.policy._build_target_networks()

    def _initialize_optimizers(self) -> None:
        """Initialize optimizer states."""
        # Get actor parameters (both actor_net and actor_output, but not targets)
        actor_params = {}
        for name, param in self.policy.parameters().items():
            if ("actor_net" in name or "actor_output" in name) and "target" not in name:
                actor_params[name] = param

        # Get critic parameters (q_net_0 and q_net_1, but not targets)
        critic_params = {}
        for name, param in self.policy.parameters().items():
            if (
                any(f"q_net_{i}" in name for i in range(self.policy.n_critics))
                and "target" not in name
            ):
                critic_params[name] = param

        # Initialize optimizer states
        self.actor_optimizer_state = self.actor_optimizer.init_state(actor_params)
        self.critic_optimizer_state = self.critic_optimizer.init_state(critic_params)

    def _create_aliases(self) -> None:
        """Create aliases for easy access to policy components."""
        self.actor = self.policy
        self.critic = self.policy
        self.critic_target = self.policy
        self.actor_target = self.policy

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

        actor_losses = []
        critic_losses = []

        for gradient_step in range(gradient_steps):
            # Sample a batch from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Convert to MLX arrays
            observations = obs_as_mlx(replay_data["observations"])
            actions = mx.array(replay_data["actions"])
            next_observations = obs_as_mlx(replay_data["next_observations"])
            rewards = mx.array(replay_data["rewards"]).flatten()
            terminated = mx.array(replay_data["terminated"]).flatten()

            # Update critics
            critic_loss = self._update_critics(
                observations, actions, next_observations, rewards, terminated
            )
            critic_losses.append(float(critic_loss))

            # Delayed policy updates: only update actor every policy_delay steps
            if (self._n_updates + gradient_step) % self.policy_delay == 0:
                # Update actor
                actor_loss = self._update_actor(observations)
                actor_losses.append(float(actor_loss))

                # Update target networks
                self._update_target_networks()

        # Increment number of updates
        self._n_updates += gradient_steps

        # Store training stats (optional logger)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record(
                "train/n_updates", self._n_updates, exclude="tensorboard"
            )
            if len(actor_losses) > 0:
                self.logger.record("train/actor_loss", safe_mean(actor_losses))
            self.logger.record("train/critic_loss", safe_mean(critic_losses))

    def _update_critics(
        self,
        observations: mx.array,
        actions: mx.array,
        next_observations: mx.array,
        rewards: mx.array,
        terminated: mx.array,
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

            # Compute target Q-values using target networks
            next_features = self.policy.extract_features(next_observations)
            next_actions = self.policy.actor_target_forward(next_features)

            # Add target policy smoothing noise
            noise = (
                mx.random.normal(shape=next_actions.shape) * self.target_policy_noise
            )
            noise = mx.clip(noise, -self.target_noise_clip, self.target_noise_clip)

            # Apply noise and ensure actions stay in bounds [-1, 1] (before
            # action space scaling)
            # Since actor_target_forward handles action scaling, work in the
            # normalized space
            if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
                low = mx.array(self.action_space.low)
                high = mx.array(self.action_space.high)
                # Convert to normalized space [-1, 1]
                next_actions_normalized = 2 * (next_actions - low) / (high - low) - 1
                # Add noise in normalized space
                next_actions_normalized = next_actions_normalized + noise
                # Clip in normalized space
                next_actions_normalized = mx.clip(next_actions_normalized, -1.0, 1.0)
                # Convert back to action space
                noisy_next_actions = low + (next_actions_normalized + 1.0) * 0.5 * (
                    high - low
                )
            else:
                noisy_next_actions = mx.clip(next_actions + noise, -1.0, 1.0)

            target_q_values = self.policy.critic_target_forward(
                next_features, noisy_next_actions
            )

            # Take minimum of target Q-values (clipped double Q-learning)
            target_q = mx.minimum(target_q_values[0], target_q_values[1])

            # Compute target values with Bellman backup
            not_terminal = 1 - terminated.reshape(-1, 1).astype(mx.float32)
            target_q = rewards.reshape(-1, 1) + not_terminal * self.gamma * target_q

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
            if (
                any(f"q_net_{i}" in name for i in range(self.policy.n_critics))
                and "target" not in name
            ):
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

    def _update_actor(self, observations: mx.array) -> mx.array:
        """Update actor network."""

        def actor_loss_fn(actor_params: Dict[str, mx.array]) -> mx.array:
            """Compute actor loss."""
            # Update policy parameters for forward pass
            old_params = self.policy.parameters()
            temp_params = {**old_params, **actor_params}
            self.policy.load_state_dict(temp_params, strict=False)

            # Forward pass through actor
            features = self.policy.extract_features(observations)
            actions = self.policy.actor_forward(features)

            # Compute Q-values for actor's actions using first critic only
            q_values = self.policy.critic_forward(features, actions)

            # Actor loss: maximize Q(s, a) -> minimize -Q(s, a)
            actor_loss = -mx.mean(q_values[0])

            # Restore original parameters
            self.policy.load_state_dict(old_params, strict=False)

            return actor_loss

        # Get actor parameters (both actor_net and actor_output, but not targets)
        actor_params = {}
        for name, param in self.policy.parameters().items():
            if ("actor_net" in name or "actor_output" in name) and "target" not in name:
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

    def _update_target_networks(self) -> None:
        """Update target networks with polyak averaging."""
        # Update target actor - create mapping from main to target parameter names
        main_actor_params = {}
        target_actor_params = {}

        for name, param in self.policy.parameters().items():
            if ("actor_net" in name or "actor_output" in name) and "target" not in name:
                main_actor_params[name] = param
                # Create target parameter name
                if "actor_net" in name:
                    target_name = name.replace("actor_net", "actor_target_net")
                elif "actor_output" in name:
                    target_name = name.replace("actor_output", "actor_target_output")
                else:
                    target_name = name

                # Find the corresponding target parameter
                all_params = self.policy.parameters()
                if target_name in all_params:
                    target_actor_params[name] = all_params[
                        target_name
                    ]  # Use main param name as key

        updated_target_actor_params = polyak_update(
            main_actor_params, target_actor_params, self.tau
        )

        # Update target critics - create mapping from main to target parameter names
        main_critic_params = {}
        target_critic_params = {}

        for name, param in self.policy.parameters().items():
            if (
                any(f"q_net_{i}" in name for i in range(self.policy.n_critics))
                and "target" not in name
            ):
                main_critic_params[name] = param
                # Create target parameter name
                target_name = name.replace("q_net_", "q_net_target_")

                # Find the corresponding target parameter
                all_params = self.policy.parameters()
                if target_name in all_params:
                    target_critic_params[name] = all_params[
                        target_name
                    ]  # Use main param name as key

        updated_target_critic_params = polyak_update(
            main_critic_params, target_critic_params, self.tau
        )

        # Apply updated parameters back to the policy
        # We need to map the parameter names back to their target names
        all_params = self.policy.parameters()

        # Update target actor parameters
        for main_name, updated_param in updated_target_actor_params.items():
            if "actor_net" in main_name:
                target_name = main_name.replace("actor_net", "actor_target_net")
            elif "actor_output" in main_name:
                target_name = main_name.replace("actor_output", "actor_target_output")
            else:
                target_name = main_name
            all_params[target_name] = updated_param

        # Update target critic parameters
        for main_name, updated_param in updated_target_critic_params.items():
            target_name = main_name.replace("q_net_", "q_net_target_")
            all_params[target_name] = updated_param

        self.policy.load_state_dict(all_params, strict=False)

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1000,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Run the TD3 off-policy training loop.
        This fills the replay buffer with environment interactions and performs
        gradient updates according to train_freq/gradient_steps.
        """
        # Reset counters
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        self._total_timesteps = total_timesteps

        # Convert callback to proper format
        from mlx_baselines3.common.callbacks import convert_callback

        callback = convert_callback(callback)

        # Initial reset
        if not hasattr(self, "_last_obs") or self._last_obs is None:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                obs0, _ = reset_out
            else:
                obs0 = reset_out
            # Ensure batch dim for non-Vec envs
            if not hasattr(self.env, "num_envs") or self.n_envs == 1:
                self._last_obs = np.expand_dims(obs0, 0)
            else:
                self._last_obs = obs0
        if not hasattr(self, "_last_episode_starts"):
            self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

        callback.on_training_start(locals(), globals())

        timesteps_since_last_train = 0

        while self.num_timesteps < total_timesteps:
            # Sample action(s)
            actions = self._sample_action(
                self.learning_starts, self.action_noise, n_envs=self.n_envs
            )

            # Step the environment (handle non-Vec single env API)
            if self.n_envs == 1 and not hasattr(self.env, "num_envs"):
                step_action = actions if np.asarray(actions).ndim == 1 else actions[0]
                obs_, reward, terminated, truncated, info = self.env.step(step_action)
                terminated_array = np.array([terminated], dtype=np.bool_)
                truncated_array = np.array([truncated], dtype=np.bool_)
                done = np.array([terminated or truncated], dtype=np.bool_)
                new_obs = np.expand_dims(obs_, 0)
                rewards = np.expand_dims(np.array([reward], dtype=np.float32), 0)
                infos = [info]
            else:
                step_output = self.env.step(actions)
                if len(step_output) == 5:
                    (
                        new_obs,
                        rewards,
                        terminated_vals,
                        truncated_vals,
                        infos,
                    ) = step_output
                    terminated_array = np.array(terminated_vals, dtype=np.bool_)
                    truncated_array = np.array(truncated_vals, dtype=np.bool_)
                    done = np.array(terminated_array | truncated_array, dtype=np.bool_)
                else:
                    new_obs, rewards, done_vals, infos = step_output
                    done = np.array(done_vals, dtype=np.bool_)
                    truncated_array = np.array(
                        [info.get("TimeLimit.truncated", False) for info in infos],
                        dtype=np.bool_,
                    )
                    terminated_array = np.array(done & ~truncated_array, dtype=np.bool_)

            # Ensure batch dimensions for replay buffer
            if not isinstance(new_obs, dict) and new_obs.ndim == len(
                self.observation_space.shape
            ):
                new_obs = np.expand_dims(new_obs, 0)
            if (
                not isinstance(self._last_obs, dict)
                and self._last_obs is not None
                and self._last_obs.ndim == len(self.observation_space.shape)
            ):
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
            self.replay_buffer.add(
                last_obs_batched,
                new_obs,
                actions_batched,
                rewards,
                terminated_array,
                truncated_array,
                infos,
            )

            self._last_obs = new_obs
            self._last_episode_starts = done

            # Increment timesteps
            self.num_timesteps += self.n_envs
            timesteps_since_last_train += self.n_envs

            # Update learning progress
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Train according to frequency
            if (
                self.num_timesteps >= self.learning_starts
                and timesteps_since_last_train >= int(self.train_freq)
            ):
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
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
            "actor_target",
        ]

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data to save."""
        data = super()._get_save_data()

        # Save TD3-specific parameters
        data.update(
            {
                "tau": self.tau,
                "gamma": self.gamma,
                "policy_delay": self.policy_delay,
                "target_policy_noise": self.target_policy_noise,
                "target_noise_clip": self.target_noise_clip,
            }
        )

        # Save optimizer states
        if hasattr(self, "actor_optimizer_state"):
            data["actor_optimizer_state"] = self.actor_optimizer_state
        if hasattr(self, "critic_optimizer_state"):
            data["critic_optimizer_state"] = self.critic_optimizer_state

        return data

    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load data from save."""
        super()._load_save_data(data)

        # Load TD3-specific parameters
        self.tau = data.get("tau", self.tau)
        self.gamma = data.get("gamma", self.gamma)
        self.policy_delay = data.get("policy_delay", self.policy_delay)
        self.target_policy_noise = data.get(
            "target_policy_noise", self.target_policy_noise
        )
        self.target_noise_clip = data.get("target_noise_clip", self.target_noise_clip)

        # Load optimizer states
        if "actor_optimizer_state" in data:
            self.actor_optimizer_state = data["actor_optimizer_state"]
        if "critic_optimizer_state" in data:
            self.critic_optimizer_state = data["critic_optimizer_state"]

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
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict actions for given observations.

        Args:
            observation: Input observations
            state: Not used in TD3
            episode_start: Not used in TD3
            deterministic: Always True for TD3 (deterministic policy)

        Returns:
            actions: Predicted actions
            state: Not used in TD3 (returns None)
        """
        return self.policy.predict(
            observation, state, episode_start, deterministic=True
        )
