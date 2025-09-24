"""
Deep Q-Networks (DQN) algorithm implementation using MLX.

Based on the original DQN paper, "Human-level control through deep
reinforcement learning": https://www.nature.com/articles/nature14236
"""

from typing import Any, Dict, List, Optional, Type, Union

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
from mlx_baselines3.common.schedules import linear_schedule
from mlx_baselines3.common.type_aliases import GymEnv, Schedule
from mlx_baselines3.common.utils import (
    polyak_update,
    safe_mean,
)
from mlx_baselines3.dqn.policies import DQNPolicy, MlpPolicy


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Networks (DQN) algorithm.

    Paper: https://www.nature.com/articles/nature14236
    Default hyperparameters are based on the original DQN paper.

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from
        learning_rate: Learning rate for the Q-network optimizer
        buffer_size: Size of the replay buffer
        learning_starts: How many steps to collect before training starts
        batch_size: Batch size for training
        tau: The soft update coefficient for target network
        gamma: Discount factor
        train_freq: Update the model every `train_freq` steps
        gradient_steps: How many gradient steps to do after each rollout
        target_update_interval: Update the target network every
            `target_update_interval` steps
        exploration_fraction: Fraction of training during which the
            exploration rate is annealed
        exploration_initial_eps: Initial value of random action probability
        exploration_final_eps: Final value of random action probability
        max_grad_norm: Maximum norm for gradient clipping
        device: Device (cpu, gpu, auto) on which the code should be run
        verbose: Verbosity level
        seed: Seed for the pseudo random generators
        optimize_memory_usage: Enable memory-optimized replay buffer
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]] = "MlpPolicy",
        env: Union[GymEnv, str] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
        optimize_memory_usage: bool = False,
        **kwargs,
    ):
        # Set hyperparameters before calling super().__init__()
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.optimize_memory_usage = optimize_memory_usage

        # Epsilon schedule for exploration
        self.exploration_schedule = linear_schedule(
            exploration_initial_eps, exploration_final_eps
        )

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            device=device,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=[gym.spaces.Discrete],
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs=dict(optimize_memory_usage=optimize_memory_usage),
        )

    def _setup_model(self) -> None:
        """Create networks and optimizer."""
        # Create policy networks
        policy_kwargs = getattr(self, "policy_kwargs", {})

        if isinstance(self.policy, str):
            if self.policy == "MlpPolicy":
                policy_class = MlpPolicy
            else:
                raise ValueError(f"Unknown policy: {self.policy}")
        else:
            policy_class = self.policy

        # Main Q-network
        self.q_net = policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **policy_kwargs,
        )

        # Target Q-network (copy of main network)
        self.q_net_target = policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **policy_kwargs,
        )

        # Initialize target network with same weights as main network
        # This is done by copying the state dict
        target_state_dict = self.q_net.state_dict()
        self.q_net_target.load_state_dict(target_state_dict)

        # Set up optimizer
        self.optimizer = AdamAdapter(learning_rate=self.learning_rate)
        self.optimizer_state = self.optimizer.init_state(self.q_net.state_dict())

        # Create replay buffer
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            **replay_buffer_kwargs,
        )

        self.policy = self.q_net

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Any] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Optional[Any]]:
        """
        Get the policy action from an observation.

        Args:
            observation: Observation
            state: Policy state (unused for DQN)
            episode_start: Episode start flags (unused for DQN)
            deterministic: Whether to use deterministic action selection

        Returns:
            Action and new state (None for DQN)
        """
        if not deterministic and np.random.rand() < self._get_exploration_rate():
            # Random action
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.action_space.sample()
            else:
                action = np.array(
                    [self.action_space.sample() for _ in range(self.n_envs)]
                )
        else:
            # Greedy action from Q-network
            action, _ = self.q_net.predict(observation, deterministic=True)

        return action, None

    def _get_exploration_rate(self) -> float:
        """Get current exploration rate (epsilon)."""
        # Calculate progress through training
        if self._total_timesteps == 0:
            progress = 0.0
        else:
            exploration_steps = int(self.exploration_fraction * self._total_timesteps)
            if exploration_steps <= 0:
                progress = 1.0
            elif self.num_timesteps >= exploration_steps:
                progress = 1.0
            else:
                progress = self.num_timesteps / exploration_steps

        progress_remaining = max(0.0, 1.0 - progress)
        return float(self.exploration_schedule(progress_remaining))

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Run the DQN off-policy training loop.
        This fills the replay buffer with environment interactions and performs
        gradient updates according to train_freq/gradient_steps.

        Args:
            total_timesteps: Total number of samples to train on
            callback: Callback for monitoring training
            log_interval: Log every n episodes
            tb_log_name: Name of the run for TensorBoard logging
            reset_num_timesteps: Whether to reset timesteps
            progress_bar: Display progress bar

        Returns:
            Trained model
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
            # Sample action(s) with epsilon-greedy exploration
            actions = self._sample_action(self.learning_starts, n_envs=self.n_envs)

            # Step the environment (handle non-Vec single env API)
            if self.n_envs == 1 and not hasattr(self.env, "num_envs"):
                step_action = actions if np.asarray(actions).ndim == 0 else actions[0]
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
            if np.asarray(actions).ndim == 0:
                actions_batched = np.expand_dims(actions, 0)
            else:
                actions_batched = actions

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

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update the DQN policy using gradient steps on batches from the replay buffer.

        Args:
            gradient_steps: Number of gradient steps to perform
            batch_size: Size of each training batch
        """
        # Track losses for logging
        q_losses = []

        # Get current parameters
        params = self.q_net.state_dict()

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self.env)

            # Pre-compute target Q-values (don't need gradients for these)
            next_q_values = self.q_net_target.predict_values(
                replay_data["next_observations"]
            )
            next_q_values = mx.max(next_q_values, axis=1)

            # Compute target Q-values using Bellman equation
            not_terminal = 1 - replay_data["terminated"].astype(mx.float32)
            target_q_values = (
                replay_data["rewards"] + not_terminal * self.gamma * next_q_values
            )

            # Define pure loss function for gradient computation
            def loss_fn(p):
                # Load params into policy temporarily for computation
                self.q_net.load_state_dict(p, strict=False)

                # Get Q-values for current observations
                q_values = self.q_net.predict_values(replay_data["observations"])

                # Select Q-values for actions taken
                current_q_values = mx.take_along_axis(
                    q_values,
                    mx.expand_dims(replay_data["actions"].astype(mx.int32), axis=-1),
                    axis=-1,
                ).squeeze(-1)

                # Huber loss (smooth L1 loss)
                def huber_loss(pred, target, delta=1.0):
                    residual = mx.abs(pred - target)
                    condition = residual <= delta
                    squared_loss = 0.5 * residual * residual / delta
                    linear_loss = residual - 0.5 * delta
                    return mx.where(condition, squared_loss, linear_loss)

                # Compute and return loss
                loss = mx.mean(huber_loss(current_q_values, target_q_values))
                return loss

            # Compute loss and gradients using centralized helper
            loss_value, grads = compute_loss_and_grads(loss_fn, params)

            # Clip gradients
            if self.max_grad_norm is not None:
                grads, grad_norm = clip_grad_norm(grads, self.max_grad_norm)

            # Update parameters using optimizer adapter
            params, self.optimizer_state = self.optimizer.update(
                params, grads, self.optimizer_state
            )

            # Ensure policy reflects latest params
            self.q_net.load_state_dict(params, strict=False)
            mx.eval(list(params.values()))

            # Track loss
            q_losses.append(float(loss_value))

            # Update target network
            if self.num_timesteps % self.target_update_interval == 0:
                polyak_update(
                    self.q_net.state_dict(), self.q_net_target.state_dict(), self.tau
                )

        # Log training info
        if len(q_losses) > 0 and hasattr(self, "logger") and self.logger is not None:
            self.logger.record("train/q_loss", safe_mean(q_losses))
            self.logger.record("train/exploration_rate", self._get_exploration_rate())

    def _sample_action(
        self, learning_starts: int, action_noise=None, n_envs: int = 1
    ) -> np.ndarray:
        """
        Sample actions according to the current exploration policy.

        Args:
            learning_starts: Number of steps before learning starts
            action_noise: Action noise object (not used for DQN)
            n_envs: Number of environments

        Returns:
            Actions to take
        """
        # Use epsilon-greedy exploration
        if (
            self.num_timesteps < learning_starts
            or np.random.rand() < self._get_exploration_rate()
        ):
            # Random action
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = np.array([self.action_space.sample() for _ in range(n_envs)])
            else:
                actions = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Greedy action from Q-network
            actions, _ = self.predict(self._last_obs, deterministic=True)

        return actions

    def _get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = {}
        if self.q_net is not None:
            params["q_net_parameters"] = dict(self.q_net.named_parameters())
        if self.q_net_target is not None:
            params["q_net_target_parameters"] = dict(
                self.q_net_target.named_parameters()
            )
        return params

    def _set_parameters(self, params: Dict[str, Any], exact_match: bool = True) -> None:
        """Set algorithm parameters."""
        if "q_net_parameters" in params and self.q_net is not None:
            # Load main Q-network parameters
            self.q_net.load_state_dict(params["q_net_parameters"], strict=exact_match)
        if "q_net_target_parameters" in params and self.q_net_target is not None:
            # Load target Q-network parameters
            self.q_net_target.load_state_dict(
                params["q_net_target_parameters"], strict=exact_match
            )

    def _excluded_save_params(self) -> List[str]:
        """
        Returns list of parameters that should not be saved.
        """
        return super()._excluded_save_params() + ["q_net_target"]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get constructor parameters for saving."""
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                target_update_interval=self.target_update_interval,
                exploration_fraction=self.exploration_fraction,
                exploration_initial_eps=self.exploration_initial_eps,
                exploration_final_eps=self.exploration_final_eps,
                max_grad_norm=self.max_grad_norm,
                optimize_memory_usage=self.optimize_memory_usage,
            )
        )
        return data

    def _get_save_data(self) -> Dict[str, Any]:
        """Get data to save."""
        data = super()._get_save_data()

        # Save DQN-specific parameters
        data.update(
            {
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps,
                "target_update_interval": self.target_update_interval,
                "exploration_fraction": self.exploration_fraction,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_final_eps": self.exploration_final_eps,
                "max_grad_norm": self.max_grad_norm,
                "optimize_memory_usage": self.optimize_memory_usage,
            }
        )

        # Save exploration schedule progress (current epsilon value)
        data["current_exploration_rate"] = self._get_exploration_rate()

        # Save optimizer state if it exists
        if hasattr(self, "optimizer_state"):
            data["q_net_optimizer_state"] = self.optimizer_state

        return data

    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load data from save."""
        super()._load_save_data(data)

        # Load DQN-specific parameters
        self.buffer_size = data.get("buffer_size", self.buffer_size)
        self.learning_starts = data.get("learning_starts", self.learning_starts)
        self.batch_size = data.get("batch_size", self.batch_size)
        self.tau = data.get("tau", self.tau)
        self.gamma = data.get("gamma", self.gamma)
        self.train_freq = data.get("train_freq", self.train_freq)
        self.gradient_steps = data.get("gradient_steps", self.gradient_steps)
        self.target_update_interval = data.get(
            "target_update_interval", self.target_update_interval
        )
        self.exploration_fraction = data.get(
            "exploration_fraction", self.exploration_fraction
        )
        self.exploration_initial_eps = data.get(
            "exploration_initial_eps", self.exploration_initial_eps
        )
        self.exploration_final_eps = data.get(
            "exploration_final_eps", self.exploration_final_eps
        )
        self.max_grad_norm = data.get("max_grad_norm", self.max_grad_norm)
        self.optimize_memory_usage = data.get(
            "optimize_memory_usage", self.optimize_memory_usage
        )

        # Load optimizer state if it exists
        if "q_net_optimizer_state" in data:
            self.optimizer_state = data["q_net_optimizer_state"]
