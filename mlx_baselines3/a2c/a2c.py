"""
Advantage Actor Critic (A2C) algorithm implementation using MLX.

A2C is an on-policy algorithm that combines value-based and policy-based
methods. It uses an actor network to learn the policy and a critic network to
learn the value function. Unlike PPO, A2C uses an unclipped policy gradient
loss with a single epoch of training per update.
"""

import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np

from mlx_baselines3.common.base_class import OnPolicyAlgorithm
from mlx_baselines3.common.buffers import RolloutBuffer
from mlx_baselines3.common.policies import ActorCriticPolicy
from mlx_baselines3.common.type_aliases import GymEnv, MlxArray, Schedule
from mlx_baselines3.common.utils import explained_variance, obs_as_mlx
from mlx_baselines3.common.vec_env import VecEnv
from mlx_baselines3.common.optimizers import (
    create_optimizer_adapter,
    clip_grad_norm,
    compute_loss_and_grads,
)
from mlx_baselines3.common.schedules import get_schedule_fn


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C) algorithm using MLX.

    Paper: https://arxiv.org/abs/1602.01783

    Args:
        policy: The policy model to use
        env: The environment to learn from
        learning_rate: The learning rate
        n_steps: Number of steps to run for each environment per update
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: Maximum value for gradient clipping
        normalize_advantage: Whether to normalize advantages
        use_rms_prop: Whether to use RMSProp optimizer (True) or Adam (False)
        rms_prop_eps: RMSProp epsilon parameter
        device: Device to use for computation
        verbose: Verbosity level
        seed: Random generator seed
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = False,
        use_rms_prop: bool = True,
        rms_prop_eps: float = 1e-5,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # A2C hyperparameters (set before super init)
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.use_rms_prop = use_rms_prop
        self.rms_prop_eps = rms_prop_eps

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
                # TODO: Add support for MultiDiscrete and MultiBinary action spaces
                # gym.spaces.MultiDiscrete,
                # gym.spaces.MultiBinary,
            ],
            **kwargs,
        )

        # Initialize training counters
        self._n_updates = 0

    def _setup_model(self) -> None:
        """Setup model: create policy, networks, buffers, optimizers, etc."""
        if self.verbose >= 1:
            print("Setting up A2C model...")

        # Create policy
        self._make_policy()

        # Initialize rollout buffer
        assert isinstance(self.env, VecEnv), "A2C requires a vectorized environment"

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
                normalize_advantage=self.normalize_advantage,
            )
            if self.verbose >= 1:
                print(f"Rollout buffer created: {self.rollout_buffer}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"Error creating rollout buffer: {e}")
            raise

        if self.verbose >= 1:
            print("A2C model setup complete")

    def _setup_optimizer(self) -> None:
        """Setup the optimizer adapter with proper schedule support."""
        if self.verbose >= 1:
            print("Setting up optimizer adapter...")

        # Create learning rate schedule function
        lr_schedule = get_schedule_fn(self.learning_rate)

        # Create optimizer adapter (RMSProp by default for A2C, Adam as alternative)
        if self.use_rms_prop:
            # RMSProp is the traditional choice for A2C
            self.optimizer_adapter = create_optimizer_adapter(
                optimizer_name="rmsprop",
                learning_rate=lr_schedule,
                alpha=0.99,
                eps=self.rms_prop_eps,
                weight_decay=0.0,
            )
        else:
            # Adam alternative
            self.optimizer_adapter = create_optimizer_adapter(
                optimizer_name="adam",
                learning_rate=lr_schedule,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )

        # Get initial parameters from policy and initialize optimizer state
        initial_params = self.policy.state_dict()
        self.optimizer_state = self.optimizer_adapter.init_state(initial_params)

        if self.verbose >= 1:
            optimizer_name = "RMSProp" if self.use_rms_prop else "Adam"
            print(f"Optimizer adapter initialized: {optimizer_name}")

    def _make_policy(self) -> None:
        """Create policy instance."""
        from mlx_baselines3.a2c.policies import get_a2c_policy_class

        if isinstance(self.policy, str):
            policy_class = get_a2c_policy_class(self.policy)
            self.policy = policy_class(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
            )
        elif isinstance(self.policy, ActorCriticPolicy):
            # Already a constructed policy instance (e.g., when loading); use as-is
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
                clipped_actions = np.clip(
                    actions_np, self.action_space.low, self.action_space.high
                )
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

        rollout_buffer.compute_returns_and_advantage(
            last_values=np.array(values), dones=dones
        )

        return True

    def train(self) -> None:
        """
        Update policy using A2C algorithm (single epoch).
        """
        # Switch to training mode
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses = []
        pg_losses = []
        value_losses = []

        # Initialize a flat parameter dict for functional updates
        params = self.policy.state_dict()

        # A2C trains for only 1 epoch (unlike PPO)
        # Do a complete pass on the rollout buffer
        for rollout_data in self.rollout_buffer.get(
            self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        ):
            actions = rollout_data["actions"]

            # Define loss as a pure function of params
            def loss_fn(p):
                # Load params into policy for forward computations
                self.policy.load_state_dict(p, strict=False)
                return self._compute_loss(rollout_data, self.policy)

            # Compute loss and gradients using centralized helper
            loss_val, grads = compute_loss_and_grads(loss_fn, params)

            # Clip gradients using improved clipping function
            if self.max_grad_norm is not None:
                grads, _ = clip_grad_norm(grads, self.max_grad_norm)

            # Update parameters using optimizer adapter
            if self.optimizer_adapter is not None and self.optimizer_state is not None:
                try:
                    params, self.optimizer_state = self.optimizer_adapter.update(
                        params, grads, self.optimizer_state
                    )
                except Exception as e:
                    warnings.warn(
                        f"Optimizer update failed: {e}. Falling back to SGD.",
                        UserWarning,
                    )
                    # Fallback to simple SGD
                    lr = 7e-4  # Default learning rate
                    params = {
                        k: params[k] - lr * grads.get(k, 0) for k in params.keys()
                    }
            else:
                # Fallback to simple SGD if optimizer not initialized
                lr = 7e-4  # Default learning rate
                params = {k: params[k] - lr * grads.get(k, 0) for k in params.keys()}

            # Ensure policy reflects latest params
            self.policy.load_state_dict(params, strict=False)
            mx.eval(list(params.values()))

            # For logging, recompute key terms with current policy
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data["observations"], actions
            )
            values = mx.flatten(values)

            # Get advantages (already computed in buffer)
            advantages = rollout_data["advantages"]

            # Unclipped policy gradient loss (key difference from PPO)
            policy_loss = -mx.mean(advantages * log_prob)

            # Value loss
            value_loss = mx.mean((rollout_data["returns"] - values) ** 2)

            # Entropy loss
            entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0

            # Store losses for logging
            pg_losses.append(float(policy_loss))
            value_losses.append(float(value_loss))
            entropy_losses.append(float(entropy_loss))

        self._n_updates += 1
        explained_var = explained_variance(
            mx.array(self.rollout_buffer.values.flatten()),
            mx.array(self.rollout_buffer.returns.flatten()),
        )

        # Log training metrics
        if self.verbose >= 1:
            print(f"Explained variance: {explained_var:.2f}")
            print(f"Policy loss: {np.mean(pg_losses):.3f}")
            print(f"Value loss: {np.mean(value_losses):.3f}")
            print(f"Entropy loss: {np.mean(entropy_losses):.3f}")

    def _compute_loss(self, rollout_data: Dict[str, MlxArray], model) -> MlxArray:
        """Compute the total loss for A2C."""
        actions = rollout_data["actions"]

        # Get current policy predictions
        values, log_prob, entropy = model.evaluate_actions(
            rollout_data["observations"], actions
        )
        values = mx.flatten(values)

        # Get advantages (already computed in buffer, optionally normalized)
        advantages = rollout_data["advantages"]

        # Unclipped policy gradient loss (key difference from PPO)
        policy_loss = -mx.mean(advantages * log_prob)

        # Value loss
        value_loss = mx.mean((rollout_data["returns"] - values) ** 2)

        # Entropy loss
        entropy_loss = -mx.mean(entropy) if entropy is not None else 0.0

        # Total loss
        return policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
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
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time() - self.start_time), 1e-8)
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed
                )
                if self.verbose >= 1:
                    print("------------------------------------")
                    print("| rollout/              |         |")
                    if hasattr(self, "ep_info_buffer") and len(self.ep_info_buffer) > 0:
                        ep_len_mean = float(
                            np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])
                        )
                        ep_rew_mean = float(
                            np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                        )
                    else:
                        ep_len_mean = 0.0
                        ep_rew_mean = 0.0
                    print(f"|    ep_len_mean        | {ep_len_mean:.1f}     |")
                    print(f"|    ep_rew_mean        | {ep_rew_mean:.1f}     |")
                    print("| time/                 |         |")
                    print(f"|    fps                | {fps}       |")
                    print(f"|    iterations         | {iteration}       |")
                    print(f"|    time_elapsed       | {int(time_elapsed)}       |")
                    print(f"|    total_timesteps    | {self.num_timesteps}       |")
                    print("------------------------------------")

            self.train()

        callback.on_training_end()

        return self

    def _setup_learn(
        self,
        total_timesteps: int,
        callback,
        reset_num_timesteps: bool,
        tb_log_name: str,
        progress_bar: bool,
    ):
        """Setup learning process."""
        import time

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0

        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        self.start_time = time.time()

        # Initialize callback system
        from mlx_baselines3.common.callbacks import convert_callback

        callback = convert_callback(callback)
        if hasattr(callback, "init_callback"):
            callback.init_callback(self)

        # Reset environment
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        return total_timesteps, callback

    def _update_info_buffer(self, infos):
        """Update the info buffer with episode information."""
        if not hasattr(self, "ep_info_buffer"):
            self.ep_info_buffer = []

        for info in infos:
            if isinstance(info, dict):
                if "episode" in info:
                    ep_info = info["episode"]
                    if "r" in ep_info and "l" in ep_info:
                        self.ep_info_buffer.append(
                            {"r": ep_info["r"], "l": ep_info["l"]}
                        )
                        # Keep only last 100 episodes
                        if len(self.ep_info_buffer) > 100:
                            self.ep_info_buffer.pop(0)

    def _update_learning_rate(self, optimizer):
        """Update learning rate in the optimizer."""
        if hasattr(optimizer, "learning_rate"):
            new_lr = self.lr_schedule(self._current_progress_remaining)
            optimizer.learning_rate = new_lr

    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving."""
        return {
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "use_rms_prop": self.use_rms_prop,
            "rms_prop_eps": self.rms_prop_eps,
        }

    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """Load algorithm-specific data."""
        for key in [
            "n_steps",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "normalize_advantage",
            "use_rms_prop",
            "rms_prop_eps",
        ]:
            if key in data:
                setattr(self, key, data[key])
