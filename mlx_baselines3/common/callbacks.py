"""Callback helpers used by training loops."""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from mlx_baselines3.common.type_aliases import GymEnv
from mlx_baselines3.common.utils import safe_mean

if TYPE_CHECKING:
    from mlx_baselines3.common.base_class import BaseAlgorithm
    from tqdm import tqdm as TqdmCallable
else:  # pragma: no cover - used only for typing
    BaseAlgorithm = Any
    TqdmCallable = Callable[..., Any]


class BaseCallback(ABC):
    """
    Abstract base class for callbacks.

    Callbacks provide a way to monitor training, save checkpoints,
    evaluate models, and implement early stopping.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the callback.

        Args:
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        self.verbose = verbose
        self.model: Optional[BaseAlgorithm] = None
        self.training_env: Optional[GymEnv] = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}

    def init_callback(self, model: BaseAlgorithm) -> None:
        """
        Initialize the callback.

        Args:
            model: Reference to the model being trained
        """
        self.model = model
        self.training_env = model.env
        self._init_callback()

    @abstractmethod
    def _init_callback(self) -> None:
        """
        Initialize the callback (to be implemented by subclasses).
        """
        pass

    def on_training_start(
        self,
        locals_: Optional[Dict[str, Any]] = None,
        globals_: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called at the beginning of training.

        Args:
            locals_: Local variables from the training function
            globals_: Global variables from the training function
        """
        # Initialize locals and globals for the callback
        if locals_ is not None:
            self.locals = locals_
        else:
            self.locals = {}
        if globals_ is not None:
            self.globals = globals_
        else:
            self.globals = {}
        self._on_training_start()

    @abstractmethod
    def _on_training_start(self) -> None:
        """
        Called at the beginning of training (to be implemented by subclasses).
        """
        pass

    def on_rollout_start(self) -> None:
        """
        Called at the beginning of a rollout.
        """
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        """
        Called at the beginning of a rollout (to be implemented by subclasses).
        """
        pass

    def on_step(self) -> bool:
        """
        Called at every step during training.

        Returns:
            If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        if self.model is not None:
            self.num_timesteps = self.model.num_timesteps
        return self._on_step()

    @abstractmethod
    def _on_step(self) -> bool:
        """
        Called at every step during training (to be implemented by subclasses).

        Returns:
            If the callback returns False, training is aborted early.
        """
        pass

    def on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        self._on_training_end()

    def _on_training_end(self) -> None:
        """
        Called at the end of training (to be implemented by subclasses).
        """
        pass

    def on_rollout_end(self) -> None:
        """
        Called at the end of a rollout.
        """
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout (to be implemented by subclasses).
        """
        pass


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.
    """

    def __init__(self, callbacks: List[BaseCallback]):
        """
        Initialize the callback list.

        Args:
            callbacks: List of callbacks to chain
        """
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        """Initialize all callbacks."""
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        """Call _on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        """Call _on_rollout_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        """
        Call _on_step for all callbacks.

        Returns:
            False if any callback returns False, True otherwise.
        """
        continue_training = True
        for callback in self.callbacks:
            if callback.on_step() is False:
                continue_training = False
        return continue_training

    def _on_rollout_end(self) -> None:
        """Call _on_rollout_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        """Call _on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end()


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints at regular intervals.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        """
        Initialize the checkpoint callback.

        Args:
            save_freq: Save the model every save_freq timesteps
            save_path: Path to the folder where checkpoints will be saved
            name_prefix: Prefix for checkpoint filenames
            save_replay_buffer: Whether to save the replay buffer (for
                off-policy algorithms)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer

    def _init_callback(self) -> None:
        """Create save directory if it doesn't exist."""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """Nothing to do on training start."""
        pass

    def _on_step(self) -> bool:
        """Save checkpoint if save_freq reached."""
        if self.model is None:
            return True
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    This callback evaluates the model on a separate environment
    and saves the best model based on evaluation performance.
    """

    def __init__(
        self,
        eval_env: GymEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize the evaluation callback.

        Args:
            eval_env: Environment for evaluation
            callback_on_new_best: Callback to trigger when new best model is found
            n_eval_episodes: Number of episodes to evaluate
            eval_freq: Evaluate the model every eval_freq timesteps
            log_path: Path to a folder where the evaluations will be saved
            best_model_save_path: Path to save the best model
            deterministic: Whether to use deterministic actions during evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results: List[np.ndarray] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[np.ndarray] = []
        self.callback_on_new_best = callback_on_new_best

    def _init_callback(self):
        """Initialize evaluation callback."""
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_training_start(self) -> None:
        """Nothing to do on training start."""
        pass

    def _on_step(self) -> bool:
        """Evaluate model if eval_freq reached."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if needed
            episode_rewards, episode_lengths = self._evaluate_policy()

            mean_reward = np.mean(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                reward_std = np.std(episode_rewards)
                length_std = np.std(episode_lengths)
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {reward_std:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {length_std:.2f}")

            # Add to current Logger
            if hasattr(self.model, "logger") and self.model.logger is not None:
                self.model.logger.record("eval/mean_reward", float(mean_reward))
                self.model.logger.record("eval/mean_ep_length", mean_ep_length)

            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(np.array(episode_rewards, dtype=float))
            self.evaluations_length.append(np.array(episode_lengths, dtype=float))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    return self.callback_on_new_best.on_step()

        return True

    def _evaluate_policy(self) -> Tuple[List[float], List[int]]:
        """Evaluate the policy for n_eval_episodes episodes."""
        if self.model is None:
            return [], []

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle new gym API
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return episode_rewards, episode_lengths


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop training when a reward threshold is reached.
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        """
        Initialize the stop training callback.

        Args:
            reward_threshold: Minimum reward to stop training
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _init_callback(self) -> None:
        """Nothing to do on init."""
        pass

    def _on_training_start(self) -> None:
        """Nothing to do on training start."""
        pass

    def _on_step(self) -> bool:
        """Check if reward threshold is reached."""
        if (
            self.model is not None
            and hasattr(self.model, "ep_info_buffer")
            and len(self.model.ep_info_buffer) > 0
        ):
            # Check recent episode rewards
            recent_rewards = [
                ep_info["r"] for ep_info in self.model.ep_info_buffer[-100:]
            ]
            if len(recent_rewards) >= 100:
                mean_reward = safe_mean(np.array(recent_rewards))
                if mean_reward >= self.reward_threshold:
                    if self.verbose > 0:
                        print(
                            "Stopping training because the mean reward "
                            f"{mean_reward:.2f} is above the threshold "
                            f"{self.reward_threshold}"
                        )
                    return False
        return True


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar during training.

    This callback shows training progress including timesteps,
    episode rewards, and other metrics.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the progress bar callback.

        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.progress_bar: Optional[TqdmCallable] = None
        self.pbar: Optional[Any] = None
        self._total_timesteps = 0

    def _init_callback(self) -> None:
        """Initialize progress bar."""
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            self.progress_bar = tqdm
        except ImportError:
            if self.verbose > 0:
                print("Warning: tqdm not available, progress bar disabled")
            self.progress_bar = None

    def _on_training_start(self) -> None:
        """Start progress bar."""
        if self.progress_bar is not None:
            self._total_timesteps = int(self.locals.get("total_timesteps", 0))
            self.pbar = self.progress_bar(total=self._total_timesteps)

    def _on_step(self) -> bool:
        """Update progress bar."""
        if self.progress_bar is not None and self.pbar is not None:
            # Update progress
            self.pbar.n = self.num_timesteps

            # Get recent episode info
            if (
                hasattr(self.model, "ep_info_buffer")
                and len(self.model.ep_info_buffer) > 0
            ):
                recent_rewards = [
                    ep_info["r"] for ep_info in self.model.ep_info_buffer[-10:]
                ]
                if len(recent_rewards) > 0:
                    mean_reward = safe_mean(np.array(recent_rewards))
                    self.pbar.set_description(f"Mean reward: {mean_reward:.2f}")

            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        """Close progress bar."""
        if self.progress_bar is not None and self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class NoOpCallback(BaseCallback):
    """Callback that performs no action; used as a placeholder."""

    def __init__(self) -> None:
        super().__init__(verbose=0)

    def _init_callback(self) -> None:  # pragma: no cover - trivial
        pass

    def _on_training_start(self) -> None:  # pragma: no cover - trivial
        pass

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:  # pragma: no cover - trivial
        pass

    def _on_rollout_start(self) -> None:  # pragma: no cover - trivial
        pass

    def _on_rollout_end(self) -> None:  # pragma: no cover - trivial
        pass


def convert_callback(callback: Optional[Any]) -> Any:
    """
    Convert a callback or list of callbacks to BaseCallback format.

    Args:
        callback: Callback(s) to convert

    Returns:
        BaseCallback instance
    """
    if callback is None:
        return NoOpCallback()

    if isinstance(callback, list):
        return CallbackList(callback)

    if not isinstance(callback, BaseCallback):
        # Assume it's a simple callback with on_step method
        # This maintains backward compatibility with existing simple callbacks
        return callback

    return callback
