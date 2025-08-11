"""Base algorithm class for all RL algorithms in MLX Baselines3."""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import cloudpickle
import gymnasium as gym
import mlx.core as mx
import numpy as np

from mlx_baselines3.common.type_aliases import (
    ActionType,
    GymEnv,
    MlxArray,
    ObsType,
    PolicyPredict,
    Schedule,
    SerializableData,
)
from mlx_baselines3.common.utils import (
    get_device,
    get_schedule_fn,
    obs_as_mlx,
    set_random_seed,
)
from mlx_baselines3.common.logger import Logger, configure_logger


class BaseAlgorithm(ABC):
    """
    Abstract base class for all RL algorithms in MLX Baselines3.
    
    This class provides the common interface and functionality shared by all
    reinforcement learning algorithms, including training, prediction, and
    model persistence.
    """

    def __init__(
        self,
        policy: Union[str, Type],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        device: str = "auto",
        verbose: int = 0,
        seed: Optional[int] = None,
        supported_action_spaces: Optional[List[Type[gym.Space]]] = None,
        **kwargs,
    ):
        """
        Initialize the base algorithm.

        Args:
            policy: Policy class or string identifier
            env: Environment or environment ID string
            learning_rate: Learning rate value or schedule function
            device: Device to use ("auto", "gpu", or "cpu")
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            seed: Random seed for reproducibility
            supported_action_spaces: List of supported action space types
            **kwargs: Additional arguments
        """
        if isinstance(env, str):
            env = gym.make(env)

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = getattr(env, "num_envs", 1)
        
        # Set device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = device
            
        self.verbose = verbose
        self.policy = policy
        self.learning_rate = learning_rate
        self.lr_schedule = get_schedule_fn(learning_rate)
        
        # Training variables
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._episode_num = 0
        self._current_progress_remaining = 1.0
        
        # Set random seed
        if seed is not None:
            set_random_seed(seed)
        self.seed = seed
        
        # Action space validation
        if supported_action_spaces is not None:
            assert isinstance(self.action_space, tuple(supported_action_spaces)), (
                f"Action space {self.action_space} is not supported. "
                f"Supported action spaces: {supported_action_spaces}"
            )
        
        # Initialize logger (will be set up properly in learn())
        self.logger = None
        
        # Episode info buffer for tracking episode metrics
        self.ep_info_buffer = []
        
        # Algorithm-specific initialization
        self._setup_model()

    @abstractmethod
    def _setup_model(self) -> None:
        """
        Setup model: create policy, networks, buffers, optimizers, etc.
        
        This method must be implemented by each algorithm.
        """
        pass

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "BaseAlgorithm":
        """
        Learn policy using the algorithm for the given number of timesteps.

        Args:
            total_timesteps: Number of samples (env steps) to train on
            callback: Callback(s) called at every step with state of the algorithm
            log_interval: Number of timesteps before logging  
            tb_log_name: Name of the tensorboard log
            reset_num_timesteps: Whether to reset timesteps counter
            progress_bar: Display a progress bar

        Returns:
            Trained algorithm instance
        """
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[MlxArray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> PolicyPredict:
        """
        Get action(s) from observation(s).

        Args:
            observation: Input observation(s)
            state: RNN state (if using recurrent policy)
            episode_start: Whether the episode has started (for recurrent policies)
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (actions, next_states)
        """
        # Convert observation to MLX format
        obs_tensor = obs_as_mlx(observation)
        
        # Add batch dimension if needed
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        if not vectorized_env:
            obs_tensor = mx.expand_dims(obs_tensor, 0) if isinstance(obs_tensor, mx.array) else {
                key: mx.expand_dims(value, 0) for key, value in obs_tensor.items()
            }

        # Get actions from policy
        actions, state = self.policy.predict(
            obs_tensor, state, episode_start, deterministic
        )

        # Convert back to numpy and remove batch dimension if needed
        actions = np.array(actions)
        if not vectorized_env:
            # For single environments, remove batch dimension
            if len(actions.shape) > 0 and actions.shape[0] == 1:
                actions = actions[0]
                
        # For discrete action spaces, return scalar; for continuous, return array
        if isinstance(self.action_space, gym.spaces.Discrete) and not vectorized_env:
            # Convert to scalar for discrete actions
            actions = actions.item() if hasattr(actions, 'item') else actions
        elif not vectorized_env:
            # For continuous actions, ensure it's an array
            actions = np.atleast_1d(actions)

        return actions, state

    def _setup_logger(
        self,
        log_path: Optional[str] = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ) -> None:
        """
        Setup logger for training.
        
        Args:
            log_path: Path to save logs
            tb_log_name: Name for TensorBoard logs
            reset_num_timesteps: Whether to reset timestep counter
        """
        if log_path is not None:
            format_strings = ["stdout", "csv", "tensorboard"]
        else:
            format_strings = ["stdout"]
            
        self.logger = configure_logger(log_path, format_strings)

    def _update_info_buffer(self, infos: List[Dict]) -> None:
        """
        Update episode info buffer with information from environments.
        
        Args:
            infos: List of info dicts from environment steps
        """
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                ep_info = info['episode']
                # Store episode reward and length
                self.ep_info_buffer.append({
                    'r': ep_info.get('r', 0.0),
                    'l': ep_info.get('l', 0.0),
                    't': ep_info.get('t', 0.0),  # time
                })
                # Keep only recent episodes (last 100)
                if len(self.ep_info_buffer) > 100:
                    self.ep_info_buffer = self.ep_info_buffer[-100:]

    def save(self, path: str) -> None:
        """
        Save model to a file.

        Args:
            path: Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Extract env_id if available
        env_id = None
        if hasattr(self.env, "spec") and self.env.spec is not None:
            env_id = self.env.spec.id
        elif hasattr(self.env, "envs") and len(self.env.envs) > 0 and hasattr(self.env.envs[0], "spec"):
            # For vectorized environments, get from first env
            env_id = self.env.envs[0].spec.id if self.env.envs[0].spec is not None else None
        
        # Prepare data to save
        # Important: avoid saving non-serializable callables (e.g., lr_schedule)
        # Persist a concrete learning rate value instead of a raw callable
        if callable(self.learning_rate):
            try:
                lr_value = float(self.lr_schedule(self._current_progress_remaining))
            except Exception:
                # Fallback to a sensible default if schedule evaluation fails
                lr_value = 3e-4
        else:
            lr_value = self.learning_rate
        
        # Try to save a policy alias string instead of the instance to avoid pickling closures
        policy_field: Any
        try:
            # Prefer common SB3-style alias
            policy_field = "MlpPolicy"
        except Exception:
            policy_field = "MlpPolicy"
        
        data = {
            "policy": policy_field,
            "policy_state": self.policy.state_dict() if hasattr(self.policy, "state_dict") else {},
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "num_timesteps": self.num_timesteps,
            "seed": self.seed,
            "learning_rate": lr_value,
            # Save a serializable representation of lr_schedule
            "lr_schedule": lr_value,
            "device": self.device,
            "verbose": self.verbose,
            "_total_timesteps": self._total_timesteps,
            "_episode_num": self._episode_num,
            "_current_progress_remaining": self._current_progress_remaining,
            "env_id": env_id,
        }
        
        # Add optimizer state if available
        if hasattr(self, "optimizer_state") and self.optimizer_state is not None:
            data["optimizer_state"] = self.optimizer_state
        
        # Add algorithm-specific data
        algorithm_data = self._get_save_data()
        data.update(algorithm_data)
        
        # Save using cloudpickle for compatibility
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
            
        if self.verbose >= 1:
            print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        device: str = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load model from a file.

        Args:
            path: Path to the saved model
            env: Environment (if None, will try to use saved environment)
            device: Device to use for loaded model
            custom_objects: Dictionary of custom objects to use during loading
            print_system_info: Whether to print system info
            force_reset: Whether to force environment reset after loading
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        import warnings
        
        # Load data
        with open(path, "rb") as f:
            data = cloudpickle.load(f)
        
        if print_system_info:
            print("System info:")
            print(f"  Device: {get_device()}")
            print(f"  MLX version: {mx.__version__}")
        
        # Create model instance
        if env is None:
            # Try to recreate environment from saved data
            env_id = data.get("env_id")
            if env_id is not None:
                try:
                    from .vec_env import DummyVecEnv
                    base_env = gym.make(env_id)
                    env = DummyVecEnv([lambda: base_env])
                    if print_system_info or kwargs.get("verbose", 0) >= 1:
                        print(f"Recreated environment from saved env_id: {env_id}")
                except Exception as e:
                    warnings.warn(f"Failed to recreate environment '{env_id}': {e}. Using CartPole-v1 as fallback.", UserWarning)
                    from .vec_env import DummyVecEnv
                    base_env = gym.make("CartPole-v1")
                    env = DummyVecEnv([lambda: base_env])
            else:
                warnings.warn("No env_id found in saved model. Using CartPole-v1 as fallback.", UserWarning)
                from .vec_env import DummyVecEnv
                base_env = gym.make("CartPole-v1")
                env = DummyVecEnv([lambda: base_env])
            
        # Extract constructor arguments
        model_kwargs = {}
        for key in ["learning_rate", "verbose", "seed"]:
            if key in data:
                model_kwargs[key] = data[key]
        model_kwargs.update(kwargs)
        
        # Create model instance
        model = cls(
            policy=data["policy"],
            env=env,
            device=device,
            **model_kwargs,
        )
        
        # Restore training state (backward compatible)
        known_state_keys = {
            "num_timesteps", "_total_timesteps", "_episode_num", "_current_progress_remaining"
        }
        for key in known_state_keys:
            if key in data:
                setattr(model, key, data[key])
            elif key == "num_timesteps":
                setattr(model, key, 0)
            elif key == "_total_timesteps":
                setattr(model, key, 0)
            elif key == "_episode_num":
                setattr(model, key, 0)
            elif key == "_current_progress_remaining":
                setattr(model, key, 1.0)
        
        # Restore optimizer state if available
        if "optimizer_state" in data and hasattr(model, "optimizer_state"):
            model.optimizer_state = data["optimizer_state"]
            if print_system_info or kwargs.get("verbose", 0) >= 1:
                print("Restored optimizer state from saved model")
        
        # Load saved policy state if present
        if "policy_state" in data and hasattr(model, "policy") and hasattr(model.policy, "load_state_dict"):
            try:
                model.policy.load_state_dict(data["policy_state"], strict=True)
            except Exception as e:
                warnings.warn(f"Failed to load policy state: {e}", UserWarning)
        
        # Load algorithm-specific data (with backward compatibility)
        try:
            model._load_save_data(data)
        except Exception as e:
            warnings.warn(f"Failed to load some algorithm-specific data: {e}. Model may not be fully restored.", UserWarning)
        
        # Warn about unknown keys for forward compatibility
        expected_keys = {
            "policy", "policy_state", "observation_space", "action_space", "n_envs", "num_timesteps",
            "seed", "learning_rate", "lr_schedule", "device", "verbose",
            "_total_timesteps", "_episode_num", "_current_progress_remaining",
            "env_id", "optimizer_state"
        }
        
        # Get algorithm-specific keys by calling _get_save_data() on a temporary instance
        try:
            temp_data = model._get_save_data()
            expected_keys.update(temp_data.keys())
        except:
            pass  # Ignore if we can't get algorithm-specific keys
        
        unknown_keys = set(data.keys()) - expected_keys
        if unknown_keys:
            warnings.warn(f"Unknown keys in saved model (skipping): {unknown_keys}. "
                         "This may indicate the model was saved with a newer version.", UserWarning)
        
        if force_reset and hasattr(env, "reset"):
            env.reset()
            
        return model

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Any]],
        exact_match: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Load parameters from a file or dictionary.

        Args:
            load_path_or_dict: Path to saved parameters or parameter dictionary
            exact_match: Whether parameter names must match exactly
            device: Device to load parameters to
        """
        if isinstance(load_path_or_dict, str):
            with open(load_path_or_dict, "rb") as f:
                params = cloudpickle.load(f)
        else:
            params = load_path_or_dict
            
        self._set_parameters(params, exact_match=exact_match)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.

        Returns:
            Dictionary containing model parameters
        """
        return self._get_parameters()

    def _get_save_data(self) -> Dict[str, Any]:
        """
        Get algorithm-specific data to save.
        
        Returns:
            Dictionary of data to save
        """
        return {}

    def _load_save_data(self, data: Dict[str, Any]) -> None:
        """
        Load algorithm-specific data.
        
        Args:
            data: Dictionary containing saved data
        """
        pass

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """
        Get algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass

    @abstractmethod
    def _set_parameters(self, params: Dict[str, Any], exact_match: bool = True) -> None:
        """
        Set algorithm parameters.
        
        Args:
            params: Parameter dictionary
            exact_match: Whether parameter names must match exactly
        """
        pass

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and decreases to 0).

        Args:
            num_timesteps: Current number of timesteps
            total_timesteps: Total number of timesteps
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[Any, List[Any]]) -> None:
        """
        Update the optimizers learning rate using the current progress remaining.

        Args:
            optimizers: Optimizer(s) to update
        """
        # Convert to list if single optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
            
        # Get current learning rate from schedule
        current_lr = self.lr_schedule(self._current_progress_remaining)
        
        # Update each optimizer
        for optimizer in optimizers:
            if hasattr(optimizer, 'learning_rate'):
                optimizer.learning_rate = current_lr
            elif hasattr(optimizer, 'lr'):
                optimizer.lr = current_lr

    def _is_vectorized_observation(self, observation: Union[np.ndarray, Dict], observation_space) -> bool:
        """
        Check if observation comes from a vectorized environment.

        Args:
            observation: The observation to check
            observation_space: The observation space

        Returns:
            True if vectorized, False otherwise
        """
        if isinstance(observation, dict):
            if len(observation) == 0:
                return False
            # Check first key
            first_key = next(iter(observation.keys()))
            return observation[first_key].shape[0] != observation_space[first_key].shape[0]
        else:
            return observation.shape[0] != observation_space.shape[0]

    def _log(self, message: str, level: int = 1) -> None:
        """
        Log a message if verbose level is sufficient.

        Args:
            message: Message to log
            level: Required verbosity level
        """
        if self.verbose >= level:
            print(message)


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms (PPO, A2C).
    
    On-policy algorithms learn from data collected using the current policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a rollout buffer.

        Args:
            env: Environment to collect rollouts from
            callback: Callback to call during rollout collection
            rollout_buffer: Buffer to store rollouts
            n_rollout_steps: Number of steps to collect

        Returns:
            True if collection was successful
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Update policy using collected rollouts.
        """
        pass


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy algorithms (SAC, TD3, DQN).
    
    Off-policy algorithms can learn from data collected using any policy.
    """

    def __init__(self, *args, replay_buffer_class=None, replay_buffer_kwargs=None, **kwargs):
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        super().__init__(*args, **kwargs)

    @abstractmethod
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update the policy using gradient steps on batches from the replay buffer.

        Args:
            gradient_steps: Number of gradient steps to perform
            batch_size: Size of each training batch
        """
        pass

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1) -> np.ndarray:
        """
        Sample actions according to the exploration policy.

        Args:
            learning_starts: Number of steps before learning starts
            action_noise: Action noise object
            n_envs: Number of environments

        Returns:
            Actions to take
        """
        # Use random actions before learning starts
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Get action from policy
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
            
        # Add action noise if specified
        if action_noise is not None and not isinstance(self.action_space, gym.spaces.Discrete):
            unscaled_action = np.clip(unscaled_action + action_noise(), -1, 1)
            
        return unscaled_action
