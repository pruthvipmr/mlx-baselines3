"""Type aliases for MLX Baselines3 to improve code readability and type safety."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np

# ============================================================================
# MLX Array Types
# ============================================================================

# Core MLX array type
MlxArray = mx.array

# Common array-like types that can be converted to MLX
ArrayLike = Union[np.ndarray, MlxArray, List, Tuple]

# Tensor dictionary for storing multiple arrays (e.g., network parameters)
TensorDict = Dict[str, MlxArray]

# Optional tensor for cases where tensor might not exist
OptionalTensor = Optional[MlxArray]

# ============================================================================
# Observation and Action Types
# ============================================================================

# Observations can be single arrays or dictionaries of arrays
ObsType = Union[MlxArray, Dict[str, MlxArray]]
NumpyObsType = Union[np.ndarray, Dict[str, np.ndarray]]

# Actions can be discrete (int) or continuous (float array)
ActionType = Union[int, MlxArray]
NumpyActionType = Union[int, np.ndarray]

# Combined observation-action pairs
Observation = Union[MlxArray, Dict[str, MlxArray]]
Action = Union[int, MlxArray]

# ============================================================================
# Environment Types
# ============================================================================

# Gymnasium environment types
GymEnv = gym.Env
GymSpace = gym.Space
GymObs = Union[np.ndarray, Dict[str, np.ndarray]]
GymAct = Union[int, np.ndarray]

# Step return types
StepReturn = Tuple[GymObs, float, bool, bool, Dict[str, Any]]
ResetReturn = Tuple[GymObs, Dict[str, Any]]

# ============================================================================
# Policy and Network Types
# ============================================================================

# Network parameter dictionaries
NetworkParams = Dict[str, MlxArray]
OptimizerState = Dict[str, Any]

# Policy prediction returns
PolicyPredict = Tuple[MlxArray, Optional[MlxArray]]  # (actions, states)
PolicyPredictWithLogProb = Tuple[
    MlxArray, MlxArray, Optional[MlxArray]
]  # (actions, log_probs, states)

# Value function outputs
ValuePredict = MlxArray
QValuePredict = MlxArray

# ============================================================================
# Training and Buffer Types
# ============================================================================

# Training batch data
RolloutBatch = Dict[str, MlxArray]
ReplayBatch = Dict[str, MlxArray]

# Buffer samples
BufferSample = Tuple[MlxArray, ...]

# Training step return (loss values)
TrainStepReturn = Dict[str, float]

# ============================================================================
# Schedule and Callback Types
# ============================================================================

# Schedule functions take training progress and return current value
# progress_remaining: 1.0 at start -> 0.0 at end
Schedule = Callable[[float], float]

# Learning rate schedules
LearningRateSchedule = Schedule

# Exploration schedules (e.g., epsilon decay)
ExplorationSchedule = Schedule

# Callback function signature
CallbackFn = Callable[[], Optional[bool]]

# ============================================================================
# Device and Configuration Types
# ============================================================================

# Device specification
Device = str  # "gpu" or "cpu" for MLX

# Configuration dictionaries
AlgorithmConfig = Dict[str, Any]
PolicyConfig = Dict[str, Any]
TrainConfig = Dict[str, Any]

# ============================================================================
# File I/O Types
# ============================================================================

# Path types for saving/loading models
PathLike = Union[str, bytes]

# Serializable data for model saving
SerializableData = Dict[str, Any]

# ============================================================================
# Gradient and Optimization Types
# ============================================================================

# Gradient dictionaries
GradientDict = Dict[str, MlxArray]

# Loss function signature
LossFn = Callable[..., MlxArray]

# Value and gradient function
ValueAndGradFn = Callable[..., Tuple[MlxArray, GradientDict]]

# ============================================================================
# Common Union Types for Flexibility
# ============================================================================

# Numeric types that can be used for hyperparameters
Numeric = Union[int, float]

# Array or scalar values
ArrayOrScalar = Union[MlxArray, float, int]

# Optional numeric values
OptionalNumeric = Optional[Numeric]

# String or None for optional parameters
OptionalStr = Optional[str]

# ============================================================================
# Algorithm-Specific Types
# ============================================================================

# Actor-Critic policy components
ActorOutput = Union[
    MlxArray, Tuple[MlxArray, MlxArray]
]  # actions or (actions, log_probs)
CriticOutput = MlxArray  # value estimates

# Q-function outputs (for SAC, TD3, DQN)
QFunctionOutput = MlxArray

# Entropy coefficient (for SAC)
EntropyCoefficientType = Union[float, str, Schedule]

# Target network update parameters
TargetUpdateParams = Dict[str, Any]
