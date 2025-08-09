# MLX Stable Baselines 3 Recreation Plan

## Project Overview

**Goal**: Create `mlx-baselines3` - a drop-in replacement for Stable Baselines 3 using Apple's MLX framework for M1/M2 Mac GPU acceleration.

**Target Interface**: Maintain exact API compatibility with SB3:
```python
# Should work identically to SB3
from mlx_baselines3 import PPO, SAC, A2C, TD3, DQN
from mlx_baselines3.common.vec_env import DummyVecEnv

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000)
model.save("model.zip")
```

**Scope**: ~3,000 lines covering 5 core algorithms + essential infrastructure.

---

## Repository Structure

```
mlx-baselines3/
├── mlx_baselines3/
│   ├── __init__.py                    # Algorithm exports
│   ├── common/
│   │   ├── __init__.py
│   │   ├── base_class.py              # BaseAlgorithm, OnPolicy/OffPolicy
│   │   ├── policies.py                # ActorCritic, QNetwork policies
│   │   ├── buffers.py                 # RolloutBuffer, ReplayBuffer
│   │   ├── vec_env/
│   │   │   ├── __init__.py
│   │   │   ├── base_vec_env.py        # VecEnv base class
│   │   │   └── dummy_vec_env.py       # Single-process vectorization
│   │   ├── utils.py                   # MLX utilities, device management
│   │   ├── distributions.py           # Action probability distributions
│   │   ├── torch_layers.py            # MLX neural network layers
│   │   ├── preprocessing.py           # Observation preprocessing
│   │   ├── callbacks.py               # Training callbacks (copy from SB3)
│   │   └── logger.py                  # Logging (minimal changes from SB3)
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── ppo.py                     # PPO algorithm implementation
│   │   └── policies.py                # PPO-specific policy classes
│   ├── sac/
│   │   ├── __init__.py
│   │   ├── sac.py                     # SAC algorithm implementation
│   │   └── policies.py                # SAC-specific policy classes
│   ├── a2c/
│   │   ├── __init__.py
│   │   ├── a2c.py                     # A2C algorithm implementation
│   │   └── policies.py                # A2C-specific policy classes
│   ├── td3/
│   │   ├── __init__.py
│   │   ├── td3.py                     # TD3 algorithm implementation
│   │   └── policies.py                # TD3-specific policy classes
│   └── dqn/
│       ├── __init__.py
│       ├── dqn.py                     # DQN algorithm implementation
│       └── policies.py                # DQN-specific policy classes
├── tests/
├── examples/
├── setup.py
└── README.md
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 weeks, ~600 lines) ✅ COMPLETED

#### 1.1 MLX Utilities (`common/utils.py`) ✅
**Purpose**: Device management and MLX-specific helper functions
**Status**: ✅ Completed (~280 lines)
**Delivered**: 2024-01-XX

**Key Functions Implemented**:
```python
def get_device() -> str:
    """Return 'gpu' if MLX GPU available, else 'cpu'"""
    
def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    
def polyak_update(params: dict, target_params: dict, tau: float) -> dict:
    """Soft update of target network parameters"""
    
def explained_variance(y_pred: mlx.core.array, y_true: mlx.core.array) -> float:
    """Compute explained variance for diagnostics"""
```

**Additional Features**:
- ✅ Tensor conversion utilities (`numpy_to_mlx`, `mlx_to_numpy`, `obs_as_mlx`)
- ✅ Learning rate scheduling (`get_linear_fn`, `get_schedule_fn`)
- ✅ Gradient utilities (`clip_grad_norm`, `update_learning_rate`)
- ✅ Safe operations (`safe_mean` with empty array protection)
- ✅ All tests passing on Apple Silicon GPU

#### 1.2 Type Aliases (`common/type_aliases.py`) ✅
**Status**: ✅ Completed (~140 lines)
**Delivered**: 2024-01-XX

**Comprehensive Type System Implemented**:
```python
# Core MLX types
MlxArray = mx.array
TensorDict = Dict[str, MlxArray]
ObsType = Union[MlxArray, Dict[str, MlxArray]]

# Training types
RolloutBatch = Dict[str, MlxArray]
ReplayBatch = Dict[str, MlxArray]
Schedule = Callable[[float], float]

# Policy types
PolicyPredict = Tuple[MlxArray, Optional[MlxArray]]
NetworkParams = Dict[str, MlxArray]
```

**Coverage**: All major RL algorithm components typed with full MLX support

#### 1.3 Base Algorithm Class (`common/base_class.py`) ✅
**Status**: ✅ Completed (~400 lines)
**Delivered**: 2024-01-XX

**Core Classes Implemented**:
- ✅ `BaseAlgorithm` - Abstract base for all RL algorithms
- ✅ `OnPolicyAlgorithm` - Base for PPO, A2C 
- ✅ `OffPolicyAlgorithm` - Base for SAC, TD3, DQN

**Essential Methods Implemented**:
```python
class BaseAlgorithm:
    def __init__(self, policy, env, learning_rate, device="auto", seed=None, **kwargs)
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> "BaseAlgorithm"
    def predict(self, observation, state=None, episode_start=None, deterministic=False)
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str, env=None, device="auto", **kwargs)
    def set_parameters(self, load_path_or_dict, exact_match=True)
    def get_parameters(self) -> Dict[str, Any]
```

**MLX Integration Features**:
- ✅ Automatic MLX GPU detection and device management
- ✅ MLX tensor handling for observations and actions
- ✅ Cloudpickle-based save/load (MLX arrays supported)
- ✅ Learning rate scheduling integration
- ✅ Progress tracking for training loops
- ✅ Vectorized environment detection
- ✅ Full test coverage with mock policies

**Testing Results**:
- ✅ All initialization, prediction, save/load tests passing
- ✅ Device handling verified (GPU detection working)
- ✅ Learning rate scheduling validated
- ✅ Parameter management tested
- ✅ Integration with existing test suite successful

---

**Phase 1 Summary**:
- **Total Lines**: ~820 lines (exceeded target due to comprehensive features)
- **Completion**: 100% of planned features + additional utilities
- **Quality**: Full test coverage, Apple Silicon GPU verified
- **Next**: Ready for Phase 2 infrastructure (buffers, vectorized envs)

---

### Phase 2: Infrastructure (3-4 weeks, ~800 lines)

#### 2.1 Vectorized Environments (`common/vec_env/`) ✅ COMPLETED

**Status**: ✅ Completed (~450 lines)
**Delivered**: 2024-01-XX (Phase 2.1)

**Base Class** (`base_vec_env.py`): ✅ (~280 lines)
```python
class VecEnv:
    """Abstract base class for vectorized environments"""
    def step(self, actions): pass
    def reset(self): pass
    def close(self): pass
    def get_attr(self, attr_name): pass
    def set_attr(self, attr_name, value): pass
```

**Key Features Implemented**:
- ✅ Complete VecEnv abstract interface with all required methods
- ✅ VecEnvWrapper base class for easy environment wrapping
- ✅ Proper type hints using MLX type aliases
- ✅ MLX tensor compatibility for observations
- ✅ Comprehensive attribute and method delegation
- ✅ Environment synchronization and rendering support

**DummyVecEnv** (`dummy_vec_env.py`): ✅ (~350 lines)
- ✅ **Enhanced from SB3** - Full MLX integration with zero PyTorch dependencies
- ✅ Handles multiple environments in single process sequentially
- ✅ Automatic episode reset with terminal observation preservation
- ✅ Dictionary observation space support
- ✅ Complete async stepping interface (step_async/step_wait)

**Additional Features**:
- ✅ `make_vec_env` utility function for easy vectorized environment creation
- ✅ Support for environment wrappers and custom vectorized environment classes
- ✅ Proper seed management across multiple environments
- ✅ Full rendering support (human and rgb_array modes)
- ✅ Environment attribute/method access with indices support
- ✅ Duplicate environment instance detection and error handling

**Testing Results**:
- ✅ 20/20 unit tests passing (100% success rate)
- ✅ Comprehensive test coverage including edge cases
- ✅ Integration tested with real gymnasium environments (CartPole-v1)
- ✅ Dictionary observation space testing
- ✅ Environment wrapper functionality validated
- ✅ Memory management and auto-reset behavior verified
- ✅ All existing imports still working (no regressions)

#### 2.2 Experience Buffers (`common/buffers.py`) ✅ COMPLETED

**Status**: ✅ Completed (~550 lines)
**Delivered**: 2024-01-XX (Phase 2.2)

**BaseBuffer Class**: ✅ (~120 lines)
- ✅ Abstract base class for all experience buffers
- ✅ Handles observation/action space setup (Box, Discrete, Dict spaces)
- ✅ MLX tensor conversion utilities
- ✅ Memory management and buffer size tracking

**RolloutBuffer** (for PPO/A2C): ✅ (~200 lines)
```python
class RolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", gae_lambda=1.0, gamma=0.99, n_envs=1):
        """Buffer for on-policy algorithms"""
    
    def add(self, obs, action, reward, episode_start, value, log_prob):
        """Add transition to buffer"""
    
    def get(self, batch_size=None):
        """Sample batch from buffer - convert to MLX arrays"""
    
    def compute_returns_and_advantage(self, last_values, dones):
        """Compute GAE advantages and returns"""
```

**Key Features Implemented**:
- ✅ Generalized Advantage Estimation (GAE) with configurable lambda
- ✅ Returns computation using discount factor (gamma)
- ✅ Efficient batch sampling with random shuffling
- ✅ Support for dictionary observation spaces
- ✅ MLX tensor conversion on data retrieval
- ✅ Memory-efficient storage (numpy → MLX conversion on demand)

**ReplayBuffer** (for SAC/TD3/DQN): ✅ (~230 lines)
```python
class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", n_envs=1):
        """Buffer for off-policy algorithms"""
    
    def add(self, obs, next_obs, action, reward, done, infos):
        """Add transition to buffer"""
    
    def sample(self, batch_size, env=None):
        """Sample random batch - convert to MLX arrays"""
```

**Advanced Features Implemented**:
- ✅ Circular buffer implementation for memory efficiency
- ✅ Random batch sampling with proper indexing
- ✅ Memory optimization mode (next_obs computed on-the-fly)
- ✅ Multi-environment support with proper indexing
- ✅ Dictionary observation space handling
- ✅ Episode boundary detection and handling

**MLX Integration Features**:
- ✅ Zero PyTorch dependencies - Pure MLX implementation
- ✅ Efficient numpy → MLX conversion on sampling/retrieval
- ✅ Support for MLX GPU acceleration when available
- ✅ Memory-optimized storage (store numpy, convert to MLX on demand)
- ✅ Proper dtype handling and conversion

**Testing Results**:
- ✅ 18/18 buffer-specific unit tests passing (100% success rate)
- ✅ Comprehensive test coverage for both buffer types
- ✅ GAE computation validation with known values
- ✅ Dictionary observation space testing
- ✅ Memory optimization testing
- ✅ Circular buffer behavior validation
- ✅ Integration testing with vectorized environments
- ✅ MLX tensor conversion testing
- ✅ Multi-environment indexing validation
- ✅ Total test suite: 42/42 tests passing (no regressions)

#### 2.3 Preprocessing (`common/preprocessing.py`) ✅ COMPLETED

**Status**: ✅ Completed (~400 lines)
**Delivered**: 2024-01-XX (Phase 2.3)

**Core Functions Implemented**: ✅ (~400 lines)
```python
def preprocess_obs(obs, observation_space, normalize_images=True):
    """Preprocess observations for neural networks"""

def is_image_space(observation_space):
    """Check if observation space contains images"""

def maybe_transpose(observation, observation_space):
    """Transpose image observations if needed"""

def normalize_image(observation, dtype=np.float32):
    """Normalize image observations from [0, 255] to [0, 1]"""

def get_obs_shape(observation_space):
    """Get observation shape after preprocessing"""

def flatten_obs(obs, observation_space):
    """Flatten observations for algorithms requiring flat input"""

def convert_to_mlx(obs):
    """Convert preprocessed observations to MLX arrays"""
```

**Key Features Implemented**:
- ✅ **Image Space Detection**: Robust detection of image observations (Box spaces with uint8 dtype)
- ✅ **Channels Format Handling**: Support for both channels-first (C,H,W) and channels-last (H,W,C) formats
- ✅ **Automatic Transposition**: Convert channels-last to channels-first for MLX/DL compatibility
- ✅ **Image Normalization**: Scale pixel values from [0,255] to [0,1] range
- ✅ **Dictionary Observations**: Full support for Dict observation spaces with mixed content
- ✅ **Batch Processing**: Handle both single observations and batches
- ✅ **Multiple Space Types**: Support for Box, Discrete, MultiBinary, MultiDiscrete spaces

**Advanced Features**:
- ✅ **Shape Computation**: Predict observation shapes after preprocessing for network design
- ✅ **Flattening Utilities**: Flatten complex observations for linear layers
- ✅ **Nested Space Detection**: Identify and handle nested observation structures
- ✅ **Memory Efficiency**: In-place operations where possible
- ✅ **MLX Integration**: Seamless conversion to MLX arrays with proper dtype handling

**Image Processing Capabilities**:
- ✅ **Format Detection**: Automatic detection of channels-first vs channels-last
- ✅ **Size Validation**: Reasonable spatial dimension checking (8-2048 pixels)
- ✅ **Channel Validation**: Support for 1, 3, 4 channel images (grayscale, RGB, RGBA)
- ✅ **Batch Transposition**: Handle batched image observations correctly
- ✅ **Type Safety**: Proper dtype conversion and validation

**MLX Conversions**:
- ✅ Replace `torch.nn.functional` operations with MLX equivalents
- ✅ Convert image normalization to MLX tensor operations
- ✅ Pure MLX implementation with zero PyTorch dependencies
- ✅ Efficient numpy → MLX conversion pipelines

**Testing Results**:
- ✅ 32/32 preprocessing-specific unit tests passing (100% success rate)
- ✅ Comprehensive test coverage for all observation space types
- ✅ Image space detection validation with edge cases
- ✅ Transposition correctness verification
- ✅ Normalization accuracy testing
- ✅ Dictionary observation space handling
- ✅ Integration testing with buffers and vectorized environments
- ✅ MLX tensor conversion validation
- ✅ Backward compatibility with existing codebase verified
- ✅ Total test suite: 74/74 tests passing (no regressions)

---

**Phase 2 Summary**:
- **Total Lines**: ~1,400 lines (vectorized environments + experience buffers + preprocessing)
- **Components Delivered**:
  - ✅ **Vectorized Environments** (~450 lines): Complete VecEnv interface, DummyVecEnv, VecEnvWrapper
  - ✅ **Experience Buffers** (~550 lines): RolloutBuffer, ReplayBuffer, BaseBuffer
  - ✅ **Preprocessing** (~400 lines): Image processing, observation normalization, MLX conversion
- **Test Coverage**: 74/74 tests passing (100% success rate)
- **Features**: Full MLX integration, zero PyTorch dependencies, comprehensive API compatibility
- **Next**: Phase 3 Neural Networks (MLX layers, policies, feature extractors)

---

### Phase 3: Neural Networks & Policies (3-4 weeks, ~800 lines) ✅ COMPLETED

#### 3.1 MLX Neural Network Layers (`common/torch_layers.py`) ✅ COMPLETED
**Status**: ✅ Completed (~320 lines)
**Delivered**: 2024-01-XX

**Purpose**: MLX-based replacements for PyTorch neural network components
**Key Components**:
```python
class MlxModule:
    """Base class for MLX neural network modules"""
    def __init__(self): pass
    def __call__(self, x): pass
    def parameters(self): pass
    
class MlxSequential(MlxModule):
    """Sequential container for MLX layers"""
    def __init__(self, *layers): pass
    
def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], 
               activation_fn: Type[nn.Module] = nn.ReLU) -> MlxSequential:
    """Create MLP with specified architecture"""
```

**Features Implemented**:
- ✅ Base MlxModule class with parameter management
- ✅ MlxSequential container for layer chaining  
- ✅ MLP creation utilities (`create_mlp`)
- ✅ Weight initialization (Xavier, orthogonal, normal)
- ✅ Activation function wrappers
- ✅ MlxLinear layer with bias support
- ✅ Feature extractors (BaseFeaturesExtractor, FlattenExtractor, MlpExtractor)
- ✅ Comprehensive test suite (25 tests passing)

#### 3.2 Action Distributions (`common/distributions.py`) ✅ COMPLETED
**Status**: ✅ Completed (~380 lines)
**Delivered**: 2024-01-XX

**Purpose**: Probability distributions for action sampling and log probabilities
**Key Components**:
```python
class Distribution:
    """Base class for action distributions"""
    def sample(self) -> mx.array: pass
    def log_prob(self, actions: mx.array) -> mx.array: pass
    def entropy(self) -> mx.array: pass
    
class CategoricalDistribution(Distribution):
    """For discrete action spaces"""
    def __init__(self, action_dim: int): pass
    
class DiagGaussianDistribution(Distribution):  
    """For continuous action spaces"""
    def __init__(self, action_dim: int): pass
```

**Features Implemented**:
- ✅ Categorical distribution for discrete actions with Gumbel-max sampling
- ✅ Diagonal Gaussian for continuous actions
- ✅ Squashed Gaussian for bounded continuous actions (SAC)
- ✅ Log probability computation with MLX and numerical stability
- ✅ Entropy calculation for regularization
- ✅ Jacobian correction for squashed distributions
- ✅ Deterministic sampling mode
- ✅ Convenience methods for easy integration
- ✅ make_proba_distribution factory function
- ✅ Comprehensive test suite (24 tests passing)

#### 3.3 Core Policies (`common/policies.py`) ✅ COMPLETED
**Status**: ✅ Completed (~430 lines)  
**Delivered**: 2024-01-XX

**Purpose**: Policy network implementations for all RL algorithms
**Key Components**:
```python
class BasePolicy:
    """Abstract base class for all policies"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs): pass
    def predict(self, observation, deterministic=False): pass
    def get_distribution(self, obs): pass
    
class ActorCriticPolicy(BasePolicy):
    """Policy for on-policy algorithms (PPO, A2C)"""
    def __init__(self, observation_space, action_space, lr_schedule, 
                 net_arch=None, activation_fn=nn.Tanh, **kwargs): pass
    def forward(self, obs, deterministic=False): pass
    def evaluate_actions(self, obs, actions): pass
    def get_distribution(self, obs): pass
    def predict_values(self, obs): pass
```

**Features Implemented**:
- ✅ BasePolicy abstract interface with full SB3 compatibility
- ✅ ActorCriticPolicy for PPO/A2C with continuous and discrete actions
- ✅ MultiInputActorCriticPolicy for dict observations
- ✅ Feature extractors integration (MLP, flatten) 
- ✅ Value function networks with proper initialization
- ✅ Action/value prediction methods with batching support
- ✅ Policy aliases (MlpPolicy, CnnPolicy, MultiInputPolicy)
- ✅ Deterministic and stochastic prediction modes
- ✅ Shared and separate features extractors
- ✅ Custom network architectures support
- ✅ Comprehensive test suite (25 tests passing)

---

**Phase 3 Summary**:
- **Total Lines**: ~1,130 lines (exceeded target due to comprehensive features)
- **Completion**: 100% of planned features + additional utilities
- **Quality**: Full test coverage, 74/74 tests passing on new components
- **Testing**: 148/148 total tests passing (100% success rate)
- **Next**: Ready for Phase 4 (First Algorithm Implementation - PPO)

---

### Phase 4: Algorithm Implementations (4-6 weeks, ~1000 lines)

### Phase 4: Algorithm Implementations (planned)

#### 4.1 PPO Implementation (`ppo/ppo.py`)

**Core Algorithm Class**:
```python
class PPO(OnPolicyAlgorithm):
    def __init__(self, policy, env, learning_rate=3e-4, n_steps=2048, batch_size=64, 
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, **kwargs):
        super().__init__(policy, env, learning_rate, **kwargs)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
    
    def train(self):
        """PPO training step with clipped objective"""
        # Convert PyTorch tensors to MLX arrays
        rollout_data = self.rollout_buffer.get(self.batch_size)
        
        for epoch in range(self.n_epochs):
            for batch in rollout_data:
                # Convert batch to MLX
                obs = mx.array(batch.observations)
                actions = mx.array(batch.actions)
                old_values = mx.array(batch.old_values)
                old_log_prob = mx.array(batch.old_log_prob)
                advantages = mx.array(batch.advantages)
                returns = mx.array(batch.returns)
                
                # Policy evaluation
                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                
                # Policy loss with clipping
                ratio = mx.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * mx.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -mx.minimum(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = mx.mean((returns - values) ** 2)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                
                # Gradient computation and optimization (MLX style)
                loss_and_grad_fn = mlx.value_and_grad(lambda p: loss)
                loss_val, grads = loss_and_grad_fn(self.policy.parameters())
                self.optimizer.update(self.policy, grads)
```

**MLX Conversion Points**:
- Replace `torch.clamp()` with `mx.clip()`
- Replace `torch.exp()` with `mx.exp()`
- Replace `torch.min()` with `mx.minimum()`
- Convert loss computation and backpropagation to MLX gradient system

#### 4.2 SAC Implementation (`sac/sac.py`)

**Key Components**:
```python
class SAC(OffPolicyAlgorithm):
    def __init__(self, policy, env, learning_rate=3e-4, buffer_size=1000000, 
                 tau=0.005, gamma=0.99, train_freq=1, target_entropy="auto", **kwargs):
        super().__init__(policy, env, learning_rate, **kwargs)
        self.tau = tau
        self.target_entropy = target_entropy
        self._setup_model()
    
    def _setup_model(self):
        """Setup actor, critic networks and target networks"""
        # Create target networks for critics
        self.critic_target = copy.deepcopy(self.policy.critic)
        
        # Automatic entropy tuning
        if self.target_entropy == "auto":
            self.target_entropy = -mx.array(self.action_space.shape).prod()
        
        self.log_ent_coef = mx.zeros((1,))  # Learnable entropy coefficient
    
    def train(self, gradient_steps: int, batch_size: int = 64):
        """SAC training with entropy regularization"""
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            
            # Convert to MLX arrays
            obs = mx.array(replay_data.observations)
            actions = mx.array(replay_data.actions)
            next_obs = mx.array(replay_data.next_observations)
            rewards = mx.array(replay_data.rewards)
            dones = mx.array(replay_data.dones)
            
            # Critic loss (twin critics)
            with mx.stop_gradient():
                next_actions, next_log_prob = self.actor(next_obs)
                next_q_values = mx.minimum(
                    self.critic_target.q1_forward(next_obs, next_actions),
                    self.critic_target.q2_forward(next_obs, next_actions)
                )
                target_q_values = rewards + (1 - dones) * self.gamma * (
                    next_q_values - self.ent_coef * next_log_prob
                )
            
            current_q_values_1 = self.critic.q1_forward(obs, actions)
            current_q_values_2 = self.critic.q2_forward(obs, actions)
            
            critic_loss = mx.mean((current_q_values_1 - target_q_values) ** 2) + \
                         mx.mean((current_q_values_2 - target_q_values) ** 2)
            
            # Actor loss
            actions_pred, log_prob = self.actor(obs)
            q_values_pred = mx.minimum(
                self.critic.q1_forward(obs, actions_pred),
                self.critic.q2_forward(obs, actions_pred)
            )
            actor_loss = mx.mean(self.ent_coef * log_prob - q_values_pred)
            
            # Entropy coefficient loss
            ent_coef_loss = -mx.mean(self.log_ent_coef * (log_prob + self.target_entropy))
            
            # Update networks using MLX optimizers
            # ... gradient computation and optimization steps
```

#### 4.3 Remaining Algorithms (A2C, TD3, DQN)

**A2C** (`a2c/a2c.py`): ~180 lines
- Simplified version of PPO without clipping
- Single gradient step over rollout data
- Replace RMSprop optimizer with MLX equivalent

**TD3** (`td3/td3.py`): ~280 lines  
- Delayed policy updates (every 2 critic updates)
- Target policy smoothing with noise
- Twin critic networks with target networks

**DQN** (`dqn/dqn.py`): ~320 lines
- Epsilon-greedy exploration with linear decay
- Target network updates every N steps
- Experience replay with prioritized sampling (optional)

---

## Core MLX Conversion Patterns

### Tensor Operations
```python
# PyTorch → MLX
torch.tensor(data) → mx.array(data)
torch.zeros(shape) → mx.zeros(shape)
torch.ones(shape) → mx.ones(shape)
torch.randn(shape) → mx.random.normal(shape)
torch.cat([a, b], dim=0) → mx.concatenate([a, b], axis=0)
torch.clamp(x, min, max) → mx.clip(x, min, max)
torch.exp(x) → mx.exp(x)
torch.log(x) → mx.log(x)
torch.mean(x) → mx.mean(x)
torch.sum(x) → mx.sum(x)
torch.min(a, b) → mx.minimum(a, b)
torch.max(a, b) → mx.maximum(a, b)
```

### Neural Networks
```python
# PyTorch → MLX
torch.nn.Linear(in_features, out_features) → mlx.nn.Linear(in_features, out_features)
torch.nn.Conv2d(in_ch, out_ch, kernel) → mlx.nn.Conv2d(in_ch, out_ch, kernel)
torch.nn.ReLU() → mlx.nn.ReLU()
torch.nn.Tanh() → mlx.nn.Tanh()
torch.nn.Sequential(*layers) → Custom MLX sequential container
F.mse_loss(pred, target) → mx.mean((pred - target) ** 2)
F.smooth_l1_loss(pred, target) → Custom Huber loss implementation
```

### Optimizers and Gradients
```python
# PyTorch → MLX
torch.optim.Adam(params, lr) → mlx.optimizers.Adam(learning_rate=lr)
torch.optim.RMSprop(params, lr) → mlx.optimizers.RMSprop(learning_rate=lr)

# Gradient computation
loss.backward() → 
# MLX equivalent:
loss_and_grad_fn = mlx.value_and_grad(loss_function)
loss_val, grads = loss_and_grad_fn(model.parameters())
optimizer.update(model, grads)
```

### Device Management
```python
# PyTorch → MLX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") →
device = "gpu" if mx.metal.is_available() else "cpu"

tensor.to(device) → 
# MLX handles device placement automatically

torch.cuda.empty_cache() → 
# Not needed in MLX (unified memory)
```

---

## Testing Strategy

### Unit Tests Structure
```
tests/
├── test_buffers.py          # RolloutBuffer, ReplayBuffer
├── test_policies.py         # ActorCritic, Critic policies  
├── test_distributions.py    # Action distributions
├── test_vec_env.py         # Vectorized environments
├── test_algorithms/
│   ├── test_ppo.py         # PPO training and prediction
│   ├── test_sac.py         # SAC training and prediction  
│   └── test_others.py      # A2C, TD3, DQN
└── test_compatibility.py   # API compatibility with SB3
```

### Benchmark Tests
```python
def test_cartpole_ppo():
    """Test PPO on CartPole-v1 environment"""
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    
    # Test should achieve >195 average reward
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)
    assert mean_reward > 195

def test_pendulum_sac():
    """Test SAC on Pendulum-v1 environment"""
    env = gym.make("Pendulum-v1") 
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    
    # Test should achieve >-200 average reward
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)
    assert mean_reward > -200
```

### Compatibility Tests
```python
def test_forest_compatibility():
    """Test compatibility with forest's Agent wrapper"""
    from forest import TradingEnv, Agent
    from mlx_baselines3 import PPO
    
    # Should work exactly like SB3
    env = TradingEnv(...)
    model = PPO("MlpPolicy", env)
    agent = Agent(model)
    agent.train(env, total_timesteps=1000)
    
    action = agent.predict(observation)
    agent.save("test_model.zip")
    
    loaded_agent = Agent.load("test_model.zip", PPO, env)
    assert loaded_agent.predict(observation) == action
```

---

## Package Configuration

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="mlx-baselines3",
    version="0.1.0",
    description="MLX implementation of Stable Baselines 3 for Apple Silicon",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.0.9",
        "gymnasium>=0.29.0", 
        "numpy>=1.20.0",
        "cloudpickle",
    ],
    extras_require={
        "tests": ["pytest", "pytest-cov"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

### __init__.py Exports
```python
# mlx_baselines3/__init__.py
__version__ = "0.1.0"

from mlx_baselines3.a2c import A2C
from mlx_baselines3.dqn import DQN  
from mlx_baselines3.ppo import PPO
from mlx_baselines3.sac import SAC
from mlx_baselines3.td3 import TD3

__all__ = ["A2C", "DQN", "PPO", "SAC", "TD3"]
```

---

## Development Milestones

### Milestone 1: PPO-Only (4-6 weeks)
- [ ] Base infrastructure (BaseAlgorithm, OnPolicyAlgorithm)
- [ ] RolloutBuffer with MLX tensor conversion
- [ ] DummyVecEnv (copy from SB3)
- [ ] ActorCriticPolicy with MLX neural networks
- [ ] PPO algorithm with clipped objective
- [ ] Save/load functionality
- [ ] CartPole benchmark passing

**Deliverable**: `from mlx_baselines3 import PPO` working with forest

### Milestone 2: SAC Addition (2-3 weeks)
- [ ] OffPolicyAlgorithm base class
- [ ] ReplayBuffer with MLX tensor conversion  
- [ ] Continuous action policies (Actor, Critic)
- [ ] SAC with entropy regularization
- [ ] Pendulum benchmark passing

### Milestone 3: Remaining Algorithms (3-4 weeks)
- [ ] A2C implementation
- [ ] TD3 with target networks and delayed updates
- [ ] DQN with epsilon-greedy and target networks
- [ ] All benchmarks passing

### Milestone 4: Production Ready (2-3 weeks)
- [ ] Comprehensive test suite
- [ ] Documentation and examples
- [ ] CI/CD pipeline
- [ ] PyPI package publishing
- [ ] Performance benchmarking vs SB3

**Total Estimated Timeline**: 12-16 weeks for complete implementation

---

## Success Criteria

1. **API Compatibility**: Drop-in replacement for SB3 in forest codebase
2. **Performance**: Achieve comparable learning performance on standard benchmarks
3. **M1 GPU Utilization**: Demonstrate GPU acceleration on Apple Silicon
4. **Stability**: Pass comprehensive test suite with >95% coverage
5. **Documentation**: Complete API documentation with examples

This plan provides a concrete roadmap for recreating Stable Baselines 3 with MLX, maintaining full compatibility while leveraging Apple Silicon GPU acceleration.
