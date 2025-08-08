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

### Phase 1: Foundation (2-3 weeks, ~600 lines)

#### 1.1 MLX Utilities (`common/utils.py`)
**Purpose**: Device management and MLX-specific helper functions

**Key Functions**:
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

**MLX Conversions**:
- Replace `torch.device` with MLX device strings
- Replace `torch.manual_seed` with appropriate MLX seeding
- Convert tensor operations to `mlx.core` functions

#### 1.2 Type Aliases (`common/type_aliases.py`)
```python
import mlx.core as mx
from typing import Union, Dict, Any

MlxArray = mx.array
TensorDict = Dict[str, MlxArray]
PyTorchObs = Union[MlxArray, Dict[str, MlxArray]]
```

#### 1.3 Base Algorithm Class (`common/base_class.py`)
**Core Class**: `BaseAlgorithm` (abstract base for all RL algorithms)

**Essential Methods**:
```python
class BaseAlgorithm:
    def __init__(self, policy, env, learning_rate, device="auto", seed=None, **kwargs):
        """Initialize algorithm with policy and environment"""
    
    def learn(self, total_timesteps: int, callback=None, **kwargs) -> "BaseAlgorithm":
        """Train the algorithm - main entry point"""
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Predict action given observation"""
    
    def save(self, path: str) -> None:
        """Save model to disk"""
    
    @classmethod
    def load(cls, path: str, env=None, device="auto", **kwargs):
        """Load model from disk"""
    
    def set_parameters(self, load_path_or_dict, exact_match=True):
        """Set model parameters"""
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
```

**MLX Conversion Points**:
- Replace `torch.save/torch.load` with MLX serialization (`mx.savez/mx.load`)
- Convert device management from PyTorch to MLX
- Replace PyTorch optimizers with MLX optimizers

### Phase 2: Infrastructure (3-4 weeks, ~800 lines)

#### 2.1 Vectorized Environments (`common/vec_env/`)

**Base Class** (`base_vec_env.py`):
```python
class VecEnv:
    """Abstract base class for vectorized environments"""
    def step(self, actions): pass
    def reset(self): pass
    def close(self): pass
    def get_attr(self, attr_name): pass
    def set_attr(self, attr_name, value): pass
```

**DummyVecEnv** (`dummy_vec_env.py`):
- **Copy unchanged from SB3** - minimal PyTorch dependencies
- Handles multiple environments in single process
- ~150 lines

#### 2.2 Experience Buffers (`common/buffers.py`)

**RolloutBuffer** (for PPO/A2C):
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

**ReplayBuffer** (for SAC/TD3/DQN):
```python
class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", n_envs=1):
        """Buffer for off-policy algorithms"""
    
    def add(self, obs, next_obs, action, reward, done, infos):
        """Add transition to buffer"""
    
    def sample(self, batch_size, env=None):
        """Sample random batch - convert to MLX arrays"""
```

**MLX Conversion Points**:
- Store data as numpy arrays, convert to MLX in `get()`/`sample()`
- Replace `torch.tensor()` calls with `mx.array()`
- Convert GAE computation to MLX operations

#### 2.3 Preprocessing (`common/preprocessing.py`)
**Key Functions**:
```python
def preprocess_obs(obs, observation_space, normalize_images=True):
    """Preprocess observations for neural networks"""

def is_image_space(observation_space):
    """Check if observation space contains images"""

def maybe_transpose(observation, observation_space):
    """Transpose image observations if needed"""
```

**MLX Conversions**:
- Replace `torch.nn.functional` operations with MLX equivalents
- Convert image normalization to MLX tensor operations

### Phase 3: Neural Networks (4-6 weeks, ~1000 lines)

#### 3.1 MLX Layers (`common/torch_layers.py`)

**Feature Extractors**:
```python
class BaseFeaturesExtractor(mlx.nn.Module):
    """Base class for feature extraction networks"""
    
class MlpExtractor(mlx.nn.Module):
    """Multi-layer perceptron feature extractor"""
    def __init__(self, feature_dim, net_arch, activation_fn=mlx.nn.ReLU):
        # Convert from PyTorch Sequential to MLX modules
    
class NatureCNN(mlx.nn.Module):
    """CNN from DQN Nature paper"""
    def __init__(self, observation_space, features_dim=512):
        # Convert CNN layers from torch.nn to mlx.nn
```

**Utility Functions**:
```python
def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn=mlx.nn.ReLU) -> List[mlx.nn.Module]:
    """Create MLP layers for policies and value functions"""
```

**MLX Conversion Matrix**:
```python
# PyTorch → MLX Layer Mappings
torch.nn.Linear → mlx.nn.Linear
torch.nn.Conv2d → mlx.nn.Conv2d  
torch.nn.ReLU → mlx.nn.ReLU
torch.nn.Tanh → mlx.nn.Tanh
torch.nn.Sequential → Custom sequential container
torch.nn.Flatten → mlx.core.flatten
```

#### 3.2 Action Distributions (`common/distributions.py`)

**Core Classes**:
```python
class Distribution:
    """Base class for action probability distributions"""
    def proba_distribution_net(self, latent_dim: int) -> mlx.nn.Module: pass
    def proba_distribution(self, action_logits): pass
    def log_prob(self, actions): pass
    def entropy(self): pass
    def sample(self): pass

class DiagGaussianDistribution(Distribution):
    """Diagonal Gaussian distribution for continuous actions"""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
    
    def proba_distribution_net(self, latent_dim: int):
        """Create network outputting mean and log_std"""
        mean_actions = mlx.nn.Linear(latent_dim, self.action_dim)
        log_std = mlx.nn.Linear(latent_dim, self.action_dim)
        return mean_actions, log_std
    
    def log_prob(self, actions):
        """Compute log probability of actions"""
        # Convert torch.distributions.Normal logic to MLX

class CategoricalDistribution(Distribution):
    """Categorical distribution for discrete actions"""
    def log_prob(self, actions):
        # Convert torch.distributions.Categorical logic to MLX
```

**MLX Conversion Points**:
- Replace `torch.distributions.Normal` with custom MLX implementation
- Convert log probability and entropy computations
- Implement sampling using MLX random number generation

#### 3.3 Policy Networks (`common/policies.py`)

**Base Policy Class**:
```python
class BasePolicy(mlx.nn.Module):
    """Base class for all policies"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
    
    def _build(self, lr_schedule): pass
    def forward(self, obs): pass
    def _get_constructor_parameters(self): pass
    def reset_noise(self): pass
    def _build_mlp_extractor(self): pass
```

**Actor-Critic Policy** (for PPO/A2C):
```python
class ActorCriticPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=mlx.nn.Tanh, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self._build(lr_schedule)
    
    def _build_mlp_extractor(self):
        """Build shared MLP feature extractor"""
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn
        )
    
    def forward(self, obs, deterministic=False):
        """Forward pass through policy and value networks"""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for policy optimization"""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
```

### Phase 4: Algorithm Implementations (4-5 weeks per algorithm)

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
