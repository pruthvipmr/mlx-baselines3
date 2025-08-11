# MLX Baselines3

A drop-in replacement for [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) (SB3) that runs on Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) framework, providing native acceleration on Apple Silicon.

## Features

- **Full SB3 API Compatibility**: Drop-in replacement with identical APIs and behavior
- **Apple Silicon Optimized**: Native MLX acceleration on M-series chips (CPU and GPU)
- **5 Core Algorithms**: PPO, A2C, DQN, SAC, TD3 with comprehensive implementations
- **Complete Infrastructure**: Policies, buffers, callbacks, logging, schedules, and vectorized environments
- **Performance Optimized**: JIT compilation and optimized training loops with 10-20% speedups
- **Comprehensive Testing**: 429+ tests with GitHub Actions CI ensuring reliability

## Supported Algorithms

| Algorithm | Type | Status | Action Spaces | Observation Spaces |
|-----------|------|--------|---------------|-------------------|
| **PPO** | On-Policy | Complete | Discrete, Box, MultiDiscrete, MultiBinary | Box, Dict |
| **A2C** | On-Policy | Complete | Discrete, Box, MultiDiscrete, MultiBinary | Box, Dict |
| **DQN** | Off-Policy | Complete | Discrete, MultiDiscrete, MultiBinary | Box, Dict |
| **SAC** | Off-Policy | Complete | Box | Box, Dict |
| **TD3** | Off-Policy | Complete | Box | Box, Dict |

## Installation

### Requirements

- **macOS**: Required for MLX framework
- **Python**: 3.10 or 3.11
- **Apple Silicon**: Recommended for best performance (M1/M2/M3/M4)

### Install from Source

```bash
git clone https://github.com/pruthvipmr/mlx-baselines3.git
cd mlx-baselines3
uv sync
```

### Dependencies

MLX Baselines3 requires:
- `mlx>=0.0.8` - Apple's machine learning framework
- `gymnasium>=0.28.0` - Environment interface
- `numpy>=1.21.0` - Numerical computing
- `cloudpickle>=1.6.0` - Serialization

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from mlx_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1")

# Train a PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load and test
model = PPO.load("ppo_cartpole")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### Advanced Features

```python
import gymnasium as gym
from mlx_baselines3 import SAC
from mlx_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mlx_baselines3.common.callbacks import EvalCallback

# Create and wrap environment
env = gym.make("Pendulum-v1")
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Setup evaluation callback
eval_env = gym.make("Pendulum-v1")
eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model/",
                           log_path="./logs/", eval_freq=1000)

# Train with advanced configuration
model = SAC("MlpPolicy", env, 
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            verbose=1)

model.learn(total_timesteps=50000, callback=eval_callback)
```

## Apple GPU Acceleration

MLX automatically uses Apple GPU when available. To monitor GPU usage:

```bash
# Monitor GPU activity
sudo powermetrics --samplers gpu_power -n 1 -i 1000
```

For optimal performance:
- Use MLX arrays stay on GPU throughout training
- Batch operations when possible  
- Enable JIT compilation (automatically used where available)

## Policy Types

All algorithms support multiple policy architectures:

- **MlpPolicy**: Multi-layer perceptron for vector observations
- **CnnPolicy**: Convolutional neural network for image observations  
- **MultiInputPolicy**: Multiple input types (dict observations)

```python
# Custom network architecture
policy_kwargs = dict(
    net_arch=[128, 128, 64],  # Hidden layer sizes
    activation_fn="tanh",     # Activation function
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

## Vectorized Environments

```python
from mlx_baselines3.common.vec_env import DummyVecEnv, make_vec_env

# Create multiple parallel environments
env = make_vec_env("CartPole-v1", n_envs=4, seed=0)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

## Callbacks and Logging

```python
from mlx_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    StopTrainingOnRewardThreshold
)

# Setup callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./checkpoints/")
eval_callback = EvalCallback(eval_env, eval_freq=500)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

model.learn(total_timesteps=10000, 
           callback=[checkpoint_callback, eval_callback, stop_callback])
```

## Hyperparameter Schedules

```python
from mlx_baselines3.common.schedules import linear_schedule

# Linear learning rate decay
model = PPO("MlpPolicy", env, 
           learning_rate=linear_schedule(1e-3),  # 1e-3 to 0
           clip_range=linear_schedule(0.2))      # 0.2 to 0
```

## Saving and Loading

```python
# Save complete model state
model.save("my_model")

# Load model (environment auto-detected if available)
model = PPO.load("my_model")

# Load with custom environment
model = PPO.load("my_model", env=custom_env)

# Save/load with VecNormalize
env.save("vec_normalize.pkl")
env = VecNormalize.load("vec_normalize.pkl", env)
```

## Performance Tips

1. **Use Apple Silicon**: M-series chips provide significant acceleration
2. **Batch Size**: Larger batches (64-256) often perform better on MLX
3. **Float32**: MLX Baselines3 enforces float32 for optimal performance
4. **JIT Compilation**: Automatically enabled for core operations (18% speedup)
5. **Memory Management**: MLX handles GPU memory automatically

## Comparison with Stable Baselines3

| Feature | Stable Baselines3 | MLX Baselines3 |
|---------|------------------|----------------|
| **Platform** | CPU, CUDA | Apple Silicon |
| **Backend** | PyTorch | MLX |
| **Memory** | Manual GPU management | Automatic |
| **Performance** | Good on NVIDIA | Optimized for Apple |
| **API** | Complete | Compatible |

## Examples

See the [`examples/`](examples/) directory for complete training scripts:

- [`train_cartpole_ppo.py`](examples/train_cartpole_ppo.py) - PPO on CartPole
- [`train_cartpole_dqn.py`](examples/train_cartpole_dqn.py) - DQN on CartPole  
- [`train_pendulum_sac.py`](examples/train_pendulum_sac.py) - SAC on Pendulum

## Caveats and Limitations

- **macOS Only**: MLX framework requires Apple platforms
- **Algorithm Coverage**: Core algorithms implemented, some SB3 features may be missing
- **Memory Usage**: MLX uses unified memory, monitor system RAM usage
- **Debugging**: MLX stack traces may differ from PyTorch
- **Ecosystem**: Smaller ecosystem compared to PyTorch-based tools

## Testing

Run the comprehensive test suite:

```bash
# All tests
uv run pytest

# Fast tests only  
uv run pytest -m "not slow"

# Specific algorithm
uv run pytest tests/test_ppo.py
```

## Acknowledgments

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for the original implementation
- [MLX](https://ml-explore.github.io/mlx/) team at Apple for the framework
- [Gymnasium](https://gymnasium.farama.org/) for environment interfaces
