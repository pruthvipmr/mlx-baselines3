# MLX Baselines3

A drop-in replacement for [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) that runs on Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) framework, providing native acceleration on Apple Silicon. Includes complete implementations of PPO, A2C, DQN, SAC, and TD3 with full API compatibility.

## Setup

### Requirements
- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10 or 3.11**

### Installation
```bash
git clone https://github.com/pruthvipmr/mlx-baselines3.git
cd mlx-baselines3
uv sync
```

### Quick Test
```bash
# Run a quick training example
uv run python examples/train_cartpole_ppo.py

# Run tests
uv run pytest
```

## Project Structure

```
mlx-baselines3/
├── mlx_baselines3/                    # Main package
│   ├── __init__.py                    # Package entry point
│   ├── common/                        # Shared infrastructure
│   │   ├── base_class.py             # BaseAlgorithm, OnPolicy/OffPolicy classes
│   │   ├── buffers.py                # RolloutBuffer, ReplayBuffer
│   │   ├── callbacks.py              # Callback system
│   │   ├── policies.py               # Base policy classes
│   │   ├── optimizers.py             # MLX optimizer adapters
│   │   ├── schedules.py              # Learning rate schedules
│   │   └── vec_env/                  # Vectorized environments
│   ├── ppo/                          # PPO algorithm
│   ├── a2c/                          # A2C algorithm  
│   ├── dqn/                          # DQN algorithm
│   ├── sac/                          # SAC algorithm
│   └── td3/                          # TD3 algorithm
├── tests/                            # Test suite (429+ tests)
├── examples/                         # Usage examples
└── pyproject.toml                    # Dependencies and config
```

### Core Components
- **Algorithms**: Complete implementations of 5 major RL algorithms
- **Policies**: MLP, CNN, and multi-input policy networks
- **Buffers**: Efficient rollout and replay buffers for on/off-policy learning
- **VecEnv**: Parallel environment execution and normalization
- **Callbacks**: Training monitoring, evaluation, and checkpointing
- **Optimizers**: MLX-compatible optimizers with gradient clipping and schedules
