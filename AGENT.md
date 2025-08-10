# MLX Baselines3 Agent Guide

## Build/Test Commands
- **Install dependencies**: `uv sync`
- **Test**: `uv run pytest`
- **Single test**: `uv run pytest tests/test_<module>.py::<test_function>`
- **Format**: `uv run black mlx_baselines3/ tests/`
- **Lint**: `uv run flake8 mlx_baselines3/ tests/`
- **Type check**: `uv run mypy mlx_baselines3/`
- **Install dev deps**: `uv sync --extra dev`

## Architecture & Structure
- **Goal**: Drop-in replacement for Stable Baselines 3 using Apple's MLX framework
- **Core modules**: `mlx_baselines3/{ppo,sac,a2c,td3,dqn}/` - individual RL algorithms
- **Common infrastructure**: `mlx_baselines3/common/` - shared base classes, policies, buffers
- **Key components**: BaseAlgorithm, OnPolicy/OffPolicy classes, RolloutBuffer, ReplayBuffer
- **MLX integration**: Replace PyTorch tensors with `mx.array()`, use `mlx.nn` modules, `mlx.optimizers`

## Code Style & Conventions
- **Imports**: Use `import mlx.core as mx`, `import mlx.nn`, prefer absolute imports
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Types**: Use type hints with `mlx.core.array` for tensors, Union types for flexible inputs
- **Error handling**: Raise specific exceptions (ValueError, RuntimeError), include descriptive messages
- **Documentation**: Follow NumPy docstring style, document all public methods and classes
- **MLX patterns**: Use `mx.array()` instead of `torch.tensor()`, `mx.clip()` instead of `torch.clamp()`
- **NO EMOJIS**: NEVER use emojis anywhere in code, comments, docstrings, or commit messages

## Package Management
- Uses `uv` (not pip/conda)
- Dependencies in `pyproject.toml`
- Lock file: `uv.lock`

## Current File Structure

```
mlx-baselines3/
├── mlx_baselines3/                    # Main package
│   ├── __init__.py                    # Package entry point with algorithm imports
│   ├── common/                        # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── base_class.py             # BaseAlgorithm, OnPolicy/OffPolicy classes
│   │   ├── buffers.py                # RolloutBuffer for on-policy algorithms
│   │   ├── distributions.py          # Action distributions (Categorical, DiagGaussian)
│   │   ├── optimizers.py             # MLX optimizer adapters (AdamAdapter, SGDAdapter)
│   │   ├── policies.py               # Base policy classes (ActorCriticPolicy, etc.)
│   │   ├── preprocessing.py          # Input preprocessing utilities
│   │   ├── schedules.py              # Learning rate and hyperparameter schedules
│   │   ├── torch_layers.py           # MLX neural network layers
│   │   ├── type_aliases.py           # Type definitions for arrays, schedules
│   │   ├── utils.py                  # General utilities
│   │   └── vec_env/                  # Vectorized environment support
│   │       ├── __init__.py
│   │       ├── base_vec_env.py       # VecEnv base class
│   │       └── dummy_vec_env.py      # DummyVecEnv implementation
│   ├── ppo/                          # PPO algorithm (✅ IMPLEMENTED)
│   │   ├── __init__.py               # Exports PPO and policy aliases
│   │   ├── policies.py               # PPO-specific policies (MlpPolicy, CnnPolicy)
│   │   └── ppo.py                    # PPO algorithm implementation
│   ├── a2c/                          # A2C algorithm (☐ TODO)
│   │   └── __init__.py               # Placeholder
│   ├── dqn/                          # DQN algorithm (☐ TODO)
│   │   └── __init__.py               # Placeholder
│   ├── sac/                          # SAC algorithm (☐ TODO)
│   │   └── __init__.py               # Placeholder
│   └── td3/                          # TD3 algorithm (☐ TODO)
│       └── __init__.py               # Placeholder
├── tests/                            # Test suite
│   ├── test_buffers.py               # Buffer functionality tests
│   ├── test_distributions.py         # Distribution math tests
│   ├── test_imports.py               # Import compatibility tests
│   ├── test_optimizers.py            # Optimizer adapter tests
│   ├── test_parameter_registry.py    # Parameter registry and state_dict tests
│   ├── test_policies.py              # Policy tests
│   ├── test_ppo.py                   # PPO algorithm tests
│   ├── test_ppo_optimizer_integration.py # PPO + optimizer integration tests
│   ├── test_preprocessing.py         # Preprocessing tests
│   ├── test_save_load_api_parity.py  # Save/load API parity tests (env_id, optimizer state)
│   ├── test_save_load_roundtrip.py   # Save/load round-trip tests
│   ├── test_torch_layers.py          # Neural network layer tests
│   └── test_vec_env.py               # VecEnv tests
├── examples/                         # Usage examples (empty - TODO)
├── notes/                            # Development documentation
│   ├── initial_plan.md               # Original project plan
│   ├── mlx-baselines3_spec.md        # Detailed technical specification
│   └── phase4_bugs.md                # Known issues and fixes
├── pyproject.toml                    # Project configuration and dependencies
├── README.md                         # Project documentation
├── AGENT.md                          # This file - development guide
└── uv.lock                           # Dependency lock file
```

## Implementation Status
- **✅ PPO**: Fully implemented with MLP/CNN policies, training loop, save/load
- **✅ Common Infrastructure**: Base classes, buffers, distributions, policies, VecEnv
- **✅ Optimizer System**: AdamAdapter, SGDAdapter, gradient clipping, LR schedules
- **✅ Parameter Registry**: Complete state_dict/load_state_dict system with validation
- **✅ Save/Load API Parity**: env_id persistence, optimizer state, backward compatibility
- **☐ A2C/DQN/SAC/TD3**: Placeholder classes that raise NotImplementedError
- **☐ ReplayBuffer**: Needed for off-policy algorithms (DQN/SAC/TD3)
- **☐ Examples**: No example scripts yet
- **☐ Advanced Features**: VecNormalize, callbacks need completion
