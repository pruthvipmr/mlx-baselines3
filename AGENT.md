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
│   │   ├── callbacks.py              # Callback system (BaseCallback, EvalCallback, etc.)
│   │   ├── distributions.py          # Action distributions (Categorical, DiagGaussian)
│   │   ├── functional_losses.py      # Pure functional loss computations for performance
│   │   ├── jit_optimizations.py      # JIT compilation optimizations (18% speedup)
│   │   ├── logger.py                 # Logging system (Logger, multiple output formats)
│   │   ├── optimized_training.py     # Optimized training loops with minimal parameter reloads
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
│   │       ├── dummy_vec_env.py      # DummyVecEnv implementation, make_vec_env utility
│   │       └── vec_normalize.py      # VecNormalize wrapper with running mean/std
│   ├── ppo/                          # PPO algorithm (✅ IMPLEMENTED)
│   │   ├── __init__.py               # Exports PPO and policy aliases
│   │   ├── policies.py               # PPO-specific policies (MlpPolicy, CnnPolicy)
│   │   ├── ppo.py                    # PPO algorithm implementation
│   │   └── optimized_ppo.py          # Performance-optimized PPO with JIT and float32 enforcement
│   ├── a2c/                          # A2C algorithm (✅ IMPLEMENTED)
│   │   ├── __init__.py               # Exports A2C and policy aliases
│   │   ├── policies.py               # A2C-specific policies (MlpPolicy, CnnPolicy)
│   │   └── a2c.py                    # A2C algorithm implementation
│   ├── dqn/                          # DQN algorithm (✅ IMPLEMENTED)
│   │   ├── __init__.py               # Exports DQN and policy aliases
│   │   ├── policies.py               # DQN-specific policies (MlpPolicy, CnnPolicy)
│   │   └── dqn.py                    # DQN algorithm implementation
│   ├── sac/                          # SAC algorithm (✅ IMPLEMENTED)
│   │   ├── __init__.py               # Exports SAC and policy aliases
│   │   ├── policies.py               # SAC-specific policies (MlpPolicy, CnnPolicy, MultiInputPolicy)
│   │   └── sac.py                    # SAC algorithm implementation
│   └── td3/                          # TD3 algorithm (✅ IMPLEMENTED)
│       ├── __init__.py               # Exports TD3 and policy aliases
│       ├── policies.py               # TD3-specific policies (TD3Policy, MlpPolicy, CnnPolicy)
│       └── td3.py                    # TD3 algorithm implementation
├── tests/                            # Test suite (429 tests total)
│   ├── test_buffers.py               # Buffer functionality tests
│   ├── test_buffer_performance.py    # Buffer performance benchmarks
│   ├── test_callbacks.py             # Callback system tests
│   ├── test_distributions.py         # Distribution math tests
│   ├── test_final_performance.py     # Final baseline vs optimized performance comparison
│   ├── test_gradient_stability.py    # Numerical stability and dtype consistency tests
│   ├── test_imports.py               # Import compatibility tests
│   ├── test_integration.py           # End-to-end integration tests (2k-5k timesteps)
│   ├── test_logger.py                # Logging system tests
│   ├── test_optimized_performance.py # Performance benchmarks for optimization components
│   ├── test_optimizers.py            # Optimizer adapter tests
│   ├── test_parameter_registry.py    # Parameter registry and state_dict tests
│   ├── test_policies.py              # Policy tests
│   ├── test_ppo.py                   # PPO algorithm tests
│   ├── test_ppo_optimizer_integration.py # PPO + optimizer integration tests
│   ├── test_a2c.py                   # A2C algorithm tests
│   ├── test_dqn.py                   # DQN algorithm tests
│   ├── test_sac.py                   # SAC algorithm tests
│   ├── test_td3.py                   # TD3 algorithm tests
│   ├── test_reproducibility.py       # Seeding and reproducibility tests
│   ├── test_schedules.py             # Schedule functionality tests
│   ├── test_preprocessing.py         # Preprocessing tests
│   ├── test_save_load_api_parity.py  # Save/load API parity tests (env_id, optimizer state)
│   ├── test_save_load_roundtrip.py   # Save/load round-trip tests
│   ├── test_torch_layers.py          # Neural network layer tests
│   ├── test_vec_env.py               # VecEnv tests
│   └── test_vec_normalize.py         # VecNormalize tests and env type assertions
├── examples/                         # Usage examples
│   ├── train_cartpole_ppo.py         # PPO training example with CartPole-v1
│   ├── train_cartpole_dqn.py         # DQN training example with CartPole-v1  
│   └── train_pendulum_sac.py         # SAC training example with Pendulum-v1
├── notes/                            # Development documentation
│   ├── initial_plan.md               # Original project plan
│   ├── mlx-baselines3_spec.md        # Detailed technical specification
│   └── phase4_bugs.md                # Known issues and fixes
├── .github/                          # GitHub Actions CI/CD
│   └── workflows/
│       └── test.yml                  # CI workflow (macOS, Python 3.10/3.11)
├── pyproject.toml                    # Project configuration and dependencies
├── README.md                         # Project documentation
├── AGENT.md                          # This file - development guide
└── uv.lock                           # Dependency lock file
```

## Implementation Status
- **✅ PPO**: Fully implemented with MLP/CNN policies, training loop, save/load
- **✅ A2C**: Fully implemented with training, policies, RMSProp/Adam optimizers (save/load has known issue)
- **✅ Common Infrastructure**: Base classes, buffers, distributions, policies, VecEnv
- **✅ Optimizer System**: AdamAdapter, SGDAdapter, RMSPropAdapter, gradient clipping, LR schedules
- **✅ Parameter Registry**: Complete state_dict/load_state_dict system with validation
- **✅ Save/Load API Parity**: env_id persistence, optimizer state, policy_state, serializable lr/lr_schedule, backward compatibility
- **✅ Action Distributions**: CategoricalDistribution, DiagGaussianDistribution, MultiCategoricalDistribution, BernoulliDistribution with action clipping
- **✅ Buffer System**: RolloutBuffer and ReplayBuffer with SB3 compatibility, >3.6M samples/s throughput
- **✅ DQN**: Fully implemented with Q-networks, epsilon-greedy exploration, target networks, Huber loss
- **✅ SAC**: Fully implemented with stochastic actor, twin critics, automatic entropy tuning, target networks, and a working off-policy learn() loop
- **✅ TD3**: Fully implemented with deterministic actor, twin critics, delayed policy updates, target policy smoothing
- **✅ VecNormalize**: Complete observation/reward normalization wrapper with save/load support
- **✅ Callbacks & Logging**: Complete callback system with BaseCallback, EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold, ProgressBarCallback; Multi-format logging (stdout, CSV, TensorBoard)
- **✅ Schedules & Hyperparams**: Complete schedule system (constant, linear, piecewise, exponential, cosine) with string parsing, SB3 compatibility, and PPO integration for lr/clip_range/ent_coef; target_kl early stopping with proper epoch counting
- **✅ Performance Optimizations**: JIT compilation framework with 18% improvement on core operations, optimized PPO implementation with float32 enforcement, functional loss computations, and comprehensive performance testing suite
- **✅ Testing & CI**: Complete test suite with 429 tests, GitHub Actions CI workflow, reproducibility tests, integration tests, and comprehensive coverage of all algorithms and infrastructure
- **✅ Examples**: Complete example scripts with PPO/DQN/SAC training, evaluation, and command-line interfaces
