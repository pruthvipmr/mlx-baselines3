# MLX Baselines3 Agent Guide

## Build/Test Commands
- **Install dependencies**: `uv sync`
- **Test**: `uv run pytest`
- **Single test**: `uv run pytest tests/test_<module>.py::<test_function>`
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check mlx_baselines3/ tests/`
- **Type check**: `uv run mypy mlx_baselines3/`
- **Install dev deps**: `uv sync --extra dev`

## Package Management
- Uses `uv` (not pip/conda)
- Dependencies in `pyproject.toml`
- Lock file: `uv.lock`

## Architecture & Structure
- **Goal**: Drop-in replacement for Stable Baselines 3 using Apple's MLX framework
- **Core modules**: `mlx_baselines3/{ppo,sac,a2c,td3,dqn}/` - individual RL algorithms
- **Common infrastructure**: `mlx_baselines3/common/` - shared base classes, policies, buffers
- **Key components**: BaseAlgorithm, OnPolicy/OffPolicy classes, RolloutBuffer, ReplayBuffer
- **MLX integration**: Replace PyTorch tensors with `mx.array()`, use `mlx.nn` modules, `mlx.optimizers`

## Code Style & Conventions

### Prime Directives
- **Clarity over cleverness**: Do not code-golf. Prefer simple, readable implementations
- **Small, surgical diffs**: Break big changes into a series of obviouswins. Avoid sprawling refactors.
- **Tests with every change**: Bug fixes come with a regression test; new features come with non-brittle tests.
- Don't churn docs/whitespace**: No cosmetic or doc-only edits unless explicitly requested

### Writing Code
- **Keep it small & explicit**: Short functions; return early; no hidden side-effects
- **No magic imports**: Absolute imports only; no `from x import *`
- **Limit dependencies**: Use the standard library unless a dependency removes substantial complexity
- **Comments explain "why"**: Avoid narrating the obvious "what"
- **Error messages are actionable**: Tell the user what went wrong and how to fix it

### Formatting
- **Type hints are pragmatic**: Prioritize public APIs and tricky internals; don't block progress on typing every corner
- **Pre-commit always passes locally**: Run linter, mypy, and a fast test subset before proposing a change
- **Format/Lint with Ruff**: `uv run ruff format .` and `uv run ruff check mlx_baselines3/ tests/` (add `--fix` for autofixes)
- **NO EMOJIS**: Never use any emojis

### Testing Policy
- **Every change includes tests**.
  -Bug fix → regression test that fails before, passes after.
  - Feature → unit tests + shape/dtype edge cases.
- **Avoid brittleness**. No tests that depend on timing, randomness (without seeding), or device-specific quirks unless the feature is device-specific.
- Property/fuzz tests welcome when invariants matter (e.g., algebraic rewrites, shape rules).
- Refactors with “no behavior change” should demonstrate equivalence (e.g., replay/process-comparison or identical outputs on a golden set).

### Change-scoping rules
- **Touch as little as possible**: Minimize blast radius
- **Keep "core" clean"**: Don't churn peripheral or poorly-tested areas without necessity
- **Split PRs logically**: Land enabling refactors first; then the 3-line feature that becomes obvious after

### Commit/PR hygiene
- **Title**: concise, imperative ("Fuse X into Y for simpler kernel schedule")
- **Body**: problem -> approach -> why it's simpler -> test/bench evidence -> risk & rollout plan
- **Scope**: one theme per PR; follow-ups for anything orthogonal

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
│   ├── ppo/                          # PPO algorithm
│   │   ├── __init__.py               # Exports PPO and policy aliases
│   │   ├── policies.py               # PPO-specific policies (MlpPolicy, CnnPolicy)
│   │   ├── ppo.py                    # PPO algorithm implementation
│   │   └── optimized_ppo.py          # Performance-optimized PPO with JIT and float32 enforcement
│   ├── a2c/                          # A2C algorithm
│   │   ├── __init__.py               # Exports A2C and policy aliases
│   │   ├── policies.py               # A2C-specific policies (MlpPolicy, CnnPolicy)
│   │   └── a2c.py                    # A2C algorithm implementation
│   ├── dqn/                          # DQN algorithm
│   │   ├── __init__.py               # Exports DQN and policy aliases
│   │   ├── policies.py               # DQN-specific policies (MlpPolicy, CnnPolicy)
│   │   └── dqn.py                    # DQN algorithm implementation
│   ├── sac/                          # SAC algorithm
│   │   ├── __init__.py               # Exports SAC and policy aliases
│   │   ├── policies.py               # SAC-specific policies (MlpPolicy, CnnPolicy, MultiInputPolicy)
│   │   └── sac.py                    # SAC algorithm implementation
│   └── td3/                          # TD3 algorithm
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
│   ├── train_cartpole_a2c.py         # A2C training example with CartPole-v1
│   ├── train_cartpole_dqn.py         # DQN training example with CartPole-v1
│   ├── train_pendulum_sac.py         # SAC training example with Pendulum-v1
│   └── train_pendulum_td3.py         # TD3 training example with Pendulum-v1
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
