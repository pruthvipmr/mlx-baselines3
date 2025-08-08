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
