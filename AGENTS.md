# MLX Baselines3 Agent Guide

## Essentials
- Manage dependencies with `uv`; run `uv sync` before working.
- Run tests via `uv run pytest`; target individual cases with standard pytest node IDs.
- Keep formatting clean with `uv run ruff format .`; lint with `uv run ruff check mlx_baselines3/ tests/`.
- Type-check critical paths using `uv run mypy mlx_baselines3/`.

## Development Principles
- Goal: match Stable Baselines 3 behavior on Apple's MLX; algorithms live in `mlx_baselines3/{ppo,a2c,sac,td3,dqn}/`, shared utilities in `mlx_baselines3/common/`.
- Prefer clear, explicit code and small, focused diffs.
- Ship a reliable test with every feature or bug fix; seed randomness when relevant.
- Use absolute imports and standard-library solutions unless an extra dependency meaningfully reduces complexity.
- Write actionable error messages and comments that capture intent.
- Don't use try catch blocks unless explicity asked for.

## Project Layout
```
mlx-baselines3/
├── mlx_baselines3/        # RL algorithms and shared infrastructure
├── tests/                 # Unit and regression suite
├── examples/              # Minimal training scripts
├── pyproject.toml         # uv project configuration
└── uv.lock                # Locked dependencies
```
