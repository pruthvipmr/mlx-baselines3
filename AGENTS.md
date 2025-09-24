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
├── mlx_baselines3/        # Library source: algorithms and shared infrastructure
│   ├── common/            # Utilities, policies, optimizers, buffers
│   ├── ppo/
│   ├── a2c/
│   ├── sac/
│   ├── td3/
│   └── dqn/
├── tests/                 # Regression + performance suite
├── examples/              # Minimal training scripts
├── pyproject.toml         # Project configuration (uv-managed)
├── uv.lock                # Resolved dependencies
├── README.md              # Project overview
└── AGENTS.md              # This guide
```
