# Contributing to Levi

## Dev Setup

```bash
git clone https://github.com/ttanv/levi.git
cd algoforge
uv sync --extra dev
pre-commit install
```

## Running Tests

```bash
pytest
```

## Linting and Formatting

```bash
ruff check levi/ tests/       # lint
ruff format --check levi/ tests/  # check formatting
ruff format levi/ tests/       # auto-format
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure tests pass (`pytest`) and linting is clean (`ruff check`, `ruff format --check`)
4. Open a PR against `main`

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) with default settings. Key conventions:

- Line length: 120
- Target: Python 3.11+
- Import sorting follows isort conventions
