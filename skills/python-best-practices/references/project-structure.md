# Project Structure & Packaging

## Recommended Layout (src layout)

```
my-project/
├── pyproject.toml          # single source of truth for project config
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── py.typed         # PEP 561 marker for type stubs
│       ├── models.py
│       ├── services.py
│       ├── cli.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_services.py
│   └── integration/
│       ├── conftest.py
│       └── test_api.py
├── .python-version          # pin Python version (used by uv, pyenv)
├── uv.lock                  # lockfile (if using uv)
└── README.md
```

**Why src layout?**
- Prevents accidentally importing from the source directory instead of installed package
- Forces you to install the package before testing (catches packaging errors early)
- Clear separation between source and project root files

## pyproject.toml (Modern Standard)

```toml
[project]
name = "my-package"
version = "1.0.0"
description = "A short description"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
authors = [
    { name = "Your Name", email = "you@example.com" },
]
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "mypy>=1.11",
    "ruff>=0.7",
    "coverage[toml]>=7.0",
]

[project.scripts]
my-cli = "mypackage.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

## uv (Modern Package Manager)

```bash
# Create a new project
uv init my-project
cd my-project

# Add dependencies
uv add httpx pydantic
uv add --dev pytest mypy ruff

# Run commands in the project environment
uv run python -m mypackage
uv run pytest
uv run mypy src/

# Sync dependencies from lockfile
uv sync

# Pin Python version
uv python pin 3.12
```

## Ruff (Linter + Formatter)

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # Ruff-specific rules
    "PTH",  # flake8-use-pathlib
    "ERA",  # eradicate (commented-out code)
    "TID",  # flake8-tidy-imports
    "PL",   # pylint rules
    "PERF", # performance anti-patterns
]
ignore = [
    "E501",   # line length (handled by formatter)
    "PLR0913", # too many arguments (sometimes necessary)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",   # allow assert in tests
    "PLR2004", # allow magic values in tests
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

```bash
# Lint and fix
ruff check --fix src/ tests/
# Format
ruff format src/ tests/
```

## Module Organization

### __init__.py Best Practices

```python
# src/mypackage/__init__.py
"""My Package -- a short description."""

# Re-export public API explicitly
from mypackage.models import User, UserCreate, UserUpdate
from mypackage.services import UserService
from mypackage.exceptions import AppError, NotFoundError

__all__ = [
    "User",
    "UserCreate",
    "UserUpdate",
    "UserService",
    "AppError",
    "NotFoundError",
]
```

### __all__ for Controlling Public API

```python
# Explicitly declare what `from module import *` exports
# Also serves as documentation of the public interface
__all__ = ["PublicClass", "public_function"]
```

### Circular Import Prevention

```python
# Strategy 1: Import at function level
def get_user() -> "User":
    from mypackage.models import User  # lazy import
    return User.query.first()

# Strategy 2: TYPE_CHECKING guard
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypackage.models import User

def process(user: User) -> None:  # string annotation at runtime
    ...

# Strategy 3: Restructure modules to break the cycle
# Move shared types to a separate module both sides can import
```

## Logging Configuration

```python
import logging

# In library code -- NEVER configure logging, just get a logger
logger = logging.getLogger(__name__)

def process(data: dict) -> None:
    logger.debug("Processing %d items", len(data))
    try:
        result = transform(data)
    except ValueError:
        logger.exception("Failed to transform data")
        raise

# In application entry point -- configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

## Environment and Configuration

```python
import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    """Load from environment with sensible defaults."""
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() in ("1", "true"))
    database_url: str = field(default_factory=lambda: os.environ["DATABASE_URL"])
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))

# For more complex needs, use pydantic-settings
```

## Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]
```
