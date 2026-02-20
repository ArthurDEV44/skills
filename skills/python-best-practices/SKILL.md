---
name: python-best-practices
description: >
  Idiomatic Python patterns, modern type hints, and best practices for writing clean, performant,
  and maintainable Python code (3.10+). Covers type annotations (generics, Protocol, TypeAlias,
  TypeGuard, overload, ParamSpec, TypeVarTuple), dataclasses and attrs, error handling (exception
  hierarchy, exception groups, custom exceptions), iterators and generators (yield, yield from,
  generator expressions, itertools), context managers (contextlib, async context managers),
  decorators (functools.wraps, decorator factories, class decorators), enum design (StrEnum,
  IntEnum, Flag), string handling (f-strings, Template, textwrap), pathlib, structural pattern
  matching (match/case), slots, descriptors, ABC and Protocol, packaging (pyproject.toml, uv, ruff),
  common anti-patterns, and Pythonic idioms. Complements specialized Python skills for frameworks
  (Django, FastAPI, Flask). Use when writing, reviewing, or refactoring Python code: (1) Type
  annotations and mypy compliance, (2) Dataclass and model design, (3) Error handling strategies,
  (4) Iterator and generator patterns, (5) Context manager usage, (6) Decorator implementation,
  (7) Enum and constant design, (8) Structural pattern matching, (9) Project structure and
  packaging, (10) Testing with pytest, (11) Async/await patterns, (12) Performance optimization,
  (13) Common anti-patterns and Pythonic rewrites.
---

# Python Best Practices

## Type Annotations (Python 3.10+)

### Modern Syntax

```python
# Use built-in generics (3.10+) -- no need to import from typing
def process(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Union with | (3.10+)
def find(key: str) -> str | None:
    ...

# Tuple types
pair: tuple[int, str] = (1, "hello")
variable_len: tuple[int, ...] = (1, 2, 3)
```

### Function Signatures

```python
from collections.abc import Callable, Iterable, Sequence

# Prefer Sequence over list for read-only parameters (accepts list, tuple, etc.)
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)

# Prefer Iterable for iteration-only parameters
def print_all(items: Iterable[str]) -> None:
    for item in items:
        print(item)

# Callable with signature
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# *args and **kwargs typing
def log(*args: str, **kwargs: int) -> None: ...
```

### TypeAlias and type Statement

```python
# Python 3.12+ type statement (preferred)
type UserId = int
type Matrix[T] = list[list[T]]
type Handler = Callable[[Request], Response]

# Pre-3.12 TypeAlias
from typing import TypeAlias
UserId: TypeAlias = int
```

### Protocol (Structural Subtyping)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Renderable(Protocol):
    def render(self) -> str: ...

# Any class with a render() -> str method satisfies Renderable
# No inheritance required -- duck typing with type safety
class Button:
    def render(self) -> str:
        return "<button>Click</button>"

def display(widget: Renderable) -> None:
    print(widget.render())

display(Button())  # OK -- Button satisfies Renderable structurally
```

**When to use Protocol vs ABC:** Use Protocol when you want structural (duck) typing -- no inheritance required. Use ABC when you want nominal typing with enforced implementation.

For advanced type system patterns (generics, TypeVar, ParamSpec, TypeGuard, overload, TypeVarTuple, Self), see `references/type-system.md`.

## Dataclasses

### Basic Usage

```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# frozen=True -> immutable (hashable, safe as dict key)
# slots=True -> __slots__ generated (less memory, faster attribute access)
```

### Defaults and Field Factories

```python
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    tags: list[str] = field(default_factory=list)  # mutable defaults need factory
    _cache: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
```

### Post-Init Processing

```python
@dataclass
class User:
    name: str
    email: str

    def __post_init__(self) -> None:
        self.email = self.email.lower().strip()
        if "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
```

### When to Use What

| Need | Use |
|------|-----|
| Simple data container | `@dataclass` |
| Immutable value object | `@dataclass(frozen=True, slots=True)` |
| Complex validation | `pydantic.BaseModel` or `attrs` with validators |
| Lightweight tuple replacement | `NamedTuple` |
| Quick grouping (no methods) | `NamedTuple` |

## Error Handling

### Exception Hierarchy

```python
# Define a base exception for your package
class AppError(Exception):
    """Base exception for the application."""

class NotFoundError(AppError):
    """Resource was not found."""
    def __init__(self, resource: str, id: str) -> None:
        self.resource = resource
        self.id = id
        super().__init__(f"{resource} not found: {id}")

class ValidationError(AppError):
    """Input validation failed."""
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"Validation error on {field}: {message}")
```

### Best Practices

```python
# GOOD: Catch specific exceptions
try:
    user = get_user(user_id)
except NotFoundError:
    return None
except DatabaseError as e:
    logger.error("DB failure: %s", e)
    raise

# BAD: Bare except or catching Exception broadly
try:
    do_something()
except:          # catches SystemExit, KeyboardInterrupt too!
    pass

# GOOD: Re-raise with context using `from`
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    raise ValidationError("body", "Invalid JSON") from e

# GOOD: Use else clause for code that should only run if no exception
try:
    f = open(path)
except OSError as e:
    handle_error(e)
else:
    with f:
        return f.read()
```

### Exception Groups (Python 3.11+)

```python
# Aggregate multiple errors from parallel operations
errors: list[Exception] = []
for task in tasks:
    try:
        task.run()
    except Exception as e:
        errors.append(e)
if errors:
    raise ExceptionGroup("Multiple task failures", errors)

# Handle with except*
try:
    run_all(tasks)
except* ValueError as eg:
    for e in eg.exceptions:
        log_validation(e)
except* OSError as eg:
    for e in eg.exceptions:
        log_io_error(e)
```

## Iterators & Generators

### Generator Functions

```python
from collections.abc import Iterator, Generator

# Simple generator -- yields values lazily
def fibonacci() -> Iterator[int]:
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Generator with send/return (rare -- prefer simple generators)
def accumulator() -> Generator[float, float, float]:
    total = 0.0
    while True:
        value = yield total
        if value is None:
            return total
        total += value
```

### Generator Expressions

```python
# Prefer generator expressions over list comprehensions for large data
total = sum(x * x for x in range(1_000_000))  # no intermediate list

# Use list comp only when you need the full list
squares = [x * x for x in range(10)]

# Nested comprehensions -- keep readable (max 2 levels)
flat = [cell for row in matrix for cell in row]
```

### itertools Patterns

```python
from itertools import chain, islice, groupby, batched, pairwise

# Chain multiple iterables
all_items = chain(list_a, list_b, list_c)

# Batch processing (3.12+)
for batch in batched(items, 100):
    process_batch(batch)

# Pairwise iteration (3.10+)
for prev, curr in pairwise(values):
    print(f"{prev} -> {curr}")

# Groupby (requires sorted input)
for key, group in groupby(sorted(users, key=lambda u: u.role), key=lambda u: u.role):
    print(f"{key}: {list(group)}")

# yield from for delegating to sub-generators
def flatten(nested: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in nested:
        yield from inner
```

## Context Managers

### The `with` Statement

```python
# File handling -- always use `with`
with open("data.json") as f:
    data = json.load(f)

# Multiple context managers (3.10+ parenthesized form)
with (
    open("input.txt") as src,
    open("output.txt", "w") as dst,
):
    dst.write(src.read())
```

### Custom Context Managers

```python
from contextlib import contextmanager

@contextmanager
def temporary_env(**kwargs: str) -> Iterator[None]:
    """Temporarily set environment variables."""
    old = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

with temporary_env(DEBUG="1", LOG_LEVEL="debug"):
    run_app()
```

### Class-Based Context Manager

```python
class DatabaseConnection:
    def __init__(self, url: str) -> None:
        self.url = url
        self.conn = None

    def __enter__(self) -> "DatabaseConnection":
        self.conn = create_connection(self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.conn:
            self.conn.close()
        return False  # don't suppress exceptions
```

## Decorators

### Preserving Signatures with functools.wraps

```python
from functools import wraps
import time

def timer(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)  # preserves __name__, __doc__, __annotations__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper
```

### Decorator Factories (with arguments)

```python
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator

@retry(max_attempts=5, delay=0.5)
def fetch_data(url: str) -> dict: ...
```

### Class Decorators

```python
def singleton(cls):
    """Ensure a class has only one instance."""
    instances: dict[type, object] = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
```

## Enum Design

### Modern Enums (Python 3.11+)

```python
from enum import Enum, StrEnum, IntEnum, Flag, auto

class Color(StrEnum):
    RED = auto()     # "red"
    GREEN = auto()   # "green"
    BLUE = auto()    # "blue"

# StrEnum values are strings -- great for APIs and serialization
print(Color.RED == "red")  # True

class Permission(Flag):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    ADMIN = READ | WRITE | EXECUTE

# Flag supports bitwise operations
perms = Permission.READ | Permission.WRITE
assert Permission.READ in perms
```

### Enum Best Practices

```python
# GOOD: Use enum instead of string constants
class Status(StrEnum):
    PENDING = auto()
    ACTIVE = auto()
    CANCELLED = auto()

# BAD: Magic strings scattered through code
status = "pending"  # typo-prone, no IDE completion

# Add methods to enums
class HttpMethod(StrEnum):
    GET = auto()
    POST = auto()
    PUT = auto()
    DELETE = auto()

    @property
    def is_safe(self) -> bool:
        return self in (HttpMethod.GET,)

    @property
    def is_idempotent(self) -> bool:
        return self in (HttpMethod.GET, HttpMethod.PUT, HttpMethod.DELETE)
```

## Structural Pattern Matching (3.10+)

```python
match command:
    case {"action": "move", "direction": str() as dir}:
        move(dir)
    case {"action": "attack", "target": str() as target, "weapon": str() as w}:
        attack(target, w)
    case {"action": "quit"}:
        quit_game()
    case _:
        print("Unknown command")

# Match with classes
match event:
    case Click(position=Point(x, y)) if x > 100:
        handle_right_click(x, y)
    case KeyPress(key="q"):
        quit()
    case KeyPress(key=str() as k):
        handle_key(k)
```

## Common Anti-Patterns

### Mutable Default Arguments

```python
# BAD: Mutable default is shared across calls
def append_to(item, target=[]):
    target.append(item)
    return target

# GOOD: Use None sentinel
def append_to(item, target: list | None = None) -> list:
    if target is None:
        target = []
    target.append(item)
    return target
```

### Bare `except` and Broad `except Exception`

```python
# BAD: Silences all errors including KeyboardInterrupt
try:
    do_work()
except:
    pass

# BAD: Catches too broadly, hides bugs
try:
    process(data)
except Exception:
    logger.error("Something went wrong")

# GOOD: Catch specific exceptions you can handle
try:
    process(data)
except ValueError as e:
    logger.warning("Invalid data: %s", e)
except ConnectionError as e:
    logger.error("Network failure: %s", e)
    raise
```

### Not Using Context Managers

```python
# BAD: Resource leak if exception occurs
f = open("file.txt")
data = f.read()
f.close()

# GOOD: Guaranteed cleanup
with open("file.txt") as f:
    data = f.read()
```

### Checking Type Instead of Using Protocols

```python
# BAD: Explicit type checking
if isinstance(obj, dict):
    ...

# GOOD: Use duck typing or Protocol
# Just use obj as if it has the interface you need
# Or define a Protocol if you want static checking
```

### Using `is` for Value Comparison

```python
# BAD: `is` checks identity, not equality
if x is 1:  # unreliable, works only for small ints
    ...

# GOOD: `==` checks equality
if x == 1:
    ...

# `is` is correct for: None, True, False, sentinel objects
if result is None:
    ...
```

## Pythonic Idioms

### EAFP over LBYL

```python
# LBYL (Look Before You Leap) -- not Pythonic
if key in dictionary:
    value = dictionary[key]
else:
    value = default

# EAFP (Easier to Ask Forgiveness than Permission) -- Pythonic
try:
    value = dictionary[key]
except KeyError:
    value = default

# Best: use dict.get()
value = dictionary.get(key, default)
```

### Unpacking and Swapping

```python
# Tuple unpacking
first, *rest = [1, 2, 3, 4]  # first=1, rest=[2, 3, 4]

# Variable swap
a, b = b, a

# Ignore values with _
_, important, _ = get_triple()
```

### Truthiness

```python
# GOOD: Use truthiness for emptiness checks
if items:          # instead of if len(items) > 0
    process(items)

if not name:       # instead of if name == ""
    name = "anonymous"

# But be explicit when 0 or False are valid values
if count is not None:  # not `if count:` (0 is falsy but valid)
    use_count(count)
```

### Walrus Operator (3.8+)

```python
# Assign and test in one expression
if (m := pattern.match(line)) is not None:
    process(m.group(1))

# In while loops
while chunk := f.read(8192):
    process(chunk)

# In comprehensions
results = [clean for raw in data if (clean := transform(raw)) is not None]
```

For async/await patterns (asyncio, aiohttp, structured concurrency, TaskGroup), see `references/async-patterns.md`.

For testing best practices (pytest, fixtures, parametrize, mocking, coverage), see `references/testing.md`.

For project structure and packaging (pyproject.toml, uv, ruff, src layout), see `references/project-structure.md`.

For performance patterns (profiling, caching, slots, __slots__, comprehension speed), see `references/performance.md`.

For advanced type system (generics, TypeVar, ParamSpec, overload, TypeGuard), see `references/type-system.md`.
