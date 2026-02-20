# Advanced Type System

## Generics (Python 3.12+)

### New Syntax with `type` and `[T]`

```python
# Python 3.12+ generic syntax (preferred)
def first[T](items: list[T]) -> T:
    return items[0]

class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# Bounded generics
from collections.abc import Hashable

def deduplicate[T: Hashable](items: list[T]) -> list[T]:
    seen: set[T] = set()
    return [x for x in items if not (x in seen or seen.add(x))]
```

### Legacy TypeVar (Pre-3.12)

```python
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)  # upper bound
N = TypeVar("N", int, float)       # constrained to specific types

def first(items: list[T]) -> T:
    return items[0]
```

## ParamSpec (Preserving Callable Signatures)

```python
from typing import ParamSpec, TypeVar
from collections.abc import Callable
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def logged(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def add(a: int, b: int) -> int:
    return a + b

# Type checker knows: add(a: int, b: int) -> int
```

## TypeVarTuple (Variadic Generics, 3.11+)

```python
from typing import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")

def first_element[*Ts](tup: tuple[*Ts]) -> ???:
    ...

# Primarily useful for tensor/array shapes and variadic tuple operations
class Array[*Shape]:
    def reshape[*NewShape](self, *shape: Unpack[NewShape]) -> "Array[*NewShape]":
        ...
```

## TypeGuard and TypeIs

```python
from typing import TypeGuard, TypeIs

# TypeGuard narrows type in the True branch only
def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

items: list[object] = ["a", "b", "c"]
if is_string_list(items):
    # items is list[str] here
    print(items[0].upper())

# TypeIs (3.13+) narrows in BOTH branches
def is_int(val: int | str) -> TypeIs[int]:
    return isinstance(val, int)

x: int | str = get_value()
if is_int(x):
    print(x + 1)      # x is int
else:
    print(x.upper())   # x is str
```

## @overload

```python
from typing import overload

@overload
def process(data: str) -> str: ...
@overload
def process(data: bytes) -> bytes: ...
@overload
def process(data: int) -> float: ...

def process(data: str | bytes | int) -> str | bytes | float:
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, bytes):
        return data.upper()
    else:
        return float(data)
```

Use `@overload` when: return type depends on input type, and you want the type checker to track this precisely.

## Self Type (3.11+)

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self  # correctly typed as the subclass type

class AdvancedBuilder(Builder):
    def set_extra(self, extra: str) -> Self:
        self.extra = extra
        return self

# AdvancedBuilder().set_name("x") returns AdvancedBuilder, not Builder
```

## Final and Literal

```python
from typing import Final, Literal

# Final prevents reassignment
MAX_RETRIES: Final = 3
API_VERSION: Final[str] = "v2"

# Literal restricts to specific values
def set_mode(mode: Literal["read", "write", "append"]) -> None: ...

# Combine for discriminated unions
from dataclasses import dataclass

@dataclass
class Success:
    kind: Literal["success"] = "success"
    value: str = ""

@dataclass
class Failure:
    kind: Literal["failure"] = "failure"
    error: str = ""

type Result = Success | Failure

def handle(result: Result) -> None:
    match result:
        case Success(value=v):
            print(f"OK: {v}")
        case Failure(error=e):
            print(f"Error: {e}")
```

## Protocols with Methods and Properties

```python
from typing import Protocol

class Sized(Protocol):
    def __len__(self) -> int: ...

class Named(Protocol):
    @property
    def name(self) -> str: ...

class Describable(Sized, Named, Protocol):
    """Combines multiple protocols."""
    def describe(self) -> str: ...
```

## Annotating Descriptors, ClassVar, InitVar

```python
from typing import ClassVar
from dataclasses import dataclass, InitVar

@dataclass
class DatabaseConfig:
    # ClassVar is excluded from __init__ and comparisons
    connection_pool: ClassVar[int] = 10

    host: str
    port: int
    password: InitVar[str]  # passed to __init__ but not stored as field

    def __post_init__(self, password: str) -> None:
        self._connection_string = f"postgresql://{self.host}:{self.port}?password={password}"
```

## Mypy Configuration Best Practices

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true                    # enable all strict checks
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true     # all functions must have type annotations
disallow_any_generics = true     # no bare list, dict (require list[int], etc.)
check_untyped_defs = true
no_implicit_optional = true      # def f(x: str = None) is an error, use str | None
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false    # relax for tests

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true
```

## Common Type Annotation Patterns

```python
from collections.abc import AsyncIterator, Mapping
from typing import Any, Never, NoReturn, TypedDict

# TypedDict for structured dictionaries
class UserDict(TypedDict):
    name: str
    age: int
    email: str | None

# NotRequired fields (3.11+)
from typing import NotRequired

class ApiResponse(TypedDict):
    data: list[dict[str, Any]]
    error: NotRequired[str]
    cursor: NotRequired[str]

# Never for functions that never return normally
def assert_never(value: Never) -> NoReturn:
    raise AssertionError(f"Unexpected value: {value}")

# Use in exhaustiveness checking
match status:
    case Status.ACTIVE:
        ...
    case Status.INACTIVE:
        ...
    case _ as unreachable:
        assert_never(unreachable)  # mypy error if a case is missing
```
