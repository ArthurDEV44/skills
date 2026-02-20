# Performance Patterns

## Profiling First

```bash
# Time-based profiling
python -m cProfile -s cumulative my_script.py

# Line-by-line profiling (install: pip install line_profiler)
# Decorate function with @profile, then:
kernprof -l -v my_script.py

# Memory profiling (install: pip install memory_profiler)
python -m memory_profiler my_script.py

# py-spy for sampling profiler (no code changes needed)
py-spy record -o profile.svg -- python my_script.py
```

**Rule:** Always profile before optimizing. Optimize the hot path, not guesses.

## Data Structure Selection

| Operation | list | deque | set | dict |
|-----------|------|-------|-----|------|
| Append end | O(1)* | O(1) | -- | -- |
| Append start | O(n) | O(1) | -- | -- |
| Lookup by index | O(1) | O(n) | -- | -- |
| Membership test | O(n) | O(n) | O(1) | O(1) |
| Insert middle | O(n) | O(n) | -- | -- |
| Delete by value | O(n) | O(n) | O(1) | O(1) |

```python
from collections import deque

# Use deque for queue/stack operations
queue: deque[str] = deque(maxlen=1000)
queue.append("item")      # right end
queue.appendleft("item")  # left end
item = queue.popleft()     # O(1) vs list.pop(0) which is O(n)

# Use set for membership testing
valid_ids: set[int] = {1, 2, 3, 4, 5}
if user_id in valid_ids:  # O(1) vs O(n) with list
    ...

# Use dict.get() with default instead of try/except for missing keys
value = cache.get(key, compute_default())
```

## __slots__ for Memory Reduction

```python
# Without __slots__: each instance has a __dict__ (~200+ bytes overhead)
class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

# With __slots__: no __dict__, significant memory savings for many instances
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

# Or use @dataclass(slots=True) for automatic __slots__
from dataclasses import dataclass

@dataclass(slots=True)
class Point:
    x: float
    y: float
```

**When to use:** Classes with many instances (data models, tree nodes, graph vertices). Saves ~40-60% memory per instance.

## Caching and Memoization

```python
from functools import lru_cache, cache

# @cache -- unbounded cache (Python 3.9+)
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# @lru_cache -- bounded cache with LRU eviction
@lru_cache(maxsize=256)
def expensive_computation(key: str) -> dict:
    return load_and_process(key)

# Check cache stats
print(expensive_computation.cache_info())
# CacheInfo(hits=100, misses=10, maxsize=256, currsize=10)

# Clear cache
expensive_computation.cache_clear()
```

### Caching with Unhashable Arguments

```python
# lru_cache requires hashable arguments
# For dict/list args, convert to hashable form

def process(config: dict) -> Result:
    return _process_cached(tuple(sorted(config.items())))

@lru_cache(maxsize=128)
def _process_cached(config_tuple: tuple) -> Result:
    config = dict(config_tuple)
    ...
```

## String Performance

```python
# BAD: String concatenation in a loop -- O(n^2)
result = ""
for item in items:
    result += str(item)

# GOOD: str.join -- O(n)
result = "".join(str(item) for item in items)

# GOOD: io.StringIO for complex string building
from io import StringIO
buffer = StringIO()
for item in items:
    buffer.write(str(item))
    buffer.write("\n")
result = buffer.getvalue()

# f-strings are fast for simple formatting
name = f"{first} {last}"  # faster than "%s %s" % (first, last) or "{} {}".format(first, last)
```

## Generator vs List Comprehension

```python
# Generator expression -- lazy, constant memory
total = sum(x * x for x in range(10_000_000))  # ~0 memory overhead

# List comprehension -- eager, stores all results
squares = [x * x for x in range(10_000_000)]  # ~80MB memory

# Use generator when:
# - Processing large datasets
# - Only need to iterate once
# - Passing to a consuming function (sum, min, max, any, all)

# Use list when:
# - Need to index, slice, or iterate multiple times
# - Need len()
# - Data fits comfortably in memory
```

## Avoiding Unnecessary Work

```python
# Use any()/all() with short-circuit evaluation
has_admin = any(user.role == "admin" for user in users)  # stops at first match
all_active = all(user.is_active for user in users)       # stops at first False

# Use dict for O(1) lookup instead of linear search
# BAD
def find_user(users: list[User], user_id: int) -> User | None:
    for user in users:
        if user.id == user_id:
            return user
    return None

# GOOD: Pre-build a lookup dict
users_by_id: dict[int, User] = {u.id: u for u in users}
user = users_by_id.get(user_id)
```

## Local Variable Access

```python
# Local variables are faster than global or attribute lookups
# CPython optimization: LOAD_FAST vs LOAD_GLOBAL/LOAD_ATTR

# BAD in hot loops
for item in large_list:
    result = math.sqrt(item)  # attribute lookup each iteration

# GOOD: localize frequently accessed functions
sqrt = math.sqrt
for item in large_list:
    result = sqrt(item)       # local variable lookup
```

## collections for Specialized Needs

```python
from collections import Counter, defaultdict, OrderedDict

# Counter for frequency counting
word_counts = Counter(words)
most_common = word_counts.most_common(10)

# defaultdict to avoid key existence checks
groups: defaultdict[str, list[str]] = defaultdict(list)
for item in items:
    groups[item.category].append(item.name)
# No need for: if category not in groups: groups[category] = []

# namedtuple for lightweight immutable records (faster than dataclass for simple cases)
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
```

## Batch Operations

```python
# BAD: N individual database queries
for user_id in user_ids:
    user = db.query(User).get(user_id)

# GOOD: Single batch query
users = db.query(User).filter(User.id.in_(user_ids)).all()

# BAD: N individual API calls
for item in items:
    await api.create(item)

# GOOD: Batch API call or concurrent execution
await api.create_batch(items)
# Or with asyncio
async with asyncio.TaskGroup() as tg:
    for item in items:
        tg.create_task(api.create(item))
```

## Structural Pattern Matching Performance

```python
# match/case compiles to efficient dispatch (3.10+)
# More readable AND faster than if/elif chains for many cases
match event.type:
    case "click":
        handle_click(event)
    case "keypress":
        handle_key(event)
    case "scroll":
        handle_scroll(event)
```

## When NOT to Optimize

- Don't optimize code that runs once (startup, configuration loading)
- Don't micro-optimize at the cost of readability
- Don't prematurely switch to C extensions -- Python is fast enough for most I/O-bound work
- Profile first, then optimize the top 1-3 bottlenecks
- Consider algorithmic improvements (O(n log n) vs O(n^2)) before constant-factor optimizations
