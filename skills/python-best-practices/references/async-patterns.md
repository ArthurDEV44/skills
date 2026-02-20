# Async/Await Patterns

## asyncio Fundamentals

### Basic Async Functions

```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main() -> None:
    data = await fetch_data("https://api.example.com/data")
    print(data)

asyncio.run(main())
```

### Concurrent Execution with gather

```python
async def fetch_all(urls: list[str]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# With error handling -- return_exceptions=True prevents one failure from cancelling all
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        logger.error("Task failed: %s", result)
    else:
        process(result)
```

## TaskGroup (Python 3.11+ -- Structured Concurrency)

```python
async def process_all(items: list[str]) -> list[Result]:
    results: list[Result] = []

    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(process_one(item, results))

    # All tasks completed (or all cancelled if any raised)
    return results

# TaskGroup advantages over gather:
# - Structured lifetime: all tasks complete before exiting the `async with` block
# - Better error handling: if one task fails, all others are cancelled
# - Exceptions are collected into an ExceptionGroup
```

### Handling TaskGroup Failures

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(risky_operation_1())
        tg.create_task(risky_operation_2())
except* ValueError as eg:
    for exc in eg.exceptions:
        logger.error("Validation error: %s", exc)
except* ConnectionError as eg:
    for exc in eg.exceptions:
        logger.error("Connection failed: %s", exc)
```

## Async Iterators and Generators

```python
from collections.abc import AsyncIterator

async def read_lines(path: str) -> AsyncIterator[str]:
    async with aiofiles.open(path) as f:
        async for line in f:
            yield line.strip()

# Async comprehension
lines = [line async for line in read_lines("data.txt") if line]

# Async for loop
async for line in read_lines("data.txt"):
    await process_line(line)
```

## Async Context Managers

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def managed_connection(url: str) -> AsyncIterator[Connection]:
    conn = await create_connection(url)
    try:
        yield conn
    finally:
        await conn.close()

async with managed_connection("postgresql://localhost/db") as conn:
    result = await conn.execute("SELECT 1")
```

## Semaphores and Rate Limiting

```python
async def fetch_with_limit(urls: list[str], max_concurrent: int = 10) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_fetch(url: str) -> str:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return await resp.text()

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(limited_fetch(url)) for url in urls]

    return [task.result() for task in tasks]
```

## Timeouts

```python
# asyncio.timeout (3.11+) -- preferred
async with asyncio.timeout(5.0):
    data = await fetch_data(url)

# asyncio.wait_for -- older API
try:
    data = await asyncio.wait_for(fetch_data(url), timeout=5.0)
except asyncio.TimeoutError:
    logger.warning("Request timed out")
```

## Queues for Producer/Consumer

```python
async def producer(queue: asyncio.Queue[str], items: list[str]) -> None:
    for item in items:
        await queue.put(item)
    await queue.put(None)  # sentinel

async def consumer(queue: asyncio.Queue[str | None]) -> None:
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
        queue.task_done()

async def main() -> None:
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(queue, items))
        for _ in range(4):  # 4 consumers
            tg.create_task(consumer(queue))
```

## Event and Condition

```python
# Event for one-time signaling
shutdown_event = asyncio.Event()

async def worker() -> None:
    while not shutdown_event.is_set():
        await asyncio.sleep(1)
        do_work()

async def shutdown() -> None:
    shutdown_event.set()  # signals all waiters
```

## Async Best Practices

- **Never mix sync and async I/O** -- use `asyncio.to_thread()` for blocking calls
- **Always use `async with` for async context managers** -- ensures proper cleanup
- **Prefer `TaskGroup` over `gather`** for structured concurrency (3.11+)
- **Use `asyncio.run()` as the single entry point** -- don't nest event loops
- **Cancel tasks gracefully** -- catch `asyncio.CancelledError` for cleanup
- **Avoid bare `await asyncio.sleep(0)`** -- use it only for explicit yield points
- **Use typed queues** -- `asyncio.Queue[ItemType]` for type safety

```python
# Run blocking code in async context
import asyncio

def cpu_heavy(data: bytes) -> bytes:
    # Blocking computation
    return process(data)

async def handle_request(data: bytes) -> bytes:
    # Run blocking code in a thread pool
    result = await asyncio.to_thread(cpu_heavy, data)
    return result
```

## Common Pitfalls

```python
# BAD: Creating coroutine without awaiting it
async def bad():
    fetch_data(url)       # coroutine created but never awaited!
    # RuntimeWarning: coroutine 'fetch_data' was never awaited

# GOOD
async def good():
    await fetch_data(url)

# BAD: Blocking the event loop
async def bad():
    time.sleep(5)         # blocks the entire event loop!

# GOOD
async def good():
    await asyncio.sleep(5)

# BAD: Ignoring task results (fire and forget leaks errors)
async def bad():
    asyncio.create_task(risky())  # exception silently lost

# GOOD: Store reference and handle errors
async def good():
    task = asyncio.create_task(risky())
    try:
        await task
    except Exception:
        logger.exception("Task failed")
```
