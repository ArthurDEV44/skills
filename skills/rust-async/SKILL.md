---
name: rust-async
description: >
  Rust async/await programming patterns, futures, streams, concurrency, and runtime usage.
  Covers async fn, .await syntax, Future trait, Poll, Pin/Unpin, join/try_join, select,
  spawn, async channels, Stream/StreamExt, cancellation safety, Send bounds, async fn in
  traits, async closures, error handling, and threads vs async tradeoffs. Use when writing,
  reviewing, or refactoring Rust async code: (1) Writing async functions, blocks, or closures,
  (2) Spawning and joining async tasks, (3) Using channels for message passing between async
  tasks, (4) Working with streams and async iteration, (5) Choosing between threads and async,
  (6) Fixing Pin/Unpin or Send compiler errors, (7) Building timeout, racing, or cancellation-safe
  patterns, (8) Combining threads with async for mixed workloads, (9) Using async fn in trait
  definitions, (10) Handling errors with ? in async contexts.
---

# Rust Async

## Core Syntax

### async fn and .await

```rust
// async fn returns impl Future<Output = T>
async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    let resp = reqwest::get(url).await?;
    resp.text().await
}
```

- `async fn` compiles to a function returning `impl Future`
- `.await` is **postfix** -- chain it: `get(url).await?.text().await?`
- Futures are **lazy** -- nothing runs until polled (via `.await` or a runtime)
- Each `.await` is a yield point where the runtime can switch tasks

### async blocks

```rust
let fut = async {
    do_something().await;
    42 // resolves to i32
};

// async move takes ownership of captured variables
let name = String::from("hello");
let fut = async move {
    println!("{name}");
};
```

### async closures (Rust 1.85+)

```rust
// AsyncFn / AsyncFnMut / AsyncFnOnce traits
let fetch = async |url: &str| -> Result<String, reqwest::Error> {
    reqwest::get(url).await?.text().await
};

// As a higher-order function parameter
async fn retry<F>(f: impl AsyncFn() -> Result<String, Error>) -> Result<String, Error> {
    for _ in 0..3 {
        if let Ok(val) = f().await {
            return Ok(val);
        }
    }
    f().await
}
```

### Running async code with tokio

```rust
#[tokio::main]
async fn main() {
    let result = fetch_data("https://example.com").await.unwrap();
    println!("{result}");
}

// Or manually with a runtime
fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        fetch_data("https://example.com").await.unwrap();
    });
}
```

## Concurrency Patterns

### tokio::spawn -- fire-and-forget with join handle

```rust
let handle = tokio::spawn(async {
    tokio::time::sleep(Duration::from_secs(1)).await;
    "done"
});
// ... do other work ...
let result = handle.await.unwrap(); // JoinError if task panicked
```

- Spawned task runs independently on the runtime's thread pool
- Task must be `Send + 'static` (no borrowed data across `.await`)
- Task is **detached** if handle is dropped (keeps running, result discarded)

### join -- run futures concurrently, wait for all

```rust
use tokio::join;

// Known at compile time -- all must succeed
let (a, b, c) = join!(fetch("a"), fetch("b"), fetch("c"));

// Dynamic collection
use futures::future::join_all;
let urls = vec!["a", "b", "c"];
let results = join_all(urls.iter().map(|u| fetch(u))).await;
```

### try_join -- short-circuit on first error

```rust
use tokio::try_join;

// Returns Err as soon as any future fails
let (user, posts) = try_join!(
    fetch_user(id),
    fetch_posts(id),
)?;

// Dynamic collection
use futures::future::try_join_all;
let results = try_join_all(urls.iter().map(|u| fetch(u))).await?;
```

### select -- race futures, take first result

```rust
use tokio::select;

select! {
    val = fetch_from_primary() => println!("primary: {val:?}"),
    val = fetch_from_replica() => println!("replica: {val:?}"),
}
```

- The losing branch is **cancelled** (dropped) -- see [cancellation safety](#cancellation-safety)
- `select!` is biased by default (checks branches top to bottom); use `biased;` to make this explicit or randomize with default behavior

### Timeout pattern

```rust
use tokio::time::{timeout, Duration};

match timeout(Duration::from_secs(5), fetch_data(url)).await {
    Ok(Ok(data)) => println!("got: {data}"),
    Ok(Err(e))   => eprintln!("request failed: {e}"),
    Err(_)       => eprintln!("timed out"),
}
```

### Yielding control

```rust
// Give runtime a chance to switch tasks
tokio::task::yield_now().await;

// CPU-bound work: yield periodically
for chunk in data.chunks(1000) {
    process(chunk);
    tokio::task::yield_now().await;
}

// Better: offload blocking work entirely
let result = tokio::task::spawn_blocking(|| {
    expensive_computation()
}).await.unwrap();
```

## Cancellation Safety

Dropping a future at an `.await` point stops execution there. This can cause bugs if partial work was done before the `.await`:

```rust
// UNSAFE with select! -- data read from stream may be lost if cancelled
select! {
    data = stream.next() => { buffer.push(data); }
    _ = shutdown.recv() => { return; }
}

// SAFE -- use cancellation-safe methods or restructure
loop {
    let item = select! {
        item = stream.next() => item,
        _ = shutdown.recv() => break,
    };
    if let Some(data) = item {
        buffer.push(data);
    }
}
```

Key rules:
- `tokio::sync::mpsc::Receiver::recv()` is cancellation safe (no data lost)
- `tokio_stream::StreamExt::next()` is cancellation safe
- `AsyncReadExt::read()` is **not** cancellation safe (bytes may be partially read)
- When in doubt, check the method's docs for "Cancel safety" section

For detailed advanced patterns, see [references/advanced-patterns.md](references/advanced-patterns.md).

## Message Passing (Async Channels)

```rust
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::channel(32); // bounded, backpressure at capacity

let sender = tokio::spawn(async move {
    tx.send("hello").await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    tx.send("world").await.unwrap();
    // tx dropped -> channel closes
});

while let Some(msg) = rx.recv().await {
    println!("got: {msg}");
}
// loop exits when all senders are dropped
sender.await.unwrap();
```

Channel types:
- `mpsc::channel(cap)` -- multi-producer, single-consumer, bounded
- `mpsc::unbounded_channel()` -- unbounded (no backpressure, risk of OOM)
- `oneshot::channel()` -- single value, single send
- `broadcast::channel(cap)` -- multi-producer, multi-consumer
- `watch::channel(init)` -- single-producer, multi-consumer, latest-value only

## Streams

For detailed Stream/StreamExt trait internals, see [references/traits-internals.md](references/traits-internals.md).

```rust
use tokio_stream::{self as stream, StreamExt};

let mut s = stream::iter(vec![1, 2, 3]).map(|n| n * 2);

while let Some(value) = s.next().await {
    println!("{value}");
}
```

### Buffered concurrent processing

```rust
use tokio_stream::StreamExt;
use futures::stream::{self, StreamExt as FutStreamExt};

// Process up to 10 futures concurrently from the stream
let results: Vec<_> = stream::iter(urls)
    .map(|url| fetch(url))
    .buffer_unordered(10) // up to 10 in-flight at once
    .collect()
    .await;
```

- `buffer_unordered(n)` -- results arrive in completion order
- `buffered(n)` -- results maintain input order
- Stream = async Iterator (items arrive over time)
- `while let Some(v) = stream.next().await` is the async iteration pattern

## Threads vs Async

For full comparison and mixing patterns, see [references/threads-vs-async.md](references/threads-vs-async.md).

| | Threads | Async |
|---|---------|-------|
| Best for | CPU-bound, parallelizable work | I/O-bound, concurrent work |
| Overhead | ~MB per thread (OS stack) | Lightweight tasks (KB) |
| Scheduling | OS preemptive | Runtime cooperative |
| Blocking | Safe to block | Never block the executor |

Combine both when needed:

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);

    // CPU-heavy work in a blocking thread
    tokio::task::spawn_blocking(move || {
        for i in 1..=10 {
            tx.blocking_send(i).unwrap(); // blocking send from sync context
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    // I/O handling in async
    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
}
```

## Error Handling in Async

```rust
// ? works naturally in async fn returning Result
async fn process(id: u64) -> Result<Output, AppError> {
    let user = fetch_user(id).await?;       // propagates error
    let data = fetch_data(user.key).await?;
    Ok(transform(data))
}

// Map errors across different types
async fn combined() -> Result<(), Box<dyn std::error::Error>> {
    let user = fetch_user(1).await?;   // reqwest::Error
    save_to_db(&user).await?;          // sqlx::Error
    Ok(())
}

// Collect results from concurrent operations
let results: Result<Vec<_>, _> = try_join_all(
    ids.iter().map(|&id| fetch_user(id))
).await;
```

## async fn in Traits (Rust 1.75+)

```rust
trait DataStore {
    async fn get(&self, key: &str) -> Option<String>;
    async fn set(&self, key: &str, value: String) -> Result<(), StoreError>;
}

impl DataStore for RedisStore {
    async fn get(&self, key: &str) -> Option<String> {
        self.client.get(key).await.ok()
    }
    async fn set(&self, key: &str, value: String) -> Result<(), StoreError> {
        self.client.set(key, value).await.map_err(StoreError::from)
    }
}
```

Limitation: `async fn` in traits does not imply `Send`. For spawnable trait methods, see [references/advanced-patterns.md](references/advanced-patterns.md).

## Official Documentation

- [The Rust Book: Async/Await](https://doc.rust-lang.org/book/ch17-00-async-await.html)
- [Async Book](https://rust-lang.github.io/async-book/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [futures crate docs](https://docs.rs/futures/latest/futures/)
- [tokio-stream docs](https://docs.rs/tokio-stream/latest/tokio_stream/)
