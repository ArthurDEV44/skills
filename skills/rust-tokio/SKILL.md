---
name: rust-tokio
description: >
  Tokio async runtime for Rust: task spawning, JoinSet, shared state, channels (mpsc/oneshot/broadcast/watch),
  sync primitives (Notify/Semaphore/Barrier/RwLock), timers (sleep/timeout/interval), async I/O, select!,
  streams, graceful shutdown, sync-async bridging, testing, tracing. Use when writing or reviewing Tokio code:
  (1) Spawning tasks with tokio::spawn, JoinSet, JoinHandle, (2) Sharing state with Arc/Mutex,
  (3) Using channels or sync primitives, (4) Working with timeout/interval/sleep,
  (5) Async TCP/UDP I/O with AsyncRead/AsyncWrite, (6) Using select! for multiplexing,
  (7) tokio-stream and StreamExt, (8) Graceful shutdown with CancellationToken/TaskTracker,
  (9) Bridging sync/async, (10) Testing with #[tokio::test], (11) Tracing, (12) Runtime Builder/LocalSet.
---

# Rust Tokio

## Project Setup

```toml
# Cargo.toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

Feature flags: `rt`, `rt-multi-thread`, `net`, `fs`, `io-util`, `io-std`, `time`, `sync`, `signal`, `process`, `macros`. Use `"full"` for all. For production, pick only what you need.

## Runtime & Entry Point

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // async code here
    Ok(())
}
```

Expands to:
```rust
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ })
}
```

Use `#[tokio::main(flavor = "current_thread")]` for single-threaded runtime (lighter, no Send requirement on spawned tasks when using LocalSet).

### Runtime Builder

```rust
let rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(4)           // default: num CPUs
    .thread_name("my-worker")
    .thread_stack_size(3 * 1024 * 1024)
    .enable_all()                // enable io + time
    .build()?;

rt.block_on(async { /* ... */ });
```

For details on Builder options and LocalSet, see [references/timers-runtime.md](references/timers-runtime.md).

## Task Spawning

```rust
// Spawn a task (requires 'static + Send)
let handle = tokio::spawn(async move {
    // ... async work ...
    "result"
});
let result = handle.await.unwrap();
```

- Tasks are lightweight (~64 bytes). Spawn thousands freely.
- **'static bound**: task must own all data. Use `move` to transfer ownership.
- **Send bound**: all data held across `.await` must be `Send`. `Rc` across `.await` fails; scope it or use `Arc`.
- `spawn_blocking(|| { ... })` for CPU-bound/blocking work on dedicated threads.

### JoinSet (managing groups of tasks)

```rust
use tokio::task::JoinSet;

let mut set = JoinSet::new();

for i in 0..10 {
    set.spawn(async move { expensive_work(i).await });
}

// Collect results as they complete (unordered)
while let Some(result) = set.join_next().await {
    let val = result?; // JoinError if task panicked
    println!("got: {val}");
}
```

- `set.abort_all()` cancels all tasks in the set
- `set.len()` / `set.is_empty()` for status checks
- `set.shutdown().await` aborts all and waits for completion
- `set.spawn_blocking(|| { ... })` for blocking tasks in the set
- Dropping a JoinSet aborts all tasks in it

### Aborting tasks

```rust
let handle = tokio::spawn(async { long_running().await });
handle.abort();  // request cancellation
// await returns JoinError with is_cancelled() == true
assert!(handle.await.unwrap_err().is_cancelled());
```

Note: `spawn_blocking` tasks cannot be aborted (they run on OS threads).

## Shared State

```rust
use std::sync::{Arc, Mutex};
type Db = Arc<Mutex<HashMap<String, Bytes>>>;

let db = Arc::new(Mutex::new(HashMap::new()));
let db_clone = db.clone();
tokio::spawn(async move {
    let mut map = db_clone.lock().unwrap();
    map.insert("key".into(), "value".into());
});
```

**std::sync::Mutex vs tokio::sync::Mutex**:
- Use `std::sync::Mutex` for short critical sections not spanning `.await` (cheaper)
- Use `tokio::sync::Mutex` only when lock must be held across `.await` points
- Never hold `MutexGuard` across `.await` with std Mutex (not `Send`)

Fix: scope the guard in a block `{ let guard = m.lock().unwrap(); ... }` before `.await`.

For high contention: shard the mutex, use message passing, or crates like `dashmap`.

## Channels

| Type | Producers | Consumers | Values | Use case |
|------|-----------|-----------|--------|----------|
| `mpsc` | Many | One | Buffered | Command queues, fan-in |
| `oneshot` | One | One | Single | Request/response |
| `broadcast` | Many | Many | All see every msg | Event notification |
| `watch` | Many | Many | Latest only | Config updates |

For detailed patterns (command enum, manager task, oneshot responses), see [references/channels.md](references/channels.md).

```rust
// mpsc
let (tx, mut rx) = tokio::sync::mpsc::channel(32);
tx.send("msg").await?;
while let Some(msg) = rx.recv().await { /* ... */ }

// oneshot
let (tx, rx) = tokio::sync::oneshot::channel();
tx.send("response").unwrap(); // no .await needed
let val = rx.await?;
```

## Sync Primitives

| Primitive | Purpose | Key method |
|-----------|---------|------------|
| `Notify` | Wake one/all waiters | `notify_one()`, `notify_waiters()`, `notified().await` |
| `Semaphore` | Limit concurrency | `acquire().await`, `try_acquire()`, `add_permits()` |
| `Barrier` | Sync N tasks at a point | `wait().await` (returns `BarrierWaitResult`) |
| `RwLock` | Async reader-writer lock | `read().await`, `write().await` |

```rust
// Notify: signal between tasks
let notify = Arc::new(tokio::sync::Notify::new());
let n = notify.clone();
tokio::spawn(async move { n.notified().await; /* woken */ });
notify.notify_one();

// Semaphore: limit concurrent operations
let sem = Arc::new(tokio::sync::Semaphore::new(3)); // 3 permits
let permit = sem.acquire().await?;
do_work().await;
drop(permit); // release

// Barrier: wait for all tasks to reach a point
let barrier = Arc::new(tokio::sync::Barrier::new(5));
let b = barrier.clone();
tokio::spawn(async move { b.wait().await; /* all 5 arrived */ });
```

For detailed patterns, see [references/sync-primitives.md](references/sync-primitives.md).

## Timers

```rust
use tokio::time::{self, Duration, Instant};

// Sleep
time::sleep(Duration::from_secs(1)).await;

// Timeout: wrap any future with a deadline
match time::timeout(Duration::from_secs(5), fetch_data()).await {
    Ok(result) => println!("got: {result:?}"),
    Err(_) => println!("timed out"),
}

// Interval: periodic ticks
let mut interval = time::interval(Duration::from_millis(100));
loop {
    interval.tick().await;
    do_periodic_work();
}
```

For MissedTickBehavior, interval_at, sleep reset patterns, see [references/timers-runtime.md](references/timers-runtime.md).

## select! Macro

```rust
tokio::select! {
    val = future_a => { /* a completed first */ }
    val = future_b => { /* b completed first */ }
    else => { /* all branches disabled */ }
}
```

- Runs on **same task** (no spawn needed), enables borrowing
- Uncompleted branches are **cancelled** (dropped)
- Randomly picks ready branch to check first (fair)
- Supports pattern matching: `Some(v) = rx.recv() => { ... }`
- Preconditions: `val = op, if condition => { ... }`
- Use `tokio::pin!(fut)` + `&mut fut` to resume across loop iterations

For full select! patterns, see [references/select-streams.md](references/select-streams.md).

## Async I/O

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Read
let n = socket.read(&mut buf).await?; // Ok(0) = EOF
socket.read_to_end(&mut vec).await?;

// Write
socket.write_all(b"hello").await?;

// Copy
let (mut rd, mut wr) = socket.split(); // same-task zero-cost
tokio::io::copy(&mut rd, &mut wr).await?;
```

For framing, buffered I/O, and codec patterns, see [references/io-framing.md](references/io-framing.md).

## TCP Server Pattern

```rust
use tokio::net::TcpListener;

let listener = TcpListener::bind("127.0.0.1:8080").await?;
loop {
    let (socket, addr) = listener.accept().await?;
    tokio::spawn(async move {
        process(socket).await;
    });
}
```

## Graceful Shutdown

For full patterns with CancellationToken and TaskTracker, see [references/shutdown-testing-tracing.md](references/shutdown-testing-tracing.md).

```rust
use tokio::signal;
tokio::select! {
    _ = server_loop() => {}
    _ = signal::ctrl_c() => { println!("shutting down"); }
}
```

Use `tokio_util::sync::CancellationToken` for multi-task shutdown coordination.
Use `tokio_util::task::TaskTracker` to wait for all spawned tasks to complete.

## Testing

```rust
#[tokio::test]
async fn my_test() {
    let result = my_async_fn().await;
    assert_eq!(result, 42);
}

// Pause time for instant sleep/timeout tests
#[tokio::test(start_paused = true)]
async fn time_test() {
    tokio::time::sleep(Duration::from_secs(100)).await; // instant
}
```

Requires feature `test-util` for `start_paused`. Mock I/O with `tokio_test::io::Builder`.

## Bridging Sync/Async

```rust
// From sync code: create runtime manually
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all().build()?;
let result = rt.block_on(async { fetch_data().await });

// From async code: run blocking work
let result = tokio::task::spawn_blocking(|| {
    heavy_computation()
}).await?;
```

## Tracing

```rust
// Cargo.toml: tracing = "0.1", tracing-subscriber = "0.3"
tracing_subscriber::fmt::init();

#[tracing::instrument]
async fn handle_request(id: u64) {
    tracing::info!("processing request");
}
```

## Official Documentation

- [Tokio API docs](https://docs.rs/tokio/latest/tokio/)
- [Tutorial overview](https://tokio.rs/tokio/tutorial)
- [Setup](https://tokio.rs/tokio/tutorial/setup)
- [Hello Tokio](https://tokio.rs/tokio/tutorial/hello-tokio)
- [Spawning](https://tokio.rs/tokio/tutorial/spawning)
- [Shared state](https://tokio.rs/tokio/tutorial/shared-state)
- [Channels](https://tokio.rs/tokio/tutorial/channels)
- [I/O](https://tokio.rs/tokio/tutorial/io)
- [Framing](https://tokio.rs/tokio/tutorial/framing)
- [Async in depth](https://tokio.rs/tokio/tutorial/async)
- [Select](https://tokio.rs/tokio/tutorial/select)
- [Streams](https://tokio.rs/tokio/tutorial/streams)
- [Bridging sync/async](https://tokio.rs/tokio/topics/bridging)
- [Graceful shutdown](https://tokio.rs/tokio/topics/shutdown)
- [Tracing](https://tokio.rs/tokio/topics/tracing)
- [Tracing next steps](https://tokio.rs/tokio/topics/tracing-next-steps)
- [Testing](https://tokio.rs/tokio/topics/testing)
- [Glossary](https://tokio.rs/tokio/glossary)
