---
name: rust-tokio
description: >
  Tokio async runtime for Rust: task spawning, shared state, channels (mpsc/oneshot/broadcast/watch),
  async I/O, select! macro, streams, graceful shutdown, sync-async bridging, testing, and tracing.
  Use when writing, reviewing, or refactoring Rust code using Tokio: (1) Setting up a Tokio project
  with Cargo.toml feature flags, (2) Spawning tasks with tokio::spawn and JoinHandle, (3) Sharing
  state with Arc and Mutex across tasks, (4) Using mpsc/oneshot/broadcast/watch channels, (5) Async
  TCP/UDP I/O with AsyncRead/AsyncWrite, (6) Using tokio::select! for multiplexing futures,
  (7) Working with tokio-stream and StreamExt, (8) Implementing graceful shutdown with
  CancellationToken, (9) Bridging sync and async code, (10) Testing with #[tokio::test] and
  time pausing, (11) Adding tracing instrumentation.
---

# Rust Tokio

## Project Setup

```toml
# Cargo.toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

Feature flags: `rt`, `rt-multi-thread`, `net`, `fs`, `io-util`, `io-std`, `time`, `sync`, `signal`, `process`, `macros`. Use `"full"` for all.

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
