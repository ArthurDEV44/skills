# Threads vs Async in Rust

## When to Use What

### Use threads for:
- **CPU-bound, parallelizable work** (data processing, encoding, compression)
- Work that benefits from true parallelism across cores
- Simple "fire and forget" background work
- Code that calls blocking OS APIs or C libraries

### Use async for:
- **I/O-bound, concurrent work** (network requests, file I/O, database queries)
- Handling many concurrent connections (web servers, chat systems)
- Lightweight concurrency without thread overhead
- Embedded systems without OS thread support

### Use both together for:
- Video encoding (thread) with progress notifications (async channel)
- Background computation (thread) feeding results to an async pipeline
- Any mixed CPU-bound + I/O-bound workload

## Detailed Comparison

| Aspect | Threads | Async Tasks |
|--------|---------|-------------|
| Memory | ~8 MB per thread (OS stack) | ~KB per task (state machine) |
| Scheduling | OS preemptive (can interrupt anytime) | Cooperative (yields at await points) |
| Creation | `thread::spawn(closure)` | `tokio::spawn(async { ... })` |
| Waiting | `handle.join().unwrap()` | `handle.await.unwrap()` |
| Sleeping | `thread::sleep(duration)` | `tokio::time::sleep(duration).await` |
| Parallelism | True parallel on multiple cores | Concurrent; parallelism via multi-thread runtime |
| Blocking | Blocking one thread doesn't block others | Blocking the executor starves all tasks |
| Cancellation | Runs to completion (no built-in cancel) | Dropped when handle is dropped or runtime shuts down |
| Send bounds | Closure must be `Send + 'static` | Future must be `Send + 'static` (for `tokio::spawn`) |
| Error propagation | `JoinHandle<T>` with `Result` | `JoinHandle<T>` with `JoinError` |

## Mixing Threads and Async

### spawn_blocking -- blocking code from async context

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Offload blocking work to tokio's blocking thread pool
    let hash = tokio::task::spawn_blocking(move || {
        compute_hash(&large_data) // CPU-intensive, OK to block
    }).await?;

    // Back in async land
    save_hash(hash).await?;
    Ok(())
}
```

### blocking_send / blocking_recv -- sync-to-async bridge

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);

    // OS thread producing data
    std::thread::spawn(move || {
        for i in 1..=10 {
            tx.blocking_send(i).unwrap(); // blocks thread, not async runtime
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    // Async consumer
    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
}
```

### block_on -- running async from sync context

```rust
fn sync_function() -> String {
    // Create a runtime to run async code from synchronous context
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        fetch_data("https://example.com").await.unwrap()
    })
}
```

**Warning:** Never call `block_on` from within an async context (deadlock). Use `spawn_blocking` instead.

### Key rules when mixing:
- **Never** use `thread::sleep()` inside async code (blocks the executor)
- **Never** call `block_on` inside an async task (deadlock)
- Use `tokio::time::sleep().await` in async context
- Use `spawn_blocking()` to run blocking code from async context
- Use `blocking_send()` / `blocking_recv()` for sync-to-async channel bridging
- Bridge between sync and async with channels when ownership boundaries are complex

## Task Hierarchy

```
Runtime (executor + thread pool)
├── Worker Thread 1
│   ├── Task A (lightweight, runtime-scheduled)
│   │   ├── Future 1
│   │   └── Future 2 (tree of sub-futures)
│   └── Task B
│       └── Future 3
├── Worker Thread 2
│   └── Task C
│       ├── Future 4
│       └── Future 5
└── Blocking Thread Pool
    ├── spawn_blocking closure 1
    └── spawn_blocking closure 2
```

- **Runtime**: manages scheduling across OS threads (work-stealing in multi-threaded mode)
- **Tasks**: boundaries for concurrent work; can migrate between worker threads
- **Futures**: most granular unit; compose into trees via `.await`
- **Blocking pool**: separate threads for blocking operations, won't starve async tasks

## Cooperative Multitasking Pitfall

Async tasks yield **only at await points**. CPU-heavy code between awaits starves other tasks:

```rust
// BAD: blocks other tasks for the entire computation
async fn process() {
    expensive_computation(); // no await = no yield point
    another_heavy_fn();
}

// GOOD: yield periodically (for divisible work)
async fn process(data: &[u8]) {
    for chunk in data.chunks(1000) {
        process_chunk(chunk);
        tokio::task::yield_now().await;
    }
}

// BEST: offload to the blocking pool (for indivisible work)
async fn process() {
    let result = tokio::task::spawn_blocking(|| {
        expensive_computation()
    }).await.unwrap();
    use_result(result).await;
}
```

## Runtime Configuration

```rust
// Multi-threaded (default with #[tokio::main])
#[tokio::main]
async fn main() { /* uses all CPU cores */ }

// Single-threaded (good for testing, embedded, or when Send is hard)
#[tokio::main(flavor = "current_thread")]
async fn main() { /* single thread, no Send requirement for local tasks */ }

// Custom thread count
#[tokio::main(worker_threads = 4)]
async fn main() { /* exactly 4 worker threads */ }
```

## Sources

- [Futures, tasks, and threads](https://doc.rust-lang.org/book/ch17-06-futures-tasks-threads.html)
- [Concurrency with async](https://doc.rust-lang.org/book/ch17-02-concurrency-with-async.html)
- [Tokio Tutorial: Spawning](https://tokio.rs/tokio/tutorial/spawning)
- [Tokio: Bridging sync and async](https://tokio.rs/tokio/topics/bridging)
