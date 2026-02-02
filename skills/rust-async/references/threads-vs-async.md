# Threads vs Async in Rust

## When to Use What

### Use threads for:
- **CPU-bound, parallelizable work** (data processing, encoding, compression)
- Work that benefits from true parallelism across cores
- Simple "fire and forget" background work
- Code that calls blocking OS APIs

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
| Memory | ~MB per thread (OS stack) | ~KB per task |
| Scheduling | OS preemptive (can interrupt anytime) | Cooperative (yields at await points) |
| Creation | `thread::spawn(closure)` | `spawn_task(async { ... })` |
| Waiting | `handle.join().unwrap()` | `handle.await.unwrap()` |
| Sleeping | `thread::sleep(duration)` | `trpl::sleep(duration).await` |
| Parallelism | True parallel on multiple cores | Concurrent; may use parallelism via work-stealing runtime |
| Blocking | Blocking one thread doesn't block others | Blocking one task blocks all tasks on that executor thread |
| Cancellation | Runs to completion | Dropped when runtime shuts down (unless awaited) |

## Mixing Threads and Async

```rust
use std::{thread, time::Duration};

fn main() {
    let (tx, mut rx) = trpl::channel();

    // Spawn an OS thread for blocking/CPU work
    thread::spawn(move || {
        for i in 1..=10 {
            tx.send(i).unwrap();
            thread::sleep(Duration::from_secs(1)); // blocking sleep OK in threads
        }
        // tx dropped here -> channel closes
    });

    // Async runtime for I/O / event handling
    trpl::block_on(async {
        while let Some(message) = rx.recv().await {
            println!("{message}");
        }
    });
}
```

### Key rules when mixing:
- Never use `thread::sleep()` inside async code (blocks the executor)
- Use `trpl::sleep().await` or `tokio::time::sleep().await` in async
- Bridge between sync and async with channels
- Use `spawn_blocking()` (tokio) to run blocking code from async context

## Task Hierarchy

```
Runtime (executor)
├── Task 1 (lightweight, runtime-managed)
│   ├── Future A
│   └── Future B (tree of sub-futures)
├── Task 2
│   └── Future C
└── Task 3
    ├── Future D
    └── Future E
```

- **Runtime**: manages scheduling across OS threads (may use work-stealing)
- **Tasks**: boundaries for concurrent work; can be moved between threads
- **Futures**: most granular unit; compose into trees via await

## Cooperative Multitasking Pitfall

Async tasks yield **only at await points**. CPU-heavy code between awaits starves other tasks:

```rust
// BAD: blocks other tasks for the entire computation
let result = async {
    expensive_computation(); // no await = no yield point
    another_heavy_fn();
};

// GOOD: yield periodically
let result = async {
    for chunk in data.chunks(1000) {
        process(chunk);
        trpl::yield_now().await; // let other tasks run
    }
};

// GOOD: offload to a thread
let result = tokio::task::spawn_blocking(|| {
    expensive_computation()
}).await.unwrap();
```

## Sources

- [Futures, tasks, and threads](https://doc.rust-lang.org/book/ch17-06-futures-tasks-threads.html)
- [Concurrency with async](https://doc.rust-lang.org/book/ch17-02-concurrency-with-async.html)
- [More futures patterns](https://doc.rust-lang.org/book/ch17-03-more-futures.html)
