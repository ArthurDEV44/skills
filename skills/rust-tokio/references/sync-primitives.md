# Tokio Sync Primitives

Beyond channels, Tokio provides async-aware synchronization primitives in `tokio::sync`.

## Notify

Basic async task notification. A task can wait until another task signals it.

```rust
use std::sync::Arc;
use tokio::sync::Notify;

let notify = Arc::new(Notify::new());

// Waiter
let n = notify.clone();
let waiter = tokio::spawn(async move {
    n.notified().await;
    println!("received notification");
});

// Notifier
notify.notify_one(); // wake one waiter
waiter.await.unwrap();
```

### Key methods

- `notify_one()` -- wake a single waiter. If no one is waiting, the permit is stored so the next `notified().await` completes immediately.
- `notify_waiters()` -- wake **all** currently waiting tasks (does not store a permit for future waiters).
- `notified().await` -- wait for a notification. Returns immediately if a permit was stored by a prior `notify_one()`.

### Producer-consumer wake pattern

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct SharedQueue<T> {
    data: Mutex<VecDeque<T>>,
    notify: Notify,
}

impl<T> SharedQueue<T> {
    fn push(&self, item: T) {
        self.data.lock().unwrap().push_back(item);
        self.notify.notify_one();
    }

    async fn pop(&self) -> T {
        loop {
            if let Some(item) = self.data.lock().unwrap().pop_front() {
                return item;
            }
            self.notify.notified().await;
        }
    }
}
```

### Notify vs channels

- Use `Notify` when you just need to **signal** (no data). Lighter than a channel.
- Use `mpsc`/`oneshot` when you need to **send values**.
- `Notify` supports one-shot or repeated wakeups in a loop.

## Semaphore

Limits the number of concurrent operations by managing a pool of permits.

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

let sem = Arc::new(Semaphore::new(3)); // 3 concurrent permits

let mut handles = vec![];
for i in 0..10 {
    let permit = sem.clone().acquire_owned().await.unwrap();
    handles.push(tokio::spawn(async move {
        do_work(i).await;
        drop(permit); // release when done
    }));
}
```

### Key methods

- `acquire().await` -- wait for a permit (returns `SemaphorePermit`, released on drop)
- `acquire_owned().await` -- returns `OwnedSemaphorePermit` (can move across tasks, requires `Arc<Semaphore>`)
- `try_acquire()` -- non-blocking, returns `Err(TryAcquireError)` if unavailable
- `acquire_many(n).await` -- acquire n permits at once
- `add_permits(n)` -- dynamically add permits
- `close()` -- close the semaphore, all pending acquires return `AcquireError`
- `available_permits()` -- check current count
- `Semaphore::MAX_PERMITS` -- maximum permits supported

### Rate limiter pattern

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

async fn rate_limited_fetch(urls: Vec<String>, max_concurrent: usize) -> Vec<String> {
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let mut set = tokio::task::JoinSet::new();

    for url in urls {
        let permit = sem.clone().acquire_owned().await.unwrap();
        set.spawn(async move {
            let result = reqwest::get(&url).await.unwrap().text().await.unwrap();
            drop(permit);
            result
        });
    }

    let mut results = vec![];
    while let Some(res) = set.join_next().await {
        results.push(res.unwrap());
    }
    results
}
```

### OwnedSemaphorePermit for async guards

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

struct Connection {
    _permit: tokio::sync::OwnedSemaphorePermit,
    // ... connection fields
}

impl Connection {
    async fn new(pool: Arc<Semaphore>) -> Self {
        let permit = pool.acquire_owned().await.unwrap();
        Connection { _permit: permit }
    }
    // permit released when Connection is dropped
}
```

## Barrier

Synchronization point where N tasks must all arrive before any can proceed.

```rust
use std::sync::Arc;
use tokio::sync::Barrier;

let barrier = Arc::new(Barrier::new(3));
let mut handles = vec![];

for i in 0..3 {
    let b = barrier.clone();
    handles.push(tokio::spawn(async move {
        println!("task {i} doing setup");
        let result = b.wait().await;
        // All 3 tasks arrive here before any proceeds
        if result.is_leader() {
            println!("task {i} is the leader");
        }
        println!("task {i} continuing");
    }));
}

for h in handles { h.await.unwrap(); }
```

- `wait().await` returns `BarrierWaitResult`
- Exactly one waiter gets `is_leader() == true` (useful for one-time initialization)
- Barrier is reusable: after all N tasks pass, it resets

## RwLock

Async reader-writer lock. Multiple readers OR one writer at a time.

```rust
use tokio::sync::RwLock;

let lock = RwLock::new(HashMap::new());

// Multiple concurrent readers
let data = lock.read().await;
println!("{:?}", *data);
drop(data); // release read lock

// Exclusive writer
let mut data = lock.write().await;
data.insert("key".into(), "value".into());
// write lock released on drop
```

### Key methods

- `read().await` -- shared read access (returns `RwLockReadGuard`)
- `write().await` -- exclusive write access (returns `RwLockWriteGuard`)
- `try_read()` / `try_write()` -- non-blocking variants
- `read_owned().await` / `write_owned().await` -- owned guards (movable across tasks, requires `Arc<RwLock<T>>`)

### When to use RwLock vs Mutex

- **RwLock**: read-heavy workloads where writes are rare. Concurrent reads improve throughput.
- **Mutex**: write-heavy or short critical sections. Simpler, less overhead.
- **std::sync::RwLock**: never use across `.await` (same as std::sync::Mutex rule)
- **tokio::sync::RwLock**: safe across `.await` but more expensive. Use only when the lock guard must span await points.

### Write-preferring behavior

Tokio's RwLock is **write-preferring**: pending writers block new readers. This prevents writer starvation but can cause reader starvation under heavy write contention.

## Sources

- [tokio::sync module](https://docs.rs/tokio/latest/tokio/sync/index.html)
- [Notify](https://docs.rs/tokio/latest/tokio/sync/struct.Notify.html)
- [Semaphore](https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html)
- [Barrier](https://docs.rs/tokio/latest/tokio/sync/struct.Barrier.html)
- [RwLock](https://docs.rs/tokio/latest/tokio/sync/struct.RwLock.html)
