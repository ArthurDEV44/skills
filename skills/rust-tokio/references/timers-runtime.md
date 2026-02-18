# Timers, Runtime Configuration, and LocalSet

## Timers (tokio::time)

Requires feature flag `time` (included in `"full"`).

### sleep

```rust
use tokio::time::{self, Duration, Instant};

// Sleep for a duration
time::sleep(Duration::from_secs(1)).await;

// Sleep until a specific instant
time::sleep_until(Instant::now() + Duration::from_secs(1)).await;
```

#### Resetting a sleep in a loop

```rust
let sleep = time::sleep(Duration::from_millis(50));
tokio::pin!(sleep);

loop {
    tokio::select! {
        () = &mut sleep => {
            println!("timer fired");
            // Reset for next iteration
            sleep.as_mut().reset(Instant::now() + Duration::from_millis(50));
        }
        msg = rx.recv() => {
            if let Some(m) = msg { process(m); } else { break; }
        }
    }
}
```

### timeout

Wraps any future with a deadline. Returns `Err(Elapsed)` if the inner future doesn't complete in time.

```rust
use tokio::time::{self, Duration};

match time::timeout(Duration::from_secs(5), fetch_data()).await {
    Ok(Ok(data)) => println!("got data: {data:?}"),
    Ok(Err(e)) => println!("fetch error: {e}"),
    Err(_) => println!("timed out after 5s"),
}
```

#### timeout_at

```rust
use tokio::time::{self, Instant, Duration};

let deadline = Instant::now() + Duration::from_secs(10);
let result = time::timeout_at(deadline, do_work()).await;
```

### interval

Produces ticks at a fixed period. First tick completes immediately.

```rust
use tokio::time::{self, Duration};

let mut interval = time::interval(Duration::from_millis(100));
loop {
    interval.tick().await; // first tick is instant
    do_periodic_work();
}
```

#### interval_at (delayed start)

```rust
use tokio::time::{self, Duration, Instant};

// Start ticking 5 seconds from now, then every 1 second
let start = Instant::now() + Duration::from_secs(5);
let mut interval = time::interval_at(start, Duration::from_secs(1));
```

#### MissedTickBehavior

Controls what happens when the consumer is slow and ticks are missed:

```rust
use tokio::time::{self, Duration, MissedTickBehavior};

let mut interval = time::interval(Duration::from_millis(50));
interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
```

| Behavior | Description | Use case |
|----------|-------------|----------|
| `Burst` (default) | Fires missed ticks immediately in rapid succession | Catch up on work |
| `Delay` | Resets from now, pushing future ticks forward | Debouncing, keep minimum gap |
| `Skip` | Skips missed ticks, resumes at next aligned tick | Periodic polling, sampling |

### Instant

Tokio's `tokio::time::Instant` wraps `std::time::Instant` but works with paused time in tests:

```rust
use tokio::time::Instant;

let start = Instant::now();
do_work().await;
let elapsed = start.elapsed(); // works correctly with paused time
```

## Runtime Builder

### Multi-thread runtime (default)

```rust
let rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(4)                // default: number of CPU cores
    .max_blocking_threads(512)        // default: 512
    .thread_name("my-worker")
    .thread_name_fn(|| {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("worker-{id}")
    })
    .thread_stack_size(3 * 1024 * 1024)
    .on_thread_start(|| { /* per-thread init */ })
    .on_thread_stop(|| { /* per-thread cleanup */ })
    .enable_all()       // enable io + time subsystems
    .build()?;
```

### Current-thread runtime

```rust
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()?;
```

- Single-threaded: all tasks run on the thread calling `block_on`
- Lower overhead than multi-thread
- Tasks don't need to be `Send`
- Use with `LocalSet` for `!Send` futures

### Runtime handle

```rust
// Get handle to current runtime from within async context
let handle = tokio::runtime::Handle::current();

// Use handle from sync code to spawn tasks
handle.spawn(async { /* ... */ });

// Enter the runtime context (for creating resources like Interval)
let _guard = handle.enter();
```

## LocalSet (for !Send futures)

`LocalSet` runs `!Send` futures on the current thread. Required for types like `Rc`, non-thread-safe libraries.

```rust
use tokio::task::LocalSet;
use std::rc::Rc;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let local = LocalSet::new();

    let data = Rc::new("not Send");
    let data2 = data.clone();

    local.spawn_local(async move {
        println!("using Rc: {}", data2);
    });

    local.spawn_local(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        println!("done: {}", data);
    });

    local.await; // runs until all local tasks complete
}
```

### run_until

```rust
let local = LocalSet::new();
local.run_until(async {
    // Can spawn_local here
    tokio::task::spawn_local(async {
        println!("local task");
    }).await.unwrap();
}).await;
```

### When to use LocalSet

- Working with `!Send` types (`Rc`, `Cell`, `RefCell` across `.await`)
- Wrapping non-thread-safe C libraries
- Single-threaded event loops (GUI, game loops)
- Must use `current_thread` runtime flavor

## Task-Local Storage

Store per-task data accessible within the task without passing it through function arguments:

```rust
tokio::task_local! {
    static REQUEST_ID: u64;
}

async fn handle_request(id: u64) {
    REQUEST_ID.scope(id, async {
        // REQUEST_ID accessible in this scope and all nested calls
        process().await;
    }).await;
}

async fn process() {
    let id = REQUEST_ID.get(); // access the value
    println!("processing request {id}");
}
```

- `task_local!` defines a `LocalKey<T>`
- `.scope(value, future)` sets the value for the duration of the future
- `.get()` retrieves a `Copy` value, `.with(|v| ...)` for non-Copy
- Panics if accessed outside a `.scope()`

## Advanced Task Control

### yield_now

Voluntarily yield back to the scheduler, allowing other tasks to run:

```rust
tokio::task::yield_now().await;
```

Use for CPU-heavy async loops that don't have natural `.await` points, preventing starvation of other tasks on the same runtime thread.

### unconstrained

Opt out of Tokio's cooperative scheduling budget for a future:

```rust
use tokio::task::unconstrained;

let result = unconstrained(async {
    // This future won't be interrupted by the coop budget
    // Use sparingly: can starve other tasks
    intensive_computation().await
}).await;
```

Use only when you need a future to run to completion without being preempted by the cooperative scheduler (e.g., critical finalization logic).

## Sources

- [tokio::time module](https://docs.rs/tokio/latest/tokio/time/index.html)
- [Runtime](https://docs.rs/tokio/latest/tokio/runtime/struct.Runtime.html)
- [Builder](https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html)
- [LocalSet](https://docs.rs/tokio/latest/tokio/task/struct.LocalSet.html)
- [task_local!](https://docs.rs/tokio/latest/tokio/macro.task_local.html)
