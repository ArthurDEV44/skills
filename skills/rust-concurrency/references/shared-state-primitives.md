# Shared State Primitives -- Deep Dive

## Table of Contents

- [RwLock](#rwlock)
- [Condvar](#condvar)
- [Barrier](#barrier)
- [OnceLock](#oncelock)
- [LazyLock](#lazylock)
- [Mutex Poisoning Recovery](#mutex-poisoning-recovery)
- [Deadlock Avoidance](#deadlock-avoidance)

## RwLock

`RwLock<T>` allows multiple concurrent readers **or** one exclusive writer. Prefer over `Mutex` when reads vastly outnumber writes.

```rust
use std::sync::{Arc, RwLock};
use std::thread;

let config = Arc::new(RwLock::new(String::from("v1.0")));
let mut handles = vec![];

// Spawn multiple readers
for i in 0..5 {
    let config = Arc::clone(&config);
    handles.push(thread::spawn(move || {
        let val = config.read().unwrap();
        println!("Reader {i}: {val}");
    }));
}

// Spawn a single writer
{
    let config = Arc::clone(&config);
    handles.push(thread::spawn(move || {
        let mut val = config.write().unwrap();
        *val = String::from("v2.0");
        println!("Writer updated config");
    }));
}

for h in handles { h.join().unwrap(); }
```

### Key properties

- `read()` returns `RwLockReadGuard` -- multiple readers can hold this simultaneously
- `write()` returns `RwLockWriteGuard` -- exclusive access, blocks all readers and writers
- `try_read()` / `try_write()` return immediately with `Err` if lock is unavailable
- RwLock is **poisonable** -- a panic while holding a write guard poisons the lock
- Read guards do not poison the lock (a panic while reading is fine)
- **Writer starvation** is possible: if readers always hold the lock, writers may never acquire it. Behavior is OS-dependent.

### When to use RwLock vs Mutex

| Scenario | Use |
|----------|-----|
| Mostly reads, infrequent writes | `RwLock` |
| Short critical sections | `Mutex` (less overhead) |
| Write-heavy workloads | `Mutex` |
| Need `try_read`/`try_write` | `RwLock` |

## Condvar

`Condvar` (condition variable) lets threads wait for a condition to become true, avoiding busy-waiting. Always used with a `Mutex`.

### Producer-consumer pattern

```rust
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

let pair = Arc::new((Mutex::new(false), Condvar::new()));

// Consumer: waits for data
let pair_clone = Arc::clone(&pair);
let consumer = thread::spawn(move || {
    let (lock, cvar) = &*pair_clone;
    let mut ready = lock.lock().unwrap();

    // wait() atomically unlocks the mutex and blocks this thread.
    // When notified, it re-acquires the lock before returning.
    // Loop to guard against spurious wakeups.
    while !*ready {
        ready = cvar.wait(ready).unwrap();
    }
    println!("Consumer: data is ready!");
});

// Producer: signals when data is ready
let (lock, cvar) = &*pair;
{
    let mut ready = lock.lock().unwrap();
    *ready = true;
}
cvar.notify_one(); // wake one waiting thread

consumer.join().unwrap();
```

### wait_while convenience

```rust
use std::sync::{Condvar, Mutex};

let pair = (Mutex::new(0u32), Condvar::new());
let (lock, cvar) = &pair;

let mut count = lock.lock().unwrap();
// wait_while handles the spurious-wakeup loop for you
count = cvar.wait_while(count, |c| *c < 5).unwrap();
assert!(*count >= 5);
```

### Key properties

- `wait(guard)` unlocks the mutex, blocks, then re-locks before returning
- **Spurious wakeups**: `wait()` can return without being notified -- always check the condition in a loop
- `wait_while(guard, condition)` handles the loop for you
- `wait_timeout(guard, duration)` adds a timeout
- `notify_one()` wakes one waiting thread; `notify_all()` wakes all
- The notification is **not queued**: if no thread is waiting when `notify` is called, the signal is lost

## Barrier

`Barrier` blocks threads until a specified number of threads have all called `wait()`. Useful for phased computation.

```rust
use std::sync::{Arc, Barrier};
use std::thread;

let n = 5;
let barrier = Arc::new(Barrier::new(n));
let mut handles = vec![];

for i in 0..n {
    let barrier = Arc::clone(&barrier);
    handles.push(thread::spawn(move || {
        println!("Thread {i}: phase 1 done");
        barrier.wait(); // all threads synchronize here
        println!("Thread {i}: phase 2 start");
    }));
}

for h in handles { h.join().unwrap(); }
```

- `wait()` returns `BarrierWaitResult`; exactly one thread gets `is_leader() == true`
- Reusable: after all threads pass through, the barrier resets automatically
- Panicking threads reduce the expected count permanently (barrier may never release)

## OnceLock

`OnceLock<T>` (Rust 1.70+) is a cell that can be written to only once, thread-safely. Replaces `once_cell::sync::OnceCell`.

```rust
use std::sync::OnceLock;

static CONFIG: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static String {
    CONFIG.get_or_init(|| {
        // Expensive initialization, runs at most once
        String::from("initialized")
    })
}

fn main() {
    println!("{}", get_config()); // triggers init
    println!("{}", get_config()); // returns cached value
}
```

### Key properties

- `get()` returns `Option<&T>` -- `None` if not yet initialized
- `get_or_init(f)` initializes with `f` if empty (blocking, thread-safe)
- `set(value)` returns `Err(value)` if already initialized (non-blocking)
- Ideal for global config, lazily-initialized singletons, one-time computation

## LazyLock

`LazyLock<T>` (Rust 1.80+) is `OnceLock` with a built-in initializer. Replaces `once_cell::sync::Lazy`.

```rust
use std::sync::LazyLock;

static GLOBAL_DATA: LazyLock<Vec<i32>> = LazyLock::new(|| {
    println!("Computing...");
    vec![1, 2, 3, 4, 5]
});

fn main() {
    println!("Before access");
    println!("{:?}", *GLOBAL_DATA); // triggers init on first access
    println!("{:?}", *GLOBAL_DATA); // uses cached value, no recomputation
}
```

- Implements `Deref<Target = T>` -- use like a normal reference
- Initialization happens on first dereference, exactly once
- Panics if the initializer panics (the panic is propagated to all waiting threads)
- Preferred over `OnceLock` when the initializer is known at definition time

## Mutex Poisoning Recovery

When a thread panics while holding a `MutexGuard`, the mutex becomes "poisoned." Subsequent `lock()` calls return `Err(PoisonError)`.

### Ignoring poison (recovering the data)

```rust
use std::sync::Mutex;

let m = Mutex::new(vec![1, 2, 3]);

// After a panic in another thread:
let data = m.lock().unwrap_or_else(|poisoned| {
    eprintln!("Lock was poisoned, recovering");
    poisoned.into_inner()
});
println!("{data:?}");
```

### Clearing poison (Rust 1.77+)

```rust
use std::sync::Mutex;

let m = Mutex::new(5);

// After recovering from poison, clear it so future lock() calls succeed
m.clear_poison();
assert!(!m.is_poisoned());
```

### When to recover vs propagate

- **Propagate** (`.unwrap()`) when the data may be in an inconsistent state after a panic -- this is the common case
- **Recover** (`.into_inner()`) when the data is self-consistent (e.g., simple counters, append-only logs) or you can validate/repair it
- **Clear** when you've verified the data integrity and want to resume normal operation

## Deadlock Avoidance

### Lock ordering

The most common deadlock: two threads acquire the same locks in different order.

```text
Thread A: lock(X) then lock(Y)
Thread B: lock(Y) then lock(X)
// -> Deadlock if A holds X and B holds Y
```

**Fix**: Establish a global ordering for locks and always acquire in that order.

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let resource_a = Arc::new(Mutex::new("A"));
let resource_b = Arc::new(Mutex::new("B"));

// ALWAYS acquire A before B in every thread
let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
let t1 = thread::spawn(move || {
    let _a = ra.lock().unwrap();
    let _b = rb.lock().unwrap();
    // use both resources
});

let (ra, rb) = (Arc::clone(&resource_a), Arc::clone(&resource_b));
let t2 = thread::spawn(move || {
    let _a = ra.lock().unwrap(); // same order as t1
    let _b = rb.lock().unwrap();
    // use both resources
});

t1.join().unwrap();
t2.join().unwrap();
```

### try_lock to avoid blocking

```rust
use std::sync::Mutex;

let m = Mutex::new(42);

match m.try_lock() {
    Ok(guard) => println!("Got lock: {}", *guard),
    Err(_) => println!("Lock is held by another thread, skipping"),
}
```

### General deadlock prevention rules

1. **Consistent lock ordering** -- number your locks and always acquire in ascending order
2. **Minimize lock scope** -- drop guards as early as possible, use explicit blocks `{}`
3. **Avoid nested locks** -- restructure to use a single lock or channels
4. **Never hold a lock across I/O** -- release the lock, do I/O, re-acquire
5. **Prefer channels** -- message passing avoids lock-based deadlocks entirely
6. **Use `try_lock()`** -- for best-effort acquisition without blocking
