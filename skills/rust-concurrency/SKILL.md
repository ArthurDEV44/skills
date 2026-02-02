---
name: rust-concurrency
description: >
  Rust concurrency and parallelism patterns: threads, message passing with channels, shared state
  with Mutex and Arc, Send and Sync marker traits, data races vs race conditions, and atomics
  with memory orderings. Use when writing, reviewing, or refactoring Rust concurrent code:
  (1) Spawning threads with std::thread and JoinHandle, (2) Using move closures with threads,
  (3) Message passing with mpsc channels (single and multiple producers), (4) Shared-state
  concurrency with Mutex and Arc, (5) Understanding or implementing Send and Sync traits,
  (6) Fixing "cannot be sent between threads safely" errors, (7) Choosing between message passing
  and shared state, (8) Working with atomics and memory orderings (SeqCst, Acquire, Release,
  Relaxed), (9) Preventing data races and understanding race conditions, (10) Building thread-safe
  types with raw pointers.
---

# Rust Concurrency

## Threads

### Spawning and joining

```rust
use std::thread;
use std::time::Duration;

let handle = thread::spawn(|| {
    for i in 1..10 {
        println!("spawned: {i}");
        thread::sleep(Duration::from_millis(1));
    }
});

// do other work...

handle.join().unwrap(); // blocks until thread finishes
```

- `thread::spawn` returns `JoinHandle<T>` -- call `.join()` to wait for completion
- When main thread ends, **all spawned threads are shut down** whether finished or not
- Where you call `.join()` matters: calling it before other work serializes execution
- Rust uses 1:1 threading model (one OS thread per language thread)

### move closures -- transferring ownership to threads

```rust
use std::thread;

let v = vec![1, 2, 3];
let handle = thread::spawn(move || {
    println!("vector: {v:?}");
});
handle.join().unwrap();
// v is no longer accessible here -- ownership moved to thread
```

- `move` forces the closure to take ownership of captured values
- Required because Rust can't guarantee the spawned thread won't outlive the borrowed data
- Without `move`, compiler errors: `closure may outlive the current function`

## Message Passing (Channels)

### Basic mpsc channel

```rust
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel(); // multiple producer, single consumer

thread::spawn(move || {
    tx.send(String::from("hello")).unwrap();
    // tx is moved -- dropped when thread ends
});

let msg = rx.recv().unwrap(); // blocks until message arrives
```

- `send()` takes ownership of the value -- prevents use-after-send
- `recv()` blocks; `try_recv()` returns immediately (non-blocking)
- Channel closes when all transmitters are dropped

### Iterating over received messages

```rust
let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    for val in ["hi", "from", "thread"] {
        tx.send(val.to_string()).unwrap();
        thread::sleep(Duration::from_secs(1));
    }
});

for received in rx {  // iterates until channel closes
    println!("Got: {received}");
}
```

### Multiple producers

```rust
let (tx, rx) = mpsc::channel();
let tx1 = tx.clone(); // clone transmitter for second producer

thread::spawn(move || { tx1.send("from thread 1").unwrap(); });
thread::spawn(move || { tx.send("from thread 2").unwrap(); });

for msg in rx {
    println!("{msg}");
}
```

## Shared State (Mutex + Arc)

### Mutex<T> -- mutual exclusion

```rust
use std::sync::Mutex;

let m = Mutex::new(5);
{
    let mut num = m.lock().unwrap(); // returns MutexGuard
    *num = 6;
} // MutexGuard dropped here -> lock released automatically

println!("m = {m:?}"); // 6
```

- `lock()` blocks until lock is acquired; returns `MutexGuard` (smart pointer)
- `MutexGuard` implements `Deref`/`DerefMut` for access and `Drop` for auto-unlock
- If lock holder panicked, `lock()` returns `Err` (mutex is "poisoned")
- Type system enforces: **cannot access data without acquiring the lock**

### Arc<T> -- thread-safe reference counting

`Rc<T>` is **not** thread-safe. Use `Arc<T>` for shared ownership across threads:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

println!("Result: {}", *counter.lock().unwrap()); // 10
```

### Choosing between channels and shared state

| | Message passing | Shared state |
|---|---|---|
| Model | Transfer ownership of data | Multiple owners with synchronized access |
| Primitives | `mpsc::channel` | `Arc<Mutex<T>>` |
| Best for | Pipelines, actor patterns | Counters, caches, shared collections |
| Analogy | Single-ownership | Multiple-ownership with interior mutability |

`RefCell<T>`/`Rc<T>` (single-threaded) maps to `Mutex<T>`/`Arc<T>` (multi-threaded)

## Send and Sync Traits

For unsafe implementations and the full Carton<T> example, see [references/send-sync-deep.md](references/send-sync-deep.md).

- **`Send`** -- safe to transfer ownership to another thread
- **`Sync`** -- safe to reference from multiple threads (`T: Sync` iff `&T: Send`)

### Auto-derivation

Types composed entirely of `Send`/`Sync` types are automatically `Send`/`Sync`. Almost all primitives are both.

### Notable exceptions

| Type | Send | Sync | Why |
|------|------|------|-----|
| `Rc<T>` | No | No | Non-atomic reference count |
| `RefCell<T>` | Yes | No | Non-thread-safe borrow checking |
| `Cell<T>` | Yes | No | Non-thread-safe interior mutability |
| `UnsafeCell<T>` | Yes | No | Foundation of interior mutability |
| `MutexGuard<T>` | No | Yes | Must unlock on same thread |
| Raw pointers | No | No | No safety guarantees |

### Manual implementation (unsafe)

```rust
struct MyBox(*mut u8);

unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}
```

Only needed for types containing raw pointers or other non-Send/Sync components where you can guarantee safety.

## Data Races vs Race Conditions

For detailed examples and unsafe pitfalls, see [references/races-and-atomics.md](references/races-and-atomics.md).

**Data race** (impossible in safe Rust): two threads access same memory concurrently, at least one writes, no synchronization. This is **Undefined Behavior**.

**Race condition** (possible in safe Rust): logic error from non-deterministic ordering. Cannot cause memory unsafety in safe code but can cause panics or wrong results:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

let data = vec![1, 2, 3, 4];
let idx = Arc::new(AtomicUsize::new(0));
let other_idx = idx.clone();

thread::spawn(move || { other_idx.fetch_add(10, Ordering::SeqCst); });

// May panic (index out of bounds) but cannot corrupt memory
println!("{}", data[idx.load(Ordering::SeqCst)]);
```

## Atomics and Memory Orderings

For full details on compiler/hardware reordering and the spinlock example, see [references/races-and-atomics.md](references/races-and-atomics.md).

Rust inherits its atomics memory model from C++20. Four orderings exposed:

| Ordering | Guarantees | Cost | Use case |
|----------|-----------|------|----------|
| `SeqCst` | Global total order, no reordering | Highest (memory fences on all platforms) | Default safe choice |
| `Release` | All prior writes visible to paired `Acquire` | Low on x86 | Store side of lock/flag |
| `Acquire` | Sees all writes before paired `Release` | Low on x86 | Load side of lock/flag |
| `Relaxed` | Only atomicity, no ordering | Lowest | Counters, statistics |

- `Acquire`/`Release` always come in pairs on the **same memory location**
- `SeqCst` is the safe default -- easy to downgrade to weaker orderings later
- On strongly-ordered platforms (x86/64), `Acquire`/`Release` are often free
- On weakly-ordered platforms (ARM), relaxed orderings provide real savings

## Official Documentation

- [Concurrency (The Rust Book)](https://doc.rust-lang.org/book/ch16-00-concurrency.html)
- [Threads](https://doc.rust-lang.org/book/ch16-01-threads.html)
- [Message passing](https://doc.rust-lang.org/book/ch16-02-message-passing.html)
- [Shared state](https://doc.rust-lang.org/book/ch16-03-shared-state.html)
- [Send and Sync](https://doc.rust-lang.org/book/ch16-04-extensible-concurrency-sync-and-send.html)
- [Concurrency (Nomicon)](https://doc.rust-lang.org/nomicon/concurrency.html)
- [Races (Nomicon)](https://doc.rust-lang.org/nomicon/races.html)
- [Send and Sync (Nomicon)](https://doc.rust-lang.org/nomicon/send-and-sync.html)
- [Atomics (Nomicon)](https://doc.rust-lang.org/nomicon/atomics.html)
