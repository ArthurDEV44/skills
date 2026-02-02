# Races and Atomics -- Deep Dive

## Table of Contents

- [Data Races](#data-races)
- [Race Conditions](#race-conditions)
- [Unsafe + Race Condition = UB](#unsafe--race-condition--ub)
- [Compiler Reordering](#compiler-reordering)
- [Hardware Reordering](#hardware-reordering)
- [Data Accesses vs Atomic Accesses](#data-accesses-vs-atomic-accesses)
- [Memory Orderings in Detail](#memory-orderings-in-detail)
- [Spinlock Example](#spinlock-example)

## Data Races

A data race occurs when ALL three conditions hold:
1. Two or more threads concurrently access the same memory location
2. At least one access is a write
3. No synchronization

Data races are **Undefined Behavior**. Safe Rust prevents them through the ownership system: you cannot alias a mutable reference.

## Race Conditions

A race condition is a logic error caused by non-deterministic ordering. Safe Rust **cannot** prevent them (mathematically impossible without controlling the scheduler).

A race condition alone **cannot violate memory safety** in safe Rust. Programs can safely deadlock or behave nonsensically with incorrect synchronization.

### Safe example -- may panic but no UB

```rust
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let data = vec![1, 2, 3, 4];
let idx = Arc::new(AtomicUsize::new(0));
let other_idx = idx.clone();

thread::spawn(move || {
    other_idx.fetch_add(10, Ordering::SeqCst);
});

// Bounds check is safe: indexing panics on out-of-bounds, no memory corruption
println!("{}", data[idx.load(Ordering::SeqCst)]);
```

## Unsafe + Race Condition = UB

```rust
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let data = vec![1, 2, 3, 4];
let idx = Arc::new(AtomicUsize::new(0));
let other_idx = idx.clone();

thread::spawn(move || {
    other_idx.fetch_add(10, Ordering::SeqCst);
});

if idx.load(Ordering::SeqCst) < data.len() {
    unsafe {
        // BUG: idx can change between the check and the access (TOCTOU)
        // Race condition + unsafe = Undefined Behavior
        println!("{}", data.get_unchecked(idx.load(Ordering::SeqCst)));
    }
}
```

## Compiler Reordering

Compilers reorder instructions when single-threaded behavior is preserved:

```text
x = 1;       x = 2;
y = 3;  -->  y = 3;
x = 2;
```

Unobservable single-threaded, but breaks multi-threaded assumptions about ordering.

## Hardware Reordering

CPUs with caches/memory hierarchies may propagate writes to shared memory lazily:

```text
initial: x = 0, y = 1

THREAD 1        THREAD 2
y = 3;          if x == 1 {
x = 1;              y *= 2;
                }
```

Expected final states: `y = 3` or `y = 6`.
Possible hardware state: `y = 2` (thread 2 sees `x = 1` but not `y = 3` yet).

**Strongly-ordered** hardware (x86/64) provides strong guarantees by default.
**Weakly-ordered** hardware (ARM) provides weaker guarantees; relaxed orderings save more.

## Data Accesses vs Atomic Accesses

**Data accesses**: Fundamentally unsynchronized. Compiler assumes single-threaded execution. Hardware propagates lazily. It is **impossible** to write correct synchronized code using only data accesses.

**Atomic accesses**: Tell compiler and hardware the program is multi-threaded. Each has an ordering that restricts allowed optimizations.

## Memory Orderings in Detail

### Sequentially Consistent (SeqCst)

- Most restrictive ordering
- Operations cannot be reordered past SeqCst operations in either direction
- All threads agree on a single global order of SeqCst operations
- Requires memory fences even on strongly-ordered platforms
- **Use as default** when unsure; easy to downgrade later

### Acquire-Release

Designed to be paired for synchronization.

**Acquire** (on loads):
- All accesses after the acquire stay after it
- Accesses before the acquire **may** reorder to after it

**Release** (on stores):
- All accesses before the release stay before it
- Accesses after the release **may** reorder to before it

**Causality**: When thread A does a Release store and thread B does an Acquire load on the **same location**, B sees all writes (atomic, relaxed, or non-atomic) that happened before A's Release.

Requirements for causality to hold:
- Must use the **same memory location**
- Must be a Release-Acquire pair between the specific threads

On strongly-ordered platforms (x86/64), Acquire-Release is often free (hardware provides it by default).

### Relaxed

- Weakest ordering -- can be freely reordered
- No happens-before relationship established
- Still atomic: read-modify-write operations happen atomically
- Use for counters, statistics, or anything that only needs atomicity
- Minimal benefit on x86 (already provides Acquire-Release semantics)
- Can be cheaper on weakly-ordered platforms (ARM)

## Spinlock Example

Acquire-Release used to build a simple spinlock:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

fn main() {
    let lock = Arc::new(AtomicBool::new(false)); // false = unlocked

    // ... distribute lock to threads ...

    // Acquire the lock
    while lock.compare_and_swap(false, true, Ordering::Acquire) {}
    // Lock acquired -- all writes after the previous Release are visible here

    // ... critical section: access shared data ...

    // Release the lock
    lock.store(false, Ordering::Release);
    // All writes before this Release will be visible to the next Acquire
}
```

- `Acquire` on lock acquisition ensures we see all writes from the previous lock holder
- `Release` on unlock ensures the next acquirer sees our writes
- This pattern is the foundation of all lock-based synchronization
