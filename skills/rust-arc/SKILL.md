---
name: rust-arc
description: >
  Rust Arc (Atomically Reference Counted) smart pointer for thread-safe shared ownership.
  Covers construction, cloning, mutation strategies (Mutex, RwLock, atomics, make_mut),
  Weak references for cycle breaking, reference counting, unwrapping, raw pointer conversion,
  and trait implementations. Use when writing, reviewing, or refactoring Rust code involving
  shared ownership across threads: (1) Sharing data between threads with Arc, (2) Choosing
  between Arc and Rc, (3) Combining Arc with Mutex or RwLock for interior mutability,
  (4) Using Weak references to break reference cycles, (5) Managing Arc reference counts,
  (6) Converting Arc to/from raw pointers, (7) Using Arc with tokio::spawn or thread::spawn.
---

# Rust Arc

`Arc<T>` = **Atomically Reference Counted** pointer. Thread-safe shared ownership of heap-allocated data.

## Construction & Cloning

```rust
use std::sync::Arc;

let data = Arc::new(vec![1, 2, 3]);
let clone = Arc::clone(&data);  // prefer Arc::clone() over data.clone() for clarity

assert_eq!(Arc::strong_count(&data), 2);
```

`Arc::clone()` increments the reference count (cheap, no data copy). Value is dropped when the last `Arc` is dropped.

## Sharing Across Threads

```rust
use std::sync::Arc;
use std::thread;

let shared = Arc::new("hello");

for _ in 0..10 {
    let shared = Arc::clone(&shared);
    thread::spawn(move || {
        println!("{shared}");
    });
}
```

`Arc<T>` is `Send + Sync` when `T: Send + Sync`. Use `Rc<T>` instead for single-threaded code (no atomic overhead).

## Mutation Strategies

Arc provides **immutable** access by default. Four approaches for mutation:

### 1. Arc + Mutex (most common)

```rust
use std::sync::{Arc, Mutex};

let data = Arc::new(Mutex::new(vec![1, 2, 3]));
let data_clone = Arc::clone(&data);

thread::spawn(move || {
    let mut vec = data_clone.lock().unwrap();
    vec.push(4);
});
```

Use `RwLock` instead of `Mutex` when reads are frequent and writes are rare.

### 2. Arc + Atomic types (lock-free)

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

let counter = Arc::new(AtomicUsize::new(0));
let c = Arc::clone(&counter);

thread::spawn(move || {
    c.fetch_add(1, Ordering::SeqCst);
});
```

### 3. make_mut (clone-on-write)

```rust
let mut data = Arc::new(vec![1, 2, 3]);
// Clones inner data only if other Arc references exist
Arc::make_mut(&mut data).push(4);
```

If this is the only reference, mutates in place. Otherwise clones the data first.

### 4. get_mut (exclusive access)

```rust
let mut data = Arc::new(5);
// Returns Some(&mut T) only if no other Arc or Weak references exist
if let Some(val) = Arc::get_mut(&mut data) {
    *val = 10;
}
```

## Weak References

`Weak<T>` prevents reference cycles that cause memory leaks. Weak references don't keep the value alive.

```rust
use std::sync::{Arc, Weak};

let strong = Arc::new(42);
let weak: Weak<i32> = Arc::downgrade(&strong);

assert_eq!(Arc::weak_count(&strong), 1);

// Upgrade Weak to Arc (returns None if value was dropped)
if let Some(val) = weak.upgrade() {
    println!("{val}");
}

drop(strong);
assert!(weak.upgrade().is_none()); // value is gone
```

### Self-referential / Cyclic Structures

```rust
use std::sync::{Arc, Weak};

struct Node {
    parent: Weak<Node>,      // weak to prevent cycle
    children: Vec<Arc<Node>>, // strong for ownership
}

// new_cyclic provides a Weak before construction completes
let node = Arc::new_cyclic(|me: &Weak<Node>| {
    Node {
        parent: me.clone(),
        children: vec![],
    }
});
```

## Reference Counting

```rust
let a = Arc::new("data");
let b = Arc::clone(&a);
let w = Arc::downgrade(&a);

Arc::strong_count(&a) // 2
Arc::weak_count(&a)   // 1
Arc::ptr_eq(&a, &b)   // true (same allocation)
```

## Unwrapping

```rust
// try_unwrap: extract value if sole owner
let x = Arc::new(5);
assert_eq!(Arc::try_unwrap(x), Ok(5));

let x = Arc::new(5);
let _y = Arc::clone(&x);
assert!(Arc::try_unwrap(x).is_err()); // fails, returns Arc back

// into_inner: returns Option, always consumes Arc
let x = Arc::new(5);
assert_eq!(Arc::into_inner(x), Some(5));
```

## Raw Pointer Conversion

```rust
let original = Arc::new("hello".to_string());

// Convert to raw pointer (does NOT decrement count)
let ptr: *const String = Arc::into_raw(original);

// Reconstruct Arc from raw (unsafe, must maintain invariants)
let restored = unsafe { Arc::from_raw(ptr) };
assert_eq!(*restored, "hello");
```

## Key Trait Implementations

| Trait | Behavior |
|-------|----------|
| `Clone` | Increments strong count (cheap) |
| `Deref` | Auto-derefs to `&T` |
| `Drop` | Decrements count; drops value when last ref gone |
| `Send + Sync` | Only when `T: Send + Sync` |
| `From<T>` | `Arc::from(value)` |
| `From<Box<T>>` | Convert Box to Arc |
| `From<String>` | Produces `Arc<str>` |
| `From<Vec<T>>` | Produces `Arc<[T]>` |
| `Display, Debug` | Delegates to inner `T` |
| `PartialEq, Eq, Hash, Ord` | Delegates to inner `T` |

## Arc vs Rc

| | `Arc<T>` | `Rc<T>` |
|---|----------|---------|
| Thread-safe | Yes (atomic ops) | No |
| Performance | Slightly slower (atomics) | Faster |
| Implements | `Send + Sync` | Neither |
| Use when | Multi-threaded sharing | Single-threaded sharing |

## Common Patterns

```rust
// Shared state in async (tokio::spawn requires 'static + Send)
let state = Arc::new(Mutex::new(HashMap::new()));
let state_clone = Arc::clone(&state);
tokio::spawn(async move {
    state_clone.lock().unwrap().insert("key", "val");
});

// Type alias for readability
type SharedDb = Arc<Mutex<HashMap<String, Vec<u8>>>>;
```

## Official Documentation

- [std::sync::Arc](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- [Arc - Rust by Example](https://doc.rust-lang.org/rust-by-example/std/arc.html)
