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

For the complete API with all method signatures and stability versions, see `references/api-reference.md`.
For advanced real-world patterns, see `references/patterns.md`.

## Construction & Cloning

```rust
use std::sync::Arc;

let data = Arc::new(vec![1, 2, 3]);
let clone = Arc::clone(&data);  // prefer Arc::clone() over data.clone() for clarity

assert_eq!(Arc::strong_count(&data), 2);
```

`Arc::clone()` increments the reference count (cheap, no data copy). Value is dropped when the last `Arc` is dropped.

### Pinned Construction

```rust
use std::sync::Arc;
use std::pin::Pin;

let pinned: Pin<Arc<String>> = Arc::pin("hello".to_string());
```

### Uninitialized Construction (since 1.82.0)

```rust
use std::sync::Arc;

let mut five = Arc::<u32>::new_uninit();
Arc::get_mut(&mut five).unwrap().write(5);
let five = unsafe { five.assume_init() };
assert_eq!(*five, 5);

// Slices too
let mut values = Arc::<[u32]>::new_uninit_slice(3);
let data = Arc::get_mut(&mut values).unwrap();
data[0].write(1);
data[1].write(2);
data[2].write(3);
let values = unsafe { values.assume_init() };
assert_eq!(*values, [1, 2, 3]);
```

### Self-Referential Construction (since 1.60.0)

```rust
use std::sync::{Arc, Weak};

struct Gadget {
    me: Weak<Gadget>,
}

impl Gadget {
    fn new() -> Arc<Self> {
        Arc::new_cyclic(|me| Gadget { me: me.clone() })
    }
}
```

`new_cyclic` provides a `Weak<T>` to the closure before the `Arc` is fully constructed. Calling `upgrade()` on this `Weak` inside the closure returns `None`.

## Sharing Across Threads

```rust
use std::sync::Arc;
use std::thread;

let shared = Arc::new("hello");

let mut handles = vec![];
for _ in 0..10 {
    let shared = Arc::clone(&shared);
    handles.push(thread::spawn(move || {
        println!("{shared}");
    }));
}
for h in handles { h.join().unwrap(); }
```

`Arc<T>` is `Send + Sync` when `T: Send + Sync`. Use `Rc<T>` instead for single-threaded code (no atomic overhead).

## Mutation Strategies

Arc provides **immutable** access by default. Four approaches for mutation:

### 1. Arc + Mutex (most common)

```rust
use std::sync::{Arc, Mutex};
use std::thread;

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
use std::thread;

let counter = Arc::new(AtomicUsize::new(0));
let c = Arc::clone(&counter);

thread::spawn(move || {
    c.fetch_add(1, Ordering::SeqCst);
});
```

### 3. make_mut (clone-on-write, since 1.4.0)

```rust
use std::sync::Arc;

let mut data = Arc::new(5);
*Arc::make_mut(&mut data) += 1;         // no clone (sole owner)

let other = Arc::clone(&data);
*Arc::make_mut(&mut data) += 1;         // clones inner data (count was 2)
*Arc::make_mut(&mut data) += 1;         // no clone (sole owner again)

assert_eq!(*data, 8);
assert_eq!(*other, 6);
```

If this is the only strong reference, mutates in place. If other strong refs exist, clones the data first. **Important:** if only Weak references exist (no other strong), `make_mut` dissociates them (existing Weak pointers will return `None` on `upgrade()`).

### 4. get_mut (exclusive access, since 1.4.0)

```rust
use std::sync::Arc;

let mut data = Arc::new(5);
// Returns Some(&mut T) only if no other Arc or Weak references exist
if let Some(val) = Arc::get_mut(&mut data) {
    *val = 10;
}
```

Unlike `make_mut`, this never clones — it returns `None` if any other references exist.

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

### Empty Weak (no allocation)

```rust
use std::sync::Weak;

let empty: Weak<i64> = Weak::new();
assert!(empty.upgrade().is_none()); // always None
```

### Cyclic Structures (parent/child pattern)

```rust
use std::sync::{Arc, Weak, Mutex};

struct Node {
    value: i32,
    parent: Weak<Node>,         // weak to prevent cycle
    children: Mutex<Vec<Arc<Node>>>, // strong for ownership
}

let root = Arc::new_cyclic(|me: &Weak<Node>| {
    Node {
        value: 0,
        parent: me.clone(),
        children: Mutex::new(vec![]),
    }
});
```

## Reference Counting & Comparison

```rust
use std::sync::Arc;

let a = Arc::new("data");
let b = Arc::clone(&a);
let w = Arc::downgrade(&a);

assert_eq!(Arc::strong_count(&a), 2);
assert_eq!(Arc::weak_count(&a), 1);
assert!(Arc::ptr_eq(&a, &b)); // same allocation (not value equality)
```

**Note:** `strong_count` and `weak_count` are not atomic snapshots — other threads may change the count between calling and observing the result.

## Unwrapping & Extraction

```rust
use std::sync::Arc;

// try_unwrap: extract value if sole strong owner (since 1.4.0)
let x = Arc::new(5);
assert_eq!(Arc::try_unwrap(x), Ok(5));

let x = Arc::new(5);
let _y = Arc::clone(&x);
assert!(Arc::try_unwrap(x).is_err()); // fails, returns Arc back in Err

// into_inner: returns Option, always consumes Arc (since 1.70.0)
// Preferred over try_unwrap when you don't need the Arc back on failure
let x = Arc::new(5);
assert_eq!(Arc::into_inner(x), Some(5));

// unwrap_or_clone: unwrap if sole owner, otherwise clone inner (since 1.76.0)
let x = Arc::new(vec![1, 2, 3]);
let y = Arc::clone(&x);
let vec = Arc::unwrap_or_clone(x); // clones because y still exists
```

**Prefer `into_inner` over `try_unwrap(...).ok()`** — the latter can silently drop the value on race conditions in multi-threaded code.

## Raw Pointer Conversion

```rust
use std::sync::Arc;

let original = Arc::new("hello".to_string());

// Convert to raw pointer (does NOT decrement count)
let ptr: *const String = Arc::into_raw(original);

// Reconstruct Arc from raw (unsafe, must maintain invariants)
let restored = unsafe { Arc::from_raw(ptr) };
assert_eq!(*restored, "hello");
```

For FFI interop, see also `increment_strong_count` / `decrement_strong_count` in `references/api-reference.md`.

## Key Trait Implementations

| Trait | Behavior |
|-------|----------|
| `Clone` | Increments strong count (cheap, no data copy) |
| `Deref` | Auto-derefs to `&T` |
| `Drop` | Decrements count; drops value when last strong ref gone |
| `Send` | When `T: Send + Sync` |
| `Sync` | When `T: Send + Sync` |
| `From<T>` | `Arc::from(value)` |
| `From<Box<T>>` | Convert Box to Arc |
| `From<String>` | Produces `Arc<str>` |
| `From<Vec<T>>` | Produces `Arc<[T]>` |
| `From<[T; N]>` | Produces `Arc<[T]>` |
| `From<CString>` | Produces `Arc<CStr>` |
| `FromIterator<T>` | Collects into `Arc<[T]>` |
| `Display, Debug` | Delegates to inner `T` |
| `PartialEq, Eq, Hash, Ord` | Compares/hashes inner `T` (not pointer) |
| `Borrow<T>, AsRef<T>` | Borrows inner `T` |
| `Read, Write, Seek` | For `Arc<File>` (shared file handles) |
| `Unpin` | Always Unpin regardless of T |
| `UnwindSafe` | Always (even if T is not) |
| `Error` | When `T: Error` |

## Arc vs Rc

| | `Arc<T>` | `Rc<T>` |
|---|----------|---------|
| Thread-safe | Yes (atomic ops) | No |
| Performance | Slightly slower (atomics) | Faster |
| Implements | `Send + Sync` | Neither |
| Weak type | `std::sync::Weak<T>` | `std::rc::Weak<T>` |
| Use when | Multi-threaded sharing | Single-threaded sharing |

**Rule of thumb:** start with `Rc` in single-threaded code, switch to `Arc` when you need to cross thread boundaries.

## Common Pitfalls

1. **Deadlock with `Arc<Mutex<T>>`**: acquiring the same mutex in multiple threads in different orders.
2. **Memory leaks from cycles**: two `Arc`s pointing to each other. Use `Weak` for back-references.
3. **Using `.clone()` instead of `Arc::clone()`**: works but obscures intent — `Arc::clone()` makes it clear you're cloning the pointer, not the data.
4. **Forgetting `into_raw` leaks**: every `Arc::into_raw` must be paired with `Arc::from_raw` or the allocation leaks.
5. **`try_unwrap(...).ok()` in threaded code**: use `into_inner` instead to avoid silent data loss from race conditions.
6. **`strong_count` for synchronization**: counts are not atomic snapshots — don't branch on them for correctness.
