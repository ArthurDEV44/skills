# Send and Sync -- Deep Dive

## Table of Contents

- [Trait Definitions](#trait-definitions)
- [Unsafe Marker Traits](#unsafe-marker-traits)
- [Why Exceptions Exist](#why-exceptions-exist)
- [Manual Implementation](#manual-implementation)
- [Negative Implementations](#negative-implementations)
- [Complete Example: Carton<T>](#complete-example-cartont)
- [MutexGuard Counter-Example](#mutexguard-counter-example)

## Trait Definitions

- **Send**: Safe to transfer ownership to another thread.
- **Sync**: Safe to share references between threads. `T` is `Sync` if and only if `&T` is `Send`.

Both are **unsafe marker traits** (no methods). Incorrectly implementing them can cause Undefined Behavior.

## Unsafe Marker Traits

- `Send` and `Sync` are automatically derived: if all fields are `Send`/`Sync`, the type is too
- Almost all primitives are both `Send` and `Sync`
- Other unsafe code may **assume** these traits are correctly implemented
- Incorrect implementation leads to UB, not just logic errors

## Why Exceptions Exist

**Raw pointers** (`*const T`, `*mut T`): Neither Send nor Sync. Act as a lint -- types containing raw pointers aren't automatically thread-safe. Most raw pointer usage should be encapsulated behind sufficient abstraction.

**`UnsafeCell<T>`**: Not Sync. Foundation of all interior mutability (`Cell`, `RefCell`). Allows unsynchronized mutation through shared references.

**`Rc<T>`**: Neither Send nor Sync. Shared, unsynchronized reference count -- incrementing/decrementing from multiple threads would be a data race.

## Manual Implementation

```rust
struct MyBox(*mut u8);

// Safety: We own the pointer exclusively, so sending to another thread is safe
unsafe impl Send for MyBox {}

// Safety: No interior mutability through &MyBox, so sharing references is safe
unsafe impl Sync for MyBox {}
```

All Rust standard collections are Send and Sync (when containing Send and Sync types) despite pervasive raw pointer use internally.

## Negative Implementations

Opt out of auto-derived Send/Sync (nightly only):

```rust
#![feature(negative_impls)]

struct SpecialThreadToken(u8);

impl !Send for SpecialThreadToken {}
impl !Sync for SpecialThreadToken {}
```

## Complete Example: Carton<T>

A heap-allocated type using raw pointers with correct Send/Sync implementation.

### Construction

```rust
use std::{mem::{align_of, size_of}, ptr, cmp::max};

pub mod libc {
    pub use ::std::os::raw::{c_int, c_void};
    pub type size_t = usize;
    unsafe extern "C" {
        pub fn posix_memalign(memptr: *mut *mut c_void, align: size_t, size: size_t) -> c_int;
    }
}

struct Carton<T>(ptr::NonNull<T>);

impl<T> Carton<T> {
    pub fn new(value: T) -> Self {
        assert_ne!(size_of::<T>(), 0, "ZSTs not supported");
        let mut memptr: *mut T = ptr::null_mut();
        unsafe {
            let ret = libc::posix_memalign(
                (&mut memptr as *mut *mut T).cast(),
                max(align_of::<T>(), size_of::<usize>()),
                size_of::<T>(),
            );
            assert_eq!(ret, 0, "Failed to allocate or invalid alignment");
        };
        let ptr = ptr::NonNull::new(memptr).expect("Guaranteed non-null if ret == 0");
        unsafe { ptr.as_ptr().write(value); }
        Self(ptr)
    }
}
```

### Deref / DerefMut

```rust
use std::ops::{Deref, DerefMut};

impl<T> Deref for Carton<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for Carton<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.0.as_mut() }
    }
}
```

### Send -- safe if T is Send

```rust
// Safety: We exclusively own the pointer. Transferring Carton to another
// thread is safe as long as T itself can be transferred.
unsafe impl<T> Send for Carton<T> where T: Send {}
```

### Sync -- safe if T is Sync

```rust
// Safety: &Carton<T> provides &T via Deref (unsynchronized).
// Carton has no interior mutability, so T: Sync suffices.
unsafe impl<T> Sync for Carton<T> where T: Sync {}
```

### Drop

```rust
mod libc {
    pub use ::std::os::raw::c_void;
    unsafe extern "C" { pub fn free(p: *mut c_void); }
}

impl<T> Drop for Carton<T> {
    fn drop(&mut self) {
        unsafe { libc::free(self.0.as_ptr().cast()); }
    }
}
```

### Alternative: delegate to Box bounds

```rust
// Instead of reasoning about raw pointer safety directly:
unsafe impl<T> Send for Carton<T> where Box<T>: Send {}
```

## MutexGuard Counter-Example

`MutexGuard<T>` is **not Send** but **is Sync**:

- **Not Send**: Underlying OS primitives require a lock to be released on the same thread that acquired it. Sending a `MutexGuard` to another thread would cause the destructor to run on a different thread.
- **Is Sync**: Sending `&MutexGuard` to another thread is fine -- dropping a reference does nothing, so the lock is never released from the wrong thread.
