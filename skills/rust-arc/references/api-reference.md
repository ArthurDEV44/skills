# Arc<T> Complete API Reference

All methods listed with stability version. Nightly-only methods are marked. Methods use associated function syntax: `Arc::method(arc, ...)`.

## Construction

| Method | Since | Signature |
|--------|-------|-----------|
| `new` | 1.0.0 | `fn new(data: T) -> Arc<T>` |
| `new_cyclic` | 1.60.0 | `fn new_cyclic<F>(data_fn: F) -> Arc<T>` where `F: FnOnce(&Weak<T>) -> T` |
| `pin` | 1.0.0 | `fn pin(data: T) -> Pin<Arc<T>>` |
| `new_uninit` | 1.82.0 | `fn new_uninit() -> Arc<MaybeUninit<T>>` |
| `new_zeroed` | 1.92.0 | `fn new_zeroed() -> Arc<MaybeUninit<T>>` |
| `new_uninit_slice` | 1.82.0 | `fn new_uninit_slice(len: usize) -> Arc<[MaybeUninit<T>]>` |
| `new_zeroed_slice` | 1.92.0 | `fn new_zeroed_slice(len: usize) -> Arc<[MaybeUninit<T>]>` |

### Nightly: Allocator-Aware Construction

| Method | Signature |
|--------|-----------|
| `new_in` | `fn new_in(data: T, alloc: A) -> Arc<T, A>` |
| `new_cyclic_in` | `fn new_cyclic_in<F>(data_fn: F, alloc: A) -> Arc<T, A>` |
| `pin_in` | `fn pin_in(data: T, alloc: A) -> Pin<Arc<T, A>>` where `A: 'static` |
| `new_uninit_in` | `fn new_uninit_in(alloc: A) -> Arc<MaybeUninit<T>, A>` |
| `new_zeroed_in` | `fn new_zeroed_in(alloc: A) -> Arc<MaybeUninit<T>, A>` |

### Nightly: Fallible Construction

| Method | Signature |
|--------|-----------|
| `try_new` | `fn try_new(data: T) -> Result<Arc<T>, AllocError>` |
| `try_new_uninit` | `fn try_new_uninit() -> Result<Arc<MaybeUninit<T>>, AllocError>` |
| `try_new_zeroed` | `fn try_new_zeroed() -> Result<Arc<MaybeUninit<T>>, AllocError>` |
| `try_pin` | `fn try_pin(data: T) -> Result<Pin<Arc<T>>, AllocError>` |

## Unwrapping & Extraction

| Method | Since | Signature | Notes |
|--------|-------|-----------|-------|
| `try_unwrap` | 1.4.0 | `fn try_unwrap(this: Arc<T>) -> Result<T, Arc<T>>` | Returns `Err(arc)` if not sole owner |
| `into_inner` | 1.70.0 | `fn into_inner(this: Arc<T>) -> Option<T>` | Preferred over `try_unwrap` — no race condition issues |
| `unwrap_or_clone` | 1.76.0 | `fn unwrap_or_clone(this: Arc<T>) -> T` where `T: Clone` | Unwraps if sole owner, clones otherwise |

### `try_unwrap` vs `into_inner`

`into_inner` is preferred when you don't need the Arc back on failure. It guarantees that across all threads, exactly one caller gets `Some(T)` and all others get `None`. Using `Arc::try_unwrap(x).ok()` can silently drop the inner value in multi-threaded code when two threads race.

## Reference Counting

| Method | Since | Signature | Notes |
|--------|-------|-----------|-------|
| `strong_count` | 1.15.0 | `fn strong_count(this: &Arc<T>) -> usize` | Not an atomic snapshot |
| `weak_count` | 1.15.0 | `fn weak_count(this: &Arc<T>) -> usize` | Not an atomic snapshot |
| `ptr_eq` | 1.17.0 | `fn ptr_eq(this: &Arc<T>, other: &Arc<T>) -> bool` | Compares pointers, ignores dyn metadata |

## Mutable Access

| Method | Since | Signature | Behavior |
|--------|-------|-----------|----------|
| `get_mut` | 1.4.0 | `fn get_mut(this: &mut Arc<T>) -> Option<&mut T>` | `Some` only if sole strong ref AND no weak refs |
| `make_mut` | 1.4.0 | `fn make_mut(this: &mut Arc<T>) -> &mut T` where `T: Clone` | Clone-on-write; dissociates weak refs if no other strong refs |
| `get_mut_unchecked` | nightly | `unsafe fn get_mut_unchecked(this: &mut Arc<T>) -> &mut T` | UB if other refs exist |

### `get_mut` vs `make_mut` behavior matrix

| Other strong refs? | Weak refs? | `get_mut` | `make_mut` |
|--------------------|------------|-----------|------------|
| No | No | `Some(&mut T)` — mutates in place | `&mut T` — mutates in place |
| No | Yes | `None` | `&mut T` — dissociates weak refs, mutates in place |
| Yes | Any | `None` | `&mut T` — clones data to new allocation |

## Weak Pointer Operations

| Method | Since | Signature |
|--------|-------|-----------|
| `downgrade` | 1.4.0 | `fn downgrade(this: &Arc<T>) -> Weak<T>` where `A: Clone` |

## Raw Pointer Operations

| Method | Since | Signature | Safety |
|--------|-------|-----------|--------|
| `into_raw` | 1.17.0 | `fn into_raw(this: Arc<T>) -> *const T` | Safe — consumes Arc, no count change |
| `from_raw` | 1.17.0 | `unsafe fn from_raw(ptr: *const T) -> Arc<T>` | Ptr must come from `into_raw` |
| `as_ptr` | 1.45.0 | `fn as_ptr(this: &Arc<T>) -> *const T` | Safe — doesn't consume Arc |
| `increment_strong_count` | 1.51.0 | `unsafe fn increment_strong_count(ptr: *const T)` | Ptr must come from `into_raw` |
| `decrement_strong_count` | 1.51.0 | `unsafe fn decrement_strong_count(ptr: *const T)` | Drops value if count reaches 0 |

### FFI Pattern with `increment/decrement_strong_count`

```rust
use std::sync::Arc;

let five = Arc::new(5);

// Pass to C code as opaque pointer
let ptr = Arc::into_raw(five);

// C code wants another reference — increment before handing out
unsafe { Arc::increment_strong_count(ptr); }

// When C code is done with its copy — decrement
unsafe { Arc::decrement_strong_count(ptr); }

// Reconstruct our original Arc
let five = unsafe { Arc::from_raw(ptr) };
assert_eq!(*five, 5);
```

## MaybeUninit Operations

| Method | Since | Signature | Safety |
|--------|-------|-----------|--------|
| `assume_init` | 1.82.0 | `unsafe fn assume_init(self) -> Arc<T>` | Caller must guarantee init |
| `assume_init` (slice) | 1.82.0 | `unsafe fn assume_init(self) -> Arc<[T]>` | All elements must be init |

## Nightly-Only Methods

| Method | Signature | Notes |
|--------|-----------|-------|
| `clone_from_ref` | `fn clone_from_ref(value: &T) -> Arc<T>` where `T: CloneToUninit` | Clones unsized values into Arc |
| `map` | `fn map<U>(this: Arc<T>, f: impl FnOnce(&T) -> U) -> Arc<U>` | Functional transform |
| `try_map` | `fn try_map<R>(this: Arc<T>, f: impl FnOnce(&T) -> R) -> R::TryType` | Fallible map |
| `into_array` | `fn into_array<const N: usize>(self) -> Option<Arc<[T; N]>>` | Slice to array (no realloc) |
| `allocator` | `fn allocator(this: &Arc<T, A>) -> &A` | Access underlying allocator |

---

# Weak<T> Complete API Reference

## Construction & Core

| Method | Since | Signature | Notes |
|--------|-------|-----------|-------|
| `Weak::new` | 1.10.0 | `const fn new() -> Weak<T>` | No allocation; `upgrade()` always returns `None` |
| `upgrade` | 1.4.0 | `fn upgrade(&self) -> Option<Arc<T>>` | `None` if all strong refs dropped |
| `strong_count` | 1.41.0 | `fn strong_count(&self) -> usize` | 0 if from `Weak::new()` or all strong dropped |
| `weak_count` | 1.41.0 | `fn weak_count(&self) -> usize` | Approximate — off by 1 possible |
| `ptr_eq` | 1.39.0 | `fn ptr_eq(&self, other: &Weak<T>) -> bool` | Two `Weak::new()` compare equal |

## Raw Pointer Operations

| Method | Since | Signature | Safety |
|--------|-------|-----------|--------|
| `as_ptr` | 1.45.0 | `fn as_ptr(&self) -> *const T` | May be dangling if all strong dropped |
| `into_raw` | 1.45.0 | `fn into_raw(self) -> *const T` | Consumes Weak, preserves weak count |
| `from_raw` | 1.45.0 | `unsafe fn from_raw(ptr: *const T) -> Weak<T>` | Ptr must come from `into_raw` |

## Weak Trait Implementations

| Trait | Since | Notes |
|-------|-------|-------|
| `Clone` | 1.4.0 | Increments weak count |
| `Debug` | 1.4.0 | Prints `(Weak)` |
| `Default` | 1.10.0 | Same as `Weak::new()` |
| `Drop` | 1.4.0 | Decrements weak count; deallocates backing memory if last weak + strong |
| `Send` | 1.4.0 | When `T: Send + Sync` |
| `Sync` | 1.4.0 | When `T: Send + Sync` |

---

# Complete Arc Trait Implementation Table

## Standard Traits

| Trait | Bounds | Since | Notes |
|-------|--------|-------|-------|
| `Clone` | `A: Allocator + Clone` | 1.0.0 | Increments strong count |
| `Deref` | — | 1.0.0 | `Target = T` |
| `Drop` | — | 1.0.0 | Decrements count, drops T at 0 |
| `Display` | `T: Display` | 1.0.0 | Delegates to T |
| `Debug` | `T: Debug` | 1.0.0 | Delegates to T |
| `Pointer` | — | 1.0.0 | Prints pointer address |
| `PartialEq` | `T: PartialEq` | 1.0.0 | Compares inner values |
| `Eq` | `T: Eq` | 1.0.0 | |
| `PartialOrd` | `T: PartialOrd` | 1.0.0 | Compares inner values |
| `Ord` | `T: Ord` | 1.0.0 | |
| `Hash` | `T: Hash` | 1.0.0 | Hashes inner value |
| `Default` | `T: Default` | 1.0.0 | `Arc::new(T::default())` |
| `Error` | `T: Error` | 1.0.0 | Delegates to T |

## Marker Traits

| Trait | Bounds | Notes |
|-------|--------|-------|
| `Send` | `T: Send + Sync` | Safe to send across threads |
| `Sync` | `T: Send + Sync` | Safe to share references across threads |
| `Unpin` | always | Regardless of T |
| `UnwindSafe` | always | |
| `RefUnwindSafe` | always | |

## Conversion Traits

| From | To | Since | Notes |
|------|----|-------|-------|
| `T` | `Arc<T>` | 1.0.0 | `Arc::from(value)` |
| `Box<T>` | `Arc<T>` | 1.21.0 | Moves from heap-to-heap |
| `String` | `Arc<str>` | 1.21.0 | |
| `&str` | `Arc<str>` | 1.21.0 | |
| `Vec<T>` | `Arc<[T]>` | 1.21.0 | |
| `&[T]` | `Arc<[T]>` where `T: Clone` | 1.21.0 | |
| `[T; N]` | `Arc<[T]>` | 1.56.0 | |
| `CString` | `Arc<CStr>` | 1.24.0 | |
| `&CStr` | `Arc<CStr>` | 1.24.0 | |
| `OsString` | `Arc<OsStr>` | 1.24.0 | |
| `PathBuf` | `Arc<Path>` | 1.24.0 | |
| `Cow<'_, str>` | `Arc<str>` | 1.45.0 | |
| `Arc<str>` | `Arc<[u8]>` | 1.0.0 | |

## Borrowing Traits

| Trait | Notes |
|-------|-------|
| `Borrow<T>` | Borrows inner T |
| `AsRef<T>` | References inner T |

## I/O Traits (for `Arc<File>`)

| Trait | Notes |
|-------|-------|
| `Read` | Shared file handle reading |
| `Write` | Shared file handle writing |
| `Seek` | Shared file handle seeking |

## Platform Traits

| Trait | Platform | Notes |
|-------|----------|-------|
| `AsFd` | Unix | File descriptor access |
| `AsRawFd` | Unix | Raw file descriptor |
| `AsHandle` | Windows | Windows handle |
| `AsSocket` | Windows | Windows socket |

## Iterator & Collection

| Trait | Since | Notes |
|-------|-------|-------|
| `FromIterator<T>` | 1.37.0 | Collects into `Arc<[T]>` |

## Async

| Conversion | Since | Notes |
|------------|-------|-------|
| `Arc<W>` → `Waker` | 1.51.0 | Where `W: Wake` |
| `Arc<W>` → `RawWaker` | 1.51.0 | Where `W: Wake` |

## Coercion Traits (compiler)

| Trait | Notes |
|-------|-------|
| `CoerceUnsized` | `Arc<T>` → `Arc<U>` where `T: Unsize<U>` |
| `DispatchFromDyn` | For dyn trait dispatch |
