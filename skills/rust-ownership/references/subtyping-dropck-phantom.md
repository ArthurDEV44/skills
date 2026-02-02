# Subtyping, Variance, Drop Check, PhantomData, and Borrow Splitting

## Subtyping

`Sub <: Super` means Sub can be used wherever Super is expected.

For lifetimes: `'long <: 'short` if `'long` completely contains `'short`.

```rust
fn debug<'a>(a: &'a str, b: &'a str) { /* ... */ }

let hello: &'static str = "hello";
{
    let world = String::from("world");
    debug(hello, &world); // 'static downgrades to 'world -- OK
}
```

## Variance Table

| Type | `'a` | `T` | `U` |
|------|------|-----|-----|
| `&'a T` | covariant | covariant | -- |
| `&'a mut T` | covariant | **invariant** | -- |
| `Box<T>`, `Vec<T>` | -- | covariant | -- |
| `UnsafeCell<T>`, `Cell<T>` | -- | **invariant** | -- |
| `fn(T) -> U` | -- | **contravariant** | covariant |
| `*const T` | -- | covariant | -- |
| `*mut T` | -- | **invariant** | -- |

- **Covariant**: `F<Sub> <: F<Super>` (subtyping passes through)
- **Contravariant**: `F<Super> <: F<Sub>` (subtyping inverted)
- **Invariant**: no subtyping relationship

### Why `&mut T` is invariant over T

```rust
fn assign<T>(input: &mut T, val: T) { *input = val; }

let mut hello: &'static str = "hello";
{
    let world = String::from("world");
    assign(&mut hello, &world); // REJECTED: would write short-lived into long-lived slot
}
println!("{hello}"); // would be use-after-free
```

### Struct variance

A struct inherits variance from its fields:
- All fields covariant over T -> struct covariant over T
- All fields contravariant over T -> struct contravariant over T
- Mixed or any invariant -> struct **invariant** over T

## Drop Check

**The big rule**: for a generic type to soundly implement `Drop`, its generic arguments must **strictly outlive** it.

```rust
struct Inspector<'a>(&'a u8);

impl<'a> Drop for Inspector<'a> {
    fn drop(&mut self) {
        println!("{}", self.0); // accesses borrowed data in destructor
    }
}
```

The borrow checker is conservative: it doesn't analyze destructor bodies, so even safe destructors that don't access borrowed data are rejected if the lifetime isn't satisfied.

### Escape hatch: `#[may_dangle]` (unstable)

Assert that a destructor won't access the marked parameter's data:

```rust
unsafe impl<#[may_dangle] 'a> Drop for Inspector<'a> {
    fn drop(&mut self) {
        // must NOT access self.0 here
    }
}
```

Do NOT use `#[may_dangle]` if the destructor calls trait methods or callbacks on the type parameter -- these could access expired data.

## PhantomData

Zero-sized marker type that simulates a field for variance and drop check purposes.

Use when a type logically owns or borrows `T` but has no field of type `T` (e.g., raw pointer wrappers).

```rust
use std::marker::PhantomData;

struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: PhantomData<&'a T>,  // bounds 'a, covariant over 'a and T
}
```

### PhantomData patterns table

| Phantom type | `'a` variance | `T` variance | Send/Sync | drop check |
|---|---|---|---|---|
| `PhantomData<T>` | -- | covariant | inherited | owns T |
| `PhantomData<&'a T>` | covariant | covariant | needs `T: Sync` | no ownership |
| `PhantomData<&'a mut T>` | covariant | **invariant** | inherited | no ownership |
| `PhantomData<*const T>` | -- | covariant | `!Send + !Sync` | no ownership |
| `PhantomData<*mut T>` | -- | **invariant** | `!Send + !Sync` | no ownership |
| `PhantomData<fn(T)>` | -- | **contravariant** | `Send + Sync` | no ownership |
| `PhantomData<fn() -> T>` | -- | covariant | `Send + Sync` | no ownership |
| `PhantomData<fn(T) -> T>` | -- | **invariant** | `Send + Sync` | no ownership |
| `PhantomData<Cell<&'a ()>>` | **invariant** | -- | `Send + !Sync` | no ownership |

Since RFC 1238: if a type has a `Drop` impl, Rust automatically treats it as owning its generic parameters (no `PhantomData` needed for drop check). `PhantomData` is still needed for variance and auto-trait control.

## Borrow Splitting

### Struct fields (works automatically)

```rust
let mut x = Foo { a: 0, b: 0, c: 0 };
let a = &mut x.a;
let b = &mut x.b;
let c = &x.c;
// All OK: disjoint fields
```

### Slices/arrays (needs split_at_mut)

```rust
let mut arr = [1, 2, 3];
let (left, right) = arr.split_at_mut(1);
// left = &mut [1], right = &mut [2, 3]  -- independent borrows
```

### Mutable iterators

The `Iterator` trait naturally supports borrow splitting because `Item` has no connection to `self`. Each call to `next()` yields a unique reference, never producing overlapping `&mut`:

```rust
// Safe mutable linked-list iterator
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.take().map(|node| {
            self.0 = node.next.as_mut().map(|n| &mut **n);
            &mut node.elem
        })
    }
}
```

For slices, use `mem::take` + `split_at_mut`:

```rust
fn next(&mut self) -> Option<Self::Item> {
    let slice = mem::take(&mut self.0);
    if slice.is_empty() { return None; }
    let (l, r) = slice.split_at_mut(1);
    self.0 = r;
    l.get_mut(0)
}
```

## Sources

- [Subtyping and variance](https://doc.rust-lang.org/nomicon/subtyping.html)
- [Drop check](https://doc.rust-lang.org/nomicon/dropck.html)
- [PhantomData](https://doc.rust-lang.org/nomicon/phantom-data.html)
- [Borrow splitting](https://doc.rust-lang.org/nomicon/borrow-splitting.html)
