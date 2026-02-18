# Subtyping, Variance, Drop Check, PhantomData, and Borrow Splitting

## Subtyping

`Sub <: Super` means Sub can be used wherever Super is expected.

For lifetimes: `'long <: 'short` if `'long` completely contains `'short`. `'static` is a subtype of every lifetime.

```rust
fn debug<'a>(a: &'a str, b: &'a str) { /* ... */ }

let hello: &'static str = "hello";
{
    let world = String::from("world");
    debug(hello, &world); // 'static downgrades to 'world -- OK
}
```

Subtyping in Rust is restricted to two cases:
1. **Lifetime subtyping**: `'long <: 'short`
2. **Higher-ranked lifetime subtyping**: `for<'a> fn(&'a T)` vs `fn(&'specific T)`

If you erase lifetimes, types have no subtyping -- only type equality.

## Variance

Variance defines how subtyping of a type parameter propagates to the containing type.

### Definitions

- **Covariant**: `F<Sub> <: F<Super>` -- subtyping passes through
- **Contravariant**: `F<Super> <: F<Sub>` -- subtyping inverted
- **Invariant**: no subtyping relationship regardless of `Sub`/`Super`
- **Bivariant**: subtyping in both directions (only for unused parameters)

### Variance Table

| Type | `'a` | `T` | `U` |
|------|------|-----|-----|
| `&'a T` | covariant | covariant | -- |
| `&'a mut T` | covariant | **invariant** | -- |
| `Box<T>`, `Vec<T>` | -- | covariant | -- |
| `UnsafeCell<T>`, `Cell<T>`, `RefCell<T>` | -- | **invariant** | -- |
| `fn(T) -> U` | -- | **contravariant** | covariant |
| `*const T` | -- | covariant | -- |
| `*mut T` | -- | **invariant** | -- |
| `PhantomData<T>` | -- | covariant | -- |

### Why `&mut T` is invariant over T

If `&mut T` were covariant, you could write a short-lived value into a long-lived slot:

```rust
fn assign<T>(input: &mut T, val: T) { *input = val; }

let mut hello: &'static str = "hello";
{
    let world = String::from("world");
    assign(&mut hello, &world);
    // If allowed: hello would point to world's data
}
println!("{hello}"); // use-after-free! world is dropped
```

The compiler rejects this because `&mut T` is invariant over `T` -- `&mut &'static str` is NOT a supertype of `&mut &'short str`.

### Why `fn(T)` is contravariant over T

A function that accepts `'static` references can be used where a function accepting shorter-lived references is expected, because `'static` references satisfy any lifetime requirement:

```rust
fn handle_static(s: &'static str) { println!("{s}"); }

// This is safe: handle_static accepts &'static str,
// which is a subtype of &'a str for any 'a
let f: fn(&str) = handle_static; // contravariance allows this
```

### Struct variance

A struct inherits variance from its fields. The rules combine field-by-field:

- All fields covariant over T -> struct covariant over T
- All fields contravariant over T -> struct contravariant over T
- Mixed or any invariant -> struct **invariant** over T

```rust
use std::cell::UnsafeCell;

struct Variance<'a, 'b, 'c, T, U: 'a> {
    x: &'a U,               // covariant in 'a; covariant in U (but see w)
    y: *const T,             // covariant in T
    z: UnsafeCell<&'b f64>,  // invariant in 'b (UnsafeCell is invariant)
    w: *mut U,               // invariant in U (overrides covariance from x)
    f: fn(&'c ()) -> &'c (), // co + contra in 'c = invariant in 'c
}
// Result: covariant in 'a, invariant in 'b, invariant in 'c,
//         covariant in T, invariant in U
```

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

### Why strict outlive matters

```rust
let (inspector, data);  // declaration order matters for drop order!
data = 42u8;
inspector = Inspector(&data);
// Drop order: inspector first, then data
// Inspector::drop reads &data -- data must still be alive -> OK

// But if reversed:
// let (data, inspector); // data dropped first!
// inspector = Inspector(&data);
// Inspector::drop would read freed data -> UNSOUND
```

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

### Drop check for type parameters

```rust
struct Container<T> { data: T }

// With Drop impl: T must strictly outlive Container
impl<T> Drop for Container<T> {
    fn drop(&mut self) {
        // Even if we don't access self.data, the compiler assumes we might
    }
}

// Without Drop impl: no extra constraint -- T can be dropped simultaneously
```

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

### Choosing the right PhantomData

Decision tree for raw pointer wrappers:

1. **Owns the data** (like `Box<T>`)? -> `PhantomData<T>` (drop check + covariant + inherited Send/Sync)
2. **Borrows the data** (like `&'a T`)? -> `PhantomData<&'a T>` (bounds lifetime + covariant)
3. **Mutable borrow** (like `&'a mut T`)? -> `PhantomData<&'a mut T>` (bounds lifetime + invariant over T)
4. **Needs Send + Sync but no ownership** (type-level tag)? -> `PhantomData<fn(T) -> T>` (invariant, Send + Sync)
5. **Non-Send/Sync marker**? -> `PhantomData<*const T>` or `PhantomData<*mut T>`

## Borrow Splitting

### Struct fields (works automatically)

The borrow checker tracks struct fields independently:

```rust
let mut x = Foo { a: 0, b: 0, c: 0 };
let a = &mut x.a;
let b = &mut x.b;
let c = &x.c;
// All OK: disjoint fields
*a += 1;
*b += 1;
println!("{c}");
```

This does NOT work through method calls -- the borrow checker sees `self` as a single unit:

```rust
// FAILS: get_a borrows all of self
// let a = x.get_a();  // &mut self
// let b = x.get_b();  // ERROR: self already borrowed
```

### Slices/arrays (needs split_at_mut)

```rust
let mut arr = [1, 2, 3, 4, 5];
let (left, right) = arr.split_at_mut(2);
// left = &mut [1, 2], right = &mut [3, 4, 5] -- independent borrows
left[0] = 10;
right[0] = 30;
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

### HashMap borrow splitting

The borrow checker can't split borrows on `HashMap` keys. Use the entry API:

```rust
use std::collections::HashMap;

let mut map = HashMap::new();
// FAILS: two mutable borrows of map
// let a = map.get_mut("key1").unwrap();
// let b = map.get_mut("key2").unwrap();

// Solution 1: entry API for insert-or-update
map.entry("key1").or_insert(0);

// Solution 2: get_many_mut (nightly) or split into separate maps
```

## Sources

- [Subtyping and variance (Nomicon)](https://doc.rust-lang.org/nomicon/subtyping.html)
- [Subtyping and variance (Rust Reference)](https://doc.rust-lang.org/reference/subtyping.html)
- [Drop check (Nomicon)](https://doc.rust-lang.org/nomicon/dropck.html)
- [PhantomData (Nomicon)](https://doc.rust-lang.org/nomicon/phantom-data.html)
- [Borrow splitting (Nomicon)](https://doc.rust-lang.org/nomicon/borrow-splitting.html)
- [Destructors (Rust Reference)](https://doc.rust-lang.org/reference/destructors.html)
