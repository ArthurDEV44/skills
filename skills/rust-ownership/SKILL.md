---
name: rust-ownership
description: >
  Rust ownership, borrowing, and lifetime system from the Nomicon, Rust Book, and Rust Reference.
  Covers ownership rules, move semantics, Copy vs Clone, references, aliasing, reborrowing,
  NLL, lifetime elision, subtyping and variance, HRTB, unbounded lifetimes, drop check,
  PhantomData, borrow splitting, interior mutability (Cell/RefCell/UnsafeCell), trait object
  lifetimes, and lifetime bounds (T: 'a). Use when writing, reviewing, or debugging Rust code
  involving ownership and lifetimes: (1) Fixing borrow checker errors, (2) Lifetime elision rules,
  (3) Subtyping and variance in generic types, (4) PhantomData for correct variance, (5) Drop
  check and destructors, (6) HRTB with closures, (7) Interior mutability patterns,
  (8) Reborrowing and two-phase borrows, (9) Trait object lifetimes (dyn Trait + 'a),
  (10) Borrow splitting on structs/slices.
---

# Rust Ownership & Lifetimes

## Ownership Rules

Three fundamental rules enforced at compile time:

1. Each value has exactly **one owner** (a variable binding)
2. When the owner goes out of scope, the value is **dropped** (destructor runs, memory freed)
3. Ownership can be **transferred** (moved), not duplicated (unless `Copy`)

```rust
let s1 = String::from("hello");
let s2 = s1;       // s1 is MOVED into s2 -- s1 is now invalid
// println!("{s1}"); // ERROR: value used after move
println!("{s2}");   // OK: s2 owns the data
```

For move semantics, Copy vs Clone, partial moves, and owned-vs-borrowed type patterns, see [references/ownership-moves-interior-mut.md](references/ownership-moves-interior-mut.md).

## Borrowing Rules

Two kinds of references: `&T` (shared/immutable) and `&mut T` (exclusive/mutable). Two rules:

1. A reference **cannot outlive** its referent
2. A mutable reference **cannot be aliased** -- at any point, you can have EITHER:
   - One or more `&T` references, OR
   - Exactly one `&mut T` reference

```rust
let mut v = vec![1, 2, 3];
let first = &v[0];     // shared borrow
let second = &v[1];    // OK: multiple shared borrows
println!("{first} {second}");
// first and second are dead (NLL)
v.push(4);             // OK: exclusive borrow, no active shared borrows
```

### Reborrowing

`&mut` references are not `Copy`, but the compiler implicitly **reborrows** them when passed to functions, creating a shorter-lived `&mut` from the original:

```rust
fn process(data: &mut Vec<i32>) { data.push(1); }

let mut v = vec![];
let r = &mut v;
process(r);    // implicit reborrow: process(&mut *r), r stays valid
process(r);    // works again -- r was reborrowed, not moved
r.push(2);     // r is still usable
```

## Aliasing

Variables alias when they refer to overlapping memory. Rust's no-aliasing guarantee on `&mut` enables compiler optimizations: caching values in registers, eliminating reads/writes, reordering operations.

```rust
// Safe to optimize: &input and &mut output cannot alias
fn compute(input: &u32, output: &mut u32) {
    let cached = *input;  // compiler can cache -- won't change via output
    if cached > 10 { *output = 2; }
    else if cached > 5 { *output *= 2; }
}
```

## Lifetimes: Core Rules

Every reference has a **lifetime** -- the region of code where it is valid. Lifetimes are mostly inferred (Non-Lexical Lifetimes / NLL since Rust 2021 edition).

1. A reference is alive from creation to **last use** (not end of scope) -- this is NLL
2. The borrowed value must **outlive** all active borrows
3. No `&mut T` may coexist with any other reference to the same data
4. At function boundaries, lifetimes must be named or elided

### Last-Use Semantics (NLL)

```rust
let mut data = vec![1, 2, 3];
let x = &data[0];
println!("{}", x);  // last use of x -- borrow ends here
data.push(4);       // OK: x is dead
```

Lifetimes can have **gaps** (invalidated then reinitialized):

```rust
let mut data = vec![1, 2, 3, 4];
let mut x = &data[0];
println!("{}", x);   // borrow 1 ends
data.push(5);        // OK
x = &data[3];        // borrow 2 starts
println!("{}", x);
```

### `'static` Lifetime

`'static` means the reference is valid for the entire program. Two forms:

```rust
let s: &'static str = "hello";         // string literal: baked into binary
static GLOBAL: i32 = 42;
let r: &'static i32 = &GLOBAL;         // reference to static item

// T: 'static means T contains no non-static references (T can be owned)
fn spawn<T: Send + 'static>(t: T) {}   // T must be self-contained
```

### Lifetime Bounds

`T: 'a` means all references inside `T` must live at least as long as `'a`:

```rust
struct Ref<'a, T: 'a> {
    data: &'a T,  // T: 'a implied by &'a T
}

// Explicit bound needed when T is not directly referenced
struct Wrapper<'a, T: 'a> {
    ptr: *const T,
    _marker: std::marker::PhantomData<&'a T>,
}
```

### Trait Object Lifetimes

Trait objects have an implicit lifetime bound:

```rust
// Box<dyn Trait> is Box<dyn Trait + 'static> (owned context)
// &'a dyn Trait is &'a (dyn Trait + 'a)        (borrowed context)
fn use_trait(t: &dyn Display);          // implicitly: &'a (dyn Display + 'a)
fn boxed_trait(t: Box<dyn Display>);    // implicitly: Box<dyn Display + 'static>
```

## Destructors Extend Borrows

If a type implements `Drop`, its destructor counts as a use at end of scope:

```rust
struct X<'a>(&'a i32);
impl Drop for X<'_> { fn drop(&mut self) {} }

let mut data = vec![1, 2, 3];
let x = X(&data[0]);
println!("{:?}", x);
data.push(4);  // ERROR: x's drop runs at scope end, borrow still active
// Fix: drop(x); before push, or restructure
```

## Interior Mutability

Allows mutation through `&T` (shared reference) by moving the borrow-check to runtime or using atomic operations. For patterns and details, see [references/ownership-moves-interior-mut.md](references/ownership-moves-interior-mut.md).

| Type | Thread-safe | Overhead | Use case |
|------|-------------|----------|----------|
| `Cell<T>` | No | Zero-cost | `Copy` types, single-threaded |
| `RefCell<T>` | No | Runtime check | Complex borrows, single-threaded |
| `UnsafeCell<T>` | No | None | Building custom abstractions |
| `Mutex<T>` | Yes | Lock | Shared mutable state across threads |
| `RwLock<T>` | Yes | Lock | Read-heavy shared state |
| `AtomicT` | Yes | Atomic ops | Simple counters, flags |

## Lifetime Elision

For detailed rules and examples, see [references/elision-hrtb-unbounded.md](references/elision-hrtb-unbounded.md).

Applied at function boundaries (in order):
1. Each input reference gets a distinct lifetime parameter
2. If exactly one input lifetime, it's assigned to all output lifetimes
3. If one input is `&self`/`&mut self`, self's lifetime is assigned to all outputs

```rust
fn substr(s: &str, until: usize) -> &str;             // elided
fn substr<'a>(s: &'a str, until: usize) -> &'a str;   // expanded

fn get_mut(&mut self) -> &mut T;                       // elided
fn get_mut<'a>(&'a mut self) -> &'a mut T;             // expanded

fn frob(s: &str, t: &str) -> &str;                    // ILLEGAL: ambiguous
```

## Subtyping & Variance

For the full variance table and examples, see [references/subtyping-dropck-phantom.md](references/subtyping-dropck-phantom.md).

`'long <: 'short` -- a longer lifetime is a subtype of a shorter one.

| Type | `'a` | `T` |
|------|------|-----|
| `&'a T` | covariant | covariant |
| `&'a mut T` | covariant | **invariant** |
| `Box<T>`, `Vec<T>` | -- | covariant |
| `Cell<T>`, `UnsafeCell<T>` | -- | **invariant** |
| `fn(T) -> U` | -- | **contra** `T`, co `U` |
| `*const T` | -- | covariant |
| `*mut T` | -- | **invariant** |

Key insight: `&mut T` is invariant over `T` to prevent writing a short-lived value where a long-lived one is expected.

## Borrow Splitting

The borrow checker allows simultaneous mutable borrows of **disjoint struct fields**:

```rust
let a = &mut x.field_a;
let b = &mut x.field_b;  // OK: different fields
```

For slices/arrays, use `split_at_mut`:

```rust
let (left, right) = slice.split_at_mut(mid);
// left and right are independent &mut slices
```

For details and mutable iterator patterns, see [references/subtyping-dropck-phantom.md](references/subtyping-dropck-phantom.md).

## Lifetime Mismatch Limits

The borrow checker is intentionally coarse-grained. Some correct programs are rejected:

```rust
impl Foo {
    fn mutate_and_share(&mut self) -> &Self { &*self }
    fn share(&self) {}
}

let loan = foo.mutate_and_share(); // &mut borrow with lifetime of loan
foo.share();                        // ERROR: &mut still alive due to loan
println!("{:?}", loan);
```

The lifetime system extends `&mut foo` to the lifetime of `loan`, blocking further borrows.

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "does not live long enough" | Reference outlives data | Return owned value or extend data's scope |
| "cannot borrow as mutable" | Overlapping borrows | Move last use of `&` ref before `&mut` borrow |
| "borrowed value does not live long enough" + Drop | Destructor extends borrow | Use `drop(x)` explicitly |
| "lifetime may not live long enough" | Signature mismatch | Adjust lifetime annotations |
| "cannot borrow `*map` as mutable more than once" | Lifetime system conservative | Restructure control flow or use entry API |
| "use of moved value" | Value moved earlier | Clone, use reference, or restructure |
| "cannot move out of borrowed content" | Moving from behind `&` or `&mut` | Clone, `mem::take`, `mem::replace`, or `Option::take` |
| "already borrowed: BorrowMutError" (runtime) | RefCell borrow conflict | Restructure to avoid overlapping `borrow()`/`borrow_mut()` |
| "closure may outlive current function" | Closure captures reference | Use `move` closure or ensure reference lives long enough |
| "cannot return reference to local variable" | Returning dangling ref | Return owned type or use lifetime parameter |

## Official Documentation

- [Ownership (Rust Book)](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
- [References and Borrowing (Rust Book)](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- [Lifetimes (Rust Book)](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)
- [Lifetimes (Nomicon)](https://doc.rust-lang.org/nomicon/lifetimes.html)
- [References (Nomicon)](https://doc.rust-lang.org/nomicon/references.html)
- [Aliasing (Nomicon)](https://doc.rust-lang.org/nomicon/aliasing.html)
- [Lifetime elision (Nomicon)](https://doc.rust-lang.org/nomicon/lifetime-elision.html)
- [Lifetime mismatch (Nomicon)](https://doc.rust-lang.org/nomicon/lifetime-mismatch.html)
- [Unbounded lifetimes (Nomicon)](https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html)
- [Higher-Rank Trait Bounds (Nomicon)](https://doc.rust-lang.org/nomicon/hrtb.html)
- [Subtyping and variance (Nomicon)](https://doc.rust-lang.org/nomicon/subtyping.html)
- [Drop check (Nomicon)](https://doc.rust-lang.org/nomicon/dropck.html)
- [PhantomData (Nomicon)](https://doc.rust-lang.org/nomicon/phantom-data.html)
- [Borrow splitting (Nomicon)](https://doc.rust-lang.org/nomicon/borrow-splitting.html)
- [Subtyping and variance (Rust Reference)](https://doc.rust-lang.org/reference/subtyping.html)
- [Lifetime elision (Rust Reference)](https://doc.rust-lang.org/reference/lifetime-elision.html)
- [Interior mutability (Rust Reference)](https://doc.rust-lang.org/reference/interior-mutability.html)
- [Destructors (Rust Reference)](https://doc.rust-lang.org/reference/destructors.html)
