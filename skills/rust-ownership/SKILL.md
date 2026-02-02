---
name: rust-ownership
description: >
  Rust ownership, borrowing, and lifetime system from the Nomicon. Covers references and aliasing
  rules, borrow checker mechanics, lifetime regions and scope desugaring, lifetime elision rules,
  subtyping and variance (covariant/contravariant/invariant), Higher-Rank Trait Bounds (HRTB),
  unbounded lifetimes, drop check, PhantomData patterns, borrow splitting, and lifetime mismatch
  limits. Use when writing, reviewing, or debugging Rust code involving ownership and lifetimes:
  (1) Fixing borrow checker errors, (2) Understanding why a reference outlives its referent,
  (3) Debugging aliased mutable reference errors, (4) Understanding lifetime elision rules,
  (5) Working with subtyping and variance in generic types, (6) Using PhantomData for correct
  variance, (7) Splitting borrows on structs/slices, (8) Understanding drop check and destructors,
  (9) Using Higher-Rank Trait Bounds with closures, (10) Fixing lifetime mismatch limitations.
---

# Rust Ownership & Lifetimes (Nomicon)

## Reference Rules

Two kinds: `&T` (shared) and `&mut T` (exclusive). Two rules:

1. A reference **cannot outlive** its referent
2. A mutable reference **cannot be aliased**

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

1. A reference is alive from creation to **last use** (not end of scope)
2. The borrowed value must **outlive** all active borrows
3. No `&mut T` may coexist with any other reference to the same data
4. At function boundaries, lifetimes must be named or elided

## Last-Use Semantics

```rust
let mut data = vec![1, 2, 3];
let x = &data[0];
println!("{}", x);  // last use of x
data.push(4);       // OK: x is dead
```

Lifetimes can have **gaps** (invalidated then reinitialized):

```rust
let mut x = &data[0];
println!("{}", x);   // borrow 1 ends
data.push(4);        // OK
x = &data[3];        // borrow 2 starts
```

## Destructors Extend Borrows

If a type implements `Drop`, its destructor counts as a use at end of scope:

```rust
struct X<'a>(&'a i32);
impl Drop for X<'_> { fn drop(&mut self) {} }

let mut data = vec![1, 2, 3];
let x = X(&data[0]);
println!("{:?}", x);
data.push(4);  // ERROR: x's drop runs at scope end
// Fix: drop(x); before push
```

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

**Variance** defines how type parameter subtyping propagates:

| Type | `'a` | `T` |
|------|------|-----|
| `&'a T` | covariant | covariant |
| `&'a mut T` | covariant | **invariant** |
| `Box<T>`, `Vec<T>` | -- | covariant |
| `Cell<T>`, `UnsafeCell<T>` | -- | **invariant** |
| `fn(T) -> U` | -- | **contra** `T`, co `U` |
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
| "cannot borrow `*map` as mutable more than once" | Lifetime system too conservative | Restructure control flow or use entry API |

## Official Documentation

- [Lifetimes](https://doc.rust-lang.org/nomicon/lifetimes.html)
- [References](https://doc.rust-lang.org/nomicon/references.html)
- [Aliasing](https://doc.rust-lang.org/nomicon/aliasing.html)
- [Lifetime elision](https://doc.rust-lang.org/nomicon/lifetime-elision.html)
- [Lifetime mismatch](https://doc.rust-lang.org/nomicon/lifetime-mismatch.html)
- [Unbounded lifetimes](https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html)
- [Higher-Rank Trait Bounds](https://doc.rust-lang.org/nomicon/hrtb.html)
- [Subtyping and variance](https://doc.rust-lang.org/nomicon/subtyping.html)
- [Drop check](https://doc.rust-lang.org/nomicon/dropck.html)
- [PhantomData](https://doc.rust-lang.org/nomicon/phantom-data.html)
- [Borrow splitting](https://doc.rust-lang.org/nomicon/borrow-splitting.html)
