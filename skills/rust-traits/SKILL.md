---
name: rust-traits
description: >
  Rust trait system best practices, patterns, and idiomatic usage. Covers trait definition,
  implementation, derive macros, dynamic dispatch (dyn), operator overloading (Add, Sub, Mul...),
  Drop, Iterator, Clone, supertraits, impl Trait syntax, fully qualified disambiguation,
  associated types, trait bounds, object safety, blanket implementations, orphan rule, newtype
  pattern, sealed traits, extension traits, and standard traits (Display, From/Into, Default,
  Send/Sync, Sized). Use when writing, reviewing, or refactoring Rust code involving traits:
  (1) Defining or implementing traits, (2) Using derive macros, (3) Operator overloading,
  (4) Dynamic dispatch with dyn Trait, (5) Implementing Iterator, Drop, or Clone,
  (6) Using supertraits, (7) Disambiguating overlapping trait methods, (8) impl Trait syntax,
  (9) Associated types and constants, (10) Object safety, (11) Blanket implementations,
  (12) Send, Sync, Sized, or marker traits.
---

# Rust Traits

## Trait Definition & Implementation

```rust
trait Summary {
    // Required method (no body)
    fn summarize(&self) -> String;

    // Default method (can be overridden)
    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..20])
    }
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}, by {}", self.title, self.author)
    }
    // preview() inherited from default
}
```

### Associated Types

Define placeholder types that implementors must specify. Prefer over generics when there is exactly one natural implementation per type:

```rust
trait Iterator {
    type Item;                    // Associated type
    fn next(&mut self) -> Option<Self::Item>;
}

impl Iterator for Counter {
    type Item = u32;              // Concrete type chosen here
    fn next(&mut self) -> Option<u32> { /* ... */ }
}
```

**Associated type vs generic parameter:**
- `trait Foo { type Bar; }` -- one impl per type, cleaner call sites
- `trait Foo<Bar> {}` -- multiple impls per type (e.g., `From<T>` for many `T`)

### Associated Constants

```rust
trait Float {
    const ZERO: Self;
    const ONE: Self;
}
impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
}
```

## Trait Bounds & Where Clauses

```rust
// Inline bound
fn print_it(item: &impl Display) { println!("{item}"); }

// Generic bound (equivalent)
fn print_it<T: Display>(item: &T) { println!("{item}"); }

// Multiple bounds with +
fn log<T: Display + Debug>(item: &T) { /* ... */ }

// Where clause (cleaner for complex bounds)
fn process<T, U>(t: &T, u: &U) -> String
where
    T: Display + Clone,
    U: Debug + Into<String>,
{
    format!("{t}: {:?}", u)
}
```

**Bound on associated type:**
```rust
fn sum_iter<I>(iter: I) -> i64
where
    I: Iterator<Item = i64>,
{ iter.sum() }
```

## Derive Macros

Apply `#[derive(...)]` to structs, enums, or unions. Built-in derives:

`Clone`, `Copy`, `Debug`, `Default`, `Eq`, `Hash`, `Ord`, `PartialEq`, `PartialOrd`

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point { x: f64, y: f64 }
```

Generated impls add trait bounds on generic params: `#[derive(Clone)]` on `Foo<T>` generates `impl<T: Clone> Clone for Foo<T>`.

## `impl Trait`

**As argument** (static dispatch, syntactic sugar for generics):
```rust
fn process(reader: impl BufRead) -> io::Result<String> { /* ... */ }
// Equivalent to: fn process<R: BufRead>(reader: R) -> ...
```

**As return type** (single concrete type, enables returning closures/iterators):
```rust
fn make_adder(y: i32) -> impl Fn(i32) -> i32 {
    move |x| x + y
}
fn doubled(v: &[i32]) -> impl Iterator<Item = i32> + '_ {
    v.iter().map(|x| x * 2)
}
```

Note: return-position `impl Trait` must return **one** concrete type. For multiple types, use `Box<dyn Trait>`.

## Dynamic Dispatch & Trait Objects

Use `dyn Trait` when the concrete type is unknown at compile time:

```rust
fn make_animal(kind: &str) -> Box<dyn Animal> {
    match kind {
        "dog" => Box::new(Dog {}),
        _     => Box::new(Cat {}),
    }
}
```

Common forms: `Box<dyn Trait>`, `&dyn Trait`, `Arc<dyn Trait>`, `Box<dyn Trait + Send + Sync>`.

**Object safety** -- a trait can only be used as `dyn Trait` if:
1. All methods have `&self`, `&mut self`, or `self: Box<Self>` receiver (or no `self`)
2. No method returns `Self` (use `-> Box<dyn Trait>` or `where Self: Sized` to exclude)
3. No generic type parameters on methods
4. No associated functions without `Self` receiver (or `where Self: Sized` bound)
5. Trait has no `Self: Sized` supertrait bound

For detailed rules, vtable mechanics, and trait object lifetimes, see [references/trait-objects.md](references/trait-objects.md).

## Operator Overloading

Implement traits from `std::ops` to overload operators:

```rust
use std::ops::Add;

impl Add for Point {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}
```

For the complete operator-trait mapping table and patterns, see [references/operators.md](references/operators.md).

## Key Trait Patterns

### Drop

```rust
impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}
// Called automatically when value goes out of scope.
// Manual: drop(value); -- value won't be dropped twice.
// Order: LIFO within a scope. Fields dropped after parent's drop().
```

### Iterator

```rust
impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        if self.count <= 5 { Some(self.count) } else { None }
    }
}
// Enables: counter.take(3), counter.skip(2), for x in counter { ... }
```

### Clone & Copy

```rust
#[derive(Clone)]        // Deep copy (heap resources)
struct Data(Box<i32>);

#[derive(Clone, Copy)]  // Bitwise copy (stack-only types)
struct Point(i32, i32);
```

`Clone` = explicit `.clone()`. `Copy` = implicit copy on assignment (requires `Clone`).

## Supertraits

Require another trait as prerequisite:

```rust
trait OutlinePrint: fmt::Display {
    fn outline_print(&self) {
        let s = self.to_string(); // Display method available
        println!("*** {} ***", s);
    }
}
// Implementing OutlinePrint requires implementing Display first.
```

Multiple supertraits: `trait CompSciStudent: Programmer + Student { ... }`

Equivalent with where clause: `trait Circle where Self: Shape { fn radius(&self) -> f64; }`

## Blanket Implementations

Implement a trait for all types satisfying a bound:

```rust
// From the standard library: any Display type gets ToString
impl<T: Display> ToString for T {
    fn to_string(&self) -> String { /* ... */ }
}
```

Write your own:
```rust
trait Greet {
    fn greet(&self) -> String;
}
impl<T: fmt::Display> Greet for T {
    fn greet(&self) -> String {
        format!("Hello, {self}!")
    }
}
```

## Coherence & Orphan Rule

You can implement a trait for a type only if **at least one of** the trait or the type is defined in your crate. This prevents conflicting implementations across crates.

**Workaround -- Newtype pattern:** Wrap external types to implement external traits:

```rust
struct Wrapper(Vec<String>);

impl fmt::Display for Wrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}
```

Use `Deref` to forward method calls to the inner type if needed.

## Disambiguation

When multiple traits have methods with the same name:

```rust
// Method with &self -- use Trait::method(&instance)
Pilot::fly(&person);
Wizard::fly(&person);

// Associated function (no self) -- use fully qualified syntax
<Dog as Animal>::baby_name()
```

General form: `<Type as Trait>::function(receiver_if_method, args...)`

For complete disambiguation examples, see [references/disambiguation.md](references/disambiguation.md).

## Standard Library Traits

For detailed coverage of Display, Debug, From/Into, Default, AsRef/AsMut, Borrow, Send/Sync, Sized, Eq vs PartialEq, and Ord vs PartialOrd, see [references/standard-traits.md](references/standard-traits.md).

## Advanced Patterns

For sealed traits, extension traits, marker traits, GATs, and trait design guidelines, see [references/advanced-patterns.md](references/advanced-patterns.md).

## Sources

- [Traits (The Rust Book)](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Advanced Traits (The Rust Book)](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html)
- [Traits (Rust by Example)](https://doc.rust-lang.org/rust-by-example/trait.html)
- [Traits (Rust Reference)](https://doc.rust-lang.org/reference/items/traits.html)
- [Trait Objects (Rust Reference)](https://doc.rust-lang.org/reference/types/trait-object.html)
- [impl Trait (Rust Reference)](https://doc.rust-lang.org/reference/types/impl-trait.html)
- [Derive attributes](https://doc.rust-lang.org/reference/attributes/derive.html)
- [std::ops](https://doc.rust-lang.org/std/ops/index.html)
