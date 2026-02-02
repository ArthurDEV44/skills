---
name: rust-traits
description: >
  Rust trait system best practices, patterns, and idiomatic usage. Covers trait definition,
  implementation, derive macros, dynamic dispatch (dyn), operator overloading (Add, Sub, Mul...),
  Drop, Iterator, Clone, supertraits, impl Trait syntax, and fully qualified disambiguation.
  Use when writing, reviewing, or refactoring Rust code involving traits: (1) Defining or
  implementing traits, (2) Using derive macros, (3) Operator overloading with std::ops traits,
  (4) Dynamic dispatch with dyn Trait and trait objects, (5) Implementing Iterator, Drop, or Clone,
  (6) Using supertraits or trait hierarchies, (7) Disambiguating overlapping trait methods,
  (8) Using impl Trait as argument or return type.
---

# Rust Traits

## Quick Reference

### Trait Definition & Implementation

```rust
trait Animal {
    fn new(name: &'static str) -> Self;  // Associated function
    fn name(&self) -> &'static str;       // Required method
    fn talk(&self) {                      // Default method
        println!("{} says hello", self.name());
    }
}

impl Animal for Dog {
    fn new(name: &'static str) -> Self { Dog { name } }
    fn name(&self) -> &'static str { self.name }
    // talk() inherited from default
}
```

### Derive Macros

Apply `#[derive(...)]` to structs, enums, or unions. Built-in derives:

`Clone`, `Copy`, `Debug`, `Default`, `Eq`, `Hash`, `Ord`, `PartialEq`, `PartialOrd`

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point { x: f64, y: f64 }
```

Generated impls add trait bounds on generic params: `#[derive(Clone)]` on `Foo<T>` generates `impl<T: Clone> Clone for Foo<T>`.

### Dynamic Dispatch (`dyn`)

Use `Box<dyn Trait>` to return different concrete types implementing the same trait:

```rust
fn make_animal(kind: &str) -> Box<dyn Animal> {
    match kind {
        "dog" => Box::new(Dog {}),
        _     => Box::new(Cat {}),
    }
}
```

`dyn Trait` = pointer with vtable, fixed size. Use when the concrete type is unknown at compile time.

### `impl Trait`

**As argument** (static dispatch, syntactic sugar for generics):
```rust
fn process(reader: impl BufRead) -> io::Result<String> { ... }
// Equivalent to: fn process<R: BufRead>(reader: R) -> ...
```

**As return type** (enables returning closures/iterators without naming types):
```rust
fn make_adder(y: i32) -> impl Fn(i32) -> i32 {
    move |x| x + y
}
fn doubled(v: &[i32]) -> impl Iterator<Item = i32> + '_ {
    v.iter().map(|x| x * 2)
}
```

## Operator Overloading

For detailed operator-trait mapping and examples, see [references/operators.md](references/operators.md).

Implement traits from `std::ops` to overload operators:

```rust
use std::ops::Add;

impl Add for Point {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}
// Now: Point { x:1, y:2 } + Point { x:3, y:4 }
```

Cross-type addition with custom `Rhs`:
```rust
impl Add<Meters> for Millimeters {
    type Output = Millimeters;
    fn add(self, rhs: Meters) -> Millimeters {
        Millimeters(self.0 + rhs.0 * 1000)
    }
}
```

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

### Clone

```rust
#[derive(Clone)]        // Deep copy for types with heap resources
struct Data(Box<i32>);

#[derive(Clone, Copy)]  // Bitwise copy for simple stack types
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

## Official Documentation

- [Traits (Rust by Example)](https://doc.rust-lang.org/rust-by-example/trait.html)
- [Derive attributes](https://doc.rust-lang.org/reference/attributes/derive.html)
- [Dynamic dispatch (dyn)](https://doc.rust-lang.org/rust-by-example/trait/dyn.html)
- [impl Trait](https://doc.rust-lang.org/rust-by-example/trait/impl_trait.html)
- [Operator overloading](https://doc.rust-lang.org/rust-by-example/trait/ops.html)
- [Add trait (core::ops)](https://doc.rust-lang.org/core/ops/trait.Add.html)
- [Operators appendix](https://doc.rust-lang.org/book/appendix-02-operators.html)
- [Drop](https://doc.rust-lang.org/rust-by-example/trait/drop.html)
- [Iterators](https://doc.rust-lang.org/rust-by-example/trait/iter.html)
- [Clone](https://doc.rust-lang.org/rust-by-example/trait/clone.html)
- [Supertraits](https://doc.rust-lang.org/rust-by-example/trait/supertraits.html)
- [Advanced traits (supertraits)](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait)
- [Disambiguation](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html)
- [Fully qualified syntax](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name)
