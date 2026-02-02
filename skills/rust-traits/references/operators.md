# Rust Operator Overloading Reference

## Operator-to-Trait Mapping

All overloadable operators live in `std::ops`.

### Arithmetic Operators

| Operator | Trait | Method | Example |
|----------|-------|--------|---------|
| `a + b` | `Add<Rhs=Self>` | `add(self, rhs) -> Output` | `Point + Point` |
| `a - b` | `Sub<Rhs=Self>` | `sub(self, rhs) -> Output` | `Point - Point` |
| `a * b` | `Mul<Rhs=Self>` | `mul(self, rhs) -> Output` | `Matrix * Matrix` |
| `a / b` | `Div<Rhs=Self>` | `div(self, rhs) -> Output` | `Vec3 / f64` |
| `a % b` | `Rem<Rhs=Self>` | `rem(self, rhs) -> Output` | `BigInt % BigInt` |
| `-a` | `Neg` | `neg(self) -> Output` | `-Vector` |

### Compound Assignment

| Operator | Trait | Method |
|----------|-------|--------|
| `a += b` | `AddAssign<Rhs=Self>` | `add_assign(&mut self, rhs)` |
| `a -= b` | `SubAssign<Rhs=Self>` | `sub_assign(&mut self, rhs)` |
| `a *= b` | `MulAssign<Rhs=Self>` | `mul_assign(&mut self, rhs)` |
| `a /= b` | `DivAssign<Rhs=Self>` | `div_assign(&mut self, rhs)` |
| `a %= b` | `RemAssign<Rhs=Self>` | `rem_assign(&mut self, rhs)` |

### Bitwise Operators

| Operator | Trait | Method |
|----------|-------|--------|
| `a & b` | `BitAnd` | `bitand(self, rhs) -> Output` |
| `a \| b` | `BitOr` | `bitor(self, rhs) -> Output` |
| `a ^ b` | `BitXor` | `bitxor(self, rhs) -> Output` |
| `!a` | `Not` | `not(self) -> Output` |
| `a << b` | `Shl` | `shl(self, rhs) -> Output` |
| `a >> b` | `Shr` | `shr(self, rhs) -> Output` |

### Bitwise Compound Assignment

| Operator | Trait |
|----------|-------|
| `a &= b` | `BitAndAssign` |
| `a \|= b` | `BitOrAssign` |
| `a ^= b` | `BitXorAssign` |
| `a <<= b` | `ShlAssign` |
| `a >>= b` | `ShrAssign` |

### Comparison Operators

| Operator | Trait |
|----------|-------|
| `==`, `!=` | `PartialEq` |
| `<`, `>`, `<=`, `>=` | `PartialOrd` (requires `PartialEq`) |

### Other Overloadable Traits

| Operator | Trait | Notes |
|----------|-------|-------|
| `*a` (deref) | `Deref` | Smart pointer dereferencing |
| `a[i]` | `Index` | Immutable indexing |
| `a[i] = v` | `IndexMut` | Mutable indexing |
| `a()` | `Fn`, `FnMut`, `FnOnce` | Callable types |

## Add Trait Definition

```rust
pub trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
```

`Rhs` defaults to `Self`. Override for cross-type operations.

## Patterns

### Same-type operation

```rust
use std::ops::Add;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Point { x: i32, y: i32 }

impl Add for Point {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}
```

### Cross-type operation

```rust
struct Millimeters(u32);
struct Meters(u32);

impl Add<Meters> for Millimeters {
    type Output = Millimeters;
    fn add(self, rhs: Meters) -> Millimeters {
        Millimeters(self.0 + rhs.0 * 1000)
    }
}
```

### Generic implementation

```rust
impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}
```

### Non-commutative operators

Implement both directions explicitly:

```rust
impl ops::Add<Bar> for Foo {
    type Output = FooBar;
    fn add(self, _rhs: Bar) -> FooBar { FooBar }
}

impl ops::Add<Foo> for Bar {
    type Output = BarFoo;
    fn add(self, _rhs: Foo) -> BarFoo { BarFoo }
}
```

## Sources

- [Add trait (core::ops)](https://doc.rust-lang.org/core/ops/trait.Add.html)
- [Operators appendix](https://doc.rust-lang.org/book/appendix-02-operators.html)
- [Operator overloading (Rust by Example)](https://doc.rust-lang.org/rust-by-example/trait/ops.html)
