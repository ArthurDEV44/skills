# Rust Design Patterns

## Strategy Pattern via Closures

Use closures instead of trait objects for simple strategy injection:

```rust
struct Processor<F: Fn(&str) -> String> {
    transform: F,
}

impl<F: Fn(&str) -> String> Processor<F> {
    fn new(transform: F) -> Self {
        Self { transform }
    }

    fn process(&self, input: &str) -> String {
        (self.transform)(input)
    }
}

// Usage
let upper = Processor::new(|s| s.to_uppercase());
let trimmed = Processor::new(|s| s.trim().to_string());
```

When the strategy is complex or needs to be named/stored heterogeneously, use trait objects:

```rust
trait Compressor: Send + Sync {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
    fn decompress(&self, data: &[u8]) -> Vec<u8>;
}

struct Pipeline {
    compressor: Box<dyn Compressor>,
}
```

## Decorator / Wrapper Pattern

```rust
struct LoggingStore<S: Store> {
    inner: S,
}

impl<S: Store> Store for LoggingStore<S> {
    fn get(&self, key: &str) -> Option<Value> {
        let result = self.inner.get(key);
        tracing::debug!(key, found = result.is_some(), "store.get");
        result
    }
}

// Composable: LoggingStore<CachingStore<PostgresStore>>
```

## Type-Level State Machines

Encode valid transitions in the type system -- illegal transitions are compile errors:

```rust
use std::marker::PhantomData;

// States (zero-sized types)
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self, key: &Key) -> Result<Door<Unlocked>, LockError> {
        if key.is_valid() {
            Ok(Door { _state: PhantomData })
        } else {
            Err(LockError::InvalidKey)
        }
    }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> {
        Door { _state: PhantomData }
    }

    fn open(&self) {
        // Only unlocked doors can be opened
    }
}

// door.open() on Door<Locked> is a compile error
```

## Extension Trait Pattern

Add methods to foreign types without violating the orphan rule:

```rust
trait IteratorExt: Iterator {
    fn take_while_inclusive<P>(self, predicate: P) -> TakeWhileInclusive<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        TakeWhileInclusive { iter: self, predicate, done: false }
    }
}

// Blanket implementation for all iterators
impl<I: Iterator> IteratorExt for I {}
```

## Handle Pattern (Opaque Identifier)

```rust
// Opaque handle -- prevents users from constructing or inspecting internals
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(u64);

impl EntityId {
    pub(crate) fn new(raw: u64) -> Self { Self(raw) }
    pub(crate) fn raw(self) -> u64 { self.0 }
}

// Users can pass EntityId around but never create or decompose it
```

## Fallible Constructor

Prefer `TryFrom` or named constructors over panicking:

```rust
struct Port(u16);

impl Port {
    pub fn new(port: u16) -> Result<Self, PortError> {
        if port == 0 {
            return Err(PortError::Zero);
        }
        Ok(Self(port))
    }
}

// Or via TryFrom
impl TryFrom<u16> for Port {
    type Error = PortError;
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
```

## Sealed Trait Pattern

Prevent external implementations of a trait:

```rust
mod private {
    pub trait Sealed {}
}

pub trait MyTrait: private::Sealed {
    fn method(&self);
}

// Only types in this crate can implement Sealed, so only they can implement MyTrait
impl private::Sealed for MyType {}
impl MyTrait for MyType {
    fn method(&self) { /* ... */ }
}
```

## Enum Dispatch

Replace dynamic dispatch (`dyn Trait`) with an enum for known variants -- avoids heap allocation and vtable indirection:

```rust
enum Shape {
    Circle(Circle),
    Rectangle(Rectangle),
    Triangle(Triangle),
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle(c) => c.area(),
            Shape::Rectangle(r) => r.area(),
            Shape::Triangle(t) => t.area(),
        }
    }
}

// Use dyn Trait only when the set of variants is truly open-ended
// Use enum dispatch when variants are known at compile time
```

## Interior Mutability Patterns

When you need shared mutable state:

| Pattern | Thread-safe | Use case |
|---------|-------------|----------|
| `Cell<T>` | No | Copy types, single-threaded |
| `RefCell<T>` | No | Non-Copy types, single-threaded, runtime borrow checking |
| `Mutex<T>` | Yes | Multi-threaded, exclusive access |
| `RwLock<T>` | Yes | Multi-threaded, many readers, few writers |
| `Atomic*` | Yes | Primitive types, lock-free |
| `OnceCell<T>` / `OnceLock<T>` | Cell: No / Lock: Yes | Initialize once, read many |
| `LazyCell<T>` / `LazyLock<T>` | Cell: No / Lock: Yes | Lazy initialization with closure |
