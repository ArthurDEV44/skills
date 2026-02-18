---
name: rust-best-practices
description: >
  Idiomatic Rust patterns, API design, and cross-cutting best practices for writing clean,
  performant, and maintainable Rust code. Covers newtype pattern, typestate pattern, builder
  pattern, RAII, zero-cost abstractions, iterator and closure idioms, enum design, Option/Result
  combinators, string handling (str vs String, Cow), conversion traits (From/Into, AsRef, Borrow,
  Deref, TryFrom), API design principles, documentation with rustdoc, Cargo workspace and feature
  management, Clippy linting, profiling, and common anti-patterns. Complements specialized Rust
  skills (rust-traits, rust-ownership, rust-async, rust-concurrency). Use when writing, reviewing,
  or refactoring Rust code: (1) Choosing idiomatic patterns (newtype, typestate, builder),
  (2) Designing public APIs with conversion traits, (3) Writing iterator chains and closures,
  (4) Enum design and match ergonomics, (5) String and Cow usage, (6) Avoiding common anti-patterns
  and clippy warnings, (7) Writing rustdoc documentation, (8) Configuring Cargo features, profiles,
  and workspaces, (9) Performance profiling and optimization, (10) Edition 2024 and MSRV management.
---

# Rust Best Practices

## Idiomatic Patterns

### Newtype Pattern

Wrap a type to give it distinct semantics, prevent mixing, and attach custom behavior:

```rust
struct Meters(f64);
struct Seconds(f64);

// Compiler prevents: let speed = meters + seconds;  -- different types!

impl Meters {
    fn new(val: f64) -> Self {
        assert!(val >= 0.0, "distance cannot be negative");
        Self(val)
    }
    fn as_f64(&self) -> f64 { self.0 }
}

// Implement Display, From, arithmetic as needed
impl std::fmt::Display for Meters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}m", self.0)
    }
}
```

**When to use:** domain types (UserId, Email, Port), enforcing invariants at construction, adding behavior to foreign types (orphan rule workaround).

### Typestate Pattern

Encode state machine transitions in the type system -- invalid states become compile errors:

```rust
struct Unvalidated;
struct Validated;

struct Form<State> {
    data: String,
    _state: std::marker::PhantomData<State>,
}

impl Form<Unvalidated> {
    fn new(data: String) -> Self {
        Self { data, _state: std::marker::PhantomData }
    }

    fn validate(self) -> Result<Form<Validated>, ValidationError> {
        if self.data.is_empty() {
            return Err(ValidationError::Empty);
        }
        Ok(Form { data: self.data, _state: std::marker::PhantomData })
    }
}

impl Form<Validated> {
    fn submit(&self) -> Result<(), SubmitError> {
        // Only validated forms can be submitted
        Ok(())
    }
}
```

### Builder Pattern

For constructing complex types with many optional fields:

```rust
pub struct ServerConfig {
    host: String,
    port: u16,
    max_connections: usize,
    tls: bool,
}

pub struct ServerConfigBuilder {
    host: String,
    port: u16,
    max_connections: usize,
    tls: bool,
}

impl ServerConfigBuilder {
    pub fn new(host: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            port: 8080,
            max_connections: 100,
            tls: false,
        }
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn max_connections(mut self, n: usize) -> Self {
        self.max_connections = n;
        self
    }

    pub fn tls(mut self, enabled: bool) -> Self {
        self.tls = enabled;
        self
    }

    pub fn build(self) -> ServerConfig {
        ServerConfig {
            host: self.host,
            port: self.port,
            max_connections: self.max_connections,
            tls: self.tls,
        }
    }
}

// Usage
let config = ServerConfigBuilder::new("localhost")
    .port(3000)
    .tls(true)
    .build();
```

### RAII (Resource Acquisition Is Initialization)

Tie resource cleanup to scope via `Drop`:

```rust
struct TempFile {
    path: std::path::PathBuf,
}

impl TempFile {
    fn new(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        std::fs::File::create(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
// File is guaranteed to be cleaned up when TempFile goes out of scope
```

Use for: file handles, locks, database connections, network sockets, temporary resources.

For more design patterns (strategy, decorator, type-level programming), see `references/design-patterns.md`.

## Enum Design & Pattern Matching

### Rich Enums

Enums carry data -- use them instead of stringly-typed code:

```rust
// BAD: stringly typed
fn process(status: &str) { /* "pending", "active", "cancelled"... */ }

// GOOD: each variant carries its own data
enum OrderStatus {
    Pending,
    Active { started_at: DateTime<Utc> },
    Shipped { tracking: String },
    Cancelled { reason: String },
}
```

### Option/Result Combinators

Prefer combinators over `match` and `if let` chains:

```rust
// Map, and_then, unwrap_or, unwrap_or_else
let name = user.name.as_deref().unwrap_or("anonymous");

let result = parse_id(input)
    .map_err(|e| AppError::InvalidId(e))
    .and_then(|id| find_user(id))
    .map(|user| user.display_name());

// ok_or / ok_or_else: Option -> Result
let user = cache.get(&id).ok_or(AppError::NotFound)?;

// transpose: Option<Result<T, E>> <-> Result<Option<T>, E>
let maybe_val: Option<Result<i32, Error>> = Some(Ok(42));
let result: Result<Option<i32>, Error> = maybe_val.transpose();
```

### Match Ergonomics

```rust
// Exhaustive matching -- compiler ensures all variants handled
match status {
    OrderStatus::Pending => handle_pending(),
    OrderStatus::Active { started_at } => handle_active(started_at),
    OrderStatus::Shipped { tracking } => ship(tracking),
    OrderStatus::Cancelled { reason } => log_cancel(reason),
}

// Use _ for catch-all only when you genuinely don't care about future variants
// Adding #[non_exhaustive] to public enums forces callers to have a _ arm

// if let for single-variant interest
if let Some(val) = optional {
    use_val(val);
}

// let-else (Rust 1.65+) for early return on mismatch
let Some(config) = load_config() else {
    return Err(AppError::NoConfig);
};
```

## Iterator & Closure Idioms

### Iterator Chains

```rust
// Prefer iterator chains over manual loops
let active_names: Vec<&str> = users.iter()
    .filter(|u| u.is_active)
    .map(|u| u.name.as_str())
    .collect();

// Use fold / reduce for accumulation
let total: i64 = orders.iter()
    .filter(|o| o.status == Status::Completed)
    .map(|o| o.amount)
    .sum();

// Chaining with Option/Result
let result: Result<Vec<_>, _> = items.iter()
    .map(|item| process(item))  // each returns Result
    .collect();                  // collect stops at first Err

// enumerate, zip, chain, take, skip, peekable, windows, chunks
for (i, item) in items.iter().enumerate() { /* ... */ }

let pairs: Vec<_> = keys.iter().zip(values.iter()).collect();
```

### Closure Best Practices

```rust
// Closures infer capture mode: &T, &mut T, or T (move)
let greeting = |name: &str| format!("Hello, {name}!");

// Use move when closure outlives current scope
let name = String::from("world");
let greet = move || println!("Hello, {name}!");
// name is moved into closure -- no longer accessible here

// Prefer closures over function pointers when you need to capture state
// Prefer fn items when no capture is needed (zero overhead, nameable)
```

### Lazy Evaluation

```rust
// Iterators are lazy -- no work until consumed
let iter = (0..1_000_000)
    .filter(|n| n % 2 == 0)
    .map(|n| n * n);
// Nothing computed yet

let first_ten: Vec<_> = iter.take(10).collect();  // only computes 10 values
```

## String Handling

### &str vs String

```rust
// &str: borrowed, read-only view -- prefer in function signatures
fn greet(name: &str) { println!("Hello, {name}!"); }

// String: owned, growable -- use when you need ownership
fn make_greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

// Accept impl AsRef<str> or Into<String> for flexible APIs
fn set_name(&mut self, name: impl Into<String>) {
    self.name = name.into();  // works with &str, String, Cow<str>
}
```

### Cow (Clone on Write)

```rust
use std::borrow::Cow;

// Returns borrowed data when possible, owned when modification needed
fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains(' ') {
        Cow::Owned(input.replace(' ', "_"))
    } else {
        Cow::Borrowed(input)  // zero allocation
    }
}
```

Use `Cow<str>` when: returning data that's usually a borrow but sometimes requires allocation, avoiding unnecessary clones.

### Formatting

```rust
// format! for String construction
let msg = format!("User {name} (id={id})");

// Display vs Debug
// Display ({}) -- user-facing, human-readable, implement manually
// Debug ({:?}) -- developer-facing, derive it: #[derive(Debug)]
// Alternate Debug ({:#?}) -- pretty-printed with indentation

// write! to an existing buffer to avoid allocation
use std::fmt::Write;
let mut buf = String::new();
write!(buf, "count: {}", count)?;
```

## Conversion Traits

### From / Into

```rust
// Implement From -- Into comes for free
impl From<Config> for ServerConfig {
    fn from(cfg: Config) -> Self {
        Self { host: cfg.host, port: cfg.port }
    }
}

// Usage: both work
let server_cfg = ServerConfig::from(cfg);
let server_cfg: ServerConfig = cfg.into();

// From<T> for T is always implemented (identity conversion)
// Use Into<T> in bounds, From<T> in implementations
fn new(addr: impl Into<String>) -> Self { /* ... */ }
```

### TryFrom / TryInto

```rust
impl TryFrom<i64> for Port {
    type Error = PortError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let port = u16::try_from(value).map_err(|_| PortError::OutOfRange)?;
        if port == 0 { return Err(PortError::Zero); }
        Ok(Port(port))
    }
}
```

### AsRef / AsMut

```rust
// Cheap reference-to-reference conversion
fn read_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<u8>> {
    std::fs::read(path.as_ref())
}
// Accepts &str, String, PathBuf, &Path, OsString, etc.
```

### Deref / DerefMut

```rust
// Smart pointer pattern -- gives transparent access to inner type
use std::ops::Deref;

struct Email(String);

impl Deref for Email {
    type Target = str;
    fn deref(&self) -> &str { &self.0 }
}

let email = Email("user@example.com".into());
println!("{}", email.len());  // str methods available directly
```

**Warning:** Don't abuse `Deref` for general inheritance. Use it only for smart-pointer-like wrappers.

## Common Anti-Patterns

### Excessive Cloning

```rust
// BAD: clone to silence borrow checker
let data = self.data.clone();
process(&data);

// GOOD: borrow instead
process(&self.data);

// If you must clone, ask: can I restructure ownership instead?
```

### Stringly Typed Code

```rust
// BAD
fn set_color(color: &str) { /* "red", "blue"... what about "banana"? */ }

// GOOD
enum Color { Red, Green, Blue, Custom(u8, u8, u8) }
fn set_color(color: Color) { /* ... */ }
```

### Overusing unwrap/expect in Library Code

```rust
// BAD: panics in library code
let val = map.get("key").unwrap();

// GOOD: propagate errors
let val = map.get("key").ok_or(MyError::MissingKey("key"))?;

// unwrap/expect are fine in:
// - Tests
// - Examples
// - Cases provably infallible (with a comment explaining why)
// - main() after all error handling is done
```

### Boolean Parameters

```rust
// BAD: what does `true` mean at the call site?
process(data, true, false);

// GOOD: use enums or named structs
enum Compression { Enabled, Disabled }
enum Encryption { Enabled, Disabled }
process(data, Compression::Enabled, Encryption::Disabled);
```

### Premature Abstraction

```rust
// BAD: trait + impl for a single concrete type
trait DataStore { fn get(&self, id: &str) -> Option<Data>; }
struct PostgresStore;
impl DataStore for PostgresStore { /* ... */ }

// GOOD (if only one implementation): just use the struct directly
struct PostgresStore;
impl PostgresStore {
    fn get(&self, id: &str) -> Option<Data> { /* ... */ }
}
// Extract a trait later when you actually need polymorphism
```

For API design principles (public API surface, generics, method naming), see `references/api-design.md`.

For performance patterns (profiling, allocation reduction, inlining, Cow), see `references/performance.md`.

For Cargo, Clippy, and tooling best practices, see `references/cargo-tooling.md`.

For documentation and rustdoc patterns, see `references/documentation.md`.
