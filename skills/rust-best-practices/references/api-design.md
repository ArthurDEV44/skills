# Rust API Design

## Public API Principles

### Minimize the Public Surface

```rust
// Expose only what users need
pub struct Client {
    // fields are private by default -- good
    config: Config,
    http: reqwest::Client,
}

// Public methods form the API contract
impl Client {
    pub fn new(config: Config) -> Self { /* ... */ }
    pub fn get(&self, url: &str) -> Result<Response, Error> { /* ... */ }

    // Internal helpers stay private
    fn build_request(&self, url: &str) -> Request { /* ... */ }
}
```

### Accept Generics, Return Concrete Types

```rust
// GOOD: accept flexible input
pub fn read_config(path: impl AsRef<Path>) -> Result<Config, Error> { /* ... */ }
pub fn set_name(&mut self, name: impl Into<String>) { self.name = name.into(); }

// GOOD: return concrete types -- callers know exactly what they get
pub fn new() -> Client { /* ... */ }

// Use impl Trait in return position for iterators and closures
pub fn active_users(&self) -> impl Iterator<Item = &User> {
    self.users.iter().filter(|u| u.is_active)
}
```

### Method Naming Conventions

| Convention | Example | Returns |
|-----------|---------|---------|
| `new` | `Vec::new()` | Constructors |
| `with_*` | `Vec::with_capacity(10)` | Constructor with parameter |
| `from_*` | `String::from_utf8(bytes)` | Conversion constructor |
| `into_*` | `vec.into_iter()` | Consuming conversion (takes self) |
| `as_*` | `slice.as_ptr()` | Cheap reference conversion (borrows) |
| `to_*` | `str.to_string()` | Expensive conversion (allocates) |
| `is_*` | `path.is_dir()` | Boolean query |
| `has_*` | `set.has_key(&k)` | Boolean containment |
| `try_*` | `u8::try_from(256)` | Fallible operation |
| `*_mut` | `vec.iter_mut()` | Mutable variant |
| `set_*` | `config.set_port(8080)` | Setter |

### Borrowing Conventions

| Method takes | Signature | Meaning |
|-------------|-----------|---------|
| `&self` | `fn len(&self) -> usize` | Read-only access |
| `&mut self` | `fn push(&mut self, val: T)` | Mutates in place |
| `self` | `fn into_inner(self) -> T` | Consumes, transforms |

### Return Type Conventions

```rust
// Fallible operations: return Result<T, E>
pub fn parse(input: &str) -> Result<Config, ParseError> { /* ... */ }

// Optional values: return Option<T>
pub fn find(&self, id: &str) -> Option<&Item> { /* ... */ }

// Never use bool for error states
// BAD: fn delete(id: &str) -> bool
// GOOD: fn delete(id: &str) -> Result<(), DeleteError>
```

## Generics Best Practices

### Minimal Trait Bounds

```rust
// BAD: over-constrained -- requires more than needed
fn process<T: Clone + Debug + Display + Send + Sync>(item: T) {
    println!("{item}");
}

// GOOD: minimal bounds
fn process<T: Display>(item: T) {
    println!("{item}");
}
```

### Where Clauses for Readability

```rust
// Inline bounds get noisy with complex constraints
// GOOD: use where clause
fn merge<K, V, M>(map: &M, key: K, value: V) -> Result<(), Error>
where
    K: Eq + Hash + Display,
    V: Serialize,
    M: MutableMap<K, V>,
{
    // ...
}
```

### Generic Return Type Flexibility

```rust
// Let callers choose the collection type
fn collect_names<B: FromIterator<String>>(users: &[User]) -> B {
    users.iter().map(|u| u.name.clone()).collect()
}

let vec: Vec<String> = collect_names(&users);
let set: HashSet<String> = collect_names(&users);
```

## Preventing Misuse

### Use the Type System

```rust
// BAD: easy to mix up parameters
fn transfer(from: u64, to: u64, amount: f64) { /* ... */ }
transfer(to_id, from_id, amount);  // compiles but wrong!

// GOOD: newtypes prevent mixup
struct AccountId(u64);
struct Amount(f64);
fn transfer(from: AccountId, to: AccountId, amount: Amount) { /* ... */ }
```

### Non-Exhaustive Enums and Structs

```rust
// For public enums that may gain variants
#[non_exhaustive]
pub enum Error {
    NotFound,
    Timeout,
    InvalidInput(String),
}
// Forces external callers to have a _ wildcard arm

// For public structs that may gain fields
#[non_exhaustive]
pub struct Config {
    pub host: String,
    pub port: u16,
}
// External code can't construct Config directly -- must use a builder/constructor
```

### Must-Use Types

```rust
#[must_use = "this Result may contain an error that should be handled"]
pub fn save(&self) -> Result<(), SaveError> { /* ... */ }

#[must_use]
pub struct Transaction { /* ... */ }
// Compiler warns if return value is ignored
```

## Error Type Design

```rust
// Use thiserror for library errors
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("invalid response: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("rate limited, retry after {retry_after}s")]
    RateLimited { retry_after: u64 },

    #[error("{0}")]
    Other(String),
}

// Guidelines:
// - One error enum per module or logical boundary
// - Implement From for automatic ? conversion
// - Provide enough context for callers to handle each variant differently
// - Use #[non_exhaustive] on public error enums
// - Use anyhow::Error for applications (not libraries)
```

## Deprecation

```rust
#[deprecated(since = "2.0.0", note = "Use `new_method` instead")]
pub fn old_method(&self) { /* ... */ }
```

## Feature Flags for Optional Functionality

```rust
// In Cargo.toml
[features]
default = ["json"]
json = ["dep:serde_json"]
xml = ["dep:quick-xml"]

// In code
#[cfg(feature = "json")]
pub fn to_json(&self) -> Result<String, serde_json::Error> { /* ... */ }
```

Keep features additive (enabling a feature should never remove functionality).
