# Rust Documentation Best Practices

## Rustdoc Basics

### Item Documentation

```rust
/// Creates a new [`Server`] with the given configuration.
///
/// The server will listen on the specified address and port. Use
/// [`ServerBuilder`] for more fine-grained configuration.
///
/// # Errors
///
/// Returns [`ServerError::Bind`] if the address is already in use.
///
/// # Examples
///
/// ```
/// let server = Server::new("127.0.0.1:8080")?;
/// server.run().await?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn new(addr: &str) -> Result<Self, ServerError> {
    // ...
}
```

### Module Documentation

```rust
//! HTTP client for the API.
//!
//! This module provides a type-safe client for interacting with
//! the REST API. The main entry point is [`Client::new`].
//!
//! # Quick Start
//!
//! ```rust
//! use mylib::client::Client;
//!
//! let client = Client::new("https://api.example.com")?;
//! let users = client.list_users().await?;
//! ```
```

## Documentation Sections

Use these standard sections in order:

```rust
/// Brief one-line description.
///
/// Extended description with more context. Explain **why** this exists,
/// not just what it does.
///
/// # Panics
///
/// Document conditions that cause panics (if any).
///
/// # Errors
///
/// Document each error variant that can be returned.
///
/// # Safety
///
/// For `unsafe` functions: document invariants the caller must uphold.
///
/// # Examples
///
/// ```
/// // working example that compiles and runs as a doc test
/// ```
```

### Panics Section

```rust
/// # Panics
///
/// Panics if `index` is out of bounds.
pub fn get(&self, index: usize) -> &T {
    &self.data[index]
}
```

### Errors Section

```rust
/// # Errors
///
/// - [`ParseError::InvalidFormat`] if the input is not valid JSON
/// - [`ParseError::MissingField`] if a required field is absent
pub fn parse(input: &str) -> Result<Config, ParseError> { /* ... */ }
```

### Safety Section

```rust
/// # Safety
///
/// - `ptr` must be a valid pointer to a `T` that was allocated by this allocator.
/// - `ptr` must not have been previously deallocated.
/// - The caller must ensure no other references to the pointed-to value exist.
pub unsafe fn dealloc<T>(ptr: *mut T) { /* ... */ }
```

## Doc Examples That Compile

All examples in `///` blocks are compiled and run as tests (`cargo test --doc`):

```rust
/// Parses a duration string like "5s", "100ms", or "2m".
///
/// # Examples
///
/// ```
/// use mylib::parse_duration;
///
/// let d = parse_duration("5s").unwrap();
/// assert_eq!(d, std::time::Duration::from_secs(5));
///
/// let d = parse_duration("100ms").unwrap();
/// assert_eq!(d, std::time::Duration::from_millis(100));
/// ```
///
/// Invalid input returns an error:
///
/// ```
/// use mylib::parse_duration;
///
/// assert!(parse_duration("not_a_duration").is_err());
/// ```
pub fn parse_duration(input: &str) -> Result<Duration, ParseError> { /* ... */ }
```

### Hidden Lines in Examples

```rust
/// ```
/// # use mylib::Config;         // hidden setup line (# prefix)
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Config::from_file("config.toml")?;
/// assert!(config.is_valid());
/// # Ok(())
/// # }
/// ```
```

Lines starting with `#` are compiled but not shown in documentation.

### Examples That Should Not Run

```rust
/// ```no_run
/// // Compiles but doesn't execute (e.g., network calls, infinite loops)
/// let server = Server::new("0.0.0.0:80")?;
/// server.run_forever().await;
/// ```

/// ```ignore
/// // Not compiled at all (last resort -- prefer no_run)
/// // Only use for pseudo-code or intentionally incomplete snippets
/// ```

/// ```compile_fail
/// // Verified to NOT compile (useful for showing what's forbidden)
/// let x: u32 = "not a number";
/// ```

/// ```should_panic
/// // Expected to panic
/// vec![1, 2, 3][99];
/// ```
```

## Linking in Documentation

```rust
/// Converts this [`Config`] into a [`Server`] using [`Server::from_config`].
///
/// See the [module-level documentation](crate::server) for architecture details.
///
/// Related: [`ServerBuilder`], [`ServerError`]
pub fn into_server(self) -> Result<Server, ServerError> { /* ... */ }
```

Link syntax:
- `[`Type`]` -- link to a type in scope
- `[`mod::Type`]` -- link to a type in another module
- `[`crate::module`]` -- absolute path from crate root
- `[display text](`Type`)` -- custom display text

## Crate-Level Documentation

In `src/lib.rs`:

```rust
//! # My Crate
//!
//! `my_crate` provides high-performance widgets for the discerning developer.
//!
//! ## Quick Start
//!
//! ```rust
//! use my_crate::Widget;
//!
//! let w = Widget::new("example");
//! w.activate()?;
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `json` | Enable JSON serialization |
//! | `async` | Enable async API |
//!
//! ## Modules
//!
//! - [`widget`] -- Core widget types and operations
//! - [`config`] -- Configuration and builder
```

## Documentation Anti-Patterns

```rust
// BAD: restates the function name
/// Gets the name.
pub fn get_name(&self) -> &str { /* ... */ }

// GOOD: explains what, why, and any nuances
/// Returns the display name for this user, falling back to email if unset.
pub fn display_name(&self) -> &str { /* ... */ }

// BAD: no examples, no error docs
/// Parses the input.
pub fn parse(input: &str) -> Result<Data, Error> { /* ... */ }

// GOOD: complete documentation
/// Parses a TOML-formatted string into a [`Data`] structure.
///
/// # Errors
///
/// Returns [`ParseError::Syntax`] if the input is malformed TOML.
///
/// # Examples
///
/// ```
/// let data = parse("[section]\nkey = \"value\"")?;
/// assert_eq!(data.get("section.key"), Some("value"));
/// # Ok::<(), mylib::ParseError>(())
/// ```
pub fn parse(input: &str) -> Result<Data, ParseError> { /* ... */ }
```

## Generating Documentation

```bash
# Build and open docs
cargo doc --open

# Include private items (for internal review)
cargo doc --document-private-items

# Run doc tests
cargo test --doc

# Check for broken links
cargo doc --no-deps 2>&1 | grep "warning"
```
