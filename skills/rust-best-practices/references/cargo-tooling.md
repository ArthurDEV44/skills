# Cargo & Tooling Best Practices

## Cargo.toml Structure

```toml
[package]
name = "my-crate"
version = "0.1.0"
edition = "2024"              # latest stable edition
rust-version = "1.85"         # MSRV -- minimum supported Rust version
description = "Brief description"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"
categories = ["web-programming"]
keywords = ["http", "server"]

[dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
insta = "1"   # snapshot testing

[features]
default = []
json = ["dep:serde_json"]

[lints]
workspace = true              # inherit from workspace

[[bench]]
name = "benchmarks"
harness = false               # use criterion instead of built-in
```

## Workspace Configuration

```toml
# Root Cargo.toml
[workspace]
members = ["crates/*"]
resolver = "3"                # edition 2024 default

[workspace.package]
edition = "2024"
rust-version = "1.85"
license = "MIT"
repository = "https://github.com/user/project"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
thiserror = "2"
anyhow = "1"
tracing = "0.1"

[workspace.lints.rust]
unsafe_code = "forbid"
unused_must_use = "deny"

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
# Allow specific pedantic lints that are too noisy
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
```

Member crates inherit:

```toml
# crates/my-crate/Cargo.toml
[package]
name = "my-crate"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
serde.workspace = true

[lints]
workspace = true
```

## Clippy Configuration

### Recommended Lint Groups

```toml
# In Cargo.toml or workspace root
[lints.clippy]
all = "warn"                  # standard lints
pedantic = "warn"             # opinionated but useful
nursery = "warn"              # newer lints, may have false positives

# Fine-tune noisy pedantic lints
module_name_repetitions = "allow"   # http::HttpClient is fine
must_use_candidate = "allow"        # too many false positives
missing_errors_doc = "allow"        # sometimes obvious
missing_panics_doc = "allow"        # too noisy
```

### Key Individual Lints

```toml
[lints.clippy]
# Correctness
unwrap_used = "warn"          # prefer ? or expect with message
expect_used = "warn"          # in library code, prefer Result
panic = "warn"                # no panics in library code

# Performance
needless_collect = "warn"     # unnecessary .collect() before iteration
large_enum_variant = "warn"   # Box large variants
unnecessary_to_owned = "warn" # .to_string() when &str works

# Style
enum_glob_use = "warn"        # prefer explicit imports
wildcard_imports = "warn"     # prefer explicit imports
manual_let_else = "warn"      # suggest let-else pattern
```

### Running Clippy

```bash
# Basic
cargo clippy

# All targets (tests, benches, examples)
cargo clippy --all-targets --all-features

# Fix automatically where possible
cargo clippy --fix

# In CI (fail on warnings)
cargo clippy --all-targets --all-features -- -D warnings
```

## Build Profiles

```toml
# Fast debug builds
[profile.dev]
opt-level = 0
debug = true

# Faster dev builds for dependencies
[profile.dev.package."*"]
opt-level = 2

# Maximum optimization for release
[profile.release]
opt-level = 3
lto = true           # link-time optimization
codegen-units = 1    # slower compile, faster binary
strip = true         # strip debug symbols
panic = "abort"      # smaller binary, no unwinding

# Release with debug info for profiling
[profile.profiling]
inherits = "release"
debug = true
strip = false
```

## Essential Cargo Commands

```bash
# Development
cargo check                   # fast type checking (no codegen)
cargo build                   # debug build
cargo build --release         # optimized build
cargo run                     # build and run
cargo test                    # run all tests
cargo test -- --nocapture     # show println! output
cargo test -p my-crate        # test specific crate

# Quality
cargo clippy                  # linting
cargo fmt                     # formatting
cargo fmt -- --check          # check formatting (CI)
cargo doc --open              # generate and view docs
cargo audit                   # security vulnerability scan
cargo deny check              # license and advisory checks

# Analysis
cargo tree                    # dependency tree
cargo tree -d                 # show duplicate dependencies
cargo tree -i some-crate      # why is this crate included?
cargo bloat --release         # binary size analysis
cargo llvm-lines --release    # monomorphization bloat

# Testing
cargo test --release          # test with optimizations
cargo test -- --test-threads=1  # sequential tests
cargo bench                   # run benchmarks
cargo fuzz run target         # fuzz testing
cargo miri test               # detect undefined behavior
```

## Edition Management

| Edition | Key Features |
|---------|-------------|
| 2015 | Original |
| 2018 | NLL, `dyn` keyword, module system changes, `async`/`await` |
| 2021 | Resolver v2, disjoint capture in closures, IntoIterator for arrays |
| 2024 | Resolver v3, MSRV-aware resolver, `unsafe_op_in_unsafe_fn` lint, gen blocks (nightly) |

```bash
# Migrate to latest edition
cargo fix --edition
# Then update edition = "2024" in Cargo.toml
```

## MSRV (Minimum Supported Rust Version)

```toml
[package]
rust-version = "1.85"  # declare in Cargo.toml
```

```bash
# Verify MSRV compliance
cargo msrv verify

# Find minimum version that compiles
cargo msrv find

# CI: test with MSRV
rustup install 1.85.0
cargo +1.85.0 check
```

Guidelines:
- Libraries: support at least 2-3 most recent stable versions
- Applications: latest stable is fine
- Increment MSRV in a minor version bump, never a patch

## CI Pipeline Essentials

```bash
# Minimum CI checks
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo doc --no-deps

# Additional quality checks
cargo audit                    # security advisories
cargo deny check               # license compliance
cargo test --release           # catch release-only bugs
```
