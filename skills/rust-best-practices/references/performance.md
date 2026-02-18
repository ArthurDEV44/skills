# Rust Performance Patterns

## Profiling First

Never optimize without measuring. Tools:

- **cargo bench** + **criterion**: micro-benchmarks with statistical rigor
- **flamegraph** (`cargo flamegraph`): visualize CPU hotspots
- **perf** / **samply**: system-level profiling on Linux
- **DHAT** (`cargo run --features dhat-heap`): heap allocation profiling
- **cargo-bloat**: find what's taking space in your binary
- **cargo-llvm-lines**: find generic monomorphization bloat

```bash
# Quick benchmark
cargo bench

# Flamegraph (Linux)
cargo flamegraph --bin myapp

# Check binary size contributors
cargo bloat --release -n 20
```

## Avoiding Allocations

### Pre-allocate Collections

```rust
// BAD: grows incrementally, multiple reallocations
let mut v = Vec::new();
for i in 0..1000 {
    v.push(i);
}

// GOOD: single allocation
let mut v = Vec::with_capacity(1000);
for i in 0..1000 {
    v.push(i);
}

// Also works for String, HashMap, HashSet
let mut s = String::with_capacity(expected_len);
let mut m = HashMap::with_capacity(expected_entries);
```

### Reuse Buffers

```rust
// BAD: allocates new String each iteration
for line in lines {
    let processed = format!("prefix: {line}");
    output.push(processed);
}

// GOOD: reuse buffer
let mut buf = String::new();
for line in lines {
    buf.clear();
    write!(buf, "prefix: {line}").unwrap();
    output.push(buf.clone());  // or process in place
}
```

### Use &str Instead of String in Intermediate Steps

```rust
// BAD: unnecessary allocation
fn is_valid(input: &str) -> bool {
    let trimmed = input.trim().to_string();  // allocates!
    trimmed.len() > 3
}

// GOOD: borrow chain
fn is_valid(input: &str) -> bool {
    input.trim().len() > 3  // no allocation
}
```

### Cow for Conditional Allocation

```rust
use std::borrow::Cow;

fn escape_html(input: &str) -> Cow<'_, str> {
    if input.contains('&') || input.contains('<') || input.contains('>') {
        Cow::Owned(
            input.replace('&', "&amp;")
                 .replace('<', "&lt;")
                 .replace('>', "&gt;")
        )
    } else {
        Cow::Borrowed(input)  // zero-cost when no escaping needed
    }
}
```

## Iterator vs Loop Performance

Iterator chains compile to the same code as hand-written loops (zero-cost abstraction):

```rust
// These produce identical assembly:
let sum: i64 = (0..1000).filter(|x| x % 2 == 0).sum();

let mut sum = 0i64;
for x in 0..1000 {
    if x % 2 == 0 { sum += x; }
}
```

Prefer iterators -- they're often **faster** because they enable LLVM auto-vectorization and avoid bounds checks.

## Stack vs Heap

```rust
// Stack: fixed size, fast, no allocation
let arr = [0u8; 1024];           // stack
let small = (1, 2, 3);           // stack

// Heap: dynamic size, slower allocation
let vec = vec![0u8; 1024];       // heap
let boxed = Box::new(LargeStruct { .. }); // heap

// SmallVec / ArrayVec: stack for small sizes, heap for overflow
use smallvec::SmallVec;
let mut sv: SmallVec<[u8; 16]> = SmallVec::new();  // stack until > 16 elements
```

## Inlining

```rust
// #[inline] -- hint to inline across crate boundaries
#[inline]
pub fn is_whitespace(c: char) -> bool {
    c == ' ' || c == '\t' || c == '\n'
}

// #[inline(always)] -- force inline (use sparingly, can bloat binary)
#[inline(always)]
fn hot_inner_loop_function() { /* ... */ }

// #[inline(never)] -- prevent inline (useful for cold error paths)
#[inline(never)]
#[cold]
fn handle_error(e: Error) { /* ... */ }
```

When to use `#[inline]`:
- Small functions called across crate boundaries
- Generic functions (already monomorphized, but hint helps)
- Hot loop bodies

## Avoiding Monomorphization Bloat

Each generic instantiation creates a separate copy of the function:

```rust
// BAD: 10 different types = 10 copies of this entire function
fn process<T: Display>(items: &[T]) {
    // lots of code...
    for item in items {
        println!("{item}");
    }
}

// GOOD: monomorphize only the type-specific part
fn process<T: Display>(items: &[T]) {
    fn process_inner(formatted: &[String]) {
        // lots of code using formatted strings...
    }
    let formatted: Vec<String> = items.iter().map(|i| i.to_string()).collect();
    process_inner(&formatted);
}
```

## Data Layout

```rust
// #[repr(C)] -- predictable layout, C-compatible
#[repr(C)]
struct Header {
    magic: u32,
    version: u16,
    flags: u16,
}

// Field ordering matters for size (padding)
// BAD: 24 bytes (due to padding)
struct Bad { a: u8, b: u64, c: u8 }

// GOOD: 16 bytes (Rust reorders by default, but be explicit for #[repr(C)])
struct Good { b: u64, a: u8, c: u8 }
```

Rust's default layout (`#[repr(Rust)]`) already optimizes field ordering. Use `#[repr(C)]` only for FFI.

## Key Performance Rules

1. **Measure first** -- don't guess where the bottleneck is
2. **Reduce allocations** -- pre-allocate, reuse buffers, use Cow
3. **Prefer iterators** -- zero-cost, enable vectorization
4. **Minimize cloning** -- borrow when possible, use `Arc` for shared ownership
5. **Use release mode** -- `cargo build --release` (10x-100x faster than debug)
6. **Enable LTO** for final binaries:

```toml
[profile.release]
lto = true
codegen-units = 1
strip = true
```
