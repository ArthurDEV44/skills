# Async Traits Internals: Future, Pin, Unpin, Stream

## The Future Trait

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
```

- `Output`: the resolved value type
- `poll()`: called by the runtime to check if the future is ready
- Returns `Poll::Ready(T)` or `Poll::Pending`
- Never poll a future again after it returns `Ready` (may panic)

### Poll enum

```rust
pub enum Poll<T> {
    Ready(T),
    Pending,
}
```

### How await compiles

```rust
// This:
let value = my_future.await;

// Becomes roughly:
loop {
    match my_future.poll(cx) {
        Poll::Ready(value) => break value,
        Poll::Pending => { /* yield to runtime */ }
    }
}
```

The runtime handles the scheduling: pausing, resuming, and interleaving poll calls across multiple futures.

### Async state machines

Each `async fn` compiles into a state machine enum with one variant per await point:

```rust
// Conceptual -- compiler generates this
enum FetchTitleFuture<'a> {
    Initial { url: &'a str },
    AfterGet { url: &'a str },
    AfterText { response_text: String },
}
```

The compiler manages state transitions and borrows automatically.

## Pin and Unpin

### The problem

Async state machines can contain **self-referential data** (a field borrowing another field). If the struct moves in memory, these internal references become dangling.

### Pin prevents movement

`Pin<P>` wraps a pointer type `P` and guarantees the pointed-to value won't move:

```rust
// Pin the future so it won't move in memory
let pinned = Pin::new(Box::new(some_future));
```

- `Pin<Box<T>>` pins the `T`, not the `Box`
- The `Box` pointer itself can move; the heap-allocated data stays put

### Unpin marker trait

- Most types are `Unpin` (safe to move even when pinned): `String`, `Vec`, `i32`, etc.
- Async state machines may be `!Unpin` (not safe to move) due to self-references
- `Unpin` only matters when using `Pin<&mut T>`

### Practical usage: pin! macro

When using `join_all` with dynamic future collections:

```rust
use std::pin::pin;

// pin! pins the future on the stack
let fut_a = pin!(async { do_a().await });
let fut_b = pin!(async { do_b().await });
let fut_c = pin!(async { do_c().await });

let futures: Vec<Pin<&mut dyn Future<Output = ()>>> =
    vec![fut_a, fut_b, fut_c];

trpl::join_all(futures).await;
```

Alternative with `Box::pin` (heap-allocated, works across scopes):

```rust
let futures: Vec<Pin<Box<dyn Future<Output = ()>>>> = vec![
    Box::pin(async { do_a().await }),
    Box::pin(async { do_b().await }),
];
trpl::join_all(futures).await;
```

### Common Pin error

```
error[E0277]: `dyn Future<Output = ()>` cannot be unpinned
```

Fix: wrap futures with `pin!()` or `Box::pin()`.

## The Stream Trait

```rust
trait Stream {
    type Item;
    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>>;
}
```

Combines Iterator (sequence of items) with Future (readiness over time):

| Trait | Yields | Readiness |
|-------|--------|-----------|
| Iterator | `Option<Item>` | Immediate |
| Future | `Output` | Async (Poll) |
| Stream | `Poll<Option<Item>>` | Async sequence |

### StreamExt

Extension trait providing convenience methods on top of Stream:

```rust
trait StreamExt: Stream {
    async fn next(&mut self) -> Option<Self::Item>
    where Self: Unpin;
    // + filter, map, fold, collect, etc.
}
```

- Auto-implemented for all `Stream` types
- Must be imported: `use trpl::StreamExt;`
- Not yet in std (provided by `futures` / `tokio-stream` crates)

### Creating streams

```rust
use trpl::StreamExt;

// From iterator
let stream = trpl::stream_from_iter(vec![1, 2, 3]);

// From async channel receiver (already a stream)
let (tx, rx) = trpl::channel();
// rx implements Stream
```

## Sources

- [Traits for async](https://doc.rust-lang.org/book/ch17-05-traits-for-async.html)
- [Streams](https://doc.rust-lang.org/book/ch17-04-streams.html)
