---
name: rust-async
description: >
  Rust async/await programming patterns, futures, streams, concurrency, and runtime usage.
  Covers async fn, .await syntax, Future trait, Poll, Pin/Unpin, spawn_task, join, select,
  async channels, Stream/StreamExt, yield_now, timeout patterns, and threads vs async tradeoffs.
  Use when writing, reviewing, or refactoring Rust async code: (1) Writing async functions or
  blocks, (2) Spawning and joining async tasks, (3) Using channels for message passing between
  async tasks, (4) Working with streams and async iteration, (5) Choosing between threads and
  async, (6) Fixing Pin/Unpin compiler errors, (7) Building timeout or racing patterns,
  (8) Combining threads with async for mixed workloads.
---

# Rust Async

## Core Syntax

### async fn and .await

```rust
// async fn returns impl Future<Output = T>
async fn fetch_title(url: &str) -> Option<String> {
    let text = trpl::get(url).await.text().await;
    Html::parse(&text)
        .select_first("title")
        .map(|t| t.inner_html())
}
```

- `async fn` compiles to a function returning `impl Future`
- `.await` is **postfix** -- chain it: `get(url).await.text().await`
- Futures are **lazy** -- nothing runs until awaited
- Each `.await` is a yield point where the runtime can switch tasks

### async blocks

```rust
let fut = async {
    do_something().await;
    42 // resolves to i32
};

let fut_move = async move {
    // takes ownership of captured variables
    tx.send(val).unwrap();
};
```

### Running async code

`main` cannot be `async`. Use a runtime's `block_on`:

```rust
fn main() {
    trpl::block_on(async {
        let result = fetch_title("https://example.com").await;
        println!("{result:?}");
    });
}
```

## Concurrency Patterns

### spawn_task -- fire-and-forget with join handle

```rust
let handle = trpl::spawn_task(async {
    trpl::sleep(Duration::from_secs(1)).await;
    "done"
});
// ... do other work ...
let result = handle.await.unwrap();
```

Task is cancelled when the runtime shuts down unless you `.await` the handle.

### join -- run futures concurrently, wait for all

```rust
// Two futures
trpl::join(fut_a, fut_b).await;

// N futures (known at compile time)
trpl::join!(fut_a, fut_b, fut_c);

// Dynamic collection (requires Pin)
use std::pin::pin;
let futures: Vec<Pin<&mut dyn Future<Output = ()>>> =
    vec![pin!(fut_a), pin!(fut_b), pin!(fut_c)];
trpl::join_all(futures).await;
```

`join` is **fair** -- polls each future equally, producing deterministic interleaving.

### select -- race futures, take first result

```rust
use trpl::Either;

match trpl::select(fut_a, fut_b).await {
    Either::Left(val_a)  => { /* fut_a won */ }
    Either::Right(val_b) => { /* fut_b won */ }
}
```

`select` is **not fair** -- always polls arguments in order. First argument is checked first.

### Timeout pattern

```rust
async fn timeout<F: Future>(
    future: F,
    max_time: Duration,
) -> Result<F::Output, Duration> {
    match trpl::select(future, trpl::sleep(max_time)).await {
        Either::Left(output) => Ok(output),
        Either::Right(_) => Err(max_time),
    }
}
```

### Yielding control

```rust
// Give runtime a chance to switch tasks (no actual sleep)
trpl::yield_now().await;

// Avoid starving other futures in CPU-bound loops:
for chunk in data.chunks(100) {
    process(chunk);
    trpl::yield_now().await;
}
```

Without yield points, a future **blocks all other futures** on the same task (cooperative multitasking).

## Message Passing (Async Channels)

```rust
let (tx, mut rx) = trpl::channel();

let sender = async move {  // move tx into block so it drops on completion
    tx.send("hello").unwrap();
    trpl::sleep(Duration::from_millis(500)).await;
    tx.send("world").unwrap();
};

let receiver = async {
    while let Some(msg) = rx.recv().await {
        println!("got: {msg}");
    }
    // loop exits when all senders are dropped
};

trpl::join(sender, receiver).await;
```

- `async move` ensures `tx` is dropped when the block ends, closing the channel
- Clone `tx` for multiple producers: `let tx2 = tx.clone();`
- `rx.recv().await` returns `None` when all senders are dropped

## Streams

For detailed Stream/StreamExt trait internals, see [references/traits-internals.md](references/traits-internals.md).

```rust
use trpl::StreamExt; // required for .next()

let mut stream = trpl::stream_from_iter([1, 2, 3].iter().map(|n| n * 2));

while let Some(value) = stream.next().await {
    println!("{value}");
}
```

- Stream = async Iterator (items arrive over time)
- Import `StreamExt` to use `.next()`, `.filter()`, `.map()`, etc.
- `while let Some(v) = stream.next().await` is the async iteration pattern (no `for` loop support yet)

## Threads vs Async

For full comparison and mixing patterns, see [references/threads-vs-async.md](references/threads-vs-async.md).

| | Threads | Async |
|---|---------|-------|
| Best for | CPU-bound, parallelizable work | I/O-bound, concurrent work |
| Overhead | ~MB per thread (OS stack) | Lightweight tasks (KB) |
| Scheduling | OS preemptive | Runtime cooperative |
| Availability | Requires OS support | Works on embedded (no_std) |

Combine both when needed:

```rust
let (tx, mut rx) = trpl::channel();

thread::spawn(move || {           // CPU-heavy work in thread
    for i in 1..=10 {
        tx.send(i).unwrap();
        thread::sleep(Duration::from_secs(1));
    }
});

trpl::block_on(async {            // I/O handling in async
    while let Some(msg) = rx.recv().await {
        println!("{msg}");
    }
});
```

## Official Documentation

- [Async/Await overview](https://doc.rust-lang.org/book/ch17-00-async-await.html)
- [Futures and syntax](https://doc.rust-lang.org/book/ch17-01-futures-and-syntax.html)
- [Concurrency with async](https://doc.rust-lang.org/book/ch17-02-concurrency-with-async.html)
- [More futures patterns](https://doc.rust-lang.org/book/ch17-03-more-futures.html)
- [Streams](https://doc.rust-lang.org/book/ch17-04-streams.html)
- [Traits for async](https://doc.rust-lang.org/book/ch17-05-traits-for-async.html)
- [Futures, tasks, and threads](https://doc.rust-lang.org/book/ch17-06-futures-tasks-threads.html)
