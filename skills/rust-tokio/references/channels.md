# Tokio Channels: Patterns and Usage

## Channel Types Summary

### mpsc (multi-producer, single-consumer)

```rust
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::channel(32); // bounded, capacity 32
let tx2 = tx.clone();                 // clone for multiple producers

tokio::spawn(async move {
    tx.send("from task 1").await.unwrap();
});
tokio::spawn(async move {
    tx2.send("from task 2").await.unwrap();
});

// Receiver cannot be cloned
while let Some(msg) = rx.recv().await {
    println!("got: {msg}");
}
// recv() returns None when all senders are dropped
```

Bounded channels provide **backpressure**: `send().await` blocks when buffer is full.

### oneshot (single value, single use)

```rust
use tokio::sync::oneshot;

let (tx, rx) = oneshot::channel();
tx.send("response").unwrap();    // no .await needed, completes immediately
let val = rx.await.unwrap();     // await the single response
```

- Neither tx nor rx can be cloned
- `send()` fails if receiver was dropped (returns the value back)
- Use `let _ = tx.send(val);` when you don't care if receiver dropped

### broadcast (every consumer sees every message)

```rust
use tokio::sync::broadcast;

let (tx, mut rx1) = broadcast::channel(16);
let mut rx2 = tx.subscribe(); // additional receivers via subscribe()

tx.send("event").unwrap();
// Both rx1 and rx2 receive "event"
```

### watch (latest value only)

```rust
use tokio::sync::watch;

let (tx, mut rx) = watch::channel("initial");
tx.send("updated").unwrap();

// Receiver sees only the most recent value
let val = rx.borrow().clone();
// Wait for changes:
rx.changed().await.unwrap();
```

## Command Pattern with Oneshot Response

Define commands as enum with embedded response channel:

```rust
use tokio::sync::{mpsc, oneshot};
use bytes::Bytes;

type Responder<T> = oneshot::Sender<mini_redis::Result<T>>;

#[derive(Debug)]
enum Command {
    Get {
        key: String,
        resp: Responder<Option<Bytes>>,
    },
    Set {
        key: String,
        val: Bytes,
        resp: Responder<()>,
    },
}
```

### Sender side (request + await response)

```rust
let (resp_tx, resp_rx) = oneshot::channel();
let cmd = Command::Get {
    key: "foo".to_string(),
    resp: resp_tx,
};
tx.send(cmd).await.unwrap();
let result = resp_rx.await.unwrap(); // await the response
```

### Manager task (process commands)

```rust
let manager = tokio::spawn(async move {
    let mut client = Client::connect("127.0.0.1:6379").await.unwrap();

    while let Some(cmd) = rx.recv().await {
        match cmd {
            Command::Get { key, resp } => {
                let res = client.get(&key).await;
                let _ = resp.send(res); // ignore if receiver dropped
            }
            Command::Set { key, val, resp } => {
                let res = client.set(&key, val).await;
                let _ = resp.send(res);
            }
        }
    }
});
```

This pattern provides:
- Single connection management (no shared mutable access)
- Natural backpressure via bounded mpsc
- Foundation for connection pooling

## Backpressure Guidelines

- Always use bounded channels (`mpsc::channel(N)`)
- Pick bounds based on expected throughput and acceptable latency
- `send().await` will wait when buffer is full, propagating backpressure
- Unbounded channels (`mpsc::unbounded_channel()`) exist but risk OOM under load

## Sources

- [Channels tutorial](https://tokio.rs/tokio/tutorial/channels)
- [Shared state tutorial](https://tokio.rs/tokio/tutorial/shared-state)
