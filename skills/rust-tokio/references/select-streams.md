# tokio::select! and Streams

## select! Macro Details

### Basic syntax

```rust
tokio::select! {
    <pattern> = <async expression> => <handler>,
    <pattern> = <async expression> => <handler>,
    else => { /* when no patterns match */ }
}
```

All async expressions run concurrently on the **same task** (no spawn). When one completes, the rest are **dropped** (cancelled).

### Pattern matching

```rust
tokio::select! {
    Some(v) = rx1.recv() => { println!("rx1: {v}"); }
    Some(v) = rx2.recv() => { println!("rx2: {v}"); }
    else => { println!("both channels closed"); }
}
```

If a pattern doesn't match (e.g., `recv()` returns `None`), remaining branches continue.

### Borrowing advantage over spawn

select! branches can borrow data (no 'static required):

```rust
async fn race(data: &[u8], addr1: SocketAddr, addr2: SocketAddr) -> io::Result<()> {
    tokio::select! {
        Ok(_) = async {
            let mut s = TcpStream::connect(addr1).await?;
            s.write_all(data).await?;  // borrows data
            Ok::<_, io::Error>(())
        } => {}
        Ok(_) = async {
            let mut s = TcpStream::connect(addr2).await?;
            s.write_all(data).await?;  // borrows data
            Ok::<_, io::Error>(())
        } => {}
        else => {}
    };
    Ok(())
}
```

Handlers can mutably borrow shared data since only one handler runs.

### Loops with select!

```rust
loop {
    let msg = tokio::select! {
        Some(msg) = rx1.recv() => msg,
        Some(msg) = rx2.recv() => msg,
        else => break,
    };
    process(msg);
}
```

### Resuming a future across loop iterations

```rust
let operation = some_async_op();
tokio::pin!(operation);  // pin! required for &mut usage

loop {
    tokio::select! {
        result = &mut operation => {
            // operation completed
            break;
        }
        Some(v) = rx.recv() => {
            // handle incoming message
            if should_restart(v) {
                operation.set(some_async_op()); // reset the future
            }
        }
    }
}
```

### Branch preconditions

```rust
let mut done = false;
tokio::select! {
    val = operation(), if !done => {
        done = true;
        handle(val);
    }
    msg = rx.recv() => { /* always active */ }
}
```

Precondition checked **before** polling. Disabled branches are skipped.

### Return values

All handlers must return the same type:

```rust
let output = tokio::select! {
    v = future_a => v,     // returns v
    v = future_b => v + 1, // returns v + 1
};
```

### Error handling

`?` in async expressions binds to the pattern; `?` in handlers propagates out:

```rust
tokio::select! {
    res = async {
        listener.accept().await?;
        Ok::<_, io::Error>(())
    } => { res?; }  // ? here propagates out of select!
    _ = shutdown_signal => {}
}
```

## Streams (tokio-stream)

```toml
# Cargo.toml
tokio-stream = "0.1"
```

### Basic iteration

```rust
use tokio_stream::StreamExt;

let mut stream = tokio_stream::iter(&[1, 2, 3]);
while let Some(v) = stream.next().await {
    println!("{v}");
}
```

### Pinning requirement

```rust
let messages = subscriber.into_stream();
tokio::pin!(messages); // required before calling .next()

while let Some(msg) = messages.next().await { /* ... */ }
```

### Stream adapters

Order matters: `filter` then `take` differs from `take` then `filter`.

```rust
let stream = subscriber.into_stream()
    .filter(|msg| match msg {
        Ok(m) if m.content.len() == 1 => true,
        _ => false,
    })
    .map(|msg| msg.unwrap().content)
    .take(3); // stop after 3 items
```

### Creating streams

```rust
// From iterator
let stream = tokio_stream::iter(vec![1, 2, 3]);

// Using async-stream crate
use async_stream::stream;
let s = stream! {
    for i in 0..3 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        yield i;
    }
};
```

### Stream trait

```rust
pub trait Stream {
    type Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Self::Item>>;
}
```

Use `StreamExt` (from tokio-stream or futures) for `.next()`, `.filter()`, `.map()`, etc.

## Sources

- [Select tutorial](https://tokio.rs/tokio/tutorial/select)
- [Streams tutorial](https://tokio.rs/tokio/tutorial/streams)
