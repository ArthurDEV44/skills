# Advanced Async Patterns

## async fn in Traits (Rust 1.75+)

Native `async fn` in traits is supported since Rust 1.75:

```rust
trait HttpClient {
    async fn get(&self, url: &str) -> Result<Response, Error>;
    async fn post(&self, url: &str, body: &[u8]) -> Result<Response, Error>;
}

impl HttpClient for MyClient {
    async fn get(&self, url: &str) -> Result<Response, Error> {
        self.inner.get(url).send().await.map_err(Error::from)
    }
    async fn post(&self, url: &str, body: &[u8]) -> Result<Response, Error> {
        self.inner.post(url).body(body.to_vec()).send().await.map_err(Error::from)
    }
}
```

### Send bound limitation

`async fn` in traits does **not** guarantee the returned future is `Send`. This matters when the trait object is used with `tokio::spawn`:

```rust
// This WON'T compile if the future isn't Send:
fn spawn_fetch(client: &dyn HttpClient) {
    // ERROR: future is not Send
    tokio::spawn(client.get("https://example.com"));
}
```

**Fix with `#[trait_variant::make(SendHttpClient: Send)]`** (trait-variant crate):

```rust
use trait_variant::make;

#[make(SendHttpClient: Send)]
trait HttpClient {
    async fn get(&self, url: &str) -> Result<Response, Error>;
}

// SendHttpClient requires Send futures -- works with tokio::spawn
fn spawn_fetch(client: Arc<dyn SendHttpClient>) {
    tokio::spawn(async move {
        client.get("https://example.com").await.unwrap();
    });
}
```

**Alternative: manual desugaring with `Box<dyn Future + Send>`:**

```rust
trait HttpClient {
    fn get(&self, url: &str) -> Pin<Box<dyn Future<Output = Result<Response, Error>> + Send + '_>>;
}

impl HttpClient for MyClient {
    fn get(&self, url: &str) -> Pin<Box<dyn Future<Output = Result<Response, Error>> + Send + '_>> {
        Box::pin(async move {
            self.inner.get(url).send().await.map_err(Error::from)
        })
    }
}
```

## Send Bounds Across .await

Futures passed to `tokio::spawn` must be `Send`. A future is `Send` if all values held across `.await` points are `Send`:

```rust
// WON'T COMPILE: Rc is !Send
async fn bad() {
    let data = Rc::new(42);
    tokio::time::sleep(Duration::from_secs(1)).await; // data held across await
    println!("{data}");
}

// FIX 1: Use Arc instead of Rc
async fn good() {
    let data = Arc::new(42);
    tokio::time::sleep(Duration::from_secs(1)).await;
    println!("{data}");
}

// FIX 2: Drop !Send value before the await
async fn also_good() {
    {
        let data = Rc::new(42);
        println!("{data}");
    } // data dropped here
    tokio::time::sleep(Duration::from_secs(1)).await;
}

// FIX 3: Use a current_thread runtime (no Send requirement)
#[tokio::main(flavor = "current_thread")]
async fn main() {
    tokio::task::LocalSet::new().run_until(async {
        tokio::task::spawn_local(async {
            let data = Rc::new(42);
            tokio::time::sleep(Duration::from_secs(1)).await;
            println!("{data}");
        }).await.unwrap();
    }).await;
}
```

Common `!Send` types to watch for: `Rc`, `RefCell` (interior mutability without sync), `MutexGuard` from `std::sync::Mutex` (use `tokio::sync::Mutex` instead), raw pointers.

### Diagnosing Send errors

The compiler error often points to the wrong line. Look for:
1. Non-Send types alive across `.await` points
2. `MutexGuard` held across `.await` (use `tokio::sync::Mutex` or drop the guard before `.await`)
3. Trait objects without `+ Send` bound

```rust
// BAD: std::sync::MutexGuard held across .await
async fn bad(mutex: &std::sync::Mutex<Vec<u8>>) {
    let mut guard = mutex.lock().unwrap();
    guard.push(1);
    some_async_fn().await; // guard still alive here!
}

// GOOD: drop guard before await
async fn good(mutex: &std::sync::Mutex<Vec<u8>>) {
    {
        let mut guard = mutex.lock().unwrap();
        guard.push(1);
    } // guard dropped
    some_async_fn().await;
}

// GOOD: use tokio's async-aware Mutex
async fn also_good(mutex: &tokio::sync::Mutex<Vec<u8>>) {
    let mut guard = mutex.lock().await;
    guard.push(1);
    some_async_fn().await; // tokio MutexGuard is Send
}
```

## Cancellation Safety

A future is **cancellation safe** if dropping it at any `.await` point does not lose data or leave state inconsistent.

### select! and cancellation

`tokio::select!` drops the losing branches. If a future has done partial work before its `.await`, that work is lost:

```rust
// UNSAFE: read bytes may be lost on cancellation
loop {
    select! {
        n = reader.read(&mut buf) => {
            // If this branch loses, bytes may have been read into buf
            // but we never process them
            process(&buf[..n?]);
        }
        _ = cancel_token.cancelled() => break,
    }
}

// SAFE: use cancellation-safe read method
loop {
    select! {
        result = reader.read(&mut buf) => {
            process(&buf[..result?]);
        }
        _ = cancel_token.cancelled() => break,
    }
}
```

### Cancellation-safe tokio methods

| Method | Safe? | Notes |
|--------|-------|-------|
| `mpsc::Receiver::recv()` | Yes | No data lost |
| `oneshot::Receiver` (as future) | Yes | Value stays in channel |
| `broadcast::Receiver::recv()` | Yes | |
| `watch::Receiver::changed()` | Yes | |
| `TcpListener::accept()` | Yes | |
| `tokio::time::sleep()` | Yes | |
| `AsyncReadExt::read()` | **No** | Bytes may be partially read |
| `AsyncReadExt::read_exact()` | **No** | |
| `AsyncWriteExt::write_all()` | **No** | Partial writes possible |

### Making code cancellation-safe

1. **Don't do work before the cancel-sensitive await:**
   ```rust
   // Move side effects after the select
   let msg = select! {
       msg = rx.recv() => msg,
       _ = shutdown.recv() => return,
   };
   // Process after select (only reached if rx won)
   process(msg).await;
   ```

2. **Use `tokio::pin!` with a persistent future in a loop:**
   ```rust
   let operation = fetch_data();
   tokio::pin!(operation);

   loop {
       select! {
           result = &mut operation => { handle(result); break; }
           msg = rx.recv() => { /* handle msg without cancelling operation */ }
       }
   }
   ```

## Recursive Async Functions

Async functions can't be directly recursive because the state machine size would be infinite. Use `BoxFuture`:

```rust
use futures::future::BoxFuture;
use futures::FutureExt;

fn traverse(node: Node) -> BoxFuture<'static, Vec<String>> {
    async move {
        let mut results = vec![fetch_data(&node).await];
        for child in node.children {
            results.extend(traverse(child).await);
        }
        results
    }.boxed() // heap-allocate to break infinite size
}
```

For `Send` futures: use `BoxFuture<'a, T>` (alias for `Pin<Box<dyn Future<Output = T> + Send + 'a>>`).
For non-`Send`: use `LocalBoxFuture<'a, T>`.

## FuturesUnordered

Process many independent futures concurrently without the overhead of spawning tasks:

```rust
use futures::stream::FuturesUnordered;
use futures::StreamExt;

let mut futs = FuturesUnordered::new();
for url in urls {
    futs.push(fetch(url));
}

while let Some(result) = futs.next().await {
    match result {
        Ok(data) => process(data),
        Err(e) => eprintln!("failed: {e}"),
    }
}
```

- Results arrive in **completion order** (not insertion order)
- All futures run on the **current task** (no `Send + 'static` required)
- Better than `join_all` when you want to process results as they arrive
- Better than `tokio::spawn` for many small futures (no per-task scheduling overhead)

### FuturesUnordered vs buffer_unordered vs tokio::spawn

| Approach | Send required | Concurrency limit | Ordering |
|----------|--------------|-------------------|----------|
| `FuturesUnordered` | No | Unbounded (manual) | Completion order |
| `buffer_unordered(n)` | No | Yes (n) | Completion order |
| `buffered(n)` | No | Yes (n) | Input order |
| `tokio::spawn` per task | Yes + 'static | Unbounded (manual) | N/A |

## Graceful Shutdown Pattern

```rust
use tokio::signal;
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let worker = tokio::spawn(worker_loop(shutdown_rx.clone()));
    let server = tokio::spawn(serve(shutdown_rx));

    // Wait for Ctrl+C
    signal::ctrl_c().await.unwrap();
    println!("shutting down...");

    // Notify all tasks
    let _ = shutdown_tx.send(true);

    // Wait for tasks to finish
    let _ = tokio::join!(worker, server);
}

async fn worker_loop(mut shutdown: watch::Receiver<bool>) {
    loop {
        tokio::select! {
            _ = do_work() => {}
            _ = shutdown.changed() => {
                println!("worker shutting down");
                break;
            }
        }
    }
}
```

Alternative: use `tokio_util::sync::CancellationToken` for a purpose-built solution.

## Sources

- [Async Book: Cancellation](https://rust-lang.github.io/async-book/part-guide/more-async-await.html)
- [Async Book: Select](https://rust-lang.github.io/async-book/part-guide/concurrency-primitives.html)
- [Tokio: Select](https://tokio.rs/tokio/tutorial/select)
- [Tokio: Graceful Shutdown](https://tokio.rs/tokio/topics/shutdown)
- [trait-variant crate](https://docs.rs/trait-variant/latest/trait_variant/)
