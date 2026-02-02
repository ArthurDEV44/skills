# Graceful Shutdown, Testing, and Tracing

## Graceful Shutdown

### Signal detection

```rust
use tokio::signal;

tokio::select! {
    _ = server_loop() => {}
    _ = signal::ctrl_c() => {
        println!("shutdown signal received");
    }
}
```

### CancellationToken (tokio-util)

```toml
tokio-util = "0.7"
```

```rust
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();

// Spawn worker with cloned token
let worker_token = token.clone();
tokio::spawn(async move {
    tokio::select! {
        _ = worker_token.cancelled() => {
            // cleanup on shutdown
        }
        _ = do_work() => {}
    }
});

// Trigger shutdown
token.cancel(); // all cloned tokens are cancelled
```

### TaskTracker (wait for all tasks)

```rust
use tokio_util::task::TaskTracker;

let tracker = TaskTracker::new();

for i in 0..10 {
    tracker.spawn(handle_connection(i));
}

// Signal no more tasks will be added
tracker.close();
// Wait for all tracked tasks to complete
tracker.wait().await;
```

### Full shutdown pattern

```rust
let token = CancellationToken::new();
let tracker = TaskTracker::new();

loop {
    tokio::select! {
        conn = listener.accept() => {
            let (socket, _) = conn.unwrap();
            let token = token.clone();
            tracker.spawn(async move {
                tokio::select! {
                    _ = process(socket) => {}
                    _ = token.cancelled() => {}
                }
            });
        }
        _ = signal::ctrl_c() => {
            token.cancel();
            break;
        }
    }
}
tracker.close();
tracker.wait().await;
println!("all tasks finished, exiting");
```

---

## Testing

### Basic async test

```rust
#[tokio::test]
async fn test_something() {
    let result = async_operation().await;
    assert_eq!(result, expected);
}
```

### Pausing time

Avoid slow tests with real `sleep`/`timeout`:

```rust
#[tokio::test(start_paused = true)]  // requires feature "test-util"
async fn fast_timeout_test() {
    // sleep completes instantly but temporal order is preserved
    tokio::time::sleep(Duration::from_secs(3600)).await;
    // interval ticks happen in correct order
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    interval.tick().await;
}
```

Manual pause/resume:

```rust
#[tokio::test]
async fn manual_pause() {
    tokio::time::pause();
    // time-dependent code runs instantly
    tokio::time::sleep(Duration::from_secs(60)).await;
}
```

### Mocking I/O with tokio_test

```rust
#[tokio::test]
async fn test_handler() {
    let reader = tokio_test::io::Builder::new()
        .read(b"Hello\r\n")
        .read(b"World\r\n")
        .build();
    let writer = tokio_test::io::Builder::new()
        .write(b"Response 1\r\n")
        .write(b"Response 2\r\n")
        .build();

    handle_connection(reader, writer).await.unwrap();
}
```

Make handlers generic over `AsyncRead + AsyncWrite` for testability:

```rust
async fn handle<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    reader: R, writer: W,
) -> io::Result<()> { /* ... */ }
```

---

## Tracing

### Setup

```toml
tracing = "0.1"
tracing-subscriber = "0.3"
```

```rust
// Simple setup
tracing_subscriber::fmt::init();

// Custom configuration
tracing_subscriber::fmt()
    .compact()
    .with_file(true)
    .with_line_number(true)
    .with_thread_ids(true)
    .init();
```

### Instrumenting functions

```rust
#[tracing::instrument]
async fn handle_request(id: u64, name: &str) {
    // Auto-creates span with function name + args as fields
    tracing::info!("processing");
}

// Skip fields, custom name
#[tracing::instrument(name = "handler", skip(self, socket))]
async fn run(&mut self, socket: TcpStream) { /* ... */ }
```

### Events (log-like)

```rust
tracing::error!("connection failed: {}", err);
tracing::warn!(%cause, "parse error");    // % = Display format
tracing::info!(user_id = 42, "logged in");
tracing::debug!(?object, "state");        // ? = Debug format
tracing::trace!("low-level detail");
```

### Tokio Console (real-time monitoring)

```toml
tokio = { version = "1", features = ["full", "tracing"] }
console-subscriber = "0.1"
```

```rust
// Replace fmt subscriber with:
console_subscriber::init();
```

```bash
RUSTFLAGS="--cfg tokio_unstable" cargo run
tokio-console  # in another terminal
```

Shows live task states, resource usage (mutexes, semaphores), helps find deadlocks.

### OpenTelemetry / Jaeger

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

let tracer = opentelemetry_jaeger::new_pipeline()
    .with_service_name("my-service")
    .install_simple()?;
let otel = tracing_opentelemetry::layer().with_tracer(tracer);

tracing_subscriber::registry()
    .with(otel)
    .with(tracing_subscriber::fmt::Layer::default())
    .try_init()?;
```

---

## Bridging Sync and Async

### From sync code: use block_on

```rust
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()?;
let result = rt.block_on(async { do_async_work().await });
```

### Spawn background tasks on a runtime

```rust
let rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)
    .enable_all()
    .build()?;

let handle = rt.spawn(async { background_work().await });
// ... do sync work ...
rt.block_on(handle)?;
```

### Actor pattern (runtime in separate thread)

```rust
use tokio::sync::mpsc;

pub struct TaskSpawner {
    send: mpsc::Sender<Task>,
}

impl TaskSpawner {
    pub fn new() -> Self {
        let (send, mut recv) = mpsc::channel(16);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all().build().unwrap();

        std::thread::spawn(move || {
            rt.block_on(async {
                while let Some(task) = recv.recv().await {
                    tokio::spawn(handle(task));
                }
            });
        });

        TaskSpawner { send }
    }

    pub fn spawn(&self, task: Task) {
        self.send.blocking_send(task).unwrap();
    }
}
```

### From async: run blocking code

```rust
let result = tokio::task::spawn_blocking(|| {
    // CPU-heavy or blocking I/O work here
    compute_hash(data)
}).await?;
```

## Sources

- [Graceful shutdown](https://tokio.rs/tokio/topics/shutdown)
- [Testing](https://tokio.rs/tokio/topics/testing)
- [Tracing](https://tokio.rs/tokio/topics/tracing)
- [Tracing next steps](https://tokio.rs/tokio/topics/tracing-next-steps)
- [Bridging sync/async](https://tokio.rs/tokio/topics/bridging)
