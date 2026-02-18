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

#### Unix-specific signals

```rust
#[cfg(unix)]
{
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sighup = signal(SignalKind::hangup())?;

    tokio::select! {
        _ = sigterm.recv() => { println!("SIGTERM"); }
        _ = sighup.recv() => { println!("SIGHUP"); }
        _ = signal::ctrl_c() => { println!("SIGINT"); }
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

#### Child tokens

```rust
let parent = CancellationToken::new();
let child = parent.child_token();

// Cancelling parent cancels child
parent.cancel();
assert!(child.is_cancelled());

// Cancelling child does NOT cancel parent
let parent2 = CancellationToken::new();
let child2 = parent2.child_token();
child2.cancel();
assert!(!parent2.is_cancelled());
```

Use child tokens for hierarchical shutdown (e.g., cancel a subsystem without stopping the whole app).

#### drop_guard

```rust
let token = CancellationToken::new();
let guard = token.clone().drop_guard();
// token is cancelled when guard is dropped
drop(guard); // triggers cancellation
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

- `tracker.spawn(future)` -- spawn and track (like `tokio::spawn` but tracked)
- `tracker.spawn_on(future, &handle)` -- spawn on specific runtime handle
- `tracker.spawn_blocking(|| { ... })` -- track a blocking task
- `tracker.token()` -- get a `TaskTrackerToken` for manual tracking
- `tracker.len()` -- number of in-flight tracked tasks

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

### Shutdown with timeout

```rust
token.cancel();
tracker.close();

if tokio::time::timeout(Duration::from_secs(30), tracker.wait()).await.is_err() {
    eprintln!("shutdown timed out, {} tasks still running", tracker.len());
    // Force exit or abort remaining tasks
}
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

### Multi-threaded test

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent() {
    // runs on multi-thread runtime
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
    tokio::time::resume(); // optional: resume real time
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

### Testing with channels

```rust
#[tokio::test]
async fn test_actor() {
    let (tx, rx) = tokio::sync::mpsc::channel(10);
    let actor = tokio::spawn(my_actor(rx));

    tx.send(Command::DoSomething).await.unwrap();
    drop(tx); // close channel, actor loop ends

    let result = actor.await.unwrap();
    assert_eq!(result, expected);
}
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
    .with_target(false)
    .with_env_filter("my_app=debug,tower_http=info")
    .init();
```

### EnvFilter (dynamic level control)

```rust
use tracing_subscriber::EnvFilter;

tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env()) // reads RUST_LOG
    .init();

// RUST_LOG="my_crate=debug,tokio=warn" cargo run
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

// Add custom fields, set level
#[tracing::instrument(level = "debug", fields(request_id = %id))]
async fn process(id: u64) { /* ... */ }
```

### Events (log-like)

```rust
tracing::error!("connection failed: {}", err);
tracing::warn!(%cause, "parse error");    // % = Display format
tracing::info!(user_id = 42, "logged in");
tracing::debug!(?object, "state");        // ? = Debug format
tracing::trace!("low-level detail");
```

### Spans (structured context)

```rust
let span = tracing::info_span!("request", method = %req.method(), path = %req.uri());
let _guard = span.enter(); // span active while guard lives

// For async code, use .instrument()
use tracing::Instrument;
async_work()
    .instrument(tracing::info_span!("work"))
    .await;
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

### Layer composition

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

tracing_subscriber::registry()
    .with(
        tracing_subscriber::fmt::layer()
            .with_filter(EnvFilter::new("info"))
    )
    .with(
        console_subscriber::ConsoleLayer::builder()
            .with_default_env()
            .spawn()
    )
    .init();
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

**Important**: Never call `block_on` inside an async context (panics). Use `spawn_blocking` instead.

## Sources

- [Graceful shutdown](https://tokio.rs/tokio/topics/shutdown)
- [Testing](https://tokio.rs/tokio/topics/testing)
- [Tracing](https://tokio.rs/tokio/topics/tracing)
- [Tracing next steps](https://tokio.rs/tokio/topics/tracing-next-steps)
- [Bridging sync/async](https://tokio.rs/tokio/topics/bridging)
- [CancellationToken](https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html)
- [TaskTracker](https://docs.rs/tokio-util/latest/tokio_util/task/struct.TaskTracker.html)
