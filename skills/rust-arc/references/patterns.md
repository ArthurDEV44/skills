# Arc Patterns & Real-World Usage

## Shared State in Async (Tokio)

`tokio::spawn` requires `'static + Send`, making `Arc` the standard way to share state across tasks.

```rust
use std::sync::Arc;
use tokio::sync::Mutex; // prefer tokio::sync::Mutex in async code
use std::collections::HashMap;

type SharedState = Arc<Mutex<HashMap<String, String>>>;

let state: SharedState = Arc::new(Mutex::new(HashMap::new()));

let state_clone = Arc::clone(&state);
tokio::spawn(async move {
    let mut map = state_clone.lock().await; // non-blocking .await
    map.insert("key".into(), "value".into());
});
```

**Use `tokio::sync::Mutex` over `std::sync::Mutex` in async code** — holding a `std::sync::Mutex` guard across `.await` points will block the entire runtime thread.

### Tokio RwLock for Read-Heavy Workloads

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

let config = Arc::new(RwLock::new(AppConfig::default()));

// Many readers concurrently
let cfg = Arc::clone(&config);
tokio::spawn(async move {
    let r = cfg.read().await;
    println!("{:?}", *r);
});

// Occasional writer
let cfg = Arc::clone(&config);
tokio::spawn(async move {
    let mut w = cfg.write().await;
    w.reload();
});
```

## Type Alias Pattern

Deeply nested `Arc<Mutex<...>>` types are common — use type aliases for readability.

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

type Db = Arc<Mutex<HashMap<String, Vec<u8>>>>;
type SharedConfig = Arc<tokio::sync::RwLock<Config>>;

fn process(db: &Db) {
    let mut data = db.lock().unwrap();
    data.insert("key".into(), vec![1, 2, 3]);
}
```

## Axum Shared State

```rust
use axum::{Router, extract::State, routing::get};
use std::sync::Arc;
use tokio::sync::RwLock;

struct AppState {
    db: DatabasePool,
    cache: RwLock<HashMap<String, String>>,
}

let shared_state = Arc::new(AppState {
    db: pool,
    cache: RwLock::new(HashMap::new()),
});

let app = Router::new()
    .route("/", get(handler))
    .with_state(shared_state);

async fn handler(State(state): State<Arc<AppState>>) -> String {
    let cache = state.cache.read().await;
    cache.get("key").cloned().unwrap_or_default()
}
```

## Thread Pool with Shared Work Queue

```rust
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    sender: mpsc::Sender<Job>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        let workers = (0..size)
            .map(|_| {
                let rx = Arc::clone(&receiver);
                thread::spawn(move || {
                    while let Ok(job) = rx.lock().unwrap().recv() {
                        job();
                    }
                })
            })
            .collect();

        ThreadPool { sender, workers }
    }

    fn execute<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.sender.send(Box::new(f)).unwrap();
    }
}
```

## Observer / Event Bus Pattern

```rust
use std::sync::{Arc, Mutex, Weak};

trait Observer: Send + Sync {
    fn on_event(&self, data: &str);
}

struct EventBus {
    listeners: Mutex<Vec<Weak<dyn Observer>>>,
}

impl EventBus {
    fn new() -> Self {
        EventBus { listeners: Mutex::new(vec![]) }
    }

    fn subscribe(&self, observer: &Arc<dyn Observer>) {
        self.listeners.lock().unwrap().push(Arc::downgrade(observer));
    }

    fn notify(&self, data: &str) {
        let mut listeners = self.listeners.lock().unwrap();
        // Clean up dead listeners and notify alive ones
        listeners.retain(|weak| {
            if let Some(strong) = weak.upgrade() {
                strong.on_event(data);
                true
            } else {
                false // listener was dropped, remove from list
            }
        });
    }
}
```

## Cache with Weak References

```rust
use std::sync::{Arc, Weak, Mutex};
use std::collections::HashMap;

struct Cache<T> {
    entries: Mutex<HashMap<String, Weak<T>>>,
}

impl<T> Cache<T> {
    fn get(&self, key: &str) -> Option<Arc<T>> {
        let entries = self.entries.lock().unwrap();
        entries.get(key)?.upgrade() // None if value was dropped
    }

    fn insert(&self, key: String, value: &Arc<T>) {
        let mut entries = self.entries.lock().unwrap();
        entries.insert(key, Arc::downgrade(value));
    }

    /// Remove entries whose values have been dropped
    fn gc(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries.retain(|_, weak| weak.strong_count() > 0);
    }
}
```

## Tree with Parent Back-References

```rust
use std::sync::{Arc, Weak, Mutex};

struct TreeNode {
    value: String,
    parent: Weak<TreeNode>,         // non-owning back-reference
    children: Mutex<Vec<Arc<TreeNode>>>, // owning forward references
}

impl TreeNode {
    fn new_root(value: String) -> Arc<Self> {
        Arc::new_cyclic(|me| TreeNode {
            value,
            parent: me.clone(),
            children: Mutex::new(vec![]),
        })
    }

    fn add_child(parent: &Arc<TreeNode>, value: String) -> Arc<TreeNode> {
        let child = Arc::new(TreeNode {
            value,
            parent: Arc::downgrade(parent),
            children: Mutex::new(vec![]),
        });
        parent.children.lock().unwrap().push(Arc::clone(&child));
        child
    }

    fn parent(&self) -> Option<Arc<TreeNode>> {
        self.parent.upgrade()
    }
}
```

## Clone-on-Write Configuration

Use `make_mut` for config objects that are read often but updated rarely.

```rust
use std::sync::Arc;

#[derive(Clone)]
struct Config {
    max_connections: usize,
    timeout_ms: u64,
}

struct Server {
    config: Arc<Config>,
}

impl Server {
    fn update_timeout(&mut self, timeout: u64) {
        // Only clones if other references exist (e.g., active handlers)
        Arc::make_mut(&mut self.config).timeout_ms = timeout;
    }

    fn config(&self) -> Arc<Config> {
        Arc::clone(&self.config) // cheap ref-count bump
    }
}
```

## FFI Bridge Pattern

Pass Arc across FFI boundaries using raw pointers.

```rust
use std::sync::Arc;

#[repr(C)]
pub struct Handle {
    _private: [u8; 0],
}

pub struct Inner {
    data: Vec<u8>,
}

/// Create a new handle (increments ref count)
#[no_mangle]
pub extern "C" fn handle_create(data: *const u8, len: usize) -> *const Handle {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let inner = Arc::new(Inner { data: slice.to_vec() });
    Arc::into_raw(inner) as *const Handle
}

/// Clone a handle (increments ref count)
#[no_mangle]
pub extern "C" fn handle_clone(ptr: *const Handle) -> *const Handle {
    unsafe { Arc::increment_strong_count(ptr as *const Inner); }
    ptr
}

/// Release a handle (decrements ref count, frees at 0)
#[no_mangle]
pub extern "C" fn handle_release(ptr: *const Handle) {
    unsafe { Arc::decrement_strong_count(ptr as *const Inner); }
}
```

## Custom Waker for Async (Arc<W> where W: Wake)

```rust
use std::sync::Arc;
use std::task::Wake;

struct MyWaker {
    thread: std::thread::Thread,
}

impl Wake for MyWaker {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

let waker = Arc::new(MyWaker {
    thread: std::thread::current(),
});
let std_waker: std::task::Waker = waker.into(); // Arc<W> -> Waker conversion
```

## Shared File Handle

`Arc<File>` implements `Read + Write + Seek`, enabling shared file access.

```rust
use std::sync::Arc;
use std::fs::File;
use std::io::Read;

let file = Arc::new(File::open("data.txt").unwrap());

let f1 = Arc::clone(&file);
let h1 = std::thread::spawn(move || {
    let mut buf = [0u8; 1024];
    let mut f = &*f1; // &File implements Read
    f.read(&mut buf).unwrap();
});
```

## Anti-Patterns to Avoid

### 1. Arc where Rc suffices
```rust
// BAD: atomic overhead for no reason in single-threaded code
let data = Arc::new(vec![1, 2, 3]);

// GOOD: use Rc in single-threaded contexts
use std::rc::Rc;
let data = Rc::new(vec![1, 2, 3]);
```

### 2. Arc<Mutex<T>> when channels would be cleaner
```rust
// Consider whether mpsc::channel or tokio::sync::mpsc is more appropriate
// than shared state for producer-consumer patterns
```

### 3. Cloning Arc inside a hot loop unnecessarily
```rust
// BAD: atomic increment/decrement on every iteration
for item in &items {
    let data = Arc::clone(&shared);
    process(item, &data);
}

// GOOD: borrow the Arc once
let data = &*shared; // or just use &shared with auto-deref
for item in &items {
    process(item, data);
}
```

### 4. Holding std::sync::Mutex guard across await
```rust
// BAD: blocks the async runtime thread
let guard = data.lock().unwrap();
some_async_fn().await; // other tasks on this thread are blocked!
drop(guard);

// GOOD: use tokio::sync::Mutex, or drop guard before await
{
    let mut guard = data.lock().unwrap();
    *guard += 1;
} // guard dropped here
some_async_fn().await;
```

### 5. Using strong_count for synchronization logic
```rust
// BAD: race condition
if Arc::strong_count(&data) == 1 {
    // Another thread might clone between check and use!
    let val = Arc::try_unwrap(data).unwrap(); // might panic
}

// GOOD: use into_inner or try_unwrap and handle the result
if let Some(val) = Arc::into_inner(data) {
    // guaranteed sole owner
}
```
