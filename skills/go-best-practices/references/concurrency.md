# Go Concurrency Patterns

## Goroutines

### Lifecycle Management

Always ensure goroutines can be stopped. Never fire-and-forget:

```go
// BAD: goroutine leak -- no way to stop it
go func() {
    for {
        doWork()
        time.Sleep(time.Second)
    }
}()

// GOOD: controllable via context cancellation
func startWorker(ctx context.Context) {
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                doWork()
            }
        }
    }()
}
```

### Goroutine per Connection / Request

```go
func serve(l net.Listener) {
    for {
        conn, err := l.Accept()
        if err != nil {
            log.Printf("accept: %v", err)
            continue
        }
        go handleConn(conn)  // one goroutine per connection
    }
}
```

## Channels

### Channel Direction in Signatures

```go
// Producer: send-only channel
func generate(ctx context.Context) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        for i := 0; ; i++ {
            select {
            case ch <- i:
            case <-ctx.Done():
                return
            }
        }
    }()
    return ch
}

// Consumer: receive-only channel
func consume(ch <-chan int) {
    for val := range ch {
        process(val)
    }
}
```

### Buffered vs Unbuffered

- **Unbuffered** (`make(chan T)`): synchronizes sender and receiver, guarantees delivery before sender proceeds
- **Buffered** (`make(chan T, n)`): decouples sender/receiver, sender blocks only when buffer full
- Default to unbuffered; use buffered only when you have a clear reason (batching, known producer/consumer rate mismatch)

### Patterns

**Fan-out / Fan-in:**

```go
func fanOut(ctx context.Context, input <-chan Job, workers int) <-chan Result {
    results := make(chan Result)
    var wg sync.WaitGroup
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range input {
                select {
                case results <- process(job):
                case <-ctx.Done():
                    return
                }
            }
        }()
    }
    go func() {
        wg.Wait()
        close(results)
    }()
    return results
}
```

**Done channel / signal:**

```go
done := make(chan struct{})
go func() {
    defer close(done)
    // work...
}()
<-done  // blocks until goroutine finishes
```

## Select

```go
select {
case msg := <-msgCh:
    handle(msg)
case err := <-errCh:
    log.Printf("error: %v", err)
case <-ctx.Done():
    return ctx.Err()
case <-time.After(5 * time.Second):
    return errors.New("timeout")
}
```

- `select` blocks until one case is ready
- If multiple cases ready, one is chosen at random (fair scheduling)
- `default` makes it non-blocking -- use for try-send / try-receive

```go
// Non-blocking send
select {
case ch <- val:
default:
    // channel full, drop or buffer elsewhere
}
```

## sync Package

### sync.WaitGroup

```go
var wg sync.WaitGroup
for _, url := range urls {
    wg.Add(1)
    go func() {
        defer wg.Done()
        fetch(url)
    }()
}
wg.Wait()
```

### sync.Mutex / sync.RWMutex

```go
type SafeMap struct {
    mu sync.RWMutex
    m  map[string]int
}

func (s *SafeMap) Get(key string) (int, bool) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    v, ok := s.m[key]
    return v, ok
}

func (s *SafeMap) Set(key string, val int) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.m[key] = val
}
```

### sync.Once

```go
var (
    instance *DB
    once     sync.Once
)

func GetDB() *DB {
    once.Do(func() {
        instance = connectDB()
    })
    return instance
}
```

### sync.Map

Use only when keys are stable (write-once, read-many) or disjoint goroutines access disjoint keys. For most cases, `sync.Mutex` + regular map is simpler and faster.

## errgroup (golang.org/x/sync/errgroup)

Run goroutines and collect the first error:

```go
g, ctx := errgroup.WithContext(ctx)

for _, url := range urls {
    g.Go(func() error {
        return fetch(ctx, url)
    })
}

if err := g.Wait(); err != nil {
    return fmt.Errorf("fetching: %w", err)
}
```

- `g.Wait()` blocks until all goroutines finish, returns first non-nil error
- Context is cancelled when any goroutine returns an error
- Use `g.SetLimit(n)` for bounded concurrency

```go
g, ctx := errgroup.WithContext(ctx)
g.SetLimit(10)  // max 10 concurrent goroutines

for _, job := range jobs {
    g.Go(func() error {
        return process(ctx, job)
    })
}
```

## Common Pitfalls

### Race Conditions

```go
// BAD: data race -- concurrent map write
go func() { m["a"] = 1 }()
go func() { m["b"] = 2 }()

// GOOD: protect with mutex or use sync.Map
```

Run `go test -race ./...` to detect races. Always run race detector in CI.

### Goroutine Leaks

Every goroutine must have a clear exit path. Common leak patterns:
- Blocked on channel with no sender/closer
- Blocked on channel with no context cancellation
- Infinite loop without exit condition

### Channel Closing Rules

- Only the **sender** closes the channel, never the receiver
- Closing an already-closed channel panics
- Sending to a closed channel panics
- Receiving from a closed channel returns zero value immediately
