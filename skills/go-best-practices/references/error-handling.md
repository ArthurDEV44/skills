# Advanced Error Handling

## Error Wrapping Chain

Build a descriptive chain where each layer adds context:

```go
// Low level
func readFile(path string) ([]byte, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("read file %s: %w", path, err)
    }
    return data, nil
}

// Mid level
func loadConfig(path string) (*Config, error) {
    data, err := readFile(path)
    if err != nil {
        return nil, fmt.Errorf("load config: %w", err)
    }
    // ...
}

// Result: "load config: read file config.yaml: open config.yaml: no such file or directory"
```

## Multi-Error Handling (Go 1.20+)

`errors.Join` combines multiple errors into one:

```go
var errs []error
for _, item := range items {
    if err := validate(item); err != nil {
        errs = append(errs, err)
    }
}
if err := errors.Join(errs...); err != nil {
    return err
}
```

`errors.Is` and `errors.As` work with joined errors -- they check each error in the tree.

## Custom Error Types with Wrapping

```go
type OpError struct {
    Op   string
    Path string
    Err  error
}

func (e *OpError) Error() string {
    return fmt.Sprintf("%s %s: %s", e.Op, e.Path, e.Err)
}

// Implement Unwrap to participate in errors.Is/As chain
func (e *OpError) Unwrap() error {
    return e.Err
}
```

## Panic and Recovery

- **panic**: only for truly unrecoverable programmer errors (index out of range, nil pointer)
- **recover**: only in deferred functions; use at package boundaries (HTTP handlers, goroutine launchers)

```go
func safeHandler(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if r := recover(); r != nil {
                log.Printf("panic recovered: %v\n%s", r, debug.Stack())
                http.Error(w, "internal error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}
```

**Rules:**
- Never use panic for normal error flow
- Never let panics escape package boundaries
- A library should never panic on bad input -- return errors instead
- Use `must` prefixed helpers only for program initialization:

```go
func mustParseURL(raw string) *url.URL {
    u, err := url.Parse(raw)
    if err != nil {
        panic(fmt.Sprintf("parse URL %q: %v", raw, err))
    }
    return u
}

var baseURL = mustParseURL("https://api.example.com")  // package-level init only
```

## Error Handling Anti-Patterns

```go
// BAD: swallowing errors
result, _ := doSomething()

// BAD: logging and returning (double handling)
if err != nil {
    log.Printf("error: %v", err)
    return err  // caller will also log it
}

// GOOD: log OR return, not both
// Return errors up to where they can be handled meaningfully
if err != nil {
    return fmt.Errorf("doing thing: %w", err)
}

// BAD: wrapping with redundant "failed to" / "error"
return fmt.Errorf("failed to open file: %w", err)
// The error already says what failed; chain reads: "failed to open file: open foo.txt: no such file"

// GOOD: just describe the operation
return fmt.Errorf("open file: %w", err)
```

## errors.Is vs errors.As vs Type Assertion

| Approach | Use When |
|----------|----------|
| `errors.Is(err, target)` | Checking for sentinel errors (value comparison, traverses wrap chain) |
| `errors.As(err, &target)` | Extracting typed error details (traverses wrap chain) |
| `err == target` | Exact equality only, ignores wrapping -- almost never correct |
| `err.(*Type)` | Type assertion, ignores wrapping -- almost never correct |
