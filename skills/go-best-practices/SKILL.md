---
name: go-best-practices
description: >
  Go (Golang) best practices, idioms, and modern patterns for writing clean, performant, and
  maintainable Go code. Covers error handling (errors.Is, errors.As, error wrapping, sentinel
  errors, custom error types), naming conventions, package design, interface usage, struct design,
  concurrency (goroutines, channels, select, sync primitives, context propagation, errgroup),
  generics (type parameters, constraints, type sets), testing (table-driven, subtests, benchmarks,
  fuzzing, testscript), modules and workspaces, and project layout. Use when writing, reviewing,
  or refactoring Go code: (1) Error handling and wrapping strategies, (2) Naming and code style,
  (3) Interface design and accept-interfaces-return-structs, (4) Concurrency patterns with
  goroutines, channels, and sync, (5) Context propagation and cancellation, (6) Generics with
  type parameters and constraints, (7) Table-driven tests, subtests, benchmarks, and fuzzing,
  (8) Package layout and module organization, (9) Struct embedding and composition,
  (10) Defer, init, and control flow idioms, (11) Slice, map, and string best practices,
  (12) Performance patterns and common pitfalls.
---

# Go Best Practices

## Naming Conventions

### General Rules

- **MixedCaps** (exported) or **mixedCaps** (unexported) -- never underscores in names
- Short, concise names; the broader the scope, the more descriptive the name
- Single-letter vars are fine in small scopes: `i` for index, `r` for reader, `ctx` for context
- Acronyms keep consistent case: `URL`, `HTTP`, `ID` (not `Url`, `Http`, `Id`)

### Package Naming

```go
package http     // lowercase, single word, no underscores
package httputil // compound is ok when needed
```

- Package name is part of the call site: `http.Get` not `http.HTTPGet`
- Avoid stutter: `http.Server` not `http.HTTPServer`
- No `util`, `common`, `misc` -- name by what it provides
- Package path is lowercase only, no underscores or mixedCaps

### Interface Naming

- Single-method interfaces: method name + `-er` suffix: `Reader`, `Writer`, `Stringer`, `Closer`
- Multi-method: descriptive noun: `ReadWriteCloser`, `Handler`

### Getter/Setter

```go
// Getter -- just the field name, no "Get" prefix
func (u *User) Name() string { return u.name }

// Setter -- "Set" prefix
func (u *User) SetName(name string) { u.name = name }
```

## Error Handling

### Basic Pattern

Always check errors immediately. Never discard errors silently:

```go
val, err := doSomething()
if err != nil {
    return fmt.Errorf("doing something: %w", err)  // wrap with context
}
```

### Error Wrapping (Go 1.13+)

```go
// Wrap with %w to allow errors.Is / errors.As unwrapping
return fmt.Errorf("opening config %s: %w", path, err)

// Use %v when you intentionally want to hide the underlying error
return fmt.Errorf("operation failed: %v", err)
```

### Sentinel Errors

```go
var ErrNotFound = errors.New("not found")
var ErrPermission = errors.New("permission denied")

// Check with errors.Is (handles wrapped errors)
if errors.Is(err, ErrNotFound) {
    // handle not found
}
```

### Custom Error Types

```go
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation: %s: %s", e.Field, e.Message)
}

// Extract with errors.As
var ve *ValidationError
if errors.As(err, &ve) {
    log.Printf("invalid field: %s", ve.Field)
}
```

### Error Strings

- Start lowercase, no punctuation at end: `"opening file"` not `"Opening file."`
- Provide context that chains: `"open config: read file: permission denied"`
- Don't repeat caller info -- each layer adds its own context

For advanced error handling patterns (multi-error, error groups, panic recovery), see `references/error-handling.md`.

## Interfaces

### Accept Interfaces, Return Structs

```go
// GOOD: accept the narrowest interface you need
func Process(r io.Reader) error { /* ... */ }

// GOOD: return concrete type -- callers decide what interface they need
func NewServer(addr string) *Server { /* ... */ }
```

### Keep Interfaces Small

```go
// GOOD: minimal interface -- easy to implement, easy to mock
type Store interface {
    Get(ctx context.Context, id string) (*Item, error)
}

// BAD: kitchen-sink interface -- hard to implement, hard to mock
type Store interface {
    Get(ctx context.Context, id string) (*Item, error)
    List(ctx context.Context, filter Filter) ([]*Item, error)
    Create(ctx context.Context, item *Item) error
    Update(ctx context.Context, item *Item) error
    Delete(ctx context.Context, id string) error
    Count(ctx context.Context) (int, error)
    // ...20 more methods
}
```

### Define Interfaces at the Consumer, Not the Producer

```go
// package order (consumer) -- defines the interface it needs
type PaymentProcessor interface {
    Charge(ctx context.Context, amount int) error
}

func (s *Service) Checkout(ctx context.Context, pp PaymentProcessor) error {
    return pp.Charge(ctx, s.total)
}

// package stripe (producer) -- just exports a concrete struct
type Client struct { /* ... */ }
func (c *Client) Charge(ctx context.Context, amount int) error { /* ... */ }
// Client implicitly satisfies order.PaymentProcessor
```

### Don't Export Interfaces for Implementation

Only export an interface when callers need polymorphism. If you have one implementation, you probably don't need an interface.

## Structs & Composition

### Struct Embedding

```go
type Logger struct {
    mu     sync.Mutex   // embed for internal use (unexported)
    output io.Writer
}

type ReadCloser struct {
    io.Reader  // promoted methods: Read()
    io.Closer  // promoted methods: Close()
}
```

- Embedding promotes methods, not fields -- it's composition, not inheritance
- Only embed types whose methods you want to expose in the outer type's API
- Avoid embedding exported types in exported structs if it creates a confusing API

### Functional Options Pattern

```go
type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func NewServer(opts ...Option) *Server {
    s := &Server{port: 8080, timeout: 30 * time.Second}
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage
srv := NewServer(WithPort(9090), WithTimeout(5*time.Second))
```

## Slices, Maps & Strings

### Slice Idioms

```go
// Prefer nil slice declaration (works with append, len, range)
var items []string           // nil slice, len=0, cap=0

// Pre-allocate when size is known
items := make([]string, 0, expectedLen)

// Copy to avoid aliasing underlying array
dst := make([]int, len(src))
copy(dst, src)

// Delete element (order-preserving)
s = append(s[:i], s[i+1:]...)

// Delete element (fast, no order)
s[i] = s[len(s)-1]
s = s[:len(s)-1]

// slices package (Go 1.21+)
slices.Sort(s)
slices.Contains(s, val)
idx := slices.Index(s, val)
```

### Map Idioms

```go
// Check existence
val, ok := m[key]
if !ok {
    // key not present
}

// Delete
delete(m, key)

// maps package (Go 1.21+)
keys := maps.Keys(m)
maps.DeleteFunc(m, func(k string, v int) bool { return v < 0 })
```

### String Building

```go
// GOOD: strings.Builder for repeated concatenation
var b strings.Builder
for _, s := range parts {
    b.WriteString(s)
}
result := b.String()

// GOOD: strings.Join for slice joining
result := strings.Join(parts, ", ")

// BAD: repeated += in a loop (O(n^2) allocations)
```

## Defer, Init & Control Flow

### Defer

```go
f, err := os.Open(path)
if err != nil {
    return err
}
defer f.Close()  // runs when enclosing function returns
```

- Defer runs LIFO (last in, first out)
- Defer args are evaluated immediately, not at execution time
- Use `defer` for cleanup: closing files, unlocking mutexes, stopping timers
- Named returns + defer for error annotation:

```go
func readConfig(path string) (cfg Config, err error) {
    defer func() {
        if err != nil {
            err = fmt.Errorf("readConfig(%s): %w", path, err)
        }
    }()
    // ...
}
```

### Init Functions

- Use sparingly -- prefer explicit initialization
- Good for: registering drivers, computing package-level tables
- Bad for: anything with side effects that callers might not expect
- Multiple init() per file is allowed but avoid it

## Context

```go
// Always pass context as first parameter, named ctx
func (s *Service) GetUser(ctx context.Context, id string) (*User, error) {
    // ...
}
```

- Never store context in structs -- pass it as a function parameter
- Use `context.WithTimeout` / `context.WithCancel` for lifecycle control
- Use `context.WithValue` sparingly -- only for request-scoped data (trace IDs, auth)
- Always call the cancel function (usually via defer)

```go
ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
defer cancel()
```

For concurrency patterns (goroutines, channels, sync, errgroup, worker pools), see `references/concurrency.md`.

For generics (type parameters, constraints, type sets, best practices), see `references/generics.md`.

For testing patterns (table-driven, subtests, benchmarks, fuzzing, mocks), see `references/testing.md`.

For project structure (module layout, package organization, cmd/internal), see `references/project-structure.md`.
