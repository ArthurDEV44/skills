# Go Project Structure

## Module Basics

```bash
# Initialize a new module
go mod init github.com/user/project

# Add dependencies
go get github.com/lib/pq@latest

# Tidy (remove unused, add missing)
go mod tidy

# Vendor dependencies
go mod vendor
```

### go.mod

```
module github.com/user/project

go 1.23

require (
    github.com/lib/pq v1.10.9
    golang.org/x/sync v0.7.0
)
```

- Pin the Go version to the minimum your code requires
- Run `go mod tidy` before every commit
- Commit both `go.mod` and `go.sum`

## Standard Project Layout

### Small Project / Library

```
mylib/
    mylib.go          // package mylib
    mylib_test.go
    helper.go
    go.mod
    go.sum
```

### Application

```
myapp/
    cmd/
        myapp/
            main.go       // package main -- entry point
        migrate/
            main.go       // package main -- secondary binary
    internal/
        server/
            server.go     // package server
            handler.go
            handler_test.go
            middleware.go
        store/
            store.go      // package store
            postgres.go
            postgres_test.go
        config/
            config.go     // package config
    pkg/                  // (optional) importable by external projects
        api/
            types.go
    go.mod
    go.sum
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `cmd/` | Main applications. Each subdirectory = one binary. `main.go` should be minimal -- parse flags, wire dependencies, call `run()` |
| `internal/` | Private code. **Cannot** be imported by other modules. Use for core business logic |
| `pkg/` | Public library code importable by other projects. Only create if you intend external consumers |
| `testdata/` | Test fixtures. Ignored by `go build`, accessible to tests |

## Package Design Principles

### One Purpose Per Package

```go
// GOOD: each package has a clear responsibility
package user     // user domain logic
package postgres // PostgreSQL store implementation
package http     // HTTP transport layer

// BAD: grab-bag packages
package util     // what does it do?
package common   // everything and nothing
package helpers  // same problem
```

### Package by Domain, Not by Layer

```go
// GOOD: domain-oriented
internal/
    order/
        order.go       // Order type, business rules
        service.go     // OrderService
        repository.go  // OrderRepository interface
        postgres.go    // PostgreSQL implementation
    user/
        user.go
        service.go

// BAD: layer-oriented (leads to import cycles)
internal/
    models/       // everything depends on this
    handlers/     // imports models, services
    services/     // imports models, repositories
    repositories/ // imports models
```

### Import Cycle Prevention

Go forbids circular imports. Strategies:
- **Interface at consumer**: define interfaces where they're used, not where they're implemented
- **Shared types package**: if two packages need the same type, extract it to a third package
- **Dependency inversion**: high-level package defines interface, low-level package implements it

## The `internal` Directory

Code under `internal/` can only be imported by code rooted at the parent of `internal/`:

```
github.com/user/project/
    internal/
        secret/       // only importable by github.com/user/project/...
    cmd/
        app/
            main.go   // CAN import internal/secret

github.com/other/project/
    main.go           // CANNOT import github.com/user/project/internal/secret
```

Use `internal/` liberally. You can always make things public later; you can't take back a public API.

## cmd/ Pattern

Keep `main.go` thin -- it's the wiring layer:

```go
// cmd/server/main.go
package main

import (
    "context"
    "log"
    "os"
    "os/signal"

    "github.com/user/project/internal/config"
    "github.com/user/project/internal/server"
)

func main() {
    if err := run(context.Background(), os.Args[1:]); err != nil {
        log.Fatal(err)
    }
}

func run(ctx context.Context, args []string) error {
    ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
    defer cancel()

    cfg, err := config.Load()
    if err != nil {
        return fmt.Errorf("load config: %w", err)
    }

    srv := server.New(cfg)
    return srv.Run(ctx)
}
```

- `run()` takes context and returns error -- testable
- `main()` only calls `run()` and handles fatal exit
- Signal handling at the top level via `signal.NotifyContext`

## Workspaces (Go 1.18+)

For multi-module repos:

```
project/
    go.work
    service-a/
        go.mod
    service-b/
        go.mod
    shared/
        go.mod
```

```
// go.work
go 1.23

use (
    ./service-a
    ./service-b
    ./shared
)
```

- Workspaces allow developing multiple modules together without `replace` directives
- `go.work` is for local development -- do not commit it (add to `.gitignore`)
- Each module still has its own `go.mod` and works independently

## Build Tags

```go
//go:build linux
// +build linux

package mypackage
```

Common uses:
- Platform-specific code: `//go:build linux`, `//go:build windows`
- Integration tests: `//go:build integration`
- Feature flags: `//go:build enterprise`

Run with tags: `go test -tags=integration ./...`

## Dependency Management Tips

- Prefer stdlib over third-party when the stdlib solution is adequate
- Vet dependencies: check maintenance status, license, transitive deps
- Use `go mod why` to understand why a dependency is needed
- Use `go mod graph` to visualize the dependency tree
- Pin major versions; let `go mod tidy` handle minor/patch
- Update deps regularly: `go get -u ./...` then `go mod tidy` and run tests
