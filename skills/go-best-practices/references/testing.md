# Go Testing Patterns

## Table-Driven Tests

The canonical Go testing pattern:

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive", 2, 3, 5},
        {"zero", 0, 0, 0},
        {"negative", -1, -2, -3},
        {"mixed", -1, 5, 4},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            if got != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.expected)
            }
        })
    }
}
```

### Why Table-Driven

- Easy to add new cases (one line per case)
- Test names appear in output: `TestAdd/positive`, `TestAdd/negative`
- Each subtest can be run individually: `go test -run TestAdd/negative`
- Subtests run in parallel if `t.Parallel()` is called

## Subtests and Parallel

```go
func TestAPI(t *testing.T) {
    t.Run("GET", func(t *testing.T) {
        t.Parallel()
        // ...
    })
    t.Run("POST", func(t *testing.T) {
        t.Parallel()
        // ...
    })
}
```

- `t.Parallel()` marks a subtest to run concurrently with other parallel subtests
- The parent test does not complete until all parallel subtests finish
- Use for independent tests to speed up the suite

## Test Helpers

```go
func setupTestDB(t *testing.T) *sql.DB {
    t.Helper()  // marks this as a helper -- errors report caller's line
    db, err := sql.Open("sqlite3", ":memory:")
    if err != nil {
        t.Fatalf("open db: %v", err)
    }
    t.Cleanup(func() {
        db.Close()
    })
    return db
}
```

- `t.Helper()` -- call at the start of helper functions so failures report the caller's location
- `t.Cleanup(func())` -- registers cleanup that runs after the test (and subtests) complete, in LIFO order
- `t.TempDir()` -- creates a temp directory, auto-cleaned after test

## Benchmarks

```go
func BenchmarkConcat(b *testing.B) {
    for b.Loop() {     // Go 1.24+: replaces for i := 0; i < b.N; i++
        concat("hello", "world")
    }
}

// With setup excluded from timing
func BenchmarkProcess(b *testing.B) {
    data := generateLargeDataset()
    b.ResetTimer()
    for b.Loop() {
        process(data)
    }
}

// Sub-benchmarks
func BenchmarkSort(b *testing.B) {
    sizes := []int{100, 1000, 10000}
    for _, size := range sizes {
        b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
            data := generateSlice(size)
            b.ResetTimer()
            for b.Loop() {
                slices.Sort(slices.Clone(data))
            }
        })
    }
}
```

Run: `go test -bench=. -benchmem ./...`

## Fuzzing (Go 1.18+)

```go
func FuzzParseJSON(f *testing.F) {
    // Seed corpus
    f.Add([]byte(`{"name": "test"}`))
    f.Add([]byte(`{}`))
    f.Add([]byte(`[]`))

    f.Fuzz(func(t *testing.T, data []byte) {
        var result map[string]any
        err := json.Unmarshal(data, &result)
        if err != nil {
            return  // invalid input is fine, just don't panic
        }
        // Re-marshal and check round-trip
        out, err := json.Marshal(result)
        if err != nil {
            t.Fatalf("marshal after successful unmarshal: %v", err)
        }
        var result2 map[string]any
        if err := json.Unmarshal(out, &result2); err != nil {
            t.Fatalf("re-unmarshal: %v", err)
        }
    })
}
```

Run: `go test -fuzz=FuzzParseJSON -fuzztime=30s`

## HTTP Handler Testing

```go
func TestGetUser(t *testing.T) {
    // Setup
    handler := NewUserHandler(mockStore)

    req := httptest.NewRequest(http.MethodGet, "/users/123", nil)
    rec := httptest.NewRecorder()

    // Execute
    handler.ServeHTTP(rec, req)

    // Assert
    if rec.Code != http.StatusOK {
        t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
    }

    var user User
    if err := json.NewDecoder(rec.Body).Decode(&user); err != nil {
        t.Fatalf("decode response: %v", err)
    }
    if user.ID != "123" {
        t.Errorf("user.ID = %q, want %q", user.ID, "123")
    }
}
```

## Test Organization

```
mypackage/
    mypackage.go
    mypackage_test.go      // same package tests (white-box, access unexported)
    export_test.go         // exports unexported symbols for external tests
    mypackage_ext_test.go  // package mypackage_test (black-box, public API only)
```

- **Same-package tests** (`package foo`): test internal logic, access unexported functions
- **External tests** (`package foo_test`): test public API, catch export issues
- Prefer external tests for API-focused testing; use same-package for internals

## Testdata and Golden Files

```go
// testdata/ directory is ignored by go build, available to tests
func TestRender(t *testing.T) {
    input, err := os.ReadFile("testdata/input.html")
    if err != nil {
        t.Fatal(err)
    }

    got := render(string(input))

    golden := "testdata/expected.html"
    if *update {  // -update flag to regenerate
        os.WriteFile(golden, []byte(got), 0644)
    }

    expected, err := os.ReadFile(golden)
    if err != nil {
        t.Fatal(err)
    }
    if got != string(expected) {
        t.Errorf("output mismatch; run with -update to regenerate golden file")
    }
}
```

## Testing Conventions

- Test file: `foo_test.go` alongside `foo.go`
- Test function: `TestXxx(t *testing.T)` -- `Xxx` starts with uppercase
- Benchmark: `BenchmarkXxx(b *testing.B)`
- Fuzz: `FuzzXxx(f *testing.F)`
- Example: `ExampleXxx()` with `// Output:` comment
- Use `t.Errorf` for non-fatal (test continues), `t.Fatalf` for fatal (test stops)
- No assertion libraries needed -- `if got != want` is idiomatic Go
- Run all tests: `go test ./...`
- With race detector: `go test -race ./...`
- With coverage: `go test -cover ./...`
- Verbose: `go test -v ./...`
