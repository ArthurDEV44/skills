# Error Patterns — Taxonomy, Heuristics, and Signal Extraction

## Table of Contents

- [Error Classification Taxonomy](#error-classification-taxonomy)
- [Diagnosis Heuristics by Classification](#diagnosis-heuristics-by-classification)
- [Language-Adaptive Signal Extraction](#language-adaptive-signal-extraction)
- [Compound Errors](#compound-errors)

---

## Error Classification Taxonomy

### Compile Errors

**Subtypes:**
- Syntax error — missing delimiter, unexpected token, invalid expression
- Import/module error — unresolved import, circular dependency, missing module
- Type mismatch — expected X, found Y; incompatible types in assignment/return
- Lifetime/borrow error (Rust) — dangling reference, moved value, conflicting borrows
- Generic constraint failure — trait not implemented, where clause not satisfied

**Key signals:**
- Compiler name in output (`rustc`, `tsc`, `gcc`, `javac`, `go build`)
- Error codes with standard prefixes (`E0308`, `TS2322`, `C2065`)
- File:line:column format in error location
- "expected ... found ..." pattern

**Common root causes:**
- API changed between versions (function signature mismatch)
- Missing derive/impl for a trait the code requires
- Incorrect import path after refactor
- Type inference failure from ambiguous context

### Type Errors

**Subtypes:**
- Static type checker errors (TypeScript `tsc`, Python `mypy`/`pyright`, Flow)
- Generic/parametric type failures
- Nullability violations (strict null checks, Option unwrap)
- Interface/protocol conformance failures

**Key signals:**
- Type checker name in output
- "Type 'X' is not assignable to type 'Y'" patterns
- `undefined is not assignable`, `null`, `None`

**Common root causes:**
- Incorrect type annotation (the annotation is wrong, not the code)
- Missing null check before access
- Generic parameter not propagated through call chain
- Third-party type definitions out of date with runtime

### Runtime Errors

**Subtypes:**
- Panic/crash — unrecoverable error, process termination
- Exception — caught or uncaught, with stack trace
- Segfault/access violation — memory corruption, null pointer
- Assertion failure — `assert!`, `assert.equal`, `expect().toBe()`

**Key signals:**
- Stack trace with function names and line numbers
- "panic", "Traceback", "Error:", "Exception", "SIGSEGV"
- Process exit codes (non-zero)
- Thread/task identifiers in concurrent code

**Common root causes:**
- Unwrap on None/Err without checking
- Index out of bounds
- Division by zero
- Null reference in object chain
- Race condition in concurrent access

### Dependency Issues

**Subtypes:**
- Resolution failure — package not found, version conflict
- Lock file conflict — lock file doesn't match manifest
- Peer dependency mismatch — incompatible version ranges
- Binary incompatibility — ABI mismatch, native extension failure

**Key signals:**
- Package manager name in output (`cargo`, `npm`, `pip`, `go`)
- "could not resolve", "version conflict", "peer dependency"
- Lock file names (`Cargo.lock`, `package-lock.json`, `poetry.lock`)

**Common root causes:**
- Manifest specifies version range that excludes available versions
- Two dependencies require conflicting versions of a shared dependency
- Package was unpublished or yanked
- Lock file committed from different platform/architecture

### Config Issues

**Subtypes:**
- Missing environment variable
- Config file parse error (YAML, TOML, JSON syntax)
- Invalid configuration value (wrong type, out of range)
- Path resolution failure (file not found, wrong working directory)

**Key signals:**
- "ENOENT", "FileNotFoundError", "No such file or directory"
- Config file names in error (`config.toml`, `.env`, `tsconfig.json`)
- Environment variable names (`DATABASE_URL`, `API_KEY`)
- "failed to parse", "invalid value"

**Common root causes:**
- Environment variable set in local dev but not in CI/production
- Config file uses tabs where spaces are required (YAML)
- Relative path assumes a specific working directory
- Config schema changed between versions

### Logic Bugs

**Subtypes:**
- Wrong output — function returns incorrect result
- Off-by-one — boundary condition handled incorrectly
- State mutation bug — unexpected modification of shared state
- Ordering bug — operations execute in wrong sequence

**Key signals:**
- "expected X, got Y" in test output (values, not types)
- Assertion failure with actual vs expected diff
- Test passes individually but fails in suite (state leakage)
- Intermittent failures (race condition or timing dependency)

**Common root causes:**
- Incorrect comparison operator (`<` vs `<=`)
- Mutating input argument instead of creating copy
- Async operations not properly awaited
- Cache returning stale data

### Performance Issues

**Subtypes:**
- Timeout — operation exceeds time limit
- Out of memory — heap exhaustion, stack overflow
- Slow query — database query takes too long
- CPU saturation — computation blocks event loop

**Key signals:**
- "timeout", "OOM", "stack overflow", "killed"
- Timing information in output (seconds, milliseconds)
- Memory sizes in error messages
- "SIGKILL" (OOM killer)

**Common root causes:**
- O(n²) algorithm on large input (nested loops over same collection)
- Missing database index on filtered/joined column
- Unbounded recursion without base case
- Loading entire dataset into memory instead of streaming

### Test Failures

**Subtypes:**
- Assertion failure — expected value doesn't match actual
- Setup/teardown failure — test environment not properly configured
- Snapshot mismatch — serialized output changed
- Timeout — test exceeds time limit (often async-related)

**Key signals:**
- Test framework name (`cargo test`, `jest`, `pytest`, `go test`)
- Test function name in output
- "FAILED", "FAIL", "✗", "×"
- Diff output (- expected, + received)

**Common root causes:**
- Code changed but tests not updated
- Test relies on external state (database, filesystem, network)
- Async test doesn't await completion
- Snapshot needs regeneration after intentional change

---

## Diagnosis Heuristics by Classification

### Rapid Diagnosis Decision Tree

```
Error input received
     │
     ├── Has compiler/type-checker name? → Compile/Type error
     │     └── Has error code? → Look up code meaning
     │     └── Has "expected/found"? → Type mismatch at stated location
     │
     ├── Has stack trace? → Runtime error
     │     └── Top frame is user code? → Bug is at top frame location
     │     └── Top frame is library? → Bug is in how user calls library
     │     └── Only library frames? → Possible library bug or env issue
     │
     ├── Has package manager output? → Dependency issue
     │     └── "not found"? → Package doesn't exist or is misspelled
     │     └── "conflict"? → Version resolution failure
     │
     ├── Has config file reference? → Config issue
     │     └── "parse error"? → Syntax error in config file
     │     └── "not found"? → Missing file or env var
     │
     ├── Has test framework output? → Test failure
     │     └── Has expected/actual diff? → Logic bug in tested code
     │     └── Has "timeout"? → Async issue or infinite loop
     │
     └── Has "timeout"/"OOM"/"slow"? → Performance issue
           └── Has query? → Database performance
           └── Has recursion in trace? → Stack overflow
```

### Signal Priority

When multiple signals are present, prioritize:
1. **Error codes** — most specific, often link directly to documentation
2. **File:line locations** — tells you exactly where to look
3. **Stack trace top frame** — the immediate failure point
4. **Library names** — determines which documentation to check
5. **Error message text** — for web search if all else fails

---

## Language-Adaptive Signal Extraction

### Rust

```
error[E0308]: mismatched types
 --> src/main.rs:42:5
  |
42 |     let x: u32 = "hello";
  |                  ^^^^^^^ expected `u32`, found `&str`
```

Extract: error code `E0308`, file `src/main.rs`, line `42`, type mismatch `u32` vs `&str`.

Rust-specific signals: `E` + 4 digits error codes, `-->` location format, borrow checker annotations (`^` underlines), lifetime annotations in error.

### TypeScript

```
error TS2322: Type 'string' is not assignable to type 'number'.
  src/utils.ts(15,3): error TS2322
```

Extract: error code `TS2322`, file `src/utils.ts`, line `15`, col `3`, type mismatch `string` vs `number`.

TypeScript-specific: `TS` + digits error codes, parenthesized `(line,col)` format.

### Python

```
Traceback (most recent call last):
  File "app.py", line 23, in process_data
    result = data["key"]
KeyError: 'key'
```

Extract: file `app.py`, line `23`, function `process_data`, error type `KeyError`, key `'key'`.

Python-specific: `Traceback` header, `File "...", line N, in func` format, exception class name at bottom.

### Go

```
panic: runtime error: index out of range [5] with length 3

goroutine 1 [running]:
main.processItems(...)
        /app/main.go:47 +0x123
```

Extract: panic type `index out of range`, index `5`, length `3`, file `/app/main.go`, line `47`, function `main.processItems`.

Go-specific: `goroutine N [state]` format, tab-indented file paths, `+0x` offset notation.

### JavaScript/Node.js

```
TypeError: Cannot read properties of undefined (reading 'map')
    at processData (/app/src/utils.js:15:22)
    at Object.<anonymous> (/app/src/index.js:8:1)
```

Extract: error type `TypeError`, property `map`, undefined object, file `/app/src/utils.js`, line `15`, col `22`, function `processData`.

JS-specific: `at function (file:line:col)` stack format, error class name prefix.

---

## Compound Errors

Some errors are compound — the first error causes a cascade. Diagnosis rules:

1. **Read errors bottom-to-top in stack traces** — the root cause is usually the deepest user-code frame.
2. **In compiler output with multiple errors** — fix the FIRST error only, then re-run. Later errors are often caused by the first.
3. **In test suites with multiple failures** — look for a shared failing setup/fixture. If tests fail independently, treat each as separate.
4. **In dependency resolution** — the conflict that appears FIRST in the output is usually the one to resolve. Later conflicts may resolve automatically.
