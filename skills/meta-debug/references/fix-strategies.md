# Fix Strategies — Ranking Criteria, Anti-Patterns, and Verification

## Table of Contents

- [Fix Strategy Ranking](#fix-strategy-ranking)
- [Fix Patterns by Error Classification](#fix-patterns-by-error-classification)
- [Fix Anti-Patterns](#fix-anti-patterns)
- [Verification Protocols](#verification-protocols)
- [When NOT to Fix](#when-not-to-fix)

---

## Fix Strategy Ranking

When multiple fix strategies exist, rank them using these three criteria in priority order:

### 1. Correctness (Highest Priority)

Does the fix address the root cause, not just suppress the symptom?

```
// BAD: suppresses the symptom
try {
  result = riskyOperation();
} catch (e) {
  result = null; // swallow error — masks the real problem
}

// GOOD: fixes the root cause
if (input !== undefined) {
  result = riskyOperation(input);
} else {
  return Err(MissingInputError);
}
```

Questions to verify correctness:
- Does this fix the error for ALL inputs, or just the one that triggered it?
- Would removing the fix cause the exact same error to return?
- Does this match how the API/library is supposed to be used (per docs)?

### 2. Safety (Second Priority)

Does the fix minimize side effects and avoid regressions?

**Blast radius assessment:**
- How many files does the fix touch?
- Does it change any public API signatures?
- Does it affect code paths other than the failing one?
- Could it break existing tests?

```
// BAD: high blast radius — changes shared utility
fn parse_config(path: &str) -> Config {
    // Changed return type, affects all 15 callers
}

// GOOD: low blast radius — fixes only the failing call site
fn handler() -> Result<(), AppError> {
    let config = parse_config("app.toml")
        .map_err(|e| AppError::Config(e))?;  // Add error handling at call site only
}
```

Safety rules:
- Prefer adding code over modifying existing code
- Prefer fixing the call site over fixing the callee (unless the callee is wrong)
- Never change function signatures unless the signature itself is the bug
- If the fix requires touching 5+ files, reconsider — there may be a simpler approach

### 3. Simplicity (Third Priority)

Is this the least invasive change that achieves correctness and safety?

```
// BAD: over-engineered fix
class RetryableErrorHandler {
  constructor(private maxRetries: number, private backoff: BackoffStrategy) {}
  async handle<T>(fn: () => Promise<T>): Promise<T> { /* ... */ }
}

// GOOD: minimal fix for a one-off timeout
const result = await fetchWithTimeout(url, { timeout: 5000 });
```

Simplicity rules:
- One-line fixes are better than multi-line fixes
- Fixes in existing files are better than new files
- Fixes that use existing project patterns are better than introducing new ones
- Don't refactor while fixing — fix first, refactor later (if requested)

---

## Fix Patterns by Error Classification

### Compile Error Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Missing import | Add the import — check if the module is in dependencies |
| Type mismatch in assignment | Fix the assignment OR fix the type annotation — determine which is wrong from context |
| Trait not implemented | Add derive macro or manual impl — check which traits the consuming code needs |
| Lifetime error | Usually fix the borrow structure — avoid `clone()` unless the data truly needs shared ownership |
| Unused variable warning | Remove the variable if dead code, prefix with `_` only if intentionally unused for now |

### Type Error Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Null/undefined access | Add null check at the appropriate level — not at every level |
| Generic type mismatch | Add explicit type parameter at call site, or fix the generic constraint |
| Interface missing field | Add the missing field to the implementation, not make it optional in the interface |
| Type assertion needed | Prefer narrowing (type guards, `is`) over assertions (`as`, `!`) |

### Runtime Error Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Unwrap on None | Use `?` operator (Rust), optional chaining (JS/TS), or explicit match |
| Index out of bounds | Add bounds check, or use `.get()` / `.get(index)` that returns Option/undefined |
| Null reference chain | Find which part of the chain is null and handle it — don't wrap entire chain in try/catch |
| Race condition | Add synchronization at the shared resource, not at every access point |

### Dependency Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Version conflict | Pin both dependents to compatible versions — check what ranges overlap |
| Missing package | Add to manifest — verify the package name is correct (check registry) |
| Peer dependency | Install the required peer at the version specified in the warning |
| Lock file stale | Regenerate lock file from manifest — don't edit lock file manually |

### Config Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Missing env var | Add to `.env.example` and document — set it in the current environment |
| Parse error | Fix the syntax — use a linter/validator for the config format |
| Wrong path | Use absolute path or resolve relative to project root, not CWD |
| Invalid value | Check documentation for valid range/type and correct the value |

### Logic Bug Fixes

| Error Pattern | Fix Strategy |
|--------------|-------------|
| Off-by-one | Fix the boundary condition — add a test for the boundary |
| Wrong comparison | Fix the operator — check if the spec says inclusive or exclusive |
| State mutation | Clone before mutation, or restructure to avoid shared mutation |
| Ordering bug | Make the dependency explicit — use await, then, or sequential composition |

---

## Fix Anti-Patterns

### The Shotgun Fix
Making changes in many places hoping one of them fixes it.

**Why it's bad:** Introduces unintended changes that may cause regressions later. Makes it impossible to understand what actually fixed the issue.

**Instead:** Identify the single root cause location and fix only there.

### The Suppression Fix
Catching/ignoring the error instead of fixing the cause.

```
// ANTI-PATTERN
try { dangerousOperation(); } catch { /* ignore */ }

// ANTI-PATTERN
result = value ?? defaultValue; // when value should never be null

// ANTI-PATTERN
@ts-ignore  // suppress type error without understanding it
```

**Why it's bad:** The underlying bug remains. It will resurface in a different form, usually harder to diagnose.

**Instead:** Understand why the error occurs and fix the condition that causes it.

### The Dependency Upgrade Fix
Upgrading a library hoping the bug is fixed in a newer version.

**Why it's bad:** May introduce new breaking changes. The bug might not be in the library. Version upgrades should be deliberate, not speculative.

**Instead:** Verify the bug is in the library (check GitHub issues), then upgrade to the specific version that contains the fix.

### The Copy-Paste Fix
Copying a solution from Stack Overflow or GitHub issues without understanding it.

**Why it's bad:** The solution may be for a different version, different context, or may have subtle bugs itself.

**Instead:** Understand WHY the solution works before applying it. Verify it matches your version and context.

### The Over-Engineering Fix
Adding abstraction layers, retry logic, or error handling infrastructure for a one-time issue.

**Why it's bad:** Adds permanent complexity for a temporary problem. The abstraction may never be used again.

**Instead:** Apply the minimal fix. If the issue recurs, then consider adding infrastructure.

### The Config Workaround
Disabling a check or safety feature to make the error go away.

```
// ANTI-PATTERN
// @ts-nocheck
// eslint-disable
// #[allow(unused)]  when the variable IS used but in a way the linter can't see
// --no-verify
```

**Why it's bad:** The check exists for a reason. Disabling it masks the real issue and reduces safety for all future code.

**Instead:** Fix the code to satisfy the check. If the check is genuinely wrong, disable it narrowly with a comment explaining why.

---

## Verification Protocols

### After Compile Error Fix

```bash
# Rust
cargo check          # fast type/borrow check
cargo build          # full build if check passes
cargo clippy         # lint for additional issues

# TypeScript
bunx tsc --noEmit     # type check only
npm run build        # full build

# Go
go build ./...       # build all packages
go vet ./...         # lint

# Python
python -m py_compile file.py  # syntax check
mypy file.py                  # type check (if configured)
```

### After Runtime Error Fix

```bash
# Run the specific failing test/command that produced the error
# Do NOT run the full test suite — focus on the failure

# Rust
cargo test test_name -- --exact

# TypeScript/JavaScript
bunx jest --testNamePattern="test name"
bunx vitest run path/to/test.ts

# Python
pytest path/to/test.py::test_name -v

# Go
go test -run TestName ./package/...
```

### After Dependency Fix

```bash
# Verify lock file is consistent with manifest
cargo check              # Rust — regenerates Cargo.lock if needed
npm install              # Node — regenerates package-lock.json
pip install -r requirements.txt  # Python
go mod tidy              # Go — cleans up go.sum
```

### After Config Fix

Verify the application starts successfully:
```bash
# Dry-run or health check — don't run the full application
cargo run -- --help         # Rust CLI
npm run dev                 # Node dev server (check for startup errors)
python -c "from app import create_app; create_app()"  # Python
```

### After Logic Bug Fix

1. Run the failing test — it should now pass.
2. Check related tests — they should still pass.
3. If no test existed, suggest adding one for the boundary condition that was fixed.

---

## Fix Minimality Constraints

Research shows multi-agent debuggers make 40-60% more changes than necessary (Stanford DebugEVAL, 2025). Apply these constraints:

### Per-Line Justification

Every modified line must trace to the root cause:

```
// GOOD: each change is justified
- let x: u32 = "hello";  // root cause: type mismatch
+ let x: u32 = 42;       // fix: correct value type

// BAD: drive-by improvements
- let x: u32 = "hello";
+ let x: u32 = 42;       // fix
+ // Added proper type annotation  // unnecessary comment
```

### Change Scope Limits

For the canonical scope thresholds shared across skills, see `@~/.claude/skills/_shared/scope-guard.md`.

| Change scope | Action |
|---|---|
| 1-3 lines changed | Ideal — proceed |
| 4-10 lines changed | Acceptable — verify each line is justified |
| 11-20 lines changed | Warning — review for drive-by fixes, consider if a simpler approach exists |
| 20+ lines changed | STOP — this is likely over-fixing. Present the diagnosis and let the user decide scope |

### What NOT to Change

- Unrelated code near the fix location
- Import ordering or formatting
- Variable names that aren't part of the bug
- Comments on surrounding code
- Error handling for unrelated paths
- Type annotations on unrelated declarations

---

## DDI-Aware Fix Iteration

The Debugging Decay Index (DDI) describes exponential capability loss during repeated fix attempts (Scientific Reports, Dec 2025). The model is: E(t) = E₀ × e^(-λt), where λ ≈ 0.5-0.8 per attempt.

### Circuit Breaker Protocol

```
Attempt 1: Apply the top-ranked fix strategy
  → Verify (re-run reproduction command)
  → If PASS: done
  → If FAIL: record what was tried and WHY it failed

Attempt 2: Apply a DIFFERENT fix strategy (not a variation of Attempt 1)
  → Verify
  → If PASS: done
  → If FAIL: STOP — escalate to user

DO NOT attempt 3+. Present:
  - Root cause diagnosis
  - Two strategies attempted with failure reasons
  - Suggested next step requiring human judgment
```

### Strategy Differentiation

The second attempt MUST use a fundamentally different approach:

| Attempt 1 approach | Attempt 2 must NOT be | Attempt 2 should be |
|---|---|---|
| Fix the call site | Fix a different call site with slight variation | Fix the callee, or change the type/interface |
| Add a null check | Add a different null check | Restructure to avoid the null path entirely |
| Update the import | Try a different import path | Check if the dependency needs updating |
| Fix the type annotation | Tweak the same annotation | Change the underlying value or function signature |

---

## Regression Test Codification

Every fix represents a learning about the system's failure mode. Codify this learning as a test:

### When to Suggest a Regression Test

- The error was a logic bug, runtime error, or test failure
- A test file already exists for the affected module
- The fix involved a non-obvious boundary condition
- The bug could plausibly recur after future refactoring

### When NOT to Suggest

- The error was a syntax/compile error (compiler catches these)
- The error was a dependency/config issue (not testable in unit tests)
- The existing test suite already covers this case
- Adding the test would be trivial or redundant

### Regression Test Structure

```
// Test guards against: [one-sentence description of what was broken]
// Related fix: [file:line where the fix was applied]
test("should handle [specific condition that triggered the bug]", () => {
  // Arrange: set up the exact condition that caused the error
  // Act: execute the code path that was broken
  // Assert: verify the correct behavior (not just absence of error)
});
```

---

## When NOT to Fix

Sometimes the correct action is NOT to apply a fix:

1. **The error is in third-party code** — report the issue upstream, apply a local workaround, and document it.
2. **The error requires architectural changes** — explain the root cause and the architectural change needed. Let the user decide the scope.
3. **The error is environment-specific** — if the error only occurs in a specific environment (CI, production) that you can't reproduce, explain the likely cause and suggest verification steps for that environment.
4. **The fix would break other things** — if the blast radius is high and you're not confident in all the implications, present the diagnosis and let the user choose the fix approach.
5. **The "error" is intentional** — some warnings or errors are expected (e.g., deprecation warnings during a planned migration). Confirm with the user before "fixing" something that may be intentional.
6. **Confidence is LOW and 2 attempts failed** — escalate to the user with diagnosis and evidence rather than continuing to iterate.
