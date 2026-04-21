# Analysis Checklist — Issue Detection Patterns

## Table of Contents

- [Dead Code Detection](#dead-code-detection)
- [Complexity Analysis](#complexity-analysis)
- [Duplication Detection](#duplication-detection)
- [SOLID Violations](#solid-violations)
- [Performance Issues](#performance-issues)
- [Frontend-Specific Issues](#frontend-specific-issues)
- [Legacy Pattern Detection](#legacy-pattern-detection)
- [Type Safety Issues](#type-safety-issues)
- [Error Handling Issues](#error-handling-issues)
- [Severity Criteria](#severity-criteria)

---

## Dead Code Detection

### Patterns to detect:

| Pattern | Indicators | Languages |
|---------|-----------|-----------|
| Unused imports | Import not referenced in file | All |
| Unused variables | Declared but never read | All |
| Unused functions | Defined but never called (check exports) | All |
| Unused types/interfaces | Defined but never referenced | TS, Rust, Go, Java |
| Commented-out code | Blocks of commented code (>3 lines) | All |
| Unreachable branches | `if (false)`, `return` before code, dead `else` branches | All |
| Unused CSS classes | Classes defined but not used in templates | CSS/SCSS |
| Unused dependencies | Listed in manifest but not imported anywhere | All |
| Feature-flagged dead code | Code behind permanently-off flags | All |

### Verification before removal:

- Check if the symbol is exported (may be used by external consumers)
- Check for dynamic references: `window[name]`, `getattr()`, reflection
- Check test files — some "unused" code is only used in tests
- Check for conditional compilation: `#[cfg()]`, `#ifdef`, platform-specific code
- If uncertain, flag as LOW severity rather than removing

---

## Complexity Analysis

### Thresholds:

| Metric | Acceptable | Warning | Critical |
|--------|-----------|---------|----------|
| Function length | <30 lines | 30-60 lines | >60 lines |
| Cyclomatic complexity | <8 | 8-15 | >15 |
| Nesting depth | <3 levels | 3-4 levels | >4 levels |
| Parameter count | <4 | 4-5 | >5 |
| Class/module size | <300 lines | 300-500 lines | >500 lines |
| File size | <500 lines | 500-800 lines | >800 lines |

### Refactoring strategies by complexity type:

**Long functions →** Extract Method
- Identify cohesive blocks that do one thing
- Name the extracted function after what it does, not how
- Pass only needed variables as parameters

**Deep nesting →** Guard Clauses + Early Returns
- Invert conditions and return early
- Extract nested blocks into separate functions
- Use flatMap/reduce instead of nested loops where appropriate

**Long parameter lists →** Parameter Object
- Group related parameters into a typed object
- Especially when the same parameter group appears in multiple functions

**God class/module →** Extract Class
- Identify distinct responsibilities
- Group related methods and state together
- Extract into focused modules with clear boundaries

---

## Duplication Detection

### Patterns to detect:

| Type | Description | Action |
|------|-------------|--------|
| Exact duplicates | Identical code blocks (>5 lines) | Extract to shared function |
| Near duplicates | Similar structure with minor variations | Parameterize the differences |
| Structural duplicates | Same algorithm with different types | Use generics/templates |
| API pattern duplication | Repeated fetch/query/handler patterns | Extract a pattern/factory |
| Test duplication | Repeated test setup/teardown | Use fixtures/helpers |

### When NOT to deduplicate:

- Two similar blocks that serve different business purposes and may diverge
- Duplication across module boundaries where coupling would be worse
- Simple one-liners (creating a function for `arr.filter(x => x > 0)` adds noise)
- Test code where readability matters more than DRY

---

## SOLID Violations

### Single Responsibility Principle (SRP)

**Indicators of violation:**
- File/class has multiple unrelated import groups
- Functions named with "and" (`fetchAndTransform`, `validateAndSave`)
- Module handles both business logic and I/O (DB, HTTP, file system)
- Changes in unrelated features require editing the same file

**Fix:** Extract responsibilities into focused modules.

### Open/Closed Principle (OCP)

**Indicators:**
- Large switch/if-else chains that grow with each new variant
- Adding a new feature requires modifying existing code (not extending)
- No extension points for behavior variation

**Fix:** Replace conditional with Strategy pattern or polymorphism.

### Liskov Substitution Principle (LSP)

**Indicators:**
- Subclass throws "not implemented" for inherited methods
- Type checks (`instanceof`, `typeof`) before calling methods
- Subtypes that silently change base behavior semantics

**Fix:** Ensure subtypes honor the base contract. Restructure hierarchy if needed.

### Interface Segregation Principle (ISP)

**Indicators:**
- Interfaces with >7 methods
- Implementors that leave methods as no-ops or throw
- Consumers that use only 1-2 methods of a large interface

**Fix:** Split into smaller, role-specific interfaces.

### Dependency Inversion Principle (DIP)

**Indicators:**
- Direct instantiation of dependencies inside business logic (`new Database()`)
- Hard-coded imports of concrete implementations
- Business logic directly calling HTTP/DB/file system without abstraction

**Fix:** Inject dependencies via constructor/parameters. Depend on interfaces.

---

## Performance Issues

### Universal patterns:

| Issue | Indicator | Severity |
|-------|-----------|----------|
| N+1 queries | Loop containing DB/API calls | HIGH |
| Missing indexes | `WHERE`/`ORDER BY` on unindexed columns | HIGH |
| Blocking I/O | Sync file/network calls on main thread | HIGH |
| Unbounded queries | `SELECT *` without `LIMIT`, fetching all rows | MEDIUM |
| Unnecessary allocations | Creating objects in hot loops | MEDIUM |
| Missing caching | Repeated identical expensive computations | MEDIUM |
| String concatenation in loops | Building strings with `+=` in loops | LOW |
| Premature optimization | Complex optimization for non-hot paths | LOW (flag, don't fix) |

### JavaScript/TypeScript specific:

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Re-renders (React) | Missing `useMemo`/`useCallback` for expensive ops, props causing child re-renders | HIGH |
| Missing keys in lists | `key={index}` or missing key prop | MEDIUM |
| Large bundle imports | `import moment from 'moment'` instead of individual functions | HIGH |
| Sync `require()` | Dynamic `require()` that defeats tree-shaking | MEDIUM |
| Memory leaks | `useEffect` without cleanup, event listeners not removed | HIGH |
| Layout thrashing | Reading DOM then writing DOM in a loop | HIGH |

### Rust specific:

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Unnecessary clones | `.clone()` where a borrow would work | MEDIUM |
| Allocation in hot loops | `Vec::new()` inside loops, `String::from()` repeated | HIGH |
| Missing `&str` parameters | Functions taking `String` when `&str` suffices | LOW |
| Unwrap in library code | `.unwrap()` instead of proper error handling | HIGH |
| Blocking in async | Sync I/O inside `async fn` without `spawn_blocking` | HIGH |

### Python specific:

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Global imports of heavy modules | `import pandas` at top level when not always needed | MEDIUM |
| List where generator suffices | `[x for x in range(1M)]` when iteration-only | MEDIUM |
| Missing `__slots__` | Classes with many instances but no `__slots__` | LOW |
| Repeated dictionary lookups | `d[key]` multiple times without local variable | LOW |

---

## Frontend-Specific Issues

### Core Web Vitals impact:

| Issue | Affects | Severity |
|-------|---------|----------|
| Render-blocking CSS/JS in `<head>` | LCP | HIGH |
| No `width`/`height` on images | CLS | HIGH |
| No `loading="lazy"` on below-fold images | LCP | MEDIUM |
| No `fetchpriority="high"` on hero image | LCP | MEDIUM |
| Long JavaScript tasks (>50ms) | INP | HIGH |
| Unminified/uncompressed assets | LCP | MEDIUM |
| No font preload, no `font-display: swap` | CLS, LCP | MEDIUM |
| Third-party scripts blocking render | LCP, INP | HIGH |
| No `<link rel="preconnect">` for external origins | LCP | LOW |
| Inline critical CSS missing | LCP | MEDIUM |

### Bundle and loading:

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Full library imports | `import _ from 'lodash'` instead of `import get from 'lodash/get'` | HIGH |
| No code splitting | Single bundle for entire app | HIGH |
| No lazy routes | All routes loaded eagerly | MEDIUM |
| Duplicate dependencies | Same library bundled in multiple chunks | MEDIUM |
| Unoptimized images | Large PNGs/JPGs without WebP/AVIF conversion | MEDIUM |
| Missing compression | No gzip/brotli on server responses | HIGH |

### Accessibility (a11y):

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Missing alt text | `<img>` without `alt` attribute | HIGH |
| Non-semantic HTML | `<div onClick>` instead of `<button>` | MEDIUM |
| Missing ARIA labels | Interactive elements without accessible names | MEDIUM |
| No skip navigation | No "Skip to main content" link | LOW |
| Poor color contrast | Text color ratio < 4.5:1 (AA) | MEDIUM |
| Missing focus indicators | `outline: none` without replacement | MEDIUM |

---

## Legacy Pattern Detection

### JavaScript/TypeScript:

| Legacy Pattern | Modern Replacement | Severity |
|---------------|-------------------|----------|
| `var` declarations | `const` / `let` | LOW |
| Callback pyramids | `async/await` | MEDIUM |
| `.then().catch()` chains | `async/await` with try/catch | LOW |
| `require()` / `module.exports` | `import` / `export` (ESM) | MEDIUM |
| Class components (React) | Functional components + hooks | MEDIUM |
| `componentDidMount` etc. | `useEffect` | MEDIUM |
| `PropTypes` runtime | TypeScript types | LOW |
| jQuery DOM manipulation | Native DOM API or framework reactivity | HIGH |
| `XMLHttpRequest` | `fetch` API | LOW |
| `moment.js` | `date-fns` or `Temporal` API | MEDIUM |

### Rust:

| Legacy Pattern | Modern Replacement | Severity |
|---------------|-------------------|----------|
| Manual `impl Iterator` | `impl IntoIterator` or iterator combinators | LOW |
| Old error patterns (`Box<dyn Error>`) | `thiserror` / `anyhow` | LOW |
| `.unwrap()` in library code | `?` operator with proper error types | HIGH |
| Manual string formatting | `format!()` with named parameters | LOW |
| Explicit lifetime where elidable | Let the compiler elide | LOW |

### Python:

| Legacy Pattern | Modern Replacement | Severity |
|---------------|-------------------|----------|
| `%` string formatting | f-strings | LOW |
| `os.path` manipulation | `pathlib.Path` | LOW |
| `type()` checks | `isinstance()` | LOW |
| Manual file open/close | `with` context manager | MEDIUM |
| No type hints | Type annotations | MEDIUM |
| `requirements.txt` only | `pyproject.toml` with dependency groups | LOW |

---

## Type Safety Issues

| Issue | Indicator | Severity |
|-------|-----------|----------|
| `any` type (TS) | Explicit or inferred `any` | MEDIUM |
| Missing return types | Public functions without explicit return type | LOW |
| Unsafe casts | `as unknown as Type`, `as any` | HIGH |
| Missing null checks | Optional values accessed without `?.` or guards | HIGH |
| Untyped function boundaries | Public API functions without parameter types | MEDIUM |
| Loose generics | `Array<any>`, `Record<string, any>` | MEDIUM |
| Type assertions over type guards | `x as T` instead of narrowing with `is` | MEDIUM |

---

## Error Handling Issues

| Issue | Indicator | Severity |
|-------|-----------|----------|
| Swallowed errors | Empty `catch {}` blocks | HIGH |
| Generic catch-all | `catch (e) { console.log(e) }` without recovery | MEDIUM |
| Missing error boundaries | React app without ErrorBoundary components | MEDIUM |
| No retry logic | Network calls without retry for transient failures | LOW |
| Panic in library code | `unwrap()`, `expect()` in Rust lib code | HIGH |
| Missing validation | User input used directly without validation | HIGH |
| No graceful degradation | Feature fails silently or crashes entirely | MEDIUM |

---

## Severity Criteria

| Severity | Definition | Action |
|----------|-----------|--------|
| CRITICAL | Causes bugs, crashes, data loss, or security vulnerabilities | Fix immediately (P0) |
| HIGH | Significant performance impact, major tech debt, blocks maintainability | Fix in this refactoring pass (P1) |
| MEDIUM | Code smell, moderate tech debt, maintainability concern | Fix if time allows (P2) |
| LOW | Style preference, minor improvement, cosmetic | Flag but may skip (P3) |
