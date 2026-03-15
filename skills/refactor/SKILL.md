---
model: opus
name: refactor
disable-model-invocation: true
description: "Multi-agent refactoring workflow that analyzes, plans, executes, simplifies, and validates code improvements across files or entire directories. Optimizes performance, cleans legacy code, enforces SOLID/clean code principles, maximizes Lighthouse scores for frontend, and improves overall code quality. Orchestrates agent-explore (architecture analysis), agent-websearch (best practices), agent-docs (API verification), and /simplify (post-execution refinement) in a 7-step pipeline: scope, analyze, research, plan, execute, simplify, validate. Use when the user explicitly invokes /refactor with a file, folder, or scope modifier. Do NOT auto-trigger for minor code changes, bug fixes, or feature implementation."
argument-hint: "[file-or-folder] [--perf|--clean|--types|--solid|--lighthouse]"
---

# refactor — Multi-Agent Code Refactoring Pipeline

## Overview

refactor is a 7-step pipeline that systematically improves code quality by combining:
1. **Scope detection** (orchestrator) — identify target files, stack, framework, and baseline metrics
2. **Analysis** (orchestrator + agent-explore) — catalog issues, code smells, performance bottlenecks
3. **Research** (agent-websearch + agent-docs) — best practices and correct API patterns for the stack
4. **Planning** (orchestrator) — structured refactoring plan with priority ranking
5. **Execution** (orchestrator) — apply changes incrementally, one logical unit per pass
6. **Simplification** (/simplify) — review changed code for reuse, clarity, and efficiency; apply refinements
7. **Validation** (orchestrator) — run tests, lints, type checks, measure before/after metrics

Steps 2b and 3 run in parallel. The orchestrator synthesizes findings and applies changes.

## Execution Flow

```
$ARGUMENTS -> [file-or-folder] [--perf|--clean|--types|--solid|--lighthouse]
     |
     v
+---------------+
|  Step 1:      |
|  SCOPE        |  <- Detect stack, read files, capture baseline metrics
|  (instant)    |
+-------+-------+
        |
        v
+-------+-------+
|  Step 2a:     |
|  SELF-ANALYZE |  <- Orchestrator reads code and catalogs issues
|  (analysis)   |
+-------+-------+
        |
        v
+-------+----------------------------+
|            PARALLEL                 |
|  +------------+  +--------------+   |
|  | Step 2b:   |  | Step 3:      |   |
|  | EXPLORE    |  | RESEARCH     |   |
|  | (codebase) |  | (web + docs) |   |
|  +-----+------+  +------+------+   |
|        |                |           |
+--------+----------------+-----------+
         |                |
         v                v
+--------+----------------+--------+
|  Step 4:                         |
|  PLAN                            |  <- Merge findings, create ranked plan
|  (orchestrator)                  |
+----------------+-----------------+
                 |
                 v
+----------------+-----------------+
|  Step 5:                         |
|  EXECUTE                         |  <- Apply changes incrementally
|  (orchestrator)                  |
+----------------+-----------------+
                 |
                 v
+----------------+-----------------+
|  Step 6:                         |
|  SIMPLIFY                        |  <- /simplify reviews changed code
|  (/simplify skill)               |
+----------------+-----------------+
                 |
                 v
+----------------+-----------------+
|  Step 7:                         |
|  VALIDATE                        |  <- Tests, lints, metrics comparison
|  (orchestrator)                  |
+----------------+-----------------+
```

## Runtime Output Format

Before each step, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Step N/7] STEP_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Between major steps, print a thin separator: `───────────────────────────────`

## Step-by-Step Execution

### Step 1 — Scope Detection (Orchestrator, Instant)

Print: `[Step 1/7] SCOPE DETECTION`

**1a. Parse arguments:**

- If `$ARGUMENTS` contains a file path → target that file
- If `$ARGUMENTS` contains a folder path → target that folder
- If `$ARGUMENTS` contains a scope modifier (`--perf`, `--clean`, etc.) → apply that modifier (see Scope Modifiers table)
- If `$ARGUMENTS` is empty → ask the user what to refactor

**1b. Identify target files:**

If user specifies a file → read that file.
If user specifies a folder → Glob for source files in that folder:
```
Glob: {folder}/**/*.{ts,tsx,js,jsx,py,rs,go,java,vue,svelte,css,scss}
```
If user says "refactor this project" → detect from manifest files and read the main source directories.

**1c. Detect the stack:**

| Signal | Detection |
|--------|-----------|
| Language | File extensions (.rs, .ts, .py, .go, .java, etc.) |
| Framework | Import statements, manifest files (package.json deps, Cargo.toml deps, etc.) |
| Build tool | vite.config, webpack.config, tsconfig, Cargo.toml, pyproject.toml |
| Test framework | jest.config, vitest.config, pytest.ini, Cargo test, go test |
| Linter | .eslintrc, biome.json, clippy, ruff, golangci-lint |

**1d. Capture baseline metrics** (run what's available):

```bash
# Frontend
npx lighthouse --output=json --chrome-flags="--headless" {url}  # if URL available
npx tsc --noEmit 2>&1 | tail -5  # type errors count
npx eslint {target} --format compact 2>&1 | tail -5  # lint errors count

# Any language
wc -l {files}  # line counts
git diff --stat  # current state

# Rust
cargo clippy --message-format=short 2>&1 | tail -10
cargo test 2>&1 | tail -5

# Python
ruff check {target} 2>&1 | tail -10
python -m pytest --co -q 2>&1 | tail -5
```

Record available metrics as the **baseline snapshot** for Step 6 comparison.

**1e. Read all target files** using the Read tool. For large scopes (>15 files), prioritize:
1. Entry points (main, index, app, lib)
2. Largest files (most likely to need refactoring)
3. Files with the most imports/dependencies (architectural hubs)

### Step 2a — Self-Analysis (Orchestrator)

Print: `[Step 2/7] ANALYSIS`

After reading the code, catalog issues across these categories. See [references/analysis-checklist.md](references/analysis-checklist.md) for the full checklist.

**Issue categories (always check all):**

1. **Dead code** — unused imports, unreachable branches, commented-out code, unused variables/functions/types
2. **Complexity** — functions >40 lines, cyclomatic complexity >10, deep nesting (>3 levels), long parameter lists (>4)
3. **Duplication** — repeated logic blocks, copy-pasted patterns, near-identical functions
4. **SOLID violations** — god classes/modules, tight coupling, missing abstractions, mixed responsibilities
5. **Performance** — N+1 queries, missing memoization, unnecessary re-renders, blocking operations, unoptimized loops
6. **Frontend-specific** — render-blocking resources, unoptimized images, missing lazy loading, large bundle imports, layout shifts, poor accessibility
7. **Legacy patterns** — callbacks instead of async/await, var instead of const/let, class components instead of hooks, CommonJS instead of ESM, deprecated APIs
8. **Type safety** — `any` types, missing null checks, unsafe casts, untyped function boundaries
9. **Error handling** — swallowed errors, missing try/catch, generic catch-all, no error boundaries

**For each issue, record:**
- Category (from list above)
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- File and line number
- Brief description
- Estimated impact (performance, maintainability, correctness)

### Step 2b — Architecture Exploration (agent-explore, Parallel)

Print: `[Step 2/7] ARCHITECTURE EXPLORATION (parallel)`

Spawn agent-explore to map the architecture and find systemic patterns:

```
Agent(
  description: "Analyze architecture for refactoring",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-explore"
)
```

The agent investigates:
- Module dependency graph and coupling analysis
- Architectural layers and boundary violations
- Shared state patterns and data flow
- Import cycles or circular dependencies
- Convention inconsistencies across the codebase

### Step 3 — Research (agent-websearch + agent-docs, Parallel with 2b)

Print: `[Step 3/7] RESEARCH (parallel)`

Spawn both agents in a SINGLE message for true parallel execution with Step 2b:

**agent-websearch** — stack-specific optimization best practices:
```
Agent(
  description: "Research {framework} optimization",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-websearch"
)
```

**agent-docs** — verify correct API usage for the detected framework/libraries:
```
Agent(
  description: "Check docs for {libraries}",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-docs"
)
```

If no specific libraries need verification, skip agent-docs.

### Step 4 — Refactoring Plan (Orchestrator)

Print: `[Step 4/7] PLANNING`

Merge all findings from Steps 2a, 2b, and 3. Create a structured plan.

**4a. Priority ranking:**

Rank all identified issues by this matrix:

| Priority | Criteria |
|----------|----------|
| P0 — Critical | Bugs, security issues, crashes, data loss risks |
| P1 — High | Performance bottlenecks, severe Lighthouse failures, N+1 queries, memory leaks |
| P2 — Medium | SOLID violations, duplication, missing types, legacy patterns |
| P3 — Low | Code style, minor optimizations, cosmetic improvements |

**4b. Group into change sets:**

Group related changes into atomic units. Each change set must:
- Be independently testable
- Not break compilation or tests if applied alone
- Touch the minimum number of files
- Have a clear before/after description

**4c. Present the plan to the user:**

Print the plan using this enhanced template with clear visual hierarchy:

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFACTORING PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Scope:** {N} files | **Language:** {lang} | **Framework:** {framework}
**Issues found:** {N} total — {N} P0 | {N} P1 | {N} P2 | {N} P3
**Change sets:** {N} | **Estimated files touched:** {N}

───────────────────────────────

### [CS-1] {Title}  ·  P{n}  ·  {files_count} file(s)

| | |
|---|---|
| **Files** | `path/to/file.ext` |
| **Issue** | {description} |
| **Action** | {what will change} |
| **Impact** | {expected improvement} |

───────────────────────────────

### [CS-2] {Title}  ·  P{n}  ·  {files_count} file(s)

| | |
|---|---|
| **Files** | `path/to/file.ext` |
| **Issue** | {description} |
| **Action** | {what will change} |
| **Impact** | {expected improvement} |

───────────────────────────────

### Skipped (out of scope)
- {issues intentionally not addressed and why}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**4d. Scope Guard — Assess plan size:**

Count the plan metrics:
- `change_sets`: total number of change sets
- `files_touched`: total unique files across all change sets
- `estimated_loc`: rough estimate of lines that will change

**If the plan exceeds these thresholds → escalate to PRD:**

| Metric | Direct execution OK | Escalate to PRD |
|--------|-------------------|-----------------|
| Change sets | ≤ 7 | > 7 |
| Files touched | ≤ 20 | > 20 |
| Estimated LOC delta | ≤ 800 | > 800 |
| Cross-service/module | Same module | Multiple services or public API changes |

If ANY threshold is exceeded:

1. Print the scope warning:
```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE ALERT — Plan exceeds single-session capacity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This refactoring touches **{N} change sets** across **{N} files** (~{N} LOC).
Direct execution risks incomplete application and difficult rollback.

**Recommendation:** Generate a PRD with phased execution via `/write-prd`.
Each phase becomes an independently implementable story.
```

2. Use AskUserQuestion to let the user decide:
```json
{
  "questions": [{
    "question": "This refactoring is too large for a single session. How would you like to proceed?",
    "header": "Scope",
    "options": [
      { "label": "Generate PRD", "description": "Launch /write-prd to create a phased plan with implementable stories (recommended)" },
      { "label": "Execute anyway", "description": "Proceed with direct execution despite the large scope" },
      { "label": "Reduce scope", "description": "Go back and select only high-priority change sets to execute now" }
    ]
  }]
}
```

- If user selects **"Generate PRD"** → invoke the `/write-prd` skill with the analysis findings as context. STOP the refactor pipeline.
- If user selects **"Execute anyway"** → proceed to Step 5 with the full plan.
- If user selects **"Reduce scope"** → ask which change sets to keep (P0/P1 only, or specific CS-N IDs), rebuild the plan, and re-evaluate the scope guard.

**4e. User confirmation — Interactive decision (if scope is within thresholds):**

Use AskUserQuestion to present an interactive menu:

```json
{
  "questions": [{
    "question": "The refactoring plan has {N} change sets across {N} files. Ready to execute?",
    "header": "Execute",
    "options": [
      { "label": "Yes, execute", "description": "Apply all {N} change sets in priority order with verification after each" },
      { "label": "Modify plan", "description": "Adjust change sets, priorities, or scope before executing" },
      { "label": "Cancel", "description": "Abort the refactoring — no changes will be made" }
    ]
  }]
}
```

- If user selects **"Yes, execute"** → proceed to Step 5.
- If user selects **"Modify plan"** → ask what to change, adjust the plan, re-present it.
- If user selects **"Cancel"** → print summary of findings without applying changes. STOP.

**GATE:** User explicitly confirms execution via the interactive menu.

### Step 5 — Execute (Orchestrator)

Print: `[Step 5/7] EXECUTION`

Apply changes incrementally, one change set at a time, highest priority first.

Before each change set, print:
```
── Applying [CS-{N}/{total}] {Title} ──
```

After each change set passes verification, print:
```
   [CS-{N}] Applied successfully
```

**Execution rules:**

1. **One change set per pass** — apply all edits for one logical change, then verify before moving to the next.
2. **Use Edit tool** for surgical changes — do NOT rewrite entire files unless the changes affect >60% of lines.
3. **Preserve existing style** — match indentation, naming conventions, import ordering of the surrounding code.
4. **Preserve behavior** — refactoring must not change observable behavior unless fixing an actual bug (P0).
5. **After each change set**, run the fastest available verification:
   - Type check (`tsc --noEmit`, `cargo check`, `mypy`)
   - Lint (`eslint`, `clippy`, `ruff`)
   - Tests if fast (<30s)

**Frontend-specific execution — Lighthouse optimization:**

Apply changes following the fix priority order from [references/optimization-patterns.md](references/optimization-patterns.md):
1. TTFB optimization (server response, caching)
2. LCP optimization (preload hero, critical CSS, SSR)
3. INP optimization (break long tasks, debounce, Web Workers)
4. CLS optimization (explicit dimensions, font-display)
5. Bundle size (tree-shaking, code splitting, lazy imports)
6. Accessibility (semantic HTML, ARIA, focus management)

### Step 6 — Simplify (/simplify Skill)

Print: `[Step 6/7] SIMPLIFICATION`

After all change sets are applied and verified individually, invoke the `/simplify` skill to perform a final refinement pass on the changed code.

**6a. Invoke /simplify:**

Use the Skill tool to invoke `/simplify`. This will trigger the code-simplifier agent which autonomously reviews all recently modified code and applies refinements for:

- **Clarity** — reduce unnecessary complexity, nesting, and redundant abstractions
- **Consistency** — enforce project coding standards and conventions (from CLAUDE.md)
- **Reuse** — identify duplicated logic introduced across change sets and consolidate
- **Readability** — improve variable/function names, remove unnecessary comments, prefer explicit code over dense one-liners
- **Balance** — avoid over-simplification that would reduce maintainability or clarity

**6b. Verify simplification:**

After `/simplify` completes, run a quick verification to ensure the refinements didn't introduce regressions:

```bash
# Type check (fastest available)
npx tsc --noEmit  # or cargo check, mypy
# Lint
npx eslint {target}  # or cargo clippy, ruff
```

If any verification fails after simplification, revert the simplification changes for the affected files and note them in the final summary.

**6c. Print result:**

```
   [Simplify] Refined {N} file(s) — {brief summary of improvements}
```

───────────────────────────────

### Step 7 — Validate (Orchestrator)

Print: `[Step 7/7] VALIDATION`

After all change sets are applied, run comprehensive validation.

**7a. Run all available checks:**

```bash
# Type checking
npx tsc --noEmit            # TypeScript
cargo check                 # Rust
mypy {target}               # Python

# Linting
npx eslint {target}         # JS/TS
cargo clippy                # Rust
ruff check {target}         # Python

# Tests
npx vitest run              # or jest, pytest, cargo test, go test
```

**7b. Compare metrics (before vs. after):**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFACTORING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Type errors | {n} | {n} | {+/-n} |
| Lint warnings | {n} | {n} | {+/-n} |
| Test results | {pass/fail} | {pass/fail} | — |
| Lines of code | {n} | {n} | {+/-n} |
| Files changed | — | {n} | — |
| [Lighthouse LCP] | {n}s | {n}s | {+/-n}s |
| [Lighthouse INP] | {n}ms | {n}ms | {+/-n}ms |
| [Lighthouse CLS] | {n} | {n} | {+/-n} |
| [Bundle size] | {n}KB | {n}KB | {+/-n}KB |
```

Metrics in brackets are included only when applicable (frontend projects with measurable endpoints).

**7c. Regression check:**

If any test fails that was passing before:
1. Identify the change set that caused the regression
2. Revert that specific change set
3. Report the regression to the user with root cause
4. Ask whether to attempt an alternative fix or skip that change set

**7d. Summary:**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Applied:** {N}/{M} change sets successfully
**Reverted:** {N} (due to regressions)
**Skipped:** {N} (per user request or out of scope)

───────────────────────────────

### Key Improvements
- {Improvement 1 with measured impact}
- {Improvement 2 with measured impact}

### Remaining Opportunities
- {Issue not addressed and why}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Hard Rules

1. ALWAYS read code before analyzing — never infer issues from file names alone.
2. ALWAYS present the plan and use AskUserQuestion for interactive confirmation before executing changes. NEVER use plain text questions like "Shall I proceed?" — always use the AskUserQuestion tool with selectable options.
3. Steps 2b and 3 run in PARALLEL — spawn all applicable agents in a single message.
4. Step 5 applies changes ONE CHANGE SET AT A TIME — never batch all changes.
5. Preserve behavior — refactoring MUST NOT change observable behavior (unless fixing a P0 bug).
6. After each change set, run the fastest available verification.
7. Agent boundaries are strict — explore reads code, docs queries Context7, websearch fetches URLs.
8. Max 3 Context7 calls — agent-docs must stay within the hard limit.
9. Every issue must have file:line, severity, and specific remediation.
10. If tests fail after a change, REVERT that change set immediately.
11. Never add features during refactoring — scope is code quality only.
12. Compress analysis before passing to agents (<500 words).
13. Scope Guard is MANDATORY — if change sets > 7 OR files > 20 OR LOC > 800, present the escalation menu. If user selects "Generate PRD", invoke `/write-prd` and STOP the refactor pipeline.
14. Print `[Step N/7]` progress headers before each step — NEVER skip progress indicators.
15. ALWAYS invoke `/simplify` after execution (Step 6) — it is a mandatory refinement pass before final validation.
16. For architectural decisions and multi-file refactoring planning, use ultrathink for deep reasoning.

## Scope Modifiers

The user can narrow the refactoring focus with these modifiers:

| Modifier | Focus |
|----------|-------|
| `--perf` or "optimize performance" | Only P1 performance issues + Lighthouse |
| `--clean` or "clean up" | Dead code, duplication, legacy patterns only |
| `--types` or "improve types" | Type safety, `any` elimination, null checks |
| `--solid` or "improve architecture" | SOLID violations, coupling, abstractions |
| `--lighthouse` or "fix lighthouse" | Frontend-only: Core Web Vitals + bundle size |
| `--all` (default) | Full analysis across all categories |

## DO NOT

- Skip the analysis phase — you need to understand the code before changing it.
- Apply changes without presenting a plan first.
- Refactor code you haven't read.
- Add new features, tests, documentation, or comments beyond what's necessary for the refactoring.
- Change public API signatures without flagging it as a breaking change.
- Rewrite entire files when surgical edits suffice.
- Over-abstract — three similar lines are better than a premature abstraction.
- Ignore existing code style — match the project's conventions, don't impose new ones.
- Run Lighthouse in the pipeline unless the user has a running dev server or provides a URL.
- Ask "Shall I proceed?" or any plain text confirmation — ALWAYS use AskUserQuestion with selectable options.
- Skip the Scope Guard — ALWAYS evaluate plan size before presenting the execution menu.
- Execute a plan that exceeds scope thresholds without presenting the escalation menu first.

## Constraints (Three-Tier)

### ALWAYS
- Read code before analyzing — never infer issues from file names alone
- Run verification after each change set
- Compress analysis before passing to agents (<500 words)
- Print progress headers before each step

### ASK FIRST
- Execute the refactoring plan (require user confirmation via AskUserQuestion)
- Proceed when scope guard thresholds exceeded (>7 change sets, >20 files, >800 LOC)
- Change public API signatures (flag as breaking change)

### NEVER
- Apply changes without presenting a plan first
- Add features, tests, or documentation beyond what's necessary for the refactoring
- Rewrite entire files when surgical edits suffice
- Skip the Scope Guard evaluation
- Use plain text confirmations — always use AskUserQuestion with selectable options

## Done When

- [ ] Stack detected and baseline metrics captured (Step 1)
- [ ] Analysis complete with categorized issues (Step 2)
- [ ] Refactoring plan presented and approved by user via AskUserQuestion (Step 4)
- [ ] All change sets applied incrementally with per-set verification (Step 5)
- [ ] `/simplify` skill invoked and post-simplification verified (Step 6)
- [ ] Before/after metrics compared and results table displayed (Step 7)
- [ ] No test regressions — all tests that passed before still pass

## References

- [Analysis Checklist](references/analysis-checklist.md) — detailed issue detection patterns per category, language-specific indicators, severity criteria
- [Optimization Patterns](references/optimization-patterns.md) — frontend Lighthouse optimization, backend performance, database queries, bundle size reduction, memory management
- [Agent Orchestration](references/agent-orchestration.md) — exact Agent tool parameters, prompt templates, parallel spawning rules, output processing
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules, call budgets, authority hierarchy
- [Scope Guard](@~/.claude/skills/_shared/scope-guard.md) — shared threshold definitions and escalation protocol
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — ALWAYS/ASK FIRST/NEVER model
