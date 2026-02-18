# Agent Team Prompt Templates

Ready-to-use prompts for each orchestration pattern. Copy and adapt to your project.

## Leader Pattern

### Code Review

```
Create an agent team to review PR #<number>. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review independently and report findings.
Wait for all teammates before synthesizing a summary.
```

### Feature Implementation

```
Create an agent team with 3 teammates using Sonnet:
- "backend": implement the API endpoints in src/api/
- "frontend": build the React components in src/components/
- "tests": write integration tests in tests/
Require plan approval before any teammate starts coding.
Each teammate owns its own files — no overlap.
```

### Refactoring

```
Create an agent team to refactor the <module> module:
- "architect": plan the new structure, define interfaces (plan mode required)
- "implementer-1": refactor src/<submodule-a>/ after architect approval
- "implementer-2": refactor src/<submodule-b>/ after architect approval
- "tester": update and run all tests after implementers finish
Set task dependencies: implementers blocked by architect, tester blocked by implementers.
```

## Swarm Pattern

### Parallel Processing

```
Create an agent team to process all <N> modules in src/modules/.
Spawn one teammate per module. Each teammate:
1. Reads its assigned module
2. Applies <transformation> (e.g., migrate to TypeScript, add error handling)
3. Runs the module's tests
4. Reports results
No cross-module dependencies — pure parallel work.
```

### Bulk Migration

```
Create an agent team to migrate all API routes from Express to Hono.
Each teammate owns a route file — no file overlap.
Teammate list:
- "auth-routes": src/routes/auth.ts
- "user-routes": src/routes/users.ts
- "product-routes": src/routes/products.ts
- "order-routes": src/routes/orders.ts
Use Sonnet for each. Run tests after each migration.
```

## Pipeline Pattern

### Research → Design → Implement

```
Create an agent team with a pipeline workflow:
1. "researcher": investigate <topic>, write findings to docs/research.md
2. "architect": read research, design solution in docs/design.md (plan approval required)
3. "implementer": implement the approved design in src/
4. "reviewer": review the implementation, file issues

Set dependencies:
- architect blocked by researcher
- implementer blocked by architect
- reviewer blocked by implementer
```

### Data Pipeline

```
Create an agent team for the data migration pipeline:
1. "analyzer": analyze the current schema, document gaps in docs/analysis.md
2. "migrator": write migration scripts based on analysis
3. "validator": run migrations on test data and verify correctness
Sequential dependencies — each step depends on the previous.
```

## Watchdog Pattern

### Quality Gate

```
Create an agent team with a watchdog:
- "dev-1": implement feature A in src/features/a/
- "dev-2": implement feature B in src/features/b/
- "watchdog": continuously review dev-1 and dev-2's work for:
  - Code style consistency
  - Security vulnerabilities (OWASP top 10)
  - Test coverage (must be > 80%)
  - No shared state or file conflicts
The watchdog should message developers with issues as they arise.
```

### Debugging with Competing Hypotheses

```
Users report <bug description>.
Create an agent team to investigate different hypotheses:
- "hyp-memory": investigate memory leaks in <component>
- "hyp-race": investigate race conditions in <module>
- "hyp-config": investigate misconfiguration in <config-files>
- "hyp-external": investigate external service failures
- "watchdog": compare findings, challenge weak hypotheses, converge on root cause
Have the watchdog actively ask teammates to disprove each other's theories.
Update docs/investigation.md with consensus.
```

## Combined Patterns

### Full Project Kickstart (Leader + Pipeline + Watchdog)

```
Create an agent team for the new <feature> feature:

Phase 1 (parallel research):
- "researcher-api": investigate the external API docs
- "researcher-codebase": map existing code patterns

Phase 2 (design — blocked by phase 1):
- "architect": design the solution based on research (plan approval required)

Phase 3 (implementation — blocked by phase 2):
- "backend": implement API layer in src/api/
- "frontend": implement UI in src/components/
- "tests": write tests in tests/

Watchdog (runs throughout phase 3):
- "reviewer": continuously review implementation quality

Use Sonnet for all teammates. Require plan approval for architect.
Set dependencies between phases.
```

### Multi-Perspective Exploration

```
I'm designing a CLI tool that helps developers track TODO comments across
their codebase. Create an agent team to explore this from different angles:
- One teammate on UX
- One on technical architecture
- One playing devil's advocate
Have them each explore the problem independently and report findings.
```

### Codebase Audit

```
Create an agent team to audit the codebase:
- "security": audit for OWASP top 10 vulnerabilities across src/
- "performance": profile and identify bottlenecks in src/
- "deps": audit dependencies for vulnerabilities and outdated packages
- "tests": analyze test coverage gaps and write missing tests

Each reviewer works independently on their domain.
Synthesize all findings into docs/audit-report.md when done.
```

### Database + API + UI Feature

```
Create an agent team for the new <feature>:
- "db": write migrations and models in src/db/ (Sonnet)
- "api": implement endpoints in src/api/ — blocked by db (Sonnet)
- "ui": build React components in src/components/ — blocked by api (Sonnet)
- "e2e": write end-to-end tests in tests/e2e/ — blocked by ui (Sonnet)

Strict pipeline: db → api → ui → e2e.
Each teammate owns its directory exclusively.
```
