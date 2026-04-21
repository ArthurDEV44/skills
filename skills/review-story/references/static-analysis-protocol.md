# Static Analysis Protocol — Phase 2.5

Deterministic checks run before AI review. Cheap, fast, high-precision.

## 2.5a — Language-Specific Tools

| Language | Linter/Checker | Command |
|----------|---------------|---------|
| Rust | Clippy + format check | `cargo clippy --all-targets 2>&1; cargo fmt --check 2>&1` |
| TypeScript/JS | ESLint + Prettier | `bunx eslint {changed_files} 2>&1; bunx prettier --check {changed_files} 2>&1` |
| Python | Ruff (lint + format) | `ruff check {changed_files} 2>&1; ruff format --check {changed_files} 2>&1` |
| Go | vet + staticcheck | `go vet ./... 2>&1; staticcheck ./... 2>&1` |

Only run tools that are already configured in the project (check for config files: `.eslintrc*`, `rustfmt.toml`, `ruff.toml`, `pyproject.toml [tool.ruff]`, etc.). Do NOT install or configure new tools.

## 2.5b — Type Checking (if configured)

| Language | Command |
|----------|---------|
| TypeScript | `bunx tsc --noEmit 2>&1` |
| Python (mypy) | `mypy {changed_files} 2>&1` |
| Rust | Already checked by `cargo clippy` |

## 2.5c — Collect Results

Parse output into structured findings:
- Errors → map to MUST_FIX
- Warnings → map to SHOULD_FIX
- Info/style → map to CONSIDER

These findings go directly into Phase 5's remediation queue. They do NOT need AI review — they are deterministic and high-precision.

## 2.5d — Git Blame Hotspot Analysis

Run `git log --format='%H' --follow {changed_files} | head -20` and `git log --shortstat main...HEAD` to identify:
- Files with high churn (changed frequently)
- Recent multi-author files (coordination risks)
- Large single commits (potential rushed changes)

Feed hotspot files as priority targets into risk hypotheses. This mirrors Anthropic Code Review's 4th agent (git blame/history analysis).

## 2.5e — Generate Risk Hypotheses

Based on the changed files, research brief, intent context, and static analysis results, generate 3-5 targeted risk hypotheses before spawning the review agents. Each hypothesis is a specific, falsifiable claim:

```
- "The auth middleware bypass on line X may allow unauthenticated access to {endpoint}"
- "The N+1 query pattern in the loop at Y:Z will cause timeout at scale"
- "The missing input validation at A:B could allow {injection type}"
- "The error handling at C:D silently swallows {specific failure mode}"
```

Pass these hypotheses to the Phase 3 and Phase 4 agents as **priority investigation targets**. An agent assigned to prove/disprove a specific hypothesis produces 3-5x more targeted findings than one scanning generically.
