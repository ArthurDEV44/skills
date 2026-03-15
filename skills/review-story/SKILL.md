---
model: opus
name: review-story
description: "End-to-end review and correction workflow for an implemented user story or a complete PRD. Orchestrates 7 phases: intake, research (/meta-code pipeline), static analysis, parallel AI code review + security audit (3-layer: SAST/Secrets/SCA), risk-tiered remediation, and executive summary report. Signal-over-noise design: pre-filters generated files, scopes to diff+1 hop, targets 2-4 high-value findings per file. Does NOT commit or push — only reviews and fixes. Invoke with /review-story [prd-path] [story-id?]."
argument-hint: "[prd-path] [story-id?]"
---

# review-story — PRD Review & Correction Pipeline

Review the following: $ARGUMENTS

## Overview

Review and correction pipeline for already-implemented user stories. Takes a PRD (single story or full PRD), researches best practices, runs static analysis, then parallel AI code review + security audit, then risk-tiered remediation. Stops after correction — no commit or push.

**Key principles:**
- Research-informed review — understand best practices before judging code
- Static-first — run deterministic checks (cheap, fast, high-precision) before AI review
- Fresh-context reviewers — subagents with no bias toward the code (Actor/Critic isolation)
- Signal over noise — target 2-4 high-value findings per file, suppress style opinions
- Scope discipline — review diff + 1 dependency hop, not the entire codebase
- Risk-tiered fixes — auto-fix low-risk issues, confirm medium-risk, escalate high-risk
- No commit — the user controls when and how to commit

## Execution Flow

```
$ARGUMENTS -> [prd-path] [story-id?]
       |
       v
+---------------+
|  Phase 1:     |
|   INTAKE      |  <- Parse PRD, map changed files, pre-filter, size check
+-------+-------+
        | stories + criteria + filtered file map + size assessment
        v
+---------------+
|  Phase 2:     |
|  RESEARCH     |  <- /meta-code pipeline (best practices context)
+-------+-------+
        | research synthesis
        v
+---------------+
|  Phase 2.5:   |
| STATIC ANALYSIS| <- Deterministic checks: lint, format, type-check (fast, cheap)
+-------+-------+
        | mechanical issues caught
        v
+-------+-------------------+
|         PARALLEL           |
|  +----------+  +----------+|
|  | Phase 3: |  | Phase 4: ||
|  |  REVIEW  |  | SECURITY ||
|  |(quality) |  | (audit)  ||
|  +----+-----+  +----+-----+|
|       |              |      |
+-------+--------------+------+
        v              v
+-------------------------+
|      Phase 5:           |
|     REMEDIATE           |  <- Risk-tiered fixes, re-verify
|  (max 3 iterations)    |
+-----------+-------------+
            | zero CRITICAL/HIGH
            v
+-------------------------+
|      Phase 6:           |
|     SUMMARY             |  <- Executive summary + structured report
+-------------------------+
```

## Phase-by-Phase Execution

### Phase 1 — INTAKE

Parse the PRD and identify the review scope.

**1a. Parse arguments:**

- `$ARGUMENTS` contains a file path → read that file as the PRD
- `$ARGUMENTS` contains a story ID (e.g., `US-001`) → review only that story
- No story ID → review ALL stories in the PRD (full PRD review mode)
- If arguments are ambiguous → ask the user with AskUserQuestion

**1b. Read and parse the PRD:**

Read the PRD file. For each story in scope, extract:
- **Story title and description**
- **Acceptance criteria** (checklist items)
- **Quality gates** (from the PRD's Quality Gates section, if present)
- **Functional requirements** (from the PRD, if present)

**1c. Map changed files:**

Identify which files implement the stories. Run in order, stop at first success:

```bash
# 1. Branch diff against main (preferred) — include stat for line counts
git diff --name-only --stat main...HEAD

# 2. Fallback: branch diff against master
git diff --name-only --stat master...HEAD

# 3. Fallback: unstaged + staged changes
git diff --name-only --stat HEAD && git diff --name-only --stat --cached

# 4. Fallback: ask the user
```

If reviewing a single story, ask: "Which of these files implement {story_title}?" — or if the file list is small (<10 files), review all.

If reviewing a full PRD, review all changed files.

**1d. Pre-filter and size assessment:**

**Pre-filter** — remove non-reviewable files from scope:
- Lock files: `package-lock.json`, `yarn.lock`, `Cargo.lock`, `pnpm-lock.yaml`, `Gemfile.lock`, `poetry.lock`
- Generated files: `*.generated.*`, `*.g.dart`, `*.pb.go`, files with `@generated` header
- Vendor directories: `vendor/`, `node_modules/`, `third_party/`
- Build output: `dist/`, `build/`, `target/`, `.next/`
- Assets: binary files, images, fonts

**Size assessment** — compute total lines changed (additions + modifications):
- **< 400 lines:** Optimal — proceed normally
- **400-500 lines:** Acceptable — proceed with note
- **> 500 lines:** Flag for chunking — AI review quality degrades sharply above this threshold. Suggest reviewing by story or by logical grouping. Ask user how to proceed.

**1e. Read all files in scope:**

Read every file identified in 1c-1d using the Read tool. Also read **1 dependency hop**: for each changed file, identify its direct importers/callers and the files it imports. This provides context without reviewing the entire codebase.

**1f. Display review scope:**

```
## Review Scope

**Mode:** {Single Story: US-XXX | Full PRD}
**Stories in scope:** {count}
**Files to review:** {count} ({total_lines_changed} lines changed)
**Files filtered out:** {count} (lock files, generated, vendor)
**Size assessment:** {OPTIMAL | ACCEPTABLE | LARGE — consider chunking}

### Stories
- US-001: {title} — {status: found/not-found in changed files}
- US-002: {title} — ...

### Files
- path/to/file.ext (+{added} -{removed})
- ...

### Context Files (1-hop dependencies, read-only context)
- path/to/caller.ext (imports changed file)
- ...

Proceed with review?
```

**GATE:** User confirms scope (or scope is reasonable and auto-proceeds).

---

### Phase 2 — RESEARCH (mandatory /meta-code pipeline)

Research best practices to inform the review. The review is only as good as the reviewer's knowledge.

**2a. Spawn agent-websearch:**

```
Agent(
  description: "Research best practices for {feature_area}",
  prompt: <see references/review-protocols.md — Research prompt>,
  subagent_type: "agent-websearch"
)
```

Focus the research on:
- Best practices for implementing this type of feature
- Common mistakes and anti-patterns
- Security considerations specific to this feature type
- Testing best practices for this domain

Wait for completion. Extract key findings, compress to <500 words.

**2b. Detect codebase and libraries:**

```
Glob: Cargo.toml
Glob: package.json
Glob: pyproject.toml
Glob: go.mod
```

**2c. Spawn agent-explore + agent-docs in parallel:**

```
// If codebase detected:
Agent(
  description: "Explore codebase patterns for {feature_area}",
  prompt: <see references/review-protocols.md — Explore prompt>,
  subagent_type: "agent-explore"
)

// If libraries identified:
Agent(
  description: "Fetch docs for {libraries}",
  prompt: <see references/review-protocols.md — Docs prompt>,
  subagent_type: "agent-docs"
)
```

Spawn both in a SINGLE message for true parallel execution.

**2d. Synthesize research into a review brief:**

Combine all findings into a concise brief that informs Phases 3 and 4. Include:
- Best practices the implementation should follow
- Common pitfalls to check for
- Correct API usage for libraries in use
- Security considerations specific to this feature

**GATE:** Research synthesis complete.

---

### Phase 2.5 — STATIC ANALYSIS (deterministic, before AI review)

Run cheap, fast, high-precision deterministic checks before spending AI tokens. These catch mechanical issues and free AI review for deeper analysis.

**2.5a. Detect and run language-specific tools:**

| Language | Linter/Checker | Command |
|----------|---------------|---------|
| Rust | Clippy + format check | `cargo clippy --all-targets 2>&1; cargo fmt --check 2>&1` |
| TypeScript/JS | ESLint + Prettier | `npx eslint {changed_files} 2>&1; npx prettier --check {changed_files} 2>&1` |
| Python | Ruff (lint + format) | `ruff check {changed_files} 2>&1; ruff format --check {changed_files} 2>&1` |
| Go | vet + staticcheck | `go vet ./... 2>&1; staticcheck ./... 2>&1` |

Only run tools that are already configured in the project (check for config files: `.eslintrc*`, `rustfmt.toml`, `ruff.toml`, `pyproject.toml [tool.ruff]`, etc.). Do NOT install or configure new tools.

**2.5b. Type checking (if configured):**

| Language | Command |
|----------|---------|
| TypeScript | `npx tsc --noEmit 2>&1` |
| Python (mypy) | `mypy {changed_files} 2>&1` |
| Rust | Already checked by `cargo clippy` |

**2.5c. Collect static analysis results:**

Parse output into structured findings:
- Errors → map to MUST_FIX
- Warnings → map to SHOULD_FIX
- Info/style → map to CONSIDER

These findings go directly into Phase 5's remediation queue. They do NOT need AI review — they are deterministic and high-precision.

**GATE:** Static analysis complete. Results stored for Phase 5.

---

### Phase 3 — CODE REVIEW (parallel with Phase 4)

Spawn a fresh-context read-only subagent for thorough code review.

```
Agent(
  description: "Code review for {story_title}",
  prompt: <see references/review-protocols.md — Code Review prompt>,
  subagent_type: "agent-explore"
)
```

The review agent checks (8 categories — priority order):

**1. Acceptance Criteria Compliance**
- For each criterion: is it fully implemented? Cite `file:line` as evidence.
- Are there gaps between criteria and implementation?
- Are there criteria that are only partially met?
- Do function/class names reflect story vocabulary?
- Does the implementation include tests for each acceptance criterion?

**2. Correctness**
- Logic errors, off-by-one, incorrect conditions
- Null/undefined/None handling
- Error path coverage
- Edge cases (empty input, boundary values, concurrent access)
- State transitions — are all valid transitions handled?

**3. Architecture & Design**
- SOLID principles adherence (single responsibility, open-closed, etc.)
- Coupling/cohesion — is the code appropriately decoupled?
- Consistent abstraction levels within functions/modules
- Appropriate use of design patterns (not over-engineered, not under-structured)

**4. Error Handling & Logging**
- Exception specificity (no bare `catch` or `catch Exception`)
- Log levels appropriate (no `error` for info-level events)
- PII/secrets in logs — sensitive data must never be logged
- Structured logging where the project uses it
- Error messages actionable for debugging

**5. Quality**
- Naming clarity, readability
- Cognitive complexity — flag functions with complexity > 10, hard-gate > 20
- DRY violations (copy-pasted blocks)
- Consistency with project conventions
- Dead code, unused imports, leftover debug statements

**6. Performance**
- Unnecessary allocations, N+1 queries
- Blocking I/O in async contexts
- Memory leaks, unbounded growth
- Missing caching for expensive operations
- Regex recompilation in loops, unindexed queries

**7. Tests**
- Coverage for new functionality (behavior-focused, not line-count obsessed)
- Edge case tests (empty, zero, max, error conditions)
- Test determinism (no timing deps, no random, no network)
- Assertion quality (testing behavior, not implementation details)
- Error handling tested (not just happy path)

**8. Best Practices (from Phase 2 research)**
- Does the implementation follow researched best practices?
- Are known anti-patterns present?
- Is API usage correct per documentation?

**Scope discipline:** Review the diff + 1 dependency hop only. If a finding concerns code outside this scope, classify it as "TRACKED — out of scope" rather than a blocking finding.

**Signal-to-noise control:** Target 2-4 high-value findings per file. Suppress style opinions and micro-optimizations. Every finding must answer: "Does this make the codebase health better or worse?" If not — omit it.

**Finding tiers:**
- **Tier 1 (always report):** Runtime errors, crashes, exploitable vulnerabilities, data loss risks
- **Tier 2 (report when impactful):** Architectural inconsistencies, measurable performance issues, missing error handling
- **Tier 3 (suppress unless egregious):** Style preferences, micro-optimizations, subjective naming

Output: Structured report with MUST_FIX / SHOULD_FIX / CONSIDER / OK findings. Each finding includes an **Impact** statement explaining WHY it matters.

---

### Phase 4 — SECURITY REVIEW (parallel with Phase 3)

Spawn a fresh-context read-only subagent for security audit.

```
Agent(
  description: "Security audit for {story_title}",
  prompt: <see references/review-protocols.md — Security prompt>,
  subagent_type: "agent-explore"
)
```

The security agent follows the `/security-review` protocol with extended coverage:

**Layer 1 — SAST (Source Analysis):**
1. Read all changed files
2. Audit against OWASP Top 10 2025 (includes LLM-specific threats + supply chain risks)
3. Check for injection, auth issues, insecure crypto, data handling
4. AI-generated code anti-patterns (eval, innerHTML, .unwrap() on user input, shell=True)

**Layer 2 — Secrets Detection:**
5. Scan for hardcoded passwords, API keys, tokens, connection strings
6. Check for secrets in config files that will be committed
7. Verify `.gitignore` covers sensitive files (`.env`, credentials, key files)

**Layer 3 — Dependency Scanning (SCA):**
8. Check `Cargo.toml` / `package.json` / `pyproject.toml` for known vulnerable dependencies
9. Flag dependencies with no maintenance (archived repos, no updates in 2+ years)
10. Check for typosquatting risks on new dependencies

**Blocking strategy (tiered):**
- CRITICAL/HIGH: Block — these represent exploitable vulnerabilities
- MEDIUM: Report — create ticket, don't block review
- LOW/INFO: Informational tracking only

Output: Structured security report with CRITICAL/HIGH/MEDIUM/LOW/INFO findings. Each finding includes CWE reference and before/after code remediation.

**Spawn Phase 3 and Phase 4 in a SINGLE message** for true parallel execution.

**GATE:** Both reviews complete.

---

### Phase 5 — REMEDIATE

Fix all issues found in Phases 3 and 4.

**5a. Consolidate and triage findings:**

Merge findings from Phase 2.5 (static analysis), Phase 3 (code review), and Phase 4 (security) into a single prioritized list:

| Priority | Source | Action |
|----------|--------|--------|
| CRITICAL (security) | Phase 4 | Fix immediately |
| HIGH (security) | Phase 4 | Fix immediately |
| MUST_FIX (review) | Phase 3 | Fix immediately |
| Static analysis errors | Phase 2.5 | Fix immediately |
| MEDIUM (security) | Phase 4 | Fix recommended |
| SHOULD_FIX (review) | Phase 3 | Fix recommended |
| Static analysis warnings | Phase 2.5 | Fix recommended |
| LOW/INFO/CONSIDER | Phase 3+4 | Fix if trivial, skip otherwise |

Display the consolidated list to the user before fixing.

**5b. Risk-tiered autonomy — classify each fix before applying:**

| Risk Level | Criteria | Action |
|------------|----------|--------|
| **LOW risk** | Unused imports, formatting, lint fixes, dead code removal, typos | Auto-fix without confirmation |
| **MEDIUM risk** | Logic changes, error handling improvements, missing validation, performance fixes | Fix and show the diff — proceed unless user objects |
| **HIGH risk** | Auth/authorization changes, data access patterns, cryptography, billing logic, API contracts | Present proposed fix, wait for explicit user confirmation |

**5c. Fix loop (max 3 iterations):**

**Iteration 1:** Fix all CRITICAL + HIGH + MUST_FIX + static errors:
1. Classify each fix by risk level (5b)
2. For LOW risk: apply silently
3. For MEDIUM risk: apply and show diff
4. For HIGH risk: present fix, await confirmation
5. After each fix: run the specific test/check that validates it
6. After all fixes: re-run quality gates + static analysis

**Iteration 2 (if needed):** Fix MEDIUM + SHOULD_FIX + static warnings:
1. Same protocol: classify, fix, test, verify
2. Re-run quality gates

**Iteration 3 (if needed):** Address remaining issues or re-fix regressions:
1. If new issues were introduced by fixes → fix them
2. If original issues persist → escalate to user
3. If the fix oscillates (fix A breaks B, fix B breaks A) → stop and escalate

**After each iteration:**
- Run quality gates (PRD-specified + language-specific)
- Run tests to verify no regressions
- Re-run static analysis to verify no new warnings introduced
- Check that previous fixes still hold

**5d. If issues persist after 3 iterations:**

Stop and present remaining issues to the user:
- What was tried
- Why it didn't resolve
- Whether the issue oscillates (two fixes conflict)
- Recommended manual action

**GATE:** Zero CRITICAL/HIGH/MUST_FIX issues remaining. Quality gates pass. Static analysis clean.

---

### Phase 6 — SUMMARY

Produce the final review report. This is the deliverable — no commit or push.

## Output Format

```markdown
## Review Report: {PRD title or Story ID}

### Executive Summary

{2-3 sentences: overall verdict, critical blocking issues if any, key insight.}

**Verdict:** {ALL_CLEAR | PASS_WITH_FIXES | ISSUES_REMAINING}
**Phase results:** Intake {PASS} | Research {PASS} | Static {PASS/n warnings} | Review {PASS/FAIL} | Security {PASS/FAIL} | Remediation {PASS/PARTIAL}

---

**Mode:** {Single Story | Full PRD}
**Files reviewed:** {count} ({total_lines_changed} lines changed)
**Files filtered:** {count} (lock/generated/vendor)
**Stories reviewed:** {count}

### Acceptance Criteria Validation

| Story | Criterion | Status | Evidence |
|-------|-----------|--------|----------|
| US-001 | {criterion_1} | PASS | `file:line` |
| US-001 | {criterion_2} | FAIL | {what's missing} |
| US-001 | {criterion_3} | PARTIAL | {what's implemented, what's not} |
| ... | ... | ... | ... |

### Static Analysis Results

- **Errors fixed:** {count}
- **Warnings fixed:** {count}
- **Remaining:** {count} (with justification)

### Findings Summary

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Code Review | {n} | {n} | {n} | {n} |
| Security | {n} | {n} | {n} | {n} |
| Static Analysis | {n} | {n} | {n} | {n} |
| **Total** | **{n}** | **{n}** | **{n}** | **{n}** |

### Issues Fixed (auto-remediated)

| ID | Severity | Category | Description | Impact | File | Fix Applied |
|----|----------|----------|-------------|--------|------|-------------|
| C-1 | CRITICAL | security | {desc} | {why it matters} | `file:line` | {what was changed} |
| H-1 | HIGH | correctness | {desc} | {why it matters} | `file:line` | {what was changed} |
| ... | ... | ... | ... | ... | ... | ... |

### Issues Requiring Human Review (if any)

| ID | Severity | Category | Description | Impact | Proposed Fix | Why Not Auto-Fixed |
|----|----------|----------|-------------|--------|-------------|-------------------|
| ... | ... | ... | ... | ... | ... | {touches auth / oscillates / novel issue} |

### Quality Gate Results

- {gate_1}: PASS / FAIL
- {gate_2}: PASS / FAIL
- Cognitive complexity: {max value found} (threshold: 10 warn, 20 gate)
- Test coverage for new code: {assessment}

### Research Insights Applied

- {insight_1 from Phase 2 that influenced a fix}
- {insight_2}

### Tracked for Later (out-of-scope findings)

- {finding that affects code outside the diff + 1 hop scope}
- {finding that would require architectural discussion}

### Recommendations

- {recommendation_1 — future improvement, not a blocking issue}
- {recommendation_2}

---
**Changes ready to commit:** {Yes — review `git diff` | No — see issues requiring human review}
```

---

## Full PRD Mode

When no story ID is provided, review the entire PRD:

**Phase 1 adapts:** Extract ALL stories, map ALL changed files.

**Phase 2 adapts:** Research focuses on the feature area of the entire PRD, not a single story.

**Phases 3-4 adapt:** Review agents receive ALL stories and ALL acceptance criteria. They evaluate completeness across the full PRD scope.

**Phase 5 adapts:** Fixes may span multiple stories. Group fixes by story for clarity.

**Phase 6 adapts:** Summary includes per-story status and an overall PRD completion assessment.

---

## Hard Rules

1. Phase 2 (RESEARCH) is MANDATORY — never review without best-practices context.
2. Phase 2.5 (STATIC ANALYSIS) runs BEFORE AI review — catch mechanical issues cheaply first.
3. Phases 3 and 4 run in PARALLEL — spawn both in a single message.
4. Phase 5 has a MAX 3 ITERATIONS — escalate to user after 3 failed fix attempts.
5. Review agents are READ-ONLY — use `agent-explore` (no Edit/Write tools).
6. NEVER commit or push — this workflow stops after correction.
7. Quality gates + static analysis run after EVERY fix iteration in Phase 5.
8. Every finding must have file:line, severity, category, impact statement, and specific remediation.
9. Acceptance criteria are checked explicitly — not assumed to pass. Cite `file:line` evidence.
10. Agent boundaries are strict: websearch does NOT read code, explore does NOT fetch URLs.
11. Compress research output before passing to review agents (<500 words).
12. In full PRD mode, review ALL stories — do not skip stories that "look fine."
13. Pre-filter generated/lock/vendor files — do NOT waste AI tokens reviewing them.
14. Flag PRs > 500 lines — suggest chunking before proceeding.
15. Scope to diff + 1 dependency hop — classify out-of-scope findings as "TRACKED."
16. Target 2-4 high-value findings per file — suppress Tier 3 (style opinions, micro-optimizations).
17. Risk-tier every fix: LOW (auto-fix), MEDIUM (fix + show diff), HIGH (await confirmation).
18. Detect oscillating fixes (A breaks B, B breaks A) — stop and escalate, never spin.
19. For complex remediation decisions involving auth, crypto, or cross-cutting concerns, use ultrathink for deep reasoning.

## Error Handling

- **PRD file not found:** Ask user for the correct path.
- **Story ID not found in PRD:** List available stories, ask user to pick.
- **No changed files detected:** Ask user which files to review, or offer to review the whole project.
- **agent-websearch fails:** Continue with codebase + docs research. Note the gap.
- **agent-explore or agent-docs fails:** Continue with available data. Note the gap.
- **Quality gates fail after all fix iterations:** Report remaining failures in Phase 6 summary.
- **No git repository:** Ask user for file list to review. Skip git-based file detection.

## DO NOT

- Skip Phase 2 (research) — uninformed reviews miss domain-specific issues.
- Skip Phase 2.5 (static analysis) — deterministic checks are cheap, fast, high-precision.
- Run review phases sequentially when they can be parallel.
- Commit or push changes — the user explicitly controls this.
- Fix issues without re-running the specific validation that caught them.
- Modify files during review phases (3 and 4) — reviews are read-only.
- Continue past 3 fix iterations — escalate to the user.
- Report style issues as security findings (or vice versa).
- Assume acceptance criteria pass without explicit verification.
- Skip stories in full PRD mode because they "look fine."
- Invent findings to justify the review — if code is clean, say so.
- Generate 20 low-value findings when 3 high-value findings would be more actionable.
- Review lock files, generated code, or vendor directories with AI tokens.
- Auto-fix HIGH-risk changes (auth, billing, crypto) without user confirmation.
- Produce findings without impact statements — every finding must answer "why does this matter?"
- Report findings outside the diff + 1 hop scope as blocking — classify them as TRACKED.

## Constraints (Three-Tier)

### ALWAYS
- Run research (Phase 2) before review — never skip
- Run static analysis before AI review (Phase 2.5 before Phases 3-4)
- Pre-filter lock files, generated code, vendor directories
- Include impact statements for every finding

### ASK FIRST
- Apply HIGH-risk fixes (auth, billing, crypto, API contracts)
- Proceed when diff exceeds 500 lines (suggest chunking)
- Skip Phase 2 research (only if user explicitly requests)

### NEVER
- Commit or push — this workflow stops after correction
- Modify files during review phases (3 and 4)
- Auto-fix HIGH-risk changes without user confirmation
- Continue past 3 fix iterations — escalate to user
- Generate 20 low-value findings when 3 would be more actionable

## Done When

- [ ] Review scope confirmed with file list and size assessment (Phase 1)
- [ ] Research completed with best-practices brief (Phase 2)
- [ ] Static analysis run and results collected (Phase 2.5)
- [ ] Code review and security audit completed in parallel (Phases 3-4)
- [ ] All CRITICAL/HIGH/MUST_FIX issues remediated (Phase 5)
- [ ] Quality gates pass and static analysis clean
- [ ] Executive summary report produced (Phase 6)
- [ ] No commit or push performed — changes ready for user review

## References

- [Review Protocols](references/review-protocols.md) — exact Agent tool parameters, prompt templates, and expected output formats for each phase
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules, call budgets, authority hierarchy
- [Scope Guard](@~/.claude/skills/_shared/scope-guard.md) — shared threshold definitions and escalation protocol
- [Synthesis Template](@~/.claude/skills/_shared/synthesis-template.md) — standardized format for research synthesis output
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — ALWAYS/ASK FIRST/NEVER model
