---
model: opus
name: implement-story
disable-model-invocation: true
description: "End-to-end workflow to implement a user story from a PRD. Orchestrates 9 phases: intake, research (/meta-code pipeline), planning, implementation, static analysis, code review, security audit, remediation loop, and conventional commit with push. Do NOT auto-trigger — this is a heavyweight pipeline that should only run on explicit invocation. Invoke with /implement-story [prd-path] [story-id]."
argument-hint: "[prd-path] [story-id]"
---

# implement-story — PRD User Story Implementation Pipeline

Implement the following: $ARGUMENTS

## Current State
- Branch: !`git branch --show-current`
- Status: !`git status --short`
- Recent commits: !`git log --oneline -5`

## Overview

End-to-end pipeline that takes a user story from a PRD through research, implementation, static analysis, review, security audit, remediation, and commit. Every phase has a gate — the pipeline only advances when the gate passes.

**Key principles:**
- Research before code — understand fully before writing a single line
- Static analysis before AI review — catch mechanical issues cheaply before spending AI tokens
- Fresh-context reviews — reviewers never see their own code
- Proof-of-fix — every issue is fixed AND verified, not just acknowledged
- Gate-based progression — no skipping phases
- Risk-tiered autonomy — LOW auto-fix, MEDIUM fix+show, HIGH await confirmation

## Execution Flow

```
$ARGUMENTS -> [prd-path] [story-id]
       |
       v
+---------------+
|  Phase 1:     |
|   INTAKE      |  <- Parse PRD, extract user story, confirm with user
+-------+-------+
        | story + acceptance criteria
        v
+---------------+
|  Phase 2:     |
|  RESEARCH     |  <- /meta-code pipeline (websearch -> explore + docs)
|  (mandatory)  |
+-------+-------+
        | research synthesis
        v
+---------------+
|  Phase 3:     |
|    PLAN       |  <- Implementation plan from research + criteria
+-------+-------+
        | approved plan
        v
+---------------+
|  Phase 4:     |
|  IMPLEMENT    |  <- Execute plan, run tests, verify criteria
+-------+-------+
        | all criteria met
        v
+------------------+
|  Phase 4.5:      |
|  STATIC ANALYSIS |  <- Linters, type-checkers, formatters
+--------+---------+
         | mechanical issues resolved
         v
+-------+-------------------+
|         PARALLEL           |
|  +----------+  +----------+|
|  | Phase 5: |  | Phase 6: ||
|  |  REVIEW  |  | SECURITY ||
|  |(quality) |  | (audit)  ||
|  +----+-----+  +----+-----+|
|       |              |      |
+-------+--------------+------+
        v              v
+-------------------------+
|      Phase 7:           |
|     REMEDIATE           |  <- Fix all issues, re-verify
|  (max 3 iterations)    |
+-----------+-------------+
            | zero CRITICAL/HIGH
            v
+-------------------------+
|      Phase 8:           |
|   COMMIT & PUSH         |  <- Conventional commit, push
+-------------------------+
```

## Runtime Output Format

Before each phase, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Phase N/9] PHASE_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Between major phases, print a thin separator: `───────────────────────────────`

## Phase-by-Phase Execution

### Phase 1 — INTAKE

Print: `[Phase 1/9] INTAKE`

Parse the PRD and extract the target user story.

**1a. Parse arguments:**

- If `$ARGUMENTS` contains a file path → read that file as the PRD
- If `$ARGUMENTS` contains a story ID (e.g., `US-001`, `US-1`, `#1`) → target that story
- If arguments are ambiguous → ask the user with AskUserQuestion

**1b. Read and parse the PRD:**

Read the PRD file. Extract:
- **Story title and description** ("As a..., I want..., so that...")
- **Acceptance criteria** (the checklist items under the story)
- **Quality gates** (from the PRD's Quality Gates section, if present)
- **Dependencies** (other stories this depends on, functional requirements)
- **Technical considerations** (from the PRD, if present)

**1c. Confirm with the user:**

Display the extracted story and acceptance criteria. Use AskUserQuestion:

```json
{
  "questions": [{
    "question": "Is this the correct story to implement?",
    "header": "Story Confirmation",
    "options": [
      { "label": "Yes, proceed", "description": "This is the correct story — start research phase" },
      { "label": "Wrong story", "description": "Let me specify a different story ID" },
      { "label": "Adjust criteria", "description": "The story is right but I want to modify acceptance criteria" }
    ]
  }]
}
```

**1d. Pre-filter scope (if reviewing existing implementation):**

When implementing on an existing codebase, pre-filter non-relevant files from scope:
- Lock files: `package-lock.json`, `yarn.lock`, `Cargo.lock`, `pnpm-lock.yaml`
- Generated files: `*.generated.*`, `*.g.dart`, `*.pb.go`, files with `@generated` header
- Vendor directories: `vendor/`, `node_modules/`, `third_party/`
- Build output: `dist/`, `build/`, `target/`, `.next/`

**GATE:** User confirms the story is correct.

---

### Phase 2 — RESEARCH (mandatory /meta-code pipeline)

Print: `[Phase 2/9] RESEARCH`

This phase replicates the `/meta-code` research pipeline. It is MANDATORY — never skip research.

**2pre. Cache check:** Read `~/.claude/projects/*/memory/` for prior research relevant to this story's domain. If fresh (<7 days) reference memories exist, incorporate them and narrow the agent-websearch scope to uncovered areas only.

**2a. Spawn Phase 1 of meta-code (agent-websearch):**

```
Agent(
  description: "Research for {story_title}",
  prompt: <see references/phase-protocols.md — Research prompt template>,
  subagent_type: "agent-websearch"
)
```

Wait for completion. Extract key findings, libraries mentioned, best practices.

**2b. Detect codebase and libraries:**

Run parallel Glob calls for manifest files:
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
  description: "Explore codebase for {story_title}",
  prompt: <see references/phase-protocols.md — Explore prompt template>,
  subagent_type: "agent-explore"
)

// If libraries identified:
Agent(
  description: "Fetch docs for {libraries}",
  prompt: <see references/phase-protocols.md — Docs prompt template>,
  subagent_type: "agent-docs"
)
```

Spawn both in a SINGLE message for true parallel execution.

**2d. Synthesize research:**

Combine all agent outputs following meta-code conflict resolution:
- Official docs > Web research > Codebase patterns
- Deduplicate findings
- Every claim traces to a source

**GATE:** Research synthesis is complete. At least agent-websearch returned results.

---

### Phase 3 — PLAN

Print: `[Phase 3/9] PLAN`

Generate a detailed implementation plan from the research and acceptance criteria.

**3a. Create the plan:**

Based on Phase 2 research + Phase 1 acceptance criteria, produce:

```markdown
## Implementation Plan: {story_title}

### Files to Create/Modify
- `path/to/file.ext` — what changes and why

### Step-by-step
1. {step} — {rationale from research}
2. ...

### Test Strategy
- Unit tests: {what to test}
- Integration tests: {if applicable}

### Quality Gates
- {commands from PRD quality gates section}
- {language-specific checks: cargo clippy, eslint, etc.}

### Risk Areas
- {potential issues identified from research}
```

**3b. Present plan to user:**

Show the plan. The user can approve, modify, or reject.

**GATE:** User approves the plan (or plan is reasonable and no user interaction configured).

---

### Phase 4 — IMPLEMENT

Print: `[Phase 4/9] IMPLEMENT`

Execute the plan step by step.

**4a. Implement each step:**

Follow the plan sequentially. For each step:
1. Read existing files before modifying
2. Make the change using Edit (prefer) or Write (new files)
3. Run quality gates after each logical group of changes

**4b. Run quality gates:**

After implementation is complete:
- Run PRD-specified quality gates (e.g., `pnpm typecheck && pnpm lint`)
- Run language-specific checks (e.g., `cargo check && cargo clippy`)
- Run tests (e.g., `cargo test`, `pnpm test`)

**4c. Verify acceptance criteria:**

Go through each acceptance criterion from Phase 1. For each one:
- Verify it is met by the implementation
- If it requires a test → confirm the test exists and passes
- If it requires visual verification → note it for the user

**GATE:** All quality gates pass. All acceptance criteria are met (or marked for manual verification).

---

### Phase 4.5 — STATIC ANALYSIS (before AI review)

Print: `[Phase 4.5/9] STATIC ANALYSIS`

Run deterministic checks before spending AI tokens on review. These catch mechanical issues cheaply.

**4.5a. Detect and run language-specific tools:**

| Language | Linter/Checker | Command |
|----------|---------------|---------|
| Rust | Clippy + format check | `cargo clippy --all-targets 2>&1; cargo fmt --check 2>&1` |
| TypeScript/JS | ESLint + Prettier | `npx eslint {changed_files} 2>&1; npx prettier --check {changed_files} 2>&1` |
| Python | Ruff (lint + format) | `ruff check {changed_files} 2>&1; ruff format --check {changed_files} 2>&1` |
| Go | vet + staticcheck | `go vet ./... 2>&1; staticcheck ./... 2>&1` |

Only run tools already configured in the project (check for config files). Do NOT install or configure new tools.

**4.5b. Type checking (if configured):**

| Language | Command |
|----------|---------|
| TypeScript | `npx tsc --noEmit 2>&1` |
| Python (mypy) | `mypy {changed_files} 2>&1` |
| Rust | Already checked by `cargo clippy` |

**4.5c. Auto-fix mechanical issues:**

If static analysis finds issues:
1. Fix lint errors and formatting issues automatically
2. Re-run quality gates to confirm fixes
3. Collect remaining warnings for Phase 7 remediation queue

**GATE:** Static analysis complete. Auto-fixable issues resolved. Remaining warnings queued for Phase 7.

---

### Phase 5 — CODE REVIEW (parallel with Phase 6)

Print: `[Phase 5/9] CODE REVIEW`

Spawn a fresh-context read-only subagent for code review.

```
Agent(
  description: "Code review for {story_title}",
  prompt: <see references/phase-protocols.md — Review prompt template>,
  subagent_type: "agent-explore"
)
```

The review agent checks:
1. **Correctness** — logic errors, edge cases, off-by-one, null handling
2. **Quality** — naming, readability, complexity, DRY violations
3. **Performance** — unnecessary allocations, N+1 queries, blocking in async
4. **Tests** — coverage gaps, missing edge cases, brittle assertions
5. **Patterns** — consistency with existing codebase conventions
6. **Acceptance criteria** — does the implementation actually satisfy each criterion?

Output: Structured review report with categorized findings (MUST_FIX, SHOULD_FIX, CONSIDER, OK).

---

### Phase 6 — SECURITY REVIEW (parallel with Phase 5)

Print: `[Phase 6/9] SECURITY REVIEW`

Spawn a fresh-context read-only subagent for security audit.

```
Agent(
  description: "Security audit for {story_title}",
  prompt: <see references/phase-protocols.md — Security prompt template>,
  subagent_type: "agent-explore"
)
```

The security agent follows the `/security-review` protocol:
1. Read all changed files (via `git diff --name-only`)
2. Audit against OWASP Top 10 + AI-generated code anti-patterns
3. Check for secrets, injection, auth issues, insecure crypto

Output: Structured security report with CRITICAL/HIGH/MEDIUM/LOW/INFO findings.

**Spawn Phase 5 and Phase 6 in a SINGLE message** for true parallel execution.

**GATE:** Both reviews complete.

---

### Phase 7 — REMEDIATE

Print: `[Phase 7/9] REMEDIATE`

Fix all issues found in Phase 4.5 (remaining warnings), Phase 5, and Phase 6.

**7a. Triage findings:**

Collect all findings from both reviews. Categorize:

| Priority | Source | Action |
|----------|--------|--------|
| CRITICAL (security) | Phase 6 | Fix immediately |
| HIGH (security) | Phase 6 | Fix immediately |
| MUST_FIX (review) | Phase 5 | Fix immediately |
| MEDIUM (security) | Phase 6 | Fix recommended |
| SHOULD_FIX (review) | Phase 5 | Fix recommended |
| LOW/INFO/CONSIDER | Phase 5+6 | Note but skip unless trivial |

**7b. Risk-tiered autonomy — classify each fix before applying:**

| Risk Level | Criteria | Action |
|------------|----------|--------|
| **LOW risk** | Unused imports, formatting, lint fixes, dead code removal, typos | Auto-fix without confirmation |
| **MEDIUM risk** | Logic changes, error handling improvements, missing validation, performance fixes | Fix and show the diff — proceed unless user objects |
| **HIGH risk** | Auth/authorization changes, data access patterns, cryptography, billing logic, API contracts | Present proposed fix, use AskUserQuestion, wait for explicit user confirmation |

**7c. Fix loop (max 3 iterations):**

For each CRITICAL/HIGH/MUST_FIX issue:
1. Classify risk level per 7b before applying
2. Apply the fix (respecting risk-tier action)
3. Run the specific test or check that validates the fix
4. Mark as resolved

After all fixes:
1. Re-run ALL quality gates
2. Re-run a lightweight verification of the fixed areas

**7d. If issues persist after 3 iterations:**

Stop and present remaining issues to the user with:
- What was tried
- Why it didn't resolve
- Recommended manual action

**GATE:** Zero CRITICAL/HIGH/MUST_FIX issues remaining. All quality gates pass.

---

### Phase 8 — COMMIT & PUSH

Print: `[Phase 8/9] COMMIT & PUSH`

Create a conventional commit and push.

**8a. Determine commit type:**

| Change Type | Prefix |
|-------------|--------|
| New feature / user story implementation | `feat` |
| Bug fix discovered during implementation | `fix` |
| Refactoring without behavior change | `refactor` |
| Tests only | `test` |
| Documentation only | `docs` |

**8b. Determine scope:**

Extract from the story title or primary area of change:
- Module name (e.g., `auth`, `api`, `ui`)
- Feature area (e.g., `cart`, `payment`, `onboarding`)

**8c. Write the commit message:**

Format:
```
type(scope): short description (imperative mood, <72 chars)

Implement {story_id}: {story_title}

Changes:
- {change_1}
- {change_2}
- {change_3}

Acceptance criteria verified:
- [x] {criterion_1}
- [x] {criterion_2}

PRD: {prd_file_path}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**8d. Stage, commit, push:**

```bash
# Stage specific files (never git add -A)
git add path/to/changed/files...

# Commit with conventional message
git commit -m "$(cat <<'EOF'
feat(scope): implement story description

...body...

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"

# Push to current branch
git push
```

**GATE:** Use AskUserQuestion to confirm before pushing:

```json
{
  "questions": [{
    "question": "Ready to commit and push? Review the staged files and commit message above.",
    "header": "Commit & Push",
    "options": [
      { "label": "Commit and push", "description": "Create the commit and push to remote" },
      { "label": "Commit only", "description": "Create the commit but do not push" },
      { "label": "Adjust", "description": "Modify the commit message or staged files before committing" }
    ]
  }]
}
```

---

## Hard Rules

1. Phase 2 (RESEARCH) is MANDATORY — never skip research, even for "simple" stories.
2. Phases 5 and 6 run in PARALLEL — spawn both in a single message.
3. Phase 7 has a MAX 3 ITERATIONS — escalate to user after 3 failed attempts.
4. Phase 8 requires USER CONFIRMATION before push.
5. Review agents are READ-ONLY — use `agent-explore` subagent type (no Edit/Write tools).
6. Quality gates run after Phase 4 AND after Phase 7 — both must pass.
7. Every Phase 5+6 finding that is CRITICAL/HIGH/MUST_FIX must be addressed in Phase 7.
8. Conventional commit format is non-negotiable — `type(scope): description`.
9. Never stage `.env`, credentials, or secret files.
10. Each phase documents its output before the next phase begins.
11. Agent boundaries are strict: websearch does NOT read code, explore does NOT fetch URLs, docs ONLY uses Context7.
12. Compress research output before passing to downstream phases (<500 words).
13. Pre-filter lock files, generated code, and vendor directories before implementation scope analysis.
14. Phase 4.5 (STATIC ANALYSIS) runs AFTER implementation and BEFORE AI review — catch mechanical issues cheaply first.
15. Risk-tier every fix in Phase 7: LOW (auto-fix), MEDIUM (fix + show diff), HIGH (await AskUserQuestion confirmation).
16. NEVER use plain text questions — ALWAYS use AskUserQuestion with selectable options.
17. Print `[Phase N/9]` progress headers before each phase — NEVER skip progress indicators.
18. For architectural decisions and multi-file implementation planning, use ultrathink for deep reasoning.

## Error Handling

- **PRD file not found:** Ask user for the correct path.
- **Story ID not found in PRD:** List available stories, ask user to pick.
- **agent-websearch fails:** Continue with codebase + docs research. Note the gap.
- **agent-explore or agent-docs fails:** Continue with available data. Note the gap.
- **Quality gates fail in Phase 4:** Fix the issues before proceeding to review.
- **All review agents fail:** Perform manual inline review as the orchestrator. Note reduced coverage.
- **Push fails:** Show error, suggest `git pull --rebase` or ask user for guidance.
- **No git repository:** Skip Phase 8 commit/push. Present changes summary instead.

## DO NOT

- Skip Phase 2 (research) for any reason — "I already know how to do this" is not acceptable.
- Start implementing before the plan is generated (Phase 3 before Phase 2).
- Run review phases sequentially when they can be parallel.
- Fix issues without re-running the specific validation that caught them.
- Push without user confirmation.
- Use `git add -A` or `git add .` — always stage specific files.
- Modify files during review phases (5 and 6) — reviews are read-only.
- Continue past 3 remediation iterations — escalate to the user.
- Invent acceptance criteria that aren't in the PRD.
- Downplay security findings to pass the gate faster.
- Use plain text questions like "Shall I proceed?" — ALWAYS use AskUserQuestion with selectable options.
- Auto-fix HIGH-risk changes (auth, billing, crypto) without AskUserQuestion confirmation.
- Review lock files, generated code, or vendor directories with AI tokens.

## Constraints (Three-Tier)

### ALWAYS
- Run research (Phase 2) before implementation — never skip
- Run quality gates after Phase 4 AND Phase 7
- Use conventional commit format: `type(scope): description`
- Compress research output before downstream phases (<500 words)
- Pre-filter lock files, generated code, and vendor directories
- Run static analysis (Phase 4.5) before AI review

### ASK FIRST
- Push to remote (Phase 8 — require explicit confirmation)
- Proceed with ambiguous arguments or unclear story
- Apply HIGH-risk fixes touching auth/billing/crypto

### NEVER
- Skip Phase 2 (research) for any reason
- Stage `.env`, credentials, or secret files
- Use `git add -A` or `git add .`
- Modify files during review phases (5 and 6)
- Continue past 3 remediation iterations
- Auto-fix HIGH-risk changes without AskUserQuestion confirmation
- Use plain text confirmations — always use AskUserQuestion with selectable options

## Done When

All 9 phases completed successfully:

- [ ] Story parsed and confirmed with user (Phase 1)
- [ ] Pre-filter applied to scope (Phase 1)
- [ ] Research completed with synthesis (Phase 2)
- [ ] Implementation plan approved by user (Phase 3)
- [ ] All acceptance criteria met and quality gates pass (Phase 4)
- [ ] Static analysis passed and auto-fixes applied (Phase 4.5)
- [ ] Code review and security audit completed in parallel (Phases 5-6)
- [ ] Zero CRITICAL/HIGH/MUST_FIX issues remaining (Phase 7)
- [ ] Conventional commit created and pushed with user confirmation (Phase 8)

## References

- [Phase Protocols](references/phase-protocols.md) — exact Agent tool parameters, prompt templates, and expected output formats for each phase
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules, call budgets, authority hierarchy
- [Scope Guard](@~/.claude/skills/_shared/scope-guard.md) — shared threshold definitions and escalation protocol
- [Synthesis Template](@~/.claude/skills/_shared/synthesis-template.md) — standardized format for research synthesis output
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — ALWAYS/ASK FIRST/NEVER model
