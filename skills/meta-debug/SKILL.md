---
model: opus
name: meta-debug
description: "Multi-agent debugging workflow that diagnoses and fixes errors with surgical precision. Orchestrates agent-explore (root cause), agent-docs (API verification), and agent-websearch (community solutions) in a 7-step pipeline: reproduce, triage, investigate, verify docs, research (conditional), fix, verify+regress. Includes DDI circuit breaker, git bisect for regressions, trajectory smell detection, and hypothesis register. Use when the user pastes an error message, stack trace, compiler output, or failing test, or says 'debug', 'fix this error', 'why is this failing', 'help me fix', 'what went wrong', 'diagnose this', or describes unexpected behavior with error context. Do NOT trigger for general code questions, feature requests, or refactoring without an error."
argument-hint: "[error message or description]"
---

# meta-debug — Multi-Agent Debugging Pipeline

## Overview

meta-debug is a 7-step debugging pipeline that diagnoses errors and applies fixes by combining:
1. **Reproduce** (orchestrator) — confirm the error is reproducible before investigating
2. **Error triage** (orchestrator) — classify, extract signals, build hypothesis register
3. **Codebase investigation** (agent-explore) — trace root cause, map dependency chain
4. **Documentation check** (agent-docs) — verify API usage, check for breaking changes
5. **Web research** (agent-websearch) — community solutions, known issues (conditional)
6. **Fix implementation** (orchestrator) — apply minimal fix with explicit minimality constraints
7. **Verify & regress** (orchestrator) — run verification, codify regression test

Steps 3 and 4 run in parallel. Step 5 runs only if Steps 3-4 don't resolve the issue.

Cross-cutting concerns: **DDI circuit breaker** (max 2 fix attempts before strategy change), **trajectory smell detection** (real-time anti-pattern monitoring), **hypothesis register** (externalized diagnostic state).

## Execution Flow

```
Error Input (message, stack trace, test output)
     │
     ▼
┌─────────────┐
│  Step 1:    │
│  REPRODUCE  │  ← Orchestrator: confirm error is reproducible
│  (fast)     │
└──────┬──────┘
       │ reproducible? YES → continue / NO → ask user
       ▼
┌─────────────┐
│  Step 2:    │
│   TRIAGE    │  ← Orchestrator: classify, hypothesis register
│  (instant)  │
└──────┬──────┘
       │ regression detected?
       ├── YES → git bisect shortcut ──┐
       │                               │
       │ error classification          │
       ▼                               ▼
┌──────┴──────────────────┐    ┌──────────────┐
│         PARALLEL         │    │  GIT BISECT  │
│  ┌──────────┐  ┌──────────┐  │  (automatic) │
│  │ Step 3:  │  │ Step 4:  │  └──────┬───────┘
│  │INVESTIGATE│  │VERIFY    │         │
│  │(codebase)│  │(docs)    │         │
│  └────┬─────┘  └────┬─────┘         │
│       │              │     │         │
└───────┼──────────────┼─────┘         │
        ▼              ▼               │
   ┌─────────────────────────┐         │
   │  Root cause identified? │◄────────┘
   │  YES → Step 6           │
   │  NO  → Step 5           │
   │  LOW confidence → ask   │
   └─────────────────────────┘
        │         │        │
        ▼         ▼        ▼
  ┌──────────┐ ┌──────┐ ┌──────────┐
  │ Step 5:  │ │ ASK  │ │ Step 6:  │
  │ RESEARCH │ │ USER │ │ FIX      │
  │(web,cond)│ │      │ │(minimal) │
  └────┬─────┘ └──────┘ └────┬─────┘
       │                      │
       └──────────┬───────────┘
                  ▼
          ┌──────────────┐
          │  Step 7:     │
          │  VERIFY &    │
          │  REGRESS     │
          └──────────────┘
               │
          DDI circuit breaker:
          fix failed? attempt < 2?
          YES → retry Step 6 with new strategy
          NO  → escalate to user
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

### Step 1 — Reproduce (Orchestrator, Fast)

Print: `[Step 1/7] REPRODUCE`

**Reproduce the error before investigating.** A fix that cannot be preceded by a deterministic failing reproduction is not trustworthy.

**1a. Determine the reproduction command:**

| Error source | Reproduction command |
|---|---|
| Compiler output | Re-run the build command (`cargo check`, `tsc --noEmit`, `go build ./...`) |
| Test failure | Re-run the specific failing test (`cargo test test_name`, `npx vitest run path`, `pytest path::test`) |
| Runtime error | Run the failing command/script as described by user |
| Linter/clippy | Re-run the linter (`cargo clippy`, `eslint`, `mypy`) |

**1b. Execute and confirm:**

- Run the reproduction command via Bash
- Confirm the same error appears in the output
- If the error does NOT reproduce: inform the user, ask for additional context (environment, state, timing), do NOT proceed to Step 2
- If the error reproduces: capture the exact output for the triage report, proceed to Step 2

**1c. Skip conditions** (reproduce step not needed):

- User pasted a complete error output and the error is clearly from a deterministic source (compiler, type checker)
- The error is visible in the code itself (syntax error, obvious type mismatch)
- The user explicitly says "I just ran this and got..."

### Step 2 — Error Triage (Orchestrator, Instant)

Print: `[Step 2/7] TRIAGE`

Parse the error and classify it. This step uses NO agents — the orchestrator handles it directly.

**2a. Extract signals from the error:**

| Signal | What to look for |
|--------|-----------------|
| File paths | Absolute or relative paths in the error output |
| Line numbers | `:42`, `line 42`, `at line 42` |
| Error codes | `E0308`, `TS2322`, `ENOENT`, HTTP status codes |
| Library names | Package names in import paths or stack frames |
| Versions | Version strings near library names in lock files or error text |
| Function names | Top of stack trace, `at function_name` |

**2b. Classify the error type:**

| Classification | Indicators |
|---------------|------------|
| **Compile error** | Compiler output, syntax errors, type mismatches, missing imports |
| **Type error** | Type checker output, generic constraint failures, inference failures |
| **Runtime error** | Panic, exception, segfault, stack trace with runtime frames |
| **Dependency issue** | Resolution failures, version conflicts, missing packages |
| **Config issue** | Environment variables, config file parsing, path errors |
| **Logic bug** | Wrong output, assertion failure, test mismatch (expected vs got) |
| **Performance issue** | Timeout, OOM, slow query, high CPU |
| **Test failure** | Test framework output, assertion messages, diff output |

**2c. Build the Hypothesis Register:**

Initialize a structured diagnostic state (maintained throughout the pipeline):

```
## Hypothesis Register
| # | Hypothesis | Status | Evidence For | Evidence Against |
|---|-----------|--------|-------------|-----------------|
| 1 | {hypothesis_1} | ACTIVE | | |
| 2 | {hypothesis_2} | ACTIVE | | |
| 3 | {hypothesis_3} | ACTIVE | | |
```

Generate 2-5 initial hypotheses based on the error classification and signals. Each hypothesis should be specific enough that a single test or observation could eliminate it.

**2d. Detect regression signal:**

Check for indicators that this is a regression (code that used to work):
- User says "this was working before", "it broke after", "regression"
- Error in code that hasn't been recently modified (check `git log --oneline -5 {file}`)
- Test that was previously passing

If regression detected → trigger **git bisect shortcut** (see below).

**2e. Prepare the triage report** (used internally for Steps 3-4):

```
## Triage Report
- Classification: {error_type}
- Error message: {core_message}
- Files involved: {file_paths_with_lines}
- Error code: {code_if_any}
- Libraries: {library_names_with_versions}
- Stack trace depth: {number_of_frames}
- Regression: {yes/no}
- Hypotheses: {numbered_list}
```

**2f. Detect codebase and libraries:**

Run parallel Glob calls for manifest files:
```
Glob: Cargo.toml
Glob: package.json
Glob: pyproject.toml
Glob: go.mod
```

If any manifest found → Step 3 is active.
If libraries identified in triage → Step 4 is active.

### Git Bisect Shortcut (Conditional, Regression Only)

When a regression is detected in Step 2d, run `git bisect` to find the causal commit before doing full LLM analysis. This is O(log n) and dramatically reduces the search space.

**Protocol:**
1. Ask the user for the last known good commit (or use `git log` to estimate)
2. Run `git bisect start HEAD {good_commit}`
3. Use the reproduction command from Step 1 as the bisect oracle: `git bisect run {command}`
4. The output identifies the exact commit that introduced the bug
5. Read the commit diff to understand what changed
6. Feed the causal commit info into the hypothesis register as strong evidence
7. Proceed to the decision gate with this evidence

**Skip bisect if:**
- No clear good/bad boundary exists
- The repo has fewer than 10 commits
- The error is not in user code (dependency/config issue)

### Step 3 — Codebase Investigation (agent-explore)

Print: `[Step 3-4/7] INVESTIGATE + VERIFY DOCS (parallel)`

Spawn agent-explore to trace the error to its root cause:

```
Agent(
  description: "Investigate {error_type} in codebase",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-explore"
)
```

The agent investigates:
- Read the files and lines referenced in the error
- Trace the call chain that leads to the failure point (use call graph when available)
- Map dependencies of the failing code (imports, types, modules)
- Check for recent changes near the error (git blame/log if useful)
- Identify the architectural context — is it a handler, middleware, model, test?
- Report confidence level: HIGH / MEDIUM / LOW

### Step 4 — Documentation Check (agent-docs)

Spawn agent-docs to verify API usage against official documentation:

```
Agent(
  description: "Check docs for {library}",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-docs"
)
```

The agent checks:
- Correct API signatures for functions involved in the error
- Known breaking changes or migration notes for the library version
- Required configuration or setup steps that may have been missed
- Deprecation warnings for any APIs used in the failing code

**Spawn Steps 3 and 4 in a SINGLE message** for true parallel execution.

### Decision Gate — Is the Root Cause Clear?

After Steps 3-4 complete, update the hypothesis register with evidence from both agents. Then evaluate:

**Update hypothesis register:**
```
| # | Hypothesis | Status | Evidence For | Evidence Against |
|---|-----------|--------|-------------|-----------------|
| 1 | {hypothesis} | CONFIRMED / ELIMINATED / ACTIVE | {from agent-explore/docs} | {contradictions} |
```

**Decision criteria:**

- **Root cause identified (HIGH confidence)** → proceed to Step 6.
  - agent-explore found the exact code causing the issue AND
  - agent-docs confirmed the correct API usage (or the error is not API-related)
  - Only ONE hypothesis remains ACTIVE, all others ELIMINATED

- **Root cause unclear (MEDIUM confidence)** → proceed to Step 5.
  - The error doesn't match any obvious code issue
  - The API usage appears correct but the error persists
  - The error message is cryptic or underdocumented
  - A known bug or platform-specific issue is suspected
  - Multiple hypotheses still ACTIVE

- **Root cause unclear (LOW confidence)** → ask the user for more context.
  - No hypothesis has strong supporting evidence
  - The error may be environment-specific, timing-dependent, or involve state not visible in the codebase
  - Present what was found so far and ask targeted discriminating questions (questions that eliminate the maximum number of hypotheses)

### Step 5 — Web Research (agent-websearch, Conditional)

Print: `[Step 5/7] RESEARCH (conditional)`

ONLY run this step if Steps 3-4 did not resolve the issue.

```
Agent(
  description: "Search for {error_message}",
  prompt: <see references/agent-orchestration.md for template>,
  subagent_type: "agent-websearch"
)
```

The agent searches for:
- The exact error message + library name + version
- GitHub issues mentioning this error
- Stack Overflow solutions with high vote counts
- Blog posts or changelogs that document this specific issue
- Whether a patch or workaround exists

### Step 6 — Fix Implementation (Orchestrator)

Print: `[Step 6/7] FIX`

Apply the fix based on all gathered evidence. Follow this protocol:

**6a. Present the diagnosis:**

```markdown
## Root Cause
[1-3 sentences explaining WHY the error occurred — the actual cause, not the symptom]

## Evidence
- [file:line — what was found there]
- [Documentation reference — what the correct behavior should be]
- [Web source — if Step 5 was needed]

## Hypothesis Register (Final)
| # | Hypothesis | Status | Key Evidence |
|---|-----------|--------|-------------|
| 1 | {winning hypothesis} | CONFIRMED | {evidence} |
| 2 | {eliminated} | ELIMINATED | {why} |
```

**6b. Choose the fix strategy:**

If multiple fixes exist, rank by:
1. **Correctness** — does it fix the actual root cause, not just the symptom?
2. **Safety** — minimal side effects, no regressions, preserves existing behavior
3. **Simplicity** — least invasive change, fewest files modified

Present the top strategy. If there are meaningful alternatives, list them briefly with trade-offs.

For fix strategy ranking criteria and anti-patterns, see `references/fix-strategies.md`.

**6c. Apply the fix with minimality constraints:**

- Use Edit tool for surgical changes — do not rewrite files unnecessarily
- Change ONLY what is necessary to fix the root cause
- Preserve existing code style and conventions
- **Minimality rule**: every modified line must have a justification traced to the root cause. If a line change cannot be justified, do not make it
- **No drive-by fixes**: do not fix unrelated issues, improve style, or add comments to surrounding code

**6d. Record the fix attempt** (for DDI circuit breaker):

```
Fix attempt #{n}: {one-line description of what was changed and why}
```

### Step 7 — Verify & Regress (Orchestrator)

Print: `[Step 7/7] VERIFY & REGRESS`

**7a. Verify the fix:**

- If the error came from a compiler: re-run the build
- If the error came from a test: re-run the failing test
- If the error came from runtime: explain how to verify manually
- If linting or type checking is available: run it

**7b. DDI Circuit Breaker** (Debugging Decay Index):

If verification fails (the fix didn't work):
- **Attempt 1 failed** → try ONE alternative fix strategy (from 6b ranking). Do NOT iterate on the same approach.
- **Attempt 2 failed** → STOP. Present what was tried, what failed, and escalate to the user with:
  - The confirmed root cause (or best hypothesis)
  - The two fix strategies that were attempted and why they failed
  - A suggested next step that requires human judgment

Research shows LLMs lose 60-80% of debugging capability within 2-3 attempts (DDI exponential decay). A strategic fresh start or human input is more effective than continued iteration.

**7c. Codify regression test:**

After the fix is verified, suggest a regression test if one doesn't exist:
- A test that would FAIL without the fix and PASS with it
- Cover the specific boundary condition or input that triggered the bug
- One sentence explaining what the test guards against

Only suggest this if:
- The error came from a logic bug, runtime error, or test failure
- A test file already exists for the affected module
- The test is genuinely useful (not trivial or redundant)

**7d. Suggest prevention:**

One sentence on how to avoid this error in the future (only if there's a meaningful preventive measure — don't add noise).

## Trajectory Smell Detection (Cross-Cutting)

Monitor the debugging session for anti-patterns that predict failure (source: Google Passerine study, 66-67% failure correlation). If any smell is detected, immediately change strategy.

| Smell | Detection | Action |
|---|---|---|
| **NO_TEST** | Agent never ran any test or build command after making a fix | STOP — run verification immediately before proceeding |
| **NO_OP_CAT** | Re-reading files that were already read with no edits in between | STOP — you're going in circles. Formulate a new hypothesis |
| **CONSECUTIVE_SEARCH** | 3+ sequential searches (Grep/Glob/Read) without editing | STOP — you have enough information. Decide and act |
| **CONSECUTIVE_EDITS** | Repeated edits to the same file without testing | STOP — run the test before making more changes |

These smells should trigger self-correction, not pipeline abort. The orchestrator should recognize the pattern and adjust.

## Hard Rules

1. Step 1 (reproduce) runs FIRST — no investigation without confirmed reproduction (unless skip conditions met).
2. Step 2 (triage) is ALWAYS done by the orchestrator — no agent spawning for triage.
3. Steps 3 and 4 run in PARALLEL — spawn both in a single message.
4. Step 5 is CONDITIONAL — only run when Steps 3-4 don't resolve the issue.
5. Step 6 runs AFTER all investigation is complete — never fix before understanding.
6. Step 7 ALWAYS verifies — never claim a fix without running the reproduction command.
7. DDI circuit breaker is MANDATORY — max 2 fix attempts, then escalate to user.
8. Agent boundaries are strict — explore reads code, docs queries Context7, websearch fetches URLs.
9. Max 3 Context7 calls — agent-docs must stay within the hard limit.
10. Compress triage report before passing to agents (<300 words).
11. Every diagnosis claim must trace to evidence (file:line, doc reference, or URL).
12. Graceful degradation — if any agent fails, continue with available data and note the gap.
13. Do NOT use TeamCreate — use simple Agent tool spawning for all agents.
14. Maintain the hypothesis register throughout — update after every step, present in final diagnosis.
15. Prefer discriminating observations — each diagnostic action should eliminate the maximum number of hypotheses.
16. Print `[Step N/7]` progress headers before each step — NEVER skip progress indicators.
17. For complex root cause analysis and multi-file investigations, use ultrathink for deep reasoning.

## Error Handling

- agent-explore returns empty: the error may be in generated/external code — note this, proceed with Steps 4-5.
- agent-docs returns empty: the library may lack Context7 coverage — note this, rely on web research.
- agent-websearch returns empty: the error may be novel — apply best-effort diagnosis from Steps 3-4.
- All agents fail: use the triage report and error message to provide the best guidance possible with an honest disclaimer.
- No codebase detected: skip Step 3, rely on Steps 4-5 and the error message itself.
- No library identified: skip Step 4, rely on Steps 3 and 5.
- Reproduction fails (Step 1): the error may be non-deterministic — ask the user about environment, timing, state. Do NOT proceed without reproduction or user confirmation to skip.
- Git bisect fails: the repo may have merge commits or non-linear history — skip bisect, proceed with normal investigation.

## DO NOT

- Skip reproduction and jump to investigation — reproduce first, investigate second.
- Skip triage and jump straight to web searching — triage prevents wasted effort.
- Apply a fix before completing investigation — understand first, fix second.
- Suggest "just Google it" — every step must add diagnostic value.
- Spawn all agents simultaneously — Steps 1-2 must complete first to inform Steps 3-4.
- Run Step 5 unconditionally — web research is the fallback, not the default.
- Include unsourced fix suggestions — every fix traces to evidence.
- Over-fix — change only what is necessary, do not refactor surrounding code. Every modified line must be justified.
- Hardcode language-specific patterns in the pipeline — use adaptive detection from triage signals.
- Iterate more than 2 fix attempts — the DDI circuit breaker is mandatory. Escalate to user.
- Re-read files without making edits — this is a trajectory smell (NO_OP_CAT). Decide and act.
- Search 3+ times without editing — this is a trajectory smell (CONSECUTIVE_SEARCH). You have enough info.

## Constraints (Three-Tier)

### ALWAYS
- Reproduce the error before investigating (unless skip conditions met)
- Maintain the hypothesis register throughout — update after every step
- Run verification after every fix attempt
- Compress triage report before passing to agents (<300 words)

### ASK FIRST
- Proceed with LOW confidence root cause (ask user for more context)
- Apply fix when multiple equally viable strategies exist

### NEVER
- Fix before understanding — investigate first, fix second
- Iterate more than 2 fix attempts (DDI circuit breaker)
- Re-read files without making edits (trajectory smell: NO_OP_CAT)
- Over-fix — change only what is necessary, no drive-by fixes

## Done When

- [ ] Error reproduced (or skip conditions documented)
- [ ] Triage report produced with hypothesis register
- [ ] Root cause identified with HIGH confidence (or escalated to user)
- [ ] Fix applied with minimality constraints (every modified line justified)
- [ ] Verification passed — reproduction command now succeeds
- [ ] Hypothesis register finalized (CONFIRMED/ELIMINATED for all hypotheses)
- [ ] Regression test suggested (if applicable)

## References

- [Error Patterns](references/error-patterns.md) — common error pattern taxonomy, diagnosis heuristics per classification, and language-adaptive signal extraction
- [Fix Strategies](references/fix-strategies.md) — fix strategy ranking criteria, common fix anti-patterns, and verification protocols
- [Agent Orchestration](references/agent-orchestration.md) — exact Agent tool parameters, prompt templates, and coordination rules for all three agents
- [Research Sources](references/research-sources.md) — academic papers and industry sources backing the pipeline design
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules, call budgets, authority hierarchy
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — ALWAYS/ASK FIRST/NEVER model
