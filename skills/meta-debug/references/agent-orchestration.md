# Agent Orchestration — Task Parameters, Prompt Templates, and Coordination

## Table of Contents

- [Agent Spawning Protocol](#agent-spawning-protocol)
- [Step 3: agent-explore Prompt Template](#step-3-agent-explore-prompt-template)
- [Step 4: agent-docs Prompt Template](#step-4-agent-docs-prompt-template)
- [Step 5: agent-websearch Prompt Template](#step-5-agent-websearch-prompt-template)
- [Parallel Spawning](#parallel-spawning)
- [Output Processing](#output-processing)
- [Orchestrator Responsibilities](#orchestrator-responsibilities)

---

## Agent Spawning Protocol

All agents are spawned using the `Agent` tool — NOT TeamCreate. meta-debug is a pipeline, not a long-lived team.

Each Agent tool call uses these parameters:

```
Agent(
  description: "3-5 word summary",
  prompt: "Detailed instructions with triage report + context",
  subagent_type: "agent-type"
)
```

**No `team_name` or `name` parameters** — these are one-shot subagent calls.

---

## Step 3: agent-explore Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Investigate {error_type} in codebase",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

### Prompt Template

```
Investigate the following error in the codebase to find the root cause.

## Error Context
{triage_report}

## Active Hypotheses to Test
{hypothesis_register — only ACTIVE hypotheses}

## Investigation Tasks

1. **Read the error location**: Read the file(s) and line(s) referenced in the error. Understand what the code is doing at the failure point.

2. **Trace the call chain**: Follow the function calls that lead to the error. If there's a stack trace, read each frame's source location. If there's no stack trace, trace backward from the error location through callers. Use call graph navigation when available (grep for function name usage across the codebase).

3. **Map dependencies**: Identify what types, functions, and modules the failing code depends on. Check if any of these dependencies have incorrect signatures, missing implementations, or version mismatches.

4. **Check for recent changes**: If the error suggests a regression (code that used to work), use git log or git blame on the relevant files to find recent modifications.

5. **Understand architectural context**: Determine what role the failing code plays — is it a handler, middleware, model, utility, test? This context affects the fix strategy.

6. **Evaluate hypotheses**: For each ACTIVE hypothesis, report whether your findings provide evidence FOR or AGAINST it. Be explicit — do not leave hypotheses unaddressed.

## Output Requirements

Return your findings with file:line references for every claim. Structure your output as:

### Error Location Analysis
[What the code does at the failure point, with file:line references]

### Call Chain
[The sequence of calls leading to the error, with file:line for each step]

### Dependencies
[Types, modules, and external crates/packages involved, with versions if visible in manifests]

### Recent Changes
[Any relevant recent modifications, or "No recent changes detected" if stable]

### Hypothesis Evaluation
[For each hypothesis: SUPPORTED (with evidence), REFUTED (with evidence), or INCONCLUSIVE (explain what's missing)]

### Root Cause Assessment
[Your assessment of WHY the error occurs — the actual cause, not the symptom. Include confidence level: HIGH (clear evidence), MEDIUM (strong indicators), LOW (hypothesis based on partial evidence)]

### Suggested Fix Direction
[Brief description of what needs to change to fix the root cause, with specific file:line targets]
```

---

## Step 4: agent-docs Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Check docs for {library}",
  prompt: <template below>,
  subagent_type: "agent-docs"
)
```

### Prompt Template

```
Look up official documentation to verify correct API usage for the code involved in the following error.

## Error Context
{triage_report}

## Libraries to Check
{library_names_with_versions}

## Documentation Focus

1. **API signatures**: Look up the exact function/method signatures involved in the error. Verify parameter types, return types, and generic constraints.

2. **Breaking changes**: Check if the library version in use has any breaking changes, deprecations, or migration notes relevant to the error.

3. **Correct usage patterns**: Find official examples showing the correct way to use the APIs involved in the error. Compare against the error context to spot usage mistakes.

4. **Required setup or configuration**: Check if there are initialization steps, feature flags, or configuration requirements that must be met for the API to work correctly.

## Important
- Use the Context7 two-step protocol: resolve-library-id first, then query-docs.
- Maximum 3 Context7 calls total. Plan queries carefully.
- Focus on the specific APIs mentioned in the error — do not provide general library overviews.
- If a version is specified, prioritize version-specific documentation.

## Output Requirements

Structure your output as:

### API Verification
[Correct signatures for the APIs involved, compared against the error. Note any mismatches.]

### Breaking Changes / Migration Notes
[Any relevant version-specific changes, or "No breaking changes found for this version"]

### Correct Usage Example
[Official code example showing the correct pattern for the user's use case]

### Required Setup
[Any setup, config, or feature flags needed, or "No special setup required"]

### Documentation Assessment
[Does the error match a known documentation pattern? Is the user's code consistent with documented usage?]
```

---

## Step 5: agent-websearch Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Search for {error_message}",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

### Prompt Template

```
Search for solutions to a specific error that could not be resolved through codebase investigation and documentation alone.

## Error Details
{triage_report}

## What Has Already Been Tried
Codebase investigation and documentation lookup did not resolve this error. The following was found:
{summary_of_steps_3_4_findings}

## Remaining Active Hypotheses
{active_hypotheses_from_register}

## Search Strategy

1. **Exact error search**: Search for the exact error message in quotes, combined with the library name and version.
2. **GitHub issues**: Search for this error in the library's GitHub repository issues.
3. **Community solutions**: Search Stack Overflow and developer forums for this error with validated solutions.
4. **Known bugs**: Check if this is a documented bug with a patch, workaround, or version fix.

## Search Queries to Run
- "{exact_error_message}" {library_name}
- {library_name} {error_code_if_any} site:github.com/issues
- {error_message_keywords} {library_name} {version} fix

## Output Requirements

Structure your output as:

### Solutions Found
[Numbered list of solutions, most authoritative first. For each:
- Source (URL)
- Solution description
- Applicability: does it match the user's version and context?
- Confidence: HIGH (official fix/patch), MEDIUM (community-validated), LOW (speculative)]

### Known Bug Status
[Is this a known bug? If so: reported when, fixed in which version, workaround available?
Or: "No known bug reports found for this error."]

### Hypothesis Evaluation
[Which remaining hypotheses are SUPPORTED or REFUTED by web findings?]

### Recommended Fix
[The single best solution from the findings, with justification for why it's the best match.]

### Sources
[All URLs consulted, formatted as markdown links]
```

---

## Parallel Spawning

When both Step 3 and Step 4 are active, spawn them in a SINGLE message with TWO Agent tool calls:

```
[Message with two tool calls]:

Agent(
  description: "Investigate {error_type} in codebase",
  prompt: <Step 3 prompt with triage report>,
  subagent_type: "agent-explore"
)

Agent(
  description: "Check docs for {library}",
  prompt: <Step 4 prompt with triage report>,
  subagent_type: "agent-docs"
)
```

This ensures true parallel execution. Both agents work simultaneously and the orchestrator waits for both to complete before evaluating the decision gate.

If only one step is applicable (e.g., codebase exists but no library identified), spawn only that one.

---

## Output Processing

### Combining Agent Outputs

After all agents complete, merge their findings for Step 6:

**Evidence hierarchy** (when agents provide conflicting information):
1. Official docs (Step 4) — highest authority for API correctness
2. Codebase evidence (Step 3) — ground truth for current state
3. Web research (Step 5) — community validation and workarounds

**Hypothesis register update:** After each agent completes, update the hypothesis register:
- Mark hypotheses as CONFIRMED, ELIMINATED, or still ACTIVE based on agent evidence
- A hypothesis is CONFIRMED only when supported by evidence from at least one agent and not contradicted by any
- A hypothesis is ELIMINATED when evidence from any agent contradicts it

**Deduplication:** If Steps 3 and 4 identify the same root cause, cite the more authoritative source and note confirmation from the other.

**Gap reporting:** If any step was skipped or returned empty, note this in the diagnosis. Example: "Documentation check was skipped (no library identified in the error)."

### Extracting the Root Cause

From the combined output, extract:
1. **Root cause statement** — one sentence explaining WHY the error occurs
2. **Evidence chain** — file:line references, doc references, or URLs that support the statement
3. **Fix target** — the specific file(s) and line(s) where the fix should be applied
4. **Confidence level** — HIGH, MEDIUM, or LOW based on the strength of evidence
5. **Hypothesis register** — final state showing which hypotheses were confirmed/eliminated

If confidence is LOW, present the diagnosis as a hypothesis and ask the user targeted discriminating questions — questions designed to eliminate the maximum number of remaining hypotheses with a single answer.

---

## Orchestrator Responsibilities

The orchestrator (main Claude session) handles:

1. **Reproduce the error** — run the failing command/test to confirm reproducibility (Step 1)
2. **Parse error input** — extract the error message, stack trace, or description from user input
3. **Execute Step 2 (triage)** — classify, extract signals, build hypothesis register, detect regressions
4. **Run git bisect** — if regression detected, find the causal commit before LLM analysis
5. **Detect codebase** — parallel Glob for manifest files
6. **Identify libraries** — from error text, import paths, and manifest files
7. **Spawn Steps 3 and/or 4** — in parallel when both applicable
8. **Wait for all active agents** — collect outputs
9. **Update hypothesis register** — integrate agent findings
10. **Evaluate decision gate** — is the root cause clear? If not, spawn Step 5 or ask user
11. **Monitor trajectory smells** — detect NO_TEST, NO_OP_CAT, CONSECUTIVE_SEARCH, CONSECUTIVE_EDITS
12. **Execute Step 6 (fix)** — apply minimal fix with explicit justification per line
13. **Execute Step 7 (verify)** — re-run reproduction command, apply DDI circuit breaker if needed
14. **Codify regression test** — suggest a test that guards against the specific bug
15. **Report results** — present diagnosis, fix, verification, and prevention to user

The orchestrator NEVER duplicates agent work. It does not explore the codebase, query Context7, or search the web itself. It only orchestrates, synthesizes, and applies fixes.
