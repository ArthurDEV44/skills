# Agent Orchestration — Task Parameters, Prompt Templates, and Coordination

## Table of Contents

- [Agent Spawning Protocol](#agent-spawning-protocol)
- [Step 2b: agent-explore Prompt Template](#step-2b-agent-explore-prompt-template)
- [Step 3a: agent-websearch Prompt Template](#step-3a-agent-websearch-prompt-template)
- [Step 3b: agent-docs Prompt Template](#step-3b-agent-docs-prompt-template)
- [Parallel Spawning](#parallel-spawning)
- [Output Processing](#output-processing)
- [Orchestrator Responsibilities](#orchestrator-responsibilities)

---

## Agent Spawning Protocol

All agents are spawned using the `Agent` tool with `subagent_type`. refactor is a pipeline, not a long-lived team.

Each Agent tool call uses these parameters:

```
Agent(
  description: "3-5 word summary",
  prompt: "Detailed instructions with analysis context",
  subagent_type: "agent-type"
)
```

---

## Step 2b: agent-explore Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Analyze architecture for refactoring",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

### Prompt Template

```
Analyze the architecture and code patterns of this codebase to identify systemic refactoring opportunities.

## Context

The user wants to refactor {scope_description}. The codebase uses {language} with {framework}.

## Analysis Focus

{analysis_summary — compressed from Step 2a, <500 words}

## Investigation Tasks

1. **Module dependency graph**: Map how modules depend on each other. Identify tightly coupled modules and circular dependencies. Use imports/requires to trace the graph.

2. **Architectural layers**: Identify the layering strategy (e.g., routes → handlers → services → repositories). Flag any layer violations (e.g., a handler directly querying the database, bypassing the service layer).

3. **Shared state patterns**: Find global state, singletons, shared mutable state, and how data flows between modules. Flag any patterns that make testing or reasoning difficult.

4. **Convention consistency**: Check naming conventions (files, functions, types), project structure patterns, and code organization. Flag inconsistencies across the codebase.

5. **Import patterns**: Identify barrel files that defeat tree-shaking, circular imports, deep relative path imports that could use aliases, and unused re-exports.

6. **Duplication across modules**: Look for structural duplication — similar patterns repeated in different parts of the codebase that suggest a missing abstraction.

## Output Requirements

Return findings with file:line references for every claim. Structure as:

### Dependency Analysis
[Module coupling map. Which modules are tightly coupled? Any circular dependencies?]

### Layer Violations
[Cases where architectural boundaries are crossed. file:line for each.]

### Shared State
[Global state, singletons, shared mutable patterns. file:line for each.]

### Convention Inconsistencies
[Naming, structure, or pattern inconsistencies across the codebase.]

### Import Issues
[Barrel files, circular imports, deep paths, unused re-exports.]

### Structural Duplication
[Similar patterns across modules that could be unified.]

### Refactoring Recommendations
[Prioritized list of systemic improvements. For each: what to change, which files, expected impact.]
```

---

## Step 3a: agent-websearch Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Research {framework} optimization",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

### Prompt Template

```
Research current best practices for optimizing and refactoring {language}/{framework} codebases.

## Context

The user is refactoring a {language} project using {framework}. Key issues identified so far:

{analysis_summary — compressed, <500 words}

## Research Focus

1. **Performance best practices for {framework} in 2025-2026**: What are the current recommended optimization patterns? Any recent framework updates that improve performance?

2. **Common anti-patterns**: What are the most common performance and quality anti-patterns in {framework} projects? How to fix them?

3. **Bundle/build optimization** (if frontend): Current best practices for reducing bundle size, improving tree-shaking, and optimizing build output for {build_tool}.

4. **Migration guides** (if legacy patterns detected): Official migration paths for any deprecated patterns found in the analysis (e.g., class to functional components, CommonJS to ESM).

5. **Lighthouse optimization** (if frontend): Current best practices for achieving 90+ Lighthouse scores with {framework}. Focus on LCP, INP, CLS.

## Output Requirements

Structure your output as:

### Performance Best Practices
[Numbered list with source URLs. Focus on actionable patterns, not general advice.]

### Anti-Patterns to Fix
[Common anti-patterns relevant to the user's stack, with solutions. Source URLs.]

### Build/Bundle Optimization
[If frontend: specific techniques for {build_tool}. Source URLs.]

### Migration Paths
[If legacy patterns found: step-by-step migration guides with links.]

### Lighthouse Tips
[If frontend: specific optimization actions ranked by impact. Source URLs.]

### Sources
[All URLs consulted]
```

---

## Step 3b: agent-docs Prompt Template

### Agent Tool Parameters

```
Agent(
  description: "Check docs for {libraries}",
  prompt: <template below>,
  subagent_type: "agent-docs"
)
```

### Prompt Template

```
Look up official documentation to verify correct API usage and find optimization patterns for the libraries used in this refactoring.

## Context

The user is refactoring a {language}/{framework} project. During analysis, the following API usage patterns were flagged for verification:

{api_usage_questions — specific questions about APIs found in the code, max 3}

## Libraries to Check
{library_names_with_versions}

## Documentation Focus

1. **Correct API patterns**: Verify that the APIs being used are the recommended way to accomplish the task. Are there newer, more performant alternatives?

2. **Performance APIs**: Look up any performance-specific APIs, hooks, or configuration options the library provides (e.g., React.memo, useMemo, virtualization helpers).

3. **Deprecated APIs**: Check if any APIs used in the code are deprecated and find their replacements.

4. **Configuration optimization**: Look up build/runtime configuration options that improve performance (e.g., compiler options, production mode settings).

## Important
- Use the Context7 two-step protocol: resolve-library-id first, then query-docs.
- Maximum 3 Context7 calls total.
- Focus on the specific APIs flagged in the analysis — do not provide general library overviews.

## Output Requirements

Structure as:

### API Verification
[For each flagged API: is the usage correct? Is there a better alternative? Code examples from docs.]

### Performance APIs
[Library-specific performance features the user should leverage. Code examples.]

### Deprecated APIs
[Any deprecated APIs found with their replacements.]

### Configuration
[Build or runtime config changes that improve performance.]

### Documentation Sources
[Context7 library IDs and query topics used]
```

---

## Parallel Spawning

Steps 2b, 3a, and 3b are spawned in a SINGLE message with multiple Agent tool calls:

```
[Message with up to 3 tool calls]:

Agent(
  description: "Analyze architecture for refactoring",
  prompt: <Step 2b prompt>,
  subagent_type: "agent-explore"
)

Agent(
  description: "Research {framework} optimization",
  prompt: <Step 3a prompt>,
  subagent_type: "agent-websearch"
)

Agent(
  description: "Check docs for {libraries}",
  prompt: <Step 3b prompt>,
  subagent_type: "agent-docs"
)
```

If no specific libraries need doc verification, omit the agent-docs call.
If no codebase context beyond the target files exists, omit agent-explore.

---

## Output Processing

### Combining Agent Outputs

After all agents complete, merge findings for Step 4 (Plan):

**Authority hierarchy** (when agents provide conflicting advice):
1. Official docs (agent-docs) — highest authority for API correctness
2. Codebase evidence (agent-explore) — ground truth for current state
3. Web research (agent-websearch) — community best practices and trends

**Deduplication:** If multiple agents identify the same issue, cite the most authoritative source and note cross-agent confirmation.

**Gap reporting:** If any agent was skipped or returned empty, note this in the plan.

### Extracting Actionable Items

From each agent's output, extract:
1. **Issue** — what's wrong or suboptimal
2. **Evidence** — file:line, doc reference, or URL
3. **Fix** — specific action to take
4. **Impact** — expected improvement (performance, maintainability, correctness)
5. **Priority** — P0-P3 based on severity criteria from analysis-checklist.md

---

## Orchestrator Responsibilities

The orchestrator (main Claude session) handles:

1. **Step 1 (Scope)** — detect files, stack, baseline metrics
2. **Step 2a (Self-Analyze)** — read code, catalog issues directly
3. **Spawn Steps 2b + 3** — in parallel, with compressed analysis context
4. **Wait for all agents** — collect outputs
5. **Step 4 (Plan)** — merge findings, rank issues, present plan
6. **Wait for user confirmation** — do not proceed without approval
7. **Step 5 (Execute)** — apply changes one change set at a time, verify after each
8. **Step 6 (Validate)** — run comprehensive checks, measure before/after metrics
9. **Report results** — present final metrics comparison and summary

The orchestrator NEVER duplicates agent work. It does not explore the broader codebase architecture (that's agent-explore), search the web (that's agent-websearch), or query Context7 (that's agent-docs). It reads the target files, analyzes them directly, orchestrates agents, and applies fixes.
