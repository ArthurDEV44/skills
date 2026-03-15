# Agent Protocols — How Each Agent Is Invoked

## Agent Spawning via Agent Tool

All agents are spawned using the `Agent` tool — NOT using TeamCreate or agent teams. meta-code is a defined pipeline, not a long-lived team collaboration.

Each Agent tool call uses these parameters:

```
Agent(
  description: "3-5 word summary",
  prompt: "Detailed instructions with user question + prior context",
  subagent_type: "agent-type"
)
```

**No `team_name` or `name` parameters** — these are one-shot subagent calls, not team members.

---

## Step 2: agent-websearch (RESEARCH)

### Agent Tool Parameters

```
Agent(
  description: "Web research on {topic}",
  prompt: <see template below>,
  subagent_type: "agent-websearch"
)
```

### Prompt Template

```
Research the following development question thoroughly using web search.

## Question
{user_question}

## Sub-questions (if decomposed)
{sub_questions_targeting_websearch — only for complex queries}

## Search Strategy
- Search for current best practices, recent articles, and official guidance related to this question.
- If the question involves specific libraries or frameworks, search for their latest documentation and changelog.
- If the question involves a technique or pattern, search for real-world examples and community consensus.
- Use 2-3 complementary searches to cover different angles.
- Include the current year (2026) in searches for time-sensitive topics.

## Output Requirements
Return your findings in this exact structure:

### Key Findings
[Numbered list of 3-8 key findings, most important first]

### Libraries & Frameworks Mentioned
[List any specific libraries, frameworks, or tools that are relevant to answering the question. Include version numbers if found. Format: "- library-name (vX.Y.Z if known)"]

### Best Practices
[Specific, actionable recommendations from authoritative sources]

### Contradictions Detected
[List any cases where different sources disagree on a claim. Format:
- Claim: "{the claim}"
  - Source A ({url}): says X
  - Source B ({url}): says Y
If no contradictions found, write "None detected."]

### Sources
[All URLs consulted, formatted as markdown links]
```

### Expected Output Structure

The orchestrator extracts from Step 2 output:
- `key_findings`: All content under "Key Findings" header
- `libraries`: Parse library names from "Libraries & Frameworks Mentioned" section
- `best_practices`: Content under "Best Practices" header
- `contradictions`: Parsed from "Contradictions Detected" section
- `sources`: URLs from "Sources" section

### Compressed Summary for Downstream Steps

Before passing to Steps 3/4, compress to:

```
## Research Context (for downstream agents)

Key findings on "{user_question}":
1. {finding_1}
2. {finding_2}
...

Relevant libraries: {lib1} {version}, {lib2} {version}

Contradictions to investigate: {contradiction_1_summary}
```

Target: <500 words. Strip URLs and detailed explanations. Keep only facts, library identifiers, and contradiction summaries.

---

## Step 3: agent-explore (EXPLORE)

### Codebase Detection (run by orchestrator BEFORE spawning)

Use Glob to check for project manifest files:

```
Glob: Cargo.toml
Glob: package.json
Glob: pyproject.toml
Glob: go.mod
```

Run these 4 Glob calls in parallel. If ANY returns a result, `codebase_exists = true`.

For additional detection (if all 4 return empty), check:
```
Glob: pom.xml
Glob: build.gradle
Glob: *.sln
Glob: Makefile
Glob: .git
```

### Agent Tool Parameters

```
Agent(
  description: "Explore codebase for {topic}",
  prompt: <see template below>,
  subagent_type: "agent-explore"
)
```

### Prompt Template

```
Explore the codebase in the current working directory to find code, patterns, and architecture relevant to the following question.

## Question
{user_question}

## Research Context
The following was found via web research (use this to guide what you look for):

{step_2_compressed_summary}

## Exploration Focus
1. Find existing code that relates to the question — functions, types, modules, handlers.
2. Identify patterns and conventions the project uses that would affect how to implement or approach this.
3. Check if there's already an implementation of what's being asked about (partial or complete).
4. Note the project's architecture, relevant dependencies, and module structure.
5. Focus your exploration on areas most relevant to the question — don't do a full project scan.

## Contradiction Check
If any of the web research recommendations conflict with how the codebase actually does things, note this explicitly:
- What the web recommends vs. what the codebase does
- Whether the codebase deviation appears intentional (custom wrapper, legacy, etc.)

## Output Requirements
Return findings with file:line references for every claim. Use the most appropriate exploration mode based on what the question requires.

At the end of your findings, include a section:
### Contradictions with Web Research
[List any conflicts between web research recommendations and actual codebase patterns. If none, write "None detected."]
```

### Expected Output

agent-explore returns structured findings with file:line references following its internal output format. The orchestrator uses this output directly in Step 5 synthesis, including the contradictions section.

---

## Step 4: agent-docs (DOCUMENT)

### Library Extraction (run by orchestrator AFTER Step 2)

Parse the Step 2 output to extract library names:

1. Read the "Libraries & Frameworks Mentioned" section from Step 2 output.
2. If Step 3 is also running (codebase exists), read the manifest file (Cargo.toml, package.json, etc.) to get exact version numbers.
3. Select the top 1-2 libraries most relevant to the user's question.
4. If no libraries identified from either source, skip Step 4.

### Agent Tool Parameters

```
Agent(
  description: "Fetch docs for {library}",
  prompt: <see template below>,
  subagent_type: "agent-docs"
)
```

### Prompt Template

```
Look up official documentation for the following libraries to answer a specific development question.

## Question
{user_question}

## Libraries to Look Up
{library_list_with_versions}

## Context
This documentation lookup is part of a comprehensive research workflow. Web research has already found general information. Your job is to provide authoritative, version-accurate API details and code examples that web research cannot reliably provide.

Focus on:
1. Exact API signatures and types relevant to the question
2. Official code examples for the specific use case
3. Version-specific behavior, deprecations, or migration notes
4. Configuration or setup requirements

## Contradiction Check
Web research made the following claims about these libraries:
{relevant_web_research_claims}

If any of these claims conflict with official documentation, note the discrepancy explicitly.

## Important
- Use the Context7 two-step protocol: resolve-library-id first, then query-docs.
- Maximum 3 Context7 calls total. Plan queries carefully.
- If a version is specified, use the versioned library ID if available.
- Do NOT repeat general information that web search would cover. Focus on precise API details and examples.

## Output Requirements
Return findings in this structure:

### Answer
[Direct answer to the question based on official docs]

### Code Examples
[Runnable snippets from official docs]

### Key API Details
[Signatures, types, parameters]

### Version Notes
[Version-specific behavior, deprecations, migration notes. If none, write "N/A."]

### Contradictions with Web Research
[List any conflicts between official docs and web research claims. If none, write "None detected."]

### Sources
[Context7 library IDs and any URLs]
```

### Expected Output

agent-docs returns structured documentation with the sections listed above. The orchestrator uses this output directly in Step 5 synthesis, including the contradictions section.

---

## Refinement Agent Prompts (Step 7)

When Step 6 (VERIFY) identifies gaps, refinement agents are spawned with focused prompts. These are narrower than the original prompts — they target specific gaps only.

### Refinement: agent-websearch

```
Agent(
  description: "Refine: {gap_description_short}",
  prompt: "A prior web research pass on '{user_question}' identified the following specific gap:

## Gap
{gap_description}

## What Was Already Found
{summary_of_existing_findings_on_this_topic}

## Task
Search specifically for information that fills this gap. Do NOT repeat broad research — focus narrowly on:
{target_query}

Return findings in the same format as before (Key Findings, Sources, Contradictions Detected).",
  subagent_type: "agent-websearch"
)
```

### Refinement: agent-explore

```
Agent(
  description: "Refine: {gap_description_short}",
  prompt: "A prior codebase exploration on '{user_question}' missed the following:

## Gap
{gap_description}

## Task
Look specifically for: {target_query}
Focus your exploration narrowly on this gap. Return findings with file:line references.",
  subagent_type: "agent-explore"
)
```

### Refinement: agent-docs

```
Agent(
  description: "Refine: {gap_description_short}",
  prompt: "A prior documentation lookup on '{user_question}' missed the following:

## Gap
{gap_description}

## Task
Look up specifically: {target_query}
Use remaining Context7 calls to fill this gap. Return findings with sources.",
  subagent_type: "agent-docs"
)
```

---

## Parallel Spawning

When both Step 3 and Step 4 are active, spawn them in a SINGLE message with TWO Agent tool calls:

```
[Message with two tool calls]:

Agent(
  description: "Explore codebase for {topic}",
  prompt: <step 3 prompt>,
  subagent_type: "agent-explore"
)

Agent(
  description: "Fetch docs for {library}",
  prompt: <step 4 prompt>,
  subagent_type: "agent-docs"
)
```

This ensures true parallel execution. Both agents work simultaneously and the orchestrator waits for both to complete before proceeding to Step 5.

If only one step is applicable (e.g., codebase exists but no libraries identified), spawn only that one.

Similarly, during refinement (Step 7), if multiple gaps target different agents, spawn all needed refinement agents in a SINGLE message for parallel execution.

---

## Orchestrator Responsibilities

The orchestrator (main Claude session) handles:

1. **Step 0 — CLASSIFY**: Assess query complexity, decompose if complex.
2. **Step 1 — CACHE CHECK**: Read memory for prior research.
3. **Spawn Step 2** — wait for completion.
4. **Process Step 2 output** — extract findings, libraries, contradictions, create compressed summary.
5. **Detect codebase** — parallel Glob for manifest files.
6. **Extract library list** — merge Step 2 libraries + manifest dependencies.
7. **Decide which steps to run** — based on codebase detection and library extraction.
8. **Spawn Steps 3 and/or 4** — in parallel when both applicable.
9. **Wait for all active agents** — collect outputs.
10. **Step 5 — SYNTHESIZE** — combine all outputs with conflict resolution, deduplication, grounding, contradiction surfacing, and confidence scoring.
11. **Step 6 — VERIFY** — score completeness, detect contradictions, identify gaps.
12. **Step 7 — REFINE** (conditional) — spawn targeted agents for gaps, merge results.
13. **Step 8 — PERSIST + OUTPUT** — write to memory if novel, deliver formatted response.

The orchestrator NEVER duplicates agent work. It does not search the web, explore the codebase, or query Context7 itself. It only classifies, orchestrates, synthesizes, verifies, and formats.
