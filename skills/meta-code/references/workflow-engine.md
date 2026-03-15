# Workflow Engine — meta-code Execution Specification

## Pattern Classification

meta-code implements an **Adaptive Pipeline + Fan-out/Fan-in hybrid** with quality gates:
- **Classifier**: Step 0 determines pipeline depth and whether to decompose.
- **Pipeline**: Step 2 (RESEARCH) must complete before Steps 3-4 start.
- **Fan-out**: Steps 3 (EXPLORE) and 4 (DOCUMENT) run in parallel.
- **Fan-in**: Step 5 (SYNTHESIZE) aggregates all results.
- **Quality Gate**: Step 6 (VERIFY) evaluates completeness and triggers refinement.
- **Refinement Loop**: Step 7 (REFINE) runs at most once for targeted gap-filling.

This combines Anthropic's "Prompt Chaining" pattern (Step 2 → 3/4) with "Parallelization — Sectioning" (Steps 3 and 4) and "Evaluator-Optimizer" (Steps 6-7).

## Step 0: CLASSIFY — Query Analysis

**Always executes first. No conditions. No agent spawning — orchestrator-only.**

**Purpose:** Assess query complexity to determine pipeline depth and whether to decompose into sub-questions.

### Classification Logic

Evaluate the query along these dimensions:
1. **Concept count**: How many distinct technical concepts does it involve?
2. **Hop count**: How many reasoning steps are needed? (single lookup vs. chain of lookups)
3. **Scope**: Single library/file vs. cross-cutting architectural concern
4. **Decision type**: Factual lookup vs. comparison vs. design decision

### Classification Decision Table

| Dimension | simple | moderate | complex |
|-----------|--------|----------|---------|
| Concepts | 1 | 2-3 | 4+ |
| Hops | 1 (direct answer) | 2 (lookup + apply) | 3+ (multi-hop reasoning) |
| Scope | Single library/API | Component or feature | Cross-cutting / architectural |
| Decision | Factual | How-to with context | Trade-off analysis |

**Assign the highest matching level.** If any dimension is `complex`, the query is complex.

### Pipeline Behavior by Level

| Level | Decomposition | Steps Executed | Refinement |
|-------|--------------|----------------|------------|
| `simple` | No | 0, 1, 2, (4 if library), 5, 8 | Never |
| `moderate` | No | 0, 1, 2, 3, 4, 5, 6, (7 if needed), 8 | If verify < 0.75 |
| `complex` | Yes (2-4 sub-questions) | All steps | If verify < 0.75 |

### Decomposition Format (complex only)

```
Sub-questions for: "{original_question}"

sq_001: {sub-question text}
  target_agent: websearch | explore | docs
  depends_on: []
  priority: 1

sq_002: {sub-question text}
  target_agent: websearch
  depends_on: [sq_001]
  priority: 2

sq_003: {sub-question text}
  target_agent: docs
  depends_on: []
  priority: 1
```

Rules:
- Maximum 4 sub-questions. If more are needed, the question should be split by the user.
- Independent sub-questions (empty `depends_on`) run in parallel.
- Dependent sub-questions wait for upstream results, which are prepended to their prompts.
- Priority 1 = critical to answering the question. Priority 3 = supplementary.

---

## Step 1: CACHE CHECK — Memory Retrieval

**Always executes. No agent spawning.**

**Purpose:** Check persistent memory for prior research that could supplement or replace agent calls.

### Execution Logic

1. Check if a memory directory exists for the current project (`~/.claude/projects/*/memory/`).
2. If it exists, read `MEMORY.md` index.
3. Scan for `reference` type entries whose description matches the query topic.
4. If matching entries found, read them and assess:
   - **Freshness**: Is the entry < 7 days old? (check retrieval date in content)
   - **Relevance**: Does it directly address the query or a sub-question?
   - **Confidence**: Was the original finding marked as high confidence?

### Decision

| Condition | Action |
|-----------|--------|
| Fresh + relevant + high confidence | Use as primary source, skip corresponding agent |
| Stale OR medium confidence | Use as supplementary context, still run agents |
| No matching entries | Proceed normally |
| No memory directory | Skip step, proceed |

---

## Step 2: RESEARCH (agent-websearch)

**Always executes. No conditions.**

**Purpose:** Establish external context — current best practices, recent changes, ecosystem landscape, relevant articles.

**Input:** The user's question (or sub-questions for `complex` queries), reformulated as 1-3 web search queries.

**Query reformulation rules:**
1. Extract the core technical topic from the user's question.
2. Add specificity: include language, framework, version if mentioned.
3. Add currency: include current year for time-sensitive topics.
4. For broad questions, craft 2-3 complementary queries covering different angles.
5. For `complex` queries: generate searches that cover each `websearch`-targeted sub-question.

**Output extraction — pass downstream as compressed summary (<500 words):**
- Key findings (numbered list, max 8 items)
- Library/framework names mentioned (used to trigger Step 4)
- Version numbers found
- Best practice recommendations
- Notable URLs for citation
- **Contradictions detected** — any conflicting claims across different sources

**Timeout:** 60 seconds. If timeout, proceed with empty research context.

---

## Step 3: EXPLORE (agent-explore) — Conditional

**Condition:** A codebase must exist in the current working directory.

**Detection logic** (run by the orchestrator BEFORE spawning):
```
Check for any of these files in the current working directory:
- Cargo.toml
- package.json
- pyproject.toml
- go.mod
- pom.xml
- build.gradle / build.gradle.kts
- *.sln / *.csproj
- Makefile / CMakeLists.txt
- composer.json
- mix.exs
- deno.json / deno.jsonc
- .git/ (fallback — if a git repo exists, there's likely a project)
```

If NONE found: Skip Step 3. Set `codebase_context = "No codebase detected in current directory."`.

**Input:** User's question + Step 2 research summary (to guide what to look for in the codebase).

**Exploration focus:**
- How the user's question relates to existing code
- Relevant files, functions, types, patterns
- Current architecture and conventions that affect the answer
- Existing implementations of similar functionality

**Required output fields:**
- Findings with file:line references
- `contradictions[]`: any conflicts between codebase patterns and web research recommendations

**Timeout:** 90 seconds. If timeout, proceed with whatever partial results were returned.

---

## Step 4: DOCUMENT (agent-docs) — Conditional

**Condition:** Specific libraries or frameworks must be identified from Step 2 output or from codebase detection.

**Library extraction logic:**
1. From Step 2 research summary: extract any library/framework names explicitly mentioned as relevant to the answer.
2. From codebase manifest (if Step 3 also runs): extract dependency names that relate to the user's question.
3. If neither source yields library names: Skip Step 4. Set `docs_context = "No specific library documentation needed."`.

**Max libraries per invocation:** 2 (due to the 3-call Context7 limit).

**Input:** User's question + library names + version information (from codebase if available).

**Required output fields:**
- API details, code examples, version notes
- `contradictions[]`: any conflicts between documentation and web research claims

**Timeout:** 45 seconds. If timeout, proceed with whatever partial results were returned.

---

## Step 5: SYNTHESIZE (orchestrator)

**Always executes. Waits for all active agents to complete.**

**Input:** All agent outputs (Step 2 always, Step 3 if codebase existed, Step 4 if libraries identified), plus cached findings from Step 1.

### Synthesis Rules

1. **Conflict resolution priority:**
   - Official documentation (Step 4) > Web research (Step 2) > Codebase patterns (Step 3)
   - Exception: if the codebase has an intentional deviation from docs (e.g., custom wrapper), note BOTH approaches.

2. **Deduplication:**
   - If Steps 2 and 4 both found the same documentation, use Step 4's version (more structured).
   - If Steps 2 and 3 both describe the same pattern, cite the codebase file:line reference (more specific).

3. **Grounding:**
   - Every claim must trace to a source (URL, file:line, or Context7 library ID).
   - If a recommendation cannot be sourced, mark it as "Based on general best practices."

4. **Contradiction surfacing (NEW):**
   - Do NOT silently resolve contradictions.
   - Apply triangulation threshold: if 2+ of 3 independent sources agree on a claim → mark as "corroborated."
   - If sources disagree → list each position with its source in the `Contested Claims` output section.
   - Contradictions from agent outputs (`contradictions[]` fields) are surfaced here.

5. **Confidence scoring (NEW):**
   Compute confidence based on these signals:

   | Signal | Weight |
   |--------|--------|
   | Number of concordant sources | High |
   | Source authority (official docs > blogs > forums) | High |
   | Source recency (current year > 2+ years old) | Medium |
   | Coverage (all sub-questions answered) | Medium |
   | Presence of unresolved contradictions | Negative |
   | Agent failures or timeouts | Negative |

   Assign level:
   - `high`: 3+ concordant authoritative sources, no unresolved contradictions, full coverage
   - `medium`: 2 sources or mixed authority, minor gaps
   - `low`: single source, significant gaps, or unresolved contradictions

6. **Code examples:**
   - If Step 3 found relevant existing code AND Step 4 found documentation examples, present the documentation example adapted to the project's conventions.
   - If only Step 4 has examples, present them as-is with a note about adapting to project conventions.
   - If only Step 2 has code snippets, present them with caveats about verifying correctness.

7. **Gap identification (for verify step):**
   - Track which sub-questions (if decomposed) are fully answered, partially answered, or unanswered.
   - Track which claims lack corroboration.

---

## Step 6: VERIFY (Quality Gate)

**Always executes after synthesis. Orchestrator-only — no agent spawning.**

**Purpose:** Evaluate synthesis quality and decide whether refinement is needed.

### Completeness Scoring (0.0 to 1.0)

Score components:
- **Question coverage** (0.4 weight): Does the answer address the full question? For decomposed queries, what fraction of sub-questions are answered?
- **Source backing** (0.3 weight): What fraction of claims have source citations?
- **Actionability** (0.2 weight): Does the answer include concrete next steps or code examples?
- **Coherence** (0.1 weight): Is the answer internally consistent? Are contradictions properly surfaced?

`completeness = 0.4 * coverage + 0.3 * source_backing + 0.2 * actionability + 0.1 * coherence`

### Contradiction Assessment

- Count unresolved contradictions from synthesis.
- If any contradictions involve safety-critical or correctness-critical claims, flag as `critical_gap`.

### Gap Identification

For each identified gap, record:
- `gap_description`: What information is missing
- `target_agent`: Which agent could fill it (websearch | explore | docs)
- `target_query`: A focused query to address the gap
- `severity`: critical | important | minor

### Decision Logic

```
IF complexity_level == "simple":
    → SKIP to Step 8 (never refine simple queries)

IF completeness >= 0.75 AND no critical_gaps:
    → SKIP to Step 8

IF completeness < 0.75 OR critical_gaps exist:
    → Proceed to Step 7 (REFINE)
    → Pass gap list with target agents and queries
```

---

## Step 7: REFINE (Conditional — max 1 iteration)

**Condition:** Step 6 completeness < 0.75 OR critical gaps identified.

**Never runs for `simple` queries. Maximum 1 refinement iteration.**

**Purpose:** Targeted gap-filling. Re-run ONLY the agents whose domain matches identified gaps.

### Execution Logic

1. From Step 6, collect all gaps with `severity` >= `important`.
2. Group gaps by `target_agent`.
3. Spawn ONLY the needed agents, with focused prompts targeting specific gaps:

```
// Only if gaps target websearch:
Agent(
  description: "Refine: {gap_description}",
  prompt: "The following specific gap was identified in prior research: {gap_description}. Search specifically for: {target_query}. Focus narrowly on this gap — do not repeat broad research.",
  subagent_type: "agent-websearch"
)

// Only if gaps target explore:
Agent(
  description: "Refine: {gap_description}",
  prompt: "Look specifically for: {gap_description}. Prior exploration missed this. Focus on: {target_query}.",
  subagent_type: "agent-explore"
)

// Only if gaps target docs:
Agent(
  description: "Refine: {gap_description}",
  prompt: "Look up specifically: {gap_description}. Prior documentation lookup missed this. Focus on: {target_query}.",
  subagent_type: "agent-docs"
)
```

4. Merge refinement results into the existing synthesis:
   - Update relevant sections with new findings.
   - Update confidence score.
   - Update gap list (remove filled gaps, keep persistent ones).

5. **Termination rule:** Do NOT re-verify or loop again. If gaps persist after 1 refinement, report them honestly in the output. Over-refinement degrades quality.

---

## Step 8: PERSIST + OUTPUT

### Memory Persistence (for `moderate` and `complex` queries only)

**Conditions for writing to memory:**
- The findings are novel (not already in memory).
- The findings are likely reusable in future conversations (not highly context-specific).
- Confidence is `medium` or `high`.
- The topic relates to a library, pattern, or architectural decision that may come up again.

**Memory entry format:**

```markdown
---
name: research-{topic-slug}
description: "Key findings on {topic} — {one-line summary}"
type: reference
---

Research on: {topic}
Date: {YYYY-MM-DD}
Confidence: {high|medium|low}

Key findings:
1. {finding with source}
2. {finding with source}

Sources:
- [Title](URL)
```

**Write to:** `~/.claude/projects/{project-path}/memory/research-{topic-slug}.md`
**Update MEMORY.md:** Add a pointer to the new file.

If a similar entry already exists, UPDATE it rather than creating a duplicate.

### Output Delivery

Follow the output format defined in SKILL.md.

---

## Execution Flow — Complete Decision Tree

```
START
  │
  ├─ Step 0: CLASSIFY query
  │     ├─ complexity_level: simple | moderate | complex
  │     └─ sub_questions[] (if complex)
  │
  ├─ Step 1: CACHE CHECK
  │     ├─ cached_findings[] (if any)
  │     └─ agents_to_skip[] (if cache is fresh)
  │
  ├─ Step 2: SPAWN agent-websearch (always)
  │     └─ Wait for completion (max 60s)
  │
  ├─ EXTRACT from Step 2 output:
  │     ├─ key_findings: string (summary, <500 words)
  │     ├─ libraries: string[] (names extracted)
  │     ├─ versions: map<string, string> (library → version)
  │     └─ contradictions: string[] (conflicting claims)
  │
  ├─ DETECT codebase: Glob for manifest files
  │     ├─ codebase_exists: bool
  │     └─ manifest_deps: string[] (dependency names from manifest)
  │
  ├─ MERGE library list: libraries ∪ (manifest_deps ∩ relevant_to_question)
  │
  ├─ DECIDE parallel steps:
  │     ├─ IF codebase_exists → spawn Step 3
  │     ├─ IF libraries.length > 0 → spawn Step 4
  │     └─ IF neither → skip to Step 5
  │
  ├─ WAIT for all spawned agents to complete
  │
  ├─ Step 5: SYNTHESIZE
  │     ├─ Apply conflict resolution, deduplication, grounding
  │     ├─ Surface contradictions (triangulation threshold: 2/3)
  │     ├─ Score confidence (high | medium | low)
  │     └─ Identify gaps
  │
  ├─ Step 6: VERIFY (quality gate)
  │     ├─ Score completeness (0.0 to 1.0)
  │     ├─ Detect critical gaps
  │     └─ DECIDE:
  │           ├─ IF simple → Step 8
  │           ├─ IF score >= 0.75 AND no critical gaps → Step 8
  │           └─ IF score < 0.75 OR critical gaps → Step 7
  │
  ├─ Step 7: REFINE (max 1 iteration)
  │     ├─ Spawn targeted agents for identified gaps
  │     ├─ Merge refinement into synthesis
  │     └─ Do NOT re-verify — proceed to Step 8
  │
  └─ Step 8: PERSIST + OUTPUT
        ├─ Write to memory (if novel + reusable)
        └─ Deliver final response
```

## Error Handling Matrix

| Scenario | Action |
|----------|--------|
| Step 2 returns empty | Proceed but note "Web research yielded no results." Steps 3-4 still run if conditions met. |
| Step 2 times out | Proceed with empty research context. Note the timeout. |
| Step 3 returns empty | Report "No relevant codebase findings." in codebase section. |
| Step 3 times out | Use partial results if any. Note timeout. |
| Step 4 Context7 unavailable | Report "Documentation lookup unavailable." Rely on Step 2 web results for docs. |
| Step 4 returns empty | Report "No documentation found for [library]." |
| Step 4 times out | Use partial results. Note timeout. |
| All agents fail | Return whatever is available with honest disclaimer. |
| Exa MCP unavailable | agent-websearch falls back to native WebSearch/WebFetch automatically. |
| Context7 resolve fails | agent-docs tries fallback plugin tools automatically. |
| Step 6 score < 0.75 but simple query | Skip refinement, output with gaps noted. |
| Step 7 refinement doesn't improve | Stop after 1 iteration, report remaining gaps. |
| Memory directory doesn't exist | Skip Step 1 and Step 8 persistence. |

## Performance Characteristics

| Step | Expected Duration | Parallelism |
|------|-------------------|-------------|
| Step 0 (CLASSIFY) | <1s | Runs alone (orchestrator) |
| Step 1 (CACHE CHECK) | <2s | Runs alone (orchestrator) |
| Step 2 (RESEARCH) | 10-30s | Runs alone |
| Step 3 (EXPLORE) | 15-60s | Parallel with Step 4 |
| Step 4 (DOCUMENT) | 5-20s | Parallel with Step 3 |
| Step 5 (SYNTHESIZE) | 5-10s | Runs alone (orchestrator) |
| Step 6 (VERIFY) | <2s | Runs alone (orchestrator) |
| Step 7 (REFINE) | 10-40s | Optional, targeted agents |
| Step 8 (PERSIST) | <2s | Runs alone (orchestrator) |
| **Total (simple, best)** | **20-40s** | |
| **Total (moderate, no refine)** | **35-80s** | |
| **Total (complex, with refine)** | **60-150s** | |
