---
model: opus
name: meta-code
description: "Intelligent multi-agent workflow that answers development questions by orchestrating web research, codebase exploration, and documentation lookup in an adaptive pipeline with quality gates, confidence scoring, and iterative refinement. Use when the user says 'meta-code', '/meta-code', 'research and answer', 'deep research', 'full analysis', or 'comprehensive answer'. Spawns agent-websearch, agent-explore, and agent-docs as subagents, synthesizes their outputs into a single grounded response with sources and confidence attribution."
argument-hint: "[question or topic to research]"
---

# meta-code — Adaptive Multi-Agent Research Pipeline

## Overview

meta-code is an adaptive pipeline that answers development questions by combining:
1. **Query classification** — assess complexity, decompose if needed
2. **Cache check** — retrieve prior research from memory
3. **Web research** (agent-websearch) — current best practices, articles, ecosystem context
4. **Codebase analysis** (agent-explore) — relevant patterns, existing code, architecture
5. **Documentation lookup** (agent-docs) — official API details, code examples, version-accurate docs
6. **Verification gate** — completeness scoring, contradiction detection
7. **Iterative refinement** — targeted gap-filling (max 1 extra iteration)
8. **Memory persistence** — write key findings for future conversations

The pipeline adapts its depth based on query complexity. Simple questions skip decomposition and refinement. Complex multi-hop questions get full treatment.

## Execution Flow

```
User Question
     │
     ▼
┌─────────────┐
│   Step 0:   │
│  CLASSIFY   │  ← Assess complexity, decompose if multi-hop
└──────┬──────┘
       │ complexity_level + sub_questions[]
       ▼
┌─────────────┐
│   Step 1:   │
│ CACHE CHECK │  ← Check memory for prior research
└──────┬──────┘
       │ cached_findings[]
       ▼
┌─────────────┐
│   Step 2:   │
│  RESEARCH   │  ← Always runs (agent-websearch)
│   (web)     │
└──────┬──────┘
       │ compressed summary + library names
       ▼
┌──────┴──────────────────┐
│         PARALLEL         │
│  ┌──────────┐  ┌──────────┐
│  │  Step 3:  │  │  Step 4:  │
│  │  EXPLORE  │  │ DOCUMENT  │
│  │(codebase) │  │  (docs)   │
│  └────┬─────┘  └────┬─────┘
│       │              │     │
└───────┼──────────────┼─────┘
        ▼              ▼
   ┌──────────────────────┐
   │       Step 5:         │
   │     SYNTHESIZE        │  ← Combine all findings + confidence scoring
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │       Step 6:         │
   │       VERIFY          │  ← Quality gate: completeness, contradictions
   └──────────┬───────────┘
              │
         ┌────┴────┐
         │ score   │
         │ < 0.75? │
         └────┬────┘
          yes │ no
              │  └──► Step 8: PERSIST + OUTPUT
              ▼
   ┌──────────────────────┐
   │       Step 7:         │
   │       REFINE          │  ← Targeted gap-filling (max 1 iteration)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │       Step 8:         │
   │   PERSIST + OUTPUT    │  ← Write to memory + deliver response
   └──────────────────────┘
```

## Runtime Output Format

Before each step, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Step N/8] STEP_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Between major steps, print a thin separator: `───────────────────────────────`

## Step-by-Step Execution

### Step 0 — CLASSIFY

Print: `[Step 0/8] CLASSIFY`

Assess query complexity BEFORE dispatching any agents. This is a fast, local analysis by the orchestrator — no agent spawning.

**Classification levels:**

| Level | Criteria | Pipeline Behavior |
|-------|----------|-------------------|
| `simple` | Single concept, one library, factual lookup | Skip decomposition. Phase 1 only + optional Phase 3. No refinement loop. |
| `moderate` | Involves 2-3 concepts, comparison, or how-to with context | Full pipeline, no decomposition. Refinement if verify fails. |
| `complex` | Multi-hop reasoning, cross-cutting concerns, architecture decision, >3 concepts | Full pipeline WITH decomposition into sub-questions. Refinement enabled. |

**Decomposition (complex queries only):**

Break the question into 2-4 sub-questions. Each sub-question carries:
- `id`: unique identifier (sq_001, sq_002, ...)
- `question`: the sub-question text
- `target_agent`: which agent should answer it (websearch, explore, docs)
- `depends_on`: list of sub-question IDs that must complete first ([] if independent)
- `priority`: 1 (critical) to 3 (nice-to-have)

Organize sub-questions as a DAG. Independent sub-questions run in parallel. Dependent ones wait for upstream results.

### Step 1 — CACHE CHECK

Print: `[Step 1/8] CACHE CHECK`

Check persistent memory (`~/.claude/projects/*/memory/`) for prior research relevant to the question.

- Read MEMORY.md index for any entries tagged as `reference` type that match the topic.
- If fresh, high-confidence cached results exist (< 7 days old), incorporate them and skip the corresponding agent dispatch.
- If stale or low-confidence, proceed normally but use cached data as supplementary context.

This step is fast — just file reads. Skip if no memory directory exists.

### Step 2 — Spawn RESEARCH (agent-websearch)

Print: `[Step 2/8] RESEARCH (web)`

Always runs first. Spawn agent-websearch via Agent tool:

```
Agent(
  description: "Web research on {topic}",
  prompt: [Phase 1 prompt template from references/agent-protocols.md],
  subagent_type: "agent-websearch"
)
```

Wait for completion. Extract from the output:
- **Key findings** — numbered list of important discoveries
- **Library names** — any libraries/frameworks mentioned as relevant
- **Compressed summary** — <500 words for passing to Steps 3/4
- **Contradictions detected** — any conflicting claims across sources

For `complex` queries with decomposition: reformulate web searches to cover each sub-question targeted at `websearch`.

### Step 3 — Detect Conditions + Spawn EXPLORE and DOCUMENT in Parallel

Print: `[Step 3-4/8] EXPLORE + DOCUMENT (parallel)`

**Codebase detection** — Run parallel Glob calls for manifest files:
- `Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`
- If any found → Step 3 EXPLORE is active
- If none found → skip EXPLORE

**Library extraction** — From Step 2 output, extract library names:
- If libraries identified → Step 4 DOCUMENT is active
- If codebase manifest exists, also check for relevant dependencies
- If no libraries found → skip DOCUMENT

Spawn both in a SINGLE message with multiple Agent tool calls for true parallelism:

```
// Only if codebase detected:
Agent(
  description: "Explore codebase for {topic}",
  prompt: [Phase 2 prompt with compressed research summary],
  subagent_type: "agent-explore"
)

// Only if libraries identified:
Agent(
  description: "Fetch docs for {library}",
  prompt: [Phase 3 prompt with library names and versions],
  subagent_type: "agent-docs"
)
```

### Step 5 — SYNTHESIZE

Print: `[Step 5/8] SYNTHESIZE`

Combine all agent outputs into a draft response. Apply these rules:

**Conflict resolution:** Official docs (Step 4) > Web research (Step 2) > Codebase patterns (Step 3)
- Exception: if the codebase has an intentional deviation, note both approaches.

**Deduplication:** If multiple phases found the same information, use the most authoritative version and cite it once.

**Grounding:** Every claim must trace to a source (URL, file:line, or Context7 library ID). Unsourced claims must be marked as "Based on general best practices."

**Contradiction surfacing:** Do NOT silently resolve contradictions. Surface them explicitly:
- If 2+ sources agree → mark as "corroborated"
- If sources disagree → list each position with its source in `contested_claims`
- Apply the triangulation threshold: 2/3 concordant sources = corroborated

**Confidence scoring:** Compute a confidence level based on:
- Number of concordant sources (more agreement = higher confidence)
- Source authority (official docs > blog posts > forum answers)
- Recency of sources (2026 > 2024)
- Coverage of the question (all sub-questions answered = higher confidence)

Assign one of: `high` (3+ concordant authoritative sources), `medium` (2 sources or mixed authority), `low` (single source or significant gaps).

### Step 6 — VERIFY (Quality Gate)

Print: `[Step 6/8] VERIFY`

After synthesis, perform a self-evaluation pass. This is done by the orchestrator, NOT a separate agent.

**Completeness check:**
- Score 0.0 to 1.0 based on: Does the answer address the full question? Are all sub-questions (if decomposed) covered? Are code examples provided when relevant?
- Threshold: 0.75

**Contradiction check:**
- Are there unresolved contradictions that could mislead the user?
- Are there claims without sources?

**Gap identification:**
- List specific areas where information is missing or weak.
- For each gap, identify which agent could fill it (websearch for factual gaps, explore for codebase gaps, docs for API gaps).

**Decision:**
- If completeness >= 0.75 AND no critical gaps → proceed to Step 8 (OUTPUT)
- If completeness < 0.75 OR critical gaps exist → proceed to Step 7 (REFINE)
- For `simple` queries: ALWAYS skip to Step 8 (no refinement)

### Step 7 — REFINE (Conditional, max 1 iteration)

Print: `[Step 7/8] REFINE`

Only runs if Step 6 identified gaps. Targeted refinement — do NOT re-run the entire pipeline.

**For factual gaps:** Spawn agent-websearch with a focused query targeting the specific gap.
**For codebase gaps:** Spawn agent-explore with a narrowed exploration focus.
**For API gaps:** Spawn agent-docs with specific API endpoints or methods to look up.

Rules:
- Maximum 1 refinement iteration. If gaps persist after refinement, report them honestly.
- Only re-run agents whose domain matches the identified gaps.
- Merge refinement results into the existing synthesis (don't rebuild from scratch).
- If issues do not decrease after refinement, stop and mark gaps in output.

### Step 8 — PERSIST + OUTPUT

Print: `[Step 8/8] PERSIST + OUTPUT`

**Persist to memory (for `moderate` and `complex` queries only):**
- If key findings are novel and likely useful in future conversations, write a `reference` type memory entry.
- Tag with: topic, source URLs, retrieval date, confidence level.
- Do NOT persist trivial or highly context-specific findings.

**Deliver the final response** following the Output Format below.

## Output Format

```markdown
## Answer

[Direct, actionable answer — 3-10 sentences. Most important finding first.]

**Confidence:** {high|medium|low} — {basis: e.g., "3 concordant sources, official docs confirmed"}

## Details

### From Web Research
[Key findings from Step 2 with source URLs.
 Or: "Web research did not yield relevant results."]

### From Codebase Analysis
[Findings from Step 3 with file:line references.
 Or: "No codebase detected." / "No relevant codebase findings."]

### From Documentation
[API details and code examples from Step 4 with Context7 sources.
 Or: "No specific library documentation needed." / "No docs found."]

### Contested Claims
[Claims where sources disagree — list each position with its source.
 Or: "No contradictions detected across sources."]

## Recommended Approach

[3-7 concrete next steps. Code examples tailored to user's context.]

## Sources
- [Source Title](URL) — annotation
- file:line — what was found
- Library: name vX.Y.Z via Context7

## Follow-up
[What additional research would materially improve this answer.
 Or: "No significant gaps identified."]
```

## Hard Rules

1. Step 0 ALWAYS runs — classify before dispatching.
2. Step 2 (RESEARCH) ALWAYS runs first among agents — web research provides foundation.
3. Steps 3 and 4 run in PARALLEL when both applicable — never sequentially.
4. Step 5 (SYNTHESIZE) only after all agents complete.
5. Step 6 (VERIFY) always runs after synthesis.
6. Step 7 (REFINE) runs at most ONCE, only if verify fails, and NEVER for `simple` queries.
7. Respect agent boundaries — websearch does NOT read code, explore does NOT fetch URLs, docs ONLY uses Context7.
8. Max 3 Context7 calls — agent-docs must stay within the hard limit.
9. Summarize before passing — compress Step 2 output before feeding to Steps 3/4 (<500 words).
10. Cite everything — every claim traces to a source.
11. Surface contradictions — never silently resolve conflicting information.
12. Confidence attribution — every answer carries a confidence level with its basis.
13. Graceful degradation — if any agent fails, continue with available data and note the gap.
14. No duplicate work — agents don't overlap domains.
15. Persist selectively — only write to memory when findings are novel and reusable.
16. Print `[Step N/8]` progress headers before each step — NEVER skip progress indicators.

## Error Handling

- If any agent returns empty results: note the gap in the relevant output section, proceed with other phases.
- If any agent times out: use partial results if available, note the timeout.
- If Exa MCP is unavailable: agent-websearch falls back to native WebSearch/WebFetch automatically.
- If Context7 is unavailable: report the failure, rely on Step 2 web results for documentation.
- If all phases fail: return whatever is available with an honest disclaimer.
- If refinement does not improve completeness: stop after 1 iteration, report remaining gaps.

## DO NOT

- Spawn all agents simultaneously — Step 2 must complete first.
- Run Steps 3 and 4 sequentially when they could be parallel.
- Pass raw Step 2 output to downstream agents — always compress first.
- Use TeamCreate for this workflow — use simple Agent tool spawning.
- Include unsourced claims in the synthesis without marking them.
- Repeat the same information across output sections.
- Silently resolve contradictions — surface them for the user.
- Skip the CLASSIFY step, even for seemingly simple questions.
- Refine more than once — one iteration max.
- Persist every result to memory — only novel, reusable findings.

## Done When

- [ ] Query classified (simple/moderate/complex) in Step 0
- [ ] Web research (Step 2) completed with compressed summary
- [ ] Applicable agents (explore/docs) spawned and completed
- [ ] Synthesis produced with confidence level and source attribution
- [ ] Verify gate (Step 6) evaluated — completeness score ≥ 0.75
- [ ] Final response follows the Output Format template
- [ ] Every claim traces to a source (URL, file:line, or Context7 library ID)

## Constraints (Three-Tier)

### ALWAYS
- Classify query complexity before dispatching agents (Step 0)
- Run web research (Step 2) first — it provides foundation for Steps 3-4
- Spawn Steps 3 and 4 in PARALLEL when both applicable
- Compress Step 2 output before feeding to Steps 3/4 (<500 words)
- Cite every claim with a source

### ASK FIRST
- Nothing — this is a read-only research workflow

### NEVER
- Spawn all agents simultaneously — Step 2 must complete first
- Pass raw Step 2 output to downstream agents — always compress first
- Include unsourced claims without marking them
- Refine more than once — one iteration max
- Silently resolve contradictions — surface them for the user

## References

- [Workflow Engine Specification](references/workflow-engine.md) — detailed execution logic, conditions, context passing, error handling, timeouts
- [Agent Protocols](references/agent-protocols.md) — exact Agent tool parameters, prompt templates, expected output formats for each agent
- [Research Notes](references/research-notes.md) — synthesized best practices from web research on multi-agent orchestration patterns
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — shared agent delegation rules, call budgets, authority hierarchy
- [Synthesis Template](@~/.claude/skills/_shared/synthesis-template.md) — standardized format for research synthesis output
