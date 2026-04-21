# Workflow Engine — Single Source of Truth

Decision tables, scoring formulas, credibility tiers, invariant validation, error handling, and performance characteristics for the meta-code pipeline. For the pipeline overview and step descriptions, see [SKILL.md](../SKILL.md).

## Classification Decision Table (Step 1a)

Evaluated at Step 1a. Assign the **highest matching level** across all dimensions.

| Dimension | simple | moderate | complex |
|-----------|--------|----------|---------|
| Concepts | 1 | 2-3 | 4+ |
| Hops | 1 (direct answer) | 2 (lookup + apply) | 3+ (multi-hop) |
| Scope | Single library/API | Component/feature | Cross-cutting / architectural |
| Decision | Factual | How-to with context | Trade-off analysis |

## Pipeline Behavior by Level

| Level | Steps Executed | Decomposition | Challenge | Evaluator | Refinement | Topology |
|-------|---------------|---------------|-----------|-----------|------------|----------|
| `simple` | 0, 1, 2, 3(fast), 5(lite), 9 — or 3→4 if library detected | No | Never | Never | Never | 1 agent (websearch) |
| `moderate` | 0-9 | Entity tagging | Signal-conditional (contested claims or low confidence) | No (self-check) | Max 1 if verify < 0.75 | 2-3 agents (websearch + explore/docs) |
| `complex` | All 0-9 | Constraint-based | Yes | Yes (independent) | Max 1 if verify < 0.75, max 2 if verify < 0.5 | 3-5 agents (2x websearch + explore/docs + evaluator) |

### Dynamic Topology Notes

The topology column reflects agent COUNT, not just which steps run:
- **Simple:** Only agent-websearch. If early-exit triggers, no further agents. If a library is detected, add agent-docs (max 2 agents).
- **Moderate:** agent-websearch + conditional explore/docs. Standard 2-3 agent topology. Self-check at Step 7.
- **Complex:** 2x parallel agent-websearch (supportive + critical angles) at Step 2 + conditional explore/docs at Step 4 + independent evaluator at Step 7. Max 5 agents total.

## Early-Exit Conditions (Step 3d)

After Step 3 boundary check, if ALL of the following are true, skip Step 4 and go directly to Step 5:

1. `query_coverage: high` — web research addressed all key aspects
2. All `must_answer` items from Step 1 are addressed by T1-T2 sources
3. No libraries identified in the handoff `libraries` field needing docs lookup
4. No codebase detected (or query is purely conceptual — not about local code)

**When NOT to early-exit:** If `must_include` contains `code_example` and the codebase exists, always run Step 4 EXPLORE to ground examples in actual project patterns.

### Simple Fast-Path

For `simple` queries that trigger early-exit, apply additional streamlining:
- **Step 3 (GATE)**: Skip 3b boundary check and 3c route detection. Only run 3a compress + 3d early-exit check.
- **Step 5 (SYNTHESIZE)**: Skip 5f-5j (deduplication, contradiction surfacing, input coverage, confidence scoring, citation audit) — these are unnecessary when only one agent contributed.
- **Step 7 (VERIFY)**: Skip entirely — go directly from Step 5 to Step 9.

This reduces simple queries to: 0 → 1 → 2 → 3(fast) → 5(lite) → 9. Target: 15-25s total.

## Query Decomposition (Step 1c)

### Moderate queries — Entity tagging

```
sq_001: {question} | target: websearch | entities: [known: X, unknown: Y] | depends_on: [] | priority: 1
sq_002: {question} | target: docs     | entities: [known: X, unknown: Z] | depends_on: [] | priority: 1
```

Tag each entity as `known` (user-specified) or `unknown` (to be retrieved). Independent sub-questions run in parallel.

### Complex queries — Constraint-based decomposition

For complex queries, decompose into atomic constraint triples with a sufficiency check:

```
constraint_001: {subject} {relation} {object} | type: factual | target: websearch | priority: 1
constraint_002: {subject} {relation} {object} | type: comparative | target: docs | priority: 1
constraint_003: {subject} {relation} {object} | type: causal | target: explore | depends_on: [001] | priority: 2
sufficiency_check: true
```

**Sufficiency check:** After each retrieval round, verify whether accumulated evidence satisfies all constraints. Halt retrieval when constraints are met — prevents retrieval drift where agents keep searching past the point of diminishing returns.

### Per Sub-question Source Priority

Each sub-question may override the global `source_priority`:

| Sub-question nature | Recommended priority |
|---|---|
| API details, configuration | `product` (T1 docs > T2 blogs) |
| Trade-offs, architecture decisions | `architecture` (T2 blogs > T1 docs > codebase) |
| Research findings, benchmarks | `academic` (T1 papers > T1 docs) |
| Current state, recent changes | `product` with recency boost |

Set via `source_priority` field in the sub-question definition.

## Source Credibility Tiers

Single authoritative definition. Referenced by SKILL.md Step 5c and synthesis-template.md.

| Tier | Weight | Examples |
|------|--------|----------|
| T1 | 1.0 | Official docs, RFCs, specs, primary research papers |
| T2 | 0.7 | Engineering blogs (major companies), reputable tech media |
| T3 | 0.4 | Community blogs, Stack Overflow, forum answers |
| T4 | 0.2 | AI-generated content, SEO-optimized articles |

### Source Priority by Query Type

| Query Type | Priority Order |
|---|---|
| `academic` | T1 papers > T1 docs > T2 eng blogs > T3 community |
| `product` | T1 docs > T2 eng blogs > T1 papers > T3 community |
| `architecture` | T2 eng blogs > T1 docs > codebase patterns > T1 papers |

Default: `product` (most common for dev questions). Set via `source_priority` in Step 1b success criteria.

### Calibration Correction

Tool-using agents (web search) are systematically over-confident because retrieved text superficially resembles correct information. Apply:
- **Web-only claims** (no docs/codebase corroboration) → treat as one tier below apparent confidence
- **Corroborated claims** (docs or codebase confirm) → keep stated confidence
- **T3-T4 only claims** → flag as `needs verification`

## Confidence Scoring

Two-dimensional model (inspired by Google ADK evaluation criteria):

### Trajectory Confidence — Were the right agents and tools used?

| Signal | Weight | Direction | Measurement |
|--------|--------|-----------|-------------|
| All planned agents executed successfully | High | Positive | Binary: all succeeded / any failed |
| Coverage (sub-questions answered) | High | Positive | answered_sub_questions / total_sub_questions |
| Search depth per finding | Medium | Positive | avg(queries_per_key_finding) — deeper = higher signal |
| Agent convergence rate | Medium | Positive | concordant_claims / total_cross_agent_claims |
| Agent failures/timeouts | Medium | Negative | Count of failed/timed-out agents |
| Coordination gaps detected (Steps 2z/4z) | Medium | Negative | Count of MAST coordination failures |
| Refinement triggered | Low | Negative | Boolean — refinement needed = trajectory gap |
| Challenge survival rate | Low | Positive | confirmed_claims / total_challenged (Step 6 only) |
| Early-exit triggered with full coverage | Low | Positive | Boolean |

### Response Confidence — Is the content factually correct?

| Signal | Weight | Direction | Measurement |
|--------|--------|-----------|-------------|
| Concordant sources (count) | High | Positive | Count of 2+ source claims |
| Source tier (T1-T2 vs T3-T4) | High | Positive | T1-T2 claims / total claims |
| Source diversity | High | Positive | unique_domains / total_sources (see below) |
| Source recency (current year) | Medium | Positive | current_year_sources / total_sources |
| Challenge results (confirmed claims) | Medium | Positive | confirmed / challenged |
| Unresolved contradictions | Medium | Negative | Count of unresolved |
| Niche topic indicator | Medium | Negative | < 3 T1-T2 sources found (see below) |

### Combined Level

| Level | Criteria |
|-------|----------|
| `high` | Both trajectory + response are strong: 3+ concordant T1-T2 sources, no unresolved contradictions, full coverage, all agents succeeded, claims survived challenge |
| `medium` | Either dimension has gaps: 2 sources or mixed tiers, minor gaps, some claims weakened by challenge, or 1 agent failed |
| `low` | Either dimension is weak: single source, significant gaps, unresolved contradictions, claims refuted, or multiple agent failures |

Output format: `**Confidence:** {level} (trajectory: {t_level}, response: {r_level}) — {basis}`

### Source Diversity Score

Measures whether sources cluster around a single authority or span multiple independent domains:

```
source_diversity = unique_domains / total_sources
```

| Score | Interpretation | Action |
|-------|---------------|--------|
| >= 0.7 | High diversity — independent corroboration | No adjustment |
| 0.5–0.7 | Moderate diversity — some clustering | No adjustment |
| < 0.5 | Low diversity — sources cluster around same authority | Downgrade response confidence by one level |

Example: 4 sources from `docs.rust-lang.org` + 1 from `blog.rust-lang.org` = 2/5 = 0.4 → low diversity despite high tier.

### Niche Topic Cap (Dunning-Kruger correction)

LLMs exhibit systematically high confidence on topics where they have the weakest training signal (arXiv:2603.09985). When limited authoritative sources exist, the model's internal confidence is unreliable.

**Rule:** If fewer than 3 T1-T2 sources were found across ALL agents (websearch + docs + explore combined), automatically cap overall confidence at `medium` regardless of other signals. Add note: "Limited authoritative sources available — confidence capped."

This prevents high-confidence answers on niche topics where the pipeline lacks sufficient grounding material.

## Completeness Scoring Formula (Step 7a)

```
completeness = 0.35 * question_coverage
             + 0.25 * source_backing
             + 0.20 * actionability
             + 0.10 * coherence
             + 0.10 * noise_ratio
```

| Component | 1.0 | 0.5 | 0.0 |
|-----------|-----|-----|-----|
| `question_coverage` | All `must_answer` items addressed | Most addressed | Major items missing |
| `source_backing` | All claims cited | Most cited | Many unsourced |
| `actionability` | All `must_include` present | Some present | None present |
| `coherence` | Consistent, contradictions surfaced | Minor issues | Internal contradictions |
| `noise_ratio` | All claims map to `must_answer` items | Some tangential claims | Many off-topic claims |

**Threshold:** 0.75. Below this → Step 8 (REFINE), unless `simple` query.

**Noise Ratio** (deterministic): Count claims in the synthesis. Count claims that directly address a `must_answer` item. `noise_ratio = on_topic_claims / total_claims`. Score: 1.0 if ratio >= 0.9, 0.5 if ratio >= 0.7, 0.0 if ratio < 0.7.

## Pipeline Invariants (Step 7b)

Deterministic checks run during Step 7 VERIFY. Each invariant produces a pass/fail result.

| # | Invariant | Check Method | Severity |
|---|-----------|-------------|----------|
| INV-1 | Every factual claim has a source URL | Scan synthesis for unsourced factual assertions | Critical |
| INV-2 | Every `must_answer` item has a response | Map `must_answer` list against synthesis sections | Critical |
| INV-3 | No T3-T4-only claims without `needs verification` flag | Cross-reference claim sources against tier table | Major |
| INV-4 | Time-sensitive claims cite current/previous year sources | Check source dates for claims about "latest", "current", "best practice" | Major |
| INV-5 | All `must_include` items present | Check for code_example, version_info, trade_offs as specified | Major |
| INV-6 | Code examples reference existing codebase entities | If `how_to` + codebase exists: quick Grep for imports, functions, framework version | Minor |
| INV-7 | Each active agent's output is represented | Verify no agent findings silently dropped | Minor |
| INV-8 | Per-claim source grounding | Segment synthesis into individual claims; verify each maps to a source URL from agent output | Major |

**Critical** invariant failure → always triggers Step 8 refinement (regardless of completeness score).
**Major** invariant failure → logged as gap, triggers Step 8 if completeness < 0.75.
**Minor** invariant failure → logged as warning in output, does not trigger refinement.

## Typed Handoff Format (Step 3a)

The canonical format for inter-agent handoff. Replace prose summaries.

```
Research context for downstream agents:

claims:
- text: "{finding}" | source: "{url}" | tier: T1|T2|T3|T4 | date: "YYYY-MM"
- text: "{finding}" | source: "{url}" | tier: T1|T2|T3|T4 | date: "YYYY-MM"

libraries: [{name: "lib", version: "X.Y.Z"}]
contradictions: ["{claim_a} (source_a) vs {claim_b} (source_b)"]
gaps: ["{what was not found}"]
query_coverage: high|medium|low
```

Target: 300-500 tokens total. Include source URLs as pointers for restorability.

## Step 5a Compression Format

After Step 4 agents return, compress their outputs before synthesis:

```
web_research: {step_3a_typed_handoff — already compressed}
codebase:
- finding: "{what}" | file: "{path}:{line}" | relevance: high|medium
- finding: "{what}" | file: "{path}:{line}" | relevance: high|medium
docs:
- api: "{function/type}" | detail: "{key info}" | version: "X.Y.Z" | source: "ctx7:{library_id}"
- api: "{function/type}" | detail: "{key info}" | version: "X.Y.Z" | source: "ctx7:{library_id}"
contradictions_cross_source: ["{web says X, docs say Y, codebase does Z}"]
```

Target: 300-500 tokens per source. Total pre-synthesis input: 900-1500 tokens max.

## Codebase Detection (Step 3c)

**Fast path:** Check for `.git` directory first. If found → `codebase_exists = true`, proceed to manifest detection for library extraction.

**Manifest detection** (parallel Glob, for library extraction only):
`Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`, `pom.xml`, `build.gradle`, `*.sln`, `composer.json`, `mix.exs`, `deno.json`

**No `.git`?** → `codebase_exists = false`, skip EXPLORE agent. Exception: if manifest files exist without `.git`, treat as codebase.

## Library Extraction (Step 3c)

1. Parse `libraries` field from Step 3a typed handoff.
2. If codebase exists, read manifest for dependency versions.
3. Select top 1-2 libraries most relevant to the question.
4. Max 2 libraries per invocation (due to 3-call ctx7 limit).

## Error Handling Matrix

| Scenario | Action |
|----------|--------|
| Step 2 returns empty | Proceed. Note "Web research yielded no results." Step 4 still runs if conditions met. |
| Step 2 times out (60s) | Proceed with empty research context. Note timeout. |
| Step 3 boundary check fails | Note gaps, pass them to Step 4 agents as specific search targets. |
| Step 3 early-exit triggers | Skip Step 4, proceed to Step 5 with web research only. |
| Step 4 EXPLORE returns empty | Report "No relevant codebase findings." |
| Step 4 EXPLORE times out (90s) | Use partial results if any. Note timeout. |
| Step 4 DOCUMENT ctx7 CLI fails | Report "Documentation lookup unavailable." Rely on Step 2. |
| Step 4 DOCUMENT ctx7 quota exhausted | Report quota error, suggest `ctx7 login`. Rely on Step 2. |
| Step 4 DOCUMENT returns empty | Report "No documentation found for [library]." |
| Step 4 DOCUMENT times out (45s) | Use partial results. Note timeout. |
| Step 6 CHALLENGE returns empty | Proceed without challenge results. Note in confidence basis. |
| Step 6 CHALLENGE times out (45s) | Use partial results. Claims not challenged are noted. |
| All agents fail | Return whatever is available with honest disclaimer. |
| Exa MCP unavailable | agent-websearch falls back to native WebSearch/WebFetch. |
| Step 7 invariant failure but simple query | Log warning, skip refinement, output with gaps noted. |
| Step 7 evaluator agent fails (complex) | Fall back to orchestrator self-check. Note in confidence basis. |
| Step 8 doesn't improve (<5% change) | Stop after 1 iteration, report remaining gaps. |
| No memory directory | Skip cache check and persistence. |
| Step 0 trivial bypass | Answer directly from model knowledge, skip pipeline. Flag as `trivial_bypass`. |
| Step 2 complex: one of two websearch agents fails | Use available results, note partial web coverage. |

<!-- Performance Characteristics — Design-time reference only, not used during pipeline execution.
Simple: 15-40s | Moderate: 50-110s | Complex: 80-190s
See git history for detailed per-step timing table if needed for diagnostics. -->

## Query Enrichment Templates (Step 1d)

**Simple queries** — Single-line rewrite:
```
websearch_instruction: "Find {specific_answer} — prioritize {source_type} sources from {current_year}"
```

**Moderate queries** — Per-agent instructions:
```
websearch_instruction: "Find X, Y, Z — prioritize {source_type} sources"
explore_instruction: "Look for patterns matching X in {likely_dirs}"
docs_instruction: "Check API for {specific_function} in {library} v{version}"
```

**Complex queries** — Full enrichment with format specification:
```
websearch_instruction: "Find X, Y, Z — prioritize {source_type}. Expected format: comparison table with columns [A, B, C]"
explore_instruction: "Trace the flow from {entry_point} through {layers}. Map dependencies."
docs_instruction: "Check API for {function} in {library} v{version}. Include migration notes from v{old}."
```

## Plan Answer Shape (complex queries only, Step 1)

Outline the expected answer structure before dispatching agents:
- The 2-4 sections the answer will need
- Which agent is responsible for each section
- What a "done" answer looks like structurally

This prevents agents from producing overlapping or disjointed outputs.

## Cache Check (Step 1)

Scan `~/.claude/projects/*/memory/` for prior research matching the topic.
- Fresh + relevant + high confidence → use as primary source, skip corresponding agent
- Stale or medium confidence → supplementary context, still run agents
- No match or no memory directory → proceed normally

## Cross-Run Learning Metadata (Step 9)

When persisting findings to memory, include both research and pipeline operational metadata:

```yaml
---
name: research-{topic-slug}
description: "{one-line summary}"
type: reference
query_strategies:
  effective: ["{search pattern that yielded T1 results}"]
  ineffective: ["{search pattern that returned noise}"]
reliable_sources: ["{domain that consistently provided T1-T2 content}"]
unreliable_sources: ["{domain that returned T4/outdated content}"]
pipeline_performance:
  complexity_level: simple|moderate|complex
  agents_spawned: [websearch, explore, docs, challenge, evaluator]
  early_exit_triggered: true|false
  refinement_needed: true|false
  refinement_gap: "{what was missing}"
  completeness_score: 0.85
  invariant_failures: ["{INV-N: description}"]
---
```

This metadata improves future cache checks (workflow-engine.md Cache Check), query enrichment (Step 1d), and identifies query patterns that systematically require refinement.
