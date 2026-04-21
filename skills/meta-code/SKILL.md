---
model: opus
name: meta-code
description: "Intelligent multi-agent workflow that answers development questions by orchestrating web research, codebase exploration, and documentation lookup in an adaptive pipeline with quality gates, confidence scoring, and iterative refinement. Use when the user says 'meta-code', '/meta-code', 'research and answer', 'deep research', 'full analysis', or 'comprehensive answer'. Spawns agent-websearch, agent-explore, and agent-docs as subagents, synthesizes their outputs into a single grounded response with sources and confidence attribution."
argument-hint: "[question or topic to research]"
assumption_baseline: "claude-opus-4-6"
last_assumption_audit: "2026-03-25"
---

# meta-code — Adaptive Multi-Agent Research Pipeline

## Pipeline Overview

A 10-step adaptive pipeline (Step 0-9) combining prompt chaining, parallelization, generator-evaluator separation, and evaluator-optimizer patterns.

`TRIAGE → ANALYZE → RESEARCH → GATE → PARALLEL → SYNTHESIZE → CHALLENGE (conditional) → VERIFY → REFINE (conditional) → OUTPUT`

Print `[Step N/9] STEP_NAME` before each step.

## Step-by-Step Execution

### Step 0 — TRIAGE

Print: `[Step 0/9] TRIAGE`

**No agent spawning — orchestrator only.** Fast pre-check before analysis.

#### 0a. Ambiguity Resolution

Count unstated dimensions (technology, version, context, scope, constraints). If **3+ dimensions are unstated**, infer the most likely values from: codebase manifests, current directory context, and the query itself. Log inferences visibly. Never ask the user — decide based on available context.

#### 0b. Trivial Query Fast-Path

If the query is a single-hop factual question with an unambiguous answer (e.g., "What is the default port for PostgreSQL?"), answer directly from model knowledge without spawning agents. Flag as `trivial_bypass` in output.

### Step 1 — ANALYZE

Print: `[Step 1/9] ANALYZE`

**No agent spawning — orchestrator only.** Five sub-tasks:

#### 1a. Classify Complexity

See [workflow-engine.md](references/workflow-engine.md) for the full classification matrix. Assign the **highest matching level** across all dimensions. If any dimension is `complex`, the query is complex.

#### 1b. Define Success Criteria

```
expected_output_type: factual_answer | comparison | architecture_decision | how_to
must_answer: [list of questions the final answer MUST address]
must_include: [code_example | version_info | trade_offs — based on query type]
source_priority: academic | product | architecture (see workflow-engine.md for tier weighting)
```

These criteria become the **explicit checklist** for Step 7 VERIFY.

#### 1c. Decompose (moderate/complex queries)

**Moderate queries** — Entity tagging:
```
sq_001: {question} | target: websearch | entities: [known: X, unknown: Y] | depends_on: [] | priority: 1
sq_002: {question} | target: docs     | entities: [known: X, unknown: Z] | depends_on: [] | priority: 1
```

**Complex queries** — Constraint-based decomposition. See [workflow-engine.md](references/workflow-engine.md) for full format including sufficiency check.

**Per sub-question source priority**: Each sub-question may override the global `source_priority` when its nature differs from the main query.

#### 1d. Reformulate Query (moderate/complex queries)

Rewrite the user question as a single, unambiguous, self-contained **canonical research question**. This eliminates implicit assumptions and creates a stable anchor for all downstream agents. Inspired by OpenAI Deep Research Instruction Agent and MA-RAG disambiguation pattern.

For `simple` queries: skip (the original question is sufficient).

The canonical question must:
1. Make all implicit assumptions explicit (technology, version, context)
2. Include scope boundaries (what is IN and OUT of scope)
3. Specify the expected output type from Step 1b

Example: User asks "how do I handle auth?" → Canonical: "What are the recommended approaches for implementing authentication in a Next.js 15 App Router application using server components, considering session management, OAuth providers, and middleware-based route protection?"

#### 1e. Enrich Query (all complexity levels)

Convert the canonical question (or original for simple) into explicit per-agent research instructions. See [workflow-engine.md](references/workflow-engine.md) for enrichment templates by complexity level.

Each instruction should be actionable, scoped, and reference the `must_answer` items it serves.

#### 1f. Plan Answer Shape (complex queries only)

Outline the expected answer structure before dispatching agents. See [workflow-engine.md](references/workflow-engine.md) for details.

#### 1g. Validate Plan

Before dispatching any agent, verify:
1. Does every `must_answer` item map to at least one agent instruction?
2. Are there redundant sub-questions that would produce overlapping agent work?
3. For moderate/complex: does the dependency graph have cycles?

If validation fails, fix the plan before proceeding.

### Step 2 — RESEARCH (agent-websearch)

Print: `[Step 2/9] RESEARCH (web)`

**Always runs first.** Agent topology varies by complexity:

**Simple/Moderate** — Single agent-websearch with the template from agent-protocols.md (includes dual-perspective search).

**Complex** — Two agent-websearch in parallel (SINGLE message): one supportive angle (best practices, official guidance), one critical angle (limitations, problems, alternatives).

**Dual-perspective search protocol** (all levels): For each of the top 3 findings, the agent runs ONE additional search with the pattern `"{finding}" limitations OR problems OR alternatives`.

Wait for all agents to complete before proceeding.

#### 2z. Output Structure Validation (MAST coordination check)

Before proceeding, verify each agent return against the MAST coordination failure pattern (36.9% of multi-agent failures are coordination breakdowns):

1. Does the output contain ALL requested sections from the prompt template?
2. Does the output address the specific sub-questions assigned (cross-reference `must_answer` items)?
3. Is the output within the output budget (1,000 tok for research/explore, 800 tok for docs)?

If any check fails → log as `coordination_gap: {agent, missing_section, unaddressed_question}` and pass the specific gap to Step 3 for downstream agents to address. This catches silent failures early rather than at Step 7.

### Step 3 — GATE (compress + boundary check + route)

Print: `[Step 3/9] GATE`

**No agent spawning — orchestrator only.** Four sub-tasks:

#### 3a. Compress to Typed Handoff Object

Convert Step 2 output into a structured handoff — NOT a prose summary. Use the **Typed Handoff Format** from [workflow-engine.md](references/workflow-engine.md). Target: 300-500 tokens total. For complex queries with 2 websearch agents, merge both outputs into a single handoff, noting where agents converged or diverged.

#### 3b. Boundary Check

Verify Step 2 output quality:
1. **Query coverage**: Did the search address all sub-questions / key aspects?
2. **Completion signal validity**: Is the coverage level warranted by the evidence count?
3. **Context preservation**: Does the compressed handoff still reference the original query intent?

If any check fails, note the gap for Step 4 agents to address specifically.

#### 3c. Route — Detect Conditions

- **Codebase exists?** → Check for `.git` first. If found → `codebase_exists = true`. Then check manifests for library extraction.
- **Libraries identified?** → From handoff `libraries` field + codebase manifest. If any → spawn DOCUMENT in Step 4.

See [workflow-engine.md](references/workflow-engine.md) for codebase detection and library extraction details.

#### 3d. Early-Exit Check

See [workflow-engine.md](references/workflow-engine.md) for the full early-exit conditions and exceptions.

### Step 4 — PARALLEL (EXPLORE + DOCUMENT)

Print: `[Step 4/9] PARALLEL (explore + document)`

Spawn agent-explore + agent-docs in a **SINGLE message** (templates in agent-protocols.md). Pass the typed handoff object from Step 3a. Only spawn each if its condition is met. Wait for all to complete.

#### 4z. Output Structure Validation (MAST coordination check)

Apply the same validation as Step 2z to Step 4 agent returns. For each agent, verify required sections and sub-question coverage. Log any `coordination_gap` for Step 5 to surface honestly rather than silently drop.

### Step 5 — SYNTHESIZE

Print: `[Step 5/9] SYNTHESIZE`

Synthesis is organized into **3 passes** to reduce cognitive load and prevent dual-task degradation:

#### Pass 1 — Compress & Merge

Structure all agent outputs and resolve overlaps before generating the answer.

**5a. Compress Step 4 Outputs** into a unified input:
```
web_research: {step_3a_typed_handoff — already compressed}
codebase: {compress Step 4 EXPLORE output into claims with file:line refs}
docs: {compress Step 4 DOCUMENT output into API facts with ctx7 refs}
```
Target: 300-500 tokens per source.

**5b. Conflict Resolution & Deduplication**
1. Official docs (Step 4 DOCUMENT) > Web research (Step 2) > Codebase patterns (Step 4 EXPLORE)
2. Exception: intentional codebase deviation → note both approaches
3. Triangulation: 2/3 concordant sources = **corroborated**, otherwise = **contested**
4. If multiple phases found the same information, use the most authoritative version and cite once
5. Do NOT silently resolve contradictions — list each position with its source in `Contested Claims`

**5c. Input Coverage Check** — Verify that **each active agent's output** is represented. If an agent returned findings not reflected in the answer, either include them or explicitly note why they were excluded.

#### Pass 2 — Generate (citation-first)

Write the synthesis with inline citations from the start.

**5d. Citation-First Generation** — Every claim must cite its source **inline** before or alongside the claim. Do NOT write claims first and add sources after.

**5e. Source Credibility Weighting** — Weight claims by source authority tier. See [workflow-engine.md](references/workflow-engine.md) for the full tier table (T1=1.0, T2=0.7, T3=0.4, T4=0.2).

#### Pass 3 — Calibrate & Audit

Score confidence and verify citation integrity as separate tasks.

**5f. Confidence Calibration Correction** — See [workflow-engine.md](references/workflow-engine.md) for the full calibration correction rules, including source diversity scoring and niche topic cap.

**5g. Confidence Scoring** — See [workflow-engine.md](references/workflow-engine.md) for the full confidence scoring table. Assign: `high` | `medium` | `low` with basis. Includes trajectory signals (search depth, agent convergence, challenge survival rate).

**5h. Citation Audit** — Run a dedicated pass focused solely on citation integrity:
1. Segment the synthesis into individual factual claims
2. For each claim, verify it maps to a specific source URL from agent output
3. Flag any claim that lacks a traceable source as `[unsourced]`
4. Verify no source URL was fabricated (every URL must come from a tool result)

This separates citation quality from synthesis quality, preventing dual-task degradation.

### Step 6 — CHALLENGE (adversarial review)

Print: `[Step 6/9] CHALLENGE`

**Signal-conditional step.** Run ONLY when at least one trigger fires:
1. `simple` queries → **always skip** (go directly to Step 7)
2. `moderate` queries → run only if Step 5 synthesis contains `contested_claims > 0` OR `confidence == low`
3. `complex` queries → **always run** (safety net for high-stakes research)

If no trigger fires for moderate queries, skip to Step 7.

Independent adversarial review of synthesis claims. The orchestrator that wrote the synthesis cannot objectively evaluate it — an independent agent with a fresh context window produces uncorrelated evaluation. Note: ICLR 2025 research shows adversarial debate does not consistently beat majority voting — this step is retained for high-signal cases only.

#### 6a. Extract Top Claims

From the Step 5 synthesis, extract the **3-5 highest-impact claims** — those that most directly answer the `must_answer` items from Step 1. Include their cited sources but NOT the full synthesis draft.

#### 6b. Spawn Challenge Agent

Spawn agent-websearch with the challenge protocol from agent-protocols.md. The agent receives ONLY the claims and their sources — NOT the full draft.

#### 6c. Integrate Challenge Results

For each challenged claim:
- **Confirmed:** Claim withstood challenge — mark as `corroborated`
- **Weakened:** Counter-evidence found — downgrade confidence, add nuance
- **Refuted:** Strong counter-evidence — remove or reframe, note contradiction

Update the synthesis before proceeding to Step 7.

### Step 7 — VERIFY

Print: `[Step 7/9] VERIFY`

Verification approach varies by complexity:
- **Simple/Moderate:** Orchestrator self-check (sub-tasks 7a-7d below)
- **Complex:** Spawn an **independent evaluator agent** that receives the success criteria from Step 1 and the synthesis draft, but has NOT participated in generating it. This enforces generator-evaluator separation.

#### 7a. Completeness Score

Score against Step 1 success criteria. See [workflow-engine.md](references/workflow-engine.md) for the scoring formula.

#### 7b. Invariant Validation

Run deterministic checks against the pipeline invariant list (see [workflow-engine.md](references/workflow-engine.md)):

1. Every factual claim has a source URL from a tool result
2. Every `must_answer` item has a corresponding answer
3. No T3-T4-only claims included without `needs verification` flag
4. Time-sensitive claims have sources from the current or previous year
5. No `must_include` items missing (code examples, version info, trade-offs)
6. If `expected_output_type` is `how_to` and codebase exists: referenced imports, functions, and framework versions exist in the codebase
7. Each active agent's output is represented in synthesis
8. Per-claim grounding: every individual factual claim maps to a source from agent output

Any invariant failure is logged as a specific, actionable gap for Step 8.

#### 7c. Noise Check

Count claims in the synthesis that do not map to any `must_answer` item. If noise ratio > 30%, flag tangential claims for removal in Step 8.

#### 7d. Evaluator Agent (complex queries only)

For complex queries, spawn a general-purpose evaluator agent with the evaluator protocol from agent-protocols.md. It receives success criteria + synthesis draft + invariant results. It has NOT generated the synthesis. Returns: pass/fail per `must_answer` item, gaps, recommended action.

**Decision:**
- `simple` → always skip to Step 9
- `completeness >= 0.75` AND no critical invariant failures → Step 9
- `completeness < 0.75` OR critical invariant failures → Step 8

For each gap, record: `gap_description`, `target_agent`, `target_query`, `severity`.

### Step 8 — REFINE (conditional, max 1 iteration)

Print: `[Step 8/9] REFINE`

Only runs if Step 7 identified gaps. Never for `simple` queries. **Max 1 iteration for moderate, max 2 for complex with verify < 0.5.**

#### Anti-Sycophancy Protocol (reinforced)

Sycophancy compounds over multi-turn interactions (arXiv:2505.23840). To prevent the refinement agent from rubber-stamping the current draft:

1. **Hard reset prompt**: Every refinement agent receives: "You have NO knowledge of any prior analysis. Start fresh from the sources below."
2. **Gap-only, never the draft**: Send the refinement agent the **gap description + relevant sources** only — NEVER the full synthesis draft. The agent generates its own independent content for the gap.
3. **Anonymization**: In the "Already Known" section, present claims WITHOUT source URLs first. The agent evaluates on substance before knowing source authority.
4. **Orchestrator merges**: The refinement agent returns its independent findings. The orchestrator merges them into the existing synthesis — the refinement agent never sees or revises the draft directly.

#### Stopping Criterion

If the refinement produces < 5% semantic change from the current synthesis, treat as converged and stop. Report remaining gaps honestly rather than loop.

Spawn ONLY agents whose domain matches identified gaps. If multiple gaps target different agents, spawn all in a SINGLE message for parallel execution.

### Step 9 — OUTPUT

Print: `[Step 9/9] OUTPUT`

**Persist to memory** (moderate/complex queries only, if findings are novel + reusable + medium+ confidence):
- Write to `~/.claude/projects/{project-path}/memory/research-{topic-slug}.md`
- Update MEMORY.md index
- Include metadata: `query_strategies`, `reliable_sources`, and `pipeline_performance` (see [workflow-engine.md](references/workflow-engine.md) for full schema)

**Deliver the response** using the [Output Format from synthesis-template.md](~/.claude/skills/_shared/synthesis-template.md).

## Hard Rules

Cross-step constraints not stated in individual step descriptions. For agent boundaries and output budgets, see [agent-boundaries.md](@~/.claude/skills/_shared/agent-boundaries.md).

1. **Delegation depth = 1**: Subagents NEVER spawn sub-subagents. Gaps return to orchestrator.
2. **Generator-evaluator separation**: For complex queries, Step 7 uses an independent evaluator agent. The orchestrator does not self-evaluate its own synthesis.
3. **Anti-sycophancy**: Refinement agents receive gap descriptions and sources only — never the full draft. Claims anonymized before authority reveal. Hard reset prompt on every refinement agent.
4. **Refinement cap**: Max 1 iteration (moderate), max 2 (complex with verify < 0.5). Never for simple. Stop if < 5% change.
5. **Graceful degradation**: If any agent fails, continue with available data and note the gap.
6. **Selective persistence**: Only novel, reusable, medium+ confidence findings. Include pipeline performance metadata.
7. **Progress indicators**: Print `[Step N/9]` headers. Never skip.
8. **No TeamCreate**: Use simple Agent tool spawning only.
9. **Narrative reframing**: All inter-agent context uses third person ("Web research found X", not "I found X").

## Error Handling

If any agent returns empty, times out, or fails: continue with available data and note the gap in the relevant output section. If all agents fail, return whatever is available with an honest disclaimer. See [workflow-engine.md](references/workflow-engine.md) for the complete error matrix.

## References

- [Workflow Engine](references/workflow-engine.md) — classification matrix, scoring formulas, credibility tiers, invariant list, confidence scoring, error matrix, performance characteristics, query enrichment templates
- [Agent Protocols](references/agent-protocols.md) — prompt templates and typed handoff format for each agent
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — CAN/CANNOT table, call budgets
- [Synthesis Template](~/.claude/skills/_shared/synthesis-template.md) — shared output format reference
