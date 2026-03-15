# Research Notes — Multi-Agent Workflow Orchestration

Synthesized from web research (Exa), Anthropic documentation, and industry frameworks (Feb-March 2026).

## 1. Recommended Orchestration Patterns

### Anthropic's Official Patterns (from "Building Effective Agents")

Anthropic identifies 5 core workflow patterns, in order of increasing complexity:

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Prompt Chaining** | Sequential steps, each LLM call processes the previous output, with optional gates between steps | Fixed, well-defined subtasks |
| **Routing** | Classify input and direct to specialized handlers | Distinct categories requiring separate handling |
| **Parallelization** | Run multiple LLM tasks simultaneously (sectioning or voting), aggregate results programmatically | Speed gains or when multiple perspectives increase confidence |
| **Orchestrator-Workers** | Central LLM dynamically breaks down tasks and delegates to workers, synthesizes results | Complex tasks where subtasks can't be predicted upfront |
| **Evaluator-Optimizer** | One LLM generates, another evaluates in iterative loops | Clear evaluation criteria, iterative refinement adds value |

**Key principle:** "Add complexity only when it demonstrably improves outcomes."

### 11 Industry Patterns (from AskAiBrain 2026 Guide)

The most relevant to meta-code:

1. **Pipeline (Sequential)** — Agents process outputs linearly. Best for well-defined sequential steps.
2. **Fan-out/Fan-in (Parallel)** — Task distributed to multiple parallel agents; aggregator synthesizes results.
3. **Supervisor (Centralized)** — Central supervisor analyzes requests and routes to specialized agents.

**meta-code maps to: Adaptive Pipeline + Fan-out/Fan-in hybrid with Evaluator-Optimizer gate.**

### Google ADK Multi-Agent Patterns

Google's ADK uses `AgentTool` to wrap agents as tools. Unlike OpenAI's handoffs (which transfer control entirely), AgentTool keeps the coordinator in charge. This matches Claude Code's Agent tool model.

## 2. Adaptive Orchestration (NEW — 2026)

### Difficulty-Aware Agentic Orchestration (DAAO)

Source: arXiv:2509.11079, updated Feb 2026.

Core problem: static workflows "either over-process simple queries or underperform on complex ones."

Three modules:
1. **VAE difficulty predictor** — predicts query complexity before agents run.
2. **Modular operator allocator** — selects which agent roles to activate.
3. **Cost/performance-aware LLM router** — assigns cheaper models to easy subtasks.

**Applied to meta-code as:** Step 0 CLASSIFY — determines pipeline depth before dispatching.

### Self-Evolving Workflows (SEW)

Source: arXiv:2505.18646.

Workflows that modify themselves based on performance. For code generation on LiveCodeBench, self-evolution produced 33% improvement over baseline.

### AgentConductor

Source: arXiv:2602.17100.

Constructs task-adapted DAG topologies whose density scales with inferred task difficulty.

**Applied to meta-code as:** Complex query decomposition into sub-question DAGs.

## 3. Quality Gates and Verification (NEW — 2026)

### Verification Gate Pattern

Source: Vadim.blog, Feb 2026.

A dedicated read-only verifier agent placed after every generative agent. Key finding from Google's 2025 DORA Report: 90% of AI-added lines were accepted when there was no separate verification step.

**Applied to meta-code as:** Step 6 VERIFY — completeness scoring and gap identification.

### LLM-as-Judge Guidelines

Source: Agent Factory / Panaversity, March 2026.

| Check Type | LLM Judge | Deterministic |
|---|---|---|
| JSON format validity | No | Yes (schema validator) |
| Response helpfulness | Yes | No |
| Code correctness | No | Yes (run tests) |
| Factual accuracy | Maybe | Cross-check known facts |
| Writing quality / safety | Yes | No |

**Applied to meta-code as:** Step 6 uses structured scoring rather than subjective evaluation.

### Confidence Scoring Pipeline

Source: Modexa, Feb 2026.

Confidence should be structural, not emotional. Produce one of three verdicts: Act, Ask, or Abstain. Based on composite signals: source count, inter-agent agreement, tool-use outcomes.

**Applied to meta-code as:** Confidence attribution (high/medium/low) in synthesis with explicit basis.

### VMAO Verification

Source: arXiv:2603.11445, March 2026.

Dedicated `ResultVerifier` agent that assesses completeness on 0-1 scale, lists gaps and contradictions, and emits recommendations.

**Applied to meta-code as:** Step 6 completeness scoring with gap identification for targeted refinement.

## 4. Iterative Refinement Patterns (NEW — 2026)

### Self-Critique Pattern

Source: agentpatterns.tech, March 2026.

Draft → critique → revise loop with constraints:
- Critique uses a fixed schema (structured risks + required changes).
- Revisions are constrained in scope.
- Audit log records what changed between versions.

### Reflexion with External Evidence

Source: Towards AI Part 3, March 2026.

When a critic identifies a gap that requires external information, it triggers a new tool call rather than just rephrasing. Separates factual gaps (require retrieval) from reasoning gaps (addressable by reflection).

**Applied to meta-code as:** Step 7 REFINE distinguishes gap types and spawns only the relevant agent.

### Critical Failure Modes

- **Self-consistency trap**: Models defend wrong answers across reflection rounds.
- **Sycophantic reflection**: Models agree with user framing rather than critiquing.
- **Quality regression**: Over-reflection introduces hedging or errors.
- **Verification Trap**: 24.3% of failed agent runs contained the correct fix but timed out in verification.

**Safeguard applied:** Maximum 1 refinement iteration. If issues don't decrease, stop and report.

## 5. Source Triangulation and Contradiction Detection (NEW — 2026)

### Multi-Agent Fact-Checking (MAFC)

Source: Scientific Reports, March 2026.

Each agent gets a unique information source (not shared pool). Credibility scoring calculates weighted score based on agent judgment and source authority.

### Fork-Merge for Fact Triangulation

Source: Zylos Research, March 2026.

Multiple agent instances independently assess the same claim, then structured debate converges on a verdict. Risk: "sycophantic convergence" — agents socially aligning.

### Contradiction Taxonomy (ContraGen)

Source: arXiv:2510.03418.

Types: definitional, temporal, quantitative, procedural contradictions.

**Applied to meta-code as:**
- Each agent output includes `contradictions[]` field.
- Triangulation threshold: 2/3 concordant sources = corroborated, otherwise = contested.
- Contradictions surfaced explicitly in output, never silently resolved.

## 6. Context Passing and Memory (UPDATED — 2026)

### Key Principles (unchanged)

- **Summarize, don't forward raw output.** Compress to <500 words.
- **Extract structured keys.** From Phase 1: key findings, library names, versions, contradictions.
- **Information flow is directional.** Step 2 → Steps 3/4. Steps 3 and 4 don't need each other.
- **The orchestrator sees all outputs.** It synthesizes during Step 5.

### Agent Memory Survey

Source: arXiv:2603.07670, March 2026.

Three memory dimensions:
- **Temporal**: working (context window), episodic (timestamped), semantic (abstracted), procedural (reusable skills)
- **Representational**: context-resident text, vector stores, structured stores, executable repos
- **Control**: heuristic, prompted, learned

Production pattern: Context + Retrieval Store — working memory in context, long-term in external stores.

### Memory Failure Modes

Source: Microsoft Research, ICLR 2026.

- **Brevity bias**: summarization silently drops domain insights.
- **Context collapse**: iterative rewriting erodes details over time.
- Solution: evolve context rather than compress it; preserve structure through versioned updates.

**Applied to meta-code as:**
- Step 1 CACHE CHECK retrieves prior research from persistent memory.
- Step 8 PERSIST writes novel findings with metadata (topic, date, confidence, sources).
- Only `reference` type memories, only when novel and reusable.

## 7. Query Decomposition (NEW — 2026)

### VMAO DAG Decomposition

Source: arXiv:2603.11445, March 2026.

Sub-questions as DAG nodes with metadata: id, question, agent_type, depends_on, priority, context_from_deps.

Results: completeness 4.2 vs. 3.1, source quality 4.1 vs. 2.6 compared to single-agent baseline.

Cost: 8.5x token usage — worth it for complex queries, wasteful for simple ones.

### DeAR: Decompose-Analyze-Rethink

Source: USTC, Oct 2025.

Builds reasoning tree iteratively. Decompose → Analyze leaves → Rethink (propagate corrections upward).

### ACL 2025 — Question Decomposition for RAG

Decomposing multi-hop queries and retrieving per sub-question outperforms single-query RAG.

**Applied to meta-code as:** Step 0 CLASSIFY decomposes complex queries into 2-4 sub-questions with dependency metadata.

## 8. Output Optimization (NEW — 2026)

### 4D-ARE Attribution Framework

Source: Tencent, arXiv:2601.04556, Jan 2026.

Four output dimensions: Results, Process, Support, Long-term implications.

### Confidence Attribution in Output

Source: Modexa + Zylos Research synthesis.

Recommended schema includes: answer, confidence score, confidence basis, contested claims, sources, gaps, recommended follow-up.

**Applied to meta-code as:** Output format includes Confidence level with basis, Contested Claims section, Follow-up section.

### Production Pipeline Design

Source: Wasowski 27-agent pipeline, March 2026.

Patterns:
- Separate specialist agents from synthesis agents.
- Use intermediate structured handoffs (not free-text) between stages.
- Treat formatting as a final dedicated phase.

## 9. Anti-Patterns to Avoid

1. **God Agent** — Single agent doing everything. Solution: specialize.
2. **Agent Loops** — Agents calling each other in cycles. Solution: strict directional flow.
3. **Redundant Work** — Multiple agents searching for the same information. Solution: clear domain boundaries.
4. **Context Explosion** — Passing too much data between agents. Solution: summarize before passing.
5. **Premature Parallelism** — Running all agents simultaneously when some need prior results. Solution: sequence Step 2, then parallelize.
6. **Over-Orchestration** — Using teams/complex coordination for simple tasks. Solution: use Agent tool directly.
7. **Ignoring Failures** — Proceeding without noting what data is missing. Solution: always report gaps.
8. **Duplicate Synthesis** — Repeating same info across output sections. Solution: deduplicate in final synthesis.
9. **Silent Contradiction Resolution** (NEW) — Averaging conflicting information without surfacing it. Solution: triangulate and surface.
10. **Verification Trap** (NEW) — Looping indefinitely in quality checks. Solution: max 1 refinement, then report gaps.
11. **Over-decomposition** (NEW) — Breaking simple queries into sub-questions. Solution: classify first, decompose only complex queries.
12. **Memory Pollution** (NEW) — Storing every result in memory. Solution: persist selectively (novel, reusable, medium+ confidence).

## Sources

- [Building Effective Agents — Anthropic](https://www.anthropic.com/research/building-effective-agents) (Dec 2024)
- [The 11 Multi-Agent Orchestration Patterns — AskAiBrain](https://www.askaibrain.com/en/posts/11-multi-agent-orchestration-patterns-complete-guide) (Jan 2026)
- [Difficulty-Aware Agentic Orchestration DAAO](https://arxiv.org/abs/2509.11079) (Feb 2026)
- [AgentConductor: Topology Evolution for Code Generation](https://arxiv.org/html/2602.17100v1) (Feb 2026)
- [SEW: Self-Evolving Agentic Workflows](https://arxiv.org/abs/2505.18646) (2025)
- [Dynamic Multi-Agent Orchestration — PromptLayer](https://blog.promptlayer.com/multi-agent-evolving-orchestration/) (Feb 2026)
- [Verified Multi-Agent Orchestration VMAO](https://arxiv.org/html/2603.11445v1) (March 2026)
- [AI Agent Reflection and Self-Evaluation — Zylos](https://zylos.ai/research/2026-03-06-ai-agent-reflection-self-evaluation-patterns) (March 2026)
- [Self-Critique Agent Pattern — agentpatterns.tech](https://www.agentpatterns.tech/en/agent-patterns/self-critique-agent) (March 2026)
- [Agent Control Patterns: Reflection — Towards AI](https://pub.towardsai.net/agent-control-patterns-part-2-reflection-a-simple-way-to-improve-answer-quality-9d039cfd5da8) (March 2026)
- [Agent Control Patterns: Reflexion — Towards AI](https://pub.towardsai.net/agent-control-patterns-part-3-reflexion-when-review-triggers-research-f56447b0ae1e) (March 2026)
- [Self-Reflection and Critique — Arun Baby](https://arunbaby.com/ai-agents/0039-self-reflection-and-critique/) (Feb 2026)
- [The Agent That Says No: Verification Gate — Vadim](https://vadim.blog/verification-gate-research-to-practice) (Feb 2026)
- [Confidence Pipeline — Modexa](https://medium.com/@Modexa/confidence-isnt-a-feeling-it-s-a-pipeline-c113aee921ca) (Feb 2026)
- [LLM-as-Judge — Agent Factory](https://agentfactory.panaversity.org/docs/Turing-LLMOps-Proprietary-Intelligence/evaluation-quality-gates/llm-as-judge) (March 2026)
- [Memory for Autonomous LLM Agents Survey](https://arxiv.org/html/2603.07670v1) (March 2026)
- [Agentic Context Engineering — Microsoft Research](https://www.microsoft.com/en-us/research/publication/agentic-context-engineering-evolving-contexts-for-self-improving-language-models/) (ICLR 2026)
- [Context-Aware Multi-Agent Frameworks — Google Developers](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/) (Dec 2025)
- [Multi-Agent Fact-Checking — Scientific Reports](https://www.nature.com/articles/s41598-026-41862-z) (March 2026)
- [Fork-Merge Patterns — Zylos](https://zylos.ai/research/2026-03-10-ai-agent-fork-merge-patterns) (March 2026)
- [ContraGen: Contradiction Detection](https://arxiv.org/html/2510.03418v1) (2025)
- [4D-ARE Attribution Framework](https://arxiv.org/pdf/2601.04556) (Jan 2026)
- [Query Decomposition for RAG — ACL 2025](https://aclanthology.org/2025.acl-srw.32/) (2025)
- [Best Practices for Claude Code — Anthropic Docs](https://docs.anthropic.com/en/docs/claude-code/best-practices)
