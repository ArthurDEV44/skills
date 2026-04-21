# Bibliography — Multi-Agent Research Pipeline

**Design-time reference only** — not loaded during pipeline execution. These sources document the research basis for pipeline design decisions. For the pipeline itself, see [SKILL.md](../SKILL.md).

Curated sources informing the meta-code pipeline design. Organized by tier: T1 sources are directly implemented in the pipeline, T2 sources provide supporting patterns.

---

## T1 — Directly Implemented

Sources whose specific patterns, formulas, or protocols are directly encoded in the pipeline.

### Orchestration Architecture

- [Building Effective Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents) — Six composable patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. Anti-pattern: framework over-reliance obscuring underlying logic.
- [Multi-Agent Research System — Anthropic Engineering](https://www.anthropic.com/engineering/multi-agent-research-system) — Lead agent + parallel subagents, CitationAgent post-processing, 90.2% improvement with heterogeneous model pairing. Effort-scaling heuristics: 1 agent for facts, 2-4 for comparisons, 10+ for complex research.
### T2 Practitioner — Boris Cherny (Claude Code creator)

- [Boris Cherny — Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny) — "A good plan one-shots the implementation"; glob+grep beats RAG for code search; adversarial review wave + challenge wave.
- [Boris Cherny — Every.to Transcript](https://every.to/podcast/transcript-how-to-use-claude-code-like-the-people-who-built-it) — Uncorrelated context windows yield better results; plan mode 2-3x success rate for complex tasks.
- [How Boris Uses Claude Code](https://howborisusesclaudecode.com) — Git worktrees as structural primitive, PostCompact hook for context continuity, staff engineer plan review, `/grill` for adversarial challenge, scheduled agent loops.

### Context Engineering

- [Effective Context Engineering — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Just-in-time loading, progressive disclosure, subagents return 1,000-2,000 token condensed summaries. Context rot from irrelevant tokens.
- [Context Engineering — Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) — 200-500 token typed objects between agents; restorable compression (drop content, preserve pointers).
- [Context Engineering — Microsoft Azure SRE Agent](https://techcommunity.microsoft.com/blog/appsonazureblog/context-engineering-lessons-from-building-azure-sre-agent/4481200/) — Typed context slices; "share memory by communicating, not communicate by sharing memory."
- [Session Memory / Context Engineering — OpenAI Cookbook](https://cookbook.openai.com/examples/agents_sdk/session_memory) — Per-handoff compressed summaries; trim-and-compress for long pipelines. Informs Step 5a compression.

### Source Triangulation & Citation

- [Cite Before You Speak — arXiv:2503.04830](https://arxiv.org/html/2503.04830) — Citation-first generation improves grounding by +13.83%. Implemented in Step 5b.
- [Dual-Perspective Claim Verification — arXiv:2602.18693](https://arxiv.org/abs/2602.18693) — Query claim AND its negation; +2-10% accuracy. Implemented in Step 2 dual-perspective protocol.
- [The Confidence Dichotomy — arXiv:2601.07264](https://arxiv.org/abs/2601.07264) — Evidence tools cause overconfidence. Implemented in Step 5d calibration correction.

### Query Decomposition

- [RT-RAG — arXiv:2601.11255](https://arxiv.org/html/2601.11255v1) — Entity tagging (known vs unknown); +7% F1. Implemented in moderate query decomposition.
- [C2RAG: Constraint-Based GraphRAG — arXiv:2603.14828](https://arxiv.org/html/2603.14828) — Atomic constraint triples with sufficiency check; suppresses retrieval drift. Implemented in complex query decomposition.

### Anti-Sycophancy & Verification

- [CONSENSAGENT — ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1141.pdf) — Agents commit to independent positions before deliberation. Implemented in Step 8 anti-sycophancy protocol.
- [Anonymization for Bias-Reduced Reasoning — arXiv:2510.07517](https://arxiv.org/html/2510.07517) — Strip identity markers during deliberation; conformity > obstinacy in most LLMs. Implemented in Step 8 refinement anonymization.
- [Peacemaker or Troublemaker — arXiv:2509.23055](https://arxiv.org/html/2509.23055v1) — Sycophancy lowest in round 1, grows progressively. Informs Step 6 CHALLENGE placement (early, not after consensus).

### Failure Modes

- [MAST Taxonomy — arXiv:2503.13657](https://arxiv.org/abs/2503.13657) — 14 failure modes across 3 categories (NeurIPS 2025). Implemented in input coverage check (Step 5h) and invariant validation (Step 7b).
- [17x Error Trap — Towards Data Science](https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/) — Unstructured multi-agent systems amplify errors 17.2x. Informs max 1 refinement iteration.

### Quality Gates & Invariant Validation

- [Cursor Agent Best Practices — Cursor Blog](https://cursor.com/blog/agent-best-practices) — TDD loop as correctness signal, conversation lifecycle discipline, verifiable quality gates. Informs pipeline invariant validation (Step 7b).
- [Codex Subagents — OpenAI Developers](https://developers.openai.com/codex/subagents) — Depth-bounded delegation (max_depth=1), wait-all collector, read-only sandbox mode. Informs Rule 9 delegation depth.
- [Devin Annual Performance Review — Cognition AI](https://cognition.ai/blog/devin-annual-performance-review-2025) — Clear upfront requirements are the strongest predictor of agent success. Informs Step 0 TRIAGE and Step 1e plan validation.

### Generator-Evaluator Separation

- [Harness Design for Long-Running Apps — Anthropic (Mar 2026)](https://www.anthropic.com/engineering/harness-design-long-running-apps) — Generator-Evaluator separation is "a strong lever" for quality. Sprint Contracts formalize typed success criteria. Assumption decay: scaffolding encodes model-capability bets that go stale. Informs Step 7d evaluator agent and Assumption Decay Notice.
- [Demystifying Evals for AI Agents — Anthropic (Jan 2026)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — pass@k vs pass^k; code-based > model-based graders; grade outcomes not paths. Informs pipeline evaluation strategy.

### Triage & Clarification

- [Deep Research API with Agents SDK — OpenAI Cookbook](https://developers.openai.com/cookbook/examples/deep_research_api/introduction_to_deep_research_api_agents) — 4-agent pipeline: Triage → Clarifying → Instruction → Research. Clarifying Agent detects ambiguity; Instruction Agent converts queries into precise research briefs with format specs and source hierarchy. Informs Step 0 TRIAGE and Step 1d query enrichment.

### Scaling & Parallelization

- [Towards a Science of Scaling Agent Systems — Google/MIT (arXiv:2512.08296)](https://arxiv.org/abs/2512.08296) — Centralized orchestration +80.9% on parallelizable tasks; 4.4x error amplification vs 17.2x for peer-to-peer. Capability saturation at ~45% single-agent baseline. Informs parallel websearch for complex queries (Step 2).
- [Cursor 2.2 Multi-Agent Judging — Cursor](https://forum.cursor.com/t/cursor-2-2-multi-agent-judging/145826/24) — Automated evaluation after parallel agents complete; AI-assisted selection with reasoning. Informs independent evaluator pattern (Step 7d).

### Per-Claim Grounding

- [ADK Evaluation — hallucinations_v1 — Google](https://google.github.io/adk-docs/evaluate/criteria/) — Segmenter + Sentence Validator: segment response into claims, validate each against grounding context. Informs INV-8 per-claim source grounding and Step 5j citation audit.

---

T2 supporting research moved to [bibliography-extended.md](bibliography-extended.md) to reduce context load. Only T1 directly-implemented sources are kept here.
