# Bibliography — T2 Supporting Research

**Design-time reference only** — not loaded during pipeline execution. These sources inform design decisions but are not directly encoded as protocols. For T1 directly-implemented sources, see [bibliography.md](bibliography.md).

## Orchestration Patterns

- [Agent Orchestration — OpenAI Agents SDK](https://openai.github.io/openai-agents-python/multi_agent/) — Agents-as-tools vs handoffs; code-driven > LLM-driven for predictable pipelines. Hybrid recommendation: code-driven routing + LLM-driven reasoning.
- [8 Named Patterns — Google Developers Blog](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/) — Sequential, Coordinator, Parallel Fan-Out, Generator-Critic, Iterative Refinement, Composite.
- [12 Architecture Patterns — Google Cloud Architecture Center](https://docs.google.com/architecture/choose-design-pattern-agentic-ai-system) — Extends ADK 8 patterns with Swarm, ReAct, Human-in-the-Loop, Custom Logic. Diamond pattern (post-generation moderation) and Peer-to-peer handoff.
- [Multi-Agent Systems — Google ADK](https://google.github.io/adk-docs/agents/multi-agents/) — `escalate=True` for quality-gated loop termination; deterministic workflow primitives.
- [Practical Guide to Building Agents — OpenAI](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) — Start single-agent; 3-layer guardrails (input filtering, tool-use validation, human-in-the-loop).
- [Cursor 2.0 Changelog](https://cursor.com/changelog/2-0) — Git worktrees as isolation primitive; plan with one model, execute with another.
- [AdaptOrch — arXiv:2602.16873](https://arxiv.org/abs/2602.16873) — Topology-aware orchestration +12-23% over static baselines. Informs early-exit conditions.
- [Boris Cherny — Lenny's Newsletter](https://www.lennysnewsletter.com/p/head-of-claude-code-what-happens) — Clean codebases boost AI productivity; complete migrations fully.
<!-- Google/MIT Scaling, OpenAI Deep Research, Codex Subagents, Cursor Agent Best Practices: see bibliography.md T1 -->
- [Cursor Automations March 2026](https://releasebot.io/updates/cursor) — Event-driven background agents with memory across runs; invariant validation subagents; parallel agent judging.

## Quality & Evaluation

- [Demystifying Evals for AI Agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Code-based > model-based graders; pass@k vs pass^k; grade outcomes not paths.
- [Effective Harnesses for Long-Running Agents — Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Progress files, git checkpoints, JSON > Markdown for state.
- [VMAO — arXiv:2603.11445](https://arxiv.org/html/2603.11445v1) — ResultVerifier with 0-1 completeness scoring, gap identification.
- [Ragas Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) — Noise Sensitivity, Context Entities Recall, Tool Call Accuracy metrics.
- [ADK Evaluation — Google](https://google.github.io/adk-docs/evaluate/) — Tool trajectory matching (EXACT, IN_ORDER, ANY_ORDER), LLM-as-judge, User Simulation evaluation.
- [Evaluation Best Practices — OpenAI](https://platform.openai.com/docs/guides/evaluation-best-practices) — Trajectory-based > final-output-only; o3 as recommended grader; chain-of-thought before scoring.
<!-- Devin Annual Performance Review: see bibliography.md T1 -->

## Confidence & Calibration

- [Confidence Calibration via Multi-Agent Deliberation — arXiv:2404.09127](https://arxiv.org/abs/2404.09127) — Post-hoc calibration via simulated group deliberation.
- [OI-MAS: Confidence-Aware Routing — arXiv:2601.04861](https://arxiv.org/html/2601.04861v1) — State-dependent routing by agent confidence at each reasoning step.

## Advanced Decomposition

- [HopRAG — arXiv:2502.12442](https://arxiv.org/abs/2502.12442) — Retrieve-reason-prune for multi-hop expansion.
- [Exploration-Exploitation Query Decomposition — arXiv:2510.18633](https://arxiv.org/abs/2510.18633) — Bandit-framed retrieval; +35% document-level precision.

## Multi-Agent Debate Failure Modes

- [Talk Isn't Always Cheap — arXiv:2509.05396](https://arxiv.org/html/2509.05396v1) — Group accuracy declines over successive debate rounds; sycophancy causes strong models to yield.
- [Deliberative Dynamics — arXiv:2510.10002](https://arxiv.org/html/2510.10002v2) — Sycophancy is format-dependent; interaction format shapes behavior beyond prompt context.

## Skills & Output

- [Claude Code Skills — Official Docs](https://code.claude.com/docs/en/skills) — SKILL.md spec, frontmatter fields, progressive disclosure.
- [Agent Skills Open Standard — Anthropic](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills) — 3-layer progressive disclosure architecture. Adopted by GitHub Copilot, Cursor, OpenAI Codex, Gemini CLI.
- [Building Agents with Claude Agent SDK — Anthropic](https://claude.com/blog/building-agents-with-the-claude-agent-sdk) — Agentic search (grep/glob) > semantic search; isolated context windows.
- [Building a C Compiler with Parallel Claudes — Anthropic](https://www.anthropic.com/engineering/building-c-compiler) — 16-agent file-based task locking; emergent specialization; environment > prompting for parallel teams.

## Context Management

- [Windsurf Cascade — Codeium](https://windsurf.com/cascade) — Continuous behavioral inference via shared timeline of developer actions; implicit context vs explicit file tagging.
- [OpenAI Agents SDK Context — OpenAI](https://openai.github.io/openai-agents-python/context/) — Session trim-and-compress; per-issue mini-summaries at handoff boundaries.

## Boris Cherny — Additional Sources (2025-2026)

<!-- How Boris Uses Claude Code: see bibliography.md T2 Practitioner -->
- [53 Tips — blog.enkr1.com](https://blog.enkr1.com/boris-cherny-claude-code-workflow/) — Opus 4.5 + thinking mode, verification as #1 tip, /btw for side-chain questions.
- [Google Agents White Paper](https://www.marktechpost.com/2025/05/06/google-releases-76-page-whitepaper-on-ai-agents-a-deep-technical-dive-into-agentic-rag-evaluation-frameworks-and-real-world-architectures/) — Typed handoffs, source credibility scoring, agent specialization > generalization.
