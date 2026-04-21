# Research Sources — Academic Papers and Industry Sources

This document lists the research backing each improvement to the meta-debug pipeline.

---

## DDI Circuit Breaker (Max 2 Fix Attempts)

- **Measuring and Mitigating Debugging Effectiveness Decay** — Scientific Reports / Nature, Dec 2025
  - LLMs lose 60-80% of debugging capability within 2-3 attempts
  - DDI model: E(t) = E_0 * e^(-lambda*t), lambda ~0.5-0.8
  - Strategic fresh start outperforms continued iteration past inflection point
- **The Debugging Decay Index: Rethinking Debugging Strategies** — arXiv, Jul 2025
- **TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging** — arXiv, Oct 2025
  - Tree search explores fix strategies; avoids greedy refinement traps

## Reproduce-First Gate

- **From Incident to Test: How Debug AI Turns Production Traces into Deterministic Reproducers** — DebuggAI, Dec 2025
- **Time-Travel CI: Deterministic Replay Meets Debug AI** — DebuggAI, Nov 2025
- **Reproducible Code Debugging AI in CI/CD** — DebuggAI, Oct 2025

## Git Bisect for Regressions

- **Agentic git bisect** — Matthew Bilyeu, Aug 2025
  - Agent evaluates good/bad criterion at each checkout without human input
- **Master Git Bisect to Find the Exact Commit That Broke Your Code** — Gun.io, May 2025

## Trajectory Smell Detection

- **Evaluating Agent-based Program Repair at Google (Passerine)** — Google Research, Jan 2025
  - NO_TEST_SMELL: present in 66% of failed trajectories
  - CONSECUTIVE_SEARCH: 67% correlation with failure on human bugs
  - NO_OP_CAT: present in 44% of failures
  - CONSECUTIVE_EDITS: correlated across bug types

## Hypothesis Register / Differential Diagnosis

- **MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs** — HKUST/Sichuan University, Dec 2025
  - Information gain maximization for diagnostic action selection
  - State tracking: externalize confirmed facts, active/eliminated hypotheses
  - Grounded reasoning: anchor to verified evidence only
- **MEDDxAgent: Modular Agent Framework for Explainable Differential Diagnosis** — ACL 2025

## Fix Minimality Constraints

- **Controllable LLM Debugging: Knowing When to Stop Matters** — Stanford NLP 191W, Mar 2025
  - Multi-agent approaches make 40-60% more code changes than necessary
  - AST-based approaches balance accuracy with minimal intervention
- **Do AI Models Help Produce Verified Bug Fixes?** — Constructor Institute, Jul 2025
  - LLM-assisted groups proposed fixes faster but required more verification rounds

## Multi-Agent Debugging Architectures

- **InspectCoder: Dynamic Analysis-Enabled Self Repair** — Zhejiang University / Alibaba, Oct 2025
  - Program Inspector + Patch Coder dual-agent model
  - 5-60% relative improvement from runtime state access
- **ROBIN: Conversational Debugging with GitHub Copilot** — Microsoft Research, IEEE VL/HCC, Aug 2024 (Best Paper)
  - "Investigate & Respond" pattern: gather context before proposing fixes
  - 2.5x improvement in bug localization, 3.5x in resolution
- **LLM4FL: Multi-Agent Fault Localization via Graph-Based Retrieval and Reflexion** — [arxiv.org/abs/2409.13642](https://arxiv.org/abs/2409.13642), Concordia/DePaul/Manitoba, Mar 2025
  - +18.55% Top-1 accuracy vs AutoFL on Defects4J v2.0.0
  - Removing call-graph causes 17-23% accuracy drop
  - Self-reflection critique improves ~11%
- **AgentFL: Scaling LLM-based Fault Localization** — [arxiv.org/abs/2403.16362](https://arxiv.org/abs/2403.16362), 2025
  - Three-phase: Comprehension, Navigation, Confirmation
  - 157/395 bugs Top-1 on Defects4J-V1.2.0, $0.074/bug avg cost

## Context Window Management

- **Cutting Through the Noise: Smarter Context Management for LLM-Powered Agents** — JetBrains Research, Dec 2025
  - Observation masking outperforms LLM summarization
  - Summarization causes agents to run 13-15% longer
- **LLM Context Window Management and Long-Context Strategies 2026** — Zylos Research, Jan 2026
  - "Lost in the middle" problem: center information retrieved less reliably

## Root Cause Analysis Automation

- **Exploring LLM-based Agents for Root Cause Analysis** — Microsoft / Washington State University, Mar 2024
  - ReAct: 35% correctness, 4% hallucination vs 40% for retrieval
- **RCACopilot** — Microsoft, arXiv, Jun 2024
  - Alert-type-based routing constrains hypothesis space
- **Leveraging Multi-Agent Framework for Root Cause Analysis** — Springer, Nov 2025

## Error Triage Best Practices

- **Automated Diagnostics & Triage** — PagerDuty Blog, Aug 2025
  - Up to 50% of incident time is diagnosis/routing, not resolution
- **BigPanda AI Triage** — Release Notes, Jan-Mar 2026
- **Using LLMs to Filter Out False Positives from Static Code Analysis** — Datadog, Oct 2025

## Regression Test Codification

- **Causal Debugging: Building a Code Debugging AI That Can Prove Its Fixes** — DebuggAI, Oct 2025
  - 12-stage pipeline including property mining, shadow execution, symbolic checks
- **Testora: LLM-based Regression Detection on PRs** — CISPA, arXiv, 2026
  - Classifies differences as intended vs unintended using PR context

## Agentic Debugging Frameworks

- **SWE-Agent** — Princeton
- **AutoCodeRover** — National University of Singapore
  - AST-level search and patch; spectrum-based fault localization
- **HyperAgent** — ICLR 2025
  - Planner, Navigator, Code Editor, Executor multi-agent architecture
- **OPENDEV** — Mar 2026
  - Terminal-native CLI agent with strict context management
- **SWE-bench SOTA (Mar 2026)**: Sonar Foundation Agent at 79.2% on SWE-bench Verified

## Effort Scaling and Context Engineering

- **Building Effective AI Agents** — [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)
  - Effort scaling: simple queries warrant 1 agent with 3-10 calls; complex need 10+ subagents
  - Simplicity-first: add multi-step agentic systems only when simpler solutions fall short
- **How We Built Our Multi-Agent Research System** — [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
  - Multi-agent uses ~15x more tokens than standard chat — requires tasks where overhead is justified
  - Parallel tool calling cut research time by up to 90% for complex queries
- **Boris Cherny (Claude Code creator): 10 Tips** — [jitendrazaa.com](https://www.jitendrazaa.com/blog/others/tips/10-claude-code-tips-from-the-creator-boris-cherny-february/)
  - "Give Claude a feedback loop and it will 2-3x the quality"
  - Plan Mode reduces token consumption ~53% (38k → 18k per session)
- **Cutting Through the Noise: Smarter Context Management** — [blog.jetbrains.com/research/2025/12](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
  - Observation masking: +2.6% solve rate, -52% cost vs unmanaged baselines
  - LLM summarization caused 15% trajectory elongation vs masking

## Bug-Inducing Commit as Fix Context

- **LLM-Based Regression Bug Repair** — University of Melbourne, Jun 2025
  - Incorporating bug-inducing commit diff yields 1.8x more successful repairs vs without

## SWE-bench SOTA (Mar 2026)

- **Sonar Foundation Agent** — [sonarsource.com](https://www.sonarsource.com/blog/introducing-sonar-foundation-agent/), 79.2% SWE-bench Verified
  - "Free workflow" over rigid pipelines — LLM determines own investigation path
  - AST symbol searching via LlamaIndex, test-driven verification
- **Cursor Debug Mode** — [cursor.com/docs/agent/debug-mode](https://cursor.com/docs/agent/debug-mode)
  - 5-step: explore+hypothesize → instrument → reproduce → analyze → fix+verify
  - Only system that explicitly captures runtime data via instrumentation
- **Google BRT Agent** — [arxiv.org/abs/2502.01821](https://arxiv.org/abs/2502.01821), Feb 2025
  - Reproduction tests improve fix generation by 30%, reduce time-to-fix by 21%
  - Dominant success bigram: (edit, test) — tight generate-validate loops

## Anti-Patterns in AI Debugging

- **AI coding anti-patterns: 6 things to avoid** — Lingo.dev, Sep 2025
- **What to Avoid When Using AI Agents to Write Code** — The Debuggers, Mar 2026
- **Copilot Anti-Patterns & Best Practices** — TechDebt.solutions, Feb 2026
- **Debugging AI-Generated Code: 8 Failure Patterns & Fixes** — Augment Code, Oct 2025
- **Multi-Model AI Code Review: Convergence Loops** — Zylos Research, Mar 2026
  - Echo chamber effect: model reviewing own output inherits same blindspots
