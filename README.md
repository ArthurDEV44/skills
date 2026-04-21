# Skills

Collection of [Claude Code](https://docs.claude.com/en/docs/claude-code) skills and subagents by [ArthurDEV44](https://github.com/ArthurDEV44).

Paired with my Claude Code configuration: [`claude-config`](https://github.com/ArthurDEV44/claude-config) (private).

## Overview

- **`skills/`** — 24 custom skills, invocable via `/skill-name` or auto-triggered by their descriptions
- **`agents/`** — 3 subagents used by multiple skills for isolated, read-only operations

Everything here is dereferenced: skills that were symlinked to `~/.agents/` on my machine are committed as real content so the repo is self-contained.

## Skills

### Meta-workflows (multi-agent orchestrators)

| Skill | Purpose |
|-------|---------|
| [meta-code](skills/meta-code) | Adaptive multi-agent research pipeline — web + codebase + docs, with synthesis, challenge, verify, refine gates |
| [meta-debug](skills/meta-debug) | Error diagnosis with surgical precision — reproduce, triage, root-cause, verify |
| [meta-archi](skills/meta-archi) | Architecture audit for LLM-readiness (CLAUDE.md, AGENTS.md, structure for AI tools) |
| [meta-audit](skills/meta-audit) | Autonomous codebase audit with PRD report — 10-phase pipeline with AutoSCORE validators |
| [meta-refact](skills/meta-refact) | Refactoring pipeline — analyze, plan, execute, simplify, validate |
| [meta-prompt](skills/meta-prompt) | Generate / transform / optimize prompts for Claude Code |
| [meta-review-prd](skills/meta-review-prd) | Independent PRD review before implementation — fresh context, zero author bias |
| [meta-speak](skills/meta-speak) | Write in Arthur's personal voice — analyzes context and tone |
| [meta-storytelling](skills/meta-storytelling) | Copywriting / storytelling for any project — landing pages, blogs, product copy |

### PRD → code lifecycle

| Skill | Purpose |
|-------|---------|
| [write-prd](skills/write-prd) | Research-informed PRD generator — autonomous, decision-making, status tracking |
| [implement-story](skills/implement-story) | PRD story → code + commit with research, review, security gates |
| [review-story](skills/review-story) | Review already-implemented code against its PRD (not the PRD document) |

### Code quality scanners

| Skill | Purpose |
|-------|---------|
| [react-doctor](skills/react-doctor) | React codebase health — security, performance, correctness, architecture |
| [rust-doctor](skills/rust-doctor) | Deep Rust analysis with `rust-doctor` CLI, senior reviewer expertise |
| [security-review](skills/security-review) | OWASP Top 10 audit with severity/confidence ratings |
| [seo-warfare](skills/seo-warfare) | SEO + GEO/AEO audit for traditional and AI search engines |

### Tool scaffolding

| Skill | Purpose |
|-------|---------|
| [api2cli](skills/api2cli) | Generate CLI + AgentSkill for any REST API |
| [mcp-server-dev](skills/mcp-server-dev) | Build MCP servers in TypeScript with `@modelcontextprotocol/sdk` |
| [remotion-best-practices](skills/remotion-best-practices) | Remotion video creation in React — rules + templates |

### Skill lifecycle

| Skill | Purpose |
|-------|---------|
| [skill-creator](skills/skill-creator) | Interactive guide for creating and updating skills |
| [skill-doctor](skills/skill-doctor) | Audit a skill for structural quality, idempotency, and best-practice compliance |

### Domain-specific

| Skill | Purpose |
|-------|---------|
| [extract-catalog](skills/extract-catalog) | Extract product metadata from supplier catalog PDFs (tiles, sanitary, furniture) |
| [frontend-design](skills/frontend-design) | Production-grade, distinctive frontend UI design — editorial, human-crafted |

### Shared

| Path | Purpose |
|------|---------|
| [_shared](skills/_shared) | Shared resources used across skills — agent boundaries, scope guard, three-tier constraints, synthesis templates |

## Subagents

Read-only subagents with isolated context windows. Used by meta-workflows for parallel, uncorrelated research.

| Agent | Purpose |
|-------|---------|
| [agent-websearch](agents/agent-websearch.md) | Web research via Exa MCP (primary) + WebSearch/WebFetch fallback |
| [agent-explore](agents/agent-explore.md) | Systematic codebase exploration — architecture mapping, flow tracing, blast-radius analysis |
| [agent-docs](agents/agent-docs.md) | Version-accurate library documentation via ctx7 CLI |

## Install

### Skills

Install all skills with [Vercel Labs' `skills` CLI](https://github.com/vercel-labs/skills):

```bash
bunx skills add ArthurDEV44/skills
```

Or cherry-pick specific skills:

```bash
bunx skills add ArthurDEV44/skills --skill meta-code --skill write-prd
```

Other useful commands: `bunx skills list`, `bunx skills update`, `bunx skills remove <name>`.

### Agents

The `skills` CLI only manages `skills/`, not `agents/`. Install the 3 subagents manually:

```bash
mkdir -p ~/.claude/agents
for agent in agent-docs agent-explore agent-websearch; do
  curl -sL "https://raw.githubusercontent.com/ArthurDEV44/skills/main/agents/${agent}.md" \
    -o ~/.claude/agents/${agent}.md
done
```

Claude Code picks everything up automatically on next session. Skills with a `name: <slug>` in their frontmatter are invocable as `/<slug>`; others auto-trigger based on their `description`.

## License

MIT
