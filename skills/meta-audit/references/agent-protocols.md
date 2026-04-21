# Agent Protocols — Prompt Templates pour meta-audit

Prompt templates exacts pour chaque agent (standard, extended, et validators) du pipeline meta-audit. Pour la logique de pipeline, voir [SKILL.md](../SKILL.md).

**Output budget :** Les agents d'audit (Phase 3) retournent des findings structures multi-dimensionnels. Le budget standard de 1,500 tokens (agent-boundaries.md) ne s'applique PAS aux agents d'audit — leur output complet est necessaire pour la synthese Phase 5. Budget audit : jusqu'a 4,000 tokens par agent de scan, 2,000 tokens par validator (Phase 6).

**Convention :** Tous les agents de scan (Phase 3) recoivent le typed handoff de Phase 1 + les grounding signals de Phase 2 (static analysis JSON). Les agents retournent leurs findings dans le format structure defini ci-dessous (Extended Findings Output Format). L'orchestrateur collecte et synthetise.

---

## Phase 1 : agent-websearch (RESEARCH)

### Prompt Research A — Best Practices Stack-Specifiques

```
Agent(
  description: "Research {language}/{framework} best practices",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

```
Research best practices and recommended patterns for {language}/{framework} codebases.

## Objective
Find authoritative guidance on building and maintaining high-quality {language}/{framework} projects.

## Search Priorities (IN ORDER)
1. **Architecture patterns** — Recommended project structure, module organization, separation of concerns for {framework}
2. **Code quality patterns** — Typing best practices, error handling patterns, naming conventions for {language}
3. **Testing best practices** — Recommended test framework, coverage expectations, test patterns for {framework}
4. **DX and tooling** — Essential dev tools, linters, formatters, pre-commit hooks for {language}
5. **Anti-patterns** — Common mistakes in {framework} projects, what to avoid

## Search Strategy
- "{framework} project structure best practices 2025 2026"
- "{language} code quality patterns"
- "{framework} testing best practices"
- "{language} tooling recommended setup"
- "{framework} common mistakes anti-patterns"

## Dual-Perspective Search
For each recommendation, also search for counter-arguments or known trade-offs.

## Output Format
### Architecture Patterns
[Recommended structures with rationale]

### Code Quality
[Typing, naming, error handling patterns]

### Testing
[Framework, patterns, coverage expectations]

### Tooling & DX
[Essential tools and configuration]

### Anti-Patterns
[Common mistakes to detect in audit]

### Sources
[All URLs with T1-T4 tier annotations]

## Task Boundaries
- Do NOT read local code
- Do NOT run ctx7 CLI
- Focus on web-accessible best practices only
```

### Prompt Research B — Issues Communes et Securite

```
Agent(
  description: "Research {stack} common issues and security",
  prompt: <template below>,
  subagent_type: "agent-websearch"
)
```

```
Research common issues, security considerations, and tech debt patterns in {language}/{framework} projects.

## Objective
Find the most frequent problems in {framework} codebases — security vulnerabilities, performance pitfalls, tech debt patterns, and quality metrics.

## Search Priorities (IN ORDER)
1. **Security** — OWASP considerations specific to {framework}, common vulnerability patterns, dependency risks
2. **Performance pitfalls** — N+1 queries, bundle size issues, async anti-patterns for {framework}
3. **Tech debt patterns** — Recurring debt patterns in {framework} projects, code smell indicators
4. **Quality metrics** — Industry benchmarks for test coverage, lint compliance, type coverage
5. **Dependency risks** — Known vulnerable packages in {framework} ecosystem

## Search Strategy
- "{framework} security vulnerabilities OWASP"
- "{framework} performance pitfalls common"
- "{framework} tech debt indicators"
- "{language} code quality metrics benchmarks"
- "{framework} dependency vulnerabilities"

## Dual-Perspective Search
For each risk identified, search for mitigations and recommended fixes.

## Output Format
### Security Considerations
[OWASP-relevant risks for this stack, with mitigations]

### Performance Pitfalls
[Common bottlenecks and anti-patterns]

### Tech Debt Patterns
[Recurring debt indicators and how to detect them]

### Quality Benchmarks
[Industry metrics and thresholds]

### Dependency Risks
[Known vulnerability patterns in ecosystem]

### Sources
[All URLs with T1-T4 tier annotations]

## Task Boundaries
- Do NOT read local code
- Do NOT run ctx7 CLI
- Focus on web-accessible information only
```

---

## Phase 3 : Mode STANDARD — 3 agents-explore specialises

3 agents paralleles remplacent le scan monolithique. Chaque agent a un mandat etroit (2 dimensions) produisant des findings plus profonds. Source : Google CodeMender — les agents specialises surpassent les reviewers monolithiques.

**Scoring criteria :** Les sous-criteres et rubriques de scoring sont definis dans [audit-dimensions.md](audit-dimensions.md). Les agents doivent aligner leurs findings sur ces sous-criteres (A1-A5, Q1-Q5, S1-S5, T1-T5, P1-P5, D1-D5) pour faciliter la synthese Phase 5.

### Focus Override

Le placeholder `{focus_override}` est remplace par :
- Si `--focus X` et la dimension est dans le scope de cet agent : `"\n\n## Focus Override\nConcentrate 60% of your exploration on the {X} dimension. For your other dimension, scan only the top 3-5 metrics."`
- Si `--focus X` et la dimension n'est PAS dans le scope de cet agent : `"\n\n## Abbreviated Mode\nThis is NOT the focus dimension. Scan only the top 3-5 key metrics per dimension. Do not deep-dive."`
- Sinon : vide

### Circuit Breaker

Si `project_profile.source_files >= 150` :
- Ajouter au prompt : `"\n\n## Circuit Breaker\nThis is a large codebase (150+ source files). For each dimension, start by scanning the 20 most structurally significant files (largest, most imported, most recently modified) before broadening. Prioritize depth on high-signal files over breadth across all files."`

### Prompt Standard: Structural (Architecture + Quality + DX)

```
Agent(
  name: "scan-structural",
  description: "Audit architecture, quality, and DX",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the STRUCTURAL dimensions of this codebase: Architecture & Structure, Code Quality, and Developer Experience. READ-ONLY scan.

## Research Context (typed handoff)
{phase_1_typed_handoff}

## Static Analysis Grounding Signals (Phase 2)
{static_analysis_signals}
Use these signals as deterministic ground truth. Do NOT re-discover issues already identified by static analysis — reference them and focus your analysis on contextualizing, prioritizing, and finding issues that static tools miss.

## Dimension 1: Architecture & Structure

Scan for:
1. Project structure pattern (vertical slices vs horizontal layers vs other)
2. File sizes: count by range (<100, 100-300, 300-500, 500-1000, >1000 LOC)
3. Top 10 largest files with LOC
4. Import depth: average hops to understand a feature
5. Circular dependencies (import cycles)
6. Module boundaries: clear or blurred?
7. Directory nesting depth
8. Entry points and feature organization
9. Re-export patterns (index files)

## Git Change Analysis
Run git commands to analyze:
- Ratio of single-file patches in last 100 commits: `git log --oneline -100 --format="" --numstat | awk '{print FILENAME}' | sort | uniq -c | sort -rn`
- Top 5 files most frequently modified together (temporal coupling): `git log --oneline -100 --name-only --format="" | sort | uniq -c | sort -rn | head -20`
- Average patch size (LOC added/removed)

## Dimension 2: Code Quality

Scan for:
1. TypeScript strict mode / Rust strict clippy / Python type hints
2. Count: `any`, `as any`, `@ts-ignore` / `unwrap()` / untyped patterns
3. Naming conventions: consistent across codebase?
4. TODO/FIXME/HACK/XXX count
5. Code duplication (similar patterns in multiple files)
6. Error handling pattern: Result/try-catch/error boundaries? Consistent?
7. Validation schemas at boundaries (Zod/joi/serde/pydantic)
8. God-functions (>50 lines) and god-files (>500 LOC)
9. Dead code: unused exports, unreachable branches

## Dimension 3: Developer Experience

Scan for:
1. README: exists? Sections: setup, architecture, contributing?
2. Available scripts: dev, test, build, lint, format?
3. CLAUDE.md / AGENTS.md presence and quality
4. .env.example: exists? Documented variables?
5. Docker/docker-compose for dev setup
6. Pre-commit hooks: husky, lefthook?
7. Lint + format: eslint, prettier, rustfmt, clippy?
8. Editorconfig / shared settings
9. Contributing guide / changelog

{focus_override}

## Output Format

For EACH dimension, provide:

### {Dimension Name}
**Sub-Criteria Scores:**
- {criterion}: {0-10} — {evidence with file:line}
- ...
**Dimension Score:** {weighted average 0-100}

**Key Metrics:**
- {metric}: {value} — {file:line}

**Issues:**
- [{CRITICAL|HIGH|MEDIUM|LOW}] {description} — {file:line} | effort: {quick-win|medium|strategic}

**Strengths:**
- {what's done well} — {file:line}

### Summary
- Total issues across 3 dimensions: {N}
- Top 3 most critical issues

## Task Boundaries
- READ-ONLY — do NOT modify any files
- Do NOT fetch URLs or search the web
- Do NOT run ctx7 CLI
- Focus on evidence-based findings with file:line references
- If a dimension has no issues, say so — don't invent problems
```

### Prompt Standard: Safety (Security + Testing)

```
Agent(
  name: "scan-safety",
  description: "Audit security and testing",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the SAFETY dimensions of this codebase: Security and Testing. READ-ONLY scan.

## Research Context (typed handoff)
{phase_1_typed_handoff}

## Static Analysis Grounding Signals (Phase 2)
{static_analysis_signals}
Use these signals as deterministic ground truth. Reference static analysis findings and focus your analysis on contextualizing, prioritizing, and finding issues that static tools miss.

## Dimension 1: Security

Scan for:
1. Hardcoded secrets: grep for API_KEY, SECRET, PASSWORD, TOKEN, private_key patterns in source files (NOT .env files)
2. .gitignore: does it exclude .env, credentials, key files?
3. .env.example: exists? Well-documented?
4. Auth pattern: JWT (with rotation?), session, OAuth, basic, none
5. Input validation on API endpoints/handlers: present? consistent?
6. Raw SQL queries (injection risk) vs ORM/query builder
7. CORS configuration: permissive or restrictive?
8. Rate limiting: present? On which endpoints?
9. CSRF protection: present?
10. XSS protection: output encoding, CSP headers
11. File upload: validation? size limits?
12. Dependency manifest analysis: parse package-lock.json/Cargo.lock/requirements.txt for known vulnerable patterns (pinned versions, yanked crates, deprecated packages)

## Dimension 2: Testing

Scan for:
1. Test files count vs source files count
2. Test framework: vitest, jest, mocha, cargo test, pytest, go test?
3. Test colocation: same directory or separate __tests__/?
4. CI configuration: .github/workflows, .gitlab-ci.yml, Jenkinsfile?
5. Tests in CI: unit? integration? e2e? Which pass?
6. Fixtures/factories: present? What pattern?
7. Mocking: heavy (mocks everywhere) vs light (real deps)?
8. Flaky test indicators: retry logic, skip markers, timeout overrides
9. Test quality: do tests verify behavior or just cover lines?
10. Missing test coverage: critical paths without tests

{focus_override}

## Output Format

For EACH dimension, provide:

### {Dimension Name}
**Sub-Criteria Scores:**
- {criterion}: {0-10} — {evidence with file:line}
- ...
**Dimension Score:** {weighted average 0-100}

**Key Metrics:**
- {metric}: {value} — {file:line}

**Issues:**
- [{CRITICAL|HIGH|MEDIUM|LOW}] {description} — {file:line} | effort: {quick-win|medium|strategic}

**Strengths:**
- {what's done well} — {file:line}

### Summary
- Total issues across 2 dimensions: {N}
- Top 3 most critical issues

## Task Boundaries
- READ-ONLY — do NOT modify any files
- Do NOT run `npm audit` or `cargo audit` — analyze lockfile patterns instead
- Do NOT fetch URLs or search the web
- Do NOT run ctx7 CLI
- Focus on evidence-based findings with file:line references
- If a dimension has no issues, say so — don't invent problems
```

### Prompt Standard: Runtime (Performance + Observability)

```
Agent(
  name: "scan-runtime",
  description: "Audit performance and observability",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the RUNTIME dimensions of this codebase: Performance and Observability. READ-ONLY scan.

## Research Context (typed handoff)
{phase_1_typed_handoff}

## Static Analysis Grounding Signals (Phase 2)
{static_analysis_signals}
Use these signals as deterministic ground truth. Reference static analysis findings and focus on contextualizing, prioritizing, and finding issues that static tools miss.

## Dimension 1: Performance

Scan for:
1. Sequential awaits in loops (should be Promise.all/join)
2. N+1 query patterns (queries inside loops, no eager loading)
3. Caching: Redis, in-memory, HTTP cache headers
4. Pagination on list endpoints
5. Bundle configuration (frontend): splitting, lazy loading
6. Memory leak patterns: event listeners, closures, unclosed resources
7. Connection pooling configuration
8. Async patterns: proper use of async/await vs blocking

## Dimension 2: Observability (sub-dimension of Performance)

Scan for:
1. Structured logging: JSON logs, log levels, contextual metadata?
2. Error tracking integration: Sentry, Datadog, Bugsnag, or similar?
3. Health check endpoints: /health, /ready, /live?
4. Metrics collection: Prometheus, StatsD, custom metrics?
5. Tracing: OpenTelemetry, Jaeger, or similar distributed tracing?
6. Log aggregation: configured destination (stdout, file, service)?

{focus_override}

## Output Format

### Performance
**Sub-Criteria Scores:**
- {criterion}: {0-10} — {evidence with file:line}
- ...
**Dimension Score:** {weighted average 0-100}

**Key Metrics:**
- {metric}: {value} — {file:line}

**Issues:**
- [{CRITICAL|HIGH|MEDIUM|LOW}] {description} — {file:line} | effort: {quick-win|medium|strategic}

**Strengths:**
- {what's done well} — {file:line}

### Observability (reported as sub-scores within Performance)
**Sub-Criteria Scores:**
- {criterion}: {0-10} — {evidence}

### Summary
- Total issues: {N}
- Top 3 most critical issues

## Task Boundaries
- READ-ONLY — do NOT modify any files
- Do NOT fetch URLs or search the web
- Do NOT run ctx7 CLI
- Focus on evidence-based findings with file:line references
- If a dimension has no issues, say so — don't invent problems
```

---

## Phase 3 : Mode EXTENDED — Prompts par Agent

Chaque agent extended recoit un prompt focalise sur sa dimension + le typed handoff de Phase 1. Tous utilisent le tool Agent() standard (pas de TeamCreate ni SendMessage).

**Scoring criteria :** Les sous-criteres et rubriques de scoring sont definis dans [audit-dimensions.md](audit-dimensions.md). Les agents doivent aligner leurs findings sur ces sous-criteres pour faciliter la synthese Phase 5.

### Prompt Extended: Architecture

```
Agent(
  name: "audit-archi",

  description: "Audit architecture and structure",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the ARCHITECTURE & STRUCTURE dimension of this codebase. READ-ONLY.

## Research Context
{phase_1_typed_handoff}

## Your Focus
You are part of a parallel audit team. YOUR dimension is Architecture & Structure.

Scan for:
1. Project structure pattern (vertical slices vs horizontal layers vs other)
2. File sizes: count by range (<100, 100-300, 300-500, 500-1000, >1000 LOC)
3. Top 10 largest files with LOC
4. Import depth: average hops to understand a feature
5. Circular dependencies (import cycles)
6. Module boundaries: clear or blurred?
7. Directory nesting depth
8. Entry points and feature organization
9. Monorepo structure (if applicable)
10. Re-export patterns (index files)

## Git Change Analysis
Run git commands to analyze:
- Ratio of single-file patches in last 100 commits
- Average patch size (LOC added/removed)
- Top 5 files most frequently modified together (temporal coupling)

## Output Format
### Score Estimate: {0-100}

### Metrics
- Average file size: {N} LOC
- Files >300 LOC: {N}, >500 LOC: {N}, >1000 LOC: {N}
- Max nesting depth: {N} levels
- Import depth: {N} average hops
- Circular dependencies: {N}
- Pattern: {vertical slices | horizontal layers | mixed | unclear}
- Git: {N}% single-file patches, avg {N} LOC per patch

### Issues
- [{severity}] {description} — {file:line}

### Strengths
- {what's done well} — {file:line}

## Task Boundaries
- READ-ONLY. Do NOT modify files.
- Focus ONLY on architecture and structure.
- Provide file:line references for every finding.

When done, return your complete findings in the output format above.
```

### Prompt Extended: Quality

```
Agent(
  name: "audit-quality",

  description: "Audit code quality and patterns",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the CODE QUALITY dimension of this codebase. READ-ONLY.

## Research Context
{phase_1_typed_handoff}

## Your Focus
You are part of a parallel audit team. YOUR dimension is Code Quality.

Scan for:
1. TypeScript strict mode / Rust strict clippy / Python type hints
2. Count: `any`, `as any`, `@ts-ignore` / `unwrap()` / untyped patterns
3. Naming conventions: consistent across codebase?
4. TODO/FIXME/HACK/XXX count
5. Code duplication (similar patterns in multiple files)
6. Error handling pattern: Result/try-catch/error boundaries? Consistent?
7. Validation schemas at boundaries (Zod/joi/serde/pydantic)
8. God-functions (>50 lines) and god-files (>500 LOC)
9. Dead code: unused exports, unreachable branches
10. Consistency: same patterns used everywhere or ad-hoc?

## Output Format
### Score Estimate: {0-100}

### Metrics
- Strict mode: {yes|no|partial}
- `any`/`unwrap` count: {N}
- TODO/FIXME/HACK count: {N}
- Naming convention: {consistent|mostly|inconsistent}
- Error handling: {exhaustive|partial|ad-hoc|absent}
- Validation: {schemas at boundaries|inline|absent}

### Issues
- [{severity}] {description} — {file:line}

### Strengths
- {what's done well} — {file:line}

## Task Boundaries
- READ-ONLY. Do NOT modify files.
- Focus ONLY on code quality.
- Provide file:line references for every finding.

When done, return your complete findings in the output format above.
```

### Prompt Extended: Security

```
Agent(
  name: "audit-security",

  description: "Audit security and dependencies",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the SECURITY dimension of this codebase. READ-ONLY.

## Research Context
{phase_1_typed_handoff}

## Your Focus
You are part of a parallel audit team. YOUR dimension is Security.

Scan for:
1. Hardcoded secrets: grep for API_KEY, SECRET, PASSWORD, TOKEN, private_key patterns in source files (NOT .env files)
2. .gitignore: does it exclude .env, credentials, key files?
3. .env.example: exists? Well-documented?
4. Auth pattern: JWT (with rotation?), session, OAuth, basic, none
5. Input validation on API endpoints/handlers: present? consistent?
6. Raw SQL queries (injection risk) vs ORM/query builder
7. CORS configuration: permissive or restrictive?
8. Rate limiting: present? On which endpoints?
9. Dependency manifest: check for known vulnerability patterns
10. CSRF protection: present?
11. XSS protection: output encoding, CSP headers
12. File upload: validation? size limits?

## Output Format
### Score Estimate: {0-100}

### Metrics
- Hardcoded secrets: {N} potential findings
- .env in .gitignore: {yes|no}
- Auth pattern: {description}
- Input validation: {at all boundaries|partial|absent}
- Raw SQL: {N} instances
- CORS: {restrictive|permissive|not configured}
- Rate limiting: {yes|no|partial}

### Issues
- [{severity}] {description} — {file:line}

### Strengths
- {what's done well} — {file:line}

## Task Boundaries
- READ-ONLY. Do NOT modify files.
- Focus ONLY on security.
- Do NOT run `npm audit` or `cargo audit` — just analyze patterns.
- Provide file:line references for every finding.

When done, return your complete findings in the output format above.
```

### Prompt Extended: Testing

```
Agent(
  name: "audit-tests",

  description: "Audit testing and CI/CD",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit the TESTING & CI/CD dimension of this codebase. READ-ONLY.

## Research Context
{phase_1_typed_handoff}

## Your Focus
You are part of a parallel audit team. YOUR dimension is Testing & CI/CD.

Scan for:
1. Test files count vs source files count
2. Test framework: vitest, jest, mocha, cargo test, pytest, go test?
3. Test colocation: same directory or separate __tests__/?
4. CI configuration: .github/workflows, .gitlab-ci.yml, Jenkinsfile?
5. Tests in CI: unit? integration? e2e? Which pass?
6. Fixtures/factories: present? What pattern?
7. Mocking: heavy (mocks everywhere) vs light (real deps)?
8. Flaky test indicators: retry logic, skip markers, timeout overrides
9. Test quality: do tests verify behavior or just cover lines?
10. Missing test coverage: critical paths without tests

## Output Format
### Score Estimate: {0-100}

### Metrics
- Test files: {N} / Source files: {N} (ratio: {N}%)
- Framework: {name}
- Colocation: {same dir|separate|mixed}
- CI: {present|absent|broken}
- Tests in CI: {yes — types: unit, integration|no}
- Fixtures: {present|absent}
- Mocking level: {heavy|moderate|light}

### Issues
- [{severity}] {description} — {file:line}

### Strengths
- {what's done well} — {file:line}

## Task Boundaries
- READ-ONLY. Do NOT modify files.
- Do NOT run tests — just analyze test code and CI config.
- Focus ONLY on testing and CI/CD.
- Provide file:line references for every finding.

When done, return your complete findings in the output format above.
```

### Prompt Extended: Performance & DX

```
Agent(
  name: "audit-perf-dx",

  description: "Audit performance and DX",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
Audit PERFORMANCE and DEVELOPER EXPERIENCE of this codebase. READ-ONLY.

## Research Context
{phase_1_typed_handoff}

## Your Focus
You are part of an audit swarm. YOUR dimension covers both Performance AND Developer Experience.

### Performance — Scan for:
1. Sequential awaits in loops (should be Promise.all/join)
2. N+1 query patterns (queries inside loops, no eager loading)
3. Caching: Redis, in-memory, HTTP cache headers
4. Pagination on list endpoints
5. Bundle configuration (frontend): splitting, lazy loading
6. Memory leak patterns: event listeners, closures, unclosed resources
7. Connection pooling configuration
8. Async patterns: proper use of async/await vs blocking

### Developer Experience — Scan for:
1. README: exists? Sections: setup, architecture, contributing?
2. Available scripts: dev, test, build, lint, format?
3. CLAUDE.md / AGENTS.md presence and quality
4. .env.example: exists? Documented variables?
5. Docker/docker-compose for dev setup
6. Pre-commit hooks: husky, lefthook?
7. Lint + format: eslint, prettier, rustfmt, clippy?
8. Editorconfig / shared settings
9. Contributing guide / changelog

## Output Format

### Performance
#### Score Estimate: {0-100}
#### Metrics
- Sequential awaits in loops: {N} instances
- N+1 patterns: {N} potential instances
- Caching: {present|absent|partial}
- Pagination: {present on list endpoints|absent}
#### Issues
- [{severity}] {description} — {file:line}
#### Strengths
- {what's done well} — {file:line}

### Developer Experience
#### Score Estimate: {0-100}
#### Metrics
- README: {complete|partial|absent}
- Scripts: {N} available ({list})
- AI config: {CLAUDE.md|AGENTS.md|none}
- .env.example: {documented|exists|absent}
- Pre-commit hooks: {yes|no}
- Lint/format: {configured and enforced|configured|absent}
#### Issues
- [{severity}] {description} — {file:line}
#### Strengths
- {what's done well} — {file:line}

## Task Boundaries
- READ-ONLY. Do NOT modify files.
- Focus on Performance + DX only.
- Provide file:line references for every finding.

When done, return your complete findings in the output format above.
```

---

## Extended Findings Output Format

Chaque agent extended retourne ses resultats directement a l'orchestrateur via le mecanisme standard du tool Agent(). Le retour doit contenir TOUTES les donnees necessaires a la synthese Phase 5.

### Template de retour des findings

Chaque agent doit structurer son output ainsi :

```
## {DIMENSION_NAME} Audit Findings

### Score Estimate: {0-100}

### Sub-Criteria Scores
- {criterion_1}: {0-10} — {evidence}
- {criterion_2}: {0-10} — {evidence}
- ...

### Key Metrics
- {metric_name}: {value} — {file:line or source}
- ...

### Issues
- [{CRITICAL|HIGH|MEDIUM|LOW}] {description} — {file:line} | effort: {quick-win|medium|strategic}
- ...

### Strengths
- {description} — {file:line}
- ...

### Git Analysis (architecture agent only)
- Single-file patch ratio: {N}%
- Average patch size: {N} LOC
- Top temporal couplings: {file_a} ↔ {file_b} ({N} co-changes)
```

### Protocole de reception

L'orchestrateur :
1. Attend la completion des 5 agents (retour direct via Agent tool)
2. Si un agent echoue → utilise les resultats partiels, note la dimension incomplete
3. Passe les findings collectes a Phase 5

---

## Phase 4 — Prompt agent-docs

Template pour la phase DOCS conditionnelle. Utilise `subagent_type: "agent-docs"` (custom subagent defini dans CLAUDE.md global).

```
Agent(
  description: "Fetch docs for {library_name}",
  prompt: <template below>,
  subagent_type: "agent-docs"
)
```

```
Fetch up-to-date documentation for {library_name} (version {version}) used in a {language}/{framework} project.

## Context
This library was flagged during a codebase audit because:
{reason — e.g., "non-standard usage pattern detected", "version-specific API question", "deprecated API usage suspected"}

## What I Need
1. Current API reference for the specific functions/patterns flagged
2. Migration guide if the version is outdated
3. Known issues or breaking changes in this version
4. Recommended usage patterns (official examples)

## Search Strategy
Use ctx7 CLI:
1. `bunx ctx7@latest resolve {library_name}` — resolve library ID
2. `bunx ctx7@latest docs {library_id}` — fetch relevant documentation (max 3 ctx7 calls total)

## Output Format
### {library_name} v{version}

**Current API:**
- {function/pattern}: {correct usage with example}

**Issues Found:**
- {deprecated API, breaking change, or non-standard pattern}: {recommended fix}

**Migration Notes (if applicable):**
- {from version X}: {what changed and how to update}

**Sources:**
- ctx7: {library_id} — {section fetched}

## Task Boundaries
- Do NOT read local code (the audit scan already did that)
- Do NOT search the web (use ctx7 only)
- Do NOT modify any files
- Max 3 ctx7 CLI calls total across all libraries
```

---

## Phase 6 — Micro-Validator Prompts

3 micro-validators purpose-scoped (inspire CodeMender, Google DeepMind). Chaque validator a un seul objectif pour maximiser la precision. Le cite-check est deterministe (pas d'agent LLM). Les deux autres sont des agent-explore.

### Prompt Validator: FP-Filter

```
Agent(
  description: "Filter false positives in audit findings",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
You are a FALSE POSITIVE FILTER for a codebase audit. Your ONLY job is to identify findings that are not real issues. READ-ONLY.

## Issues List to Filter
{issues_list_from_phase_5}

## Your Mandate

You have ONE objective: identify false positives and duplicates. Do NOT evaluate scores, do NOT add new findings, do NOT check evidence grounding (another validator handles that).

### False Positive Detection
For each issue, check if the flagged pattern is intentional:
- `unwrap()` in test code → acceptable (check if file is in tests/ or has .test. in name)
- `any` in type guards, generic utilities, or FFI bindings → intentional
- Missing validation on internal-only endpoints → may be by design (check if endpoint is exported/public)
- Console.log in development-only code or debug modules → acceptable
- TODO/FIXME with issue tracker references (e.g., TODO(#123)) → managed debt, not unmanaged
- Permissive CORS in development config only → check if production config exists separately
- Large files that are auto-generated (migrations, lock files, schemas) → not god-files

For each confirmed false positive: verify by reading the actual code at file:line.

### Deduplication
Identify issues describing the same root cause from different dimensions:
- "God-file detected" (Architecture) + "File too large" (Quality) → merge as Architecture
- "No input validation" (Security) + "Missing Zod schemas" (Quality) → merge as Security
- Same file:line cited in multiple issues → likely same root cause

Recommend merging with the most actionable framing.

## Output Format

### False Positives
- ISS-{NNN}: {reason verified at file:line} → Recommend: {remove | downgrade to LOW}

### Duplicates
- ISS-{NNN} + ISS-{MMM}: {same root cause} → Merge as ISS-{NNN}: {recommended framing}

### Stats
- Issues reviewed: {N}
- False positives found: {N}
- Duplicates found: {N}

## Task Boundaries
- READ-ONLY — do NOT modify any files
- Do NOT evaluate scores or add findings
- Do NOT check evidence grounding (cite-check handles that)
- Do NOT search the web
- Verify by reading actual code — do NOT guess
```

### Prompt Validator: Score-Coherence

```
Agent(
  description: "Validate score-to-findings coherence",
  prompt: <template below>,
  subagent_type: "agent-explore"
)
```

```
You are a SCORE COHERENCE VALIDATOR for a codebase audit. Your ONLY job is to verify that scores match findings. READ-ONLY.

## Audit Summary with Scores
{audit_summary_from_phase_5}

## Issues List (post fp-filter)
{issues_list_post_fp_filter}

## Validation from cite-check
{cite_check_results}

## Your Mandate

You have ONE objective: verify that scores are consistent with findings. Do NOT filter false positives (already done). Do NOT check evidence grounding (already done).

### Score-to-Findings Coherence
For each dimension and sub-criterion:
1. A sub-criterion scored 8+/10 should have 0-1 issues max. If it has 3+ issues → score too high.
2. A sub-criterion scored 3-/10 should have 2+ issues. If it has 0 → score too low, or findings are missing.
3. If an issue was marked [UNGROUNDED] by cite-check → its associated sub-criterion score should increase (the problem may not exist).
4. If multiple issues were merged as duplicates → adjust counts accordingly.

### Completeness Check
For each dimension scoring <60:
- Count the remaining grounded findings for that dimension
- Are there enough to explain the low score?
- If not → flag as gap: "{dimension}: score {N} but only {M} grounded findings — scan may have missed issues"

### Conflict Resolution (AgentAuditor pattern)
When two agents flagged conflicting severity for the same code:
- Read the actual code at file:line
- Evaluate which agent's assessment has stronger evidence
- Recommend the severity backed by better evidence, not the average

### N/A Validation
For sub-criteria marked N/A:
- Is N/A justified? (genuinely insufficient evidence, not just missed by agent)
- If >2 N/A in a dimension → recommend re-scan for that dimension

## Output Format

### Score Adjustments
- {dimension}.{sub_criterion}: {current} → {recommended}: {reason with evidence}

### Gaps
- {dimension}: {what's missing, with justification}

### Conflicts Resolved
- ISS-{NNN}: {agent_a severity} vs {agent_b severity} → {recommended}: {evidence-based reason}

### N/A Review
- {sub_criterion}: N/A → {justified | recommend re-scan}: {reason}

### Overall Confidence: {HIGH | MEDIUM}

## Task Boundaries
- READ-ONLY — do NOT modify any files
- Do NOT filter false positives (already done by fp-filter)
- Do NOT add new findings — only validate existing scores
- Do NOT search the web
- Every adjustment must cite evidence (file:line or metric)
```

---

## Source Tier Definitions

Tiers d'evaluation de la fiabilite des sources. Utilises dans les typed handoffs et les outputs de research.

| Tier | Definition | Exemple | Poids dans les decisions |
|------|-----------|---------|------------------------|
| T1 | Documentation officielle, RFC, specs | docs.anthropic.com, rust-lang.org/reference | Fort — fait autorite |
| T2 | Blog officiel du framework/langage | nextjs.org/blog, blog.rust-lang.org | Fort — intention des mainteneurs |
| T3 | Conference talk, interview d'auteur, article d'expert reconnu | RustConf talk, Pragmatic Engineer interview | Moyen — expertise contextualisee |
| T4 | Blog communautaire, forum, StackOverflow | dev.to, Reddit, StackOverflow answers | Faible — a cross-checker avec T1/T2 |

---

## Typed Handoff Compression (Phase 1 → Phase 2)

Apres Phase 1, l'orchestrateur compresse les outputs des deux agents en typed handoff :

```
Research context for audit:

stack: {language}/{framework}

best_practices:
- text: "{finding}" | source: "{url}" | tier: T1|T2|T3|T4
- ...

common_issues:
- text: "{issue pattern}" | source: "{url}" | tier: T1|T2|T3|T4
- ...

security_concerns:
- text: "{OWASP concern}" | source: "{url}" | tier: T1|T2|T3|T4
- ...

quality_benchmarks:
- metric: "{name}" | threshold: "{value}" | source: "{url}"
- ...

libraries_to_check: [{name, version}]
gaps: ["{what was not found}"]
```

Target: 1000-1500 tokens (aligne sur la guidance Anthropic Engineering). URLs comme pointeurs (compression restorable). Inclure assez de detail pour que les agents Phase 3 puissent contextualiser sans re-rechercher.
