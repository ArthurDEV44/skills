# Agent Protocols — Prompt Templates pour meta-archi

## Spawning Protocol

Tous les agents sont lances via l'outil `Agent` avec `subagent_type`. Pas de `team_name` ni `name` — ce sont des appels one-shot.

---

## Phase 1 — Research (agent-websearch x3, parallel)

### Agent A — Anthropic & Boris Cherny (Claude Code creator)

```
Agent(
  description: "Research Anthropic + Boris Cherny AI architecture",
  prompt: <Prompt A ci-dessous>,
  subagent_type: "agent-websearch"
)
```

#### Prompt A

```
Recherche approfondie sur les meilleures pratiques d'architecture de codebase pour Claude Code, en ciblant specifiquement Anthropic (editeur) et Boris Cherny (createur de Claude Code).

## Axes de recherche

1. **Anthropic / Claude Code (officiel)** :
   - CLAUDE.md best practices officielles : format, taille, contenu recommande vs exclu
   - Anti-pattern "over-specified CLAUDE.md" : "if your CLAUDE.md is too long, Claude ignores half of it"
   - Context window management : auto-compaction, strategies de preservation
   - Skills architecture : progressive disclosure, chargement a la demande
   - Subagents : architecture `.claude/agents/`, isolation de contexte, Writer/Reviewer pattern
   - Hooks vs CLAUDE.md advisory : quand utiliser des hooks deterministes vs des instructions
   - Progressive disclosure : syntaxe `@path/to/file.md`, hierarchie multi-niveaux
   - Position sur AGENTS.md (issue #31005 GitHub — 3000+ upvotes, zero reponse Anthropic). Claude Code ne lit PAS AGENTS.md nativement en mars 2026

2. **Boris Cherny (createur de Claude Code)** :
   - "Glob and grep bat le RAG" — noms de fichiers expressifs > documentation opaque
   - Code quality = "double-digit percent impact on engineering productivity"
   - Separation planification/implementation (plan mode)
   - CLAUDE.md comme memoire vivante auto-generee
   - 5 worktrees paralleles pour 20-30 PRs/jour — modules faiblement couples
   - "Infrastructure-first philosophy"
   - "The most important thing: give Claude a way to verify its work — 2-3x quality"
   - Threads, blog posts, interviews, talks

3. **Anthropic Engineering Blog** :
   - Articles sur l'architecture de coding agents
   - Patterns de verification automatisee (tests comme feedback loop)

## Strategie de recherche
- Chercher "Boris Cherny Claude Code" sur tous les canaux
- Chercher "Anthropic CLAUDE.md best practices 2025 2026"
- Chercher "Claude Code skills subagents architecture"
- Chercher "Anthropic AGENTS.md position"
- 5-6 recherches complementaires

## Format de sortie

### Anthropic Official Recommendations
[Findings numerotes avec URLs]

### Boris Cherny Insights
[Findings numerotes avec URLs]

### Key Architectural Principles
[Synthese : principes actionnables pour l'architecture]

### AGENTS.md Position
[Position officielle ou absence]

### Contradictions Detected
[Divergences entre sources]

### Sources
[URLs consultees]
```

---

### Agent B — Cross-tools (Google, OpenAI, Cursor)

```
Agent(
  description: "Research Google OpenAI Cursor AI config practices",
  prompt: <Prompt B ci-dessous>,
  subagent_type: "agent-websearch"
)
```

#### Prompt B

```
Recherche approfondie sur les meilleures pratiques d'architecture de codebase pour les AI coding agents, ciblant Google AI, OpenAI, et Cursor.

## Axes de recherche

1. **Google AI / Gemini** :
   - Jules : support natif AGENTS.md, patterns recommandes
   - .aiexclude : specification (syntaxe .gitignore)
   - Gemini Code Assist : MCP integration, patterns d'architecture

2. **OpenAI / Codex** :
   - AGENTS.md specification : hierarchie de decouverte, limite 32 KiB, AGENTS.override.md
   - Les 6 sections les plus efficaces (GitHub blog, 2500+ repos) : Commands, Testing, Project Structure, Code Style, Git Workflow, Boundaries
   - Pattern three-tier : "always do / ask first / never do"
   - GitHub Copilot : copilot-instructions.md, 4000 chars limit

3. **Cursor** :
   - Migration .cursorrules → .cursor/rules/*.mdc
   - Frontmatter .mdc : description (<200 chars), globs, alwaysApply
   - 4 types de regles : Always (alwaysApply:true), Auto-attach (globs), Agent (description pour routing), Manual (@mention)
   - Taille recommandee : <500 lignes par fichier .mdc (soft limit, pas de hard limit)

4. **Convergence cross-tools** :
   - AGENTS.md adopte par 25+ outils, 60 000+ repos, gouverne par l'Agentic AI Foundation sous la Linux Foundation
   - Fondateurs : OpenAI Codex, Amp, Google Jules, Cursor, Factory.ai
   - Claude Code ne lit PAS AGENTS.md nativement (mars 2026) malgre 3000+ upvotes sur GitHub issue #31005
   - Markdown comme format universel
   - Hierarchie par proximite (AGENTS.md le plus proche du fichier edite prend precedence)

## Strategie de recherche
- Chercher "AGENTS.md specification OpenAI Codex 2025 2026"
- Chercher "Cursor .mdc rules format best practices 2025"
- Chercher "Google Jules .aiexclude AGENTS.md"
- 5-6 recherches complementaires

## Format de sortie

### Google AI / Gemini Recommendations
[Findings avec URLs]

### OpenAI / Codex Recommendations
[Findings avec URLs]

### Cursor Recommendations
[Findings avec URLs]

### Cross-Tool Convergence
[Tableau comparatif]

### Quantitative Limits (validated)
| Tool | File | Size Limit | Key Constraint |
|------|------|-----------|----------------|
| Codex | AGENTS.md | 32 KiB | Configurable 64 KiB |
| Copilot | instructions | 4000 chars | Code review only |
| Cursor | .mdc | <500 lines (soft) | 4 types regles : Always, Auto-attach, Agent, Manual |
| Jules | AGENTS.md | N/A doc | Auto-detection |

### Sources
[URLs consultees]
```

---

### Agent C — Metriques quantitatives et benchmarks

```
Agent(
  description: "Research LLM architecture metrics and benchmarks",
  prompt: <Prompt C ci-dessous>,
  subagent_type: "agent-websearch"
)
```

#### Prompt C

```
Recherche approfondie sur les metriques quantitatives et benchmarks pour mesurer la "LLM-readiness" d'une architecture de codebase.

## Axes de recherche

1. **SWE-bench (priorite #1)** :
   - SWE-bench Verified : 2+ fichiers = hard (avg 2.0 vs 1.03 easy), 55.56% hard multi-fichiers vs 3.09% easy (Ganhotra 2025)
   - SWE-bench Pro (nouveau standard, 1865 taches multi-langages) : 4.1 fichiers, 107.4 LOC en moyenne
   - Patches Verified : 1.7 fichiers, 33 LOC, 3 fonctions (ICLR 2024)
   - SWE-bench Pro : chute performance sur codebases inconnues (Opus 4.5 : 80.9% Verified → 45.9% Pro)
   - Scaffolding impact : jusqu'a 20% de variation selon les outils (scaffolding > model choice parfois)

2. **Type systems et erreurs LLM** :
   - arXiv 2504.09246 : type-constrained generation reduit erreurs >50%
   - ClassEval (ICSE 2024) : performance class-level < method-level

3. **Context window management** :
   - Auto-compaction Claude Code : seuil reel (~83.5% mainstream, ~95% subagents, PAS "40% rule")
   - Context rot : precision chute 92% → 63% quand info enterree au milieu (Chroma research)
   - Aider benchmark : format "whole" bloque sur grands fichiers

4. **Naming impact sur LLM** :
   - arXiv 2307.12488v5 : anonymisation noms → Java -75%, Python -65%
   - Function definition names = impact individuel le plus fort
   - Dimension manquante de la plupart des frameworks de scoring

4. **Frameworks de scoring** :
   - Factory.ai Agent Readiness : 8 piliers, 5 niveaux, seuil 80%
   - Kodus agent-readiness : 7 piliers, 39 checks, open-source
   - Comparer dimensions et seuils

5. **Build & Verification metrics** :
   - Impact de la CI/CD sur la performance des agents
   - Correlation entre feedback loops et qualite de generation
   - Factory.ai : Build System comme pilier critique

6. **Dev Environment metrics** :
   - Impact de la reproductibilite de l'environnement
   - Lock files, containerisation, version pinning

## Strategie de recherche
- 5-6 recherches complementaires sur SWE-bench, Factory.ai, Kodus, arXiv

## Format de sortie

### Validated Metrics
| Metric | Finding | Source | Confidence |
|--------|---------|--------|------------|

### Unvalidated Claims
[Liste : "94%", "40% rule", seuils LOC]

### Scoring Frameworks Compared
[Factory.ai vs Kodus : dimensions, seuils, approche]

### Sources
[URLs]
```

---

## Phase 3 — Scan (agent-explore)

```
Agent(
  description: "Deep architecture scan for LLM readiness",
  prompt: <Prompt Scan ci-dessous>,
  subagent_type: "agent-explore"
)
```

### Prompt Scan

```
Scan d'architecture approfondi du codebase dans le repertoire courant pour evaluer sa "LLM-readiness".

Stack detecte : {stack_profile}
Research findings (compresses) : {compressed_phase1_findings}

## Ce que tu dois mesurer

### 1. Structure de fichiers
- Nombre total de fichiers source
- Distribution des tailles : tranches <100, 100-300, 300-500, 500-1000, >1000 LOC
- Top 10 des plus gros fichiers avec LOC
- Profondeur maximale de nesting
- Nombre moyen de fichiers par dossier (L1 et L2)

### 2. Pattern architectural
- horizontal layers | vertical slices | mixte — avec evidence
- Colocalisation tests : same-dir | adjacent | separate
- Colocalisation types : inline | adjacent | global

### 3. Graphe d'imports
- Top 10 fichiers avec le plus d'imports
- Dependencies circulaires
- Index files et re-exports
- Import paths : relatifs ou aliases

### 4. Type safety
TS: strict mode, any count, Zod/validation
Rust: unwrap count, error handling, serde

### 5. Documentation & AI config
- CLAUDE.md : taille, sections, anti-pattern over-specified, @imports, compaction instructions
- AGENTS.md : taille, 6 sections, <32KiB, three-tier boundaries
- .cursor/rules/*.mdc vs .cursorrules legacy
- Hooks (.claude/settings.json) : regles critiques en hooks deterministes ?
- .claude/agents/ : structure, YAML frontmatter, tools scope ?
- copilot-instructions.md, .aiexclude
- README : setup instructions ?
- Ratio commentaires/code (echantillon 10 fichiers)

### 6. Tests
- Framework, colocalisation, ratio fichiers test/source, CI

### 7. Build & Verification
- Commandes build documentees (package.json scripts, Makefile, Cargo.toml)
- Tests executables en 1 commande
- CI presente (.github/workflows/, .gitlab-ci.yml)
- Feedback loop automatisee (pre-commit hooks, lint-staged, husky)
- Lock files presents et commites

### 8. Dev Environment
- Lock files (package-lock.json, pnpm-lock.yaml, Cargo.lock)
- .env.example ou .env.template
- Scripts de setup (setup.sh, Makefile)
- Containerisation (Dockerfile, docker-compose.yml, devcontainer.json)
- Version pinning (.node-version, .tool-versions, rust-toolchain.toml)

### 9. Git Change Analysis
```bash
git log --oneline --stat -100 | grep -c "1 file changed"
git log --oneline -100 | wc -l
git log --stat -100 --format="" | grep "insertion\|deletion" | head -20
```
- Ratio patches mono-fichier, taille moyenne, couplage temporel, god-files

### 10. Naming Expressiveness
- Echantillon 20 fonctions : ratio intent-revealing vs cryptique
- Grep patterns cryptiques : single-letter vars hors boucles, `doStuff`, `handleIt`, `process`, `data`, `tmp`
- Grep magic strings/numbers : literals non assignes a des constantes
- Coherence convention : camelCase vs snake_case vs mixte
- Migration completeness : frameworks/patterns concurrents ?

## Instructions
- Glob pour patterns, Grep count pour metriques, Read pour configs, Bash (read-only) pour git
- Echantillonner si >500 fichiers

## Format de sortie

### File Structure Metrics
| Metrique | Valeur |
|----------|--------|

### Top 10 Largest Files
| File | LOC |
|------|-----|

### Architectural Pattern
- **Pattern :** {horizontal|vertical|mixed}
- **Evidence :** {details}
- **Test colocation :** {type}
- **Type colocation :** {type}

### Import Graph
| Metrique | Valeur |
|----------|--------|

### Type Safety
| Metrique | Valeur |
|----------|--------|

### AI Config Status
| Config | Present | Lines | Quality |
|--------|---------|-------|---------|

### Test Infrastructure
| Metrique | Valeur |
|----------|--------|

### Build & Verification
| Metrique | Valeur |
|----------|--------|
| Build command documented | {yes/no} — {command} |
| Tests in 1 command | {yes/no} — {command} |
| CI present | {yes/no} — {details} |
| Feedback loop | {pre-commit/lint-staged/none} |
| Lock files | {present+committed/present/missing} |

### Dev Environment
| Metrique | Valeur |
|----------|--------|
| Lock files | {present+committed/present/missing} |
| .env.example | {yes/no} |
| Setup script | {yes/no} — {path} |
| Container | {devcontainer/dockerfile/docker-compose/none} |
| Version pinning | {tool} or none |

### Git Change Analysis
| Metrique | Valeur |
|----------|--------|

### Context Budget Estimation
- Fichiers par feature : {N}
- Tokens par feature : ~{N}
- % fenetre Claude (200K) : {N}%
```

---

## Parallel Spawning

### Phase 1 — Trois agent-websearch en parallele

Lancer dans un SEUL message avec TROIS appels Agent tool :

```
[Message avec trois tool calls] :

Agent(
  description: "Research Anthropic + Boris Cherny AI architecture",
  prompt: <Prompt A>,
  subagent_type: "agent-websearch"
)

Agent(
  description: "Research Google OpenAI Cursor AI config practices",
  prompt: <Prompt B>,
  subagent_type: "agent-websearch"
)

Agent(
  description: "Research LLM architecture metrics and benchmarks",
  prompt: <Prompt C>,
  subagent_type: "agent-websearch"
)
```

---

## Structured Handoff — Compression entre phases

### Phase 1 → Phase 3

Compresser les findings en <1000 mots en format PHASE_OUTPUT :

```
## PHASE_OUTPUT — Phase 1: RESEARCH
- **Findings:**
  1. CLAUDE.md : concis <150L, anti-pattern over-specified, @imports pour progressive disclosure
  2. Boris : glob/grep > RAG, code quality = double-digit impact, verification = 2-3x quality
  3. Hooks = deterministe, CLAUDE.md = advisory
  4. AGENTS.md : 25+ outils, 32 KiB, 6 sections, three-tier boundaries
  5. .cursor/rules/*.mdc <100L, .cursorrules deprecie
  6. SWE-bench : 3+ fichiers = hard, patches moyens 33 LOC / 1.7 fichiers
  7. Types : reduction >50% erreurs compilation (arXiv 2504.09246)
  8. Build & Verification : levier n°1 qualite (Factory.ai, Boris Cherny)
  9. Dev Environment : reproductibilite = prerequis (Factory.ai, Kodus)
  10. {stack-specific findings}
- **Decisions:** rubrique 10 dimensions construite
- **Next phase needs:** stack_profile pour adapter le scan
- **Confidence:** {0-100}
- **Gaps:** {ce qui n'a pas pu etre trouve}
```

### Phase 3 → Phase 4

Compresser le scan en <500 mots — garder les metriques brutes :

```
## PHASE_OUTPUT — Phase 3: SCAN
- **Findings:**
  - {N} fichiers, ~{N} LOC
  - {N} fichiers >300, {N} >500, {N} >1000
  - Max nesting: {N}, Pattern: {type}
  - Circular deps: {N}, TS strict: {yes/no}, any: {N}
  - CLAUDE.md: {status}, AGENTS.md: {status}
  - Hooks: {N} deterministes, @imports: {yes/no}
  - .claude/agents/: {status}
  - Test coverage: {N}%, CI: {yes/no}
  - Build: {1-command?}, Lock files: {status}
  - Dev env: setup script {yes/no}, container {yes/no}
  - Git: {N}% mono-file, avg patch {N} LOC
  - Context budget: ~{N} tokens/feature ({N}% fenetre)
- **Decisions:** metriques recueillies pour scoring
- **Next phase needs:** scoring sur 10 dimensions
- **Confidence:** {0-100}
- **Gaps:** {ce qui n'a pas pu etre mesure}
```

---

## Orchestrator Responsibilities

L'orchestrateur gere :

1. **Phase 1** — Spawner les trois agent-websearch en parallele, attendre, extraire, fusionner, PHASE_OUTPUT
2. **Phase 2** — Detection locale (Glob, Read, Bash) — aucun agent, PHASE_OUTPUT
3. **Phase 3** — Spawner agent-explore avec contexte compresse, attendre, PHASE_OUTPUT
4. **Phase 4** — SCORING : appliquer rubrique aux metriques — calcul local, PHASE_OUTPUT
5. **Phase 5** — CROSS-REF : croiser best practices avec architecture — local, PHASE_OUTPUT
6. **Phase 6** — GENERATE : produire artefacts — ecriture locale, PHASE_OUTPUT
7. **Phase 7** — REPORT : rapport markdown + resume terminal

L'orchestrateur NE duplique PAS le travail des agents. Il ne fait PAS de web search ni d'exploration de code (sauf Phase 2 detection).
