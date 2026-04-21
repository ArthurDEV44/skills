---
model: opus
name: meta-audit
description: "Audit autonome complet de codebase avec rapport PRD. Pipeline en 10 phases : intake & strategie, research web, static analysis (grounding signals), deep scan (3 agents standard ou 5 agents extended), docs conditionnels, extraction & scoring AutoSCORE 2-pass, validation 4 micro-validators, self-brainstorm evidence-based, PRD generation write-prd compliant, output. Scoring hybride : static analysis + verificateurs deterministes + jugement LLM. Mode re-audit incremental. Invoke with /meta-audit [project-path?] [--focus area?] [--re-audit path]."
argument-hint: "[project-path?] [--focus area?] [--re-audit path?]"
allowed-tools: Read, Grep, Glob, Bash, Agent, Write(tasks/*, .meta/*)
---

**Pre-detected context:**
- Stack files: !`ls package.json Cargo.toml pyproject.toml go.mod 2>/dev/null || echo "none detected"`
- AI config: !`ls CLAUDE.md AGENTS.md .cursor/rules/*.mdc 2>/dev/null || echo "none"`
- Source file count: !`find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -o -name "*.rs" -o -name "*.py" -o -name "*.go" \) -not -path "*/node_modules/*" -not -path "*/target/*" -not -path "*/.venv/*" -not -path "*/vendor/*" -not -path "*/dist/*" 2>/dev/null | wc -l`

# meta-audit — Audit Complet de Codebase avec Rapport PRD Autonome

## Overview

Pipeline autonome en 10 phases qui audite un codebase complet et produit un rapport structure au format PRD (compatible `/write-prd`, `/implement-story`, `/review-story`).

**Innovations cles :**
- **AutoSCORE 2-pass** (arXiv 2509.21910, domaine: educational scoring — pattern transfere au code audit) — Extraction structuree des evidences, puis scoring sur le JSON d'extraction. Separe les deux passes pour eviter la conflation.
- **Static analysis comme grounding** — Les outils du stack (eslint, clippy, ruff, semgrep) fournissent des signaux deterministes aux agents avant le scan LLM.
- **4 micro-validators** (inspire CodeMender, Google DeepMind) — Citation verification deterministe + false positive detection + score coherence, chacun purpose-scoped.
- **Self-brainstorm** — Opus repond aux questions de `/write-prd` en s'appuyant sur les findings valides. Zero interaction utilisateur.
- **Structured agent returns** — Chaque agent retourne ses findings dans un format structure (score, metrics, issues, strengths). L'orchestrateur collecte et synthetise.

**Scaling intelligent :**
- **Standard** (<200 fichiers source) — 3 agents-explore specialises en parallele (structural, safety, runtime)
- **Extended** (200+ fichiers source) — 5 agents-explore en parallele, un par dimension d'audit

**Outputs :**
1. Un PRD d'audit complet (`./tasks/audit-prd-{project}.md`) — epics et stories de remediation
2. Un status JSON (`./tasks/audit-prd-{project}-status.json`)
3. Un rapport brut d'audit (`.meta/audit-report-{project}.md`) — scores, metriques, findings detailles, precision estimee

## Execution Flow

```
$ARGUMENTS -> [project-path?] [--focus?]
     |
     v
+-------------------+
|    Phase 0:       |
|    INTAKE &       |  <- Detect stack, measure size, choose strategy
|    STRATEGY       |
+--------+----------+
         |
         v
+-------------------+
|    Phase 1:       |
|    RESEARCH       |  <- agent-websearch x2 (parallel)
|    (web)          |     Best practices + common issues for detected stack
+--------+----------+
         | typed handoff (1000-1500 tokens)
         v
+-------------------+
|    Phase 2:       |
|    STATIC         |  <- eslint/clippy/ruff/semgrep JSON outputs
|    ANALYSIS       |     Grounding signals for Phase 3 agents
+--------+----------+
         |
         v
+-------------------+
|    Phase 3:       |
|    DEEP SCAN      |  <- 3 agents (standard) ou 5 agents (extended)
|  agent-explore    |     Return structured findings to orchestrator
+--------+----------+
         |
         v
+-------------------+
|    Phase 4:       |
|    DOCS           |  <- agent-docs (conditional, si libraries cles)
|    (optional)     |
+--------+----------+
         |
         v
+-------------------+
|    Phase 5:       |
|    EXTRACTION &   |  <- AutoSCORE 2-pass: extract JSON then score
|    SCORING        |     Merge findings, score 6 dimensions, rank issues
+--------+----------+
         |
         v
+-------------------+
|    Phase 6:       |
|    VALIDATION     |  <- 4 micro-validators (cite-check, fp-filter, score-coherence)
|    (3 validators) |     Precision gate 85%
+--------+----------+
         |
         v
+-------------------+
|    Phase 7:       |
|    SELF-BRAINSTORM|  <- Opus repond aux questions write-prd
|    (autonome)     |     Evidence-based, zero interaction
+--------+----------+
         |
         v
+-------------------+
|    Phase 8:       |
|    PRD GENERATION |  <- Format write-prd, epics = dimensions, stories = fixes
|                   |     Self-validation checklist
+--------+----------+
         |
         v
+-------------------+
|    Phase 9:       |
|    OUTPUT         |  <- Save PRD + status JSON + audit report
+-------------------+
```

## Runtime Output Format

Avant chaque phase, afficher :

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Phase N/9] NOM_DE_PHASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Entre les phases : `───────────────────────────────`

En mode extended (200+ fichiers), afficher aussi :
```
[EXTENDED] 5 agents d'audit lances en parallele
[EXTENDED] audit-archi     -> Architecture & Structure
[EXTENDED] audit-quality   -> Code Quality & Patterns
[EXTENDED] audit-security  -> Security & Dependencies
[EXTENDED] audit-tests     -> Testing & CI/CD
[EXTENDED] audit-perf-dx   -> Performance & DX
```

## Compaction Instructions

Si le contexte approche la limite, l'auto-compaction DOIT preserver :
1. Le `project_profile` complet de Phase 0
2. Les outputs JSON de static analysis (Phase 2)
3. Les findings structures retournes par les agents Phase 3
4. Tous les IDs d'issues (`ISS-NNN`) avec leur severity
5. Tous les scores par dimension et sous-critere
6. Les chemins des fichiers de sortie

## Phase-by-Phase Execution

### Phase 0 — INTAKE & STRATEGY

Print: `[Phase 0/9] INTAKE & STRATEGY`

**0a. Parse $ARGUMENTS :**

| Argument | Effet |
|----------|-------|
| `project-path` | cd vers ce path avant l'audit (defaut: cwd) |
| `--focus X` | Concentre l'audit sur une dimension specifique (architecture, quality, security, testing, performance, dx) |
| `--re-audit path` | Mode incremental : charge un `audit-report-*.md` precedent et diff les findings (Phase 3 scanne uniquement les fichiers modifies + dimensions degradees) |

**0b. Detecter le stack :**

Lancer en parallele :
```
Glob: Cargo.toml, package.json, pyproject.toml, go.mod
Glob: next.config.*, tsconfig.json, vite.config.*
Glob: CLAUDE.md, AGENTS.md, .cursor/rules/*.mdc
Glob: docker-compose.*, Dockerfile
Glob: .github/workflows/*.yml, .gitlab-ci.yml
```

**0c. Mesurer la taille du projet :**

Utiliser Glob pour compter les fichiers source par type, puis Bash pour le LOC :

```
Glob: **/*.ts, **/*.tsx, **/*.js, **/*.jsx (exclure node_modules, dist)
Glob: **/*.rs (exclure target)
Glob: **/*.py (exclure .venv, __pycache__)
Glob: **/*.go (exclure vendor)
```

Puis pour estimer le LOC total :
```bash
find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.rs" -o -name "*.py" -o -name "*.go" \) \
  -not -path "*/node_modules/*" -not -path "*/target/*" -not -path "*/.venv/*" -not -path "*/vendor/*" -not -path "*/dist/*" \
  | xargs wc -l 2>/dev/null | tail -1
```

**0d. Decision de strategie :**

```
SI fichiers_source >= 200:
    mode = EXTENDED (5 agents paralleles, un par dimension)
SINON:
    mode = STANDARD (3 agents paralleles, dimensions groupees)
```

**0e. Construire le profil projet :**

```
project_profile = {
  name: "{directory name or manifest name}",
  path: "{absolute path}",
  language: "TypeScript | Rust | Python | Go | ...",
  framework: "Next.js | Axum | FastAPI | ...",
  source_files: N,
  loc_estimate: N,
  mode: "STANDARD | EXTENDED",
  existing_ai_config: ["CLAUDE.md", ...] | [],
  has_tests: true | false,
  has_ci: true | false,
  maturity: "early-stage | growth | mature",  // Infere de: git shortlog -sn | wc -l, presence monitoring, CI sophistication
  focus: "{dimension}" | null,
  re_audit: "{path to previous audit report}" | null
}
```

**GATE :** Stack detecte. Taille mesuree. Strategie choisie.

---

### Phase 1 — RESEARCH (agent-websearch x2, parallele)

Print: `[Phase 1/9] RESEARCH — Best practices pour {stack}`

Lancer **deux agent-websearch en parallele** dans un SEUL message :

**Agent A — Best practices stack-specifiques :**

```
Agent(
  description: "Research {language}/{framework} best practices",
  prompt: <see references/agent-protocols.md — Prompt Research A>,
  subagent_type: "agent-websearch"
)
```

Cible :
- Patterns d'architecture recommandes pour {framework}
- Code quality patterns et conventions du langage
- Testing best practices pour {stack}
- DX et tooling recommande
- Anti-patterns connus

**Agent B — Securite et issues communes :**

```
Agent(
  description: "Research {stack} common issues and security",
  prompt: <see references/agent-protocols.md — Prompt Research B>,
  subagent_type: "agent-websearch"
)
```

Cible :
- OWASP considerations pour {stack}
- Vulnerabilites courantes des dependencies {framework}
- Performance pitfalls communs
- Tech debt patterns recurrents dans les projets {framework}
- Metriques de qualite et seuils du marche

Attendre la completion. Compresser en **typed handoff** (1000-1500 tokens) :

```
Research context for audit:

stack: {language}/{framework}

best_practices:
- text: "{finding}" | source: "{url}" | tier: T1|T2|T3|T4 | date: "YYYY-MM"
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
contradictions: ["{claim_a} vs {claim_b}"]
gaps: ["{what was not found}"]
query_coverage: high|medium|low
```

**GATE :** Research complete. Typed handoff pret (1000-1500 tokens).

---

### Phase 2 — STATIC ANALYSIS (grounding signals)

Print: `[Phase 2/9] STATIC ANALYSIS — Grounding signals pour {stack}`

Executer les outils de static analysis du stack detecte. Les outputs JSON servent de **grounding signals** pour les agents Phase 3 — ils fournissent des faits deterministes que les agents LLM contextualisent et priorisent.

**2a. Selectionner les outils selon le stack :**

| Stack | Outils | Commande |
|-------|--------|----------|
| TypeScript/JS | eslint | `bunx eslint . --format json --no-error-on-unmatched-pattern 2>/dev/null \| head -500` |
| TypeScript | tsc | `bunx tsc --noEmit --pretty false 2>&1 \| head -100` |
| Rust | clippy | `cargo clippy --message-format=json 2>/dev/null \| head -500` |
| Python | ruff | `ruff check . --output-format=json 2>/dev/null \| head -500` |
| Python | mypy | `mypy . --no-error-summary --no-color 2>&1 \| head -100` |
| Go | go vet | `go vet ./... 2>&1 \| head -100` |
| Tout stack | semgrep | `semgrep --config=auto --json --quiet 2>/dev/null \| head -500` |

**2b. Regles d'execution :**

1. Executer uniquement les outils installes/configurables dans le projet (check `package.json`, `Cargo.toml`, etc.)
2. Si aucun outil disponible → skip avec : `[Phase 2/9] STATIC ANALYSIS — Skip (aucun outil disponible)`
3. Limiter chaque output a 500 lignes pour rester dans le budget contexte
4. Capturer les codes de sortie : 0 = clean, >0 = findings
5. Ne PAS installer d'outils — utiliser uniquement ce qui est deja configure

**2c. Formatter les grounding signals :**

```
static_analysis_signals = {
  tools_run: ["eslint", "tsc"],
  summary: {
    errors: N,
    warnings: N,
    info: N
  },
  top_findings: [
    { tool: "eslint", rule: "no-explicit-any", count: N, severity: "warn", files: ["src/api.ts:42", ...] },
    { tool: "tsc", code: "TS2345", count: N, message: "Argument of type...", files: [...] },
    ...
  ],
  clean_dimensions: ["security" if 0 semgrep findings, ...]
}
```

**GATE :** Static analysis complete. Grounding signals prets pour Phase 3.

---

### Phase 3 — DEEP SCAN

Print: `[Phase 3/9] DEEP SCAN — Mode {STANDARD | EXTENDED}`

Chaque agent recoit :
1. Le typed handoff de Phase 1 (research context)
2. Les grounding signals de Phase 2 (static analysis)
3. L'instruction de retourner ses findings dans le format structure defini dans agent-protocols.md

#### Mode STANDARD (<200 fichiers source)

Lancer **3 agents-explore specialises en parallele** dans un SEUL message :

```
Agent(
  name: "scan-structural",
  description: "Audit architecture, quality, and DX",
  prompt: <see references/agent-protocols.md — Prompt Standard: Structural>,
  subagent_type: "agent-explore"
)
Agent(
  name: "scan-safety",
  description: "Audit security and testing",
  prompt: <see references/agent-protocols.md — Prompt Standard: Safety>,
  subagent_type: "agent-explore"
)
Agent(
  name: "scan-runtime",
  description: "Audit performance and observability",
  prompt: <see references/agent-protocols.md — Prompt Standard: Runtime>,
  subagent_type: "agent-explore"
)
```

| Agent | Dimensions | Focus |
|-------|-----------|-------|
| `scan-structural` | Architecture + Quality + DX | Structure, modules, boundaries, typing, naming, git analysis, tooling, docs |
| `scan-safety` | Security + Testing | Secrets, auth, validation, OWASP, test coverage, CI, fixtures |
| `scan-runtime` | Performance + Observability | N+1, async, caching, memory, logging, error tracking, health endpoints |

Si `--focus` est specifie, l'agent couvrant la dimension ciblee recoit un focus override (60% d'effort), les deux autres font un scan abrege (3-5 metriques cles par dimension).

**Circuit breaker (demand-driven) :** Si le codebase a 150+ fichiers source, chaque agent commence par les entry points (main, handlers, routes) et trace les call chains critiques avant d'elargir. Priorite a la profondeur sur les chemins critiques plutot qu'a la couverture exhaustive de tous les fichiers.

#### Mode EXTENDED (200+ fichiers source)

Lancer **5 agents-explore en parallele** dans un **SEUL message**, un par dimension d'audit. Utilise le tool Agent() standard (retour direct a l'orchestrateur).

```
Agent(
  name: "audit-archi",
  description: "Audit architecture and structure",
  prompt: <see references/agent-protocols.md — Prompt Extended: Architecture>,
  subagent_type: "agent-explore"
)
Agent(
  name: "audit-quality",
  description: "Audit code quality and patterns",
  prompt: <see references/agent-protocols.md — Prompt Extended: Quality>,
  subagent_type: "agent-explore"
)
Agent(
  name: "audit-security",
  description: "Audit security and dependencies",
  prompt: <see references/agent-protocols.md — Prompt Extended: Security>,
  subagent_type: "agent-explore"
)
Agent(
  name: "audit-tests",
  description: "Audit testing and CI/CD",
  prompt: <see references/agent-protocols.md — Prompt Extended: Testing>,
  subagent_type: "agent-explore"
)
Agent(
  name: "audit-perf-dx",
  description: "Audit performance and DX",
  prompt: <see references/agent-protocols.md — Prompt Extended: Perf & DX>,
  subagent_type: "agent-explore"
)
```

| Agent | Dimension | Focus |
|-------|-----------|-------|
| `audit-archi` | Architecture & Structure | Modules, boundaries, imports, layers, file organization |
| `audit-quality` | Code Quality & Patterns | Naming, typing, error handling, DRY, consistency |
| `audit-security` | Security & Dependencies | OWASP, secrets, auth, input validation, dependency vulns |
| `audit-tests` | Testing & CI/CD | Coverage, quality, fixtures, CI config, test patterns |
| `audit-perf-dx` | Performance & DX | N+1, async, caching, docs, tooling, dev workflow |

Si `--focus` est specifie, l'agent cible fait un deep-dive, les autres font un scan abrege.

**Circuit breaker extended :** Chaque agent commence par les entry points de sa dimension (routes pour security, test files pour testing, config pour DX) et trace les chemins critiques avant d'elargir.

Les agents retournent leurs findings dans le format structure defini dans [references/agent-protocols.md](references/agent-protocols.md) — Extended Findings Output Format. L'orchestrateur collecte les retours pour la synthese Phase 5.

**GATE :** Scan complet. Findings de toutes les dimensions collectes depuis les retours agents.

---

### Phase 4 — DOCS (conditionnel)

Print: `[Phase 4/9] DOCS — Documentation des libraries cles`

**Condition :** Si le typed handoff de Phase 1 identifie des libraries specifiques avec des questions ouvertes, OU si le scan de Phase 3 revele des usages non-standards de libraries.

Si la condition est remplie :

```
Agent(
  description: "Fetch docs for {library}",
  prompt: <see references/agent-protocols.md — Phase 4 — Prompt agent-docs>,
  subagent_type: "agent-docs"
)
```

Max 2 libraries, 3 ctx7 calls total.

Si la condition n'est pas remplie, skip avec : `[Phase 4/9] DOCS — Skip (aucune library specifique a documenter)`

**GATE :** Docs collectes (ou phase skippee).

---

### Phases 5-8 — SCORING, VALIDATION, BRAINSTORM, PRD

Lire les instructions completes dans [references/pipeline-phases.md](references/pipeline-phases.md).

Resume :
- **Phase 5** — AutoSCORE 2-pass : extraction JSON des evidences, puis scoring pointwise sur le JSON. Classification des issues avec deployability AXIOM.
- **Phase 6** — 4 micro-validators : cite-check (deterministe), fp-filter (agent-explore), score-coherence (agent-explore), spike-resolver (deterministe+agent-docs — resout les questions techniques avant PRD). Precision gate 85%.
- **Phase 7** — Self-brainstorm : Opus repond aux questions write-prd en citant des evidences validees. Filtre Regle 4 (CLAUDE.md contradictions) et Regle 5 (maturite projet). Voir [references/self-brainstorm.md](references/self-brainstorm.md).
- **Phase 8** — PRD generation au format exact write-prd + self-validation checklist (15 items) + rapport brut d'audit. Problem Statement reflete le scope filtre. Technical Considerations pre-resolues.

**GATES :** Chaque phase a sa gate detaillee dans le fichier de reference.

---

### Phase 9 — OUTPUT

Print: `[Phase 9/9] OUTPUT`

**9a. Creer les repertoires si necessaire :**

```bash
mkdir -p tasks .meta
```

**9b. Sauvegarder les fichiers :**

```
./tasks/audit-prd-{project-name}.md          — le PRD
./tasks/audit-prd-{project-name}-status.json — le status tracker
.meta/audit-report-{project-name}.md         — le rapport brut
```

**9c. Afficher le resume terminal :**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIT CODEBASE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Projet :** {project_name}
**Stack :** {language} / {framework}
**Mode :** {STANDARD (3 agents) | EXTENDED (5 agents)}

**Score global : {N}/100 — Grade {grade}**
**Precision estimee : {N}%**

| Dimension | Score | Grade | N/A |
|-----------|-------|-------|-----|
| Architecture & Structure | {N} | {grade} | {N} |
| Code Quality | {N} | {grade} | {N} |
| Security | {N} | {grade} | {N} |
| Testing | {N} | {grade} | {N} |
| Performance | {N} | {grade} | {N} |
| Developer Experience | {N} | {grade} | {N} |

**Static Analysis :** {N} errors, {N} warnings ({tools used})

**Issues identifiees :** {N} total ({N} grounded, {N} ungrounded retires)
  - CRITICAL: {N}  HIGH: {N}  MEDIUM: {N}  LOW: {N}

**PRD genere :** {N} epics, {N} stories
  - P0 (Must Have): {N} stories
  - P1 (Should Have): {N} stories
  - P2 (Could Have): {N} stories

**Fichiers :**
- `tasks/audit-prd-{name}.md` — PRD de remediation
- `tasks/audit-prd-{name}-status.json` — Status tracker
- `.meta/audit-report-{name}.md` — Rapport d'audit detaille

**Next Steps :**
- `/implement-story tasks/audit-prd-{name}.md US-001` pour demarrer
- `/review-story tasks/audit-prd-{name}.md` pour reviewer
- `/meta-archi` pour un audit LLM-readiness complementaire

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**GATE :** Tous les fichiers sauvegardes. Resume affiche.

---

## Hard Rules

### Pipeline
1. Phase 0 (INTAKE) TOUJOURS en premier — detecter le stack et choisir la strategie avant tout.
2. Phase 1 (RESEARCH) TOUJOURS avant tout scan — les best practices guident l'audit.
3. Phase 1 lance DEUX agent-websearch en PARALLELE — jamais sequentiellement.
4. Phase 2 (STATIC ANALYSIS) TOUJOURS avant le deep scan — les grounding signals guident les agents.
5. Phase 3 en mode EXTENDED lance 5 agents-explore en parallele via Agent() standard (un par dimension).
6. Phase 3 en mode STANDARD lance 3 agents-explore specialises en parallele (structural, safety, runtime).
7. Phase 4 (DOCS) est CONDITIONNELLE — skip si aucune library specifique a documenter.
8. Phase 5 (EXTRACTION & SCORING) separe l'extraction du scoring en 2 passes — jamais de scoring direct sur les findings bruts.
9. Phase 6 (VALIDATION) lance 4 micro-validators — cite-check deterministe, fp-filter LLM, score-coherence LLM, spike-resolver deterministe+agent-docs.
10. Phase 7 (SELF-BRAINSTORM) cite une evidence specifique validee pour CHAQUE decision. Filtre les stories contradisant CLAUDE.md (Regle 4) et prematurees pour le stade projet (Regle 5).
11. Phase 8 utilise le format EXACT de `/write-prd` — compatible implement-story/review-story. Le Problem Statement reflete le scope filtre, pas l'audit original. Les Technical Considerations sont pre-resolues par le spike-resolver.
12. Phase 8 execute la self-validation checklist de write-prd — 15 items obligatoires.

### Scaling
13. Le seuil de 200 fichiers source declenche le mode EXTENDED (5 agents) — en-dessous, mode STANDARD (3 agents).
14. En mode EXTENDED, les 5 agents tournent en PARALLELE via le tool Agent() standard — jamais en sequentiel.
15. Chaque agent recoit le typed handoff de Phase 1 + les grounding signals de Phase 2 — pas les outputs bruts.
16. `--focus X` fonctionne en STANDARD et EXTENDED : l'agent cible deep-dive, les autres font un scan abrege.

### Qualite
17. Chaque issue a un ID unique (`ISS-NNN`), un sous-critere (`A1`, `S3`...), une severity, un score deployability AXIOM, et une evidence (file:line ou URL).
18. Les stories PRD sont independamment completables en une session AI agent.
19. Max 25 stories — au-dela, suggerer des phases de release.
20. Le rapport est en FRANCAIS.
21. Les heuristiques sans base empirique sont marquees comme telles.
22. Les verificateurs deterministes et la static analysis fixent un plancher de score — le LLM ne peut que monter.
23. Un sous-critere sans evidence suffisante est marque N/A — jamais de score invente.
24. La precision estimee est calculee et affichee — precision gate a 85%.

### Idempotence (source: Anthropic agent design, Agent Patterns)
25. Les sous-criteres SANS verificateur deterministe produisent des issues **INFO** (pas MEDIUM/LOW). Les issues INFO apparaissent dans le rapport brut mais **ne generent PAS de stories** dans le PRD. Voir audit-dimensions.md — section "Regle d'idempotence".
26. **Cap par sous-critere : max 3 issues.** Au-dela, emettre les 3 plus critiques + compteur.
27. **Cap re-scan : max 1 par dimension.** Si le re-scan ne produit pas d'evidence, le sous-critere reste N/A.
28. Le score-coherence validator ne peut PAS recommander un re-scan plus d'une fois par dimension. Si un conflit entre agents persiste apres adjudication, garder le score existant ("when evidence is equal, keep existing score").

## Error Handling

| Scenario | Action |
|----------|--------|
| Pas de codebase detecte | Avertir l'utilisateur. Proposer de specifier le path via $ARGUMENTS. |
| agent-websearch echoue | Continuer avec connaissances du modele. Noter "Research web indisponible." |
| agent-explore echoue | Faire un scan superficiel avec Glob/Grep depuis l'orchestrateur. |
| Agent extended echoue | Utiliser les resultats partiels. Noter la dimension incomplete. |
| Static analysis echoue | Continuer sans grounding signals. Noter "Phase 2 skippee." |
| Trop d'issues (>30) | Regrouper par theme. Prioriser les CRITICAL/HIGH. Suggerer un phasing. |
| Score global >85 | Feliciter. Lister les optimisations marginales. Ne pas inventer de problemes. |
| ctx7 echoue (Phase 4) | Skip la phase DOCS. S'appuyer sur la research web. |
| Self-validation echoue | Corriger l'item defaillant. Max 2 corrections. Si echec apres 2 → sauver avec `[VALIDATION_WARNINGS]`. |
| Validator echoue | Continuer avec les validators restants. Noter "Validator {name} skippee." |
| --re-audit sans fichier precedent | Avertir et lancer un audit complet standard. |
| Precision < 85% | Signaler `[PRECISION_WARNING]` dans le rapport. Ne pas bloquer l'audit. |

## DO NOT

- Modifier des fichiers du projet — ce workflow est READ-ONLY (sauf `tasks/`, `.meta/`).
- Demander des questions a l'utilisateur pendant l'audit — le self-brainstorm est AUTONOME.
- Scorer sans metriques — chaque score tracable a une mesure ou marque N/A.
- Generer des stories non-actionnables ("ameliorer la qualite") — chaque story a un scope precis.
- Generer des stories a partir d'issues INFO (sous-criteres sans verificateur deterministe) — les issues INFO sont des observations, pas des corrections.
- Utiliser des connaissances generiques dans le self-brainstorm — citer des evidences d'audit.
- Lancer toutes les phases simultanement — respecter l'ordre sequentiel (research → static → scan → docs → extraction → validation → brainstorm → PRD).
- Passer les outputs bruts aux agents — toujours compresser en typed handoff.
- Ignorer le format write-prd — le PRD doit etre 100% compatible implement-story.
- Gonfler ou minimiser les scores.
- Ecrire le PRD en anglais (sauf les noms techniques et code).
- Forcer un score 0-10 quand l'evidence est insuffisante — utiliser N/A.
- Installer des outils de static analysis non-presents dans le projet.

## Constraints (Three-Tier)

Applique le modele [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md).

### ALWAYS
- Detecter le stack avant de lancer l'audit
- Lancer la research web en premier
- Executer la static analysis avant le deep scan
- Scorer chaque dimension avec des metriques (ou N/A si insuffisant)
- Citer des evidences dans le self-brainstorm
- Self-valider le PRD avant sauvegarde
- Calculer et afficher la precision estimee
- Ecrire en francais

### ASK FIRST
- Rien — ce workflow est entierement autonome

### NEVER
- Modifier des fichiers du projet (sauf tasks/ et .meta/)
- Interrompre l'audit pour poser des questions
- Scorer sans metriques mesurables (utiliser N/A)
- Presenter des heuristiques comme des faits empiriques
- Citer des sources inexistantes ou non-verifiees

## Done When

- [ ] Phase 0 — Stack detecte, taille + LOC mesures, strategie choisie
- [ ] Phase 1 — Research complete avec typed handoff (1000-1500 tokens)
- [ ] Phase 2 — Static analysis executee, grounding signals prets
- [ ] Phase 3 — Scan multi-dimensionnel complet, findings structures collectes depuis les retours agents
- [ ] Phase 4 — Docs collectes (ou phase skippee avec justification)
- [ ] Phase 5 — Extraction JSON + scoring 2-pass complete, issues classees avec deployability AXIOM
- [ ] Phase 6 — 4 micro-validators completes, precision estimee calculee
- [ ] Phase 7 — Self-brainstorm complet, toutes decisions citent des evidences validees
- [ ] Phase 8 — PRD genere, self-validation passee (15 items, ou `[VALIDATION_WARNINGS]`)
- [ ] Phase 9 — PRD + status JSON + rapport brut sauvegardes
- [ ] Resume affiche dans le terminal avec precision estimee
- [ ] Tous les agents d'audit completes

## References

- [Pipeline Phases 5-8](references/pipeline-phases.md) — scoring AutoSCORE, validation, self-brainstorm, PRD generation
- [Audit Dimensions](references/audit-dimensions.md) — rubrique de scoring, metriques, seuils par dimension
- [Agent Protocols](references/agent-protocols.md) — prompt templates pour chaque agent (standard, extended, validators)
- [Self-Brainstorm Protocol](references/self-brainstorm.md) — comment Opus repond aux questions PRD de maniere autonome
- [PRD Template](@~/.claude/skills/write-prd/references/prd-template.md) — format exact du PRD (shared avec write-prd)
- [Brainstorm Protocols](@~/.claude/skills/write-prd/references/brainstorm-protocols.md) — question templates et self-validation checklist (shared avec write-prd)
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — CAN/CANNOT table, budgets d'appels
- [Three-Tier Constraints](@~/.claude/skills/_shared/three-tier-constraints.md) — modele ALWAYS/ASK FIRST/NEVER

## Methodological References

- AutoSCORE (arXiv 2509.21910) — scoring pointwise 2-pass (domaine: educational scoring, pattern transfere au code audit)
- AXIOM (arXiv 2512.20159) — calibration multisource code quality, echelle deployability adaptee (0-5 specifique a ce skill)
- CodeMender (Google DeepMind, blog post Oct 2025) — judge agent purpose-scoped, deterministic verifiers before LLM
- RepoAudit (arXiv 2501.18160, ICML 2025) — demand-driven codebase traversal, 78% precision
- DeCE (arXiv 2509.16093) — decomposition precision/recall pour evaluation LLM
- AgentAuditor (arXiv 2602.09341) — adjudication evidence-based multi-agent via reasoning trees
- arXiv 2504.09246 — les types stricts reduisent de >50% les erreurs de compilation dans la generation de code AI
- Anthropic Engineering — context engineering, structured agent returns, 1000-2000 token handoffs
- Boris Cherny — verification-before-posting, fresh-context pattern, grep-beats-vector-search
