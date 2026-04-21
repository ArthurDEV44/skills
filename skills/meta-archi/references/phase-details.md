# Phase Details — Instructions detaillees par phase

Ce fichier contient les instructions detaillees pour chaque phase du pipeline meta-archi.
Charge a la demande via progressive disclosure depuis SKILL.md.

---

## Phase 2 — DETECT (orchestrateur, instant)

### 2a. Detection du stack

Lancer en parallele :

```
Glob: Cargo.toml, package.json, pyproject.toml, go.mod
Glob: next.config.*, tsconfig.json
Glob: CLAUDE.md, AGENTS.md
Glob: .cursorrules, .cursor/rules/*.mdc
Glob: .github/copilot-instructions.md, .aiexclude
Glob: .claude/settings.json
```

| Signal | Detection |
|--------|-----------|
| Langage principal | Extensions, manifest principal |
| Framework | next.config (Next.js), Cargo.toml [dependencies] (Axum, Actix), etc. |
| Monorepo | Cargo workspace, pnpm-workspace.yaml, turborepo.json, nx.json |
| Build tool | vite.config, webpack, turbopack, cargo |
| Test framework | vitest, jest, cargo test, pytest |
| AI config existant | CLAUDE.md, AGENTS.md, .cursor/rules/*.mdc, .cursorrules (legacy), .github/copilot-instructions.md, .aiexclude |
| Hooks Claude Code | .claude/settings.json → hooks section |

### 2b. Mesurer la taille du projet

Utiliser Glob (pas find) pour compter les fichiers source, puis Bash pour le LOC :

```
Glob: **/*.{ts,tsx,rs,js,jsx}  (exclure node_modules/, target/)
→ compter le nombre de resultats

Bash: wc -l sur les fichiers trouves par Glob
→ LOC total estime
```

### 2c. Construire le stack_profile

```
stack_profile = {
  language: "TypeScript" | "Rust" | "TypeScript + Rust" | ...,
  framework: "Next.js App Router" | "Axum" | ...,
  monorepo: true | false,
  monorepo_tool: "cargo workspace" | "turborepo" | "nx" | ...,
  project_size: "small (<50 files)" | "medium (50-200)" | "large (200+)",
  loc_estimate: N,
  existing_ai_config: ["CLAUDE.md", "AGENTS.md", ".cursor/rules/", ...] | [],
  has_hooks: true | false,
  test_framework: "vitest" | "cargo test" | ...,
  build_commands: { build: "...", test: "...", lint: "..." },
  entry_points: ["src/app/", "src/main.rs", ...]
}
```

---

## Phase 3 — SCAN (agent-explore)

Spawner un agent-explore avec le prompt defini dans `references/agent-protocols.md — Prompt Scan`.

L'agent mesure :

### 3a. Structure de fichiers
- Profondeur maximale de nesting
- Nombre moyen de fichiers par dossier
- Distribution des tailles de fichiers (fichiers >300, >500, >1000 lignes)
- Top 10 fichiers les plus gros avec LOC

### 3b. Pattern architectural
- Horizontal layers vs vertical slices vs mixte
- Colocalisation : tests a cote du code ? types a cote de l'implementation ?
- Separation des concerns : domain/infra/application layers ?

### 3c. Graphe d'imports
- Top 10 fichiers avec le plus d'imports
- Dependencies circulaires
- Profondeur d'import : combien de hops pour comprendre une feature ?
- Index files : re-exports strategiques ?

### 3d. Type safety
- TypeScript : strict mode ? `any` count ? branded types ?
- Rust : `unwrap()` count vs proper error handling ?
- Schemas de validation (Zod, serde) ?

### 3e. Documentation & AI config
- CLAUDE.md : taille, qualite, anti-pattern "over-specified" ?
- AGENTS.md : taille, 6 sections cles, limite 32 KiB ?
- .cursor/rules/*.mdc vs .cursorrules legacy ?
- Hooks : regles critiques en hooks deterministes ou en CLAUDE.md advisory ?
- @imports pour progressive disclosure ?
- Compaction instructions presentes dans CLAUDE.md ?
- .claude/agents/ : structure, YAML frontmatter, tools scope ?

### 3f. Tests
- Framework, colocalisation, couverture estimee, CI config

### 3g. Build & Verification
- Commandes de build documentees (package.json scripts, Makefile, Cargo.toml) ?
- Tests executables en une seule commande ?
- CI presente (.github/workflows/, .gitlab-ci.yml) ?
- Feedback loop automatisee (pre-commit hooks, lint-staged) ?
- Lock files presents et a jour ?

### 3h. Dev Environment
- Lock files (package-lock.json, pnpm-lock.yaml, Cargo.lock) ?
- .env.example ou .env.template ?
- Scripts de setup (setup.sh, Makefile install) ?
- Containerisation (Dockerfile, docker-compose.yml, devcontainer.json) ?
- Version pinning (.node-version, .tool-versions, rust-toolchain.toml) ?

### 3i. Git Change Analysis (metrique empiriquement validee par SWE-bench)
- Ratio patches mono-fichier sur les derniers 100 commits
- Taille moyenne des patches (LOC ajoutes/supprimes)
- Top 5 fichiers les plus frequemment modifies ensemble (couplage temporel)
- Frequence de modification des god-files identifies en 3a

---

## Phase 5 — CROSS-REF (orchestrateur)

### 5a. Pour chaque best practice de la rubrique :

| Status | Critere |
|--------|---------|
| CONFORME | La best practice est respectee |
| ECART | La best practice n'est pas respectee — impact LLM |
| ABSENT | Le concept n'est pas du tout present |
| NON APPLICABLE | Ne s'applique pas au stack |

### 5b. Pour chaque ecart, construire :

```
{
  best_practice: "Description",
  source_url: "URL",
  source_confidence: "empirique" | "officiel" | "heuristique",
  dimension: "File & Directory Organization | Vertical Cohesion | ...",
  impact_llm: "Pourquoi ca impacte la comprehension LLM — concret",
  current_state: "Etat actuel (file:line si applicable)",
  recommended_state: "Etat cible",
  before: "Structure/code actuel",
  after: "Structure/code recommande",
  effort: "quick-win | medium | strategic",
  priority: 1-5 (1 = plus impactant)
}
```

### 5c. Classer par impact LLM :

1. **Quick Wins** — changements faciles, impact immediat (renommer, ajouter CLAUDE.md)
2. **Restructurations moderees** — refactor de modules, colocalisation de tests
3. **Refactors architecturaux** — passage en vertical slices, restructuration du monorepo

---

## Phase 6 — GENERATE (orchestrateur)

Si `--generate` est passe OU si l'AI config est absente :

### 6a. Generer CLAUDE.md

Respecter :
- < 150 lignes — anti-pattern Anthropic : "if your CLAUDE.md is too long, Claude ignores half of it"
- Chaque ligne doit repondre a : "Supprimer cette ligne causerait-il une erreur de Claude ?" Si non, couper.
- Sections : Commands, Code Style, Architecture, Testing, Boundaries
- Specifique au stack (commandes build/test/lint reelles)
- Liens `@path/to/file` pour progressive disclosure si necessaire
- NE PAS inclure ce que Claude peut inferer du code
- Regles critiques → recommander en commentaire de migrer vers des hooks

Ecrire dans `.meta/generated-CLAUDE.md` (PAS a la racine).

### 6b. Generer AGENTS.md

Respecter :
- Format standard AGENTS.md (https://agents.md/) — 25+ outils, Linux Foundation
- Taille < 32 KiB (limite Codex), contenu cle dans les 4000 premiers chars (limite Copilot)
- 6 sections validees (GitHub blog, 2500+ repos) : Commands, Testing, Project Structure, Code Style, Git Workflow, Boundaries
- Pattern three-tier Boundaries : "Always safe / Ask first / Never"
- Un exemple de code reel > trois paragraphes de description

Ecrire dans `.meta/generated-AGENTS.md`.

### 6c. Generer le plan de remediation

```markdown
## Plan de remediation LLM-readiness

### Phase 1 — Quick Wins (1-2 heures)
1. {action} — Impact: {dimension}, Score +{N} — Confiance: {empirique|officiel|heuristique}

### Phase 2 — Restructurations (1-2 jours)
1. {action} — Impact: {dimension}, Score +{N}

### Phase 3 — Refactors architecturaux (1+ semaine)
1. {action} — Impact: {dimension}, Score +{N}

**Score projete apres remediation complete : {N}/100 ({grade})**
```

Ecrire dans `.meta/remediation-plan.md`.

---

## Phase 7 — REPORT (orchestrateur)

### 7a. Creer `.meta/` si necessaire.

### 7b. Generer le rapport dans `.meta/archi-llm-report.md`

Voir la section "Output Format" dans SKILL.md pour le template exact.

### 7c. Afficher le resume dans le terminal

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE LLM-READINESS AUDIT COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Score global : {N}/100 — Grade {grade}**
**Stack :** {language} / {framework}

| Dimension | Score | Grade |
|-----------|-------|-------|
| ... | ... | ... |

**Context budget :** ~{N} tokens/feature ({N}% fenetre)

**Quick Wins :** {N} (impact immediat)
**Restructurations :** {N} (moyen terme)
**Refactors :** {N} (strategiques)

**Score projete (quick wins) :** {N}/100
**Score projete (complet) :** {N}/100

Artefacts generes :
- `.meta/archi-llm-report.md` (rapport complet)
- `.meta/generated-CLAUDE.md` (si applicable)
- `.meta/generated-AGENTS.md` (si applicable)
- `.meta/remediation-plan.md` (plan d'action)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
