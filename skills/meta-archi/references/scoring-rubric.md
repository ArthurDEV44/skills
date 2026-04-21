# Scoring Rubric — LLM-Readiness Dimensions

## Vue d'ensemble

Le score LLM-readiness mesure a quel point l'architecture d'un codebase facilite la comprehension et la generation de code par les AI coding agents. Il est compose de **10 dimensions** ponderees, scorees de 0 a 100.

**Calibration :** Les seuils sont marques par leur niveau de confiance :
- **Empirique** — valide par SWE-bench, papers academiques, ou benchmarks reproductibles
- **Officiel** — recommande par Anthropic, Google, OpenAI, ou GitHub dans leur documentation
- **Heuristique** — convention de praticiens largement adoptee mais sans validation empirique directe

---

## Dimension 1 — File & Directory Organization (10%)

**Ce qui est mesure :** Taille des fichiers, single responsibility, profondeur de nesting, discoverability.

**Pourquoi ca impacte les LLM :**
- Un fichier de 2000 lignes injecte du bruit dans le context window — l'agent doit filtrer le non-pertinent
- Les LLM generent des fonctions atomiques — code organise de la meme facon = meilleur alignement
- Le vrai constraint technique : les limites d'output tokens rendent l'edition de fichiers longs difficile (Aider benchmark)
- Boris Cherny : "glob and grep bat le RAG" — noms descriptifs et structure plate facilitent la recherche agentique
- Plus le nesting est profond, plus l'agent consomme de tokens de navigation

**Metriques fichiers :**

| Score | LOC moyen | Fichiers >500 LOC | Fichiers >1000 LOC | Confiance |
|-------|-----------|--------------------|--------------------|-----------|
| 90-100 | <200 | 0 | 0 | Heuristique |
| 75-89 | <300 | <3% | 0 | Heuristique |
| 60-74 | <400 | <10% | <3% | Heuristique |
| 45-59 | <500 | <20% | <5% | Heuristique |
| 30-44 | <700 | <30% | <10% | Heuristique |
| 0-29 | >700 | >30% | >10% | Heuristique |

**Metriques directory :**

| Score | Max depth | Nommage | Fichiers/dossier | Confiance |
|-------|-----------|---------|------------------|-----------|
| 90-100 | <=3 | Descriptif, coherent | 3-15 | Heuristique |
| 75-89 | <=4 | Descriptif | 3-20 | Heuristique |
| 60-74 | <=5 | Correct | 2-25 | Heuristique |
| 45-59 | <=6 | Mix bon/cryptique | Variable | Heuristique |
| 30-44 | <=7 | Cryptique (`utils/`, `misc/`) | >25 ou <2 | Heuristique |
| 0-29 | >7 | Incoherent | Extremes | Heuristique |

**Score = moyenne ponderee (fichiers 60%, directory 25%, migration completeness 15%).**

**Sous-critere : Migration Completeness**
- Absence de frameworks/patterns concurrents (ex: pas de mix REST + GraphQL sans separation claire)
- Cohérence des conventions : un seul ORM, un seul test framework, un seul state management
- Boris Cherny : "Always make sure that when you start a migration, you finish the migration" — une codebase half-migrated confuse le modele autant que les humains
- Impact : le modele ne peut pas inferer quel pattern est canonique → generation inconsistante

**Stack-specific :**
- **Rust :** +50 LOC aux seuils (Rust plus verbose a cause des impls/derives).
- **Next.js :** page.tsx et layout.tsx doivent rester <150 LOC.

---

## Dimension 2 — Vertical Cohesion (10%)

**Ce qui est mesure :** Colocalisation feature : tests, types, et implementation dans le meme espace.

**Pourquoi ca impacte les LLM :**
- Un agent qui lit `features/auth/` a le tableau complet sans sauter entre dossiers
- Chaque hop entre dossiers consomme des tokens de contexte
- SWE-bench Verified : 2+ fichiers = hard (avg 2.0 fichiers hard vs 1.03 easy, Ganhotra 2025) — vertical slices minimisent les fichiers par feature
- SWE-bench Pro : 4.1 fichiers en moyenne, 107.4 LOC — les taches realistes sont encore plus exigeantes en localite
- Boris Cherny : "glob and grep bat le RAG" — colocalisation rend la recherche agentique efficace

**Metriques :**

| Score | Critere | Confiance |
|-------|---------|-----------|
| 90-100 | Pure vertical slices : feature folders avec tests+types+code colocalises. Index file par feature. | Heuristique (SWE-bench indirect) |
| 75-89 | Vertical predominant, quelques exceptions (types partages OK, tests colocalises). | Heuristique |
| 60-74 | Mix vertical/horizontal. Tests dans meme dossier OU adjacent. | Heuristique |
| 45-59 | Horizontal predominant avec regroupement logique. | Heuristique |
| 30-44 | Pure horizontal layers. Tests separes dans `__tests__/` ou `test/`. | Heuristique |
| 0-29 | Aucune organisation coherente. | Heuristique |

---

## Dimension 3 — Import Simplicity (8%)

**Ce qui est mesure :** Complexite du graphe d'imports, profondeur de navigation, circularites.

**Pourquoi ca impacte les LLM :**
- Chaque hop d'import = un fichier supplementaire a lire
- Imports circulaires = boucles de comprehension impossibles
- Re-exports propres = API lisible sans lire l'implementation
- SWE-bench Verified : nombre de fichiers a charger = facteur le plus predictif de difficulte
- SWE-bench Pro (1865 taches multi-langages, nouveau standard) : 4.1 fichiers en moyenne — chaque hop d'import supplementaire rapproche du seuil critique

**Metriques :**

| Score | Max hops | Circular deps | Index files | Import style | Confiance |
|-------|----------|---------------|-------------|--------------|-----------|
| 90-100 | <=2 | 0 | Systematiques | Alias (@/) | Heuristique |
| 75-89 | <=3 | 0 | Presents | Alias predominant | Heuristique |
| 60-74 | <=4 | 1-2 | Partiels | Mix alias/relatif | Heuristique |
| 45-59 | <=5 | 3-5 | Rares | Relatif predominant | Heuristique |
| 30-44 | <=6 | 5-10 | Absents | Relatif profond (../../..) | Heuristique |
| 0-29 | >6 | >10 | Absents | Chaotique | Heuristique |

**Sous-critere : Static Traceability**
- Quel % des appels peut etre resolu statiquement (sans runtime) ?
- Absence de magic strings comme identifiants de comportement
- Dynamic dispatch limite (pas de `eval()`, reflection minimale, decorateurs generatifs limites)
- Convention-over-configuration explicite (pas de routing par nom de fichier implicite sans documentation)
- Impact : l'agent ne peut tracer que les chemins statiquement resolvables — le reste est une boite noire

**Stack-specific :**
- **Rust :** `mod.rs`/`lib.rs` = index files. `pub use` = re-export. Dynamic dispatch via `dyn Trait` mesurable.
- **Next.js :** `@/` alias dans tsconfig.paths est le standard. App Router = convention-over-config (documenter les conventions dans CLAUDE.md).

---

## Dimension 4 — Type Expressiveness (13%)

**Ce qui est mesure :** Rigueur du systeme de types et capacite a encoder le sens semantique.

**Pourquoi ca impacte les LLM :**
- La generation type-constrained reduit les erreurs de compilation de >50% (arXiv 2504.09246)
- Les branded types disent "c'est un UserId, pas un string" — meilleure generation
- ClassEval (ICSE 2024) : LLM performent mieux au method-level qu'au class-level — types clairs reduisent la complexite
- SWE-bench 2025 : Python/Go > TS/JS en performance agent — types forts = meilleure comprehension

**Metriques TypeScript :**

| Score | strict | `any` count | Branded types | Validation | Confiance |
|-------|--------|-------------|---------------|------------|-----------|
| 90-100 | true | 0 | Present | Zod/io-ts | Empirique |
| 75-89 | true | 1-5 | Quelques | Zod/io-ts | Empirique |
| 60-74 | true | 5-15 | Non | Aucun | Mixte |
| 45-59 | false | 15-50 | Non | Aucun | Mixte |
| 30-44 | false | 50-100 | Non | Aucun | Heuristique |
| 0-29 | false | >100 | Non | Aucun | Heuristique |

**Metriques Rust :**

| Score | unwrap() count | Error handling | serde | Confiance |
|-------|----------------|----------------|-------|-----------|
| 90-100 | 0-2 (tests only) | Result<> + thiserror | Partout | Heuristique |
| 75-89 | <10 | Result<> + anyhow | Predominant | Heuristique |
| 60-74 | 10-30 | Mix Result/unwrap | Partiel | Heuristique |
| 45-59 | 30-60 | unwrap predominant | Rare | Heuristique |
| 30-44 | 60-100 | unwrap partout | Absent | Heuristique |
| 0-29 | >100 | panic!() partout | Absent | Heuristique |

---

## Dimension 5 — AI Config Quality (12%)

**Ce qui est mesure :** Presence, taille, qualite et couverture cross-tool de la configuration AI, incluant la distinction hooks/advisory et le progressive disclosure.

**Pourquoi ca impacte les LLM :**
- CLAUDE.md est charge a chaque session — c'est la premiere chose que l'agent lit
- Anti-pattern Anthropic : "if your CLAUDE.md is too long, Claude ignores half of it"
- AGENTS.md = standard cross-tool dominant (25+ outils, 60 000+ repos, Agentic AI Foundation / Linux Foundation). Claude Code ne le lit PAS nativement (mars 2026)
- Les hooks sont **deterministes** (garantis), CLAUDE.md est **advisory** (peut etre ignore) — source: Anthropic officiel
- Claude Code supporte **24 lifecycle events** (10 categories) et **4 types de handlers** (command, http, prompt, agent)
- Progressive disclosure via `@path/to/file` reduit la charge initiale sur le context window
- Section "Compact Instructions" dans CLAUDE.md controle ce qui est preserve lors de l'auto-compaction (~83.5%)
- Structure `.claude/agents/` avec YAML frontmatter (14 champs) permet la delegation specialisee

**Metriques :**

| Score | CLAUDE.md | AGENTS.md | Hooks | @imports / Compact Instructions | .claude/agents/ | Confiance |
|-------|-----------|-----------|-------|---------------------------------|-----------------|-----------|
| 90-100 | Present <150L, chirurgical, section Compact Instructions | Present <32KiB, 6 sections, three-tier | Regles critiques en hooks (multiple event types) | @imports + Compact Instructions | Structure 14 champs YAML (incl. memory, effort, isolation) | Officiel |
| 75-89 | Present <200L | Present, 4/6 sections | Quelques hooks PreToolUse/PostToolUse | @imports partiels | Existe avec model, tools, description | Officiel |
| 60-74 | Present <300L | Absent | Absent | Non | Non | Mixte |
| 45-59 | Present >300L (over-specified) | Absent | Absent | Non | Non | Officiel (anti-pattern) |
| 30-44 | Absent | Present seul | Absent | Non | Non | Mixte |
| 0-29 | Absent | Absent | Absent | Non | Non | Officiel |

**Sous-criteres (checklist pour scorer) :**

1. **CLAUDE.md health :**
   - Taille <150L ? (anti-pattern >300L)
   - Sections presentes : Commands, Code Style, Architecture, Testing, Boundaries ?
   - Section `## Compact Instructions` pour controle de la compaction ?
   - Contient des regles qui devraient etre des hooks ? (lint, format, no-commit → hooks)

2. **Progressive disclosure :**
   - Utilise `@path/to/file.md` pour importer du contenu additionnel ?
   - Hierarchie CLAUDE.md multi-niveaux (global, racine, sous-repertoires) ?
   - Skills avec `disable-model-invocation: true` pour exclure du contexte de base ?

3. **Hooks vs Advisory (24 lifecycle events) :**
   - `.claude/settings.json` present avec section hooks ?
   - Categories de hooks utilises (parmi 10 : Session, Prompt, Tools, Notifications, Agents, Compaction, Config, Filesystem, Worktrees, Completion) ?
   - Types de handlers utilises (command, http, prompt, agent) ?
   - Regles critiques (lint, format, block writes) en hooks deterministes ?
   - PreCompact/PostCompact hooks pour gestion de la compaction ?
   - Hooks dans frontmatter skills/subagents (scope delimite) ?

4. **Subagents (14 champs frontmatter) :**
   - `.claude/agents/` existe ?
   - Champs de base : name, description, model, tools/disallowedTools ?
   - Champs avances : permissionMode, maxTurns, skills (preload), mcpServers (scoped) ?
   - Champs infrastructure : memory (persistance cross-session), background, effort (low/medium/high/max), isolation (worktree) ?
   - Scope d'outils restreint par agent (Writer vs Reviewer pattern) ?

5. **Cross-tool :**
   - AGENTS.md present ? <32 KiB ? 6 sections ? Three-tier boundaries ? (note: Claude Code ne le lit pas nativement, mais 25+ autres outils oui)
   - .cursor/rules/*.mdc avec 4 types de regles (Always, Auto-attach, Agent, Manual) — pas .cursorrules legacy ?
   - .github/copilot-instructions.md ?
   - .aiexclude (Google Gemini) ?

---

## Dimension 6 — Test Infrastructure (10%)

**Ce qui est mesure :** Tests executables, framework, colocalisation, couverture.

**Pourquoi ca impacte les LLM :**
- Les tests sont le feedback loop #1 pour l'auto-verification (Anthropic : "tests as feedback loop")
- Boris Cherny : "the most important thing: give Claude a way to verify its work — 2-3x quality"
- Des tests colocalises sont plus faciles a decouvrir et executer par l'agent
- Le framework doit etre clair pour que l'agent sache quelle commande lancer

**Metriques :**

| Score | Coverage ratio | Colocalisation | Framework | CI | Confiance |
|-------|----------------|----------------|-----------|-----|-----------|
| 90-100 | >60% | Same-dir | Configure + clair | Oui | Officiel |
| 75-89 | 40-60% | Same-dir | Configure | Oui | Heuristique |
| 60-74 | 20-40% | Adjacent | Configure | Non | Heuristique |
| 45-59 | 10-20% | Separe | Configure | Non | Heuristique |
| 30-44 | <10% | Separe | Basique | Non | Heuristique |
| 0-29 | 0% | N/A | Absent | Non | Heuristique |

---

## Dimension 7 — Documentation Density (10%)

**Ce qui est mesure :** Qualite de la documentation inline, README, types-as-docs.

**Pourquoi ca impacte les LLM :**
- Types self-documenting (branded, enums nommes) reduisent les tokens pour comprendre
- Commentaires "why" (pas "what") donnent du contexte que le code ne fournit pas
- README avec setup instructions aide l'agent a demarrer
- Anti-pattern Anthropic : ne pas inclure ce que Claude peut inferer — meme logique pour les commentaires

**Metriques :**

| Score | Types | Comments (why) | README | Ratio comments/code | Confiance |
|-------|-------|----------------|--------|---------------------|-----------|
| 90-100 | Self-documenting (branded, enums) | Strategiques (decisions non-evidentes) | Setup + architecture | 3-8% | Heuristique |
| 75-89 | Stricts | Quelques-uns | Setup | 2-10% | Heuristique |
| 60-74 | Presents | Rares | Basique | 1-15% | Heuristique |
| 45-59 | Faibles (any, unwrap) | Absents | Absent | <1% | Heuristique |
| 30-44 | Absents | Absents | Absent | <0.5% | Heuristique |
| 0-29 | Aucun type system | Aucun | Aucun | 0% | Heuristique |

---

## Dimension 8 — Change Locality (7%)

**Ce qui est mesure :** Localite des changements dans l'historique git.

**Pourquoi ca impacte les LLM (HAUTE confiance — donnees SWE-bench) :**
- **55.56% des issues "hard"** necessitent des modifications multi-fichiers vs **3.09% des "easy"** (Ganhotra 2025)
- SWE-bench Verified patches : **1.7 fichiers, 33 LOC, 3 fonctions** (ICLR 2024). Hard = avg **2.0 fichiers** vs Easy = **1.03**
- SWE-bench Pro (nouveau standard, 1865 taches multi-langages) : **4.1 fichiers, 107.4 LOC** en moyenne — +875% vs Verified
- Performance chute a partir de **2+ fichiers touches** (pas 3+)
- Un codebase ou 80% des changements sont mono-fichier est drastiquement plus AI-friendly

**Metriques :**

| Score | Ratio mono-fichier (100 commits) | Taille patch moyenne | Confiance |
|-------|----------------------------------|---------------------|-----------|
| 90-100 | >80% | <50 LOC | Empirique (SWE-bench) |
| 75-89 | 60-80% | 50-100 LOC | Empirique |
| 60-74 | 40-60% | 100-200 LOC | Empirique |
| 45-59 | 25-40% | 200-500 LOC | Empirique |
| 30-44 | 10-25% | >500 LOC | Empirique |
| 0-29 | <10% | >1000 LOC | Empirique |

**Cas special :** Pas de git = N/A, redistribuer le poids (7%) sur les autres dimensions.

---

## Dimension 9 — Build & Verification (12%)

**Ce qui est mesure :** Capacite de l'agent a construire, tester et verifier son propre travail.

**Pourquoi ca impacte les LLM (HAUTE importance — confirme par Anthropic et Factory.ai) :**
- Boris Cherny : "Probably the most important thing: give Claude a way to verify its work. 2-3x the quality."
- Factory.ai : Build System est l'un des 8 piliers de l'Agent Readiness, niveau 1 de maturite
- Kodus : 6 checks CI/CD dedies — un agent qui ne peut pas lancer les tests est aveugle
- Un feedback loop (test → erreur → fix) est le levier principal de qualite

**Metriques :**

| Score | Build en 1 commande | Tests en 1 commande | CI presente | Feedback loop | Lock files | Confiance |
|-------|---------------------|---------------------|-------------|---------------|------------|-----------|
| 90-100 | Oui, documentee | Oui, documentee | Oui + branch protection | Pre-commit hooks + CI | A jour | Officiel (Anthropic, Factory.ai) |
| 75-89 | Oui | Oui | Oui | Pre-commit hooks | Presents | Officiel |
| 60-74 | Oui, non documentee | Oui, non documentee | Oui, basique | Aucun | Presents | Mixte |
| 45-59 | Multi-etapes | Multi-etapes | Non | Aucun | Presents | Heuristique |
| 30-44 | Complexe/fragile | Complexe/fragile | Non | Aucun | Manquants | Heuristique |
| 0-29 | Casse ou inexistant | Casse ou inexistant | Non | Aucun | Manquants | Heuristique |

**Sous-criteres :**
1. **Commandes documentees :** package.json scripts, Makefile, ou CLAUDE.md listant build/test/lint
2. **Execution one-shot :** `pnpm test` ou `cargo test` fonctionne sans setup prealable
3. **CI :** .github/workflows/, .gitlab-ci.yml, etc. — tests executes automatiquement
4. **Feedback loop :** pre-commit hooks (husky, lint-staged), PostToolUse hooks Claude Code
5. **Lock files :** package-lock.json/pnpm-lock.yaml/Cargo.lock presents et commites

---

## Dimension 10 — Dev Environment (8%)

**Ce qui est mesure :** Reproductibilite et facilite de setup de l'environnement de developpement.

**Pourquoi ca impacte les LLM :**
- Factory.ai : Dev Environment est l'un des 8 piliers de l'Agent Readiness
- Kodus : 5 checks dedies — sans env reproductible, l'agent ne peut pas bootstrapper
- Un agent qui ne peut pas installer les dependances et lancer le projet est bloque des le depart
- Les lock files garantissent que l'agent travaille avec les memes versions que l'equipe

**Metriques :**

| Score | Lock files | .env.example | Setup script | Container | Version pinning | Confiance |
|-------|------------|--------------|--------------|-----------|-----------------|-----------|
| 90-100 | Presents + commites | Present | Oui (1 commande) | devcontainer.json | .tool-versions ou equivalent | Heuristique (Factory.ai, Kodus) |
| 75-89 | Presents + commites | Present | Oui | Dockerfile | Partiel | Heuristique |
| 60-74 | Presents | Absent | Manuel documente | Non | Non | Heuristique |
| 45-59 | Presents | Absent | Manuel non documente | Non | Non | Heuristique |
| 30-44 | Partiels ou desynchronises | Absent | Aucun | Non | Non | Heuristique |
| 0-29 | Absents | Absent | Aucun | Non | Non | Heuristique |

**Sous-criteres :**
1. **Lock files :** presents, commites, a jour (pas de drift)
2. **Variables d'environnement :** .env.example ou .env.template avec toutes les variables documentees
3. **Setup script :** un script ou une commande unique pour bootstrapper (make setup, ./scripts/setup.sh)
4. **Containerisation :** Dockerfile, docker-compose.yml, ou devcontainer.json
5. **Version pinning :** .node-version, .tool-versions, rust-toolchain.toml

---

## Dimension 11 — Naming Expressiveness (7%)

**Ce qui est mesure :** Qualite et expressivite des identifiants (fonctions, variables, fichiers, dossiers).

**Pourquoi ca impacte les LLM (HAUTE confiance — donnees empiriques) :**
- arXiv 2307.12488v5 : anonymisation des identifiants → Java **-75%** de precision (70%→17%), Python **-65%** (68%→24%)
- Les function definition names ont l'impact individuel le plus fort sur la comprehension LLM
- Python plus sensible que Java (moins de types statiques comme compensation)
- Boris Cherny : "glob and grep bat le RAG" — prerequis : noms expressifs et descriptifs pour que la recherche agentique fonctionne
- Des noms cryptiques forcent l'agent a lire l'implementation — des noms expressifs permettent de comprendre l'intention sans lire le corps

**Metriques :**

| Score | Noms de fonctions | Noms de fichiers | Coherence convention | Magic strings | Confiance |
|-------|-------------------|------------------|---------------------|---------------|-----------|
| 90-100 | Intent-revealing, verbe+nom (`getUserById`, `validate_payment`) | Descriptifs, coherents avec le contenu | Convention unique (camelCase OU snake_case), appliquee partout | 0 | Empirique (arXiv) |
| 75-89 | Descriptifs, quelques abreviations acceptables | Descriptifs | Convention predominante, rares ecarts | 1-3 | Empirique |
| 60-74 | Mix descriptif/cryptique | Corrects | Convention visible mais pas stricte | 3-10 | Mixte |
| 45-59 | Abreviations frequentes (`usr`, `mgr`, `proc`) | Mix bon/generique (`utils`, `helpers`) | Conventions mixtes | 10-20 | Heuristique |
| 30-44 | Cryptiques (`fn1`, `doStuff`, `handleIt`) | Generiques (`misc`, `stuff`, `temp`) | Pas de convention | 20-50 | Heuristique |
| 0-29 | Single-letter vars en dehors des boucles, noms trompeurs | Incoherents | Chaotique | >50 | Heuristique |

**Stack-specific :**
- **Rust :** Conventions fortes : snake_case fonctions, CamelCase types, SCREAMING_SNAKE_CASE constantes. Bonus si respecte les conventions idiomatiques Rust.
- **TypeScript/React :** PascalCase composants, camelCase fonctions, kebab-case fichiers. Bonus si coherent.

**Mesure pratique :**
- Grep pour patterns cryptiques : single-letter vars (`\b[a-z]\b` hors boucles), `doStuff`, `handleIt`, `process`, `data`, `tmp`, `temp`
- Grep pour magic strings/numbers : literals non assignes a des constantes nommees
- Echantillon 20 fonctions : ratio intent-revealing / cryptique

---

## Calcul du Score Global

```
score_global = (
  file_directory_org * 0.08 +
  vertical_cohesion * 0.09 +
  import_simplicity * 0.08 +
  type_expressiveness * 0.12 +
  ai_config_quality * 0.12 +
  test_infrastructure * 0.10 +
  documentation_density * 0.08 +
  change_locality * 0.07 +
  build_verification * 0.12 +
  dev_environment * 0.07 +
  naming_expressiveness * 0.07
)
```

**Justification des poids :**
- **Type Expressiveness (12%)** : supporte par donnees empiriques (arXiv 2504.09246, -74.8% erreurs compilateur)
- **AI Config Quality (12%)** : impact direct et immediat, source officielle (Anthropic, 24 hooks, 14 champs subagent)
- **Build & Verification (12%)** : levier n°1 de qualite (Boris Cherny "2-3x quality"), Factory.ai pilier
- **Test Infrastructure (10%)** : feedback loop critique pour auto-verification (Anthropic, Boris)
- **Vertical Cohesion (9%)** : SWE-bench indirect + Boris parallelisme
- **File & Dir Org, Import Simplicity, Documentation (8% chacun)** : impact important, support mixte
- **Change Locality (7%)** : confiance haute (SWE-bench Verified + Pro) mais mesure retrospective
- **Naming Expressiveness (7%)** : **NOUVEAU** — empirique (arXiv 2307.12488, -75% precision sans noms)
- **Dev Environment (7%)** : impact reel mais indirect (Factory.ai, Kodus)

## Context Budget Estimation

```
feature_token_budget = avg_files_per_feature * avg_file_loc * 3.5 (tokens/ligne)
context_percentage = feature_token_budget / 200000 * 100

Seuils d'alerte :
- <10% : Excellent — beaucoup de marge
- 10-20% : Bon — confortable
- 20-30% : Attention — agent contraint
- >30% : Alerte — restructuration recommandee
```

## Grille d'interpretation

| Score | Grade | Verdict |
|-------|-------|---------|
| 90-100 | A+ | Architecture optimale — performance LLM maximale |
| 75-89 | A | Tres bon — quelques optimisations possibles |
| 60-74 | B | Bon — ecarts significatifs a corriger |
| 45-59 | C | Moyen — LLM perd du temps et du contexte |
| 30-44 | D | Faible — restructuration recommandee |
| 0-29 | F | Critique — LLM inefficace sur ce codebase |

## Ajustements par Stack

### Next.js (App Router)
- **Bonus +5** si App Router avec conventions (loading, error, page, layout)
- **Bonus +5** si Server Components par defaut avec Client Components explicites

### Rust
- **Bonus +5** si Cargo workspace pour monorepo
- **Bonus +5** si `lib.rs` sert d'index propre
- **Ajustement LOC** : +50 LOC aux seuils de File & Directory Organization

### Fullstack (Next.js + Rust)
- **Bonus +10** si types partages (OpenAPI, shared crate)
- **Malus -10** si duplication de modeles sans shared types

## Frameworks de scoring comparables

| Framework | Dimensions | Nature | Reference |
|-----------|-----------|--------|-----------|
| Factory.ai Agent Readiness | 9 piliers, 60+ criteres, 5 niveaux (incl. Task Discovery, Product & Experimentation) | Proprietaire, eval LLM | factory.ai/news/agent-readiness |
| Kodus agent-readiness | 7 piliers, 39 checks (incl. CI/CD, Code Health) | Open-source (MIT) | github.com/kodustech/agent-readiness |
| meta-archi | 11 dimensions, score 0-100 | Skill Claude Code | Ce fichier |

**Benchmarks de reference :**
| Benchmark | Taches | Langages | Patch moyen | Reference |
|-----------|--------|----------|-------------|-----------|
| SWE-bench Verified | 500 | Python | 1.7 fichiers, 33 LOC | ICLR 2024 |
| SWE-bench Pro | 1865 | Python, Go, TS, JS | 4.1 fichiers, 107.4 LOC | Scale AI, 2025 |

**Dimensions uniques a meta-archi :** Change Locality (valide SWE-bench), AI Config Quality etendu (hooks 24 events, @imports, .claude/agents/ 14 champs frontmatter), Naming Expressiveness (valide arXiv 2307.12488).
**Dimensions que Factory.ai/Kodus couvrent et meta-archi aussi :** Build, Testing, Documentation, Dev Environment, Code Quality (via Type Expressiveness + File Org).
**Dimensions Factory.ai/Kodus exclusives :** Task Discovery, Product & Experimentation (Factory.ai), Security (Kodus).
