---
model: opus
name: meta-archi
description: "Audit complet d'architecture de codebase pour optimiser la comprehension par les LLM (Claude Code, Cursor, Copilot, Codex). Pipeline en 7 phases : recherche des best practices (Anthropic/Boris Cherny, Google/OpenAI/Cursor), scan de l'architecture actuelle, scoring LLM-readiness sur 11 dimensions (incl. Naming Expressiveness arXiv-backed, 24 hooks Claude Code, 14 champs subagent, SWE-bench Pro), generation CLAUDE.md/AGENTS.md, remediation plan. Supporte Next.js, Rust, et fullstack. Use when the user says 'meta-archi', '/meta-archi', 'optimise pour LLM', 'architecture AI-friendly', 'LLM-ready', 'audit architecture', 'score LLM', 'genere CLAUDE.md', 'genere AGENTS.md', 'structure pour Claude', 'architecture pour AI', 'optimize for AI'. Do NOT trigger for general architecture discussions without an audit intent, or for code quality audits (use /meta-best-practices)."
argument-hint: "[focus-area?] [--generate] [--score-only]"
---

# meta-archi — Audit & Optimisation d'Architecture pour Comprehension LLM

## Critical Rules (MUST follow — placed first for attention priority)

1. Phase 1 lance TOUJOURS trois agent-websearch en PARALLELE dans un SEUL message.
2. Chaque score DOIT etre justifie par des metriques mesurables — pas de scoring au feeling.
3. Les seuils sans base empirique DOIVENT etre marques "heuristique".
4. Chaque ecart DOIT expliquer son impact specifique sur la comprehension LLM.
5. L'orchestrateur NE duplique PAS le travail des agents.
6. Ecrire les artefacts generes dans `.meta/` uniquement — JAMAIS a la racine.
7. Chaque phase produit un PHASE_OUTPUT structure (voir Handoff Protocol).

## Overview

Pipeline autonome en 7 phases qui audite l'architecture d'un codebase et score sa LLM-readiness sur **11 dimensions** ponderees.

**Principes :** Specifique LLM (chaque recommandation justifiee par son impact AI) | Quantitatif (scores, metriques mesurables) | Actionnable (before/after concrets) | Source (URL ou file:line pour chaque recommandation) | Stack-aware | Evidence-based (empirique > officiel > heuristique) | Rapport en francais.

## Execution Flow

```
$ARGUMENTS -> [focus-area?] [--generate] [--score-only]
     |
     v
[Phase 1] RESEARCH ──── 3x agent-websearch en parallele
     | PHASE_OUTPUT (findings + rubric)
     v
[Phase 2] DETECT ─────── Stack, manifests, taille (instant, orchestrateur)
     | PHASE_OUTPUT (stack_profile)
     v
[Phase 3] SCAN ────────── Architecture profonde + git (agent-explore)
     | PHASE_OUTPUT (archi_profile)
     v
[Phase 4] SCORE ───────── 10 dimensions, 0-100 (orchestrateur)
     | PHASE_OUTPUT (scores + ecarts)
     v  [--score-only?] ──> Phase 7
[Phase 5] CROSS-REF ──── Best practices vs architecture (orchestrateur)
     | PHASE_OUTPUT (ecarts classes)
     v
[Phase 6] GENERATE ────── CLAUDE.md + AGENTS.md + remediation (orchestrateur)
     | PHASE_OUTPUT (artefacts)
     v
[Phase 7] REPORT ──────── .meta/archi-llm-report.md
```

## Structured Handoff Protocol

Chaque phase produit un bloc structure avant de passer a la suivante. Ce format reduit la perte de signal inter-phases (source: Factory.ai context compression research, precision 4.04 vs 3.43 sans structure).

```
## PHASE_OUTPUT — Phase {N}: {NAME}
- **Findings:** [max 10 items, format dense]
- **Decisions:** [choix faits et pourquoi]
- **Next phase needs:** {ce que la phase suivante doit recevoir}
- **Confidence:** {0-100}
- **Gaps:** [ce qui n'a pas pu etre trouve/mesure]
```

## Phase-by-Phase Summary

### Phase 1 — RESEARCH (agent-websearch x3, parallel)

Print: `[Phase 1/7] RESEARCH — Best practices architecture LLM-friendly`

Lancer **trois agent-websearch en parallele** dans un SEUL message :
- **Agent A** — Anthropic & Boris Cherny : CLAUDE.md best practices, hooks vs advisory, skills, subagents, context management. Voir `references/agent-protocols.md — Prompt A`.
- **Agent B** — Cross-tools : Google Jules, OpenAI AGENTS.md spec, Cursor .mdc rules, Copilot, convergence. Voir `references/agent-protocols.md — Prompt B`.
- **Agent C** — Metriques : SWE-bench, type systems, frameworks de scoring (Factory.ai, Kodus). Voir `references/agent-protocols.md — Prompt C`.

Extraire : max 20 findings, rubrique de scoring, URLs, contradictions, claims non verifiees. **GATE:** Produire PHASE_OUTPUT avant de passer a Phase 2.

### Phase 2 — DETECT (orchestrateur, instant)

Print: `[Phase 2/7] DETECT — Stack & structure du projet`

Detection automatique via Glob/Read/Bash. Voir `references/phase-details.md — Phase 2` pour le detail. **GATE:** Produire stack_profile + PHASE_OUTPUT avant de passer a Phase 3.

### Phase 3 — SCAN (agent-explore)

Print: `[Phase 3/7] SCAN — Analyse d'architecture profonde`

Spawner agent-explore avec le contexte compresse de Phase 1+2. Voir `references/agent-protocols.md — Prompt Scan` et `references/phase-details.md — Phase 3` pour les 9 sections d'analyse (structure, patterns, imports, types, AI config, tests, build & verification, dev environment, git). **GATE:** Produire PHASE_OUTPUT avant de passer a Phase 4.

### Phase 4 — SCORE (orchestrateur)

Print: `[Phase 4/7] SCORE — LLM-Readiness Scoring (10 dimensions)`

Evaluer sur **11 dimensions**. Voir `references/scoring-rubric.md` pour la rubrique complete.

| Dimension | Poids | Confiance dominante |
|-----------|-------|---------------------|
| File & Directory Organization | 8% | Heuristique (+ sous-critere Migration Completeness) |
| Vertical Cohesion | 9% | Heuristique (SWE-bench Verified + Pro indirect) |
| Import Simplicity | 8% | Heuristique (+ sous-critere Static Traceability) |
| Type Expressiveness | 12% | Empirique (arXiv 2504.09246, -74.8% erreurs) |
| AI Config Quality | 12% | Officiel (Anthropic — 24 hooks, 14 champs subagent, Compact Instructions) |
| Test Infrastructure | 10% | Officiel (Anthropic, Boris "2-3x quality") |
| Documentation Density | 8% | Heuristique |
| Change Locality | 7% | Empirique (SWE-bench Verified + Pro) |
| Build & Verification | 12% | Officiel (Anthropic, Factory.ai) |
| Dev Environment | 7% | Heuristique (Factory.ai, Kodus) |
| Naming Expressiveness | 7% | **Empirique** (arXiv 2307.12488, -75% precision sans noms) |

Score global = somme ponderee. Context budget estimation incluse. Si `--score-only`, sauter a Phase 7. **GATE:** Produire PHASE_OUTPUT avant de passer a Phase 5.

### Phase 5 — CROSS-REF (orchestrateur)

Print: `[Phase 5/7] CROSS-REFERENCE — Best practices vs architecture actuelle`

Croiser findings Phase 1 avec scan Phase 3+4. Pour chaque ecart : source, confidence, impact LLM, before/after, effort. Classer en Quick Wins / Restructurations / Refactors. Voir `references/phase-details.md — Phase 5`. **GATE:** Produire PHASE_OUTPUT avant de passer a Phase 6.

### Phase 6 — GENERATE (orchestrateur)

Print: `[Phase 6/7] GENERATE — Artefacts AI config`

Si `--generate` ou AI config absente. Generer dans `.meta/` : generated-CLAUDE.md (<150L), generated-AGENTS.md (<32KiB, 6 sections, three-tier), remediation-plan.md. Voir `references/phase-details.md — Phase 6`. **GATE:** Produire PHASE_OUTPUT avant de passer a Phase 7.

### Phase 7 — REPORT (orchestrateur)

Print: `[Phase 7/7] REPORT — Generation du rapport`

Generer `.meta/archi-llm-report.md` + afficher le resume terminal. Voir `references/phase-details.md — Phase 7` pour le format de sortie terminal.

## Output Format — Rapport

Le rapport `.meta/archi-llm-report.md` suit ce format :

```markdown
# Rapport Architecture LLM-Ready — {projet}

**Date :** {YYYY-MM-DD} | **Stack :** {stack} | **Taille :** {N} fichiers, ~{N} LOC

## Score LLM-Readiness : {N}/100 — Grade {grade}

| Dimension | Score | Poids | Grade | Verdict |
|-----------|-------|-------|-------|---------|
| File & Directory Organization | {N}/100 | 10% | {grade} | {1 phrase} |
| Vertical Cohesion | {N}/100 | 10% | {grade} | {1 phrase} |
| Import Simplicity | {N}/100 | 8% | {grade} | {1 phrase} |
| Type Expressiveness | {N}/100 | 13% | {grade} | {1 phrase} |
| AI Config Quality | {N}/100 | 12% | {grade} | {1 phrase} |
| Test Infrastructure | {N}/100 | 10% | {grade} | {1 phrase} |
| Documentation Density | {N}/100 | 10% | {grade} | {1 phrase} |
| Change Locality | {N}/100 | 7% | {grade} | {1 phrase} |
| Build & Verification | {N}/100 | 12% | {grade} | {1 phrase} |
| Dev Environment | {N}/100 | 7% | {grade} | {1 phrase} |
| Naming Expressiveness | {N}/100 | 7% | {grade} | {1 phrase} |

### Context Budget
- Fichiers par feature : {N} | Tokens : ~{N} | % fenetre : {N}% (seuil: 30%)

## Quick Wins / Restructurations / Refactors
[Chaque ecart : dimension, impact LLM, source+confiance, score impact, effort, avant/apres]

## AI Config existant
[CLAUDE.md, AGENTS.md, Cursor, Hooks vs Advisory, .claude/agents/]

## Score projete apres remediation
| Scenario | Score | Grade |
|----------|-------|-------|
| Actuel / Quick wins / + Restructurations / Complet |

## Sources
[URLs + file:line + confiance]

## Methodologie
[Pipeline, agents, date, calibration]
```

## Constraints (Three-Tier)

### ALWAYS
- 3 agent-websearch en parallele en Phase 1
- PHASE_OUTPUT structure entre chaque phase
- Metriques mesurables pour chaque score
- Impact LLM explique pour chaque ecart
- Source URL + file:line + before/after + confiance pour chaque ecart
- Artefacts dans `.meta/` uniquement
- Rapport en francais
- Score projete + context budget inclus

### ASK FIRST
- Rien — workflow autonome, ne modifie aucun fichier existant du projet (ecrit uniquement dans `.meta/`)

### NEVER
- Modifier des fichiers du projet (sauf `.meta/`)
- Ecrire CLAUDE.md/AGENTS.md a la racine
- Scorer sans metriques mesurables
- Recommander sans impact LLM
- Ecrire en anglais
- Presenter des heuristiques comme empiriques
- Citer la "40% rule" (mythe — auto-compaction a ~83.5% mainstream, ~95% subagents)
- Dupliquer le travail des agents
- Gonfler les scores ou inventer des ecarts

## Error Handling

- **Pas de codebase :** Rapport template avec best practices seules.
- **agent-websearch echoue :** Continuer avec best practices connues. Noter le gap.
- **agent-explore echoue :** Scan superficiel Glob/Grep depuis orchestrateur. Noter.
- **>500 fichiers source :** Echantillonner (entry points, plus gros fichiers, core modules).
- **AI config existe :** Analyser et scorer, ne pas regenerer sauf `--generate`.
- **Score >85 :** Feliciter, lister optimisations marginales, ne pas inventer de problemes.
- **Pas de git :** Change Locality = N/A, redistribuer le poids sur 9 dimensions.

## Done When

- [ ] 7 phases completees avec PHASE_OUTPUT a chaque transition
- [ ] 11 dimensions scorees avec metriques mesurables
- [ ] `.meta/archi-llm-report.md` genere
- [ ] Resume affiche dans le terminal
- [ ] Chaque ecart explique son impact LLM avec before/after
- [ ] Aucun fichier du projet modifie (sauf `.meta/`)
- [ ] Heuristiques marquees comme telles

## References

- [Agent Protocols](references/agent-protocols.md) — prompts exacts pour chaque agent
- [Scoring Rubric](references/scoring-rubric.md) — rubrique complete, seuils par stack
- [Remediation Patterns](references/remediation-patterns.md) — before/after par dimension
- [Phase Details](references/phase-details.md) — instructions detaillees phases 2-7
