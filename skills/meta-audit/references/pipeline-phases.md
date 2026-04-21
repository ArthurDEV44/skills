# Pipeline Phases 5-8 — Scoring, Validation, Brainstorm, PRD

Phases de synthese et generation du pipeline meta-audit. Pour les phases de collecte (0-4) et d'output (9), voir [SKILL.md](../SKILL.md). Pour les prompts agents, voir [agent-protocols.md](agent-protocols.md).

---

## Phase 5 — EXTRACTION & SCORING (AutoSCORE 2-pass)

Print: `[Phase 5/9] EXTRACTION & SCORING`

Methode inspiree d'AutoSCORE (arXiv 2509.21910, domaine: educational scoring — pattern transfere au code audit) : separer l'extraction des evidences du scoring en deux passes distinctes.

**5a. PASS 1 — Extraction structuree :**

Lire les findings retournes par les agents Phase 3 + le typed handoff Phase 1 (research) + Phase 4 (docs). Pour chaque sous-critere de chaque dimension, extraire un JSON structure :

```json
{
  "criterion": "A1",
  "criterion_name": "Module boundaries",
  "evidence_found": [
    {"type": "file", "ref": "src/api/handlers.ts:42", "description": "..."},
    {"type": "static_analysis", "tool": "eslint", "rule": "...", "count": N},
    {"type": "url", "ref": "https://...", "tier": "T1"}
  ],
  "evidence_missing": ["No domain/infra separation detected"],
  "deterministic_floor": 3,
  "deterministic_source": "ls -d src/domain src/infra → 0 results",
  "static_analysis_signal": "eslint: 0 import-cycle warnings"
}
```

Regles d'extraction :
1. **Deduplication** — meme finding de multiple sources → garder la version la plus detaillee
2. **Conflit** — research vs codebase → noter les deux positions
3. **Attribution** — chaque evidence garde sa source (URL, file:line, tool:rule, ctx7)
4. **N/A** — si aucune evidence trouvee pour un sous-critere, marquer `"status": "N/A — insufficient evidence"` au lieu d'inventer

**5b. PASS 2 — Scoring sur le JSON d'extraction :**

Pour chaque sous-critere, scorer de 0 a 10 en se basant UNIQUEMENT sur le JSON d'extraction :

1. Lire le plancher deterministe (`deterministic_floor`) et le signal static analysis
2. Lire la rubrique de scoring de [audit-dimensions.md](audit-dimensions.md)
3. Attribuer un score >= plancher deterministe
4. Si `"status": "N/A"` → exclure du calcul de la moyenne. Signaler les dimensions avec >2 sous-criteres N/A comme necessitant un re-scan. **Cap re-scan : max 1 re-scan par dimension.** Si le re-scan ne produit toujours pas d'evidence, le sous-critere reste N/A definitivement.

| Dimension | Poids | Sous-criteres |
|-----------|-------|---------------|
| Architecture & Structure | 20% | A1-A5 (voir [audit-dimensions.md](audit-dimensions.md)) |
| Code Quality | 20% | Q1-Q5 |
| Security | 15% | S1-S5 |
| Testing | 15% | T1-T5 |
| Performance | 15% | P1-P5 |
| Developer Experience | 15% | D1-D5 |

Score dimension = `avg(sous-criteres non-N/A) * 10`
Score global = `sum(dimension.score * dimension.weight)`

**5c. Classer les issues :**

Trier par : impact (severity x scope) > effort (quick-win < medium < strategic)

```
issues = [
  {
    id: "ISS-NNN",
    dimension: "Architecture",
    sub_criterion: "A2",
    title: "{description courte}",
    severity: "CRITICAL | HIGH | MEDIUM | LOW",
    evidence: "file:line ou URL",
    effort: "quick-win | medium | strategic",
    deployability: 0-5,  // echelle inspiree d'AXIOM (arXiv 2512.20159)
    impact_description: "{pourquoi c'est un probleme}",
    source: "agent | static_analysis | research"
  }
]
```

**Echelle deployability** (inspiree d'AXIOM, arXiv 2512.20159 — effort de remediation pour le deploiement) :

| Score | Signification | Mapping story |
|-------|--------------|---------------|
| 5 | Production-ready, aucun changement | Skip |
| 4 | Fonctionnel, tweaks qualite mineurs | XS |
| 3 | Fonctionnel, refactoring qualite majeur | S-M |
| 2 | Corrections fonctionnelles mineures | M-L |
| 1 | Refactoring majeur necessaire | L-XL |
| 0 | Fondamentalement defaillant | Epic |

**5d. Preparer le resume d'audit :**

```
audit_summary = {
  score_global: N/100,
  grade: "A+ | A | B | C | D | F",
  scores_by_dimension: {
    architecture: { score: N, sub_scores: {A1: N, A2: N, ...}, na_count: N },
    ...
  },
  top_issues: [top 15 issues triees par impact],
  strengths: [top 5 points forts],
  total_issues: N,
  static_analysis_summary: { errors: N, warnings: N },
  na_dimensions: ["{dimensions necessitant re-scan}"]
}
```

Interpretation des grades :

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 90-100 | A+ | Codebase exemplaire |
| 75-89 | A | Tres bon — optimisations mineures |
| 60-74 | B | Correct — ecarts significatifs |
| 45-59 | C | Moyen — remediation recommandee |
| 30-44 | D | Faible — remediation urgente |
| 0-29 | F | Critique — refonte necessaire |

**GATE :** Extraction JSON complete. Scores pointwise calcules. Issues classees. Resume d'audit pret.

---

## Phase 6 — VALIDATION (4 micro-validators)

Print: `[Phase 6/9] VALIDATION — 4 micro-validators`

Phase de controle qualite inspiree de CodeMender (Google DeepMind, blog post Oct 2025) — agents purpose-scoped au lieu d'un critique omnibus. Chaque validator a un seul objectif.

**6a. Lancer cite-check + fp-filter en PARALLELE :**

**Validator 1 — CITE-CHECK (deterministe) :**

Pas d'agent LLM. L'orchestrateur verifie mecaniquement chaque citation :

```
Pour chaque issue dans audit_summary.top_issues:
  SI evidence contient "file:line":
    1. Glob: verifier que le fichier existe
    2. Read: lire la ligne citee
    3. Verifier que le contenu correspond a la description de l'issue
    SI echec → marquer [UNGROUNDED]
  SI evidence contient URL:
    Garder tel quel (non-verifiable localement)
```

**Validator 2 — FP-FILTER (LLM, agent-explore) :**

```
Agent(
  description: "Filter false positives in audit findings",
  prompt: <see agent-protocols.md — Prompt Validator: FP-Filter>,
  subagent_type: "agent-explore"
)
```

Le fp-filter verifie :
1. `unwrap()` dans du code de test → acceptable
2. `any` dans des type guards ou generics → intentionnel
3. Missing validation sur endpoints internes-only → by design
4. Console.log dans du code dev-only → acceptable
5. TODO markers avec references issue tracker → dette geree, pas non-geree
6. Issues dupliquees sous angles differents → fusionner

**6b. Apres cite-check + fp-filter, lancer score-coherence :**

**Validator 3 — SCORE-COHERENCE (LLM, agent-explore) :**

```
Agent(
  description: "Validate score-to-findings coherence",
  prompt: <see agent-protocols.md — Prompt Validator: Score-Coherence>,
  subagent_type: "agent-explore"
)
```

Le score-coherence verifie :
1. Un sous-critere score 8+/10 → 0-1 issues max
2. Un sous-critere score 3-/10 → 2+ issues minimum
3. Dimensions avec score <60 → assez de findings pour expliquer ?
4. Conflit entre agents sur la meme issue → adjudication par qualite d'evidence (AgentAuditor, arXiv 2602.09341)

**6c. Validator 4 — SPIKE-RESOLVER (deterministe + agent-docs) :**

Certains findings generent des questions techniques resolvables avec les outils disponibles (git log, agent-docs, grep). Les laisser comme "Technical Considerations" ouvertes dans le PRD produit un PRD non-actionnable qui necessite une investigation manuelle avant implementation.

```
Pour chaque issue dans audit_summary.top_issues:
  SI la description contient "verifier si", "check if", "spike", ou reference un fichier/service manquant:
    1. SI reference un fichier "manquant" ou "supprime":
       → git log --all --oneline --diff-filter=D -- '**/filename*'
       → Si trouve: noter "Spike resolu: fichier supprime dans commit {hash} ({message})"
       → Si pas trouve: noter "Spike resolu: fichier n'a jamais existe"
    2. SI reference une capacite de library ("supporte-t-elle X ?"):
       → Agent(agent-docs): verifier la documentation de la library
       → Noter "Spike resolu: {library} {supporte|ne supporte pas} {feature}"
    3. SI reference un comportement runtime ("est-ce que X fait Y ?"):
       → Grep/Read le code source pour verifier
       → Noter "Spike resolu: {description du comportement observe}"
```

Les resolutions de spikes sont injectees dans les stories correspondantes (section "Spike resolu:") et retirees de "Technical Considerations". Seules les questions veritablement non-resolvables restent dans Technical Considerations.

**Why:** Les stories P0 avec des zones d'ombre non-resolvees ne sont pas implementables. L'audit a acces aux memes outils que le developpeur — autant resoudre les spikes pendant l'audit plutot que de les deleguer.

**6d. Calculer la precision estimee :**

```
precision_estimate = (issues_grounded - issues_false_positive) / total_issues
```

**Precision gate :** Si precision_estimate < 85%, signaler explicitement dans le rapport :
`[PRECISION_WARNING] Fiabilite reduite — {N}% des findings retoquees par les validators.`

**Output des validators :**
```
validation_result = {
  cite_check: {
    grounded: N,
    ungrounded: ["{ISS-NNN}: {reason}"]
  },
  fp_filter: {
    false_positives: ["{ISS-NNN}: {reason} → remove | downgrade"],
    duplicates_merged: ["{ISS-NNN + ISS-MMM} → {ISS-NNN}"]
  },
  score_coherence: {
    adjustments: ["{dimension}.{sub_criterion}: {old} → {new}: {reason}"],
    gaps: ["{dimension}: {missing analysis}"],
    conflicts_resolved: ["{ISS-NNN}: {agent_a} vs {agent_b} → {resolution}"]
  },
  spike_resolver: {
    resolved: ["{ISS-NNN}: {question} → {answer} (method: git_log | agent_docs | grep)"],
    unresolvable: ["{ISS-NNN}: {question} — requires runtime/manual verification"]
  },
  precision_estimate: N%,
  confidence: "HIGH | MEDIUM"
}
```

Apres validation, mettre a jour le `audit_summary` avec les corrections.

**GATE :** Findings valides. Faux positifs retires. Scores coherents. Spikes resolus. Precision estimee calculee.

---

## Phase 7 — SELF-BRAINSTORM (autonome)

Print: `[Phase 7/9] SELF-BRAINSTORM — Resolution autonome des questions PRD`

L'IA repond aux questions du brainstorm `/write-prd` en s'appuyant exclusivement sur les findings de l'audit valides par les micro-validators (Phase 6).

Lire le protocole complet dans [self-brainstorm.md](self-brainstorm.md).

Le protocole definit 6 rounds : Vision & Scope, Technical Decisions, Prioritization MoSCoW, Edge Cases, Quality Gates, Devil's Advocate. Chaque round est auto-resolu avec des regles de decision basees sur les evidences de l'audit.

**Regle absolue :** Chaque reponse DOIT citer un finding specifique valide par les micro-validators Phase 6 (file:line, URL, ou metrique). Aucune reponse basee sur des connaissances generiques.

**GATE :** Toutes les decisions prises. Toutes citent des evidences.

---

## Phase 8 — PRD GENERATION

Print: `[Phase 8/9] PRD GENERATION`

**8a. Generer le PRD :**

Utiliser le format exact de [/write-prd references/prd-template.md](@~/.claude/skills/write-prd/references/prd-template.md) :

- **Problem Statement** : Resume de l'audit FILTRE — score global, dimensions faibles, et UNIQUEMENT les problemes adresses par les stories retenues (pas les findings filtres vers Non-Goals). Si des stories ont ete filtrees par Regle 4 (CLAUDE.md) ou Regle 5 (maturite), le Problem Statement reflete le scope reduit, pas l'audit original.
- **Overview** : Plan de remediation structure
- **Goals** : Amelioration des scores par dimension (baseline actuel → cible)
- **Target Users** : Developpeurs du codebase (maintainers, contributors, onboarding)
- **Research Findings** : Compressed research de Phase 1 (competitive context = best practices du marche)
- **Assumptions & Constraints** : Contraintes du stack + hypotheses de remediation
- **Quality Gates** : Commandes reelles du projet (Phase 7e)
- **Epics** : Un epic par dimension necessitant remediation (score <75)
  - Exemple : `EP-001: Architecture & Structure Remediation`
- **Stories** : Issues converties en stories avec criteres d'acceptation
  - Quick-wins → P0, taille XS-S (deployability 4-3)
  - Medium → P1, taille M-L (deployability 2)
  - Strategic → P2, taille L-XL (deployability 1-0)
  - **INFO severity → PAS de story generee.** Les issues INFO (sous-criteres sans verificateur deterministe: naming, convention adherence, test quality, feature organization, etc.) apparaissent dans le rapport d'audit brut mais ne produisent PAS de stories dans le PRD. Elles sont des observations utiles pour le developpeur, pas des corrections actionnables. Voir [audit-dimensions.md](audit-dimensions.md) — section "Regle d'idempotence".
- **NFRs** : Metriques cibles par dimension (chiffres specifiques)
- **Edge Cases** : Risques de la remediation (Phase 7d)
- **Risks & Mitigations** : Issues les plus critiques
- **Non-Goals** : Ce que cet audit ne couvre PAS
- **Technical Considerations** : Questions resolues par le spike-resolver (Phase 6c) avec leurs reponses. Titre: "Technical Considerations (Resolved)". Seules les questions veritablement non-resolvables avec les outils disponibles restent ouvertes — et dans ce cas, la story correspondante est marquee comme necessitant un spike pre-implementation.
- **Risks & Mitigations** : Les risques dont les spikes sont resolus (Phase 6c) ont leur probabilite et mitigation mises a jour. Ne pas conserver "Spike + feature flag" comme mitigation si le spike est resolu et montre que le risque est faible.
- **Non-Goals** : Inclure explicitement les stories filtrees par Regle 4 (CLAUDE.md) et Regle 5 (maturite) avec leur justification (ex: "— le projet utilise volontairement des integration tests avec real DB")
- **Success Metrics** : Score actuel → score cible par dimension, ajuste pour refléter uniquement les stories retenues (pas les scores cibles originaux si des stories ont ete filtrees)

Wrapper le PRD dans `[PRD]...[/PRD]`.

**8b. Generer le status JSON :**

Suivre le schema de [/write-prd references/prd-template.md](@~/.claude/skills/write-prd/references/prd-template.md) — Status File Schema.

**8c. Self-validation :**

Executer la checklist de [/write-prd references/brainstorm-protocols.md](@~/.claude/skills/write-prd/references/brainstorm-protocols.md) — PRD Self-Validation Checklist (15 items).

Pour chaque item : citer la section PRD qui le satisfait. Si un item echoue → corriger avant sauvegarde. Max 2 corrections. Si la validation echoue apres 2 corrections → sauvegarder avec un bloc `[VALIDATION_WARNINGS]` listant les items en echec.

**8d. Generer le rapport d'audit brut :**

Format plus detaille que le PRD, dans `.meta/audit-report-{project}.md` :
- Scores detailles avec metriques brutes et JSON d'extraction
- Tous les findings avec file:line et source (agent, static_analysis, research)
- Graphe de dependances des issues
- Decisions du self-brainstorm avec justifications
- Precision estimee et resultats des 3 validators
- Static analysis summary

**GATE :** PRD valide. Status JSON genere. Rapport brut genere.
