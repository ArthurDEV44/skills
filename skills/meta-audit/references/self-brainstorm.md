# Self-Brainstorm Protocol — Resolution Autonome des Questions PRD

Comment Opus repond aux questions de brainstorm `/write-prd` de maniere autonome, en s'appuyant exclusivement sur les findings de l'audit.

## Principe Fondamental

Le self-brainstorm remplace l'interaction humaine par l'evidence d'audit. Chaque reponse DOIT citer au moins un finding specifique (file:line, URL, ou metrique de l'audit).

**Ce qui change par rapport a write-prd :**

| write-prd | meta-audit self-brainstorm |
|-----------|---------------------------|
| L'utilisateur repond aux questions | Opus repond base sur les findings d'audit |
| Options presentees pour choix humain | Decision automatique basee sur l'evidence |
| 6-8 rounds interactifs | 6 rounds auto-resolus en une passe |
| Business context fourni par l'utilisateur | Business context infere du code et de la research |

**Ce qui ne change PAS :**
- Le format des questions reste identique (reference research, options A/B/C/D)
- La self-validation checklist reste obligatoire (15 items)
- Le format PRD de sortie reste identique
- Les edge cases, quality gates, et devil's advocate restent obligatoires

---

## Regles de Decision

### Regle 1 — Evidence First

Chaque decision doit etre tracable a une source :

| Source | Poids | Exemple |
|--------|-------|---------|
| Finding d'audit avec file:line | Poids fort | "Architecture score 45/100 — 8 god-files >1000 LOC detectes (src/api/handlers.ts:1-1247)" |
| Best practice de Phase 1 avec URL | Poids fort | "Best practice: vertical slices (source: Anthropic engineering blog)" |
| Metrique mesurable | Poids fort | "Test coverage ratio: 12% (15 test files / 125 source files)" |
| Pattern observe dans le code | Poids moyen | "Error handling inconsistant: try-catch dans 30% des handlers, ignore dans 70%" |
| Inference logique basee sur evidence | Poids faible | "Pas de rate limiting detecte → probable absence de protection DDoS" |

**INTERDIT :** Reponses basees uniquement sur les connaissances generiques du modele sans reference a un finding.

### Regle 2 — Conservative par Defaut

Quand l'evidence est ambigue ou insuffisante :

- **Securite** → Toujours choisir l'option la plus stricte
- **Architecture** → Toujours choisir l'option incremental (pas de big-bang refactor)
- **Scope** → Toujours sous-estimer plutot que sur-estimer
- **Effort** → Toujours majorer l'effort estime de 20%
- **Priorite** → En cas de doute, SHOULD HAVE (pas MUST HAVE)

### Regle 3 — Transparence

Pour chaque decision, noter :
- `[HIGH CONFIDENCE]` — 2+ evidences concordantes
- `[MEDIUM CONFIDENCE]` — 1 evidence directe ou 2+ indirectes
- `[LOW CONFIDENCE]` — inference logique sans evidence directe → marquer dans le PRD

### Regle 4 — Respect des decisions d'architecture du projet

Avant de generer une story, verifier si CLAUDE.md ou AGENTS.md contient une decision architecturale explicite qui la contredit. Si oui, la story est automatiquement filtree vers Non-Goals avec la justification.

Exemples de contradictions a detecter :
- **Testing strategy** : si CLAUDE.md dit "integration tests with real DB, not mocks" → ne pas generer de stories de tests unitaires avec MockDatabase
- **Repository pattern** : si CLAUDE.md dit "concrete types, not trait objects" → ne pas generer de stories pour passer aux trait objects
- **Intentional design decisions** : si CLAUDE.md documente un choix explicite (ex: "Intentional design decision: concrete types because...") → respecter ce choix

**Why:** Un audit automatise ne connait pas le contexte des decisions humaines. CLAUDE.md est la source de verite pour la philosophie du projet. Generer des stories qui contredisent des decisions explicites produit du bruit et erode la confiance dans l'audit.

**How to apply:** Pour chaque story candidate, grep CLAUDE.md pour les mots-cles pertinents (testing, mock, architecture, pattern). Si une contradiction est trouvee, deplacer la story dans Non-Goals avec: "— {raison du CLAUDE.md}".

### Regle 5 — Filtre de maturite projet

Evaluer le stade du projet avant d'inclure des stories d'infrastructure lourde :

| Signal | Early-stage | Growth | Mature |
|--------|------------|--------|--------|
| Team size (contributors git) | 1-2 | 3-10 | 10+ |
| Users/traffic signals (monitoring, analytics) | Aucun ou minimal | Basic | Prometheus, Grafana, APM |
| Deploy infra | Simple (1 server, Vercel) | Multi-env | Kubernetes, blue/green |
| CI sophistication | Basic (lint, test) | + security scan, coverage | + canary, perf tests |

Stories filtrees par maturite :

| Story type | Requis a partir de | Avant = Non-Goals |
|------------|-------------------|-------------------|
| Prometheus /metrics endpoint | Growth (monitoring infra existante) | Early-stage |
| Per-user rate limiting | Growth (NAT/proxy enterprise) | Early-stage |
| LATERAL JOIN / query optimization | Growth (charge mesuree, profiling) | Early-stage (sauf si TODO dans le code = backlog) |
| Full observability stack (OTEL, tracing) | Growth | Early-stage |
| cargo deny / licence compliance | Growth (630+ deps, compliance reqs) | Early-stage (sauf si >500 deps) |

**Why:** Un projet early-stage (1-2 devs, pas de monitoring) n'a pas besoin d'un endpoint Prometheus. Generer des stories L/XL d'infrastructure pour un SaaS naissant dilue les quick-wins et surcharge le backlog.

**How to apply:** En Phase 0, noter le stade du projet dans `project_profile.maturity`. En Round 3, filtrer les stories dont le type ne correspond pas au stade. Les stories filtrees vont dans Non-Goals avec: "— premature pour le stade {stage} du projet".

---

## Round 1 — Vision & Scope (auto-resolu)

### Questions cles et comment les resoudre

**Q1 : Quel est l'objectif principal de la remediation ?**

Resolution :
1. Identifier les dimensions avec score <60 → remediation urgente
2. Identifier les dimensions avec score 60-74 → amelioration planifiee
3. Si toutes >75 → maintenance et optimisation mineure
4. L'objectif = amener les dimensions les plus faibles au-dessus de 75

Evidence : scores de Phase 5, issues CRITICAL/HIGH

**Q2 : Quelles dimensions prioriser ?**

Resolution :
1. Dimensions <60 → P0 (Must Have)
2. Dimensions 60-74 → P1 (Should Have)
3. Dimensions 75-89 → P2 (Could Have)
4. Dimensions 90+ → Won't Have (deja bon)
5. Exception : Security toujours en P0 si score <80 (meme si >60)

Evidence : scores de Phase 5, severity des issues par dimension

**Q3 : Qui sont les "utilisateurs" de cette remediation ?**

Resolution automatique :
- **Utilisateur primaire** : Developpeurs maintainers du codebase
- **Utilisateur secondaire** : Nouveaux contributeurs (onboarding)
- **Utilisateur tertiaire** : AI coding agents (Claude Code, Cursor, Copilot)

Evidence : fichiers contributing.md, CLAUDE.md, AGENTS.md detectes en Phase 0

**Q4 : Quel est le scope ?**

Resolution :
1. Compter les issues par severity : CRITICAL + HIGH = scope minimum
2. Ajouter MEDIUM issues des dimensions <60
3. Si total stories > 20 → proposer un phasing en 2-3 releases
4. Si total stories < 8 → scope compact, merge les petits epics

Evidence : nombre d'issues par severity de Phase 5

---

## Round 2 — Technical Decisions (auto-resolu)

**Q1 : Strategie de remediation ?**

Resolution :
1. Si issues sont independantes (pas de dependencies circulaires) → approche incremental (quick-wins d'abord)
2. Si issues forment un cluster (ex: architecture affecte quality, qui affecte testing) → remediation par cluster
3. JAMAIS de big-bang refactor sauf si score <30 ET l'utilisateur confirme

Evidence : graphe de dependencies des issues de Phase 5

**Q2 : Quels outils utiliser pour la remediation ?**

Resolution automatique basee sur le stack detecte :

| Stack | Outils recommandes |
|-------|-------------------|
| TypeScript/Next.js | eslint, prettier, vitest, zod, tsx strict |
| Rust | clippy, rustfmt, cargo test, cargo audit |
| Python | ruff, mypy/pyright, pytest, pydantic |
| Go | golangci-lint, go vet, go test |

Evidence : stack_profile de Phase 0, tooling detecte en Phase 3

**Q3 : Contraintes techniques ?**

Resolution :
1. Lister les dependencies majeures et leurs versions
2. Identifier les integrations externes (APIs, DBs, services)
3. Tout changement DOIT etre backward-compatible (sauf evidence que le breaking change est accepte)

Evidence : manifest files, integration patterns detectes en Phase 3

**Q4 : Risques de la remediation ?**

Resolution :
1. Chaque refactor architectural → risque de regression
2. Chaque migration de types → risque de compilation
3. Chaque changement de securite → risque de break auth
4. Mitigation : chaque story doit avoir un critere de non-regression

Evidence : test coverage ratio, CI config de Phase 3

---

## Round 3 — Prioritization MoSCoW (auto-resolu)

Pour chaque issue de Phase 5, appliquer la matrice :

| Severity | Effort | → Priority |
|----------|--------|------------|
| CRITICAL | any | MUST HAVE (P0) |
| HIGH + quick-win | low | MUST HAVE (P0) |
| HIGH + medium effort | medium | SHOULD HAVE (P1) |
| HIGH + strategic | high | SHOULD HAVE (P1) |
| MEDIUM + quick-win | low | SHOULD HAVE (P1) |
| MEDIUM + medium/strategic | medium+ | COULD HAVE (P2) |
| LOW | any | COULD HAVE (P2) |

Exception : Si une issue LOW debloque d'autres issues → remonter d'un niveau.

**Filtre post-matrice :**

Apres la classification MoSCoW, appliquer les filtres de Regle 4 (CLAUDE.md) et Regle 5 (maturite) :

1. Pour chaque story MUST/SHOULD/COULD, verifier si CLAUDE.md contient une decision qui la contredit → si oui, deplacer vers Won't Have (Non-Goals) avec justification
2. Pour chaque story de type infrastructure (monitoring, observability, optimization) → verifier si le stade du projet la justifie → si non, deplacer vers Won't Have (Non-Goals)
3. Les stories filtrees sont documentees dans Non-Goals du PRD avec leur raison de filtrage

Evidence : severity et effort de chaque issue de Phase 4, CLAUDE.md decisions, project_profile.maturity

---

## Edge Cases (auto-resolu)

Generer les edge cases de la remediation elle-meme :

| # | Categorie | Resolution |
|---|----------|------------|
| 1 | Breaking changes | Identifier les signatures publiques touchees. Critere : 0 breaking change sauf si explicitement planifie. |
| 2 | Regressions de tests | Chaque story doit executer les tests existants. Si tests cassent → la story est blocked. |
| 3 | Migration de donnees | Si le refactor touche des schemas DB, creer une story de migration dediee (P0). |
| 4 | Compatibility dependencies | Verifier que les upgrades de deps sont compatibles. Si doute → spike story. |
| 5 | CI/CD disruption | Les changements de CI doivent etre valides sur une branche avant merge. |
| 6 | Rollback | Chaque story doit etre revertable par un `git revert` propre. |

Evidence : test coverage, CI config, dependencies detectees en Phase 3

---

## Quality Gates (auto-resolu)

Generer les quality gates basees sur le stack detecte :

1. Lire le stack_profile de Phase 0
2. Identifier les commandes reelles du projet (package.json scripts, Makefile, Cargo.toml)
3. Si pas de commandes detectees, utiliser les defauts du framework

| Stack | Quality Gates par defaut |
|-------|------------------------|
| TypeScript/Next.js | `pnpm typecheck && pnpm lint && pnpm test` |
| Rust | `cargo check && cargo clippy -- -D warnings && cargo test` |
| Python | `ruff check . && mypy . && pytest` |
| Go | `go build ./... && go vet ./... && go test ./...` |

Toujours preferer les commandes reelles du projet aux defauts.

Evidence : scripts detectes en Phase 0/Phase 2, CI config

---

## Devil's Advocate (auto-resolu)

Challenger systematiquement les decisions :

**Challenge 1 — "Et si on ne fait rien ?"**
- Quantifier le cout du statu quo : nombre d'issues CRITICAL/HIGH non-resolues
- Quantifier le risque : surface de securite exposee, taux de regression probable
- Evidence : scores actuels, issues severity

**Challenge 2 — "Le scope est-il realiste ?"**
- Verifier que total stories <= 20
- Verifier que chaque story est completable en une session AI agent
- Si >20 stories, proposer un phasing avec releases intermediaires

**Challenge 3 — "Manque-t-il des dimensions ?"**
- Cross-check les 6 dimensions auditees avec les findings de Phase 1
- Si la research mentionne un aspect non-couvert, le noter dans Open Questions

**Challenge 4 — "L'effort est-il sous-estime ?"**
- Pour chaque story L ou XL, verifier que le scope est bien defini
- Pour les stories touchant >3 fichiers, verifier qu'elles ne devraient pas etre splitees
- Appliquer la regle : si une story semble XL, elle doit probablement etre splitee

Evidence : story sizes, file counts, dependency graph

---

## Output du Self-Brainstorm

Le self-brainstorm produit un **decision record** utilise par Phase 8 (PRD Generation) :

```
self_brainstorm_decisions = {
  vision: {
    objective: "{remediation objective}",
    priority_dimensions: [{dimension, score, target_score}],
    users: ["maintainers", "contributors", "AI agents"],
    scope: "{compact | standard | phased}",
    confidence: "HIGH | MEDIUM"
  },
  technical: {
    strategy: "incremental | cluster | phased",
    tools: [{name, purpose}],
    constraints: ["{constraint}"],
    risks: ["{risk with mitigation}"]
  },
  prioritization: {
    must_have: ["{ISS-NNN}"],
    should_have: ["{ISS-NNN}"],
    could_have: ["{ISS-NNN}"],
    wont_have: ["{ISS-NNN}"]
  },
  edge_cases: ["{edge case with resolution}"],
  quality_gates: ["{command} - {description}"],
  devils_advocate: {
    status_quo_cost: "{quantified}",
    scope_realistic: true | false,
    missing_dimensions: ["{if any}"],
    effort_underestimated: ["{stories to watch}"]
  }
}
```

Chaque champ cite au moins une evidence de l'audit.
