# Audit Dimensions — Scoring Pointwise par Sous-Criteres

6 dimensions d'audit avec sous-criteres atomiques, metriques, seuils, et verificateurs deterministes. Methode basee sur AutoSCORE (arXiv 2509.21910 — decomposition en criteres atomiques, scoring pointwise pour eliminer le biais de conflation) et AXIOM (arXiv 2512.20159 — calibration multisource deterministe+LLM) pour des scores reproductibles et auditables.

## Vue d'ensemble

| Dimension | Poids | Focus | Sous-criteres |
|-----------|-------|-------|---------------|
| Architecture & Structure | 20% | Organisation, modules, boundaries, coupling | 5 |
| Code Quality | 20% | Typing, patterns, naming, error handling | 5 |
| Security | 15% | OWASP, secrets, auth, validation | 5 |
| Testing | 15% | Coverage, quality, CI, fixtures | 5 |
| Performance | 15% | Async, queries, memory, caching, observability | 5 |
| Developer Experience | 15% | Docs, tooling, onboarding, scripts | 5 |

## Methode de Scoring

**Scoring pointwise :** chaque sous-critere est evalue independamment (0-10), puis le score dimension = moyenne ponderee des sous-criteres * 10. Source : AutoSCORE (arXiv 2509.21910) — evaluer un critere a la fois elimine les biais de conflation. Le scoring se fait en deux passes : extraction structuree des evidences (Phase 5a), puis scoring sur le JSON d'extraction (Phase 5b).

**Verificateurs deterministes :** pour chaque sous-critere, un check machine-verifiable (grep, file existence, config parsing) fixe un plancher objectif. Le LLM peut ajuster a la hausse sur la base de son analyse qualitative, mais ne peut PAS descendre en-dessous du plancher deterministe.

```
dimension_score = avg(sub_criterion.score for sc in sub_criteria) * 10
score_global = sum(dimension.score * dimension.weight)
```

### Regle d'escalation Security

**Exception :** Security est TOUJOURS en P0 (Must Have) si son score est <80, meme si les autres dimensions sont >60. Rationale : les failles de securite ont un cout exponentiel et ne peuvent pas attendre.

---

## Dimension 1 — Architecture & Structure (20%)

**Ce qui est mesure :** Comment le code est organise, decouple, et navigable.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| A1 | Module boundaries | 25% | Bounded contexts clairs, separation domain/infra nette | Structure coherente, quelques leaks | Mix, boundaries reconnaissables | Boundaries floues | Monolith non-structure |
| A2 | File sizing | 20% | Moyen <300 LOC, 0 fichiers >500 | <3 fichiers >500 LOC | <5 fichiers >500, quelques >1000 | God-files multiples | Majorite >500 LOC |
| A3 | Import depth | 20% | <3 hops moyen, 0 circular | <5 hops, <3 circular | >5 hops ou >5 circular | Imports profonds + circulaires | Spaghetti imports |
| A4 | Feature organization | 20% | Vertical slices ou clear entry points | Pattern reconnaissable | Mix horizontal/vertical | Pattern inclair | Pas de pattern |
| A5 | Git health (temporal coupling) | 15% | >70% single-file patches, low coupling | >50% single-file, moderate coupling | Mixed, quelques tight couplings | Frequent multi-file patches | High temporal coupling |

### Verificateurs deterministes

```bash
# A1: Check for domain/infra separation
ls -d src/domain src/infra src/core src/application 2>/dev/null | wc -l  # >0 = structured

# A2: God-files count
find . -type f \( -name "*.ts" -o -name "*.rs" -o -name "*.py" \) -not -path "*/node_modules/*" -not -path "*/target/*" -exec wc -l {} + | awk '$1>500{c++}END{print c+0}'

# A3: Circular deps (TS)
grep -r "from.*\.\." --include="*.ts" --include="*.tsx" -l | wc -l  # heuristic for deep imports

# A5: Single-file patch ratio
git log --oneline -100 --name-only --format="" | sort | uniq -c | sort -rn | head -5
```

### Impact
Une architecture claire permet aux agents AI et aux humains de naviguer et modifier le code avec un minimum de contexte charge.

---

## Dimension 2 — Code Quality (20%)

**Ce qui est mesure :** Qualite du code, patterns, coherence, typing, gestion d'erreurs.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| Q1 | Type safety | 25% | Strict mode, 0 `any`/`unwrap` | <5 escape hatches | 5-20 escape hatches | >20, typing lache | Pas de typing |
| Q2 | Error handling | 25% | Exhaustif, schemas de validation aux frontieres | Present, quelques trous | Partiel, inconsistant | Ad-hoc, try-catch sporadique | Absent |
| Q3 | Naming & consistency | 20% | Convention unique respectee a 95%+ | Coherent a 90%+ | Quelques inconsistances | Naming mixte | Chaotique |
| Q4 | DRY & duplication | 15% | Pas de duplication significative | <3 patterns dupliques | 3-10 patterns similaires | Duplication significative | Copy-paste partout |
| Q5 | Code debt markers | 15% | 0 TODO/FIXME/HACK | <5 markers | 5-15 markers | 15-30 markers | >30 markers |

### Verificateurs deterministes

```bash
# Q1: Count type escape hatches
grep -rn "any\b\|as any\|@ts-ignore\|@ts-nocheck" --include="*.ts" --include="*.tsx" | wc -l  # TS
grep -rn "unwrap()\|\.expect(" --include="*.rs" | wc -l  # Rust

# Q3: Check strict mode
grep -l '"strict": true' tsconfig.json 2>/dev/null | wc -l  # TS
grep -l "deny(warnings)" build.rs Cargo.toml 2>/dev/null | wc -l  # Rust

# Q5: Debt markers count
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.ts" --include="*.tsx" --include="*.rs" --include="*.py" | wc -l
```

### Impact
Un code bien type et coherent reduit les erreurs de compilation et les bugs logiques. Les schemas de validation a la frontiere systeme previennent les erreurs runtime. Les types stricts reduisent de >50% les erreurs de compilation dans la generation de code AI (arXiv 2504.09246).

---

## Dimension 3 — Security (15%)

**Ce qui est mesure :** Surface d'attaque, gestion des secrets, auth, validation des inputs.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| S1 | Secrets management | 25% | 0 secret dans le code, .env dans .gitignore, .env.example documente | 0 secret, .gitignore OK | Pas de secrets evidents | Potentiel de secrets | Secrets dans le code |
| S2 | Auth & session | 20% | JWT/session avec rotation, CSRF protege | Auth robuste sans rotation | Auth basique | Auth fragile | Pas d'auth |
| S3 | Input validation | 25% | Validation schemas a chaque frontiere (Zod/serde) | Validation sur endpoints principaux | Validation inconsistante | Validation manquante sur endpoints cles | Pas de validation |
| S4 | Injection surface | 15% | 0 raw SQL, CSP headers, output encoding | <3 raw queries, CORS restrictif | CORS permissif, quelques raw queries | Raw SQL significatif, pas de CORS | Injection ouverte |
| S5 | Dependency health | 15% | Dependencies a jour, lockfile propre, no deprecated | <3 deps outdated | Quelques deps outdated | Dependencies avec vulns connues | CVEs critiques non-patchees |

### Verificateurs deterministes

```bash
# S1: Secrets scan
grep -rn "API_KEY\|SECRET\|PASSWORD\|TOKEN\|private_key" --include="*.ts" --include="*.tsx" --include="*.rs" --include="*.py" -l | grep -v ".env" | grep -v "node_modules" | wc -l

# S1: .gitignore check
grep -c "\.env" .gitignore 2>/dev/null  # >0 = present

# S3: Validation schemas presence
grep -rn "z\.\|Zod\|joi\|yup\|serde\|pydantic\|validator" --include="*.ts" --include="*.tsx" --include="*.rs" --include="*.py" -l | wc -l

# S5: Lockfile analysis (TS) — check for deprecated/outdated patterns
test -f package-lock.json && echo "lockfile present" || echo "no lockfile"
test -f Cargo.lock && echo "lockfile present" || echo "no lockfile"
```

### Impact
Les failles de securite ont un cout exponentiel. Un secret dans le code est un incident en attente. L'absence de validation des inputs est la porte d'entree #1 des attaques (OWASP A03:2021 — Injection).

---

## Dimension 4 — Testing (15%)

**Ce qui est mesure :** Couverture, qualite des tests, CI/CD, fixtures.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| T1 | Coverage ratio | 25% | >70% fichiers avec test correspondant | >50% | 30-50% | <30% | 0 tests |
| T2 | Test quality | 25% | Verifie le comportement, edge cases, fixtures | Tests corrects, quelques trous | Tests basiques, happy path | Tests fragiles, mocks lourds | Tests cassees ou inutiles |
| T3 | CI integration | 20% | CI passe tests a chaque PR, multiple types (unit+integ+e2e) | CI avec tests | CI presente mais fragile | CI absente ou cassee | Pas de CI |
| T4 | Test patterns | 15% | Fixtures/factories, colocalisation, naming clair | Patterns coherents | Mix de patterns | Patterns ad-hoc | Pas de pattern |
| T5 | Flakiness & reliability | 15% | 0 skip/retry, tests deterministes | <3 skipped tests | Quelques retries/timeouts | Flaky tests visibles | Tests non-fiables |

### Verificateurs deterministes

```bash
# T1: Coverage ratio
TEST_FILES=$(find . -type f \( -name "*.test.*" -o -name "*.spec.*" -o -name "*_test.*" \) -not -path "*/node_modules/*" | wc -l)
SRC_FILES=$(find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.rs" -o -name "*.py" \) -not -path "*/node_modules/*" -not -path "*/target/*" -not -name "*.test.*" -not -name "*.spec.*" | wc -l)
echo "$TEST_FILES / $SRC_FILES"

# T3: CI presence
ls .github/workflows/*.yml .gitlab-ci.yml Jenkinsfile 2>/dev/null | wc -l

# T5: Flaky indicators
grep -rn "skip\|xit\|xdescribe\|retry\|timeout:" --include="*.test.*" --include="*.spec.*" | wc -l
```

### Impact
Les tests sont le filet de securite de toute remediation. Sans tests, les corrections de l'audit risquent d'introduire des regressions. Une bonne couverture permet aux agents AI de valider leurs modifications automatiquement.

---

## Dimension 5 — Performance (15%)

**Ce qui est mesure :** Patterns de performance, async, queries, memory, caching, observabilite.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| P1 | Async patterns | 20% | Promise.all/join partout, 0 sequential awaits | <3 sequential awaits | Async partiel | Patterns sync bloquants | Performance non-consideree |
| P2 | Query efficiency | 25% | 0 N+1, eager loading, indexes | <3 N+1 potentiels | N+1 probables | N+1 visibles | Requetes dans les boucles partout |
| P3 | Caching & pagination | 20% | Caching strategique, pagination sur tous les list endpoints | Caching present, pagination partielle | Pas de caching | Pas de caching ni pagination | Pas de pagination sur endpoints de liste |
| P4 | Memory & resources | 15% | 0 memory leaks, connection pooling | Patterns corrects | Quelques patterns risques | Memory leaks potentiels | Fuites evidentes |
| P5 | Observability | 20% | Structured logging, error tracking, health checks, tracing | Logging + error tracking | Logging basique | Console.log seulement | Pas de logging |

### Verificateurs deterministes

```bash
# P1: Sequential awaits in loops
grep -rn "for.*await\|while.*await" --include="*.ts" --include="*.tsx" --include="*.js" | wc -l

# P2: Queries in loops (heuristic)
grep -rn "\.find\|\.query\|\.select\|\.where" --include="*.ts" --include="*.tsx" -A2 | grep -c "for\|while\|\.map\|\.forEach"

# P5: Observability indicators
grep -rn "Sentry\|datadog\|prometheus\|opentelemetry\|pino\|winston\|bunyan\|tracing::" --include="*.ts" --include="*.tsx" --include="*.rs" --include="*.py" -l | wc -l
```

### Impact
Les bottlenecks de performance degradent l'UX et coutent en infrastructure. L'observabilite est le prerequis pour detecter les problemes en production. Les N+1 queries et les awaits sequentiels sont les plus frequents et les plus faciles a corriger.

---

## Dimension 6 — Developer Experience (15%)

**Ce qui est mesure :** Documentation, tooling, onboarding, scripts, workflow de dev.

### Sous-criteres

| # | Sous-critere | Poids | 9-10 | 7-8 | 5-6 | 3-4 | 0-2 |
|---|-------------|-------|------|-----|-----|-----|-----|
| D1 | Documentation | 25% | README complet (setup, arch, contrib), .env.example documente | README avec setup | README basique | README minimal | Pas de README |
| D2 | Scripts & automation | 20% | Scripts pour toutes les taches (dev, test, build, lint, format, migrate) | Scripts principaux | Quelques scripts | Peu de scripts | Pas de scripts |
| D3 | AI-readiness | 20% | CLAUDE.md + AGENTS.md de qualite | CLAUDE.md present | CLAUDE.md basique | Pas de CLAUDE.md | Pas de config AI |
| D4 | Lint & format | 20% | Lint + format auto (pre-commit hooks), editorconfig | Lint + format configure | Lint configure | Lint non-enforce | Pas de lint |
| D5 | Dev environment | 15% | Docker dev setup, env vars documentes, onboarding <30min | Docker ou setup simple | Setup documentable | Setup complexe | Setup impossible sans aide |

### Verificateurs deterministes

```bash
# D1: README completeness
test -f README.md && grep -c "setup\|install\|getting started\|architecture\|contributing" README.md || echo 0

# D2: Scripts count
test -f package.json && node -e "console.log(Object.keys(require('./package.json').scripts||{}).length)" 2>/dev/null || echo 0
test -f Makefile && grep -c "^[a-z].*:" Makefile 2>/dev/null || echo 0

# D3: AI config
ls CLAUDE.md AGENTS.md .cursor/rules/*.mdc 2>/dev/null | wc -l

# D4: Lint & format config
ls .eslintrc* .prettierrc* rustfmt.toml .ruff.toml pyproject.toml 2>/dev/null | wc -l
ls .husky/pre-commit .lefthook.yml .pre-commit-config.yaml 2>/dev/null | wc -l
```

### Impact
Un bon DX reduit le temps d'onboarding de jours a heures. Les scripts et la documentation permettent aux agents AI de comprendre et executer les commandes du projet sans deviner. Un CLAUDE.md bien fait multiplie l'efficacite de Claude Code.

---

## Calcul du Score Global

```
dimension_score = avg(sub_criterion.score for sc in sub_criteria) * 10

score_global = (
  architecture.score * 0.20 +
  quality.score * 0.20 +
  security.score * 0.15 +
  testing.score * 0.15 +
  performance.score * 0.15 +
  dx.score * 0.15
)
```

## Interpretation

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 90-100 | A+ | Codebase exemplaire — maintenance minimale requise |
| 75-89 | A | Tres bon — quelques optimisations ciblees |
| 60-74 | B | Correct — ecarts significatifs a planifier |
| 45-59 | C | Moyen — remediation recommandee |
| 30-44 | D | Faible — remediation urgente |
| 0-29 | F | Critique — dette technique majeure |

## Mapping Issues → Severity

| Severity | Critere | Exemples | Genere une story? |
|----------|---------|----------|--------------------|
| CRITICAL | Securite, perte de donnees, crash en production | Secret dans le code, SQL injection, auth contournee | Oui — P0 |
| HIGH | Bug potentiel, regression probable, perf degradee | N+1 queries sur hot path, no error handling sur API critiques, 0 tests sur module critique | Oui — P0 |
| MEDIUM | Qualite reduite, maintenance plus difficile — **uniquement si un verificateur deterministe le confirme** | God-files (>500 LOC mesure), type escapes (count>5), debt markers (count>15) | Oui — P1 |
| LOW | Amelioration de confort, DX, documentation — avec metrique mesurable | README sans setup (mesurable), scripts manquants (count), lint non-configure (presence/absence) | Oui — P2 |
| INFO | Observation basee sur un jugement semantique sans verificateur deterministe. **Ne genere PAS de story.** | Naming "inconsistant" (subjectif), "convention adherence" (semantique), "test quality" (qualitatif), "feature organization" (interpretatif) | **Non** — informatif uniquement |

### Regle d'idempotence (source: Anthropic agent design, Agent Patterns)

Les sous-criteres SANS verificateur deterministe (A1 module boundaries, A4 feature organization, Q2 error handling, Q3 naming, Q4 DRY, T2 test quality, T4 test patterns, P3 caching, P4 memory, S2 auth & session) produisent des observations utiles mais non-reproductibles. Sur ces sous-criteres :
- Le LLM peut scorer et commenter, mais les issues detectees sont **INFO** (pas MEDIUM/LOW)
- Les issues INFO apparaissent dans le rapport d'audit mais **ne generent PAS de stories dans le PRD**
- Seuls les sous-criteres avec un verificateur deterministe (A2, A3, A5, Q1, Q5, S1, S3, S4, S5, T1, T3, T5, P1, P2, P5, D1-D5) peuvent produire des issues MEDIUM/LOW/HIGH/CRITICAL

**Cap par sous-critere :** Maximum 3 issues par sous-critere. Au-dela, emettre les 3 plus critiques + "et {N} occurrences similaires".
