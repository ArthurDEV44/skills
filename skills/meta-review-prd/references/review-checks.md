# Review Checks — Verification Detaillee par Categorie

Chaque check produit un verdict (`PASS`, `WARN`, `NOTE`, `FAIL`) avec une severity (`Critical`, `Major`, `Minor`, `Info`).

## Definitions des verdicts

| Verdict | Quand l'utiliser | Test requis | Produit un "Fix:" |
|---------|-----------------|-------------|-------------------|
| `FAIL` | Erreur structurelle verifiable qui bloque l'implementation | Test binaire (presence/absence, valide/invalide) | Oui |
| `WARN` | Issue fixable avec une action concrete et non-ambigue | Test binaire structurel | Non — observation avec evidence |
| `NOTE` | Observation structurelle ou jugement semantique sans action concrete | Aucun test strict requis | Non — informatif uniquement |
| `PASS` | Check reussi | — | Non |

**Regle d'or:** Si un check repose sur un **jugement semantique** ("vague", "irrealiste", "tension indirecte", "mal sizee"), il produit `NOTE`, jamais `WARN` ou `FAIL`. Seuls les tests binaires structurels (fichier existe/n'existe pas, cle presente/absente, numero manquant/present) produisent `WARN` ou `FAIL`.

**Cap:** Maximum 3 findings par check. Au-dela, emettre les 3 plus critiques + "et {N} occurrences similaires".

---

## Categorie 1 — Contradictions CLAUDE.md [Critical]

**Principe:** CLAUDE.md est la source de verite pour la philosophie du projet. Toute story qui contredit une decision explicite est un faux positif de l'audit.

**Methode:**

1. Lire CLAUDE.md completement
2. Pour chaque story, extraire les mots-cles de l'approche proposee
3. Grep CLAUDE.md pour ces mots-cles et les termes opposes
4. **Contradiction directe** (PRD propose X, CLAUDE.md dit explicitement "ne pas faire X") → `FAIL`
5. **Tension indirecte** (PRD propose X, CLAUDE.md ne le mentionne pas mais la philosophie semble differente) → `NOTE` (jugement semantique, pas un test binaire)

**Output par story:**
```
[CHECK-1] CLAUDE.md Contradiction [Critical]
Story: US-NNN — {title}
PRD claims: {what the story proposes}
CLAUDE.md says: {the contradicting statement, with section reference}
Verdict: FAIL | NOTE
Recommendation: {specific action — only for FAIL}
```

---

## Categorie 2 — Maturite Projet [Major]

**Principe:** Les stories d'infrastructure lourde ne sont pertinentes qu'a partir d'un certain stade du projet.

**Methode:**

1. Evaluer le stade du projet:
   ```bash
   # Contributors
   git shortlog -sn --no-merges | wc -l

   # Monitoring signals
   grep -r "prometheus\|grafana\|datadog\|sentry\|opentelemetry" . \
     --include="*.toml" --include="*.json" --include="*.yml" --include="*.yaml" 2>/dev/null | head -5

   # CI sophistication
   ls .github/workflows/*.yml 2>/dev/null | wc -l
   cat .github/workflows/*.yml 2>/dev/null | grep -c "coverage\|canary\|perf\|benchmark"

   # Deploy complexity
   ls docker-compose* Dockerfile kubernetes/ k8s/ 2>/dev/null
   ```

2. Classifier:
   - **Early-stage**: 1-2 contributors, pas de monitoring, CI basique, deploy simple
   - **Growth**: 3-10 contributors, monitoring basique, CI avec security scan, multi-env
   - **Mature**: 10+ contributors, full observability, CI avancee, k8s/blue-green

3. Pour chaque story, verifier si son type est adapte au stade:

   | Type de story | Requis a partir de | Avant = WARN |
   |---|---|---|
   | Endpoint Prometheus /metrics | Growth | Early-stage |
   | Per-user rate limiting | Growth | Early-stage |
   | Batch optimization (LATERAL JOIN, N+1) | Growth (si profiling mesure) | Early-stage |
   | Full OTEL/tracing stack | Growth | Early-stage |
   | cargo deny / licence audit | Growth (500+ deps) | Early-stage (<500 deps) |
   | RBAC / multi-tenant isolation | Growth | Early-stage |

**Output par story:**
```
[CHECK-2] Maturity Filter [Major]
Story: US-NNN — {title}
Story type: {infrastructure | optimization | compliance}
Project maturity: {early-stage | growth | mature}
Verdict: PASS | WARN
Recommendation: {Keep | Move to Non-Goals — premature for {stage} stage}
```

---

## Categorie 3 — Spikes Non-Resolus [Major]

**Principe:** Une story P0 avec une question technique non-resolue n'est pas implementable. Si la question est resolvable avec les outils disponibles, elle doit etre resolue dans le PRD.

**Methode:**

1. Scanner chaque story pour des patterns de spike:
   - "Verifier si..." / "Check if..." / "Spike obligatoire"
   - Branches conditionnelles ("Si oui: ... Si non: ...")
   - References a des fichiers qui "devraient exister"
   - "Technical Considerations" avec des questions ouvertes

2. Pour chaque spike detecte, tenter de le resoudre:

   **Fichier manquant/supprime:**
   ```bash
   ls {path} 2>/dev/null
   git log --all --oneline --diff-filter=D -- '**/{filename}'
   git log --all --oneline --diff-filter=R -- '**/{filename}'
   ```

   **Capacite de library:**
   ```
   Agent(agent-docs): "Read project manifest for version, then check:
   does {library} {version} support {feature}?"
   ```

   **Comportement du code:**
   ```
   Grep/Read: verifier le code source directement
   ```

3. Verdict:
   - PRD contient "Spike resolu" avec la reponse → `PASS`
   - Spike resolu par le reviewer pendant le review → `PASS` avec note "Resolution: {answer} — ajouter au PRD"
   - Spike non-resolvable et documente comme tel → `PASS`
   - PRD contient un spike resolvable mais non-resolu → `WARN`

**Output par spike:**
```
[CHECK-3] Unresolved Spike [Major]
Story: US-NNN — {title}
Question: {the open question}
Resolution: {answer found} | UNRESOLVABLE
Verdict: PASS | WARN
Recommendation: {Add spike resolution to story | Flag as requiring manual investigation}
```

---

## Categorie 4 — Coherence Interne du PRD [Major]

**Principe:** Le PRD doit etre auto-coherent. Chaque section doit refleter le meme scope.

### 4a. Numerotation continue
- Stories US-001 a US-NNN sans trou, epics EP-001 a EP-NNN sans trou
- Trou ou doublon → `FAIL`

### 4b. Problem Statement vs Stories
- Probleme mentionne sans story correspondante → `FAIL` (test binaire: probleme nomme dans Problem Statement, zero stories qui le referent)
- Story sans probleme mentionne → `NOTE` (jugement semantique — la story peut servir un but non-explicite)

### 4c. Goals vs Stories
- Dimension avec delta >0 sans story correspondante → `NOTE` (jugement semantique sur le mapping goal→story)
- Score cible irrealiste par rapport au nombre/taille des stories → `NOTE` (jugement semantique sur "irrealiste")

### 4d. Technical Considerations
- Question ouverte resolvable par les outils → `WARN`
- Question non-resolue et non-referencee par une story → `FAIL`
- Note: si CLAUDE.md absent, skip ce check

### 4e. Risks vs Spikes
- Spike resolu mais risque non-mis a jour → `WARN`
- Risque dit "Spike + feature flag" mais spike deja resolu → `FAIL`

### 4f. Non-Goals
- Stories filtrees non-documentees dans Non-Goals → `NOTE` (jugement semantique — l'auteur peut avoir des raisons non-documentees)

### 4g. NFRs et Success Metrics
- Metriques referencant des stories supprimees → `FAIL`
- Metriques incoherentes avec les stories retenues → `WARN`

**Output par check:**
```
[CHECK-4{a-g}] {Check name} [Major]
Finding: {description}
Expected: {what should be}
Actual: {what is in the PRD}
Verdict: PASS | WARN | FAIL
Fix: {specific correction}
```

---

## Categorie 5 — Actionnabilite des Stories [Minor]

**Principe:** Chaque story doit etre implementable en une session AI agent sans ambiguite. Couvre les proprietes IEEE 830: verifiable, traceable, modifiable.

### 5a. References de code valides
- Chaque `file:line` verifiable via `Read {file} offset={line} limit=5`
- Fichier inexistant ou ligne non-correspondante → `WARN`

### 5b. Criteres d'acceptation mesurables (IEEE 830: Verifiable)
- Chaque AC mappe a une commande de test ou etape de verification manuelle
- **Definition concrete de "vague":** un AC est vague si et seulement s'il utilise un verbe comparatif sans metrique cible ET sans commande de verification. Exemples vagues: "ameliorer les performances", "optimiser le chargement", "assurer la qualite". Exemples NON-vagues: "LCP < 1.5s" (metrique), "bun run typecheck passe" (commande), "le widget s'affiche sans scroll horizontal" (verification visuelle binaire).
- AC avec verbe comparatif sans metrique ET sans commande → `WARN` (test binaire: presence d'un verbe comparatif + absence de metrique/commande)
- **Cap: max 3 ACs vagues signales par story.** Au-dela, emettre les 3 + compteur.

### 5c. Taille coherente
- Story XS avec >5 ACs → mal sizee → `NOTE` (jugement semantique — le sizing depend du contexte)
- Story L avec 1-2 ACs → sous-specifiee → `NOTE` (jugement semantique)

### 5d. Non-regression definie
- Section "Non-regression" avec commandes executables requise
- "cargo test passe" = acceptable. Rien = `WARN`

### 5e. Dependencies inter-stories (IEEE 830: Traceable)
- Si US-003 utilise le service cree dans US-001, la dependance est documentee
- Story A cree un fichier, Story B le modifie, B ne declare pas "Blocked by A" → `WARN` (test binaire: fichier en commun + absence de dependance declaree)

### 5f. Modifiabilite (IEEE 830: Modifiable)
- Fichier touche par 3+ stories = couplage eleve, risque de conflits
- Detecte par agent-explore → `NOTE` (observation structurelle — le couplage est une propriete du scope, pas une erreur fixable sans changer le scope. Emettre comme contexte utile pour l'implementation, pas comme correction.)
