---
model: opus
name: meta-review-prd
description: "Review independant d'un PRD avant implementation. Contexte frais, zero biais de l'auteur. Verifie: contradictions CLAUDE.md, maturite projet, spikes non-resolus, coherence interne, actionnabilite des stories (IEEE 830). Produit un rapport severity-weighted avec corrections et JSON machine-readable. Invoke with /meta-review-prd <path-to-prd>"
argument-hint: "<path-to-prd.md>"
allowed-tools: Read, Grep, Glob, Bash, Agent
---

# meta-review-prd — Review Independant de PRD avant Implementation

## Pourquoi ce skill existe

Le generateur de PRD (humain ou `/meta-audit` ou `/write-prd`) a des angles morts. Il accumule du contexte, fait des hypotheses non-verifiees, et produit des stories qui peuvent contredire la philosophie du projet ou etre prematurees. Un reviewer dans un **contexte frais** detecte ce que l'auteur ne voit plus.

**Principe fondamental:** Le reviewer ne partage AUCUN contexte avec l'auteur. Il lit le PRD froid, verifie chaque claim contre le codebase reel, et juge objectivement.

## Execution Flow

```
$ARGUMENTS -> <path-to-prd.md>
     |
     v
[Phase 0] INTAKE         — Parse PRD, extraire structure
     |
     v
[Phase 1] GROUND TRUTH   — CLAUDE.md + maturite + snapshot codebase
     |
     v
[Phase 2] STORY REVIEW   — agent-explore (refs + deps) || agent-docs (spikes)
     |
     v
[Phase 3] PRD REVIEW     — Coherence interne (7 checks)
     |
     v
[Phase 4] VERDICT        — Rapport severity-weighted + self-check + JSON
```

## Phase-by-Phase Execution

### Phase 0 — INTAKE

Print: `[Phase 0/4] INTAKE — Lecture du PRD`

**0a. Lire le PRD complet:**

```
Read: $ARGUMENTS (le fichier PRD)
```

Si le fichier n'existe pas ou n'est pas un PRD (pas de `[PRD]` wrapper ou pas d'epics/stories) → avertir et arreter.

**0b. Extraire la structure:**

Pour chaque story, noter: ID (US-NNN), titre, priorite (P0/P1/P2), taille (XS/S/M/L/XL), epic (EP-NNN), nombre d'acceptance criteria, references de code (file:line), spike present/resolu, section non-regression presente.

Pour le PRD global, noter les sections presentes: Problem Statement, Goals, Technical Considerations, Risks, Non-Goals, NFRs, Success Metrics.

---

### Phase 1 — GROUND TRUTH

Print: `[Phase 1/4] GROUND TRUTH — Contexte projet`

Collecter le contexte independamment du PRD — c'est la verite terrain contre laquelle on verifie.

**1a. Lire CLAUDE.md:**

```
Read: CLAUDE.md (racine du projet)
```

Extraire: decisions architecturales explicites, stack/framework, commandes build/test/lint, contraintes documentees.

**1b. Evaluer la maturite projet:**

Suivre la methode complete dans [review-checks.md](references/review-checks.md) — Categorie 2. Classifier: early-stage | growth | mature.

**1c. Snapshot codebase:**

```bash
find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.rs" -o -name "*.py" -o -name "*.go" \) \
  -not -path "*/node_modules/*" -not -path "*/target/*" -not -path "*/.venv/*" -not -path "*/vendor/*" \
  2>/dev/null | wc -l
```

---

### Phase 2 — STORY REVIEW

Print: `[Phase 2/4] STORY REVIEW — Verification de {N} stories`

Pour chaque story, executer les checks des categories 1-3 et 5 de [review-checks.md](references/review-checks.md).

**2a. Lancer agent-explore pour verifier code refs, dependances, et couplage:**

```
Agent(
  description: "Verify PRD code refs and inter-story deps",
  subagent_type: "agent-explore",
  prompt: "
    Verify the following from a PRD against the actual codebase.

    PART 1 — Code reference verification:
    For each file:line reference, check:
    1. Does the file exist?
    2. Does the line content match the PRD's description?
    3. If missing, check git log --diff-filter=D for deletion history

    References to verify:
    {list of all file:line references from all stories}

    PART 2 — Inter-story dependency detection (CHECK-5e):
    For each story, identify which files it would touch based on its description.
    Flag cases where 2+ stories modify the same file — undocumented dependencies.

    PART 3 — Modifiability risk (CHECK-5f):
    Flag files touched by 3+ stories — high coupling risk, merge conflict potential.

    Return structured results for all 3 parts.
  "
)
```

**2b. En parallele, resoudre les spikes avec agent-docs (si applicable):**

Skip si aucune story n'a de spike non-resolu referencant une library.

```
Agent(
  description: "Resolve library capability spikes",
  subagent_type: "agent-docs",
  prompt: "
    The following PRD stories have unresolved technical questions about libraries.

    IMPORTANT: First read the project manifest (Cargo.toml, package.json,
    pyproject.toml, or go.mod) to determine the exact version of each library
    in use. Then check documentation for that specific version.

    For each spike:
    {list of spike questions with library names}

    Return for each:
    - story: US-NNN
    - library: {name} @ {version from manifest}
    - question: {the spike question}
    - answer: {resolved answer from docs}
    - source: {documentation reference}
  "
)
```

Lancer 2a et 2b dans un **SEUL message** pour parallelisme reel.

**2c. Appliquer les checks par story:**

Pour chaque story, evaluer sequentiellement:
1. **CHECK-1: CLAUDE.md contradiction** — Contredit une decision explicite?
2. **CHECK-2: Maturite** — Adaptee au stade du projet?
3. **CHECK-3: Spikes** — Questions techniques resolues?
4. **CHECK-5a: Code refs** — References valides?
5. **CHECK-5b: Criteres mesurables** — ACs verifiables? Chaque AC mappe a un test ou verification manuelle?
6. **CHECK-5c: Taille coherente** — Sizing vs nombre d'ACs?
7. **CHECK-5d: Non-regression** — Section presente avec commandes executables?
8. **CHECK-5e: Dependencies** — Dependances inter-stories documentees?
9. **CHECK-5f: Modifiabilite** — Fichiers touches par 3+ stories?

Chaque check produit: `PASS | WARN | NOTE | FAIL` avec severity et justification.

**Definitions des verdicts par check:**
- `FAIL` — Bloque l'implementation. Le PRD contient une erreur verifiable (contradiction, reference cassee, incoherence structurelle). Un fix concret existe.
- `WARN` — Issue fixable avec une action concrete et non-ambigue (ex: "ajouter `Blocked by US-001` dans Dependencies"). Ne bloque pas.
- `NOTE` — Observation structurelle sans action concrete disponible (ex: couplage eleve sur un fichier, tension indirecte, score-cible subjectif). Informatif uniquement. Ne se transforme jamais en "fix".
- `PASS` — Check passe sans probleme.

**Regle d'idempotence:** Un check ne peut emettre WARN ou NOTE que s'il passe un **test binaire structurel** (presence/absence d'un element, reference valide/invalide, comptage). Les jugements semantiques flous ("vague", "irrealiste", "tension") produisent NOTE, jamais WARN ou FAIL.

**Cap par check:** Maximum 3 findings par check. Si un check detecte plus de 3 occurrences du meme pattern, emettre les 3 plus critiques + "et {N} occurrences similaires".

---

### Phase 3 — PRD REVIEW

Print: `[Phase 3/4] PRD REVIEW — Coherence interne`

Appliquer les checks de la categorie 4 de [review-checks.md](references/review-checks.md):

1. **CHECK-4a: Numerotation** — US-001 a US-NNN sans trou ni doublon
2. **CHECK-4b: Problem Statement vs Stories** — Chaque probleme a une story
3. **CHECK-4c: Goals vs Stories** — Scores cibles realistes
4. **CHECK-4d: Technical Considerations** — Questions resolues ou referees
5. **CHECK-4e: Risks vs Spikes** — Risques a jour post-resolution
6. **CHECK-4f: Non-Goals** — Stories filtrees documentees
7. **CHECK-4g: NFRs et Success Metrics** — Coherents avec stories retenues

---

### Phase 4 — VERDICT

Print: `[Phase 4/4] VERDICT`

**4a. Compiler le rapport avec severity weighting:**

| Severity | Checks concernes |
|----------|-----------------|
| **Critical** | CHECK-1 FAIL only (contradiction CLAUDE.md directe) |
| **Major** | CHECK-3 (spikes), CHECK-4a/4d/4e/4g FAIL (coherence structurelle) |
| **Minor** | CHECK-5a-5d (actionnabilite) |
| **Informational** | CHECK-1 NOTE (tension indirecte), CHECK-2 (maturite), CHECK-4b/4c/4f (coherence semantique), CHECK-5e/5f (couplage) |

**4b. Determiner le verdict:**

Le verdict est base **uniquement sur les FAILs**. Les WARNs et NOTEs ne changent pas le verdict — ils sont informatifs.

| Condition | Verdict |
|-----------|---------|
| 0 FAIL | **READY** — PRD implementable tel quel |
| 0 FAIL, 1+ WARN ou NOTE | **READY** — Implementable. Notes incluses pour information |
| 1+ Critical FAIL | **NOT READY** — Contradiction architecturale, revision requise |
| 1-3 Major/Minor FAIL, 0 Critical | **NEEDS FIXES** — Corrections requises avant implementation |
| 4+ FAIL (tout severity) | **NOT READY** — Revision significative requise |

**Principe anti-cycle (source: Anthropic agent design, Agent Patterns):** Le review est un **single-pass structurel**. Le reviewer emet ses findings en une passe et s'arrete. Il ne re-verifie jamais ses propres findings. Un verdict READY est terminal — re-executer le skill sur le meme PRD inchange DOIT produire le meme verdict. Si un check repose sur un jugement semantique flou, il produit NOTE (informatif), pas WARN.

**4c. Afficher le rapport:**

```markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRD REVIEW COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PRD:** {path}
**Stories reviewees:** {N}
**Checks executes:** {N}

**Verdict: {READY | NEEDS FIXES | NOT READY}**

| | PASS | WARN | NOTE | FAIL |
|--|------|------|------|------|
| Critical |  {N} |  — |  — |  {N} |
| Major    |  {N} |  {N} |  — |  {N} |
| Minor    |  {N} |  {N} |  — |  {N} |
| Info     |  — |  — |  {N} |  — |

## Corrections Requises (FAIL only)

### {CHECK-ID} [{severity}] — {story ou PRD-level}
**Probleme:** {description}
**Evidence:** {PRD dit X (section/ligne)} vs {codebase montre Y (file:line)}
**Fix:** {correction specifique}

---

## Observations (WARN + NOTE — informatif, ne bloque pas)

- **{CHECK-ID}** [WARN] ({story}): {description} | Evidence: {PRD ref} vs {codebase ref}
- **{CHECK-ID}** [NOTE] ({story}): {observation structurelle}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**IMPORTANT — Seuls les FAILs produisent des "Fix:".** Les WARNs et NOTEs sont des observations. Le rapport ne doit JAMAIS presenter un WARN ou NOTE comme une correction a effectuer. Si le verdict est READY, le PRD est implementable tel quel — les observations sont du contexte utile, pas des blockers.

**4d. Self-check du rapport:**

Avant de livrer, verifier:
- Chaque FAIL cite 2 sources (PRD + codebase) et un "Fix:" actionnable
- Chaque WARN cite au moins 1 source (evidence)
- Les NOTEs sont des observations factuelles sans "Fix:" ni "Recommendation:"
- Le verdict correspond aux counts FAIL uniquement (WARNs/NOTEs ne changent pas le verdict)
- Aucun WARN ne contient un jugement semantique flou ("vague", "irrealiste", "tension") — ceux-ci doivent etre NOTE

Si une violation est detectee, corriger avant affichage.

**4e. JSON machine-readable:**

Ajouter en fin de rapport:

```json
{
  "verdict": "READY | NEEDS_FIXES | NOT_READY",
  "counts": { "pass": N, "warn": N, "note": N, "fail": N },
  "severity_breakdown": {
    "critical": { "pass": N, "fail": N },
    "major": { "pass": N, "warn": N, "fail": N },
    "minor": { "pass": N, "warn": N, "fail": N },
    "info": { "note": N }
  },
  "fixes": [
    {
      "check": "CHECK-1",
      "story": "US-003",
      "severity": "critical",
      "action": "Move to Non-Goals: contradicts CLAUDE.md testing strategy"
    }
  ],
  "observations": [
    {
      "check": "CHECK-5f",
      "story": "US-002",
      "type": "note",
      "observation": "auth-branded-panel.tsx touched by 4 stories — consider sequential implementation"
    }
  ]
}
```

---

## Hard Rules

### Pipeline
1. Lire le PRD complet en Phase 0 — pas de skip.
2. Collecter le ground truth (CLAUDE.md + maturite) AVANT de reviewer les stories.
3. Les agents Phase 2 tournent en PARALLELE (code refs + docs spikes).
4. Chaque FAIL cite DEUX sources: le PRD (section/ligne) ET le codebase (file:line, git log, ou CLAUDE.md section).
5. Le verdict est base **uniquement sur les counts FAIL** — les WARNs et NOTEs sont informatifs et ne changent pas le verdict.

### Idempotence (source: Anthropic agent design, Agent Patterns)
6. **Single-pass, then stop.** Le reviewer emet ses findings en une passe structurelle. Il ne re-verifie jamais ses propres findings. Pas de boucle self-critic.
7. **Tests binaires uniquement pour FAIL et WARN.** Un check ne peut emettre FAIL ou WARN que s'il passe un test structurel verifiable (presence/absence, reference valide/invalide, comptage). Si le test est un jugement semantique ("vague", "irrealiste", "tension indirecte"), le resultat est NOTE.
8. **Cap par check: 3 findings max.** Si un pattern se repete >3 fois, emettre les 3 plus critiques + compteur.
9. **READY est terminal.** Re-executer le skill sur un PRD inchange DOIT produire le meme verdict. Si ca ne converge pas, le check est mal defini — le corriger dans review-checks.md, pas dans le PRD.

### Objectivite
10. Le reviewer n'a AUCUN contexte de la conversation qui a genere le PRD.
11. Ne pas inventer de problemes pour justifier le review. Si le PRD est bon → READY.
12. Ne pas proposer de nouvelles stories. Le scope = verifier, pas etendre.
13. Ne pas re-scorer les dimensions. Le review verifie le PRD, pas l'audit.

### ALWAYS
- Lire CLAUDE.md avant de reviewer les stories
- Verifier chaque reference file:line contre le codebase reel
- Tenter de resoudre les spikes non-resolus avec les outils disponibles
- Citer deux sources par FAIL finding (PRD + codebase)
- Donner un verdict objectif base sur FAIL counts uniquement
- Executer le self-check (Phase 4d) avant livraison

### NEVER
- Modifier le PRD ou tout fichier du projet — ce workflow est READ-ONLY
- Proposer des stories hors du scope du PRD existant
- Ignorer les decisions explicites de CLAUDE.md
- Donner un verdict sans avoir execute tous les checks
- Generer un nouveau PRD — si NOT READY, lister les fixes
- Presenter un WARN ou NOTE comme une "correction requise" ou un "fix"
- Emettre un WARN ou FAIL base sur un jugement semantique flou (utiliser NOTE)

## Error Handling

| Scenario | Action |
|----------|--------|
| PRD introuvable | Avertir. Lister les fichiers `tasks/audit-prd-*.md` et `tasks/*-prd-*.md` disponibles. |
| CLAUDE.md absent | Skip CHECK-1 et CHECK-4d. Mentionner: "CLAUDE.md absent — contradiction + technical considerations checks skipped." |
| agent-explore echoue | Fallback sur Grep/Glob direct pour code refs. CHECK-5e/5f marques "agent fallback — manual verification recommended." |
| agent-docs echoue | Marquer les spikes comme "unresolvable — agent-docs unavailable." |
| PRD vide ou mal formate | Avertir avec les sections manquantes. Verdict: NOT READY. |

## Done When

- [ ] Phase 0 — PRD parse, structure extraite
- [ ] Phase 1 — Ground truth collecte (CLAUDE.md + maturite + snapshot)
- [ ] Phase 2 — Toutes les stories reviewees (checks 1-3, 5a-5f)
- [ ] Phase 3 — Coherence interne verifiee (7 checks)
- [ ] Phase 4 — Rapport severity-weighted avec findings actionnables
- [ ] Chaque FAIL cite 2 sources, chaque WARN porte evidence
- [ ] Self-check du rapport execute (Phase 4d)
- [ ] JSON machine-readable inclus

## References

- [Review Checks](references/review-checks.md) — 5 categories de checks avec severity, methodes et output format
- [Agent Boundaries](@~/.claude/skills/_shared/agent-boundaries.md) — CAN/CANNOT table, budgets d'appels
