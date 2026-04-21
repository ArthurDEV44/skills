# Remediation Patterns — Before/After par Dimension

## Vue d'ensemble

Pour chaque dimension du scoring LLM-readiness, ce document fournit des patterns de remediation concrets avec des exemples before/after. Chaque pattern inclut : ecart typique, impact LLM specifique, before/after, effort estime, niveau de confiance.

---

## Dimension 1 — File & Directory Organization

### Pattern 1.1 : Decomposer un god-file

**Ecart :** Un fichier >500 LOC contenant plusieurs responsabilites.

**Impact LLM :** L'agent doit lire 500+ lignes pour comprendre une seule fonction. Context window pollue. Le format d'edition "whole" bloque sur les tres grands fichiers (Aider benchmark). Confiance: heuristique.

**Before :**
```
src/services/
└── user-service.ts    # 800 LOC — auth, profile, preferences, notifications
```

**After :**
```
src/features/
├── auth/
│   ├── index.ts           # re-exports
│   ├── login.ts           # 60 LOC
│   ├── register.ts        # 70 LOC
│   └── auth.test.ts
├── profile/
│   ├── index.ts
│   ├── get-profile.ts     # 40 LOC
│   ├── update-profile.ts  # 50 LOC
│   └── profile.test.ts
└── notifications/
    ├── index.ts
    ├── service.ts         # 85 LOC
    └── notifications.test.ts
```

**Effort :** Medium (1-2 heures par fichier)

---

### Pattern 1.2 : Aplatir le nesting excessif

**Ecart :** >5 niveaux de nesting.

**Impact LLM :** L'agent consomme des tokens de navigation a chaque niveau. Confiance: heuristique.

**Before :**
```
src/modules/user/features/profile/components/avatar/
└── AvatarUploader.tsx   # 7 niveaux
```

**After :**
```
src/features/profile/
├── AvatarUploader.tsx   # 3 niveaux
├── ProfileForm.tsx
└── profile.test.tsx
```

**Effort :** Medium

---

### Pattern 1.3 : Extraire les composants React monolithiques

**Ecart :** Composant React >300 LOC avec JSX inline, hooks et logique metier.

**Impact LLM :** ClassEval (ICSE 2024) : les LLM performent mieux au method-level qu'au class-level. Confiance: empirique.

**Before :**
```tsx
// Dashboard.tsx — 450 LOC
export function Dashboard() {
  // 15 hooks, 200+ lignes de JSX
}
```

**After :**
```
features/dashboard/
├── index.ts
├── Dashboard.tsx           # 50 LOC — layout + composition
├── DashboardHeader.tsx     # 40 LOC
├── FilterBar.tsx           # 50 LOC
├── DataTable.tsx           # 90 LOC
├── useDashboardData.ts     # 30 LOC
├── types.ts                # 20 LOC
└── Dashboard.test.tsx
```

**Effort :** Medium (1 heure par composant)

---

## Dimension 2 — Vertical Cohesion

### Pattern 2.1 : Migration horizontal → vertical

**Ecart :** Architecture en couches horizontales — 4+ dossiers pour une feature.

**Impact LLM :** Comprendre "auth" requiert controllers/auth + services/auth + models/user + types/auth = 4+ fichiers. SWE-bench : 55% des issues "hard" sont multi-fichiers. Confiance: heuristique (SWE-bench indirect).

**Before :**
```
src/
├── controllers/auth.controller.ts
├── services/auth.service.ts
├── models/user.model.ts
├── types/auth.types.ts
└── __tests__/auth.test.ts
```

**After :**
```
src/features/auth/
├── index.ts
├── auth.controller.ts
├── auth.service.ts
├── auth.types.ts
└── auth.test.ts
```

**Effort :** Strategic (1-2 jours)

---

### Pattern 2.2 : Colocaliser les tests

**Ecart :** Tests dans un dossier separe.

**Impact LLM :** L'agent ne decouvre pas les tests en explorant le module. Confiance: heuristique.

**Before :**
```
src/auth/service.rs
tests/auth_test.rs       # loin du code
```

**After :**
```
src/auth/
├── mod.rs
├── service.rs
└── tests.rs             # ou #[cfg(test)] mod tests
tests/                   # integration tests only
└── api_integration.rs
```

**Effort :** Quick win (30 min)

---

## Dimension 3 — Import Simplicity

### Pattern 3.1 : Ajouter des index files

**Ecart :** Pas d'index files — imports pointent vers les fichiers internes.

**Impact LLM :** Pas de surface d'API claire. Boris Cherny : "glob and grep bat le RAG" — index files facilitent la decouverte. Confiance: heuristique.

**Before :**
```typescript
import { login } from '../auth/auth.service';
import { AuthResponse } from '../auth/auth.types';
```

**After :**
```typescript
// features/auth/index.ts
export { login, register } from './auth.service';
export type { AuthResponse } from './auth.types';

// Usage :
import { login, AuthResponse } from '@/features/auth';
```

**Effort :** Quick win (15 min par module)

---

### Pattern 3.2 : Eliminer les imports relatifs profonds

**Ecart :** Imports `../../../` — fragiles et opaques.

**Impact LLM :** L'agent ne peut pas determiner le chemin absolu sans compter les `..`. Confiance: heuristique.

**Before :**
```typescript
import { Database } from '../../../infrastructure/database/connection';
```

**After :**
```typescript
// tsconfig.json: "paths": { "@/*": ["./src/*"] }
import { Database } from '@/infrastructure/database/connection';
```

**Effort :** Quick win (1 heure)

---

## Dimension 4 — Type Expressiveness

### Pattern 4.1 : Eliminer les `any`

**Ecart :** >10 `any` dans le code source.

**Impact LLM :** Chaque `any` est un trou — l'agent ne sait pas quel type circule. Type-constrained generation reduit erreurs >50% (arXiv 2504.09246). Confiance: empirique.

**Before :**
```typescript
function processData(data: any): any {
  return data.items.map((item: any) => item.name);
}
```

**After :**
```typescript
interface InputData {
  items: Array<{ name: string; id: number }>;
}

function processData(data: InputData): string[] {
  return data.items.map((item) => item.name);
}
```

**Effort :** Medium (variable)

---

### Pattern 4.2 : Remplacer unwrap() par Result (Rust)

**Ecart :** `unwrap()` en dehors des tests.

**Impact LLM :** L'agent ne peut pas raisonner sur les chemins d'erreur. Confiance: heuristique.

**Before :**
```rust
fn get_user(id: &str) -> User {
    let conn = pool.get().unwrap();
    let user = conn.query_one("...", &[&id]).unwrap();
    serde_json::from_row(user).unwrap()
}
```

**After :**
```rust
fn get_user(id: &str) -> Result<User, UserError> {
    let conn = pool.get()?;
    let user = conn.query_opt("...", &[&id])?
        .ok_or_else(|| UserError::NotFound(id.to_string()))?;
    Ok(serde_json::from_row(user)?)
}
```

**Effort :** Medium (1-2 heures par module)

---

## Dimension 5 — AI Config Quality

### Pattern 5.1 : Creer un CLAUDE.md optimal

**Ecart :** CLAUDE.md absent.

**Impact LLM :** L'agent doit deviner commandes, conventions, boundaries. Confiance: officiel (Anthropic).

**After :** (voir scoring-rubric.md pour le template complet)
```markdown
# Project Name

## Commands
- Build: `pnpm build`
- Test: `pnpm test`
- Lint: `pnpm lint`

## Code Style
- TypeScript strict — no `any`
- Imports: `@/` alias

## Architecture
- Feature-based: `src/features/{feature}/`

## Testing
- Vitest, colocated

## Boundaries
- Never modify `src/generated/`
```

**Effort :** Quick win (30 min)

---

### Pattern 5.2 : Implementer hooks pour regles critiques (24 lifecycle events)

**Ecart :** Regles critiques uniquement en CLAUDE.md (advisory).

**Impact LLM :** Hooks = deterministe, CLAUDE.md = advisory (peut etre ignore). Claude Code supporte 24 events en 10 categories et 4 types de handlers (command, http, prompt, agent). Confiance: officiel (Anthropic).

**Before :**
```markdown
<!-- CLAUDE.md -->
- Always run `pnpm lint` before committing
- Never modify `src/generated/`
```

**After :**
```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "command": "pnpm lint --fix ${file}"
    }],
    "PreToolUse": [{
      "matcher": "Write|Edit",
      "command": "echo ${file} | grep -q 'src/generated/' && exit 1 || exit 0"
    }],
    "PreCompact": [{
      "command": "echo 'Compaction triggered — preserve API changes context'"
    }],
    "Stop": [{
      "command": "pnpm test --run 2>&1 | tail -5"
    }]
  }
}
```

**Hooks avances disponibles :** SessionStart (env setup), SubagentStart/SubagentStop (monitoring), PreCompact/PostCompact (preservation contexte), CwdChanged/FileChanged (reactive), http (webhooks externes), prompt (eval LLM yes/no), agent (spawn subagent verificateur).

**Effort :** Quick win (30 min)

---

### Pattern 5.3 : Ajouter progressive disclosure via @imports

**Ecart :** CLAUDE.md monolithique >150 lignes.

**Impact LLM :** "Lost in the Middle" — les regles au milieu d'un long document sont ignorees. @imports permettent de charger du contenu a la demande. Confiance: officiel (Anthropic).

**Before :**
```markdown
# CLAUDE.md — 300 lignes, tout inline
## Commands (30L) ## Style (50L) ## Architecture (80L) ## API Docs (100L) ## Testing (40L)
```

**After :**
```markdown
# CLAUDE.md — 80 lignes, essentiel uniquement
## Commands
...
## Style
...
## Architecture
@docs/architecture.md
## Testing
@docs/testing-guide.md
```

**Effort :** Quick win (30 min)

---

### Pattern 5.4 : Creer AGENTS.md cross-tool

**Ecart :** AGENTS.md absent.

**Impact LLM :** 25+ outils lisent AGENTS.md nativement (Codex, Jules, Copilot, Cursor). Confiance: officiel (OpenAI, Google, GitHub).

**After :**
```markdown
# AGENTS.md

## Commands
...
## Testing
...
## Project Structure
...
## Code Style
...
## Git Workflow
...
## Boundaries
### Always safe
- Read any file, run tests
### Ask first
- Modify DB schema, update deps
### Never
- Modify `src/generated/`, commit secrets
```

**Effort :** Quick win (20 min)

---

### Pattern 5.5 : Migrer .cursorrules → .cursor/rules/*.mdc

**Ecart :** .cursorrules legacy.

**Impact LLM :** Format .mdc permet des regles contextuelles par globs. Confiance: officiel (Cursor).

**Before :**
```
.cursorrules    # 200 lignes monolithiques
```

**After :**
```
.cursor/rules/
├── general.mdc          # alwaysApply: true, type "Always"
├── react-components.mdc # globs: ["**/*.tsx"], type "Auto-attach"
├── api-routes.mdc       # globs: ["src/app/api/**"], type "Auto-attach"
├── architecture.mdc     # description: "Architecture decisions", type "Agent" (AI routing)
└── testing.mdc          # globs: ["**/*.test.*"], type "Auto-attach"
```

**4 types de regles Cursor :** Always (alwaysApply:true), Auto-attach (globs), Agent (description pour routing AI), Manual (@mention). Soft limit 500L par fichier, description <200 chars.

**Effort :** Quick win (30 min)

---

### Pattern 5.6 : Ajouter section Compact Instructions dans CLAUDE.md

**Ecart :** Pas de controle sur ce qui est preserve lors de l'auto-compaction.

**Impact LLM :** L'auto-compaction (~83.5% du context window) peut perdre des informations critiques. La section Compact Instructions indique a Claude ce qu'il doit preserver. Confiance: officiel (Anthropic).

**Before :**
```markdown
# CLAUDE.md — pas de section Compact Instructions
## Commands ...
## Style ...
```

**After :**
```markdown
# CLAUDE.md
## Compact Instructions
Preserve: API changes context, current task state, test results
Focus: architecture decisions and error patterns over implementation details

## Commands ...
## Style ...
```

**Effort :** Quick win (10 min)

---

### Pattern 5.7 : Architecture subagent avec frontmatter complet

**Ecart :** Pas de subagents dedies ou subagents avec frontmatter minimal.

**Impact LLM :** Les subagents avec frontmatter complet (14 champs) permettent la delegation specialisee avec isolation de contexte, scope d'outils restreint, et persistance cross-session. Pattern Writer/Reviewer confirme par Anthropic. Confiance: officiel (Anthropic).

**Before :**
```
# Pas de .claude/agents/ ou agents sans frontmatter
```

**After :**
```markdown
<!-- .claude/agents/code-reviewer.md -->
---
name: code-reviewer
description: Reviews code changes for quality, security, and consistency
model: sonnet
tools:
  - Read
  - Grep
  - Glob
  - Bash
permissionMode: plan
maxTurns: 10
effort: high
---

Review the code changes focusing on:
1. Security vulnerabilities
2. Performance regressions
3. Consistency with project conventions
```

**14 champs disponibles :** name, description, tools, disallowedTools, model, permissionMode, maxTurns, skills, mcpServers, hooks, memory, background, effort, isolation.

**Effort :** Medium (1 heure)

---

## Dimension 6 — Test Infrastructure

### Pattern 6.1 : Colocaliser les tests

**Ecart :** Tests dans `__tests__/` ou `test/` separe.

**Impact LLM :** L'agent modifie le code et ne voit pas le test — ne peut pas auto-verifier. Tests = feedback loop #1 (Anthropic, Boris Cherny). Confiance: officiel.

**Before :**
```
src/features/auth/service.ts
test/features/auth/service.test.ts
```

**After :**
```
src/features/auth/service.ts
src/features/auth/service.test.ts
```

**Effort :** Quick win (30 min)

---

## Dimension 7 — Documentation Density

### Pattern 7.1 : Commentaires "why" strategiques

**Ecart :** Aucun commentaire ou commentaires "what" inutiles.

**Impact LLM :** L'agent ne comprend pas les decisions non-evidentes. Confiance: officiel + heuristique.

**Before :**
```typescript
// Get the user  ← inutile
const user = await getUser(id);
const timeout = 30000;  // ← pourquoi 30s ?
```

**After :**
```typescript
const user = await getUser(id);
// 30s timeout required by payment provider SLA — do not reduce below 25s
// without coordinating with billing team (#JIRA-1234)
const timeout = 30000;
```

**Effort :** Quick win (ongoing)

---

## Dimension 8 — Change Locality

### Pattern 8.1 : Reduire le couplage temporel

**Ecart :** Changements touchent systematiquement 3+ fichiers.

**Impact LLM :** SWE-bench : 55% des issues "hard" sont multi-fichiers vs 3% des "easy". Confiance: empirique.

**Before :**
```
# Ajouter "phone" requiert 5 fichiers :
src/types/user.ts
src/models/user.ts
src/services/user.ts
src/controllers/user.ts
src/validators/user.ts
```

**After :**
```
# Ajouter "phone" requiert 1-2 fichiers :
src/features/user/user.ts      # type + schema + logique + validation
src/features/user/user.test.ts
```

**Strategies :** Colocaliser types/schema/logique, Zod/serde comme source-of-truth, feature folders, identifier fichiers "hub".

**Effort :** Strategic (1-2 semaines)

---

### Pattern 8.2 : Decomposer les fichiers "hub"

**Ecart :** Un fichier modifie dans >20% des commits.

**Impact LLM :** Point de couplage + conflit frequents. Confiance: empirique (git analysis).

**Before :**
```
# 47% des commits touchent src/lib/api.ts
```

**After :**
```
src/features/auth/api.ts       # auth API calls
src/features/billing/api.ts    # billing API calls
src/shared/api-client.ts       # HTTP client (rarement modifie)
```

**Effort :** Medium (1-2 jours)

---

## Dimension 9 — Build & Verification

### Pattern 9.1 : Documenter les commandes de build

**Ecart :** Commandes de build non documentees — l'agent doit deviner.

**Impact LLM :** Boris Cherny : "the most important thing: give Claude a way to verify its work — 2-3x quality". Sans commandes claires, pas de feedback loop. Confiance: officiel (Anthropic).

**Before :**
```
# Aucune doc — faut lire package.json, deviner les scripts
```

**After :**
```markdown
# CLAUDE.md
## Commands
- Build: `pnpm build`
- Test all: `pnpm test`
- Test single: `pnpm test -- path/to/file`
- Lint: `pnpm lint`
- Type check: `pnpm typecheck`
- Dev: `pnpm dev`
```

**Effort :** Quick win (10 min)

---

### Pattern 9.2 : Ajouter la CI

**Ecart :** Pas de CI — l'agent ne peut pas verifier que ses changements passent.

**Impact LLM :** Factory.ai : Build System est un pilier critique. Sans CI, pas de verification automatisee. Confiance: officiel (Factory.ai).

**Before :**
```
# Pas de .github/workflows/
```

**After :**
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - run: pnpm install --frozen-lockfile
      - run: pnpm typecheck
      - run: pnpm lint
      - run: pnpm test
```

**Effort :** Quick win (30 min)

---

### Pattern 9.3 : Ajouter des pre-commit hooks

**Ecart :** Pas de feedback loop automatisee avant commit.

**Impact LLM :** Les pre-commit hooks empechent les regressions avant meme le push. Confiance: heuristique (Factory.ai, Kodus).

**Before :**
```
# Linting et formatting manuels
```

**After :**
```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"]
  }
}
```

**Effort :** Quick win (20 min)

---

## Dimension 10 — Dev Environment

### Pattern 10.1 : Ajouter .env.example

**Ecart :** Variables d'environnement non documentees.

**Impact LLM :** L'agent ne sait pas quelles variables sont requises et cree du code qui echoue au runtime. Confiance: heuristique (Factory.ai, Kodus).

**Before :**
```
# Pas de .env.example — faut demander a l'equipe
```

**After :**
```bash
# .env.example
DATABASE_URL=postgresql://localhost:5432/myapp
REDIS_URL=redis://localhost:6379
API_KEY=sk-test-your-key-here
# Required for auth
AUTH_SECRET=generate-with-openssl-rand-base64-32
```

**Effort :** Quick win (15 min)

---

### Pattern 10.2 : Ajouter un script de setup

**Ecart :** Setup multi-etapes non documente.

**Impact LLM :** L'agent (et les nouveaux developpeurs) perdent du temps a deviner le setup. Confiance: heuristique.

**Before :**
```
# README mentionne vaguement "install dependencies" sans details
```

**After :**
```bash
#!/bin/bash
# scripts/setup.sh
set -e
echo "Installing dependencies..."
pnpm install
echo "Setting up environment..."
cp .env.example .env
echo "Setting up database..."
pnpm db:migrate
echo "Ready! Run 'pnpm dev' to start."
```

**Effort :** Quick win (20 min)

---

### Pattern 10.3 : Ajouter le version pinning

**Ecart :** Pas de version pinning — versions differentes entre devs/CI/agent.

**Impact LLM :** L'agent peut utiliser une version differente de Node/Rust et rencontrer des erreurs inexplicables. Confiance: heuristique.

**Before :**
```
# Chacun utilise sa version de Node
```

**After :**
```
# .node-version
20.11.0

# .tool-versions (asdf)
nodejs 20.11.0
pnpm 9.1.0
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "1.77"
```

**Effort :** Quick win (5 min)

---

## Dimension 11 — Naming Expressiveness

### Pattern 11.1 : Renommer les identifiants cryptiques

**Ecart :** Fonctions, variables ou fichiers avec des noms cryptiques ou generiques.

**Impact LLM :** arXiv 2307.12488v5 : anonymisation des identifiants → Java -75% precision, Python -65%. Les function definition names ont l'impact individuel le plus fort. Boris Cherny : "glob and grep bat le RAG" — prerequis : noms expressifs. Confiance: empirique.

**Before :**
```typescript
function proc(d: any): any {
  const r = d.items.filter((i: any) => i.s === 'active');
  return r.map((i: any) => ({ n: i.name, v: i.val }));
}
```

**After :**
```typescript
function filterActiveItems(dataset: ItemCollection): ActiveItemSummary[] {
  const activeItems = dataset.items.filter((item) => item.status === 'active');
  return activeItems.map((item) => ({ name: item.name, value: item.value }));
}
```

**Effort :** Medium (variable — peut etre assiste par Claude Code)

---

### Pattern 11.2 : Extraire les magic strings en constantes nommees

**Ecart :** Literals non assignes a des constantes.

**Impact LLM :** L'agent ne comprend pas la semantique de `'active'` ou `30000`. Une constante nommee donne le contexte. Confiance: heuristique.

**Before :**
```typescript
if (user.role === 'admin') { setTimeout(fn, 30000); }
```

**After :**
```typescript
const ADMIN_ROLE = 'admin' as const;
const PAYMENT_PROVIDER_TIMEOUT_MS = 30_000; // SLA requirement
if (user.role === ADMIN_ROLE) { setTimeout(fn, PAYMENT_PROVIDER_TIMEOUT_MS); }
```

**Effort :** Quick win (ongoing)

---

## Prioritisation des remediations

### Tier 1 — Quick Wins (1-2 heures total)
1. Documenter les commandes build (Pattern 9.1) — +10 Build & Verification — Confiance: officiel
2. Creer CLAUDE.md avec Compact Instructions (Pattern 5.1 + 5.6) — +10-15 AI Config — Confiance: officiel
3. Creer AGENTS.md (Pattern 5.4) — +5-10 AI Config — Confiance: officiel
4. Ajouter .env.example (Pattern 10.1) — +5-10 Dev Environment — Confiance: heuristique
5. Ajouter index files (Pattern 3.1) — +5-10 Import Simplicity — Confiance: heuristique
6. Colocaliser les tests (Pattern 6.1) — +10-15 Test Infra + Vertical Cohesion — Confiance: officiel
7. Implementer hooks critiques (Pattern 5.2) — +3-5 AI Config — Confiance: officiel
8. Ajouter CI (Pattern 9.2) — +10-15 Build & Verification — Confiance: officiel
9. Extraire magic strings (Pattern 11.2) — +3-5 Naming Expressiveness — Confiance: heuristique

### Tier 2 — Restructurations (1-2 jours)
1. Decomposer god-files (Pattern 1.1) — +10-20 File & Dir Org — Confiance: heuristique
2. Eliminer les `any` (Pattern 4.1) — +10-15 Type Expressiveness — Confiance: empirique
3. Remplacer unwrap (Pattern 4.2) — +10-15 Type Expressiveness — Confiance: heuristique
4. Decomposer fichiers "hub" (Pattern 8.2) — +5-10 Change Locality — Confiance: empirique
5. @imports progressive disclosure (Pattern 5.3) — +3-5 AI Config — Confiance: officiel
6. Renommer identifiants cryptiques (Pattern 11.1) — +5-10 Naming Expressiveness — Confiance: empirique
7. Architecture subagent (Pattern 5.7) — +3-5 AI Config — Confiance: officiel

### Tier 3 — Refactors architecturaux (1+ semaine)
1. Migration horizontal → vertical (Pattern 2.1) — +20-30 Vertical Cohesion + Change Locality — Confiance: heuristique
2. Extraire composants monolithiques (Pattern 1.3) — +10-20 File & Dir Org — Confiance: empirique
3. Reduire couplage temporel (Pattern 8.1) — +10-20 Change Locality — Confiance: empirique
4. Terminer les migrations incompletes — +5-15 File & Dir Org (migration completeness) — Confiance: officiel (Boris Cherny)
