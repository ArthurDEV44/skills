---
model: opus
name: saas-storytelling
description: >
  Storytelling et copywriting pour tout SaaS GenAI. Genere du contenu structure
  par type : landing-page, blog, article, dashboard-ui, changelog, feature-announcement.
  Part du probleme utilisateur, cree de la tension, resout par le produit. Ton direct,
  dense, zero bullshit marketing. Use when the user says "storytelling", "landing page copy",
  "blog post", "write a blog", "feature announcement", "changelog entry", "dashboard copy",
  "microcopy", "saas-storytelling", "write copy for", "copywriting", "onboarding copy",
  "empty state", "value prop", or asks to write marketing/product content for a SaaS product.
  Do NOT use for SEO-only tasks (use seo-warfare), visual design (use frontend-design),
  or PRD generation (use write-prd).
argument-hint: "[type] [context]"
---

# saas-storytelling — GenAI SaaS Narrative Engine

## Phase 0: CLASSIFY

Parse `$ARGUMENTS` pour extraire le type et le contexte.

**0a. Extraction des parametres :**

- `$ARGUMENTS` format attendu : `[type] [context]`
- `type` = premier mot, doit matcher un des types ci-dessous
- `context` = tout le reste — sujet, feature, angle demande

**Types valides :**

| Type | Description | Output |
|------|-------------|--------|
| `landing-page` | Page complete ou section de landing | Hero + value props + social proof + CTA |
| `blog` | Article de blog long format (800-1500 mots) | Hook + corps structure + CTA |
| `article` | Piece editoriale / thought leadership | Hook + argumentation + conclusion |
| `dashboard-ui` | Microcopy pour l'interface produit | Tooltips, empty states, onboarding, labels |
| `changelog` | Note de mise a jour produit | What changed + why it matters + what's next |
| `feature-announcement` | Annonce de feature (blog + social) | Announcement post + tweet thread + email |

**0b. Si le type n'est pas reconnu** → demander au user.
**0c. Si le contexte est vide** → demander au user quel sujet ou quelle feature.
**0d. Detecter la langue :**
- Si `$ARGUMENTS` contient du francais → generer en francais
- Si `$ARGUMENTS` contient de l'anglais → generer en anglais
- Si ambigu → demander au user

## Execution Flow

```
$ARGUMENTS → [type] [context]
       |
       v
+---------------+
|  Phase 0:     |
|  CLASSIFY     |  ← Parse type + context, detect langue
+-------+-------+
        |
        v
+---------------+
|  Phase 1:     |
|  RESEARCH     |  ← Decouvrir le produit, persona, concurrence
+-------+-------+
        |
        v
+---------------+
|  Phase 2:     |
|  ANGLE        |  ← Definir l'angle narratif specifique (PAS generique)
+-------+-------+
        |
        v
+---------------+
|  Phase 3:     |
|  DRAFT        |  ← Generer le contenu structure selon le type
+-------+-------+
        |
        v
+---------------+
|  Phase 4:     |
|  BULLSHIT     |  ← Scanner et eliminer le copy generique
|  FILTER       |
+-------+-------+
        |
        v
+---------------+
|  Phase 5:     |
|  OUTPUT       |  ← Livrer le contenu final + variantes
+-------+-------+
```

## Phase 1: RESEARCH

**Goal:** Decouvrir le produit et comprendre le contexte specifique de cette generation.

**1a. Decouverte du produit (Product Discovery) :**

Scanner le codebase pour construire un profil produit :

```
Glob: README.md, CLAUDE.md, package.json, Cargo.toml, pyproject.toml
Glob: **/landing/**,  **/marketing/**, **/content/**
Glob: **/changelog*, **/CHANGELOG*
```

Extraire et synthetiser dans ce format :

```yaml
product_profile:
  name: "{nom du produit}"
  tagline: "{tagline existante si trouvee}"
  category: "{type de SaaS — ex: AI research tool, dev platform, analytics}"
  differentiators: ["{diff_1}", "{diff_2}", "{diff_3}"]
  target_personas: ["{persona_1}", "{persona_2}"]
  competitors: ["{concurrent_1}", "{concurrent_2}"]
  tone_existing: "{ton detecte dans le copy existant}"
  pricing_model: "{freemium | subscription | usage-based | BYOK | unknown}"
```

Si le codebase ne permet pas de deduire le produit :
- Verifier si l'utilisateur a fourni du contexte dans `$ARGUMENTS`
- Si insuffisant → demander au user de decrire le produit en 2-3 phrases (nom, ce que ca fait, pour qui)

**1b. Pour les types `blog`, `article`, `feature-announcement` :**

Spawn `agent-websearch` pour rechercher :

```
Agent(
  description: "Research context for {type} about {context}",
  prompt: "Research the following topic in the {product_category} space: '{context}'. Find: (1) How competitors position similar features, (2) What pain points users express about this topic on Reddit/HN/Twitter, (3) Current trends and angles used in successful SaaS content about this topic. Return: 3 competitor angles, 3 user pain points with quotes, 2 trending angles. Keep output under 400 words.",
  subagent_type: "agent-websearch"
)
```

**1c. Pour les types `landing-page`, `dashboard-ui` :**

Si un codebase est present, spawn `agent-explore` :

```
Agent(
  description: "Explore UI patterns for {context}",
  prompt: "Find all UI components, pages, and copy related to '{context}' in this codebase. Extract: (1) Existing microcopy and labels, (2) Component structure and layout, (3) Tone and vocabulary already used. Return a structured inventory of existing copy.",
  subagent_type: "agent-explore"
)
```

**GATE:** Le `product_profile` est construit — au minimum `name` + `category` + 1 `differentiator`. Persona cible identifie + probleme utilisateur clair. Si agent-websearch echoue, continuer avec le contexte produit interne.

---

## Phase 2: ANGLE (le plus important)

**Goal:** Definir UN angle narratif specifique, pas un pitch generique.

Un bon angle repond a cette question : **"Quelle tension specifique ce contenu va-t-il creer et resoudre ?"**

**2a. Choisir le framework narratif :**

| Framework | Quand l'utiliser | Structure |
|-----------|-----------------|-----------|
| **Problem-Agitation-Solution** | Landing pages, features | Probleme → ca empire → {product_name} resout |
| **Before/After/Bridge** | Changelogs, announcements | Avant c'etait penible → maintenant c'est resolu → voici comment |
| **Contrarian Take** | Blogs, articles | "Tout le monde pense X, mais en realite Y" |
| **Show Don't Tell** | Dashboard UI, onboarding | Pas d'explication — l'interface parle d'elle-meme |
| **User Story** | Case studies, social proof | Un vrai scenario, une vraie frustration, un vrai resultat |

Consult `references/narrative-frameworks.md` for detailed templates and examples of each framework.

**2b. Formuler l'angle en UNE phrase :**

Format : `"Pour [persona], le probleme c'est [frustration specifique]. {product_name} [resolution concrete] parce que [mecanisme unique]."`

Exemples de bons vs mauvais angles :
- ❌ "{product_name} revolutionne [domaine]" (generique, interdit)
- ✅ "[Persona] passe [temps] a [tache frustrante]. {product_name} [action concrete] en [duree/mecanisme]."
- ❌ "Boostez votre productivite avec l'IA" (vide, interdit)
- ✅ "Vous avez [situation concrete]. {product_name} [resolution] parce que [mecanisme specifique du produit]."

**2c. Valider l'angle :**
- [ ] L'angle mentionne un persona specifique (pas "les utilisateurs")
- [ ] L'angle contient une frustration concrete et mesurable
- [ ] L'angle decrit un mecanisme du produit (pas juste "IA")
- [ ] L'angle ne contient aucune formule de la Blacklist (voir Phase 4)

**GATE:** L'angle passe les 4 criteres de validation. Si un critere echoue, reformuler avant de passer a Phase 3.

---

## Phase 3: DRAFT

**Goal:** Generer le contenu structure selon le type.

### Type: `landing-page`

Consult `references/landing-page-anatomy.md` for section-by-section guidance.

```markdown
## [HERO]
Headline: [Tension en 8-12 mots — PAS une description produit]
Subheadline: [Resolution en 1 phrase — comment {product_name} resout]
CTA primaire: [Action specifique, pas "Commencer"]
CTA secondaire: [Alternative basse friction]

## [PROBLEME]
[3-4 phrases qui decrivent le quotidien frustrant du persona SANS mentionner le produit]

## [VALUE PROPS] (3 blocs)
### [VP1: Feature → Benefice]
[2 phrases : ce que ca fait → pourquoi ca change la donne]

### [VP2: Feature → Benefice]
[2 phrases]

### [VP3: Feature → Benefice]
[2 phrases]

## [SOCIAL PROOF]
[Temoignage ou metrique — si pas disponible, indiquer {{PLACEHOLDER: temoignage a collecter}}]

## [OBJECTION HANDLER]
[Repondre a la principale objection du persona]

## [CTA FINAL]
[Reprise de la tension initiale + CTA]
```

### Type: `blog`

```markdown
## Metadata
title: [< 60 chars, inclut le mot-cle longue traine]
meta_description: [< 155 chars, inclut le hook]
slug: [url-friendly]
target_keyword: [mot-cle principal]
internal_links: [pages a lier]

## [HOOK] (premier paragraphe)
[Ouvrir sur une scene, un chiffre, ou une question qui cree de la tension — JAMAIS sur "Dans un monde ou..."]

## [CORPS] (3-5 sections avec H2)
### [H2: Question ou affirmation forte]
[Argumentation + exemples concrets + lien vers feature si pertinent]

### [H2: ...]
[...]

## [CONCLUSION + CTA]
[Boucler sur la tension du hook + appel a l'action]
```

### Type: `article`

Meme structure que `blog` mais :
- Ton plus editorial, moins commercial
- Pas de CTA agressif — CTA subtil en fin d'article
- Longueur : 1200-2000 mots
- Angle : thought leadership, pas promotion produit

### Type: `dashboard-ui`

```markdown
## Microcopy Inventory

### [Composant/Page]

| Element | Copy FR | Copy EN | Note |
|---------|---------|---------|------|
| Page title | ... | ... | |
| Empty state headline | ... | ... | Doit donner envie d'agir, pas juste "Rien ici" |
| Empty state body | ... | ... | Expliquer QUOI faire, pas ce qui manque |
| Empty state CTA | ... | ... | Verbe d'action specifique |
| Tooltip: [feature] | ... | ... | Max 120 chars, expliquer le POURQUOI pas le COMMENT |
| Error message | ... | ... | Dire ce qui s'est passe + comment resoudre |
| Success message | ... | ... | Confirmer + next step |
| Onboarding step N | ... | ... | |
```

**Regles microcopy :**
- Empty states = opportunite de conversion, pas un cul-de-sac
- Tooltips : expliquer pourquoi cette feature existe, pas comment cliquer
- Erreurs : jamais "Une erreur est survenue", toujours dire QUOI et COMMENT fixer
- Succes : confirmer l'action + proposer l'etape suivante
- Labels : verbe d'action > nom abstrait ("Importer des documents" > "Import")

### Type: `changelog`

```markdown
## [Version X.Y.Z] — [Date]

### [Headline: benefice, pas feature]

**Avant :** [la situation frustrante]
**Maintenant :** [ce qui change concretement]

#### Ce qui a change
- [changement 1] — [pourquoi ca compte]
- [changement 2] — [pourquoi ca compte]

#### Prochaine etape
[Ce sur quoi l'equipe travaille, lien vers roadmap si applicable]
```

### Type: `feature-announcement`

Generer 3 assets :

```markdown
## 1. Blog Post (format blog ci-dessus)
[800-1200 mots]

## 2. Thread Social (5-7 posts)
Post 1: [Hook — le probleme]
Post 2: [Agitation — pourquoi c'est pire qu'on pense]
Post 3: [Ce qu'on a construit]
Post 4: [Comment ca marche (concret)]
Post 5: [Resultat / metrique]
Post 6: [CTA]

## 3. Email Announcement
Subject: [< 50 chars, curiosite > description]
Preview text: [< 90 chars]
Body: [Before/After/Bridge, < 200 mots, 1 CTA]
```

**GATE:** Le contenu genere est complet pour le type demande (tous les blocs du template sont remplis). Aucun bloc ne contient de placeholder non-justifie (sauf social proof si pas de donnees reelles).

---

## Phase 4: BULLSHIT FILTER

**Goal:** Scanner et eliminer tout copy generique. C'est le garde-rail le plus important.

**4a. Blacklist — Formules interdites :**

Chercher et remplacer systematiquement :

| Formule interdite | Pourquoi | Remplacement |
|-------------------|----------|--------------|
| "Revolutionnez votre [X]" | Vide, surutilise | Decrire le changement concret |
| "Boostez votre productivite" | Generique | Donner le gain mesurable |
| "Solution innovante" | Tout le monde le dit | Decrire le mecanisme specifique |
| "Alimentee par l'IA" / "Powered by AI" | Obvious en 2026 | Nommer le modele ou la technique |
| "Dans un monde ou..." | Ouverture paresseuse | Commencer par le probleme |
| "Grace a notre technologie de pointe" | Fluff corporate | Expliquer ce que la techno fait |
| "Seamless / Transparent / Intuitive" | Adjectifs vides | Montrer avec un exemple |
| "Debloquez le potentiel de..." | Jargon marketing | Dire ce qui est concretement possible |
| "Game-changer" / "Disruptif" | Buzzwords | Prouver avec un fait |
| "Passez au niveau superieur" | Generique | Decrire le niveau en question |
| "Gain de temps considerable" | Vague | "Reduit de 4h a 20 minutes" |
| "Tout-en-un" / "All-in-one" | Sur-promesse | Lister les 3 choses que ca fait vraiment |
| "Simple et puissant" | Contradiction vague | Montrer la simplicite avec un screenshot ou une etape |
| "Faites-en plus avec moins" | Cliche | Decrire le "plus" et le "moins" |

**4b. Tests de qualite :**

Pour CHAQUE paragraphe du draft, verifier :

- [ ] **Test du "So what?"** — Si on peut repondre "et alors?" a une phrase, elle est trop vague → la reformuler avec un fait specifique
- [ ] **Test du concurrent** — Si on peut remplacer "{product_name}" par n'importe quel concurrent et que la phrase reste vraie, elle est generique → la rendre specifique au produit
- [ ] **Test du chiffre** — Chaque claim de performance ou de benefice est-il etaye par un chiffre, un scenario, ou un exemple ? Si non → ajouter
- [ ] **Test de la tension** — Le contenu cree-t-il une tension (probleme, frustration, risque) avant de resoudre ? Si non → restructurer

**4c. Rewrite :**

Pour chaque element qui echoue un test, reecrire en appliquant la regle. Ne pas juste supprimer — transformer en quelque chose de specifique.

**GATE:** Zero formule de la Blacklist presente. Tous les paragraphes passent les 4 tests. Si des elements echouent encore, les reecrire une deuxieme fois (max 2 iterations).

---

## Phase 5: OUTPUT

**Goal:** Livrer le contenu final structure.

**5a. Format de sortie :**

```markdown
## Contenu Final

**Produit:** {product_name}
**Type:** {type}
**Langue:** {FR | EN}
**Angle:** {l'angle defini en Phase 2}
**Persona cible:** {persona principal}

---

{contenu structure selon le template du type}

---

## Notes Editoriales
- **Variante de headline testable :** {alternative headline pour A/B test}
- **Mots-cles longue traine :** {si blog/article — 3-5 keywords}
- **Liens internes suggeres :** {pages du site a lier}
- **Social proof a collecter :** {temoignages ou metriques manquants}
```

**5b. Si le type est `feature-announcement` :** livrer les 3 assets (blog + thread + email) dans un seul output.

**5c. Si le type est `dashboard-ui` :** livrer le tableau FR + EN avec toutes les entrees.

## Hard Rules

1. **Phase 0 ALWAYS runs** — classifier avant de generer quoi que ce soit
2. **Phase 1 Product Discovery est obligatoire** — ne jamais generer de contenu sans comprendre le produit
3. **Phase 2 (ANGLE) est obligatoire** — ne jamais generer de contenu sans angle valide
4. **Phase 4 (BULLSHIT FILTER) est obligatoire** — chaque draft est filtre, sans exception
5. **Max 2 iterations de rewrite** en Phase 4 — si le copy est encore generique apres 2 passes, livrer avec un warning
6. **Respecter le template du type** — tous les blocs doivent etre presents
7. **Bilingue pour `dashboard-ui`** — toujours FR + EN
8. **Citations sourced** — tout chiffre ou claim doit avoir une source ou etre marque `{{a verifier}}`
9. **Framework-aware** — si un codebase existe, generer du copy compatible avec les composants existants
10. **Product-specific** — chaque phrase doit etre specifique au produit decouvert en Phase 1, jamais generique
11. **Print `[Phase N/5]` progress headers** avant chaque phase — ne jamais sauter l'indicateur

## DO NOT

1. Utiliser une formule de la Blacklist Phase 4 — meme reformulee, meme "ironiquement"
2. Ouvrir un blog par "Dans un monde ou..." ou "A l'ere de l'IA..." — interdit sans exception
3. Generer du copy sans avoir defini l'angle en Phase 2 — la sequence est non-negociable
4. Ecrire "les utilisateurs" quand tu peux ecrire un persona specifique — toujours nommer le role
5. Promettre des features qui n'existent pas — si pas dans le codebase ou le contexte, ne pas inventer
6. Utiliser des superlatifs sans preuve ("le meilleur", "le plus rapide", "le seul") sauf si demontrable
7. Generer du dashboard-ui copy sans version anglaise — le produit est international
8. Mettre un CTA avec un pricing non confirme — utiliser "Essayer {product_name}" par defaut
9. Faire du name-dropping de concurrents dans le copy public — comparer en interne, pas en output client
10. Generer plus de 2000 mots pour un blog sans demande explicite du user
11. Hardcoder des details produit sans les avoir valides dans le codebase ou avec le user

## Error Handling

| Scenario | Fallback |
|----------|----------|
| **Produit non identifiable** | Demander au user de decrire le produit en 2-3 phrases (nom, ce que ca fait, pour qui). |
| **agent-websearch echoue** | Baser l'angle sur le contexte produit interne uniquement. Marquer "Competitive context unavailable" dans les notes editoriales. |
| **agent-explore echoue** | Travailler sans contexte codebase. Generer du copy sans references aux composants existants. |
| **Type non reconnu** | Demander au user de choisir parmi les 6 types valides. |
| **Contexte insuffisant** | Demander au user de preciser le sujet, la feature, ou l'audience ciblee. |
| **Langue ambigue** | Demander au user. Defaut : francais. |
| **Pas de codebase** | Travailler uniquement a partir du contexte fourni par le user. |
| **Bullshit Filter echoue apres 2 iterations** | Livrer avec un warning listant les phrases encore generiques + suggestion de reformulation manuelle. |

## References

- [Narrative Frameworks](references/narrative-frameworks.md) — story arcs, PAS, BAB, contrarian, show don't tell
- [Landing Page Anatomy](references/landing-page-anatomy.md) — section-by-section high-converting patterns
- [Voice & Tone](references/voice-tone.md) — calibration du ton par persona et format, registre FR vs EN
- [Conversion Patterns](references/conversion-patterns.md) — CTA, objection handling, social proof, pricing psychology

## Done When

- [ ] Phase 0 (Classify) completed — type and context identified
- [ ] Phase 1 (Research) completed — product profile built
- [ ] Phase 2 (Angle) completed — angle passes 4 validation criteria
- [ ] Phase 3 (Draft) completed — all template blocks filled
- [ ] Phase 4 (Bullshit Filter) passed — zero blacklisted formulas, all 4 tests pass
- [ ] Phase 5 (Output) delivered with editorial notes and variantes

## Constraints (Three-Tier)

### ALWAYS
- Run Phase 0 (Classify) before generating content
- Build product profile (Phase 1) before writing
- Define and validate an angle (Phase 2) before drafting
- Run Bullshit Filter (Phase 4) on every draft — no exceptions
- Generate FR + EN for `dashboard-ui` type

### ASK FIRST
- Product identity when not detectable from codebase
- Content type when `$ARGUMENTS` doesn't match valid types
- Language when ambiguous

### NEVER
- Use blacklisted formulas from Phase 4 — even reformulated or "ironic"
- Open a blog with "Dans un monde ou..." or "A l'ere de l'IA..."
- Promise features not in the codebase or user context
- Use superlatifs without proof ("le meilleur", "le plus rapide")
- Generate dashboard-ui copy without English version
