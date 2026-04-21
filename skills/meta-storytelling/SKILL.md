---
model: opus
name: meta-storytelling
description: >
  Storytelling et copywriting pour tout type de projet — SaaS, ecommerce, marketplace,
  app mobile, agence, portfolio, infoproduit, media, produit physique, dev tool,
  cours en ligne, association, etc. Le LLM detecte automatiquement le type de projet
  et adapte l'angle narratif, le ton, les mecanismes de conversion et les preuves sociales
  pour etre pertinent sur le format (landing-page, blog, article, product-ui, changelog,
  feature-announcement). Part du probleme utilisateur, cree de la tension, resout par
  le produit. Ton direct, dense, zero bullshit marketing. Use when the user says
  "storytelling", "landing page copy", "blog post", "write a blog", "feature announcement",
  "changelog entry", "product copy", "microcopy", "meta-storytelling", "write copy for",
  "copywriting", "onboarding copy", "empty state", "value prop", "home page copy",
  "about page", "pitch", "ecom copy", "product description", or asks to write
  marketing/product content for any project. Do NOT use for SEO-only tasks (use
  seo-warfare), visual design (use frontend-design), or PRD generation (use write-prd).
argument-hint: "[type] [context]"
---

# meta-storytelling — Universal Narrative Engine

## Phase 0: CLASSIFY

Parse `$ARGUMENTS` pour extraire le type de contenu et le contexte.

**0a. Extraction des parametres :**

- `$ARGUMENTS` format attendu : `[type] [context]`
- `type` = premier mot, doit matcher un des types ci-dessous
- `context` = tout le reste — sujet, feature, angle demande

**Types de contenu valides :**

| Type | Description | Output |
|------|-------------|--------|
| `landing-page` | Page complete ou section d'accueil | Hero + value props + social proof + CTA |
| `blog` | Article de blog long format (800-1500 mots) | Hook + corps structure + CTA |
| `article` | Piece editoriale / thought leadership | Hook + argumentation + conclusion |
| `product-ui` | Microcopy pour l'interface (app, site, dashboard, checkout) | Tooltips, empty states, onboarding, labels, erreurs |
| `changelog` | Note de mise a jour | What changed + why it matters + what's next |
| `feature-announcement` | Annonce de feature / nouveaute (blog + social) | Announcement post + thread + email |

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
|  RESEARCH     |  ← Detecter le type de projet, persona, concurrence
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

**Goal:** Detecter le type de projet, decouvrir le produit, comprendre le contexte specifique.

**1a. Project Type Detection (obligatoire) :**

Analyser le codebase, les assets et le contexte pour classifier le projet. Cette classification conditionne TOUT le reste (ton, mecanismes de conversion, preuves sociales, CTAs, objections).

**Signaux de detection :**

| Signal | Type probable |
|--------|---------------|
| `package.json` avec Stripe + multi-tenant + auth + dashboard | `saas` |
| `package.json` avec Shopify / WooCommerce / Medusa / cart / checkout | `ecommerce` |
| App native (Swift, Kotlin, Flutter, React Native) + store listing | `mobile-app` |
| Pages type "Services" + "Portfolio" + formulaire de contact B2B | `agency` |
| Pages projets + bio + CV + pas de checkout | `portfolio` |
| Stripe Checkout + page unique + upsells + "Accedez au cours" | `infoproduct` |
| Catalogue + vendeurs + commissions + recherche multi-acteurs | `marketplace` |
| Articles editoriaux dominants + publicite + newsletter | `media` |
| Produit tangible + fiches specifications + stock + livraison | `physical-product` |
| API + docs developpeurs + CLI + pricing usage-based | `devtool` |
| Modules + videos + quiz + certification | `course` |
| Don + mission + impact + beneficiaires + transparence financiere | `nonprofit` |
| Aucun signal clair | `generic` |

**Si aucun signal clair** → demander au user : "Quel type de projet ? (SaaS / ecommerce / app mobile / agence / portfolio / infoproduit / marketplace / media / produit physique / devtool / cours / asso / autre)".

**Scanner le codebase pour construire le profil :**

```
Glob: README.md, CLAUDE.md, package.json, Cargo.toml, pyproject.toml, Gemfile, composer.json
Glob: **/landing/**,  **/marketing/**, **/content/**, **/(home|index|about|pricing|services)*
Glob: **/changelog*, **/CHANGELOG*
```

**1b. Construire le `project_profile` :**

```yaml
project_profile:
  type: "{saas | ecommerce | mobile-app | agency | portfolio | infoproduct | marketplace | media | physical-product | devtool | course | nonprofit | generic}"
  name: "{nom du produit / marque / projet}"
  tagline: "{tagline existante si trouvee}"
  category: "{sous-categorie — ex: AI research tool, fashion DTC, legal agency, photography portfolio}"
  differentiators: ["{diff_1}", "{diff_2}", "{diff_3}"]
  target_personas: ["{persona_1}", "{persona_2}"]
  competitors: ["{concurrent_1}", "{concurrent_2}"]
  tone_existing: "{ton detecte dans le copy existant}"
  monetization: "{subscription | usage-based | one-time | commission | freemium | ads | donation | service-fee | retainer | free}"
  conversion_goal: "{signup | purchase | download | book-call | submit-form | subscribe-newsletter | donate | enroll}"
```

Adapter `monetization` et `conversion_goal` selon le `type` detecte :

| type | monetization typique | conversion_goal typique |
|------|---------------------|------------------------|
| `saas` | subscription / freemium / usage-based | signup / book-demo |
| `ecommerce` | one-time (panier) | purchase |
| `mobile-app` | freemium / subscription / one-time | download / signup |
| `agency` | retainer / project-based | book-call / submit-brief |
| `portfolio` | free (acquisition) | contact / view-project |
| `infoproduct` | one-time / payment-plan | purchase / enroll |
| `marketplace` | commission / service-fee | signup-buyer / signup-seller |
| `media` | subscription / ads / sponsorship | subscribe-newsletter / become-member |
| `physical-product` | one-time | purchase / pre-order |
| `devtool` | usage-based / subscription / free OSS | install / signup / star |
| `course` | one-time / payment-plan | enroll / free-trial-lesson |
| `nonprofit` | donation / grant | donate / volunteer / subscribe |

Si le codebase ne permet pas de deduire le produit :
- Verifier si l'utilisateur a fourni du contexte dans `$ARGUMENTS`
- Si insuffisant → demander au user de decrire le projet en 2-3 phrases (nom, ce que c'est, pour qui, comment ca monetise)

**1c. Pour les types `blog`, `article`, `feature-announcement` :**

Spawn `agent-websearch` pour rechercher :

```
Agent(
  description: "Research context for {type} about {context}",
  prompt: "Research the following topic in the {project_profile.category} space ({project_profile.type}): '{context}'. Find: (1) How competitors / peers position similar content, (2) What pain points the target persona expresses on Reddit/HN/Twitter/Instagram/TikTok (choose sources by project type), (3) Current trends and angles used in successful content about this topic. Return: 3 competitor angles, 3 user pain points with quotes, 2 trending angles. Keep output under 400 words.",
  subagent_type: "agent-websearch"
)
```

**1d. Pour les types `landing-page`, `product-ui` :**

Si un codebase est present, spawn `agent-explore` :

```
Agent(
  description: "Explore UI patterns for {context}",
  prompt: "Find all UI components, pages, and copy related to '{context}' in this codebase. Extract: (1) Existing microcopy and labels, (2) Component structure and layout, (3) Tone and vocabulary already used. Return a structured inventory of existing copy.",
  subagent_type: "agent-explore"
)
```

**GATE:** Le `project_profile` est construit — au minimum `type` + `name` + `category` + 1 `differentiator`. Persona cible identifie + probleme utilisateur clair. Si agent-websearch echoue, continuer avec le contexte interne.

---

## Phase 2: ANGLE (le plus important)

**Goal:** Definir UN angle narratif specifique, adapte au type de projet, pas un pitch generique.

Un bon angle repond a cette question : **"Quelle tension specifique ce contenu va-t-il creer et resoudre pour ce persona, sur ce type de projet ?"**

**2a. Choisir le framework narratif :**

| Framework | Quand l'utiliser | Structure |
|-----------|-----------------|-----------|
| **Problem-Agitation-Solution** | Landing pages, features, services d'agence | Probleme → ca empire → {name} resout |
| **Before/After/Bridge** | Changelogs, announcements, transformations (cours, coaching) | Avant c'etait penible → maintenant c'est resolu → voici comment |
| **Contrarian Take** | Blogs, articles, pieces editoriales | "Tout le monde pense X, mais en realite Y" |
| **Show Don't Tell** | Product-UI, onboarding, portfolio, ecommerce hero | Pas d'explication — l'interface / l'image / le case parle d'elle-meme |
| **User Story** | Case studies, social proof, pages de projet portfolio/agence | Un vrai scenario, une vraie frustration, un vrai resultat |
| **Mission-Driven** | Nonprofit, media independant, marque a valeurs | Pourquoi on existe → ce qu'on change → comment tu participes |
| **Transformation Arc** | Infoproduct, course, coaching | Etat initial du persona → parcours → etat vise |

Consult `references/narrative-frameworks.md` for detailed templates and examples of each framework.

**2b. Formuler l'angle en UNE phrase :**

Format : `"Pour [persona], le probleme c'est [frustration specifique]. {name} [resolution concrete] parce que [mecanisme unique au projet]."`

Exemples de bons vs mauvais angles :
- ❌ "{name} revolutionne [domaine]" (generique, interdit)
- ✅ "[Persona] passe [temps] a [tache frustrante]. {name} [action concrete] en [duree/mecanisme]."
- ❌ "Boostez votre productivite avec l'IA" (vide, interdit)
- ✅ "Vous avez [situation concrete]. {name} [resolution] parce que [mecanisme specifique du produit ou du service]."

**2c. Valider l'angle :**
- [ ] L'angle mentionne un persona specifique (pas "les utilisateurs" / "les clients")
- [ ] L'angle contient une frustration concrete et mesurable
- [ ] L'angle decrit un mecanisme specifique (feature produit, methodologie, savoir-faire, philosophie editoriale)
- [ ] L'angle ne contient aucune formule de la Blacklist (voir Phase 4)

**GATE:** L'angle passe les 4 criteres de validation. Si un critere echoue, reformuler avant de passer a Phase 3.

---

## Phase 3: DRAFT

**Goal:** Generer le contenu structure selon le type de contenu ET le type de projet.

### Type: `landing-page`

Consult `references/landing-page-anatomy.md` for section-by-section guidance.

Le CTA primaire doit correspondre au `project_profile.conversion_goal` (signup, purchase, book-call, download, donate, enroll…). Les sections peuvent varier selon le type de projet :
- `saas` / `devtool` → Hero + Problem + Value Props + How it works + Social proof + Pricing teaser + FAQ + CTA
- `ecommerce` / `physical-product` → Hero produit + Benefices + Visuels + Details produit + Reviews + FAQ + Garanties + CTA panier
- `agency` / `portfolio` → Hero positionnement + Services/Projets + Process + Cas clients + Bio/Equipe + CTA contact
- `infoproduct` / `course` → Hero transformation + Pour qui / Pas pour qui + Programme + Temoignages + Instructeur + Pricing + Garantie + FAQ + CTA
- `marketplace` → Hero double-face (adapter selon visiteur buyer ou seller) + Offre + Preuves volume/qualite + Onboarding + CTA
- `media` → Hero mission + Ligne editoriale + Pieces phares + Newsletter signup
- `nonprofit` → Hero mission + Impact + Beneficiaires + Transparence + Appel don/action

```markdown
## [HERO]
Headline: [Tension en 8-12 mots — PAS une description produit]
Subheadline: [Resolution en 1 phrase — comment {name} resout]
CTA primaire: [Action specifique alignee avec conversion_goal — pas "Commencer"]
CTA secondaire: [Alternative basse friction]

## [PROBLEME]
[3-4 phrases qui decrivent le quotidien frustrant du persona SANS mentionner le produit]

## [VALUE PROPS] (3 blocs)
### [VP1: Feature/Benefice/Promesse]
[2 phrases : ce que ca fait → pourquoi ca change la donne]

### [VP2]
[2 phrases]

### [VP3]
[2 phrases]

## [SOCIAL PROOF]
[Temoignage, metrique, logos, case study — adapter au type de projet. Si pas disponible, indiquer {{PLACEHOLDER: preuve sociale a collecter}}]

## [OBJECTION HANDLER]
[Repondre a la principale objection du persona pour ce type de projet]

## [CTA FINAL]
[Reprise de la tension initiale + CTA aligne avec conversion_goal]
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
[Argumentation + exemples concrets + lien vers feature/service/projet si pertinent]

### [H2: ...]
[...]

## [CONCLUSION + CTA]
[Boucler sur la tension du hook + appel a l'action aligne avec conversion_goal]
```

### Type: `article`

Meme structure que `blog` mais :
- Ton plus editorial, moins commercial
- Pas de CTA agressif — CTA subtil en fin d'article
- Longueur : 1200-2000 mots
- Angle : thought leadership, point de vue d'auteur, pas promotion produit

### Type: `product-ui`

Microcopy pour l'interface. Adapter les composants listes au type de projet (app SaaS, checkout ecommerce, flow d'inscription a un cours, tableau de bord donateurs, onboarding d'une app mobile, etc.).

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
- Empty states = opportunite de conversion / retention, pas un cul-de-sac
- Tooltips : expliquer pourquoi cette feature existe, pas comment cliquer
- Erreurs : jamais "Une erreur est survenue", toujours dire QUOI et COMMENT fixer
- Succes : confirmer l'action + proposer l'etape suivante
- Labels : verbe d'action > nom abstrait ("Finaliser ma commande" > "Checkout")
- Si `ecommerce` → privilegier le rassurance copy (livraison, retour, garantie) sur CTAs panier
- Si `mobile-app` → penser contraintes de taille (labels courts, tooltips native-like)
- Si `nonprofit` / `media` → privilegier l'implication emotionnelle et la transparence

**Bilingue par defaut** : FR + EN pour tout projet qui a une portee internationale. Si projet clairement mono-langue (asso locale, agence francophone), generer uniquement la langue cible.

### Type: `changelog`

```markdown
## [Version X.Y.Z ou Date] — [Titre court benefice]

### [Headline: benefice, pas feature]

**Avant :** [la situation frustrante]
**Maintenant :** [ce qui change concretement]

#### Ce qui a change
- [changement 1] — [pourquoi ca compte]
- [changement 2] — [pourquoi ca compte]

#### Prochaine etape
[Ce sur quoi l'equipe travaille, lien vers roadmap si applicable]
```

Adapter le format au type de projet :
- `ecommerce` → "Nouveautes boutique" (nouvelle collection, restock, nouvelles options)
- `media` → "Ce qui a change dans la redaction" (nouvelle rubrique, nouveau format)
- `course` → "Mises a jour du programme" (nouveau module, bonus, Q&A live)

### Type: `feature-announcement`

Generer 3 assets :

```markdown
## 1. Blog Post (format blog ci-dessus)
[800-1200 mots]

## 2. Thread Social (5-7 posts)
Post 1: [Hook — le probleme]
Post 2: [Agitation — pourquoi c'est pire qu'on pense]
Post 3: [Ce qu'on a construit / lance]
Post 4: [Comment ca marche (concret)]
Post 5: [Resultat / metrique / exemple]
Post 6: [CTA]

## 3. Email Announcement
Subject: [< 50 chars, curiosite > description]
Preview text: [< 90 chars]
Body: [Before/After/Bridge, < 200 mots, 1 CTA aligne avec conversion_goal]
```

Choisir les plateformes sociales selon `project_profile.type` :
- `saas` / `devtool` → X/Twitter + LinkedIn
- `ecommerce` / `physical-product` → Instagram + TikTok
- `agency` / `portfolio` / `infoproduct` → LinkedIn + Twitter
- `nonprofit` / `media` → Instagram + LinkedIn + Facebook
- `mobile-app` → Instagram + TikTok + App Store / Play Store description

**GATE:** Le contenu genere est complet pour le type demande (tous les blocs du template sont remplis). Aucun bloc ne contient de placeholder non-justifie (sauf social proof si pas de donnees reelles).

---

## Phase 4: BULLSHIT FILTER

**Goal:** Scanner et eliminer tout copy generique. C'est le garde-rail le plus important.

**4a. Blacklist — Formules interdites (universelles, tout type de projet) :**

| Formule interdite | Pourquoi | Remplacement |
|-------------------|----------|--------------|
| "Revolutionnez votre [X]" | Vide, surutilise | Decrire le changement concret |
| "Boostez votre productivite" | Generique | Donner le gain mesurable |
| "Solution innovante" | Tout le monde le dit | Decrire le mecanisme specifique |
| "Alimentee par l'IA" / "Powered by AI" | Obvious | Nommer le modele ou la technique |
| "Dans un monde ou..." | Ouverture paresseuse | Commencer par le probleme |
| "Grace a notre technologie de pointe" | Fluff corporate | Expliquer ce que la techno fait |
| "Seamless / Transparent / Intuitive" | Adjectifs vides | Montrer avec un exemple |
| "Debloquez le potentiel de..." | Jargon marketing | Dire ce qui est concretement possible |
| "Game-changer" / "Disruptif" | Buzzwords | Prouver avec un fait |
| "Passez au niveau superieur" | Generique | Decrire le niveau en question |
| "Gain de temps considerable" | Vague | "Reduit de 4h a 20 minutes" |
| "Tout-en-un" / "All-in-one" | Sur-promesse | Lister les 3 choses que ca fait vraiment |
| "Simple et puissant" | Contradiction vague | Montrer la simplicite avec un exemple |
| "Faites-en plus avec moins" | Cliche | Decrire le "plus" et le "moins" |
| "Qualite premium / haut de gamme" | Ecom cliche | Nommer le materiau, le process, le fournisseur |
| "Nous sommes passionnes par..." | Agency/portfolio cliche | Montrer la passion par un choix fort |
| "Rejoignez une communaute..." | Vide | Quantifier + decrire la communaute |
| "Transformez votre vie" | Infoproduct cliche | Decrire la transformation concrete |

**4b. Tests de qualite :**

Pour CHAQUE paragraphe du draft, verifier :

- [ ] **Test du "So what?"** — Si on peut repondre "et alors?" a une phrase, elle est trop vague → la reformuler avec un fait specifique
- [ ] **Test du concurrent / pair** — Si on peut remplacer "{name}" par n'importe quel concurrent (ou n'importe quelle autre agence, boutique, asso du meme secteur) et que la phrase reste vraie, elle est generique → la rendre specifique au projet
- [ ] **Test du chiffre / exemple** — Chaque claim de performance, benefice ou promesse est-il etaye par un chiffre, un scenario, ou un exemple concret ? Si non → ajouter
- [ ] **Test de la tension** — Le contenu cree-t-il une tension (probleme, frustration, risque, aspiration) avant de resoudre ? Si non → restructurer

**4c. Rewrite :**

Pour chaque element qui echoue un test, reecrire en appliquant la regle. Ne pas juste supprimer — transformer en quelque chose de specifique.

**GATE:** Zero formule de la Blacklist presente. Tous les paragraphes passent les 4 tests. Si des elements echouent encore, les reecrire une deuxieme fois (max 2 iterations).

---

## Phase 5: OUTPUT

**Goal:** Livrer le contenu final structure.

**5a. Format de sortie :**

```markdown
## Contenu Final

**Projet:** {name}
**Type de projet:** {project_profile.type}
**Type de contenu:** {type}
**Langue:** {FR | EN}
**Angle:** {l'angle defini en Phase 2}
**Persona cible:** {persona principal}
**Conversion goal:** {project_profile.conversion_goal}

---

{contenu structure selon le template du type}

---

## Notes Editoriales
- **Variante de headline testable :** {alternative headline pour A/B test}
- **Mots-cles longue traine :** {si blog/article — 3-5 keywords}
- **Liens internes suggeres :** {pages du site a lier}
- **Social proof a collecter :** {temoignages ou metriques manquants}
- **Plateformes de diffusion recommandees :** {selon project_profile.type}
```

**5b. Si le type est `feature-announcement` :** livrer les 3 assets (blog + thread + email) dans un seul output.

**5c. Si le type est `product-ui` :** livrer le tableau FR + EN (ou mono-langue si projet local) avec toutes les entrees.

## Hard Rules

1. **Phase 0 ALWAYS runs** — classifier avant de generer quoi que ce soit
2. **Phase 1 Project Type Detection + Product Discovery sont obligatoires** — ne jamais generer de contenu sans identifier le type de projet ET le produit
3. **Phase 2 (ANGLE) est obligatoire** — ne jamais generer de contenu sans angle valide
4. **Phase 4 (BULLSHIT FILTER) est obligatoire** — chaque draft est filtre, sans exception
5. **Max 2 iterations de rewrite** en Phase 4 — si le copy est encore generique apres 2 passes, livrer avec un warning
6. **Respecter le template du type** — tous les blocs doivent etre presents
7. **Bilingue FR + EN par defaut pour `product-ui`** sauf si le projet est clairement mono-langue
8. **Citations sourced** — tout chiffre ou claim doit avoir une source ou etre marque `{{a verifier}}`
9. **Context-aware** — si un codebase / des assets existent, generer du copy compatible
10. **Project-specific** — chaque phrase doit etre specifique au projet decouvert en Phase 1, jamais generique
11. **Type-adapted CTA** — le CTA principal doit coller au `conversion_goal` du type de projet (purchase pour ecom, book-call pour agence, donate pour asso, etc.)
12. **Print `[Phase N/5]` progress headers** avant chaque phase — ne jamais sauter l'indicateur

## DO NOT

1. Utiliser une formule de la Blacklist Phase 4 — meme reformulee, meme "ironiquement"
2. Ouvrir un blog par "Dans un monde ou..." ou "A l'ere de l'IA..." — interdit sans exception
3. Generer du copy sans avoir defini l'angle en Phase 2 — la sequence est non-negociable
4. Ecrire "les utilisateurs" / "les clients" quand tu peux ecrire un persona specifique — toujours nommer le role
5. Promettre des features / services / produits qui n'existent pas — si pas dans le codebase ou le contexte, ne pas inventer
6. Utiliser des superlatifs sans preuve ("le meilleur", "le plus rapide", "le seul") sauf si demontrable
7. Appliquer un template de landing-page SaaS a un projet ecommerce (ou inversement) — adapter au `project_type`
8. Mettre un CTA qui ne correspond pas au `conversion_goal` du projet
9. Faire du name-dropping de concurrents dans le copy public — comparer en interne, pas en output client
10. Generer plus de 2000 mots pour un blog sans demande explicite du user
11. Hardcoder des details projet sans les avoir valides dans le codebase ou avec le user

## Error Handling

| Scenario | Fallback |
|----------|----------|
| **Type de projet non detectable** | Demander au user de choisir parmi la liste (saas / ecommerce / mobile-app / agency / portfolio / infoproduct / marketplace / media / physical-product / devtool / course / nonprofit / generic). |
| **Produit non identifiable** | Demander au user de decrire le projet en 2-3 phrases (nom, ce que c'est, pour qui, comment ca monetise). |
| **agent-websearch echoue** | Baser l'angle sur le contexte interne uniquement. Marquer "Competitive context unavailable" dans les notes editoriales. |
| **agent-explore echoue** | Travailler sans contexte codebase. Generer du copy sans references aux composants existants. |
| **Type de contenu non reconnu** | Demander au user de choisir parmi les 6 types valides. |
| **Contexte insuffisant** | Demander au user de preciser le sujet, la feature, ou l'audience ciblee. |
| **Langue ambigue** | Demander au user. Defaut : francais. |
| **Pas de codebase** | Travailler uniquement a partir du contexte fourni par le user. |
| **Bullshit Filter echoue apres 2 iterations** | Livrer avec un warning listant les phrases encore generiques + suggestion de reformulation manuelle. |

## References

- [Narrative Frameworks](references/narrative-frameworks.md) — story arcs, PAS, BAB, contrarian, show don't tell, mission-driven, transformation arc
- [Landing Page Anatomy](references/landing-page-anatomy.md) — section-by-section high-converting patterns, variantes par type de projet
- [Voice & Tone](references/voice-tone.md) — calibration du ton par persona, format, type de projet, registre FR vs EN
- [Conversion Patterns](references/conversion-patterns.md) — CTA, objection handling, social proof, pricing / donation / booking psychology

## Done When

- [ ] Phase 0 (Classify) completed — type de contenu et contexte identifies
- [ ] Phase 1 (Research) completed — project_profile construit avec type de projet detecte
- [ ] Phase 2 (Angle) completed — angle passes 4 validation criteria
- [ ] Phase 3 (Draft) completed — all template blocks filled, adaptes au project_type
- [ ] Phase 4 (Bullshit Filter) passed — zero blacklisted formulas, all 4 tests pass
- [ ] Phase 5 (Output) delivered with editorial notes and variantes

## Constraints (Three-Tier)

### ALWAYS
- Run Phase 0 (Classify) before generating content
- Detect project type in Phase 1 before writing
- Build project profile with type + monetization + conversion_goal
- Define and validate an angle (Phase 2) before drafting
- Run Bullshit Filter (Phase 4) on every draft — no exceptions
- Adapt CTA and structure to the detected project type
- Generate FR + EN for `product-ui` type (unless project is clearly mono-langue)

### ASK FIRST
- Project type when not detectable from codebase
- Product identity when insufficient context
- Content type when `$ARGUMENTS` doesn't match valid types
- Language when ambiguous

### NEVER
- Use blacklisted formulas from Phase 4 — even reformulated or "ironic"
- Open a blog with "Dans un monde ou..." or "A l'ere de l'IA..."
- Apply a SaaS landing template to an ecommerce / agency / nonprofit project (or inversely)
- Promise features / services / products not in the codebase or user context
- Use superlatifs without proof
- Output a CTA misaligned with the project's conversion_goal
