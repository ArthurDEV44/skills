# Landing Page Anatomy — High-Converting GenAI SaaS

## La Structure qui Convertit

Une landing page n'est pas une brochure. C'est un argumentaire structure ou chaque section
a un role precis dans la decision de l'utilisateur. L'ordre n'est pas arbitraire — il suit
la psychologie de la decision.

## Section 1: HERO (above the fold)

**Role:** Capter l'attention en 3 secondes. L'utilisateur decide ici s'il scrolle ou part.

### Composants

| Element | Regle | Anti-pattern |
|---------|-------|--------------|
| **Headline** | 8-12 mots. Tension, pas description. | "La plateforme de [X] IA" |
| **Subheadline** | 1 phrase. Resolution de la tension. | Repeter le headline en plus long |
| **CTA primaire** | Verbe d'action specifique. Couleur contrastee. | "En savoir plus" / "Decouvrir" |
| **CTA secondaire** | Alternative basse friction (demo, video, docs). | Pas de CTA secondaire du tout |
| **Visual** | Screenshot reel ou animation produit. Pas de stock photo. | Illustration abstraite IA |

### Patterns de headlines par angle

**Par tension :**
- "[Chiffre] [objets]. [Une action]. [Le resultat specifique]."
- "[Ce que le persona fait manuellement]. [En combien de temps avec le produit]."
- "[Affirmation surprenante qui contredit l'idee recue du domaine]."

**Par benefice :**
- "[Benefice concret] sans [la contrepartie habituelle]."
- "[Action du persona], [resultat qu'il obtient]."

**Par douleur :**
- "[La chose frustrante que le persona fait]. [Pas avec {product_name}]."
- "Arretez de [tache penible]. [Ce que le produit fait a la place]."

### CTA — Regles universelles

- ✅ "[Verbe] + [objet specifique au produit]" (ex: "Analyser mon premier document")
- ✅ "[Action a basse friction]" (ex: "Voir une demo en 2 min")
- ❌ "Commencer" / "Get Started" (trop vague)
- ❌ "Demander une demo" (trop B2B enterprise pour du self-serve)
- ❌ "En savoir plus" (ne dit pas ce qu'on va apprendre)

---

## Section 2: PROBLEME (le mirror)

**Role:** L'utilisateur doit se reconnaitre. "C'est exactement mon probleme."

### Regles
- Decrire le quotidien frustrant SANS mentionner le produit
- Utiliser le vocabulaire du persona (pas du jargon tech)
- Etre specifique : des scenarios, pas des abstractions
- Max 4-5 phrases

### Template

```
[Scenario concret du quotidien — action specifique du persona]
[Ce que le persona fait actuellement pour resoudre — l'outil ou le process]
[Pourquoi ca ne marche pas — le cout cache, la perte de temps, le risque]
[La consequence : temps perdu, argent perdu, frustration, risque professionnel]
```

---

## Section 3: VALUE PROPOSITIONS (les 3 pilliers)

**Role:** Montrer comment le produit resout chaque aspect du probleme.

### Regles
- Exactement 3 blocs (pas 4, pas 5 — la regle de 3)
- Chaque bloc : Feature → Benefice → Preuve
- Feature = ce que ca fait (1 ligne)
- Benefice = pourquoi ca compte pour le persona (1 ligne)
- Preuve = chiffre, comparaison, ou scenario (1 ligne)

### Template

```
### [Benefice en 4-6 mots]
[Feature concrete en 1 phrase]
[Pourquoi ca change la donne en 1 phrase]
[Chiffre ou comparaison : "X au lieu de Y"]
```

### Comment choisir les 3 VP

Prioriser par force de differenciateur :
1. **VP1** = le differenciateur le plus fort (ce que les concurrents ne font PAS)
2. **VP2** = le differenciateur technique (le COMMENT, le mecanisme)
3. **VP3** = le differenciateur d'experience (la facilite, le prix, l'accessibilite)

---

## Section 4: SOCIAL PROOF

**Role:** Transferer la confiance d'un tiers vers le produit.

### Hierarchie d'efficacite

1. **Temoignage nomme avec photo** — le plus efficace
2. **Metrique d'usage** — "X documents analyses ce mois"
3. **Logos d'entreprises/universites** — si clients notables
4. **Note/avis** — Product Hunt, G2, etc.
5. **Open source stars** — si applicable

### Regles
- Un temoignage specifique > dix temoignages generiques
- Le temoignage doit mentionner le PROBLEME resolu, pas juste "c'est genial"
- Si pas de social proof disponible, utiliser `{{PLACEHOLDER: temoignage a collecter — persona: [X], scenario: [Y]}}` plutot qu'inventer

### Template temoignage

```
"[Quote qui mentionne le probleme avant + le resultat apres — avec un chiffre si possible]"
— [Prenom Nom], [Role], [Entreprise/Universite]
```

---

## Section 5: OBJECTION HANDLER

**Role:** Repondre a la raison principale pour laquelle l'utilisateur hesite.

### Objections generiques SaaS GenAI

| Objection | Reponse type | Format |
|-----------|-------------|--------|
| "Mes donnees sont sensibles" | Politique de securite, chiffrement, compliance | FAQ ou badge securite |
| "C'est cher / j'ai deja un abo IA" | Comparaison de cout reel, ROI | Calculateur ou tableau |
| "C'est encore un wrapper [LLM]?" | Architecture differente, valeur ajoutee | Diagramme ou comparaison |
| "C'est complique a mettre en place ?" | Time-to-value : "pret en X minutes" | Mini tuto inline |
| "Pourquoi pas juste [concurrent] ?" | Differenciateur specifique | Tableau Before/After |

### Framework AIDA pour objections

1. **Acknowledge** — Valider la preoccupation
2. **Inform** — Donner l'info factuelle
3. **Demonstrate** — Montrer la preuve
4. **Act** — Proposer l'etape suivante

Consult `references/conversion-patterns.md` for detailed AIDA templates.

---

## Section 6: CTA FINAL

**Role:** Convertir l'utilisateur qui a lu toute la page.

### Regles
- Reprendre la tension du hero (boucler la boucle)
- CTA identique au hero (coherence)
- Ajouter un element de friction-reduite (pas de countdown fake)
- Optionnel : rappeler que c'est gratuit/sans engagement si applicable

### Template

```
## [Rappel de la tension en 1 phrase]
[1 phrase de resolution]
[CTA bouton — meme texte que le hero]
[Texte sous le bouton : "Pas de carte bancaire requise" ou equivalent]
```
