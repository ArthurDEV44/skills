# Conversion Patterns — CTA, Objection Handling, Social Proof

## CTA Design Principles

### La Hierarchie CTA

Chaque page a maximum 2 types de CTA :

| Type | Role | Exemple |
|------|------|---------|
| **Primaire** | L'action principale qu'on veut | "Essayer avec vos documents" |
| **Secondaire** | Alternative basse friction | "Voir une demo", "Lire la doc" |

### Regles CTA

1. **Verbe d'action specifique** — jamais "Commencer", "Decouvrir", "En savoir plus"
2. **Le CTA repond a "Qu'est-ce qui se passe quand je clique ?"** — si la reponse n'est pas claire, reformuler
3. **Coherence** — le meme CTA primaire partout sur la page (hero = footer = sidebar)
4. **Contraste visuel** — le CTA primaire est le seul element de sa couleur
5. **Texte sous le CTA** — reduire la friction ("Pas de carte bancaire", "2 minutes", "Annulable a tout moment")

### CTA par type de friction

| Friction user | CTA primaire | Texte sous CTA |
|---------------|-------------|----------------|
| "Ca va me prendre du temps" | "[Verbe] mon premier [objet]" | "Pret en 2 minutes, pas de config" |
| "Mes donnees sont sensibles" | "[Verbe] un [objet] test" | "Vos donnees restent privees et chiffrees" |
| "C'est cher" | "[Action] — gratuit pour commencer" | "Pas de carte bancaire requise" |
| Pas de friction identifiee | "Essayer {product_name}" | "[Benefice principal en 5 mots]" |

---

## Objection Handling

### Framework AIDA pour les objections

Pour chaque objection majeure :

1. **Acknowledge** — Valider la preoccupation (pas la balayer)
2. **Inform** — Donner l'info factuelle
3. **Demonstrate** — Montrer la preuve (screenshot, chiffre, temoignage)
4. **Act** — Proposer l'etape suivante

### Matrice d'objections type GenAI SaaS

#### "Mes donnees sont sensibles"

| Etape | Template |
|-------|---------|
| Acknowledge | "Vos [type de donnees] sont confidentiels. Normal de s'en soucier." |
| Inform | "[Politique de traitement des donnees — stockage, retention, chiffrement]." |
| Demonstrate | "[Certifications, compliance, architecture]. {{a verifier: SOC2/RGPD/etc.}}" |
| Act | "Testez avec un [objet] non-sensible d'abord." |

#### "Pourquoi pas juste [concurrent gratuit] ?"

| Etape | Template |
|-------|---------|
| Acknowledge | "[Concurrent] est excellent pour [son use case principal]." |
| Inform | "Mais il n'est pas concu pour [le use case specifique de {product_name}]. [Explication du differenciateur technique]." |
| Demonstrate | "Test : faites [action] avec [concurrent], puis avec {product_name}. [Critere de comparaison objectif]." |
| Act | "Essayez les deux sur le meme [objet] — la difference est immediate." |

#### "C'est cher / j'ai deja un abo IA"

| Etape | Template |
|-------|---------|
| Acknowledge | "Les abonnements IA s'empilent vite." |
| Inform | "[Explication du modele de pricing et pourquoi il est avantageux — usage-based, BYOK, pas de markup, etc.]." |
| Demonstrate | "{{a verifier: cout moyen par session / par mois / comparaison avec alternatives}}" |
| Act | "[CTA vers calculateur de cout ou page pricing]." |

#### "C'est complique a mettre en place ?"

| Etape | Template |
|-------|---------|
| Acknowledge | "Personne n'a envie de passer une heure en setup." |
| Inform | "[Nombre d'etapes pour commencer]. [Time-to-first-value]." |
| Demonstrate | "[Screenshot ou GIF du onboarding]. {{a verifier: temps reel mesure}}" |
| Act | "[CTA basse friction — essayer directement]." |

---

## Social Proof Patterns

### Pattern 1: Le Temoignage Narratif

Le plus efficace. Structure Before/After avec le persona.

```
"[Situation avant — specifique et relatable]
[Decouverte / moment de bascule]
[Resultat concret — chiffre si possible]"

— [Prenom Nom], [Titre], [Organisation]
```

**Exemple :**
```
"Je passais un vendredi sur deux a reverifier mes citations pour les articles du cabinet.
Depuis qu'on utilise {product_name}, je fais la meme verification en 30 minutes.
Mon vendredi, je le passe sur les dossiers clients."

— Thomas M., Avocat associe, Cabinet X
```

### Pattern 2: La Metrique d'Usage

Quand on n'a pas de temoignages nommes.

```
[Chiffre] [unite] [depuis quand]
```

**Exemples :**
- "47,000 documents analyses ce mois"
- "Utilise par 1,200+ chercheurs dans 40 pays"
- "99.7% de citations verifiables"

**Regles :**
- Le chiffre doit etre reel — jamais inventer
- Si pas de chiffre reel disponible → `{{PLACEHOLDER: metrique a mesurer}}`
- Arrondir a la baisse, pas a la hausse (cree plus de credibilite)

### Pattern 3: Logos

Pour le B2B et l'academique.

```
Utilise par des chercheurs de :
[Logo Universite 1] [Logo Universite 2] [Logo Entreprise 1] [Logo Cabinet 1]
```

**Regles :**
- Minimum 4 logos pour que ce soit credible
- Pas de logos sans autorisation — `{{PLACEHOLDER: logos a obtenir}}`
- Les logos academiques sont plus credibles que les logos entreprise pour un outil de recherche

### Pattern 4: User Generated

Screenshots de tweets, avis Product Hunt, discussions Reddit/HN.

```
Screenshot: tweet de @user
"[citation du tweet]"
[date] — [nombre de likes/retweets]
```

**Regles :**
- Toujours demander l'autorisation avant de reproduire
- Preferer les avis qui mentionnent un use case specifique
- Eviter les avis type "Amazing tool!!!" (pas informatif)

---

## Pricing Psychology (si applicable)

### Principes

1. **Ancrage** — Montrer le cout SANS {product_name} d'abord (ex: 4h de travail = 200€ de temps)
2. **Comparaison** — "Moins cher que [alternative la plus connue] pour [use case specifique]"
3. **Transparence** — Montrer la formule de calcul, pas juste le prix final
4. **Pas de dark patterns** — pas de prix barre fictif, pas de countdown, pas de "3 places restantes"

### Framework de presentation prix usage-based

```
## Vous ne payez que ce que vous utilisez

| Ce que vous faites | Unite | Cout approximatif |
|-------------------|-------|-------------------|
| [Action legere] | [unite] | [prix bas — montrer l'accessibilite] |
| [Usage moyen] | [unite] | [prix moyen — ancrer contre l'alternative] |
| [Usage intensif (1 journee)] | [unite] | [prix — toujours moins cher que l'alternative] |

[Phrase de cloture : pas d'abonnement / prix transparent / etc.]
```

---

## Anti-Patterns de Conversion

### Ce qui tue la conversion

| Anti-pattern | Pourquoi ca echoue | Alternative |
|-------------|-------------------|-------------|
| Formulaire a 8 champs | Friction excessive | Email seul au signup |
| "Demander une demo" comme seul CTA | Trop d'engagement | Self-serve + demo optionnelle |
| Pop-up immediat | Irritant, hausse le bounce | CTA inline contextuel |
| Countdown timer | Sent le fake, detruit la confiance | Pas d'urgence artificielle |
| "Limited offer" | Mensonge visible | Ne pas utiliser |
| Slider de temoignages auto | Personne ne lit | Temoignage statique unique, fort |
| Video auto-play | Hostile | Video click-to-play |
| "Trusted by 10,000+ users" sans preuve | Claim vide | Montrer les vrais logos/metriques |
