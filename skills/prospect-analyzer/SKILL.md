---
model: opus
name: prospect-analyzer
description: Pipeline d'analyse de prospect et génération de messages LinkedIn sur-mesure. Extrait les données des URLs (LinkedIn profil, LinkedIn entreprise, site web), analyse le prospect contre l'ICP StriveX, identifie les parallèles et pain points, génère un message de connexion LinkedIn personnalisé et une séquence multi-touch. Trigger sur "analyse ce prospect", "prospect-analyzer", "analyse prospect", "prépare un message pour", "prospection pour", ou quand des URLs LinkedIn/site web de prospect sont fournies.
argument-hint: "[linkedin-url] [company-url?] [website-url?]"
tags: [prospection, linkedin, outreach, analyse, b2b, strivex, personnalisation]
version: 1.0.0
---

# Prospect Analyzer — Pipeline Claude Code

> Extraire. Analyser. Scorer. Rédiger. Valider. Un seul workflow, un message qui convertit.

## Architecture du pipeline

```
Phase 1 — EXTRACTION (Parallèle)          Phase 2 — ANALYSE (Séquentiel)
┌──────────────────────┐                   ┌──────────────────────┐
│ Agent A: LinkedIn     │──┐               │                      │
│ profil prospect       │  │               │  Synthèse des données│
├──────────────────────┤  ├──────────────▶│  Scoring ICP (100pts)│
│ Agent B: LinkedIn     │  │               │  Pain points          │
│ page entreprise       │  │               │  Parallèles StriveX  │
├──────────────────────┤  │               │  Signaux d'intent     │
│ Agent C: Site web     │──┘               │                      │
│ + recherche signaux   │                  └──────────┬───────────┘
└──────────────────────┘                              │
                                                      ▼
                                          ┌──────────────────────┐
                                          │   GATE — Score < 60  │
                                          │                      │
                                          │  Cold / Discard ?    │
                                          │  → STOP pipeline     │
                                          │  → Verdict rapide    │
                                          │  → Pas de fichiers   │
                                          │  → "Passe au suivant"│
                                          └──────────┬───────────┘
                                                     │ Score ≥ 60
                                                     ▼
Phase 3 — RÉDACTION (Séquentiel)          Phase 4 — VALIDATION (Gate)
┌──────────────────────┐                   ┌──────────────────────┐
│                      │                   │                      │
│  Message connexion   │                   │  Check vouvoiement   │
│  Message valeur J+3  │──────────────────▶│  Check < 300 chars   │
│  Message suivi J+14  │                   │  Check pas de pitch  │
│  Message sortie J+21 │                   │  Check CTA unique    │
│                      │                   │  Check ton StriveX   │
└──────────────────────┘                   └──────────────────────┘
```

---

## Déclenchement

Ce workflow se déclenche quand l'utilisateur fournit **au moins 1 URL** parmi :
- URL de profil LinkedIn du prospect
- URL de la page LinkedIn de l'entreprise du prospect
- URL du site web de l'entreprise du prospect

Format d'invocation typique :
```
Analyse ce prospect :
- LinkedIn : https://linkedin.com/in/...
- Entreprise LinkedIn : https://linkedin.com/company/...
- Site : https://example.com
```

---

## Phase 1 — EXTRACTION (Agents en parallèle)

**Objectif :** Extraire un maximum de données structurées de chaque URL fournie.

**Exécution :** Lancer **simultanément** autant d'agents `agent-websearch` que d'URLs fournies (2-3 en parallèle).

### Agent A — Profil LinkedIn du prospect

Prompt pour l'agent :
```
Recherche et extrais toutes les informations disponibles sur ce profil LinkedIn : {URL_LINKEDIN_PROFIL}

Données à extraire (JSON structuré) :
- nom_complet
- headline (titre LinkedIn exact)
- localisation
- nombre_connexions (approximatif)
- nombre_followers
- poste_actuel : { titre, entreprise, durée, description }
- postes_precedents : [{ titre, entreprise, durée }]
- formation : [{ école, diplôme, année }]
- competences_affichees : []
- certifications : []
- a_propos (section About complète)
- activite_recente : { posts_recents (sujets/thèmes), commentaires_recents, frequence_publication }
- centres_interet_visibles : []
- recommandations_recues : nombre + extraits si visibles
- langues : []
- projets_mentionnes : []
- url_site_perso (si indiqué)

Cherche aussi sur le web :
- Articles/interviews mentionnant cette personne
- Posts LinkedIn récents indexés par Google
- Présence sur d'autres plateformes (Twitter/X, GitHub, Product Hunt)

Retourne TOUTES les données trouvées, même partielles. Indique [NON TROUVÉ] pour les champs introuvables.
```

### Agent B — Page LinkedIn de l'entreprise

Prompt pour l'agent :
```
Recherche et extrais toutes les informations disponibles sur cette page LinkedIn d'entreprise : {URL_LINKEDIN_ENTREPRISE}

Données à extraire (JSON structuré) :
- nom_entreprise
- secteur
- taille (tranche d'effectifs)
- siege_social
- site_web
- date_creation
- description_complete
- specialites : []
- type_entreprise (startup, PME, ETI, grand groupe)
- nombre_employes_linkedin
- croissance_employes (tendance)
- postes_ouverts : [{ titre, lieu, date }]  ← CRITIQUE pour les signaux d'intent
- publications_recentes : [{ sujet, engagement }]
- technologies_mentionnees : []
- clients_mentionnes : []
- levees_de_fonds : [{ montant, date, source }]
- dirigeants_visibles : [{ nom, poste }]

Cherche aussi sur le web :
- Articles presse récents (Maddyness, French Tech, TechCrunch FR, Les Echos, BPI)
- Avis Glassdoor/Indeed (tendances, pas les détails)
- Présence Product Hunt
- Repos GitHub publics
- Stack technique (BuiltWith, Wappalyzer, ou indices visibles)

Retourne TOUTES les données trouvées, même partielles.
```

### Agent C — Site web de l'entreprise + Signaux

Prompt pour l'agent :
```
Analyse le site web de cette entreprise : {URL_SITE_WEB}

Extrais :
1. CONTENU :
   - Proposition de valeur principale (headline, hero)
   - Services/produits proposés
   - Pricing affiché (si visible)
   - Clients/références affichés
   - Équipe affichée
   - Blog (derniers articles, sujets)
   - Pages clés (about, contact, careers)

2. TECHNIQUE (observable de l'extérieur) :
   - Stack détectable (headers HTTP, meta tags, scripts visibles)
   - CMS utilisé (WordPress, Webflow, custom, etc.)
   - Performance perçue (rapide/lent au chargement)
   - HTTPS actif ou non
   - Responsive ou non
   - Dernière mise à jour visible (copyright, dates blog)
   - SEO basique (meta title, meta description, og:tags)

3. SIGNAUX D'OPPORTUNITÉ :
   - Le site est-il obsolète ? (design daté, pas responsive, technologies anciennes)
   - Y a-t-il des erreurs visibles ? (liens cassés, pages 404, images manquantes)
   - Le site utilise-t-il l'IA ? (chatbot, recommandations, personnalisation)
   - Y a-t-il un produit SaaS ? Si oui, quel stade ? (landing page, MVP, produit mature)
   - Le site mentionne-t-il un besoin de recrutement tech ?

Retourne une analyse structurée complète.
```

### Checkpoint Phase 1

Après réception des résultats des agents, **NE PAS sauvegarder de fichier à ce stade**. Les données brutes sont conservées en mémoire pour le scoring Phase 2. Les fichiers ne seront créés que si le prospect est qualifié (score ≥ 60).

**Gate Phase 1 :** Si aucune donnée exploitable n'a été extraite (toutes les URLs ont échoué), informer l'utilisateur et proposer des alternatives (recherche manuelle, URLs différentes). Ne pas passer à la Phase 2 avec des données vides.

---

## Phase 2 — ANALYSE ET SCORING

**Objectif :** Transformer les données brutes en intelligence actionnable pour la rédaction.

**Exécution :** Séquentiel — nécessite les outputs complets de Phase 1.

### Étape 2.1 — Scoring ICP (100 points)

Appliquer la grille de scoring définie dans le CLAUDE.md du projet :

| Critère | Points | Sous-critères |
|---------|--------|---------------|
| **Fit ICP** | 40 | Titre du prospect (15), taille entreprise (10), secteur (10), localisation France (5) |
| **Qualité enrichissement** | 25 | Email trouvé/vérifié (15), téléphone (5), LinkedIn actif (5) |
| **Signaux d'intent** | 25 | Levée récente <90j (10), recrutement tech actif (5), stack compatible/obsolète (5), engagement LinkedIn (5) |
| **Timing** | 10 | Levée <90j (5), post récent <7j (3), actif sur la plateforme (2) |

**Scoring :**
- **Hot (≥ 80)** → Outreach immédiat, message hautement personnalisé
- **Warm (60-79)** → Séquence standard, personnalisation bonne
- **Cold (40-59)** → Veille, pas d'outreach immédiat, noter pour plus tard
- **Discard (< 40)** → Ne pas contacter, ne correspond pas à l'ICP

**Gate Phase 2.1 (STOP/GO) :**

- **Score < 60 (Cold ou Discard) → STOP le pipeline immédiatement.**
  - NE PAS passer aux Phases 3 et 4 (pas de rédaction de messages)
  - NE PAS créer de fichier diagnostic
  - NE PAS ajouter le prospect aux fichiers de leads
  - Afficher uniquement le **verdict rapide** (voir format ci-dessous) et recommander de passer au prospect suivant
  - Si l'utilisateur insiste explicitement, reprendre le pipeline à partir de la Phase 2.2

- **Score ≥ 60 (Warm ou Hot) → continuer le pipeline** normalement vers Phase 2.2, puis Phase 3, Phase 4, sauvegarde des fichiers.

**Format du verdict rapide (score < 60) :**
```markdown
# Verdict — {Nom du prospect}

**Score ICP : {X}/100 — {Cold/Discard}. Pas un bon prospect pour StriveX.**

## Pourquoi ce prospect ne matche pas
- {Raison 1 — la plus importante}
- {Raison 2}
- {Raison 3}

## Données clés
- **Prospect :** {Nom} — {Titre} chez {Entreprise}
- **Entreprise :** {description courte, taille, stade}
- **Localisation :** {lieu}

**Recommandation : passe au suivant.** Ce prospect ne justifie pas le temps d'une séquence outreach.
```

### Étape 2.2 — Identification des parallèles StriveX ↔ Prospect

Chercher systématiquement les points de connexion entre StriveX et le prospect :

```
PARALLÈLES DIRECTS :
- Le prospect est-il dans un secteur où StriveX a déjà livré ? (comparer au portfolio)
- Le prospect utilise-t-il une stack que StriveX maîtrise ? (Next.js, React, Rust, Python)
- Le prospect a-t-il un besoin visible que StriveX résout ? (MVP, site, IA, audit)
- Le prospect a-t-il un problème technique observable ? (site lent, obsolète, pas d'IA)

PARALLÈLES INDIRECTS :
- Points communs géographiques (Nantes, Pays de la Loire, France)
- Points communs de réseau (connexions mutuelles, groupes LinkedIn partagés)
- Points communs de centres d'intérêt (IA, SaaS, Product Hunt, open-source)
- Parcours similaires (fondateur solo, transition tech, vibecoding)

CAS CLIENT SIMILAIRE :
- Quel projet du portfolio StriveX ressemble le plus à ce que le prospect fait/cherche ?
- Quel résultat concret de ce projet peut être cité ? (conversion, délai, technologie)
```

### Étape 2.3 — Identification des pain points

Identifier 3-5 pain points potentiels classés par probabilité :

```
Pour chaque pain point :
- Description : {le problème identifié}
- Preuve : {d'où vient cette hypothèse — donnée concrète extraite}
- Gravité : haute / moyenne / basse
- StriveX peut résoudre : oui / partiellement / non
- Angle de message : {comment formuler ça dans un message}
```

### Étape 2.4 — Choix de la stratégie de message

Basé sur le scoring et les pain points, choisir l'approche :

| Contexte | Stratégie |
|----------|-----------|
| Signal d'achat fort (levée, recrutement, post de besoin) | **Rebond sur signal** — référencer directement le signal |
| Problème technique visible (site obsolète, pas d'IA) | **Mini-diagnostic gratuit** — observer et partager une insight |
| Parallèle fort avec un projet du portfolio | **Cas client similaire** — référencer le résultat obtenu |
| Engagement LinkedIn du prospect (posts, commentaires) | **Rebond sur contenu** — commenter un de ses posts puis connecter |
| Aucun signal clair mais bon fit ICP | **Question ouverte** — poser une question liée à son domaine |

---

## Phase 3 — RÉDACTION DES MESSAGES

**Objectif :** Générer une séquence complète de messages LinkedIn personnalisés.

**Exécution :** Séquentiel — nécessite l'output complet de Phase 2.

### Règles absolues de rédaction (HARD RULES)

Ces règles sont NON NÉGOCIABLES. Chaque message DOIT les respecter :

1. **Vouvoiement systématique** — jamais de tutoiement
2. **Français** — sauf si le prospect est clairement anglophone
3. **Pas de pitch au premier message** — apporter de la valeur d'abord
4. **CTA unique** — une seule action demandée par message
5. **Ton : direct, technique, professionnel mais pas corporate** — pas de "je me permets", pas de "bonjour cher", pas de jargon marketing creux
6. **90/10** — 90% du message parle du prospect, 10% de contexte sur StriveX
7. **Court** — lisible sur mobile, pas de pavés de texte
8. **Spécifique** — chaque phrase doit contenir un élément que seul ce prospect peut reconnaître
9. **Ne jamais mentionner "CTO externalisé"** — Arthur est un builder de produits tech
10. **Pas de formules d'excuse** — pas de "désolé de vous déranger", pas de "je me permets de vous contacter"

### Message 1 — Note de demande de connexion (J1)

**Contraintes :**
- **Max 300 caractères** (limite LinkedIn)
- 1-2 phrases seulement
- Doit contenir un trigger spécifique (signal d'achat, observation, post récent)
- Pas de pitch, pas de lien
- Doit donner envie d'accepter la connexion

**Structure :**
```
[Accroche spécifique basée sur le trigger identifié en Phase 2] + [Question ouverte ou proposition de valeur en 1 phrase]
```

**Anti-patterns à éviter :**
- "J'aimerais vous ajouter à mon réseau professionnel"
- "Je me permets de vous contacter car..."
- "J'ai vu votre profil et..."
- Toute mention de services ou prix
- Tout lien externe

### Message 2 — Message de valeur post-connexion (J3)

**Contraintes :**
- **Max 500 caractères**
- Envoyé 2-3 jours après acceptation de la connexion
- Doit apporter de la valeur nouvelle (insight, observation, mini-diagnostic)
- Pas encore de pitch — on établit la crédibilité
- Si stratégie "mini-diagnostic" : partager 1-2 observations concrètes sur leur produit/site

**Structure :**
```
[Remerciement bref pour la connexion — 1 phrase max]
[Observation/insight spécifique sur leur entreprise/produit — 2-3 phrases]
[Question ouverte qui invite au dialogue — 1 phrase]
```

### Message 3 — Message de suivi avec cas client (J14)

**Contraintes :**
- **Max 600 caractères**
- Envoyé uniquement si pas de réponse au Message 2
- Doit apporter un angle NOUVEAU (pas de répétition)
- Citer un cas client StriveX similaire avec un résultat concret
- Première introduction possible du lien cal.com si le ton le permet

**Structure :**
```
[Référence à un élément nouveau — post récent du prospect, actualité du secteur]
[Cas client similaire avec résultat chiffré — 1-2 phrases]
[CTA soft : question ou proposition de call — 1 phrase]
```

### Message 4 — Message de sortie propre (J21)

**Contraintes :**
- **Max 300 caractères**
- Message final — porte ouverte, pas de pression
- Professionnel et respectueux
- Laisse la possibilité de revenir plus tard

**Structure :**
```
[Acknowledge que le timing n'est peut-être pas le bon — 1 phrase]
[Porte ouverte pour le futur — 1 phrase]
[Souhait sincère de réussite — optionnel, 1 phrase]
```

---

## Phase 4 — VALIDATION

**Objectif :** Vérifier que chaque message respecte les règles avant de le présenter.

**Exécution :** Séquentiel — auto-évaluation immédiate après la rédaction.

### Checklist de validation (pour chaque message)

```
□ Vouvoiement respecté (aucun "tu", "ton", "ta", "tes", "toi")
□ Longueur respectée (Message 1 ≤ 300 chars, Message 2 ≤ 500, Message 3 ≤ 600, Message 4 ≤ 300)
□ Pas de pitch dans le Message 1
□ Pas de "je me permets", "désolé de vous déranger", "cher/chère"
□ Pas de mention "CTO externalisé" ou "CTO as a Service"
□ CTA unique par message (pas deux demandes)
□ Au moins 1 élément spécifique au prospect par message (nom, entreprise, produit, post, signal)
□ Ton direct et technique, pas corporate
□ Aucun emoji (sauf si Arthur en utilise habituellement)
□ Français correct, pas de fautes
□ Aucun lien dans le Message 1
□ Lien cal.com uniquement dans Message 3 (si approprié)
□ Chaque message apporte un angle/une valeur nouvelle vs le précédent
```

**Si un message échoue la validation :** le réécrire immédiatement en corrigeant les points défaillants. Max 2 itérations de correction.

---

## Output final

**IMPORTANT :** Cet output complet n'est généré QUE pour les prospects avec un score ≥ 60 (Warm ou Hot). Pour les prospects Cold (< 60) ou Discard (< 40), le pipeline s'est arrêté à la Gate Phase 2.1 avec un verdict rapide — aucun fichier n'est créé, aucun message n'est rédigé.

Présenter le résultat complet à l'utilisateur dans ce format :

```markdown
# Analyse Prospect — {Nom du prospect}

## Résumé express
- **Prospect :** {Nom} — {Titre} chez {Entreprise}
- **Score ICP :** {X}/100 ({Hot/Warm/Cold/Discard})
- **Stratégie retenue :** {nom de la stratégie}
- **Meilleur parallèle StriveX :** {projet portfolio similaire}

## Données clés extraites
{Top 10 des données les plus pertinentes pour la prospection}

## Pain points identifiés
{Liste des 3-5 pain points avec preuves}

## Parallèles StriveX ↔ {Entreprise}
{Liste des connexions identifiées}

---

## Séquence de messages LinkedIn

### J1 — Demande de connexion
> {Message 1}

**Trigger :** {quel signal déclenche ce message}
**Caractères :** {count}/300

### J3 — Message de valeur
> {Message 2}

**Angle :** {quelle valeur est apportée}
**Caractères :** {count}/500

### J14 — Suivi avec cas client
> {Message 3}

**Cas cité :** {quel projet StriveX}
**Caractères :** {count}/600

### J21 — Message de sortie
> {Message 4}

**Caractères :** {count}/300

---

## Recommandations
- {Conseil spécifique pour ce prospect — par ex. "commenter d'abord son post du {date} avant d'envoyer la connexion"}
- {Action préparatoire si nécessaire}
- {Timing recommandé pour le premier contact}
```

### Sauvegarde (uniquement si score ≥ 60)

**IMPORTANT :** Ne créer des fichiers QUE pour les prospects qualifiés (Warm ≥ 60 ou Hot ≥ 80). Pour les Cold/Discard, aucun fichier n'est créé — le verdict rapide suffit.

Sauvegarder l'analyse complète dans :
```
diagnostics/{nom_prospect_slug}.md
```

Et ajouter le prospect au fichier de leads approprié selon son score :
- **Hot (≥ 80)** → `leads/hot.json`
- **Warm (60-79)** → `leads/warm.json`

```json
{
  "nom": "",
  "entreprise": "",
  "titre": "",
  "linkedin_url": "",
  "site_url": "",
  "score_icp": 0,
  "categorie": "hot|warm",
  "date_analyse": "",
  "date_premier_contact": null,
  "stage": "identifié",
  "pain_points": [],
  "strategie": "",
  "notes": ""
}
```

---

## Rappel — Qui est StriveX (contexte pour la personnalisation)

**Arthur Jean** — Fondateur solo, builder de produits tech (SaaS, Software, GenAI, Software AI). Basé à Nantes. Vibecoder : ex-dev full-stack 5 ans, ne code plus mais comprend profondément stacks, architectures, langages et infras. Fait aussi du conseil marketing (funnels, growth, Product Hunt, PostHog, UX/UI).

**Ce que StriveX fait :** Construit des produits tech durables — MVPs SaaS (8-10k€, 30j), sites vitrines (dès 990€), intégration IA (RAG, multi-agents), audit & rescue. Positionnement anti-agence : intérêts alignés, 100% propriété du code, pas de dette technique.

**Portfolio de référence :** Azuna (conciergerie), Au Sommet de Chez Vous (artisan), Dress Night (e-commerce), OpenbookLM (SaaS IA, open-source 91+ stars), Delgres Céramique (e-commerce 3D), Quantum Oracle (IA + QRNG).

**Lien booking :** https://cal.com/arthurjean/30min

## Done When

- [ ] Phase 1 (Extraction) completed — data extracted from all provided URLs
- [ ] Phase 2 (Scoring) completed — ICP score calculated on 100 points
- [ ] Gate Phase 2.1 evaluated — Cold/Discard stops pipeline with verdict rapide
- [ ] For Warm/Hot: pain points identified and parallèles StriveX mapped
- [ ] For Warm/Hot: 4 messages LinkedIn rédigés (J1/J3/J14/J21)
- [ ] Phase 4 (Validation) passed — all messages respect character limits and rules
- [ ] Files saved: diagnostic and lead entry (only for score ≥ 60)

## Constraints (Three-Tier)

### ALWAYS
- Spawn extraction agents in parallel (Phase 1)
- Score ICP before drafting messages — Gate Phase 2.1 is mandatory
- Validate every message against the checklist (Phase 4)
- Use vouvoiement in all messages

### ASK FIRST
- Resume pipeline for Cold/Discard prospects (only if user insists)
- Proceed when no exploitable data extracted from any URL

### NEVER
- Draft messages for prospects scoring < 60 (Cold/Discard)
- Mention "CTO externalisé" or "CTO as a Service"
- Include a pitch in Message 1 (connexion)
- Use tutoiement in any message
- Create files for unqualified prospects (score < 60)
