# Tone Patterns — Extracted from 6700+ Arthur Messages

## Source
Analyse de `~/.claude/history.jsonl` (6702 prompts) + conversations completes dans `~/.claude/projects/` (1152 sessions, 2.9GB) + feedback explicite d'Arthur sur le skill linkedin-comment.

## Marqueurs lexicaux frequents (par ordre de frequence)

### Connecteurs / transitions
- "Du coup" — transition principale, remplace "donc" ou "alors"
- "En revanche" — opposition mesuree
- "Par contre" — opposition directe
- "Ok bon" — transition apres reflexion
- "Parfait !" — validation, souvent en debut de message
- "En fait" / "Enfaite" — correction/precision
- "C'est pas mal" — approbation moderee
- "Je trouve que" — opinion personnelle

### Demarrage de phrase
- "J'aimerai que tu..." — pattern de requete le plus frequent (note : il ecrit "j'aimerai" sans le s)
- "Utilise X pour Y" — delegation d'outil
- "Effectue des recherches sur..." — demande de recherche
- "Modifie / Supprime / Ajoute" — imperatif direct
- "Fix cette erreur" / "Fix ce probleme" — mix FR/EN naturel
- "Je viens de tester" — retour d'experience
- "Comme tu peux voir" — reference a une capture/screenshot
- "Ok maintenant" — enchainement de taches

### Expressions recurrentes
- "code propre et de qualite" — son standard qualite
- "sans bruit ni dette technique" — ce qu'il refuse dans le code
- "meilleures pratiques a tous les niveaux" — son exigence
- "je ne suis pas fan de" — rejet poli mais ferme
- "ca manque de" — feedback visuel/design
- "c'est quoi les standard et tendances ?" — curiosite sur les best practices
- "ce n'est pas ce que je veux" — recadrage clair
- "il faut que" — imperatif
- "plutot" — preference sans forcer ("plutot avec des couleurs cyan")
- "voir capture" / "voir ci-jointe" — reference visuelle

### Validation / accord
- "Oui" + action suivante ("Oui commit et push")
- "Parfait !" — souvent suivi d'une nouvelle instruction
- "C'est bien" — validation + mais ("c'est bien, mais rajoute un peu de personnalite")
- "Ok c'est fait" — confirmation d'action effectuee
- "Je valide" — approbation formelle

### Rejet / feedback negatif
- "Trop AI Slop" — rejet direct (anti-IA detectable)
- "je ne suis pas fan de" — rejet soft
- "ca fait bizarre" — feedback visuel
- "c'est trop [adjectif]" — trop sombre, trop long, trop corporate
- "Non je ne veux pas" — refus clair, pas d'ambiguite
- "ce n'est pas adapte" — rejet fonctionnel

## Patterns syntaxiques

### Ponctuation
- Virgules abondantes (souvent la ou un point serait plus correct)
- Points d'exclamation rares sauf "Parfait !"
- Pas de points de suspension sauf hesitation genuine ("je ne sais pas si c'est de ta faute mais...")
- Pas de guillemets francais (« ») — utilise les guillemets droits si besoin

### Structure de message typique
1. Reaction courte (validation ou rejet) — "Ok", "Parfait", "Non"
2. Contexte/observation — "je trouve que ca manque de..."
3. Instruction precise — "Utilise /X pour Y"
4. Condition ou precision — "mais ne modifie pas la font"

### Typos frequentes (ne pas corriger, ca fait partie du ton)
- "j'aimerai" au lieu de "j'aimerais" (systematique)
- "fonctionné" au lieu de "fonctionner"
- "que tu es" au lieu de "que tu aies"
- "qu'il n'y est pas" au lieu de "qu'il n'y ait pas"
- "effectué" au lieu de "effectuer"
- "recomandés" au lieu de "recommandés"
- "emraude" au lieu de "émeraude"
- Espaces manquants entre mots parfois

## Registres selon contexte

### Avec Claude Code (conversationnel pur)
- Tutoiement
- Imperatif direct sans formule
- Pas de "s'il te plait" / "merci"
- Fragments de phrase
- Mix FR/EN constant

### LinkedIn (public, un cran au-dessus)
- Tutoiement
- Phrases completes mais courtes
- Un seul point developpé (pas de thread-like)
- Angle builder, pas analyste
- Questions genuines en fin de message

### Email / prospection (le plus formel)
- Vouvoiement possible
- "Je me permets de" — rare mais utilisé en prospection
- Toujours une proposition concrète (appel 15 min)
- Personnalisation obligatoire (mention du projet du prospect)

## Exemples bruts (verbatim)

> "Ok maintenant dans la sidebar nous avons une cards actuellement 'Plan Gratuit' mais je veux plutot quelque chose de promotionel plus marketing mettre en avant des choses je trouve ca plus interessant."

> "C'est bien, mais rajoute un peu de personnalite, c'est trop corporate et change Fractional CTO par un autre terme car cette appelation fait en sorte que des solo founders et startups me contacte pour etre leur CTO, or ce n'est pas ce que je veux"

> "Trop AI Slop et trop long, le fait que je n'utilise plus opencode et uniquement claude code c'est que claude code est plus ergonomique, bien optimise"

> "Ok bon tout a l'air de fonctionne en revanche je trouve ca tres contraignant la redirection forcee lorsque je suis connecte vers le dashboard"

> "Super ! Par contre je trouve que l'icon key comme ca n'est pas tres adapte pour une api key c'est quoi les standard et tendances ?"

> "Du coup je fais quoi je mets un pouce leve, je laisse coule ou je commente avec ceci"
