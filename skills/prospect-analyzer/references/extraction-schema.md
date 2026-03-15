# Schéma d'extraction — Données prospect

## Structure de données attendue post-Phase 1

```typescript
interface ProspectData {
  // Métadonnées
  date_analyse: string; // ISO 8601
  urls_analysees: {
    linkedin_profil?: string;
    linkedin_entreprise?: string;
    site_web?: string;
  };

  // Profil LinkedIn
  profil: {
    nom_complet: string;
    headline: string;
    localisation: string;
    nombre_connexions: string; // "500+" ou chiffre
    nombre_followers: number;
    poste_actuel: {
      titre: string;
      entreprise: string;
      duree: string;
      description: string;
    };
    postes_precedents: Array<{
      titre: string;
      entreprise: string;
      duree: string;
    }>;
    formation: Array<{
      ecole: string;
      diplome: string;
      annee: string;
    }>;
    competences: string[];
    a_propos: string;
    activite_recente: {
      themes_posts: string[];
      frequence: string; // "quotidien", "hebdomadaire", "mensuel", "rare"
      dernier_post_date: string;
    };
    langues: string[];
    url_site_perso: string | null;
    presence_autres_plateformes: {
      twitter?: string;
      github?: string;
      product_hunt?: string;
      autre?: string[];
    };
  };

  // Entreprise LinkedIn
  entreprise: {
    nom: string;
    secteur: string;
    taille: string; // "1-10", "11-50", "51-200", etc.
    siege: string;
    site_web: string;
    date_creation: string;
    description: string;
    specialites: string[];
    type: "startup" | "pme" | "eti" | "grand_groupe" | "association" | "autre";
    croissance_employes: "forte_croissance" | "croissance" | "stable" | "décroissance";
    postes_ouverts: Array<{
      titre: string;
      lieu: string;
      date_publication: string;
    }>;
    technologies_mentionnees: string[];
    levees_de_fonds: Array<{
      montant: string;
      date: string;
      source: string;
    }>;
    dirigeants: Array<{
      nom: string;
      poste: string;
    }>;
    presse_recente: Array<{
      titre: string;
      source: string;
      date: string;
      url: string;
    }>;
  };

  // Site web
  site_web: {
    proposition_valeur: string;
    services_produits: string[];
    pricing_visible: string | null;
    clients_references: string[];
    equipe_visible: string[];
    blog: {
      actif: boolean;
      derniers_sujets: string[];
      derniere_date: string;
    };
    technique: {
      stack_detectee: string[];
      cms: string | null;
      https: boolean;
      responsive: boolean;
      performance: "rapide" | "moyen" | "lent";
      derniere_maj: string;
      seo_basique: {
        meta_title: string;
        meta_description: string;
        og_tags: boolean;
      };
    };
    signaux_opportunite: {
      site_obsolete: boolean;
      erreurs_visibles: boolean;
      utilise_ia: boolean;
      produit_saas: {
        present: boolean;
        stade: "landing" | "mvp" | "mature" | null;
      };
      recrutement_tech_mentionne: boolean;
    };
  };
}
```

## Champs prioritaires pour le scoring

Ces champs ont le plus d'impact sur le score ICP et la stratégie de message :

1. `profil.poste_actuel.titre` — détermine le fit ICP (fondateur, DG, CMO, etc.)
2. `entreprise.taille` — filtre PME/startup
3. `entreprise.postes_ouverts` — signal d'intent critique (recrutement tech = besoin)
4. `entreprise.levees_de_fonds` — signal d'intent et timing
5. `site_web.signaux_opportunite` — pain points techniques observables
6. `site_web.technique.stack_detectee` — parallèle technique avec StriveX
7. `profil.activite_recente` — engagement LinkedIn et timing
8. `entreprise.secteur` — fit ICP sectoriel

## Champs prioritaires pour la personnalisation

Ces champs alimentent directement la rédaction des messages :

1. `profil.a_propos` — comprendre ce qui motive le prospect
2. `profil.activite_recente.themes_posts` — sujets pour le rebond
3. `entreprise.description` — comprendre le business
4. `site_web.proposition_valeur` — ce qu'ils disent d'eux-mêmes
5. `profil.poste_actuel.description` — comment le prospect décrit son rôle
6. `site_web.signaux_opportunite` — matière pour le mini-diagnostic
7. `entreprise.presse_recente` — actualités pour le rebond
