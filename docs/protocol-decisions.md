# Décisions de protocole (2026-10-13)

## 1. Famille de transformations T
T n'est pas un réglage de robustesse : elle définit l'invariance imposée (ce que T change = nuisible, ce qu'elle préserve = identité). Une banque est un objet écrit et versionné (`configs/transforms/*.yml`, module `cibirl/transforms.py`).

- `generic` : crop modéré + jitter de luminosité/contraste/saturation + niveaux de gris rares. Appliquée telle quelle à tous les datasets (tableau principal). Ne touche jamais à la teinte.
- `natural` : `generic` + flip horizontal. Images naturelles seulement (un chiffre ou une forme du jouet miroir est une autre identité).
- `informed_color` : `generic` + redraw complet de la teinte. Information a priori DÉCLARÉE sur le nuisible, présentée comme condition expérimentale.
- `toy_oracle` : redraw exact du fond du jouet (borne haute du a priori).

Résultat central sur Colored MNIST (et le jouet avec raccourci) : présenter les DEUX régimes `generic` et `informed_color`. L'écart mesure à quel point le résultat dépend d'un a priori sur le nuisible.

Règle d'admission : une opération entre dans une banque seulement si la connaissance du domaine, écrite AVANT de voir les résultats, dit qu'elle préserve l'identité. Jamais choisie ni réglée en regardant les labels. Les forces sont fixées à l'avance ; leur effet est étudié par un balayage de sensibilité, pas en gardant la meilleure valeur.

## 2. Règle d'arrêt (sans labels)
Entraînement jusqu'à convergence (`cibirl/convergence.py`) ou `max_epochs`. Convergence = sur les `window` (10) dernières epochs : partition argmax stable (changement <= 1 %, au renommage des clusters près), nombre de clusters constant, plateau de twin_acc (+-0,01) et de L_pos+L_neg (2 % relatif), après `min_epochs` (40). Le modèle évalué est celui de la dernière epoch. Les labels ne servent qu'à surveiller (`eval_*` dans metrics.jsonl).

Limite : "convergé" = stable, pas "bon". À lire avec `clusters_used`, `m_std`.

## 3. Constats du jouet (suites 100 epochs, 5 graines ; diagnostics sur 1 à 2 graines)
Résultats finaux à 100 epochs (dernière epoch, sans sélection) : `toy` ACC 0,72 +- 0,16 ; `toy_spur0.9` 0,64 +- 0,08 ; `toy_no_standardize` 0,39 +- 0,02. Ce sont des états APRÈS dégradation : à 20-40 epochs plusieurs runs avaient 5 clusters purs (ACC 1,00).

Faits établis :
- La représentation m est bonne : la moyenne de chaque classe est le centre le plus proche pour 100 % des images dans 9 runs sur 10 (hors ablation). Le défaut n'est pas l'encodeur.
- La politique reproduit l'argmax de la récompense seule (ACC politique = ACC récompense dans presque tous les runs) : les fusions viennent de la récompense / de la mémoire, pas de PPO.
- Deux problèmes distincts :
  1. Dégradation de la récompense par la compression : avec beta = 1e-3, KL tombe à ~40 et twin_acc passe de ~0,83 à ~0,5, l'entropie de la politique monte à ~1,1 (quasi aléatoire), churn ~0,6. Association nette (KL >= 80 : twin 0,79-0,82 ; KL < 60 : 0,50-0,54), confirmée par contrôle : avec beta = 1e-4 ou 0, twin_acc reste ~0,8 (graines 1 et 3).
  2. Instabilité de l'affectation, indépendante de beta : fusion brutale de deux classes (ex. graine 0 oracle : 5 clusters, ACC 1,00 de l'epoch 18 à 55, puis 4 clusters, ACC 0,80, sans signe dans twin_acc / KL), ou contamination des clusters (beta = 0 : 5 clusters mais ACC 0,71-0,77). Avec beta plus petit l'ACC finale n'est pas meilleure.
- Avec une mémoire idéale (moyennes de classes), la récompense n'ouvre presque jamais un nouveau slot dès que 2 slots sont occupés (décisions justes ~0,52-0,60 à 2 slots occupés) : elle "rejoint le plus proche". Réserve : les résumés idéaux diffèrent peut-être de ceux vus à l'entraînement (moyenne de 2 vues).
- Reproductibilité inter-plateformes (Windows / Linux) : bonne pour les graines 0 et 1, partielle pour la graine 3 (dynamique chaotique : de minuscules écarts numériques changent la trajectoire).

Il n'existe pas encore de critère sans labels qui détecte le problème 2 (twin_acc et KL ne bougent pas pendant une fusion). Le choix de la meilleure epoch sur les labels est interdit ; on rapporte la dernière epoch ET la trajectoire (`eval_*`).

La règle d'arrêt (fenêtre 5 -> 10, mesure invariante au renommage) a été ajustée en regardant des runs de la graine 0 : choix à signaler, à tester en sensibilité (fenêtre 5 / 10 / 20).

## 5. Prochaines étapes (décidées le 2026-10-13 soir)
- Ne pas lancer `toy_regimes` tel quel : il mesurerait une dynamique dégradée (beta = 1e-3).
- Problème 1 : réduire beta (1e-4 testé) ou viser un KL cible ; signaler dans les runs une baisse de twin_acc (indicateur sans labels).
- Problème 2 : voir 5.1 (trois variantes testées, aucune ne le corrige). Prochain candidat : l'écart entre les résumés vus par la récompense pendant l'entraînement des experts et ceux de l'apprenant.

### 5.1 Suite `toy_dynamics` (beta = 1e-4, jouet sans raccourci, banque oracle, 100 epochs fixes, 5 graines)
Référence contre trois variantes (`configs/suites/toy_dynamics.yml`). ACC en fin de run, moyenne +- écart-type sur les graines ; "ACC max" est la meilleure epoch (labels, information seulement).

| condition | ACC finale | ACC moy. 20 dernières | ACC max | clusters finaux | twin_acc | entropie politique |
|---|---|---|---|---|---|---|
| ref | 0,74 +- 0,17 | 0,77 | 0,92 | 4,2 | 0,81 | 0,49 |
| consolidate (mémoire = argmax en fin d'epoch) | 0,60 +- 0,00 | 0,64 | 0,84 | 3,0 | 0,81 | 0,11 |
| cold (T = 0,5, alpha = 0) | 0,74 +- 0,08 | 0,80 | 0,96 | 4,0 | 0,81 | 0,21 |
| orbits5 (B = 5) | 0,72 +- 0,20 | 0,76 | 0,92 | 3,6 | 0,65 | 0,72 |

Lecture :
- Aucune variante n'améliore l'ACC finale de façon distinguable de la référence (n = 5, écart-type 0,1 a 0,2). `cold` réduit la dispersion (0,08 contre 0,17) et gagne un peu en moyenne sur les 20 dernières epochs ; à confirmer sur plus de graines.
- `consolidate` est pire : 3 clusters sur les 5 graines, ACC 0,60 partout. Sans tirage aléatoire, les fusions ne sont jamais défaites.
- L'hypothèse "le bruit d'exploration contamine la mémoire" n'est pas soutenue : l'entropie de la politique est déjà ~0,1 dès l'epoch ~12 dans toutes les conditions, donc les actions échantillonnées sont presque l'argmax pendant que les fusions se produisent. La fusion vient de la dynamique déterministe récompense + mémoire.
- `orbits5` rend la tâche experte plus dure (twin_acc 0,65 au lieu de 0,81) sans gain final, avec à peine moins de chutes brutales d'ACC (8,0 contre 9,8 en moyenne, dans le bruit).
- Constat de structure (3 orbites, 2 vues) : dans les états experts, aucun slot n'a plus de 2 membres, donc la récompense n'est entraînée que sur des résumés = moyenne de 1 ou 2 vues, alors que l'apprenant lui présente des centres de classes moyennés sur des centaines d'images. Non testé à ce stade.

## 4. Données de suivi
`results/` est versionné dans git (pour documenter les résultats). Suites : `configs/suites/*.yml`, lancées par `scripts/run_suite.py`.
