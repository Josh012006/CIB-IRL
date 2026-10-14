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
- Problème 1 : DÉCIDÉ le 2026-10-13 : `beta = 1e-4` devient le défaut de `TrainConfig` (cibirl/train.py). Les runs `results/toy_*` faits avec beta = 1e-3 sont l'état précédent. À surveiller dans les runs : une baisse de twin_acc (indicateur sans labels).
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

### 5.2 Suite `toy_oracle_link` (mêmes conditions que la référence ; beta = 1e-4, 100 epochs ; 3 graines, `rho` sur 5)
Deux questions : (a) l'absence de signal d'attraction entre orbites d'une même classe chez l'expert explique-t-elle les fusions ? (b) le terme anti-effondrement `rho` (désactivé par défaut) aide-t-il ?
- `expert_oracle_link` : DIAGNOSTIC ORACLE qui utilise les vraies classes (jamais une méthode ; `run.yml` porte `uses_labels_for_training: true`). Dans un épisode, avec probabilité p, les orbites d'une même classe partagent un slot (attraction entre orbites, pas de répulsion dans la classe). C'est ce que produirait un relinker de phase 2 parfait avec p = 1. Il mêle deux effets : l'attraction entre orbites et des résumés experts plus gros (jusqu'a 4 vues).
- `rho` : -rho x entropie du marginal de la récompense sur les slots, mesuré sur les états experts.

Graines 0 à 2 :

| condition | ACC finale | ACC moy. 20 dernieres | epochs avec ACC >= 0,95 | clusters |
|---|---|---|---|---|
| ref | 0,80 +- 0,16 | 0,83 | 31 | 4,0 |
| link 1,0 | 0,74 +- 0,04 | 0,68 | 0 | 4,7 |
| link 0,5 | 0,70 +- 0,08 | 0,79 | 35 | 4,0 |
| rho 0,1 | 0,93 +- 0,09 | 0,94 | 67 | 4,7 |
| rho 1,0 | 0,84 +- 0,12 | 0,94 | 77 | 4,7 |

`rho` sur les 5 graines (ref : 0,74 +- 0,17 / 0,77 / 24 epochs / 4,2 clusters) : rho 0,1 -> 0,84 +- 0,13 / 0,81 / 40 / 4,8 ; rho 1,0 -> 0,82 +- 0,10 / 0,85 / 57 / 4,8.

Lecture :
- Des experts parfaits (link 1,0) ne corrigent pas les fusions : ACC 0,71 a 0,80, jamais 0,95, alors que twin_acc = 1,0 (la récompense prédit parfaitement l'expert). Le signal inter-orbites manquant n'est donc pas la cause principale.
- La représentation m n'est pas en cause : le centre de classe le plus proche donne 100 % de bonnes affectations dans toutes les conditions (graines 0 a 2).
- Test de la mémoire idéale (slots = vrais centres de classe) : avec les 5 slots occupés, récompense et politique placent 100 % des images dans leur classe ; avec 1 a 4 slots occupés, 50 a 95 % seulement (décider d'ouvrir un slot ou de rejoindre un slot existant sous une mémoire partielle). link 1,0 améliore la récompense seule avec 1 slot occupé (0,92 contre 0,52 a 0,76), pas la politique.
- `rho` est le seul levier qui bouge quelque chose, de façon modeste sur 5 graines (ACC finale +0,08 a +0,10, un peu plus d'epochs a >= 0,95, plus proche de 5 clusters), avec des écarts de l'ordre de l'écart-type : indice, pas preuve. Les graines 3 et 4 avec rho 1,0 atteignent 1,0 puis redescendent vers 0,6 vers l'epoch 100 : la dérive tardive reste.

### 5.3 Suite `toy_formation` (mêmes conditions que la référence ; 5 graines ; diagnostics de placement)
Hypothèse testée : « au début la mémoire est peu informative, l'incertitude contamine les clusters et la contamination persiste ».
Nouveautés : `cibirl/diagnostics.py` (statistiques `eval_place`, `eval_purity`, `eval_depleted` par epoch, avec les labels, évaluation seulement ; sans effet sur l'entraînement : 0 écart sur 48 valeurs et mêmes ACC/NMI que l'ancienne référence), `oracle_warm_memory` (DIAGNOSTIC ORACLE : 5 images par classe dans la mémoire au moment où l'apprenant commence à agir), `burn_in_epochs` / `burn_in_fraction` (sans labels : la mémoire est reconstruite à partir d'un tirage de 10 % des membres de chaque cluster, compté en epochs depuis le début des placements).

Test 1, sur la référence :
- La mémoire est pleine presque tout le temps : sur 37 500 placements des epochs 10 à 14 (l'apprenant ne place rien avant l'epoch 10, à cause du préchauffage de 150 itérations de la récompense), 63 seulement ont lieu avec moins de 5 slots occupés. L'hypothèse « mémoire partielle » est donc écartée (elle était une extrapolation du test à mémoire idéale, qui ne décrit pas les états réellement visités).
- Aucun slot ne se vide pendant une run (0 sur 20). Les erreurs de placement (21 à 37 % des placements) viennent d'une mémoire pleine ; elles reflètent surtout les classes fusionnées.

Interventions (ACC finale, moyenne +- écart-type ; référence 0,74 +- 0,17) : warm 0,80 +- 0,13 ; burn3 0,78 +- 0,15 ; burn8 0,69 +- 0,20. Aucune ne se distingue de la référence.
Réserve sur warm : la politique non entraînée écrase la mémoire de départ en ~25 placements (pureté 0,6 à 0,8 à l'epoch 12 malgré un départ propre).

Ce qui contredit la contamination précoce comme cause principale : 8 runs sur 20 atteignent un état pur (ACC >= 0,95 pendant au moins 10 epochs consécutives) et 4 le perdent ensuite. Graine 4 : pureté ~1,0, politique déterministe, ACC 1,0 de l'epoch 13 à 24, puis rupture brutale (churn de 0,04 à 0,41, un cluster sur cinq se défait) sans signe dans twin_acc ni dans les pertes de la récompense. Elle survient aux epochs 24 à 29 dans quatre conditions différentes (référence, warm, burn3, burn8), tandis que la graine 2 reste pure jusqu'au bout dans trois d'entre elles. L'équilibre pur est donc instable pour certaines graines, indépendamment de l'historique de la mémoire.
Prochaine étape proposée : autopsie de la graine 4 (sauvegarder m et les marges de la politique aux epochs 22 à 28).

### 5.4 Suite `toy_rho` (10 graines) : correction de 5.2
Même cadre que la référence (beta = 1e-4, 100 epochs). Résultats sur 10 graines (ACC finale, moyenne +- écart-type ; différence appariée avec la référence +- erreur standard) :

| condition | ACC finale | ACC moy. 20 dernières | epochs >= 0,95 | clusters | différence finale vs ref |
|---|---|---|---|---|---|
| ref | 0,70 +- 0,16 | 0,73 | 24 | 4,2 | |
| rho 0,1 | 0,77 +- 0,15 | 0,78 | 26 | 4,5 | +0,07 +- 0,06 |
| rho 0,3 | 0,71 +- 0,10 | 0,74 | 14 | 4,0 | +0,01 +- 0,07 |

Le gain de `rho` vu sur 3 à 5 graines (5.2) ne se confirme pas : +0,07 est à environ 1,2 erreur standard pour rho 0,1, et rho 0,3 ne change rien (moins d'epochs à >= 0,95). Conclusion : `rho` n'est pas un levier contre les fusions ; il reste désactivé par défaut. Avec 10 graines la référence vaut 0,70 +- 0,16 (0,74 sur les graines 0 à 4), ce qui donne l'ordre de grandeur du bruit entre jeux de graines.

### 5.5 Autopsie de la graine 4 de la référence (pure aux epochs 13-24, casse à l'epoch 25)
Rejeu déterministe (1 thread, identique à la suite), mesures sur les epochs 18-30. Les labels ne servent qu'à l'évaluation.

* **Ce qui ne bascule pas** : la représentation (précision du plus proche centroïde de classe dans m = 1,00 à tous les epochs) ; la politique et la reward sous une mémoire idéale (moyennes vraies des classes) restent justes à 100 %, même après la casse.
* **Ce qui bascule** : le contenu de la mémoire. À l'epoch 25 deux classes se retrouvent dans le même slot de la mémoire échantillonnée (slots dominants [3,1,4,2,0] -> [3,3,2,1,0]) ; les résumés deviennent des mélanges, la marge de la politique tombe de 4,3 à 0,3-0,9, et ensuite une classe entière de 300 images change de prédiction d'un epoch à l'autre.
* **Mesure de péremption** (avant chaque `recompute`) : distance moyenne entre l'intention stockée à la pose et l'intention actuelle, rapportée à la plus petite distance entre moyennes de classes.

| | graine 4 (casse) | graine 2 (stable) |
|---|---|---|
| dérive absolue des intentions stockées | 0,10-0,15 | 0,10-0,16 |
| plus petite distance entre classes (dans m) | 0,33 (ep 16) -> 0,14 (ep 25) -> 0,06 (ep 27+) | 0,60 -> 1,2-1,3 (croît) |
| dérive / distance | 0,2-2,1 (> 1 à partir de l'epoch 27) | 0,05-0,23 |

Lecture : la dérive entre epochs est du même ordre dans les deux graines ; ce qui diffère est la distance minimale entre classes, qui s'effondre dans la graine 4 (médiane des distances de paires 0,80 -> 0,32, alors que la plus grande reste ~10 : une classe très éloignée écrase les autres dans un coin de l'espace standardisé). Quand la dérive dépasse la séparation, des résumés construits en cours d'epoch ne désignent plus la bonne classe. Limites : une seule graine instable et un seul contrôle ; la dérive dépasse déjà la séparation aux epochs 19-21 de la graine 4 sans casse, donc ce n'est pas une condition suffisante ; la cause de l'effondrement de la séparation n'est pas établie ; la géométrie de la graine 2 (max/médiane des paires) n'a pas été mesurée.
Pistes à tester ensuite : (a) recalcul des résumés plus souvent qu'une fois par epoch ; (b) geler ou ralentir l'encodeur après la formation ; (c) comprendre pourquoi la séparation minimale décroît (rôle de la standardisation BN avec une classe très éloignée, et de `L_neg` seul sur les mouvements libres).

## 4. Données de suivi
`results/` est versionné dans git (pour documenter les résultats). Suites : `configs/suites/*.yml`, lancées par `scripts/run_suite.py`.
