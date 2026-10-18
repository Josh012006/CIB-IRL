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

Option `stop.memory: true` (5.25) : exige en plus que la partition de la mémoire soit celle de la sortie (`mem_gap` <= 0,01 sur la fenêtre) ; sinon `mem_gap` est seulement loggé.
Option `final_cleanup: true` (5.28) : après l'entraînement, politique figée, fusionne les slots doublons de la mémoire (plus grand saut dans les distances entre résumés, garde `cleanup_ratio` 10) et relit la sortie ; protocole final : arrêt à la convergence de la sortie + ce dernier pas. `cleanup_mode` (5.29) choisit comment : `merge` (5.28), `relabel` (A : sortie lue avant la fusion, slots renommés), `reintroduce` / `reintroduce_no_open` (C : balayages des images, argmax figé) ; les quatre sont enregistrés dans chaque run.

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

### 5.6 Les trois analyses de la casse : géométrie de m sur 10 graines, recalcul fréquent, encodeur gelé
Même cadre que la référence (beta = 1e-4, 100 epochs ; les résultats de `ref` sont ceux de `toy_formation`, reproduits à l'identique). Les labels ne servent qu'à l'évaluation.

**(c) Géométrie de m sur les 10 graines de `toy_rho/ref` (rejeu de 50 epochs).**
* La séparation minimale entre classes ne prédit pas la casse : la graine 9 a une séparation de 0,03 à 0,26 entre les epochs 20 et 49 (souvent plus petite que la dérive des intentions stockées, 0,15 à 0,26) et reste pure 37 epochs sur 40 ; la graine 2 a une séparation de 1,2 et reste pure.
* Ce qui s'associe à l'instabilité (graine 4 : casse ; graine 3 : jamais pure, ACC qui oscille) est l'effondrement de la distance MÉDIANE entre moyennes de classes : 0,57 (graine 3) et 0,80 (graine 4) à l'epoch 14, contre 3,7 à 7,0 pour les 8 autres graines ; une classe reste à distance ~10 pendant que les quatre autres se tassent. Association sur 2 graines, pas une preuve.
* BN n'est pas la cause directe : la compression est déjà présente avant la standardisation (distance médiane brute 1,0 à l'epoch 14 pour la graine 4, contre 8,7 pour la graine 9). Hypothèse non testée : `L_psi` (cohérence d'orbite) n'a aucun terme de répulsion entre classes, et la variance unitaire par dimension imposée par BN est satisfaite par une seule classe éloignée ; `L_neg` agit sur la reward et ne passe pas par m (l'intent n'est pas entraîné par la reward).
* Les graines 0, 1, 5, 7, 8 (ACC 0,5 à 0,7, jamais pures) n'ont pas cet effondrement (distance médiane >= 3,7 à l'epoch 14) : leur échec est un autre mode, la fusion à la formation.

**(a) et (b) : `toy_stale`, 5 graines.** Options ajoutées (désactivées par défaut, chemin par défaut inchangé bit à bit) : `recompute_every` (recalcul des résumés toutes les n itérations), `freeze_repr_after` + `repr_lr_scale` (4 epochs après les premiers placements, l'encodeur et l'intent predictor sont gelés, poids et statistiques BN, ou ralentis).

| condition | ACC finale par graine (0,1,2,3,4) | moyenne | diff. appariée vs ref | epochs >= 0,9 par graine |
|---|---|---|---|---|
| ref | 0,60 0,80 1,00 0,52 0,80 | 0,745 | | 0, 5, 89, 0, 28 |
| rec5 | 0,80 1,00 1,00 0,53 0,51 | 0,766 | +0,02 +- 0,09 | 0, 11, 89, 0, 0 |
| rec1 | 0,75 0,80 0,60 0,60 1,00 | 0,749 | +0,01 +- 0,11 | 0, 0, 0, 0, 88 |
| freeze4 | 0,70 0,72 1,00 0,73 1,00 | 0,832 | +0,09 +- 0,06 | 0, 0, 89, 0, 88 |
| slow4 (x0,1) | 0,70 0,70 1,00 0,71 1,00 | 0,824 | +0,08 +- 0,06 | 0, 0, 89, 0, 79 |

* Graine 4 (celle qui cassait) : `freeze4` donne ACC 1,00 avec 88 epochs purs et aucune casse (réf. : 28 epochs purs, 11 casses). Le rejeu est identique à la référence jusqu'à l'epoch 13 (L_psi, L_neg, KL, m_std), donc l'amélioration est causée par le gel seul. `slow4` : 79 epochs purs, 2 casses. `rec1` : 88 epochs purs, mais `rec1` dégrade la graine 2, pure en référence (ACC 0,60, aucun epoch pur), parce que le recalcul à chaque itération modifie la formation ; `rec5` n'aide pas la graine 4 (0 epoch pur). Les deux options de recalcul ont donc un effet net nul.
* Graine 2 (contrôle) : gel et ralentissement ne changent rien (89 epochs purs). Graines 0, 1, 3 (jamais pures en référence) : aucune variante ne produit de phase pure (ACC finale 0,70 à 0,80 ; le gain de la graine 3, 0,52 -> 0,73 avec gel, ne l'amène pas au régime pur).
* Les gains moyens de `freeze4` et `slow4` (+0,09 et +0,08, environ 1,5 erreur standard) ne sont pas significatifs à 5 graines ; ce qui est établi est le mécanisme sur la graine 4.

**Lecture.** L'instabilité tardive de la graine 4 est bien un problème de péremption des intentions stockées : figer m (ou presque) la supprime sans effet sur une graine déjà stable. Mais elle ne concerne qu'une graine sur cinq. Le problème dominant reste la fusion à la formation (graines 0, 1, 3 ; 5, 7, 8 sur les 10 graines), que ni la mémoire, ni le recalcul, ni le gel, ni `rho`, ni des experts oracles n'ont réglée. Le gel est une règle de calendrier (4 epochs après les premiers placements), valable sur le jouet où la représentation est terminée à l'epoch 12 (précision du centroïde = 1,00) ; sur des données réelles, il faudrait un critère sans labels pour décider du moment (par exemple la stabilité de m entre epochs). Conséquence : `freeze_repr_after` reste désactivé par défaut.
Prochaine étape proposée : traiter la formation, où le défaut se situe dans les premiers epochs d'actions (nombre de clusters ouverts et fusions initiales), avec l'hypothèse de l'absence de répulsion de `L_psi` entre classes comme piste (mesurer la distance médiane entre classes dès l'epoch 10 sur les graines qui fusionnent).

### 5.7 Formation : géométrie de m aux epochs 0-14, puis rééchantillonnage de la politique depuis l'epoch 10
Rejeu des 10 graines de `toy_rho/ref` (1 thread, identique aux suites) ; labels pour l'évaluation seulement. Scripts de mesure hors repo.

**Géométrie de m à la formation.** L'hypothèse « `L_psi` n'a pas de répulsion entre classes, donc m est mal séparé quand l'apprenant commence à agir » est écartée pour les fusions de formation : les graines bloquées en ACC 0,5 à 0,7 (0, 5, 7) ont la MEILLEURE géométrie à l'epoch 10 (distance minimale entre moyennes de classes 1,0 à 1,3 ; séparation minimale / dispersion intra-classe 150 à 250), tandis que les graines pures (2, 9) ont 0,4 à 0,6 (60 à 130). L'effondrement de la distance médiane des graines 3 et 4 (1,45 et 1,18 à l'epoch 5, contre 4,5 à 7,1) est installé avant que l'apprenant ne place une image : il vient de la phase d'experts seuls (et il reste associé à l'instabilité tardive, voir 5.6).

**Rééchantillonnage depuis l'epoch 10.** On entraîne jusqu'à la fin de l'epoch 9 (l'apprenant n'a encore rien placé), on copie tout l'état (réseaux, optimiseurs, experts, mémoire), puis on continue 8 epochs, 8 fois, en ne changeant que le générateur aléatoire de torch (tirage des placements, minibatchs PPO, épisodes experts, vues). Nombre de runs à ACC >= 0,9 à l'epoch 17 :

| graine (devenir dans la référence) | pur à l'epoch 17 / 8 | ACC moyenne | ACC min-max |
|---|---|---|---|
| 0 (bloquée) | 1 | 0,69 | 0,5-1,0 |
| 1 (bloquée) | 0 | 0,71 | 0,7-0,7 |
| 5 (bloquée) | 3 | 0,84 | 0,7-1,0 |
| 7 (bloquée) | 2 | 0,69 | 0,5-1,0 |
| 2 (pure) | 3 | 0,80 | 0,5-1,0 |
| 9 (pure) | 4 | 0,86 | 0,6-1,0 |
| 4 (casse à l'epoch 25) | 5 | 0,90 | 0,7-1,0 |
| 3 (instable) | 2 | 0,78 | 0,4-1,0 |
| total | 20 / 64 | | |

Lecture : à représentation identique, le devenir à la formation est en grande partie une loterie de la phase politique. Les graines « pures » ne le sont que dans 3 ou 4 reprises sur 8, et trois des quatre graines « bloquées » deviennent pures dans au moins une reprise. Les 5 slots sont toujours ouverts (5 clusters à l'epoch 17 dans 63 reprises sur 64, 4 dans une) : le défaut n'est pas un manque de clusters mais des slots mélangés. La représentation (m) et la reward sont donc prêtes à l'epoch 10 (voir 5.1 et 5.3, mémoire idéale à 100 %) ; ce qui décide est la façon dont une politique non entraînée remplit la mémoire pendant ses premiers epochs. Limites : l'état à l'epoch 17 n'est pas l'état final (une reprise pure peut casser plus tard, comme la graine 4) ; 8 reprises par graine ; une seule configuration du jouet.
Piste, non testée : entraîner la politique avant qu'elle n'écrive dans la mémoire (par exemple, sans labels, la rapprocher de softmax(reward) sur des états d'épisodes experts, ou la faire jouer plusieurs epochs sans mémoire persistante), pour que les premiers placements ne remplissent pas la mémoire au hasard. À valider sur les mêmes reprises (8 par graine, epoch 10) avant tout run de 100 epochs.

### 5.8 Formation : phase « professeur » de la politique et mémoire jetée (reprises depuis l'epoch 10)
Même protocole que 5.7 : état copié à la fin de l'epoch 9, puis reprises qui ne diffèrent que par le hasard (6 reprises par graine et par variante, 8 graines ; référence : 8 reprises par graine de 5.7). Mesure : ACC >= 0,9 à la fin de 8 epochs de placements persistants (epoch 17 pour A et la référence ; B compte 3 epochs de plus, jetés). Labels pour l'évaluation seulement.

Variantes (options désactivées par défaut, chemin par défaut inchangé bit à bit, 117 tests) :
* **A, `policy_teacher_iters=30`** : pendant les 30 premières itérations de placement (2 epochs), l'apprenant échantillonne ses actions dans softmax(reward) sur sa propre mémoire, et la politique est distillée vers softmax(reward) sur ces états (KL, même nombre de pas que PPO ; le critique est ajusté aux retours) ; ensuite la politique agit seule, avec PPO. Choisie plutôt qu'une distillation sur des états d'épisodes experts pour que le regroupement de l'expert n'entre pas dans la politique (principe : l'expert ne paramètre pas l'apprenant ; ici la politique n'apprend que de la reward, comme avant).
* **B, `memory_discard_epochs=3`** : la mémoire écrite pendant les 3 premiers epochs de placement est jetée à la fin de chacun ; la politique garde ce qu'elle a appris.

| graine | référence (pur / 8) | A (pur / 6) | B (pur / 6) |
|---|---|---|---|
| 0 | 1 | 6 | 0 |
| 1 | 0 | 4 | 2 |
| 2 | 3 | 5 | 2 |
| 3 | 2 | 0 | 1 |
| 4 | 5 | 5 | 3 |
| 5 | 3 | 5 | 0 |
| 7 | 2 | 6 | 0 |
| 9 | 4 | 4 | 0 |
| total | 20 / 64 (31 %) | 35 / 48 (73 %) | 8 / 48 (17 %) |

* **A divise par plus de deux le taux d'échec de formation** (de 69 % à 27 %) et rend pures, dès l'epoch 10, des graines qui ne l'étaient presque jamais (0 : 6/6, 7 : 6/6, 1 : 4/6). Cela soutient l'explication par les premiers placements d'une politique non entraînée : leur donner la reward comme guide suffit à améliorer fortement la formation, sans toucher à la représentation, à la reward ni à la structure de la mémoire.
* **B n'aide pas** (17 %, moins que la référence) : jeter la mémoire laisse la politique apprendre sur des états contaminés.
* **La graine 3 n'est pas réparée** (0/6) : c'est la graine à distance médiane effondrée (5.7), où la géométrie de m, et non la politique, limite le résultat.
* Limites : état à l'epoch 17 seulement (une phase pure peut casser plus tard, comme la graine 4 en 5.5) ; 6 reprises par graine ; pas encore de run de 100 epochs ; 30 itérations est une valeur choisie sans réglage.
Suite proposée : `configs/suites/toy_teacher.yml` (5 graines, 100 epochs : référence, `teacher30`, `teacher30_freeze4`), pour voir si les phases pures obtenues tiennent jusqu'à l'epoch 100 et si le gel de 5.6 supprime les casses tardives.

### 5.9 Pourquoi la reward n'ouvre pas : analyse de `L_pos` et `L_neg`, puis test `free_debias_scale`
**Analyse (formules, `cibirl/losses.py`).** Notations : $V$ = actions valides (slots occupés + un seul slot vide) ; $s_k$ = score de la reward pour $k$ ; $q(k)=e^{s_k}/\sum_{j\in V}e^{s_j}$ ; $k^+$ = slot du jumeau (autre vue de la même image) ; $\tau^+=1/K$.
* `L_pos` (mouvements forcés) : $L_{pos}=-\log q(k^+)$, donc $\partial L_{pos}/\partial s_{\text{ouvrir}}=q(\text{ouvrir})\ge 0$ : à chaque mouvement forcé le score d'ouvrir est poussé vers le bas ; rien dans `L_pos` ne le remonte (pas de terme positif sur les mouvements libres). Mesure : $q(\text{ouvrir})\approx 0{,}04$ sur les mouvements forcés.
* `L_neg` (mouvements libres) : $\ell(u,v)=-\log(1-q(k(v)))$, $L_{neg}(u)=\max\{(\bar\ell(u)-\tau^+\ell^+)/(1-\tau^+),0\}$ avec $\ell^+=\mathbb{E}_{\text{forcés}}[-\log(1-q(k^+))]$. Le gradient est nul dès que $\bar\ell\le\tau^+\ell^+$, soit (un seul mauvais slot) $q\le 1-e^{-\tau^+\ell^+}$ : une zone morte. $\ell^+$ est mesuré sur des jumeaux (cas faciles) alors que les faux négatifs cachés sont des images de même classe mais différentes (plus difficiles) : la correction retranche trop, et d'autant plus que le modèle est bon sur les jumeaux. Mesures (epoch 20, graines 0, 2, 4) : $\ell^+=3{,}0$ à $5{,}1$ ; répulsion nulle sur 72 à 82 % des mouvements libres ; à l'epoch 9, 55 à 72 % (moyenne 68 %).
* Les autres pertes (`L_psi`, KL, `L_inv`, bonus d'entropie) ne poussent pas vers moins de clusters ; `L_psi` tasse les classes dans m (5.7) sans les fusionner ; `rho` est le seul terme anti-fusion et est désactivé (5.4).
* Ce que dit la reward apprise (epoch 20, états reconstruits avec les vraies classes) : si une classe manque de la mémoire, $q(\text{ouvrir})$ moyen = 0,54 / 0,33 / 0,30 (graines 2 / 4 / 0), idéal 1 ; avec un slot mélangeant deux classes, la reward préfère le mélange à ouvrir dans 75 / 90 / 95 % des paires. Réserve : ces états ont 4 slots occupés, hors de la distribution des épisodes experts (au plus 2 slots occupés quand il faut ouvrir avec 3 orbites).

**Test : option `free_debias_scale`** ($\tau^+\to s\,\tau^+$ sur les mouvements libres seulement ; 1 = inchangé, bit à bit ; 0 = plus de correction), appliquée dès l'epoch 0. Protocole de 5.7 : 10 epochs, sonde de la reward, copie, 6 reprises de 8 epochs par graine (8 graines), seul le hasard change. Référence : 20/64 reprises pures (31 %), ACC finale moyenne 0,78.

| | référence ($s=1$) | $s=0{,}5$ | $s=0$ |
|---|---|---|---|
| mouvements libres sans gradient de répulsion (epoch 9) | 68 % | 54 à 68 % | 0 à 9 % |
| $q(\text{ouvrir})$ quand une classe manque (epoch 9, moyenne des 8 graines) | 0,38 | 0,46 | 0,53 |
| reprises pures à l'epoch 17 | 20/64 (31 %) | 16/48 (33 %) | 18/48 (38 %) |
| ACC finale moyenne | 0,78 | 0,79 | 0,83 |

* La zone morte disparaît avec $s=0$ et la reward ouvre mieux dans les 8 graines ($q(\text{ouvrir})$ monte de 0,38 à 0,53 en moyenne), donc le mécanisme décrit existe bien.
* Mais le taux de formation pure ne change pas de façon distinguable (38 % contre 31 %, écart du même ordre que le bruit à ce nombre de reprises), et la reward n'ouvre toujours que dans environ la moitié des cas. La zone morte est donc un facteur réel mais pas la cause suffisante. Comparaison : la phase « professeur » de 5.8 donne 73 %.
* Piste suivante, non testée : les épisodes experts n'exposent jamais plus de 2 slots occupés au moment d'ouvrir (3 orbites), alors que la politique doit ouvrir avec 4 slots occupés ; combiner `free_debias_scale=0` avec plus d'orbites par épisode (`orbits_per_episode`), ou ajouter un terme positif pour ouvrir.

### 5.10 Deux tests : plancher de répulsion avec $\rho$, et 5 orbites par épisode expert
Protocole de 5.9 : option appliquée dès l'epoch 0, 10 epochs, sonde de la reward, copie, 6 reprises de 8 epochs par graine (8 graines), seul le hasard change. Références : 20/64 reprises pures à l'epoch 17 (31 %), ACC finale moyenne 0,78, $q(\text{ouvrir})$ quand une classe manque = 0,38.

* **Test 1, `rho=0,1` + `neg_floor=0,1`.** Une constante ajoutée à $L_{neg}$ n'a pas de gradient ; le plancher est donc placé à l'intérieur du max : $L_{neg}(u)=\max\{\frac{\bar\ell(u)-\tau^+\ell^+}{1-\tau^+},\ \varepsilon\,\bar\ell(u)\}$ avec $\varepsilon=0{,}1$, où $\bar\ell(u)$ est la répulsion moyenne non corrigée ($\ell(u,v)=-\log(1-q(k(v)))$ moyenné sur les vues $v$ d'autres orbites). Ainsi une part $\varepsilon$ de la répulsion simple subsiste toujours, y compris dans la zone morte.
* **Test 2, `orbits_per_episode=5`** (au lieu de 3) : l'expert ouvre avec jusqu'à 4 slots occupés.

| | référence | test 1 | test 2 |
|---|---|---|---|
| reprises pures à l'epoch 17 | 20/64 (31 %) | 14/48 (29 %) | 4/48 (8 %) |
| ACC finale moyenne | 0,78 | 0,80 | 0,73 |
| $q(\text{ouvrir})$ si une classe manque (epoch 9) | 0,38 | 0,42 | 0,79 |

* **Test 1 : aucun effet** sur la formation (29 % contre 31 %, ACC 0,80 contre 0,78) ni sur la sonde de la reward (0,42 contre 0,38). Le plancher et $\rho$ ne suffisent pas à changer le comportement.
* **Test 2 : la reward apprend bien à ouvrir** (0,79 contre 0,38, en hausse dans les 8 graines, de 0,70 à 0,90), ce qui soutient l'hypothèse que le manque d'exemples d'ouverture avec plusieurs slots occupés (au plus 2 avec 3 orbites) contribue au défaut mesuré en 5.9 ; **mais la formation est moins bonne** (8 % de reprises pures, ACC 0,73). Les partitions obtenues sont toujours fausses des deux manières : graine 2, deux classes fusionnées dans un cluster alors qu'un cluster reste vide ; graine 7, une classe coupée en deux clusters (130 + 170) et deux autres fusionnées. Ouvrir plus souvent n'a donc pas supprimé les fusions et a ajouté des coupures ; avec 5 orbites sur 5 classes, la probabilité que deux orbites d'un épisode soient de la même classe est plus forte (plus de faux négatifs côté expert), ce qui peut expliquer une partie des coupures (non testé).
* Aucun run de 100 epochs n'a été lancé pour ces deux variantes (la formation, première condition de la stabilité, n'est pas meilleure) ; la suite `configs/suites/toy_open.yml` est prête si l'on veut tout de même mesurer la stabilité.
* Bilan de 5.8 à 5.10 : la phase « professeur » de la politique (73 %) reste la seule intervention qui améliore nettement la formation ; les corrections de la reward ($s=0$, plancher, 5 orbites) améliorent la sonde de la reward (surtout 5 orbites) sans améliorer la formation.

### 5.11 Changement de contexte : MNIST, chiffres 1 à 6 (2026-10-14)
Pour savoir si l'échec du jouet (fusions de classes) se reproduit sur des classes bien distinguables, run unique, fixé avant résultat : `data=mnist`, chiffres 1 à 6, 300 images par chiffre tirées au hasard avec la graine (1800 images), banque `digits` (crop + rotation $\pm 15^\circ$ + luminosité / contraste, sans flip), $K=6$, $B=5$ orbites, $n=2$ vues, 2000 épisodes experts, réglages du jouet pour le reste (`configs/mnist.yml`, suite `configs/suites/mnist_digits.yml`). 3 graines, arrêt à la convergence (règle sans labels) ou à 200 epochs.

| graine | epochs | convergé | ACC | NMI | clusters | kNN(10) pixels | kNN(10) $z$ | kNN(10) $m$ | k-means $m$ | retrouver le jumeau par $m$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | non | 0,43 | 0,33 | 5 | 0,93 | 0,78 | 0,68 | 0,33 | 8 % |
| 1 | 200 | non | 0,41 | 0,31 | 6 | 0,94 | 0,86 | 0,69 | 0,33 | 3 % |
| 2 | 200 | non | 0,42 | 0,32 | 5 | 0,94 | 0,80 | 0,65 | 0,33 | 3 % |

(« kNN » : plus proches voisins sur 1800 images, labels utilisés pour l'évaluation seulement ; $z$ = code de l'encodeur, $m$ = intention ; k-means sur les pixels : ACC 0,67 à 0,76.)

* **Aucune graine ne converge** : ACC entre 0,25 et 0,51 de l'epoch 15 à l'epoch 200 (autour de 0,4), et en moyenne 34 à 37 % des images changent de cluster d'une epoch à l'autre (de l'epoch 40 à 200).
* **Le chiffre 1 est isolé** (≈ 290 sur 300 dans un cluster, dans les 3 graines) ; les chiffres 2 à 6 sont répartis dans 3 à 4 clusters, sans correspondance chiffre-cluster.
* **Ce n'est pas le même tableau que le jouet** : sur le jouet la représentation $m$ était parfaite (plus proche moyenne de classe : 1,0) et l'échec venait de la partition. Ici $m$ est déjà mauvais avant la politique (kNN 0,65 à 0,69 contre 0,93 pour les pixels ; $z$ garde plus : 0,78 à 0,86 ; le jumeau d'une vue est retrouvé par $m$ dans 3 à 8 % des cas).
* La précision « twin » de la reward est ≈ 0,7 au départ, puis baisse à 0,4–0,5 sur les graines 1 et 2 (reste ≈ 0,7 sur la graine 0) ; la KL passe d'environ 900 à 170–540.
* Non testé (hypothèses) : compression trop forte ($\beta$), tâche de l'expert trop difficile avec $B=5$ (avec 6 classes, deux orbites d'un épisode sont de la même classe dans ≈ 91 % des cas mais comptées comme différentes), banque de transformations. Un seul réglage a été essayé, sans ajustement.

### 5.12 MNIST : où se perd l'information (sonde de $z$ et de $m$ au cours de l'entraînement, 2026-10-15)
Même run que 5.11 (mêmes graines, même config), sonde `scripts/probe_representation.py` aux epochs 0, 2, 4, 6, 7, 8, 9, 11, 14, 19, 24, 29, 39, 49 (la sonde ne perturbe pas l'entraînement : à l'epoch 2 de la graine 0 les chiffres sont identiques au run complet). Résultats dans `results/mnist_digits/probe/seed{0,1,2}.jsonl`, figure `results/mnist_digits/probe/representation.png`. L'apprenant commence à placer à l'epoch 8 : avant, seule la phase expert agit.

* **Dès la phase expert, $m$ est mauvais et ne s'améliore pas.** kNN(10) sur $m$ : 0,71 à 0,78 à l'epoch 0 (encodeur aléatoire), puis 0,53 à 0,71 à l'epoch 9, et 0,65 à 0,68 à l'epoch 49 ; les pixels bruts donnent 0,93. L'entraînement sur les experts ne l'a pas amélioré par rapport à l'initialisation. Le k-means sur $m$ reste entre 0,24 et 0,44 sur toute la période (k-means sur les pixels : 0,67 à 0,76).
* $z$ garde plus de classe que $m$ (kNN 0,71 à 0,88, environ 0,75 à l'epoch 49) : une partie de la perte est entre $z$ et $m$ (l'intention, entraînée par $L_\psi$ seul).
* La KL de l'encodeur passe de ≈ 70 (epoch 0) à 700–960 dès l'epoch 2 puis monte jusqu'à 1300 (graine 1) : le code porte beaucoup de bits. $L_\psi$ descend à $-32$ à $-41$ à l'epoch 8 puis à $-43$ à $-45$ à l'epoch 29, ce qui correspond (hypothèse non vérifiée) à une variance de $q_\psi$ collée à son plancher plutôt qu'à une vraie cohérence d'orbite.
* Retrouver le jumeau d'une vue par plus proche voisin dans $m$ : 3 à 7 % seulement (hasard : 0,06 %).
* La précision « twin » de la reward est ≈ 0,7 dès l'epoch 4 et n'évolue guère pendant la phase expert (0,65 à 0,77 de l'epoch 2 à 9) ; sa baisse sur les graines 1 et 2 apparaît plus tard, pendant la phase apprenant.
* **Lecture.** Le problème de MNIST se situe en amont de la politique et de la mémoire : à l'epoch 9, avant tout placement, $m$ ne sépare déjà pas les chiffres. Le signal « distinguer des images différentes » semble être appris au niveau de l'instance (la KL est élevée) plutôt qu'au niveau de la classe ; le test de cette lecture (par exemple, $\beta$ plus grand, ou 3 orbites) n'est pas fait.

### 5.13 MNIST : ni la compression ni le nombre d'orbites n'expliquent la mauvaise représentation (2026-10-15)
Trois variantes, appliquées dès l'epoch 0, sondées par `scripts/probe_representation.py` (3 graines, jusqu'à l'epoch 19) : `beta=0,001` (au lieu de $10^{-4}$) avec une montée de $\beta$ raccourcie à 60 itérations (`beta_warmup_iters=60`, sinon $\beta$ n'atteint sa valeur qu'à l'epoch ≈ 16), `orbits_per_episode=3` (au lieu de 5), et les deux ensemble. Résultats dans `results/mnist_digits/probe_beta1e-3/`, `probe_orbits3/`, `probe_beta1e-3_orbits3/` ; moyenne sur les 3 graines [min–max].

| epoch 19 | kNN(10) sur $m$ | k-means sur $m$ | kNN(10) sur $z$ | KL | précision « twin » | ACC de la politique |
|---|---|---|---|---|---|---|
| référence ($\beta=10^{-4}$, 5 orbites) | 0,62 [0,60–0,65] | 0,29 [0,24–0,35] | 0,76 | 936 | 0,72 | 0,41 [0,37–0,44] |
| $\beta=10^{-3}$ | 0,63 [0,59–0,65] | 0,33 [0,29–0,37] | 0,74 | 552 | 0,72 | 0,38 [0,36–0,40] |
| 3 orbites | 0,64 [0,63–0,64] | 0,33 [0,30–0,37] | 0,76 | 761 | 0,81 | 0,35 [0,30–0,42] |
| $\beta=10^{-3}$ + 3 orbites | 0,65 [0,53–0,75] | 0,29 [0,24–0,38] | 0,78 | 539 | 0,81 | 0,40 [0,39–0,41] |

(Pixels : kNN 0,93, k-means 0,67–0,76 ; encodeur aléatoire : kNN sur $m$ 0,71–0,78.) À l'epoch 9 les écarts sont du même ordre (kNN sur $m$ : 0,62 à 0,67 pour les quatre).

* **Aucune des variantes n'améliore $m$** : les écarts entre variantes sont dans la dispersion des graines. Réduire la KL de 936 à environ 540 ne change pas la structure de classe de $m$ ; réduire le nombre d'orbites augmente la précision « twin » (0,72 → 0,81, ce qui est attendu : moins de slots à départager) sans effet sur $m$ ni sur l'ACC.
* Les hypothèses « compression trop forte » et « trop de faux négatifs de l'expert avec 5 orbites » ne sont donc pas soutenues par ces tests (pour ces valeurs, 20 epochs, 3 graines).
* Restent non testées : la force de la banque de transformations (hypothèse 3 de l'utilisateur), l'architecture (l'intention $m$ n'est entraînée que par $L_\psi$), la taille de la partie expert (30 % de 1800 images).

### 5.14 MNIST : la banque de transformations (hypothèse 3), en pixels et avec une banque plus forte (2026-10-15)
**Géométrie de la banque en pixels** (`scripts/probe_bank.py`, graine 0, 1800 images ; deux vues par image, la banque appliquée 1 ou 2 fois comme dans une orbite ; distances euclidiennes médianes entre pixels ; résultats dans `results/mnist_digits/bank_geometry_seed0.json`) :

| banque | applications | distance entre 2 vues d'une image | distance entre 2 images de même classe | distance entre 2 images de classes différentes | jumeau = plus proche vue | plus proche vue de même classe |
|---|---|---|---|---|---|---|
| `digits` | 1 | 6,84 | 9,02 | 9,87 | 19 % | 92 % |
| `digits` | 2 | 7,86 | 9,23 | 9,89 | 8 % | 87 % |
| `digits_strong` | 1 | 9,24 | 10,15 | 10,61 | 5 % | 80 % |
| `digits_strong` | 2 | 10,14 | 10,74 | 11,04 | 2 % | 67 % |

* Avec `digits`, deux vues d'une même image ne sont pas beaucoup plus proches (6,8) que deux images de même classe (9,0) : en pixels, l'instance n'est pas triviale à reconnaître (19 % ; 8 % avec 2 applications), alors que la classe l'est davantage (plus proche vue de même classe : 92 %). L'hypothèse « les orbites se résolvent trop facilement au niveau de l'instance » n'est pas soutenue **en pixels** (elle reste possible dans l'espace appris).
* **Banque plus forte** (`configs/transforms/digits_strong.yml` : crop $\geq 0{,}5$, rotation $\pm 30^\circ$, jitter 0,3 ; fixée avant résultat, une étape de sensibilité de `digits`), sondée comme en 5.13 (3 graines, epoch 19, dossier `results/mnist_digits/probe_digits_strong/`) :

| epoch 19 | kNN(10) sur $m$ | k-means sur $m$ | kNN(10) sur $z$ | KL | précision « twin » | retrouver le jumeau par $m$ | ACC de la politique |
|---|---|---|---|---|---|---|---|
| `digits` | 0,62 [0,60–0,65] | 0,29 | 0,76 | 936 | 0,72 | 4 % | 0,41 [0,37–0,44] |
| `digits_strong` | 0,56 [0,51–0,61] | 0,30 | 0,66 | 585 | 0,58 | 1 % | 0,38 [0,36–0,40] |

* **Une banque plus forte dégrade $m$ et $z$ au lieu de les améliorer** (kNN sur $m$ 0,56 contre 0,62 ; sur $z$ 0,66 contre 0,76) et rend la tâche de l'expert plus difficile (précision « twin » 0,58 contre 0,72). L'ACC de la politique ne change pas (0,38 contre 0,41, dans la dispersion).
* Conclusion partielle : pour ce sens de variation (banque plus forte) l'hypothèse 3 n'est pas soutenue. Une banque **plus faible** n'a pas été testée.

### 5.15 MNIST : l'intention $m$ est de rang 1, variance fixe de $L_\psi$ (2026-10-15)
**Sonde étendue** (`scripts/probe_representation.py`, 3 graines, epochs 0 à 29, `results/mnist_digits/probe_intent/`) :

* **$m$ est de rang 1 dès l'epoch 2** : le premier axe principal de $m$ porte 99,4 à 99,5 % de la variance à l'epoch 2 et 99,9 à 100 % à partir de l'epoch 9 (rapport de participation 1,00 ; déjà 93 à 96 % à l'epoch 0, avec un encodeur presque aléatoire). Les 16 dimensions sont des copies d'une même variable : la contrainte « variance 1 par dimension » du BatchNorm est satisfaite sans rien porter de plus. À l'epoch 200, $m$ est quasi binaire : ≈ 290 images (le chiffre 1) d'un côté, ≈ 1500 de l'autre, ce qui explique pourquoi seul le 1 est isolé dans 5.11. **Les runs du jouet ont aussi un $m$ de rang 1** (`ref_seed0` à `4`, rapport de participation 1,00) : un axe unique suffit à porter 5 classes qui valent chacune une valeur ; ce n'est donc pas une différence jouet / MNIST, mais une limite qui devient bloquante dès que les classes ont une variabilité interne.
* **La variance prédite par l'intention atteint son plancher** ($\log\sigma^2=-6$) : part des paires (image, dimension) au plancher 0 % à l'epoch 2, 0 à 67 % à l'epoch 9, 66 à 84 % à l'epoch 19, 83 à 95 % à l'epoch 29 (selon la graine). $L_\psi$ vaut alors ≈ $-40$ à $-45$ (minimum $-48$) : le terme $\log\sigma^2$ domine. La cohérence des vues est réelle (erreur RMS par dimension entre deux vues 0,002 à 0,023 à l'epoch 200, pour un écart-type 1,0 entre images) ; en entraînement comme en inférence, la statistique de batch ou la statistique courante du BatchNorm donne les mêmes chiffres.
* **Pas de mémorisation des images d'expert** : $m$ vaut autant sur les 30 % d'images qui fabriquent les experts que sur autant d'images tirées parmi les autres (écart moyen sur les epochs ≥ 4 : kNN $-0{,}001$, jumeau retrouvé $0{,}000$, écart-type 0,02).

**Test « variance fixe »** (`psi_fixed_logvar=0.0` : $L_\psi=\tfrac12\sum_j (m^*_j-\mu_j)^2$ au lieu du terme avec variance prédite ; option de diagnostic, `None` par défaut ; 3 graines, `results/mnist_digits/probe_psi_fixed/`), moyenne [min–max] :

| epoch 19 | kNN(10) sur $m$ | k-means sur $m$ | kNN(10) sur $z$ | premier axe de $m$ | KL | ACC de la politique |
|---|---|---|---|---|---|---|
| référence | 0,62 [0,60–0,65] | 0,29 | 0,76 | 100 % | 936 | 0,41 [0,37–0,44] |
| variance fixe | 0,71 [0,69–0,73] | 0,37 | 0,85 | 100 % | 94 | 0,42 [0,41–0,43] |

* La variance fixe **améliore $z$ et $m$** (kNN sur $z$ 0,85 contre 0,76 ; sur $m$ 0,71 contre 0,62, soit le niveau de l'encodeur aléatoire à l'epoch 0) et divise la KL par dix : la variance apprise poussait l'encodeur à porter beaucoup de bits pour rendre les vues cohérentes. **Elle ne change ni le rang 1 de $m$ ni l'ACC de la politique** (0,42 contre 0,41).
* Conclusion partielle : la variance apprise de $L_\psi$ dégrade la représentation, mais le goulot qui borne la clusterisation est le rang 1 de $m$ ; avec une seule direction, la distinction la plus simple et la plus invariante (le 1 contre le reste) l'emporte. Piste non testée : imposer des dimensions décorrélées (blanchiment ou pénalité de covariance à la place de la seule standardisation par dimension).

### 5.16 MNIST : décorréler les dimensions de $m$ (2026-10-15)
Cause visée : en 5.15, $m$ est de rang 1 (la standardisation par BatchNorm donne à chaque dimension une variance 1, mais elles peuvent toutes être des copies d'une même variable). Deux options (`intent_decorrelate`, défaut `none` = comme le README ; toutes deux des choix d'implémentation hors README, comme le BatchNorm lui-même) :

* `cov` : BatchNorm conservé, plus la pénalité $\lambda_{\text{cov}}\,\frac{1}{d}\sum_{i\neq j} C_{ij}^2$ sur $m$, où $C$ est la covariance de $m$ sur le batch, $d=16$ le nombre de dimensions et $\lambda_{\text{cov}}=10$ (`intent_cov_weight`, une seule valeur, non ajustée). Elle vaut $d-1$ si toutes les dimensions sont des copies, 0 si elles sont décorrélées.
* `whiten` : blanchiment de Cholesky du batch à la place du BatchNorm (statistiques courantes à l'inférence).

Chacune seule et avec la variance fixe de 5.15 ; 3 graines, epochs 0 à 19, dossiers `results/mnist_digits/probe_dec_*/` ; moyenne [min–max].

| epoch 19 | kNN(10) sur $m$ | k-means sur $m$ | kNN(10) sur $z$ | rapport de participation de $m$ | jumeau retrouvé par $m$ | précision « twin » | ACC de la politique |
|---|---|---|---|---|---|---|---|
| référence | 0,62 [0,60–0,65] | 0,29 | 0,76 | 1,0 | 4 % | 0,72 | 0,41 [0,37–0,44] |
| variance fixe (5.15) | 0,71 | 0,37 | 0,85 | 1,0 | 5 % | 0,70 | 0,42 |
| **`cov`** | **0,98 [0,98–0,98]** | **0,95 [0,93–0,97]** | **0,98** | **13,6 [12,9–14,4]** | **51 %** | **0,98** | **0,80 [0,71–0,92]** |
| `cov` + variance fixe | 0,95 [0,95–0,96] | 0,70 [0,60–0,79] | 0,96 | 14,8 | 22 % | 0,91 | 0,61 [0,45–0,83] |
| `whiten` | 0,41 [0,39–0,42] | 0,41 | 0,71 | 1,4 [1,0–2,3] | 1 % | 0,58 [0,21–0,77] | 0,38 [0,33–0,42] |
| `whiten` + variance fixe | 0,55 [0,50–0,59] | 0,36 | 0,72 | 5,1 [4,1–6,4] | 3 % | 0,28 | 0,30 [0,25–0,39] |

(Pixels : kNN 0,93, k-means 0,67–0,76. À l'epoch 9, `cov` : kNN sur $m$ 0,96, ACC 0,61 [0,58–0,64].)

* **La pénalité de covariance seule lève le goulot** : le rang de $m$ passe de 1 à environ 14 sur 16, $m$ et $z$ séparent presque parfaitement les chiffres (kNN 0,98, au-dessus des pixels à 0,93) et l'ACC de la politique à l'epoch 19 passe de 0,41 à 0,80 (une graine à 0,92, une à 0,71). La cause du rang 1 de 5.15 est donc soutenue de façon causale par ce test.
* **Avec la variance fixe en plus, c'est moins bon** que la pénalité seule (ACC 0,61, jumeau retrouvé 22 % contre 51 %) : sans le terme de variance appris, la cohérence des vues est moins fortement imposée.
* **Le blanchiment de Cholesky échoue** (rang 1,4 à 5, précision « twin » instable de 0,21 à 0,77) : il ne remplace pas la pénalité (pas étudié plus loin : $\varepsilon=10^{-4}$ de régularisation, un seul réglage).
* Limites : 20 epochs, 3 graines, une seule valeur de $\lambda_{\text{cov}}$, un seul jeu de données ; rien n'est encore dit de la convergence ni de la stabilité (run complet lancé, section suivante).

### 5.17 Retour au jouet : pénalité de covariance + $\rho$ + plancher de répulsion (2026-10-15)
Question posée après 5.15-5.16 : si l'on fait « comme s'il n'y avait jamais eu de souci » (intention $m$ corrigée), les trois choix jugés justifiés (pénalité de covariance sur $m$, poids 10 ; $\rho=0{,}1$ ; `neg_floor=0,1`) suppriment-ils les fusions du jouet ? Protocole fixé avant résultat : suite `configs/suites/toy_cov.yml`, jouet (`configs/toy.yml`, banque `toy_oracle`), les 6 graines de `toy_rho/ref` qui avaient fusionné (0, 1, 3, 5, 7, 8), arrêt à la convergence (règle sans labels de la section 2, `min_epochs` 40) ou à 200 epochs. Deux conditions : `ref` = ancien défaut (rejoue la dégénérescence), `fix` = `intent_decorrelate=cov` + `rho=0,1` + `neg_floor=0,1`.

| | `ref` | `fix` |
|---|---|---|
| ACC finale par graine (0, 1, 3, 5, 7, 8) | 0,80 / 0,80 / 0,51 / 0,80 / 0,80 / 0,60 | 1,00 / 1,00 / 1,00 / 0,70 / 1,00 / 1,00 |
| ACC moyenne $\pm$ écart-type | 0,72 $\pm$ 0,12 | 0,95 $\pm$ 0,11 |
| clusters utilisés | 4, 4, 4, 4, 4, 3 | 5 dans les 6 graines |
| convergées (règle sans labels) | 0/6 (200 epochs) | 5/6 (40, 61, 40, 48, 61 epochs) |
| première epoch avec ACC $\ge$ 0,95 | 77 à 150 (3 graines sur 6 jamais) | 15 à 21 (5 graines) |
| rang de $m$ final (participation, 1 = rang 1) | 1,0 dans les 6 graines (part de la 1re composante = 1,00) | 2,0 à 4,0 (part de la 1re composante 0,26 à 0,52) |

* `ref` reproduit la dégénérescence (jamais convergé, ACC oscillante, 4 clusters : deux classes fusionnées ; graine 8 : 3 clusters). Dans le jouet aussi, $m$ final est exactement de rang 1 sous l'ancien défaut (5.15 l'avait noté), alors que la pénalité lui redonne plusieurs dimensions indépendantes.
* `fix` règle 5 graines sur 6 : ACC 1,00, 5 clusters, convergence entre 40 et 61 epochs, sans rechute après convergence. La graine 5 ne converge pas : ACC 0,70 à 0,80 sur 200 epochs, 5 clusters mais deux classes fusionnées et une troisième coupée en deux (156 + 144 images), donc le même mode d'échec, plus rare, n'est pas supprimé.
* Limites. (1) Les 6 graines ont été choisies parce qu'elles avaient échoué avec `ref` : cela mesure la correction sur des cas difficiles, pas un taux d'échec global ; les 4 autres graines de `toy_rho` ne sont pas rejouées. (2) `fix` regroupe trois changements ; ce run ne dit pas lequel agit (5.4 et 5.10 avaient montré que $\rho$ et le plancher seuls ne changeaient rien, ce qui désigne la pénalité, sans le prouver). (3) 6 graines, une seule valeur de $\lambda_{\text{cov}}$ (10, non ajustée). (4) `ref` est arrêté à 200 epochs ici ; les chiffres de 5.1 à 5.10 sont à 100 epochs.
* Hors sujet jouet : le run MNIST complet de 5.16 (`mnist_digits_cov`) s'est interrompu vers l'epoch 30 (processus tué, pas d'erreur de code) ; il n'est donc pas terminé et n'a pas été relancé. Les logs partiels (epochs 28 à 32) donnaient une ACC de suivi 0,96 / 0,95 / 0,70 pour les graines 0, 1, 2 ; ce ne sont pas des résultats finaux.

### 5.18 Jouet : vérifications de la pénalité de covariance (2026-10-15)
Trois vérifications demandées après 5.17, suite `configs/suites/toy_cov_check.yml`, jouet, même réglage que `fix` de 5.17 (pénalité de covariance, `rho=0,1`, `neg_floor=0,1`) sauf mention, arrêt à la convergence (règle sans labels, `min_epochs` 40) ou à **300 epochs**. Pour permettre des graines différentes par expérience, `scripts/run_suite.py` accepte maintenant une clé `seeds` par expérience (test ajouté dans `tests/test_suite.py`).

**(a) Les 4 autres graines de `toy_rho` (2, 4, 6, 9), condition `fix`.** Les 4 convergent à ACC 1,00 avec 5 clusters, en 43, 46, 50 et 40 epochs. Avec les 6 graines de 5.17, `fix` converge à ACC 1,00 sur 9 graines sur 10 ; la graine 5 est la seule qui échoue.

**(b) Graine 5, poursuivie jusqu'à 300 epochs.** Le run reprend à l'identique les 200 premières epochs de 5.17 (vérifié : mêmes valeurs), puis ne converge pas : ACC entre 0,70 et 0,80, 4 ou 5 clusters, environ 5 % des images changent de cluster à chaque epoch (epochs 200 à 300 comme 100 à 200). La partition reste fausse de la même façon : classes 1 et 2 toujours dans un même cluster ; la classe 0 est tantôt entière (4 clusters, ACC 0,80, c'est le cas à l'epoch 300), tantôt coupée en deux (5 clusters, ACC 0,70 ; à l'epoch 200 : 145 + 155 images). Diagnostic sur le modèle final (epoch 200 du run de 5.17 et epoch 300 de ce run, mêmes constats ; labels pour l'évaluation seulement ; scripts de mesure hors repo) :
* **La représentation n'est pas en cause.** $m$ sépare parfaitement les 5 classes (kNN(10) = 1,00, k-means = 1,00), son rang est celui des graines qui marchent (participation 3,9 contre 4,0) et les classes y sont très compactes (distance entre centres $\ge 1500$ fois l'écart moyen intra-classe). La pénalité ne dégrade donc pas $m$ sur cette graine ; sa valeur finale est faible ($L_{cov}=0{,}010$, soit $\lambda L_{cov}=0{,}10$).
* **L'erreur se fige à la formation.** Comme la graine 0, la graine 5 hésite entre les epochs 10 et 17 (nombre de clusters 1 à 5, partitions changeantes) ; la graine 0 sort de l'hésitation (ACC 1,00 à l'epoch 18, partition pure à l'epoch 21), la graine 5 se fixe à l'epoch 17 sur 5 clusters faux (erreur de placement 0,2 sans changement ensuite).
* **La reward sait ouvrir, mais pas sur un slot mélangé.** Mémoire propre (les 4 autres classes dans 4 slots, summary = centre vrai de classe, la classe $c$ absente, un slot libre) : la reward choisit « ouvrir » pour 100 % des images de chaque classe (probabilité moyenne 1,00). Dans l'état réel, les images des classes 1 et 2 rejoignent le slot mélangé (summary = point milieu de deux classes) avec une probabilité de 1,00, et même quand on libère un slot par contrefactuel, la probabilité d'ouvrir reste 0,00 ; les images de la classe 0 se répartissent 0,50 / 0,50 entre les deux moitiés de la mémoire (la reward est indifférente entre elles, d'où la partition argmax tantôt entière, tantôt coupée) ; la politique recopie la reward. Un summary qui mélange deux classes n'apparaît jamais chez l'expert (un slot expert contient 1 ou 2 vues d'une même orbite), la reward y est hors distribution. Référence : sous l'ancien défaut (`ref`), la même sonde donne « ouvrir » pour une seule classe sur 5 (probabilité 0,00 pour les 4 autres).
* **Piège de combinatoire.** Ici $K=5$ est aussi le nombre de classes : une classe coupée en deux occupe le slot qui manquerait pour séparer les deux classes fusionnées. Sortir de l'état demande de déplacer ensemble les 150 images d'une moitié de la classe 0, ce que des décisions image par image ne font pas.

**(c) Sensibilité à $\lambda_{cov}$** (graines 3 et 8, les deux où l'ancien défaut avait le plus échoué, ACC 0,51 et 0,60 ; et graine 5 pour $\lambda \in \{1, 3, 30\}$). Valeur par défaut 10 (résultats de 5.17).

| $\lambda_{cov}$ | graine 3 | graine 8 | graine 5 |
|---|---|---|---|
| 0,1 | 1,00 (46 epochs) | 1,00 (49) | non testé |
| 1 | 1,00 (45) | 1,00 (49) | 0,80, 4 clusters, « convergé » à l'epoch 123 |
| 3 | 1,00 (40) | 1,00 (92) | 1,00 (51) |
| 10 | 1,00 (40) | 1,00 (61) | 0,80 (pas de convergence à 300) |
| 30 | 1,00 (49) | 0,41, 4 clusters, pas de convergence | 1,00 (57) |
| 100 | 0,25, 2 clusters, pas de convergence | 0,44, 5 clusters, pas de convergence | non testé |

* **Fenêtre utile : $\lambda$ entre 3 et 10 environ.** Dès $\lambda=0{,}1$ le rang de $m$ est restauré (participation 4,0) et les graines 3 et 8 convergent. À $\lambda=3$ les trois graines convergent à ACC 1,00. À $\lambda \ge 30$ des échecs apparaissent (graine 8 à 30, graines 3 et 8 à 100).
* **Pourquoi un $\lambda$ trop grand échoue.** Aux échecs ($\lambda=100$ graines 3 et 8, $\lambda=30$ graine 8), $m$ sépare encore les classes (kNN(10) = 1,00) mais elles ne sont plus compactes (distance entre centres de 4 à 22 fois l'écart intra-classe, contre 400 à 1700 dans les runs qui convergent), le rang final mesuré est plus bas (participation 1,1 à 1,7, contre 3 à 4) et 40 à 50 % des images changent de cluster à chaque epoch ($\lambda=100$ graines 3 et 8, $\lambda=30$ graine 8).
* **Graine 5.** Non monotone en $\lambda$ (échec à 1 et 10, succès à 3 et 30) ; ce n'est donc pas un effet de « pénalité trop forte » ni « trop faible » à $\lambda=10$, mais plutôt le piège de formation du point (b) tiré ou non selon la trajectoire. À $\lambda=1$ la règle d'arrêt sans labels déclare la convergence sur une partition fausse et stable (ACC 0,80, 4 clusters) : la règle ne protège pas contre une partition stable mais mauvaise.
* Limites : 2 à 3 graines par valeur de $\lambda$, un seul run par cellule, une seule valeur de $\rho$ et de plancher ; la graine 5 est la seule où $\lambda=3$ et $\lambda=10$ diffèrent, ce qui ne suffit pas pour changer le défaut (10) ; une vérification de $\lambda=3$ sur les autres graines serait nécessaire pour le faire.

### 5.19 Jouet avec plus de slots que de classes : $K=10$ pour 5 classes (2026-10-15)
Décision de travail : $\lambda_{cov}=3$ devient le poids standard (`TrainConfig.intent_cov_weight` passe de 10 à 3 ; la pénalité reste activée par `intent_decorrelate=cov`), et $K$ doit être plus grand que le nombre réel de classes, pour que la convergence ne soit pas imposée par le nombre de slots (avec $K=5$ pour 5 classes, une partition en 5 clusters peut être forcée par la capacité). Suite `configs/suites/toy_cov_k10.yml` : jouet, $K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, les 10 graines de `toy_rho` (0 à 9), arrêt à la convergence (règle sans labels, `min_epochs` 40) ou à 300 epochs.

| graine | epochs | convergé | ACC finale | clusters finaux | part des 100 dernières epochs avec ACC $\ge$ 0,95 | clusters utilisés (min-max, 100 dernières epochs) | images qui changent de cluster par epoch |
|---|---|---|---|---|---|---|---|
| 0 | 300 | non | 1,00 | 5 | 0,22 | 4-10 | 0,086 |
| 1 | 300 | non | 1,00 | 5 | 0,04 | 5-10 | 0,197 |
| 2 | 300 | non | 0,81 | 8 | 0,19 | 4-10 | 0,115 |
| 3 | 300 | non | 0,71 | 9 | 0,23 | 5-10 | 0,153 |
| 4 | 298 | oui | 1,00 | 5 | 0,66 | 5-7 | 0,061 |
| 5 | 300 | non | 0,70 | 10 | 0,04 | 5-10 | 0,268 |
| 6 | 300 | non | 0,80 | 10 | 0,21 | 5-10 | 0,137 |
| 7 | 300 | non | 0,89 | 9 | 0,15 | 3-10 | 0,135 |
| 8 | 300 | non | 0,86 | 9 | 0,13 | 3-10 | 0,207 |
| 9 | 300 | non | 0,83 | 9 | 0,42 | 5-10 | 0,105 |

Moyenne : ACC 0,86 $\pm$ 0,11, clusters utilisés 7,9 $\pm$ 2,0, **1 graine sur 10 converge** (graine 4, à l'epoch 298). Rappel : à $K=5$, `fix` convergeait sur 9 graines sur 10 (5.17 et 5.18, avec $\lambda=10$ ; $\lambda=3$ n'a été testé qu'à $K=5$ sur les graines 3, 5 et 8, toutes à ACC 1,00).

* **Les partitions finales sont pures, mais trop fines.** Les 10 partitions finales ont une pureté de 1,00 (aucune fusion de classes, y compris la graine 5 qui fusionnait à $K=5$) ; l'échec a changé de nature : les classes sont éclatées en plusieurs clusters (de 0 à 4 classes coupées en au moins deux clusters de plus de 10 % de la classe, selon la graine ; 0 pour les graines 0, 1 et 4 qui finissent à 5 clusters), avec quelques très petits clusters (4 à 30 images). Les graines 0 et 1 finissent à 5 clusters et ACC 1,00 mais seulement par passage à l'epoch 300 : la figure montre un nombre de clusters qui oscille entre 4 et 10 sur toute la durée.
* **L'ACC sous-estime le défaut.** L'ACC est un appariement un-à-un, elle chute quand une classe est coupée ; la mesure pertinente ici est le nombre de clusters et sa stabilité.
* **Sonde de la reward (graines 0, 3 et 5 ; labels pour l'évaluation seulement).** Avec les 5 classes présentes (summary = centre de classe) et 5 slots libres, la reward et la politique rattachent chaque image à son slot de classe avec une probabilité de 1,00 et n'ouvrent jamais (probabilité d'ouvrir 0,00). Ce n'est donc pas la reward finale qui pousse à ouvrir quand tout est en place. En revanche, avec une classe répartie sur deux slots (summaries = moyennes de deux moitiés de la classe), la reward donne exactement 0,50 / 0,50 : ces deux summaries sont identiques puisque $m$ est quasi constant dans une classe (compacité de 5.18), donc la reward ne peut pas préférer l'un des slots, et **rien dans l'objectif ne pousse à regrouper deux slots d'une même classe**. À $K=5$ c'est la capacité qui force le regroupement ; à $K=10$ plus rien ne le fait.
* **Hypothèses non testées** pour l'origine des découpages pendant l'entraînement : (1) `rho` (terme anti-effondrement : entropie du marginal de la reward sur les slots, qui récompense la répartition sur plusieurs slots) et `neg_floor` (répulsion minimale) peuvent favoriser les découpages quand $K$ est grand ; (2) l'exploration par échantillonnage de la politique ouvre des slots dans les états transitoires et rien ne les referme. Un run avec `rho=0` et `neg_floor=0` à $K=10$ trancherait (1).
* Limites : un run par graine, un seul $K$ (10), un seul $\lambda$ (3), pas de comparaison à $K=5$ sur les 10 graines avec $\lambda=3$ ; le critère de convergence (partition stable à 1 % près pendant 10 epochs) est strict, mais aucune graine hors la graine 4 ne s'en approche (changement de partition moyen entre 6 et 27 % par epoch).

### 5.20 $K=10$ sans `rho` ni plancher de répulsion (2026-10-15)
Test de l'hypothèse (1) de 5.19 : `rho` et `neg_floor` favorisent-ils les découpages quand $K$ dépasse le nombre de classes ? Suite `configs/suites/toy_cov_k10_norho.yml` : identique à `toy_cov_k10` (jouet, $K=10$, $\lambda_{cov}=3$, mêmes 10 graines, arrêt à la convergence ou à 300 epochs) avec `rho=0` et `neg_floor=0`.

| graine | epochs | convergé | ACC finale | clusters finaux | pureté finale | part des 100 dernières epochs avec ACC $\ge$ 0,95 | clusters utilisés (min-max) | images qui changent de cluster par epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | non | 0,85 | 8 | 1,00 | 0,03 | 4-10 | 0,206 |
| 1 | 300 | non | 0,91 | 9 | 1,00 | 0,14 | 4-10 | 0,126 |
| 2 | 300 | non | 0,63 | 10 | 1,00 | 0,20 | 5-10 | 0,173 |
| 3 | 300 | non | 0,75 | 9 | 1,00 | 0,07 | 5-10 | 0,215 |
| 4 | 300 | non | 0,40 | 2 | 0,40 | 0,37 | 2-10 | 0,114 |
| 5 | 300 | non | 0,83 | 10 | 1,00 | 0,10 | 5-10 | 0,169 |
| 6 | 300 | non | 0,60 | 3 | 0,60 | 0,03 | 2-10 | 0,219 |
| 7 | 70 | oui | 1,00 | 5 | 1,00 | 0,86 | 1-5 | 0,026 |
| 8 | 300 | non | 0,71 | 10 | 1,00 | 0,10 | 1-10 | 0,219 |
| 9 | 300 | non | 0,84 | 9 | 1,00 | 0,09 | 5-10 | 0,217 |

Moyenne : ACC 0,75 $\pm$ 0,17, clusters utilisés 7,5 $\pm$ 2,9, **1 graine sur 10 converge** (la graine 7, en 70 epochs). Référence avec `rho` et plancher (5.19) : ACC 0,86 $\pm$ 0,11, clusters 7,9 $\pm$ 2,0, 1 graine sur 10 (la graine 4, à l'epoch 298).

* **`rho` et le plancher ne sont pas la cause des découpages.** Sans eux, le comportement est le même : 9 graines sur 10 ne convergent pas, le nombre de clusters monte jusqu'à 10 sur les 100 dernières epochs, et les 7 partitions finales à plus de 5 clusters sont pures et éclatées (aucune fusion de classes, 2 à 4 classes coupées chacune). La graine qui converge n'est pas la même (7 au lieu de 4) ; avec un run par graine, ce changement de graine n'est pas interprétable.
* **Retirer `rho` et le plancher n'améliore rien et dégrade légèrement.** L'ACC moyenne baisse (0,75 contre 0,86), surtout parce que les graines 4 et 6 terminent sur 2 et 3 clusters avec des classes fusionnées (pureté 0,40 et 0,60) ; avec les deux termes, aucune partition finale n'était impure. Ce n'est qu'un instantané à l'epoch 300 : ces deux graines passent aussi par 10 clusters pendant la même période (min-max 2-10), donc la différence est surtout la dispersion du nombre de clusters. Écart sur 10 graines, un run chacune : indice, pas preuve.
* Reste donc l'hypothèse (2) de 5.19 : rien dans l'objectif ne referme deux slots d'une même classe (la reward ne les distingue pas, 5.19), et la politique échantillonne. Non testé.

### 5.21 Pourquoi la graine 4 converge-t-elle à $K=10$ ? Formes, couleurs, formation (2026-10-15)
Question : qu'est-ce qui distingue la seule graine convergée de `toy_cov_k10` (graine 4) des graines qui ne convergent pas (3 et 5) ? Analyse sur les runs existants (20 runs de 5.19 et 5.20), sans nouveau run ; labels utilisés pour l'évaluation seulement. Figures : `results/toy_cov_k10/clusters_seed{3,4,5}.png` (galeries d'images par cluster) et `results/toy_cov_k10/shapes_by_seed.png` (formes des 5 classes par graine).

* **Les clusters de la graine 4 sont 5 clusters purs de 300 images, un par classe, avec tous les fonds mélangés dans chaque cluster.** Dans les graines 3 et 5, les classes sont coupées en sous-clusters qui suivent la couleur ou la luminosité du fond (le nuisance), pas la forme.
* **Couleurs : pas de différence.** Le fond est tiré uniformément dans $[0\,;0{,}6]^3$ pour toutes les graines. Dans une classe, la couleur moyenne du fond se prédit linéairement avec $R^2$ de 0,84 à 0,97 depuis $m$ et environ 1,00 depuis $z$, sur les 20 runs, convergés ou non (graine 4 avec `rho` : 0,94). La fuite du fond est générale et ne sépare pas les graines. (À l'échelle de $10^{-3}$ de la distance entre classes : voir 5.23.)
* **Formes : pas de propriété qui sépare.** La graine 4 a les formes les plus blanches (part de blanc 0,50 à 0,62, contre 0,25 à 0,69 ailleurs), donc le moins de fond ; sur 10 graines, plus de blanc va avec moins de clusters (corrélation de rang $-0{,}3$ à $-0{,}7$ selon la condition), non significatif et non causal. La graine qui converge sans `rho` (graine 7) a des formes ordinaires, et les graines 4 et 7 échouent chacune dans l'autre condition.
* **Ce qui diffère : la formation.** Dans 19 runs sur 20, le nombre de clusters dépasse 5 dès les epochs 10 à 18 et reste au-dessus de 5 pendant 72 à 100 % des epochs (après l'epoch 40). La seule exception est la graine 7 sans `rho` : 5 clusters dès l'epoch 12, jamais plus, convergence à l'epoch 70. La graine 4 avec `rho` passe par 8-9 clusters, ne revient à 5 que tout à la fin (5 à 7 clusters et ACC $\ge$ 0,95 pendant 66 % des 100 dernières epochs) et converge à l'epoch 298.
* **Comparaison des 20 runs.**

| condition | graines convergées | clusters (moyenne des 100 dernières epochs, moyenne sur graines) | part des epochs à plus de 5 clusters |
|---|---|---|---|
| avec `rho` et plancher (5.19) | 1/10 (graine 4) | 7,4 | 89 % |
| sans (5.20) | 1/10 (graine 7) | 7,5 | 86 % |

Sans `rho`, deux graines (4 et 6) finissent à 2 et 3 clusters avec des classes fusionnées ; avec `rho`, aucune partition finale n'est impure.
* **Lecture.** Aucune propriété des données ne sépare les graines qui convergent ; la convergence ressemble à un accident de formation (le placement initial donne directement la bonne partition, ou y revient de justesse). Un mécanisme qui referme les doublons, ou qui évite d'en créer pendant la formation, manque. Hypothèse (non testée) pour le bruit tardif du nombre de clusters : l'argmax bascule entre des slots doublons indiscernables.

### 5.22 $K=10$ avec orbites expertes idéales (oracle, 2026-10-15)
Question : l'imperfection de l'expert (deux images d'une même classe dans deux orbites différentes, sans signal d'attraction entre elles) est-elle ce qui empêche la convergence à $K=10$ ? Suite `configs/suites/toy_cov_k10_link.yml` : identique à `toy_cov_k10` ($K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, 10 graines, arrêt à la convergence ou à 300 epochs) avec `expert_oracle_link=1,0`.
* **DIAGNOSTIC ORACLE, jamais une méthode** : il utilise les vraies classes pour construire les trajectoires expertes (`run.yml` : `uses_labels_for_training: true`). Dans chaque épisode, toutes les orbites d'une même classe forment un seul groupe : même slot, mouvement forcé (rejoindre le slot du jumeau) pour les orbites suivantes, pas de répulsion à l'intérieur d'une classe. La cible de $L_\psi$ reste l'orbite (seuls le slot, le type de mouvement et la répulsion passent à la classe). La règle d'arrêt reste sans labels.

| graine | epochs | convergé | ACC finale | clusters finaux | pureté finale | part des 100 dernières epochs avec ACC $\ge$ 0,95 | clusters utilisés (min-max) | images qui changent de cluster par epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | non | 0,92 | 6 | 0,96 | 0,63 | 5-10 | 0,055 |
| 1 | 300 | non | 0,75 | 9 | 1,00 | 0,05 | 5-10 | 0,211 |
| 2 | 300 | non | 0,90 | 8 | 1,00 | 0,00 | 7-10 | 0,235 |
| 3 | 300 | non | 0,79 | 9 | 1,00 | 0,05 | 6-10 | 0,185 |
| 4 | 300 | non | 0,91 | 8 | 1,00 | 0,08 | 5-10 | 0,171 |
| 5 | 300 | non | 0,75 | 9 | 1,00 | 0,06 | 5-10 | 0,247 |
| 6 | 300 | non | 0,89 | 10 | 1,00 | 0,17 | 5-10 | 0,103 |
| 7 | 300 | non | 0,92 | 6 | 1,00 | 0,30 | 5-10 | 0,134 |
| 8 | 300 | non | 0,63 | 10 | 1,00 | 0,08 | 5-10 | 0,239 |
| 9 | 300 | non | 0,90 | 7 | 1,00 | 0,03 | 5-10 | 0,231 |

Moyenne : ACC 0,84 $\pm$ 0,10, clusters utilisés 8,2 $\pm$ 1,4, **0 graine sur 10 converge**. Figure : `results/toy_cov_k10_link/summary.png`.

| condition (10 graines) | convergées | ACC finale | clusters (100 dernières epochs) | epochs à plus de 5 clusters | $L_{pos}$ final | twin_acc final |
|---|---|---|---|---|---|---|
| `toy_cov_k10` (avec `rho`, plancher) | 1 | 0,86 | 7,4 | 89 % | 0,274 | 0,83 |
| `toy_cov_k10_norho` | 1 | 0,75 | 7,5 | 86 % | 0,276 | 0,83 |
| `toy_cov_k10_link` (oracle) | 0 | 0,84 | 8,0 | 94 % | 0,000 | 1,00 |

* **Les orbites idéales n'améliorent pas la convergence** : aucune graine ne converge, le nombre de clusters reste à 8,0 en moyenne (7,4 sans oracle), et 94 % des epochs ont plus de 5 clusters. La tâche experte est résolue parfaitement ($L_{pos}$ = 0,000, twin_acc = 1,00 contre 0,83), sans effet sur l'apprenant : l'imperfection de l'expert n'est donc pas la cause de l'absence de convergence à $K=10$.
* **Sonde de la reward** (mémoire propre : les 5 classes présentes dans les slots 0 à 4 avec comme summary la moyenne de $m$ sur toute la classe, sur 20 images ou sur 5 images, slots 5 à 9 vides ; 10 graines de `toy_cov_k10` et de `toy_cov_k10_link`) : la probabilité de rejoindre le slot de sa classe est de 1,00 et celle d'ouvrir de 0,00, dans les 3 variantes et dans les deux suites. La reward finale ne pousse pas à ouvrir quand les slots sont propres et c'est vrai avec ou sans orbites idéales (complète la sonde de 5.19, qui n'avait que des summaries de classe entière).
* **Erreurs de placement (évaluation).** « Ouvre à tort » est rare dans les trois conditions (3,6 à 9,5 par run et par phase). « Rejoint à tort » (rejoindre un slot dont la classe dominante est autre) par run, epochs $<30$ puis $\ge 30$ : `k10` 4 536 puis 9 748 ; `k10_norho` 4 174 puis 3 850 ; `k10_link` 6 210 puis 2 022. Attention à la définition (`cibirl/diagnostics.py`) : ouvrir un slot est jugé « à tort » seulement si un slot de dominante identique existe déjà ; pendant la formation, quand les slots sont encore mélangés, ouvrir un nouveau slot est donc jugé correct même s'il crée un doublon. Cohérent avec l'hypothèse que les doublons naissent dans cette phase mélangée et que rien ne les referme ; non testé directement.
* **Reste donc côté apprenant** : la formation (5.21) et l'absence de mécanisme qui regroupe deux slots d'une même classe (5.19). Limites : un run par graine ; l'oracle mêle attraction entre orbites et résumés experts plus gros (comme en 5.2) ; $L_\psi$ reste au niveau de l'orbite ; le test ne dit pas ce qu'un relinker de phase 2 réel (non parfait) donnerait.

### 5.23 Intention $m$ : les images d'une même classe sont déjà confondues, mais deux slots d'une classe sont à égalité (2026-10-16)
Question (suite de 5.22) : si l'intention et la représentation sont correctes, pourquoi des images censées être ensemble ne le sont-elles pas, même avec des orbites expertes idéales ? Analyse sans nouveau run, sur les fichiers déjà sauvegardés (`intents.npy`, `preds.npy`, `checkpoint.pt`) de runs de `toy_cov` (graines 3 et 5), `toy_cov_k10` (3, 4, 5) et `toy_cov_k10_link` (2, 5, 7) ; labels pour l'évaluation seulement. Figure : `results/toy_cov_k10_link/intent_scales.png`. Rappel : `m_dim` est resté à 16 ; ce que $L_{cov}$ a changé n'est pas la dimension de $m$ mais son rang effectif.

* **$m$ est quasi une fonction de la classe.** Part de la variance de $m$ qui n'est pas expliquée par la classe ($1-\eta^2$) : de $5\times10^{-7}$ à $2\times10^{-5}$ ; pour $z$ : de $5\times10^{-6}$ à $3\times10^{-4}$. Distances dans $m$ : entre classes de 3,2 à 5,6 ; entre deux images d'une même classe de 0,002 à 0,013 ; entre deux vues d'une même image (orbite, fond redessiné) de 0,003 à 0,019 ; entre les centres de deux sous-clusters d'une même classe de $10^{-4}$ à $4\times10^{-3}$. La séparation intra-classe est de l'ordre de $10^{-3}$ de la séparation entre classes, et du même ordre que l'écart entre deux vues d'une même image : $m$ est invariant au fond au bruit près. Le fond reste lisible linéairement (5.21, $R^2$ 0,84 à 0,97), mais à cette échelle de $10^{-3}$ ; ce n'est pas lui qui sépare les clusters aux yeux de l'apprenant.
* **Ce que $L_{cov}$ a changé.** Sans $L_{cov}$ (6 graines de `toy_cov`, $K=5$), la première valeur propre de la covariance de $m$ porte 100 % de la variance (rang 1) et les deux classes les plus proches sont en moyenne à 0,06 (minimum 0,00) : les classes se superposent sur une droite, ce qui explique les fusions d'avant. Avec $L_{cov}$ : quatre directions de poids proches (0,30 / 0,26 / 0,23 / 0,21 à $K=10$, $\lambda=3$), les 5 classes forment un simplexe régulier de $\mathbb{R}^4$, distance minimale entre deux classes 3,98 en moyenne (minimum 2,89 ; 4,01 et 2,93 avec orbites idéales). C'est le rang maximal possible pour un code de 5 classes. Le défaut est donc passé de « classes superposées » à « une classe dans deux slots ».
* **Deux slots d'une même classe sont exactement à égalité pour la reward** (14 paires de slots doublons dans 4 runs : `k10` graines 3 et 5, `k10_link` graines 2 et 5). Les summaries des deux slots diffèrent de $10^{-4}$ à $10^{-3}$ (une seule paire à 0,17, graine 3 classe 2, avec un petit slot de 65 images). L'écart de logit de la reward entre les deux slots vaut 0,00 en moyenne (au plus 0,04) avec un écart-type de 0,00 sur les images de la classe, contre 19 à 70 d'écart avec les autres slots occupés : la reward est saturée pour la classe et indifférente entre les doublons. La part des images qui préfèrent un des deux slots va de 0,27 à 0,58 selon la paire, sans lien avec le slot où elles sont : ce sont des égalités départagées par du bruit. Les tailles des deux slots sont proches de 50/50 (150/150, 162/138, 152/148, 160/140), ce qui est ce que donnerait un tirage au hasard sur une égalité.
* **Lecture (non testée directement).** Les orbites idéales améliorent un point que la représentation avait déjà résolu (séparer les classes), et sont neutres sur ce qui manque : un état où deux slots portent la même classe n'apparaît jamais dans les épisodes experts (une mémoire d'épisode a un slot par groupe), donc rien n'apprend à la reward à préférer l'un des deux. À $K=5$ la capacité forçait la fermeture ; à $K=10$ une égalité exacte n'a aucune raison de se défaire (un slot de 150 images ne se vide pas par tirage au hasard). Les pistes à discuter portent donc sur le départage des égalités entre slots d'une même classe, pas sur l'expert ni sur la représentation.
* Limites : 8 runs sur 3 conditions pour les distances de $m$, 4 runs pour les écarts de reward entre doublons ; l'invariance est mesurée avec un seul tirage de fond par image ; je n'ai pas séparé le rôle de la reward et celui de la politique (les deux sont à égalité d'après la sonde de 5.19).

### 5.24 A priori CRP sur les placements (2026-10-16)
Suite de 5.23 : à égalité de reward entre deux slots d'une même classe, rien ne départage les deux ; `rho` n'y change rien (il porte sur les indices de slots des états experts, pas sur les slots de l'apprenant). On remplace l'a priori implicite de la politique (uniforme sur les slots valides, via le bonus d'entropie) par un a priori explicite sur les partitions, le processus du restaurant chinois (CRP).

* **Définition.** Pour l'image en cours de placement, avec $n_k$ la taille du slot $k$ sans cette image, $n$ le nombre d'images placées et $\alpha>0$ la concentration :
$$\pi_0(k\mid s)=\frac{n_k}{n-1+\alpha}\ \text{(slot occupé)},\qquad \pi_0(\text{ouvrir}\mid s)=\frac{\alpha}{n-1+\alpha}.$$
Un slot dont l'image est le seul membre compte comme un nouveau cluster (masse $\alpha$). La politique devient $\pi(a\mid s)\propto\exp(\mathrm{score}_\omega(s,a))\,\pi_0(a\mid s)$ : $\log\pi_0$ est un décalage additif fixe des logits, utilisé pour l'échantillonnage, le ratio PPO, l'entropie, $L_{inv}$, la phase teacher et `predict` (chaque image y est retirée de son propre slot). Aucun label n'intervient, l'a priori ne lit que les tailles de slots.
* **Pourquoi un décalage des logits et pas une reward shapée.** La politique n'a aucune entrée de taille de slot (entrées : $z$, $m$, summaries, indicateurs de slots vides) : elle ne pourrait pas représenter une reward qui dépend de $\log n_k$, et elle ne peut pas défaire le décalage. Où la reward est indifférente (scores égaux) l'a priori seul décide, en proportion des tailles ; ailleurs les écarts de logit réels (19 à 70, 5.23) écrasent $\log n_k$ (environ 5).
* **Code.** `cibirl/memory.py` : `crp_log_prior` ; `cibirl/policy.py` : `log_probs` et `act` prennent `log_prior` ; `cibirl/train.py` : `TrainConfig.slot_prior` (`uniform` par défaut = ancien comportement, `crp`) et `crp_alpha` (1,0), `Rollout.log_prior`, branchement dans `learner_rollout`, `policy_step`, `distill_step`, `_invariance` et `predict`. Tests : `tests/test_slot_prior.py` (masses exactes, retrait de l'image, départage des égalités, échantillonnage proportionnel aux tailles, un balayage de re-placement qui conserve les proportions au lieu de revenir à 50/50 sous a priori uniforme, `predict`) ; 166 tests passent au total. Le défaut est bit-identique à l'ancien code (même hash des métriques et de la partition sur un entraînement de 4 epochs avec $L_{cov}$, `rho` et plancher).
* **Suite `configs/suites/toy_cov_k10_crp.yml`** : identique à `toy_cov_k10` ($K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, 10 graines, arrêt à la convergence sans labels ou à 300 epochs) avec `slot_prior=crp` et $\alpha=1$ ou $\alpha=10$.

Résultat de la partition évaluée (argmax de la politique, a priori inclus) : **20 runs sur 20 convergent, ACC 1,000, NMI 1,000, 5 clusters**, en 55.4 epochs en moyenne pour $\alpha=1$ et 61.1 pour $\alpha=10$ (contre 1 graine sur 10 dans 5.19 et 0 sur 10 dans 5.22, à $K=10$). Pas de différence visible entre les deux $\alpha$ à ce niveau.

Mais le même run lu sans l'a priori à l'inférence et l'état de la mémoire donnent une autre image (`results/toy_cov_k10_crp/summary.png`) :

| graine | $\alpha=1$ : epochs | clusters sans a priori | slots mémoire | classes coupées en mémoire | $\alpha=10$ : epochs | clusters sans a priori | slots mémoire | classes coupées en mémoire |
|---|---|---|---|---|---|---|---|---|
| 0 | 41 | 8 | 9 | 1 | 50 | 9 | 10 | 2 |
| 1 | 51 | 6 | 6 | 1 | 51 | 6 | 6 | 0 |
| 2 | 54 | 6 | 6 | 1 | 90 | 7 | 9 | 2 |
| 3 | 61 | 6 | 6 | 1 | 60 | 9 | 9 | 2 |
| 4 | 71 | 9 | 9 | 3 | 47 | 5 | 6 | 1 |
| 5 | 53 | 7 | 7 | 0 | 52 | 10 | 10 | 2 |
| 6 | 85 | 7 | 7 | 2 | 91 | 8 | 8 | 2 |
| 7 | 45 | 6 | 7 | 2 | 73 | 6 | 6 | 1 |
| 8 | 46 | 7 | 8 | 3 | 50 | 8 | 10 | 2 |
| 9 | 47 | 6 | 6 | 1 | 47 | 8 | 10 | 3 |

Moyennes ($\alpha=1$ puis $\alpha=10$) : clusters sans a priori à l'inférence 6.8 et 7.6 (ACC 0.91 et 0.90) ; slots occupés en mémoire à l'arrêt 7.1 et 8.4 ; classes coupées en mémoire (au moins deux slots de plus de 10 % de la classe) 1.5 et 1.7, 18 runs sur 20 avec au moins une classe coupée ; pureté de la mémoire 0,99 à 1,00.

* **L'effet est surtout un départage à la lecture.** Sur les 20 runs, la sortie est juste parce que l'argmax envoie chaque image dans le plus gros des slots doublons (maximum a posteriori sous le CRP), pas parce que la mémoire s'est refermée : sans l'a priori à l'inférence, la même politique donne 5 à 10 clusters (ACC 0,77 à 1,00), comme avant.
* **L'a priori change quand même la dynamique de formation.** Moyennes sur les 10 graines : $K_{eff}$ de la mémoire aux epochs 20 à 30 : 6,3 ($\alpha=1$) et 6,8 ($\alpha=10$) contre 8,1 sans a priori ; slots occupés 7,6 et 8,3 contre 9,9 ; images qui changent de slot par epoch (epochs 40 à 50) 0,19 contre 0,35. $K_{eff}$ passe par environ 4,8 à l'epoch 12 avec l'a priori (valeur la plus proche de 5) puis remonte à 6-7. $\alpha=1$ ouvre moins que $\alpha=10$, dans le sens attendu.
* **Aucune fermeture des doublons déjà formés sur la durée observée.** $K_{eff}$ passe de 6,27 (epochs 20 à 30) à 6,04 (epochs 40 à 50) pour $\alpha=1$, de 6,80 à 6,76 pour $\alpha=10$ (8,08 à 8,06 sans a priori). C'est compatible avec la dérive neutre attendue : à reward égale, la taille d'un slot suit une marche sans dérive, de temps de fixation de l'ordre de $n/2$ epochs pour un slot de $n$ images (estimation par diffusion, non testée), soit un ordre de grandeur de 150 epochs pour 300 images, plus que la durée de ces runs.
* **Limite de la règle d'arrêt.** La règle (partition argmax stable, nombre de clusters constant, plateau des pertes) ne regarde que la partition évaluée, qui inclut maintenant l'a priori : elle s'arrête dès `min_epochs` (41 à 91 epochs) sans voir les doublons de la mémoire. La convergence déclarée est donc celle de la sortie, pas celle de la mémoire.
* **Non testé** : l'effet de longue durée sur la mémoire (runs sans arrêt anticipé jusqu'à 300 epochs), une règle d'arrêt qui regarde aussi la mémoire, le choix de $\alpha$. Un run par graine et par $\alpha$.

### 5.25 Convergence de la mémoire, dérive du CRP, exposant $\nu$ et $\alpha=0{,}1$ (2026-10-16)
Suite de 5.24 : la sortie était juste (argmax avec a priori) mais la mémoire gardait des slots doublons, et la règle d'arrêt ne regardait que la sortie. Trois choses faites ensemble, mesurées sur 5 graines (0 à 4), $K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, un run par graine et par condition.

* **Critère de convergence mémoire** (`cibirl/convergence.py`, option `stop.memory: true`, désactivée par défaut). $\text{mem\_gap}$ = part des images qui ne gardent pas leur cluster entre la partition de la mémoire (placements échantillonnés en fin d'epoch) et la partition de sortie, au renommage près (`partition_change`). Le run s'arrête quand la règle de 2 est vraie et que $\text{mem\_gap}\le 0{,}01$ sur toute la fenêtre. C'est équivalent à « reconstruire la mémoire depuis la sortie ne change plus rien ». Deux slots d'une classe que la sortie fusionne mais que la mémoire garde séparés donnent un $\text{mem\_gap}$ grand. `mem_gap` et `mem_excess` ($=K_{mém}-K_{sortie}$, nombres de slots non vides) sont loggés à chaque epoch dans tous les cas ; `final_metrics.json` contient `output_converged_epoch` (règle sans mémoire) et `memory_converged_epoch`. Aucun label.
* **Exposant $\nu$ sur l'a priori** (`TrainConfig.crp_power`, 1,0 par défaut = 5.24, bit-identique). La politique devient $\pi(a\mid s)\propto e^{\,s_\omega(s,a)}\,\pi_0(a\mid s)^{\nu}$, avec $s_\omega$ le score de la politique et $\pi_0$ le prior CRP de 5.24, renormalisé. À une constante près, $\log P(\text{partition})\approx K\log\alpha-n\,H(C)$ ($K$ slots occupés, $n$ images, $H(C)$ entropie des tailles de clusters ; pour une affectation déterministe $H(C)=I(X;C)$) : $\nu$ pondère la compression de la variable cluster. À égalité de scores, la part d'un slot évolue en proportion de $(n_a/n_b)^{\nu}$ : $\nu=1$ est une martingale (dérive neutre), $\nu>1$ est un retour positif vers le plus gros slot. Tests : 173 passent au total (7 nouveaux : critère mémoire, exposant, balayage de re-placement avec $\nu=1$ qui garde la part et $\nu=2$ qui l'envoie à plus de 0,8).
* **Suite `configs/suites/toy_cov_k10_mem.yml`** (arrêt mémoire, 300 epochs au plus), trois conditions qui ne changent chacune qu'une chose : `mem_a1` ($\alpha=1$, $\nu=1$ : le temps seul), `mem_nu2` ($\nu=2$), `mem_a01` ($\alpha=0{,}1$). Figure : `results/toy_cov_k10_mem/summary.png`.

| graine | $\alpha=1$ : sortie / mémoire (epoch), clusters, ACC | $\nu=2$ : sortie / mémoire, clusters, ACC | $\alpha=0{,}1$ : sortie / mémoire, clusters, ACC |
|---|---|---|---|
| 0 | 41 / 120, 5, 1,0 | 70 / 70, 3, 0,6 | 42 / 42, 5, 1,0 |
| 1 | 51 / 216, 5, 1,0 | 53 / 53, 2, 0,4 | 40 / non (7 slots), 5, 1,0 |
| 2 | 54 / 289, 4, 0,8 | 40 / 42, 5, 1,0 | 70 / 70, 4, 0,8 |
| 3 | 61 / 240, 5, 1,0 | 61 / 61, 4, 0,8 | 62 / 80, 5, 1,0 |
| 4 | 71 / 171, 4, 0,8 | 67 / 76, 4, 0,8 | 43 / 277, 4, 0,8 |

Moyennes : ACC 0,92 ($\alpha=1$), 0,72 ($\nu=2$), 0,92 ($\alpha=0{,}1$) ; epochs d'arrêt 207, 60, 154 ; mémoire convergée 5 sur 5, 5 sur 5, 4 sur 5. Sorties propres (5 clusters, ACC 1,0, mémoire = sortie) : 3 sur 5, 1 sur 5, 2 sur 5.

* **Le temps seul ferme les doublons ($\alpha=1$, 5 sur 5).** La sortie convergeait à 41–71 epochs, la mémoire à 120–289 (médiane 216), de l'ordre de la dérive neutre estimée en 5.24 (150–200 epochs, estimation de diffusion). Le critère mémoire demande donc 2 à 5 fois plus d'epochs que la règle de sortie (2,4 à 5,4).
* **Mais la durée expose une autre panne : un cluster de classe absorbé par un autre.** $\alpha=1$ : 2 graines sur 5 finissent à 4 clusters (ACC 0,8) ; graine 2 : 5 clusters purs pendant 200 epochs, puis une salve à l'epoch 202 ($L_{neg}$ 0,10 → 0,40, churn 0,3) et une classe absorbée à l'epoch 208 ; graine 4 : sortie instable entre 4 et 5 clusters de l'epoch 38 à 148 (avec une fenêtre propre de 61 à 96, où s'est arrêté le run de 5.24), puis 4 clusters à partir de l'epoch 149. Même nombre de cas avec $\alpha=0{,}1$ (graines 2 et 4) ; $\nu=2$ : 4 sur 5. Les runs de 5.24 (arrêtés à 41–91 epochs par la règle de sortie) montraient 5 clusters sur 20 sur 20 : ce sont les mêmes runs, reproduits bit à bit (`L_pos` identique à chaque epoch pour les 5 graines communes, seule la durée change), observés plus tôt. La perte de classe arrive après leur arrêt de 5.24 (graine 2 : epoch 209 ; graine 4 : epoch 149, la graine 4 étant déjà instable avant).
* **Ce qui précède les deux pertes définitives ($\alpha=1$, graines 2 et 4), lu sur `metrics.jsonl`.** Graine 2 : 5 clusters purs de l'epoch 36 à 208, puis une salve aux epochs 203-204 ($L_{pos}$ 0,29 → 0,45 ; $L_{neg}$ 0,13 → 0,25 → 0,40 ; churn 0,05 → 0,30 ; $L_{cov}$ 0,92 → 0,75), le slot fusionné apparaît à l'epoch 209 et reste jusqu'à l'epoch 289. Graine 4 : à l'epoch 38 la mémoire contient un slot de deux classes (pureté 0,80, 20 % de placements faux) et la sortie alterne entre 4 et 5 clusters ; elle est propre de 61 à 96 et de 117 à 148, la mémoire est entièrement fermée aux epochs 146-148 (5 slots, `mem_gap` 0), puis une salve (aux epochs 147 à 150 : $L_{pos}$ 0,45 ; $L_{neg}$ 0,22 puis 0,46 ; churn 0,57 ; $K_{eff}$ 2,0) laisse un slot de deux classes de l'epoch 149 à la fin. Sur les 15 runs ($\alpha=1$ de cette suite et suite 5.26), 21 salves de $L_{neg}$ (plus de 1,8 fois la médiane des 10 epochs précédents et plus de 0,2) surviennent pendant que la sortie est propre : 6 sont suivies d'une perte de classe dans les 10 epochs, dont 5 guérissent en 1 epoch ; seule la graine 2 ($\alpha=1$) est suivie d'une perte définitive (la graine 4 a une salve moins nette, à la limite du seuil). Les salves sont donc fréquentes et en général sans suite ; leur cause n'est pas identifiée (pas de checkpoint par epoch pour la tester). La perte définitive demande une salve et un état où la reward ne s'oppose pas à la fusion (ci-dessous).
* **Les salves coïncident avec des changements de palier de $L_{cov}$ (2026-10-16, analyse de `metrics.jsonl`, sans nouveau run).** $L_{cov}$ ne dérive pas : elle reste plate puis change de palier d'un coup, et les mêmes paliers reviennent d'une graine à l'autre (environ 0,97, 0,73, 0,53, 0,35, 0,2 pour les graines 1, 2, 3 ; 2,65 puis 2,1 et 1,6-1,9 pour la graine 4 en $\alpha=0{,}1$ et en gate). Définition d'un palier : médiane des 10 epochs suivantes différente de plus de 15 % de celle des 10 précédentes, après l'epoch 30. Sur les 25 runs des suites 5.25 ($\alpha=1$ et $\alpha=0{,}1$) et 5.26, 27 paliers : 18 (67 %) ont une salve de $L_{neg}$ à $\pm2$ epochs du palier, contre 2,3 % (16 sur 702) pour des fenêtres de 5 epochs éloignées de tout palier ; 17 des 32 salves de $L_{neg}$ ont un palier de $L_{cov}$ à moins de 3 epochs. Un palier n'entraîne pas la perte : les graines 1 et 3 ($\alpha=1$) en ont 1 et 3 et finissent propres ; sur les 27 paliers, une seule perte définitive (graine 2, epoch 209), 6 pertes passagères. Graine 0 : $L_{cov}$ plate (1,25 à 1,31) mais le run s'arrête à l'epoch 120, avant tout palier. Graine 4 : $L_{cov}$ plate de l'epoch 40 à 171 (2,62 à 2,77) : sa perte (epoch 149) n'a pas de palier de $L_{cov}$ et n'est pas expliquée par ceci. Hypothèse, non testée : $L_{cov}$ pénalise les covariances hors diagonale dans la base des coordonnées de $m$, qui n'est pas invariante par rotation ; le code de classe (simplexe à 4 dimensions dans 16) se réoriente par paliers vers les axes, ce qui déplace toutes les intentions et les résumés de slots d'un coup et perturbe la reward et la politique.
* **Test de l'orientation du code autour d'un palier de $L_{cov}$ (2026-10-16, deux relances avec un checkpoint par epoch, `--save-epochs`, `scripts/analyze_code_orientation.py`).** Les relances reproduisent exactement les runs d'origine (écart maximal 0,0 sur toutes les métriques). Graine 2 ($\alpha=1$, perte définitive à l'epoch 209), epochs 195 à 212 ; témoin : graine 3 (propre), palier de $L_{cov}$ à l'epoch 155 (0,51 → 0,40), epochs 150 à 162. Les centroïdes de classe utilisent les vraies classes : c'est une analyse, jamais un critère d'entraînement. De $t-1$ à $t$, pour les centroïdes de classe de $m$ : $\Delta D$ = variation relative de la matrice des distances entre centroïdes (invariante par rotation) ; $\Delta C$ = variation relative des centroïdes centrés ; $\Delta C_R$ = la même après le meilleur alignement orthogonal (Procrustes) ; $\theta$ = plus grand angle principal entre les sous-espaces à 4 dimensions engendrés par les centroïdes en $t-1$ et en $t$ ; $S$ = déplacement moyen d'une image dans $m$ divisé par la distance moyenne entre centroïdes. **Résultat.** Hors événement : $\Delta D$ 0,02 à 0,04, $\Delta C$ 0,10 à 0,15, $\theta$ 7° à 15°. Au palier : graine 2 (epoch 202) $\Delta D=0{,}069$, $\Delta C=0{,}386$, $\Delta C_R=0{,}084$, $\theta=46{,}1^\circ$, $S=1{,}02$ ; graine 3 (epoch 155) $\Delta D=0{,}063$, $\Delta C=0{,}342$, $\Delta C_R=0{,}084$, $\theta=38{,}6^\circ$. Le palier est donc une rotation globale du sous-espace de classe (centroïdes ×3, mais l'essentiel disparaît après alignement) plus une déformation modeste (distances entre classes +6 à 10 % contre ≈3 %) ; chaque image se déplace d'environ une distance inter-classes ($S=1{,}0$ à $1{,}2$ aux epochs 202 à 204 de la graine 2 et 156-157 de la graine 3, contre 0,1 à 0,5 d'ordinaire). La part de variance du code de classe portée par ses 4 coordonnées les plus actives monte de 0,42 à 0,47 (graine 2) et de 0,52 à 0,59 (graine 3) : alignement partiel sur les axes, cohérent avec la pénalité $L_{cov}$ non invariante par rotation. **Ce que le test exclut.** L'événement est le même dans le témoin propre que dans le run qui perd une classe, avec la même amplitude : une rotation, même de 40° à 46°, ne suffit pas à perdre une classe. Une politique figée à l'epoch $t-1$ appliquée à la représentation de l'epoch $t$ donne encore ACC 1,00 dans les deux runs : la rotation ne casse pas la partition par elle-même. **Ce qui diffère (une seule comparaison, donc indicatif).** La perturbation dure ≈6 epochs dans la graine 2 ($S$ de 0,7 à 1,2 aux epochs 202 à 207, $L_{cov}$ stabilisée à 0,74 seulement à l'epoch 205) contre ≈2 dans la graine 3 ($S\approx1{,}15$ aux epochs 156-157, retour à 0,4 dès l'epoch 158). La classe est perdue à l'epoch 208-209, 6 à 7 epochs après le palier. **Hypothèse, non testée :** le palier ne perd pas la classe ; c'est la durée du désordre qui le suit (reward et politique qui rattrapent le nouveau code) qui laisse à une paire de classes le temps de se fusionner dans la mémoire, état ensuite stable (voir ci-dessous).
* **Ce n'est pas géométrique.** Pour les 6 runs fusionnés examinés (checkpoints finaux), la distance entre les centroïdes de $m$ des deux classes fusionnées (4,2 à 6,2) est dans la plage de toutes les paires de classes (par exemple graine 2 : 4,19 pour la paire fusionnée, 4,19 à 4,83 pour les 10 paires ; graine 4 en $\alpha=0{,}1$ : 5,83 pour la paire, 5,19 à 5,83 pour les paires).
* **L'état fusionné est stable parce que la reward elle-même n'y objecte pas.** Pour les images de la classe absorbée, avec comme mémoire la partition de sortie : logit de reward (ouvrir − rejoindre le slot fusionné) entre −16,4 et −0,2 dans 5 des 6 runs (−2,2 à −16,4 pour $\alpha=1$ ; −1,7 à −11,3 pour $\alpha=0{,}1$ ; −1,3 et −0,2 pour $\nu=2$ graine 3), contre +19 à +70 en faveur du bon slot quand les slots sont purs (5.23). Le résumé du slot fusionné est à mi-chemin des deux classes, un état que l'expert ne montre jamais : la reward extrapole vers « rejoindre ». À cela s'ajoute le biais de taille de l'a priori, $\nu\log(n/\alpha)$ : +6,4 ($\alpha=1$, $n=600$), +8,7 ($\alpha=0{,}1$), +12,8 ($\nu=2$). Ce biais augmente en $\log n$ ; changer $\alpha$ de $10$ ne le déplace que de $\ln 10\approx2{,}3$. (Dans le sixième run, $\nu=2$ graine 1, 4 classes dans un slot, la reward préfère ouvrir de +1,7 à +3,0 mais le biais de +14,2 l'emporte.)
* **$\nu=2$ : sous-segmentation.** Les runs restent à 1 cluster jusqu'aux epochs 25 à 31 (11 à 16 pour les autres conditions) puis s'arrêtent à 2 à 5 clusters (ACC 0,40 à 1,00) ; convergés très vite (42–76 epochs) mais faux. Le retour positif n'est pas utilisable tel quel.
* **$\alpha=0{,}1$ : plus rapide quand ça marche, pas meilleur.** Mémoire convergée à 42, 70, 80 et 277 epochs, une graine non convergée à 300 epochs (7 slots pour 5 clusters de sortie, $\text{mem\_gap}$ 0,15) ; 2 absorptions sur 5, comme $\alpha=1$.
* **Lecture.** La force de la préférence de taille n'est pas le bon levier : trop faible elle laisse dériver (temps ≈ 200 epochs), trop forte ($\nu=2$) elle gèle une partition sous-segmentée, et même à $\nu=1$ elle ne se sépare pas de la reward dans le régime ambigu (résumé mixte). Le défaut identifié est le comportement de la reward sur un slot contaminé, hors de la distribution des états experts (slots toujours purs).
* **Non testé** (pistes, non lancées) : un a priori appliqué seulement entre slots que la reward ne distingue pas (scores à $\delta$ près) plutôt que partout ; des états experts avec un slot contaminé (une orbite d'une autre classe fusionnée dans le slot, action experte = ouvrir), à discuter car cela touche l'expert ; un arrêt au premier $\text{mem\_gap}\le0{,}01$ avec garde contre l'absorption ; la reproductibilité d'une même graine (un run par cellule ici).

### 5.26 A priori CRP seulement entre slots que la politique ne distingue pas (2026-10-16)
Suite de 5.25 : on retient que $\nu>1$ (sous-segmentation) et $\alpha=0{,}1$ (pas mieux) ne conviennent pas, et on garde $\alpha=1$, $\nu=1$. L'hypothèse testée : dans le régime ambigu (un slot dont le résumé est mixte), le biais de taille de l'a priori pèse contre « ouvrir » et maintient les fusions de classes. On le restreint donc aux slots que les scores ne départagent pas.

* **Définition** (`memory.gated_prior_offset`, `TrainConfig.crp_gate` $=\delta$, négatif = désactivé = 5.24). Soit $s_a$ le score de la politique pour le slot valide $a$ et $S=\{a:\ s_a\ge\max_b s_b-\delta\}$ l'ensemble des slots à $\delta$ nats du meilleur. Le décalage de logit devient
$$\text{offset}_a=\begin{cases}\log\pi_0(a)-c & a\in S\\ 0 & a\notin S\end{cases},\qquad c=\log\!\sum_{b\in S}e^{s_b+\log\pi_0(b)}-\log\!\sum_{b\in S}e^{s_b},$$
avec $\pi_0$ l'a priori CRP de 5.24. La constante $c$ est choisie pour que l'a priori ne fasse que redistribuer la masse de $S$ : la probabilité totale de $S$ est exactement inchangée. Un slot que les scores classent clairement plus bas n'est ni favorisé ni pénalisé par sa taille. Les scores sont lus sans gradient pour définir $S$. Même fonction pour l'échantillonnage, le ratio PPO, l'entropie, $L_{inv}$, la phase teacher et `predict`.
* **Code.** `cibirl/memory.py` (`gated_prior_offset`), `cibirl/policy.py` (`PolicyNet.prior_gate`), `cibirl/train.py` (`crp_gate`, teacher et distillation). Sans gate le comportement est identique à 5.24. Tests dans `tests/test_slot_prior.py` (5 nouveaux : masse de $S$ conservée, départage inchangé à égalité, slot d'ouverture non taxé hors de $S$ avec un rapport de cotes exact, pas de gradient par le gate, entraînement avec teacher) ; 178 tests passent.
* **Suite `configs/suites/toy_cov_k10_gate.yml`** : même base que `toy_cov_k10_mem` ($K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, $\alpha=1$, $\nu=1$, arrêt avec critère mémoire, 300 epochs au plus), graines 0 à 4, $\delta=1$ et $\delta=5$ ; référence : `mem_a1` (même graines, a priori partout). Figure : `results/toy_cov_k10_gate/summary.png`.

| graine | $\delta=1$ : mémoire (epoch), slots mémoire, ACC | $\delta=5$ : mémoire (epoch), slots mémoire, ACC | référence $\alpha=1$ sans gate : mémoire, clusters, ACC |
|---|---|---|---|
| 0 | 194, 5, 1,0 | non (300), 6, 1,0 | 120, 5, 1,0 |
| 1 | non (300), 10, 1,0 | 246, 5, 1,0 | 216, 5, 1,0 |
| 2 | non (300), 9, 1,0 | non (300), 5 (fermé aux 2 derniers epochs), 1,0 | 289, 4, 0,8 |
| 3 | non (300), 10, 1,0 | 105, 5, 1,0 | 240, 5, 1,0 |
| 4 | non (300), 8, 1,0 | 237, 5, 1,0 | 171, 4, 0,8 |

* **Plus d'absorption de classe.** Les 10 runs gatés finissent à 5 clusters de sortie et ACC 1,000 (NMI 1,000), contre 2 absorptions sur 5 sans gate (test exact de Fisher unilatéral sur 0/10 contre 2/5 : $p\approx0{,}1$, échantillon petit). C'est l'effet visé : le biais de taille ne pèse plus contre « ouvrir » dans le régime ambigu.
* **$\delta=5$ referme presque tout.** Mémoire convergée dans 3 runs sur 5 (105, 237 et 246 epochs), un quatrième est fermé à l'epoch 299 (le critère exige 10 epochs consécutifs, il lui en manque), un cinquième garde un doublon à 300 epochs (6 slots, $\text{mem\_gap}=0{,}089$). **$\delta=1$ referme mal** : 1 run sur 5 (epoch 194), les 4 autres gardent 8 à 10 slots à 300 epochs.
* **Pourquoi $\delta=1$ ne referme pas : pas parce que les jumeaux sortent de $S$.** Sur les 5 runs non fermés (checkpoints finaux), l'écart de score de la politique entre slots jumeaux est 0,00 à 0,01 nat et l'écart de reward 0,000 : les jumeaux sont toujours dans $S$ et le départage par la taille y est identique à 5.24 (dérive neutre). Ce qui change est la création de doublons : slots en excès ($K_{mém}-K_{sortie}$) à l'epoch 40, moyennes sur 5 graines : 4,4 ($\delta=1$), 4,2 ($\delta=5$), 2,8 (sans gate). Hypothèse, non testée directement : sans gate le biais $\log(\alpha/n)$ taxe aussi « ouvrir » pendant la formation et limite les doublons ; avec le gate cette taxe disparaît dès que « ouvrir » est à plus de $\delta$ nats, il y a plus de doublons à refermer, et la dérive neutre est lente (plusieurs paires, parfois trois slots pour une classe). C'est la même taxe qui, dans l'état ambigu, maintenait les absorptions.
* **Limites.** 5 graines, un run par cellule. Le comportement entre $\delta=5$ et $\delta=\infty$ (= 5.25) n'est pas exploré. $\delta$ est un paramètre de plus (en nats de score de la politique), sans valeur motivée au-delà de « distingué = écart de plus de $\delta$ » ; les écarts de reward observés entre classes différentes sont de 19 à 70 nats (5.23), entre jumeaux 0.

### 5.27 Balayage de la concentration $\alpha$ du CRP, 0,5 à 1,2 (2026-10-17)
Suite `configs/suites/toy_cov_k10_alpha.yml` : même base que `toy_cov_k10_mem` ($K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, $\nu=1$), $\alpha\in\{0{,}5;\ 0{,}6;\ \dots;\ 1{,}2\}$, graines 0 à 4, 40 runs, arrêt quand la sortie est stable et que la mémoire est la sortie (`stop.memory`, `mem_tol` 0,01) ou à 400 epochs. Aucun label. Figure : `results/toy_cov_k10_alpha/summary.png`. $\alpha=1{,}0$ reproduit `mem_a1` de 5.25 sur les 5 graines (écart maximal 0 sur toutes les métriques).

| $\alpha$ | sortie juste (5 clusters, ACC 1,00) | mémoire convergée | epochs de convergence mémoire : médiane [min-max] | non convergés à 400 epochs (graine : slots mémoire, mem_gap) |
|---|---|---|---|---|
| 0,5 | 5/5 | 5/5 | 262 [61-302] | aucun |
| 0,6 | 5/5 | 4/5 | 90 [80-198] | graine 2 : 6, 0,036 |
| 0,7 | 5/5 | 5/5 | 80 [41-251] | aucun |
| 0,8 | 5/5 | 4/5 | 205 [77-319] | graine 3 : 8, 0,062 |
| 0,9 | 5/5 | 5/5 | 213 [142-279] | aucun |
| 1,0 | 3/5 | 5/5 | 216 [120-289] | aucun |
| 1,1 | 5/5 | 4/5 | 180 [47-339] | graine 1 : 7, 0,004 |
| 1,2 | 5/5 | 4/5 | 200 [112-229] | graine 0 : 7, 0,092 |

* **Sortie.** Convergée entre les epochs 40 et 80 (médiane 51) dans les 40 runs. Sortie juste dans 38 runs sur 40.
* **Mémoire.** Convergée dans 36 runs sur 40, médiane 191 epochs, de 41 à 339. Les 4 runs non convergés à 400 epochs gardent 6 à 8 slots pour 5 clusters de sortie (sortie juste dans les 4).
* **Pas de tendance en $\alpha$ pour le temps de fermeture** : les médianes (262, 90, 80, 205, 213, 216, 180, 200) varient moins que la dispersion à l'intérieur d'une même valeur (par exemple 41 à 251 pour $\alpha=0{,}7$).
* **Classes perdues** : 2 runs sur 40, tous deux à $\alpha=1{,}0$ (graines 2 et 4, 4 clusters, ACC 0,8), aucun aux 7 autres valeurs (0 sur 35). Ce sont les mêmes graines qui absorbent à $\alpha=0{,}1$ (5.25). Ni $\alpha=0{,}9$ ni $\alpha=1{,}1$ ne les perdent : la perte n'est pas monotone en $\alpha$. Le choix de $\alpha=1$ comme cas à problème est post hoc (c'est la valeur où les pertes avaient été vues) ; la fréquence des pertes est de l'ordre de 2 sur 40 à 5 graines par valeur, ce qui ne permet pas de conclure à un effet de $\alpha$.
* **Limites.** 5 graines par valeur, un run par cellule ; les pertes dépendent de la trajectoire (graine $\times$ $\alpha$), comme le montre la reproductibilité bit à bit à $\alpha$ fixé et l'absence de lien monotone entre valeurs voisines ; les 4 runs non convergés pourraient fermer au-delà de 400 epochs.

### 5.28 Protocole final : arrêt à la convergence de la sortie, puis fusion des doublons de la mémoire (2026-10-17)
Décision (suite de 5.23 à 5.27) : la convergence est celle de la **sortie** (règle sans labels de la section 2, critères 1 à 3, sans critère mémoire), puis un dernier pas, politique et encodeur figés, fusionne les slots de la mémoire qui portent deux fois un même cluster. Aucun paramètre n'est ajouté à l'entraînement.

* **Pourquoi la fusion plutôt qu'une réinsertion** (test hors entraînement, 50 checkpoints finaux de 5.25 et 5.27, politique figée, labels pour l'évaluation seulement). (A) Fusion par distance : les résumés de slots doublons sont à $\le 0{,}0021$, ceux de clusters différents à $\ge 3{,}02$ (rapport $>1400$) ; tout seuil entre 0,01 et 3 donne la même partition. (B) Reconstruire la mémoire à vide en replaçant les images une à une (argmax figé, avec le prior CRP) : 56 reconstructions exactes sur 100 (2 ordres d'images par run), 18 runs sur 50 ratés dans les deux ordres, et des fusions de classes (1 à 4 clusters) ; (B) dépend de l'ordre et n'est pas retenue.
* **Règle de fusion** (`cibirl/cleanup.py`, `TrainConfig.final_cleanup`, `cleanup_ratio`). On trie les distances entre les résumés des slots non vides ; si le plus grand rapport entre deux distances consécutives est $\ge r$ (garde $r=10$), toutes les paires sous ce saut sont des doublons et sont fusionnées (liaison simple, le slot le plus gros est gardé, résumés recalculés) ; sinon on ne fusionne rien. Aucune distance n'est à choisir : le saut est $\ge 53$ quand des doublons existent et $\le 1{,}19$ quand il n'y en a pas (voir ci-dessous), la garde 10 n'est pas ajustée. Avec 2 slots ou moins il n'y a rien à comparer et rien n'est fusionné. Puis la sortie est relue (`predict`, mémoire figée).
* **Protocole.** Suite `configs/suites/toy_cov_k10_final.yml` : jouet, $K=10$, $\lambda_{cov}=3$, `rho=0,1`, `neg_floor=0,1`, CRP $\alpha=1$, $\nu=1$, **100 graines (0 à 99)**, arrêt à la convergence de la sortie (`min_epochs` 40), `final_cleanup: true`. Aucun label dans l'entraînement ni dans l'arrêt. Résumé : `results/toy_cov_k10_final/summary.txt` et `summary.png`.

| 100 graines | résultat |
|---|---|
| convergence de la sortie (règle sans labels) | 100/100, epoch 40 à 95 (médiane 51, moyenne 53) |
| sortie juste à l'arrêt (5 clusters, ACC 1,00), avant nettoyage | 96/100 |
| slots de la mémoire, avant puis après nettoyage | 7,13 [3-10] puis 4,96 [3-6] |
| mémoire = sortie ($\text{mem\_gap}\le0{,}01$), avant puis après nettoyage | 22/100 puis 96/100 |
| runs avec doublons fusionnés / sans doublon | 79 / 21 ; 217 fusions, dont 0 entre classes différentes |
| plus grand saut : avec doublons / sans | min 53 (médiane 2458) / max 1,19 (médiane 1,03) |
| distances : doublons fusionnés / clusters distincts | max 0,094 (médiane par run 0,0018) / min 2,36 |
| sortie juste après nettoyage (lecture `predict`, protocole prévu) | **93/100** |

* **Les 7 sorties fausses.** (i) Quatre l'étaient déjà à la convergence de la sortie (graines 12, 30, 59, 94 : 4, 4, 4 et 3 clusters) : la règle d'arrêt s'arrête sur une partition stable mais sous-segmentée (5.18). (ii) Trois sont causées par le nettoyage (graines 19, 80, 81) : leur sortie était juste (5 clusters, ACC 1,00) avant. Mécanisme vérifié sur la graine 19 : la mémoire avait ses 10 slots occupés (deux par classe), donc l'action « ouvrir » n'était pas possible ; après fusion, 5 slots vides apparaissent et la politique préfère « ouvrir » à « rejoindre » pour deux classes (écart de score de la politique +54 et +29 nats, alors que l'écart de reward est $-3{,}4$ et $-3{,}0$ : la reward préfère rejoindre) ; ces images sont toutes envoyées au premier slot vide (groupe d'outliers de `predict`), qui regroupe deux classes. Parmi les 16 runs dont la mémoire était pleine avant nettoyage, 3 sont dans ce cas (0 étaient faux avant).
* **Lectures alternatives, post hoc** (calculées après avoir vu ces résultats, sur les checkpoints déjà nettoyés ; `scripts/readings_after_cleanup.py`) : sortie lue sans l'action « ouvrir » (chaque image va dans un cluster existant) : **96/100**, soit les mêmes 4 échecs qu'à la convergence et plus aucun dû au nettoyage ; partition portée par la mémoire nettoyée : 82/100 (quelques images mal placées par les placements échantillonnés de la dernière epoch, ACC 0,99 par exemple, plus les 4 sorties fausses). Le choix de la lecture sans « ouvrir » est donc post hoc ; le valider demande des graines neuves.
* **Limites.** Jouet seulement, $\alpha=1$, un run par graine. Les 4 échecs de sous-segmentation à l'arrêt (4 %, borne supérieure à 95 % de 10 % avec 4 sur 100) ne sont pas traités. La garde $r=10$ a une marge d'un facteur 5 (53 contre 10) du côté « doublons » et de 8 (10 contre 1,19) du côté « pas de doublon » sur ces 100 runs ; à l'arrêt, les distances entre doublons ($\le0{,}094$) sont plus grandes qu'en fin de longue durée ($\le0{,}0021$, 5.25).


### 5.29 Fermer la mémoire sans fausser la politique : relabel (A) et réintroduction (C) (2026-10-17)
Point de départ : les 3 sorties cassées par le nettoyage en 5.28 (graines 19, 80, 81). Relire la politique sur la mémoire fusionnée la place dans un état qu'elle n'a pas vu à l'entraînement (des slots vides alors que la mémoire y était pleine) ; elle y préfère « ouvrir ». Deux façons de fermer la mémoire sans cela, comparées à la fusion de 5.28 sur le **même entraînement** (mêmes 100 graines, chaque variante part d'une copie de la même mémoire ; l'entraînement est identique à 5.28, vérifié : mêmes chiffres pour « à l'arrêt » et « merge »). Aucun label dans l'entraînement ni dans l'arrêt.

* **A, `relabel`.** La sortie est celle lue à l'arrêt (avant la fusion) ; la fusion de 5.28 (saut, garde 10) ne sert qu'à renommer ses slots. La politique n'est jamais interrogée sur la mémoire fusionnée.
* **C, `reintroduce`** (`cibirl/cleanup.py`, `reintroduce`). Depuis la mémoire d'avant fusion, des balayages d'images dans un ordre aléatoire (graine de la config) : chaque image est retirée de son slot, puis replacée là où l'argmax de $\pi(a\mid z,m,C)$ l'envoie, avec le prior CRP de l'entraînement (image retirée de son slot) et les résumés mis à jour à chaque placement. C'est une epoch d'entraînement avec l'argmax à la place du tirage, politique figée. À égalité entre deux doublons, le prior favorise le plus gros : chaque déplacement l'agrandit, le plus petit se vide (la dérive neutre devient un cliquet). On s'arrête quand un balayage ne déplace aucune image (plafond de 20). Aucune distance, aucune garde. **La sortie de C est la partition où les balayages s'arrêtent** (mémoire = sortie par construction) ; `reintroduce_no_open` retire l'action « ouvrir » (les images sont réintroduites DANS les clusters existants), `reintroduce` la garde.
* **Suite** `configs/suites/toy_cov_k10_final_ac.yml` (100 graines, 0 à 99, mêmes réglages que 5.28, `cleanup_mode: relabel`), résumé `results/toy_cov_k10_final_ac/summary_variants.txt` (`scripts/summarize_variants.py`). « juste » = 5 clusters et ACC 1,00 ; « mémoire juste » = la partition portée par la mémoire est elle-même juste.

| 100 graines | sortie juste | mémoire = sortie | slots (moy.) | mémoire juste | perdus par rapport à l'arrêt |
|---|---|---|---|---|---|
| à l'arrêt (avant tout nettoyage) | 96 | 22 | 7,13 | 19 | |
| `merge` (5.28) | 93 | 96 | 4,96 | 82 | 19, 80, 81 |
| **A `relabel`** | **96** | 99 | 4,96 | 82 | aucun |
| C `reintroduce` | 92 | 100 | 5,15 | 92 | 18, 19, 80, 81 |
| **C `reintroduce_no_open`** | **96** | **100** | **4,95** | **96** | aucun |

* **A** garde les 96 sorties justes et corrige les 3 cassées par `merge`. La mémoire reste celle des placements échantillonnés de la dernière epoch : 82 mémoires justes, les autres ont des images mal placées (14 runs, ACC de la mémoire 0,990 en moyenne, 0,90 au minimum) plus les 4 échecs. Une seule sortie diffère de sa mémoire (graine 55, écart 0,10).
* **C avec « ouvrir »** ne ferme pas la mémoire quand elle était pleine : sur les 16 runs à mémoire pleine, 4 (graines 18, 19, 80, 81) gardent 10 slots, avec 10 images qui changent de slot à chaque balayage (plafond de 20 balayages atteint dans ces 4 runs seulement). Mécanisme : dès qu'un slot se vide, « ouvrir » redevient possible et la politique le choisit pour une dizaine d'images (même faiblesse que 5.28). Non retenue.
* **C sans « ouvrir »** converge en au plus 2 balayages dans les 100 runs (médiane de 120 images déplacées au premier, aucune au second). La mémoire fermée porte une partition **exactement juste** dans les 96 runs dont la sortie est juste (ACC 1,00 dans chacun, 5 slots), sans les images mal placées de A (96 mémoires justes contre 82). Aucune sortie perdue, aucune juste devenue fausse.
* **Les 4 échecs restants sont les mêmes pour A et C** (graines 12, 30, 59, 94 : sous-segmentation déjà présente à l'arrêt, 4, 4, 4 et 3 clusters) : ni A ni C ne les traitent, le nettoyage ne crée pas de cluster.

* **Ce que A et C ne corrigent pas.** La politique relue (`predict`) sur la mémoire fermée échoue encore dans 7 runs (4 + 19, 80, 81) pour `reintroduce_no_open` : la préférence pour « ouvrir » dès que des slots sont vides est une faiblesse de la politique hors des états vus à l'entraînement (mémoire pleine), que A et C contournent sans la corriger. Pour assigner des images NOUVELLES avec `predict` sur la mémoire fermée, ce défaut serait à traiter ; non testé.
* **Limites.** Jouet, $\alpha=1$, une graine = un run, mêmes graines (0 à 99) que 5.28. Les quatre lectures étaient prévues avant le lancement des 100 runs, mais `reintroduce_no_open` vient d'un pilote sur 5 graines (0, 12, 19, 80, 81) et de la lecture sans « ouvrir » de 5.28, faite après avoir vu ces mêmes graines : son avantage sur A (mémoire juste 96 contre 82) est à confirmer sur des graines neuves (100 à 199). Ordre aléatoire des balayages : une seule graine d'ordre par run (`cfg.seed`).

### 5.30 Validation sur 100 graines neuves : A est retenu (2026-10-18)
Règle fixée avant le lancement (en-tête de `configs/suites/toy_cov_k10_final_ac_val.yml`) : C (`reintroduce_no_open`) est retenu si sa sortie est juste dans au moins autant de runs que A **et** si sa mémoire est juste dans plus de runs que A ; sinon A. Suite `toy_cov_k10_final_ac_val` : graines 100 à 199, mêmes réglages, même code que 5.29 (aucun réglage touché). Résumé : `results/toy_cov_k10_final_ac_val/summary_variants.txt`.

| graines 100 à 199 | sortie juste | mémoire = sortie | slots (moy.) | mémoire juste | perdus par rapport à l'arrêt |
|---|---|---|---|---|---|
| à l'arrêt | 98 | 21 | 7,16 | 19 | |
| `merge` | 98 | 97 | 5,02 | 91 | aucun |
| **A `relabel`** | **98** | 97 | 5,02 | 91 | aucun |
| C `reintroduce` | 95 | 100 | 5,07 | 95 | 102, 106, 181 |
| C `reintroduce_no_open` | 97 | 100 | 4,97 | 97 | 102 |

* **Décision : A.** C sans « ouvrir » a la mémoire la plus juste (97 contre 91) mais perd une sortie juste (97 contre 98) : la condition « au moins autant de sorties justes » n'est pas remplie.
* **Cumul des 200 graines (0 à 199)**, sortie juste / mémoire juste : A 194 / 173 ; C sans « ouvrir » 193 / 193 ; `merge` 191 / 173 ; C avec « ouvrir » 187 / 187. A et C sans « ouvrir » sont à 1 run l'un de l'autre sur la sortie (194 et 193 sur 200) ; la différence nette est sur la mémoire (193 contre 173), qui n'est pas ce qu'on évalue.
* **Échecs de A** : 12, 30, 59, 94 (5.29) et 122, 152 (4 clusters, ACC 0,80, déjà faux à l'arrêt), soit 6 sur 200. Aucune fusion entre classes différentes (0 sur 214 fusions dans cette suite).
* **Graine 102** (juste à l'arrêt, perdue par C) : à l'arrêt la mémoire a 9 slots et une partition juste à 0,68 seulement ; après la fusion, 7 slots, mémoire juste à 0,82 ; les balayages de C descendent à 4 clusters (ACC 0,80) en 3 balayages. C relit la mémoire, A ne lit que la sortie : C hérite des défauts de la mémoire, A non.
* **Convergence** : 100/100 selon la règle de sortie (epoch min 40, médiane 47, max 100). C avec « ouvrir » atteint le plafond de 20 balayages dans 2 runs, C sans « ouvrir » dans aucun (au plus 3 balayages).
* **Défaut de la politique relue** : `predict` sur la mémoire fermée par C : 97/100 (comme la sortie de C ici).
* **Limites.** Jouet, $\alpha=1$, un run par graine, un seul ordre de balayage par run. La règle départage sur la sortie ; elle ne dit rien de l'usage de la mémoire fermée pour assigner des images nouvelles.

## 4. Données de suivi
`results/` est versionné dans git (pour documenter les résultats). Suites : `configs/suites/*.yml`, lancées par `scripts/run_suite.py`.
