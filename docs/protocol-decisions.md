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

## 4. Données de suivi
`results/` est versionné dans git (pour documenter les résultats). Suites : `configs/suites/*.yml`, lancées par `scripts/run_suite.py`.
