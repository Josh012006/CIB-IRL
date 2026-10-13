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

## 3. Constats provisoires du jouet (UNE graine, à confirmer sur 5)
- Oracle sans raccourci : ACC 1,00 avec 5 clusters à 30 epochs et à la convergence (42 epochs) ; une première version du critère (sans renommage) a laissé le run continuer jusqu'à 79 epochs, où 2 classes fusionnent (4 clusters, ACC 0,80). L'entraînement long dérive donc vers des fusions : à étudier (entropie de la politique ~0, rich-get-richer ?).
- Banque `generic` + raccourci 0,9 : pas de convergence en 150 epochs, la partition oscille (ACC 0,54 à 0,79), twin_acc baisse de 0,82 à ~0,65.
- Le critère (fenêtre 5 -> 10, mesure invariante au renommage) a été ajusté en regardant ces deux runs : choix à signaler, à tester en sensibilité (fenêtre 5 / 10 / 20).

## 4. Données de suivi
`results/` est versionné dans git (pour documenter les résultats). Suites : `configs/suites/*.yml`, lancées par `scripts/run_suite.py`.
