"""Table of a suite run with `final_cleanup: true` (docs 5.28): output at the stop, memory before/after the clean-up, merges.
The labels only EVALUATE (acc, merges_across_classes). Usage: python scripts/summarize_cleanup.py results/toy_cov_k10_final"""
import glob
import json
import sys

import numpy as np
from scipy.stats import beta

R = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "results/toy_cov_k10_final"
runs = []
for f in sorted(glob.glob(R + "/final_seed*/final_metrics.json"), key=lambda p: int(p.split("final_seed")[1].split("/")[0])):
    runs.append(json.load(open(f, encoding="utf-8")))
n = len(runs); print("runs finis:", n)
if not n: raise SystemExit
ok = lambda a: a["clusters"] == 5 and a["acc"] > 0.999
A = [r["cleanup"]["after"] for r in runs]; Bf = [r["before_cleanup"] for r in runs]
print("convergées (règle de sortie):", sum(r["converged"] for r in runs), "/", n, "; epoch de convergence: min %d médiane %d max %d" % (
    min(r["epochs_run"] for r in runs), np.median([r["epochs_run"] for r in runs]), max(r["epochs_run"] for r in runs)))
print("sortie juste avant nettoyage: %d/%d ; après: %d/%d" % (sum(map(ok, Bf)), n, sum(map(ok, A)), n))
print("mémoire = sortie (mem_gap<=0.01) avant: %d/%d ; après: %d/%d" % (sum(b["mem_gap"] <= 0.01 for b in Bf), n, sum(a["mem_gap"] <= 0.01 for a in A), n))
print("slots mémoire avant: moyenne %.2f [%d-%d] ; après: moyenne %.2f [%d-%d]" % (np.mean([b["memory_clusters"] for b in Bf]), min(b["memory_clusters"] for b in Bf), max(b["memory_clusters"] for b in Bf),
      np.mean([a["memory_clusters"] for a in A]), min(a["memory_clusters"] for a in A), max(a["memory_clusters"] for a in A)))
print("slots mémoire après == clusters de sortie: %d/%d" % (sum(a["memory_clusters"] == a["clusters"] for a in A), n))
print("fusions entre classes différentes (éval.):", sum(r["cleanup"]["merges_across_classes"] for r in runs), "sur", sum(r["cleanup"]["merged_slots"] for r in runs), "fusions")
mj = [r["cleanup"]["jump"] for r in runs if r["cleanup"]["merged_slots"]]; nj = [r["cleanup"]["jump"] for r in runs if not r["cleanup"]["merged_slots"] and r["cleanup"]["jump"] is not None]
print("runs avec doublons fusionnés: %d ; saut min parmi eux: %.0f ; runs sans doublon: %d ; saut max parmi eux: %.2f" % (
    len(mj), min(mj) if mj else float("nan"), len(nj), max(nj) if nj else float("nan")))
tw = [r["cleanup"]["threshold"] for r in runs if r["cleanup"]["threshold"] is not None]
print("distance max entre doublons fusionnés: %.4f" % max(tw) if tw else "", end="")
dmin = [min(d for d in r["cleanup"]["slot_distances"] if d > (r["cleanup"]["threshold"] or 0)) for r in runs if r["cleanup"]["slot_distances"]]
print(" ; distance min entre clusters distincts: %.2f" % min(dmin))
bad = [(r["name"], r["before_cleanup"]["clusters"], r["before_cleanup"]["acc"], r["cleanup"]["after"]["clusters"], r["cleanup"]["after"]["acc"]) for r in runs if not ok(r["cleanup"]["after"])]
print("sorties non justes après nettoyage:", bad)
k = sum(not ok(a) for a in A); print("échecs de sortie: %d/%d ; borne sup. 95%% (Clopper-Pearson): %.3f" % (k, n, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0))
