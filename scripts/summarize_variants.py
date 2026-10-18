"""Compare the ways of closing the memory in a suite run with `final_cleanup: true` (docs 5.29): the output at the stop (pre), merge
(5.28), relabel (A), reintroduce and reintroduce_no_open (C). All variants come from the same trained model and the same memory of each
run, so they are compared run by run. The labels only EVALUATE.
Usage: python scripts/summarize_variants.py results/toy_cov_k10_final_ac   (prints, and writes summary_variants.txt there)"""
import glob
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import beta

R = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "results/toy_cov_k10_final_ac"
NAMES = ("merge", "relabel", "reintroduce", "reintroduce_no_open")
K_TRUE = 5


def main():
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(R + "/final_seed*/final_metrics.json"),
                                                                  key=lambda p: int(p.split("final_seed")[1].split("/")[0]))]
    n = len(runs)
    print("runs finis:", n)
    if not n:
        return
    ok = lambda a: a["clusters"] == K_TRUE and a["acc"] > 0.999                          # the output is right
    mem_ok = lambda a: a["memory_clusters"] == K_TRUE and a["memory_acc"] > 0.999        # the memory holds the right partition
    ep = [r["epochs_run"] for r in runs]
    print("convergées (règle de sortie): %d/%d ; epoch: min %d médiane %d max %d" % (sum(r["converged"] for r in runs), n, min(ep), np.median(ep), max(ep)))
    pre = [r["before_cleanup"] for r in runs]
    var = {v: [r["cleanup"]["variants"][v] for r in runs] for v in NAMES}
    rows = [("sortie à l'arrêt (pre)", pre)] + [(v, var[v]) for v in NAMES]
    print("\n%-24s %8s %10s %14s %9s %12s %14s" % ("", "juste", "échecs", "borne 95% sup", "mém=sortie", "slots (moy)", "mémoire juste"))
    for name, rr in rows:
        k = sum(not ok(a) for a in rr)
        print("%-24s %6d/%d %10d %14.3f %7d/%d %12.2f %12d/%d" % (name, n - k, n, k, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0,
              sum(a["mem_gap"] <= 0.01 for a in rr), n, np.mean([a["memory_clusters"] for a in rr]),
              sum(mem_ok(a) for a in rr), n))
    print("\nslots de la mémoire: pre %d-%d ; " % (min(a["memory_clusters"] for a in pre), max(a["memory_clusters"] for a in pre)) +
          " ; ".join("%s %d-%d" % (v, min(a["memory_clusters"] for a in var[v]), max(a["memory_clusters"] for a in var[v])) for v in NAMES))
    print("\npar rapport à la sortie à l'arrêt, run par run:")
    for v in NAMES:
        lost = [r["seed"] for r, a, b in zip(runs, pre, var[v]) if ok(a) and not ok(b)]
        won = [r["seed"] for r, a, b in zip(runs, pre, var[v]) if not ok(a) and ok(b)]
        print("  %-20s perdus (juste -> faux): %s ; gagnés (faux -> juste): %s" % (v, lost, won))
    print("\nsorties fausses à l'arrêt:", [r["seed"] for r, a in zip(runs, pre) if not ok(a)])
    for v in NAMES:
        print("sorties fausses, %-20s %s" % (v + ":", [r["seed"] for r, a in zip(runs, var[v]) if not ok(a)]))
    full = [i for i, a in enumerate(pre) if a["memory_clusters"] == runs[i].get("num_slots", 10) or a["memory_clusters"] >= 10]
    print("\nruns dont la mémoire était pleine à l'arrêt: %d ; sorties fausses parmi eux: " % len(full) +
          " ; ".join("%s %d" % (v, sum(not ok(var[v][i]) for i in full)) for v in NAMES))
    print("fusions entre classes différentes (éval.): %d sur %d fusions" % (sum(r["cleanup"]["merges_across_classes"] for r in runs), sum(r["cleanup"]["merged_slots"] for r in runs)))
    for v in ("reintroduce", "reintroduce_no_open"):
        sw = [a["sweeps"] for a in var[v]]
        print("\n%s: balayages moy %.1f max %d ; plafond atteint (20 balayages, une image bouge encore): %d/%d ; images déplacées au 1er balayage: médiane %d" % (
            v, np.mean(sw), max(sw), sum(s >= 20 for s in sw), n, np.median([a["moved"][0] for a in var[v]])))
        print("   la politique relue sur cette mémoire (predict): juste %d/%d" % (sum(a["predict_clusters"] == K_TRUE and a["predict_acc"] > 0.999 for a in var[v]), n))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text, end="")
    if Path(R).exists() and glob.glob(R + "/final_seed*/final_metrics.json"):
        Path(R, "summary_variants.txt").write_text(text, encoding="utf-8")
