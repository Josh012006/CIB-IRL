"""Two post-hoc analyses of the runs of configs/suites/mnist10_final.yml (docs/protocol-decisions.md 5.36). Nothing is trained.

    python scripts/analyze_mnist10.py results/mnist10_final --seeds 0 1

(1) READOUT. The clustering is read again from the intent m of every image (intents.npy) with classical algorithms, to see
    whether the information is in m although the policy does not use it: k-means (K = 10 and K = 15), a Gaussian mixture,
    Ward agglomerative clustering, and a 5-fold kNN(10) as a ceiling.
(2) CLEAN-UP RATIO. `cibirl/cleanup.py` merges twin slots when the largest jump between the sorted distances of the summaries is
    at least `ratio` (retained: 10). Here `ratio` runs over several values on the memory BEFORE the clean-up (memory_pre.pt):
    which slots are merged, ACC of the output with the slots renamed (mode `relabel`), clusters left, and the merges that join
    two different digits (labels used for evaluation only). A smaller ratio can only merge more.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cibirl.cleanup import find_twins          # noqa: E402
from eval.metrics import cluster_acc           # noqa: E402
from scripts import datasets                   # noqa: E402

RATIOS = (10.0, 9.0, 5.0, 3.0, 2.0, 1.5, 1.25, 1.1, 1.0)


def readout(m, y, K):
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    out = {"kNN(10) 5-fold": float(cross_val_score(KNeighborsClassifier(10), m, y, cv=5).mean())}
    for k in (10, K):
        out[f"k-means K={k}"] = cluster_acc(KMeans(k, n_init=10, random_state=0).fit_predict(m), y, K)
    for k in (10, K):
        out[f"GMM K={k}"] = cluster_acc(GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit_predict(m), y, K)
    out["Ward K=10"] = cluster_acc(AgglomerativeClustering(10, linkage="ward").fit_predict(m), y, K)
    return out


def ratio_sweep(run, y, K):
    mem = torch.load(run / "memory_pre.pt", map_location="cpu", weights_only=False)
    pre = np.load(run / "variants.npz")["pre"]
    rows = []
    for ratio in RATIOS:
        rep, info = find_twins(mem["summaries"], mem["counts"], ratio)
        lookup = {int(k): int(r) for k, r in rep.items()}
        pred = np.array([lookup.get(int(p), int(p)) for p in pre])
        dominant = {int(k): int(np.bincount(y[pre == k], minlength=10).argmax()) for k in np.unique(pre)}
        across = sum(1 for k, r in lookup.items() if k != r and k in dominant and r in dominant and dominant[k] != dominant[r])   # a slot empty in the output has no digit
        groups = {}
        for k, r in lookup.items():
            if k != r:
                groups.setdefault(r, []).append(k)
        detail = "; ".join(f"{dominant.get(r)} <- " + ",".join(f"{dominant.get(k)}({int((pre == k).sum())})" for k in ks)
                           for r, ks in groups.items())
        rows.append({"detail": detail, "ratio": ratio, "merged": info["merged_pairs"], "across_digits": across,
                     "clusters": int(len(np.unique(pred))), "acc": cluster_acc(pred, y, K)})
    return rows, info["jump"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    for s in a.seeds:
        run = Path(a.root) / f"final_seed{s}"
        y = datasets.build("mnist", s, digits=list(range(10)), n_per_class=300).labels.numpy()
        K = int(next(l.split(":")[1] for l in (run / "config.yml").read_text().splitlines() if l.startswith("num_slots")))
        print(f"\n=== seed {s} ===")
        print("readout from m (ACC, one-to-one matching):")
        for k, v in readout(np.load(run / "intents.npy"), y, K).items():
            print(f"  {k:16s} {v:.3f}")
        rows, jump = ratio_sweep(run, y, K)
        print(f"clean-up ratio sweep (largest jump of the distances: {jump:.2f}); mode relabel")
        print("  ratio  merged  across digits  clusters  ACC   merges: kept digit <- merged digits(size)")
        for r in rows:
            print(f"  {r['ratio']:5.2f}  {r['merged']:6d}  {r['across_digits']:13d}  {r['clusters']:8d}  {r['acc']:.3f}   {r['detail']}")


if __name__ == "__main__":
    main()
