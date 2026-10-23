"""Post-hoc comparison of two runs of the MNIST-10 final suite (docs/protocol-decisions.md 5.36/5.38). Nothing is trained.

    python scripts/compare_seeds_mnist10.py results/mnist10_final --seeds 0 1

Questions: (a) where the two policies differ per digit (cluster composition), (b) whether the digit pairs are separated in the
learned intent m (linear and kNN accuracy on pairs, centroid distance / within-class spread), (c) whether the two representations
differ in their neighbourhoods (10-NN overlap) and whether the images that one seed mixes are the ones the other seed mixes.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import datasets                                      # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA   # noqa: E402
from sklearn.model_selection import cross_val_score               # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors   # noqa: E402


def pair_stats(m, y, a, b):
    ia, ib = np.where(y == a)[0], np.where(y == b)[0]
    idx = np.concatenate([ia, ib]); yy = (y[idx] == b).astype(int); X = m[idx]
    lin = cross_val_score(LDA(), X, yy, cv=5).mean()
    knn = cross_val_score(KNeighborsClassifier(10), X, yy, cv=5).mean()
    ca, cb = m[ia].mean(0), m[ib].mean(0)
    ra = np.linalg.norm(m[ia] - ca, axis=1).mean(); rb = np.linalg.norm(m[ib] - cb, axis=1).mean()
    return lin, knn, np.linalg.norm(ca - cb) / ((ra + rb) / 2)


def main():
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--seeds", type=int, nargs=2, default=[0, 1]); p.add_argument("--cross", action="store_true"); a = p.parse_args()
    if a.cross:
        return cross(Path(a.root), a.seeds)
    root = Path(a.root); y = datasets.build("mnist", 0, digits=list(range(10)), n_per_class=300).labels.numpy()
    R = {}
    for s in a.seeds:
        run = root / f"final_seed{s}"
        R[s] = dict(m=np.load(run / "intents.npy"), pred=np.load(run / "preds.npy"))
    print("== composition of the policy: digit -> clusters (count)")
    for s in a.seeds:
        print(f"seed {s}")
        for d in range(10):
            c = np.bincount(R[s]["pred"][y == d], minlength=R[s]["pred"].max() + 1)
            top = np.argsort(-c)[:4]
            print(f"  digit {d}: " + "  ".join(f"c{k}:{c[k]}" for k in top if c[k] > 0))
    print("== pair separation in m: LDA acc / kNN acc / centroid distance over mean radius")
    pairs = [(3, 5), (8, 9), (4, 9), (3, 8), (5, 8), (7, 9), (0, 6), (1, 7), (2, 7), (5, 6)]
    print("pair    " + "   ".join(f"seed{s}: lin  knn  d'  " for s in a.seeds))
    for (u, v) in pairs:
        print(f"{u}-{v}    " + "   ".join("      %.3f %.3f %.2f" % pair_stats(R[s]["m"], y, u, v) for s in a.seeds))
    print("== intrinsic geometry of m")
    for s in a.seeds:
        m = R[s]["m"]; w = np.mean([np.linalg.norm(m[y == d] - m[y == d].mean(0), axis=1).mean() for d in range(10)])
        gm = m.mean(0); b = np.mean([np.linalg.norm(m[y == d].mean(0) - gm) for d in range(10)])
        ev = np.linalg.eigvalsh(np.cov(m.T))[::-1]; pr = ev.sum() ** 2 / (ev ** 2).sum()
        print(f"  seed {s}: within-class radius {w:.3f}  centroid spread {b:.3f}  ratio {b / w:.2f}  participation ratio {pr:.1f}  m_std {m.std(0).mean():.3f}")
    print("== neighbourhoods")
    nn = {s: NearestNeighbors(n_neighbors=11).fit(R[s]["m"]).kneighbors(R[s]["m"])[1][:, 1:] for s in a.seeds}
    s0, s1 = a.seeds
    ov = np.array([len(set(nn[s0][i]) & set(nn[s1][i])) / 10 for i in range(len(y))])
    print(f"  mean 10-NN overlap between seeds: {ov.mean():.3f}")
    for s in a.seeds:
        same = (y[nn[s]] == y[:, None]).mean(1)
        print(f"  seed {s}: fraction of the 10 neighbours with the same digit: " + " ".join(f"{d}:{same[y == d].mean():.2f}" for d in range(10)))
    print("== are the mixed images the same?  (3 and 5 only: image goes to a cluster whose majority digit is the other one)")
    for s in a.seeds:
        dom = {k: np.bincount(y[R[s]["pred"] == k], minlength=10).argmax() for k in np.unique(R[s]["pred"])}
        R[s]["wrong35"] = np.array([(y[i] in (3, 5)) and dom[R[s]["pred"][i]] != y[i] for i in range(len(y))])
        print(f"  seed {s}: {R[s]['wrong35'].sum()} images of 3/5 in a cluster dominated by another digit")
    both = (R[s0]["wrong35"] & R[s1]["wrong35"]).sum()
    print(f"  in both: {both}   expected if independent: {R[s0]['wrong35'].sum() * R[s1]['wrong35'].sum() / 600:.1f}")
    # same-digit ambiguity: share of images whose kNN-majority (in m of seed s) differs from the label, per seed, 3/5 only
    for s in a.seeds:
        pr = np.array([np.bincount(y[nn[s][i]], minlength=10).argmax() for i in range(len(y))])
        bad = (pr != y) & np.isin(y, [3, 5]); print(f"  seed {s}: 3/5 images whose 10-NN majority (in m) is another digit: {bad.sum()}")



# --------------------------------------------------------------------------------------------------------------------
# Second part: is the difference in the network or in the sample of images? Each seed draws its OWN 300 images per digit
# (datasets.build_mnist), so the images of the two seeds are different and a comparison image by image is meaningless.
# Here every trained network reads BOTH samples (cross), and the raw pixels are read as a control of how hard each sample is.
#     python scripts/compare_seeds_mnist10.py results/mnist10_final --seeds 0 1 --cross
# --------------------------------------------------------------------------------------------------------------------
def cross(root, seeds):
    import torch
    from cibirl import transforms
    from cibirl.train import TrainConfig, Trainer
    torch.set_num_threads(1)
    data = {s: datasets.build("mnist", s, digits=list(range(10)), n_per_class=300) for s in seeds}
    y = data[seeds[0]].labels.numpy()
    four = [3, 5, 8, 9]

    def geometry(m):
        w = np.mean([np.linalg.norm(m[y == d] - m[y == d].mean(0), axis=1).mean() for d in range(10)])
        gm = m.mean(0); b = np.mean([np.linalg.norm(m[y == d].mean(0) - gm) for d in range(10)])
        ps = [pair_stats(m, y, u, v) for u in four for v in four if u < v]
        return w, b / w, np.mean([p[0] for p in ps]), np.min([p[2] for p in ps]), np.mean([p[1] for p in ps])
    print("== networks x samples: within-radius, spread/radius, mean LDA acc over pairs of {3,5,8,9}, min d' over those pairs, mean kNN acc")
    for ns in seeds:
        d = root / f"final_seed{ns}"
        cfg = TrainConfig.from_yaml(str(d / "config.yml")); bank = transforms.build(str(d / "transform.yml"))
        tr = Trainer(cfg, data[ns].images, bank, device="cpu", eval_labels=data[ns].labels)
        ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        for k in ("encoder", "intent"): getattr(tr, k).load_state_dict(ck[k]); getattr(tr, k).eval()
        for ds in seeds:
            m = tr.intents(data[ds].images).cpu().numpy()
            print(f"  network of seed {ns} on images of seed {ds}: radius {geometry(m)[0]:.3f}  spread/radius {geometry(m)[1]:.2f}  "
                  f"LDA {geometry(m)[2]:.3f}  min d' {geometry(m)[3]:.2f}  kNN {geometry(m)[4]:.3f}")
    print("== control: raw pixels of each sample (no learning)")
    for ds in seeds:
        X = data[ds].images.reshape(len(y), -1).numpy()
        ps = [pair_stats(X, y, u, v) for u in four for v in four if u < v]
        print(f"  images of seed {ds}: mean LDA {np.mean([p[0] for p in ps]):.3f}  kNN {np.mean([p[1] for p in ps]):.3f}  "
              f"(3-5: {pair_stats(X, y, 3, 5)[1]:.3f}  8-9: {pair_stats(X, y, 8, 9)[1]:.3f}  5-8: {pair_stats(X, y, 5, 8)[1]:.3f})")



if __name__ == "__main__":
    main()
