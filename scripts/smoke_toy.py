"""End-to-end smoke test on synthetic data: does the Phase 1 pipeline run, and does it find the classes?
Usage: python scripts/smoke_toy.py [--config configs/toy.yml configs/ablations/gamma0.yml] [--set seed=3] [--epochs 30]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cibirl.train import TrainConfig, Trainer      # noqa: E402
from toy_data import background_redraw, make_toy   # noqa: E402


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / max((ha * hb) ** 0.5, 1e-12))


def cluster_acc(pred, labels, k):
    conf = np.zeros((k, int(labels.max()) + 1))
    for p, l in zip(pred, labels):
        conf[p, l] += 1
    r, c = linear_sum_assignment(-conf)
    return conf[r, c].sum() / len(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/toy.yml"],
                    help="YAML files merged left to right, e.g. configs/toy.yml configs/ablations/gamma0.yml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. --set gamma=0 seed=3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--spurious", type=float, default=0.0, help="prob. that the background colour follows the class")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    cfg = TrainConfig.from_yaml(*[root / c for c in args.config], **overrides)
    print("config:", {k: v for k, v in vars(cfg).items()}, flush=True)

    x, y, bg = make_toy(n_per_class=300, n_classes=args.classes, spurious=args.spurious, seed=cfg.seed)
    tr = Trainer(cfg, x, background_redraw)

    def log(r):
        print(f"ep {r['epoch']:2d} it {r['iter']:4d} | L_psi {r.get('L_psi', 0):7.3f} L_pos {r.get('L_pos', 0):5.3f} "
              f"L_neg {r.get('L_neg', 0):5.3f} twin {r.get('twin_acc', 0):.2f} KL {r.get('KL', 0):6.1f} "
              f"L_inv {r.get('L_inv', 0):.3f} m_std {r.get('m_std', 0):.2f} | ent {r.get('pi_entropy', 0):.2f} "
              f"| slots {r['nonempty_slots']} K_eff {r['K_eff']:.2f} churn {r['churn']:.2f}", flush=True)

    tr.fit(args.epochs, log_fn=log)
    pred = tr.predict().cpu().numpy()
    y_np, bg_np = y.numpy(), bg.numpy()
    print(f"\nACC {cluster_acc(pred, y_np, cfg.num_slots):.3f}  NMI(cluster,class) {nmi(pred, y_np):.3f}  "
          f"NMI(cluster,background) {nmi(pred, bg_np):.3f}  clusters used {len(np.unique(pred))}")
