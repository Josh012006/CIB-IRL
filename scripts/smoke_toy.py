"""End-to-end smoke test on synthetic data: does the Phase 1 pipeline run, and does it find the classes?
Usage: python scripts/smoke_toy.py [--epochs 30] [--spurious 0.0]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--spurious", type=float, default=0.0)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, y, bg = make_toy(n_per_class=300, n_classes=args.classes, spurious=args.spurious, seed=args.seed)
    cfg = TrainConfig(num_slots=args.slots, image_size=16, orbits_per_episode=3, num_expert_episodes=2000,
                      expert_fraction=0.3, learner_batch=100, reward_warmup_iters=150, beta_warmup_iters=300,
                      intent_standardize=not args.no_standardize, seed=args.seed)
    tr = Trainer(cfg, x, background_redraw)

    def log(r):
        print(f"ep {r['epoch']:2d} it {r['iter']:4d} | L_psi {r.get('L_psi', 0):7.3f} L_pos {r.get('L_pos', 0):5.3f} "
              f"L_neg {r.get('L_neg', 0):5.3f} twin {r.get('twin_acc', 0):.2f} KL {r.get('KL', 0):6.1f} "
              f"L_inv {r.get('L_inv', 0):.3f} m_std {r.get('m_std', 0):.2f} | ent {r.get('pi_entropy', 0):.2f} "
              f"| slots {r['nonempty_slots']} K_eff {r['K_eff']:.2f} churn {r['churn']:.2f}", flush=True)

    tr.fit(args.epochs, log_fn=log)
    pred = tr.predict().cpu().numpy()
    y_np, bg_np = y.numpy(), bg.numpy()
    print(f"\nACC {cluster_acc(pred, y_np, args.slots):.3f}  NMI(cluster,class) {nmi(pred, y_np):.3f}  "
          f"NMI(cluster,background) {nmi(pred, bg_np):.3f}  clusters used {len(np.unique(pred))}")
