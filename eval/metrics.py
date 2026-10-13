"""Clustering metrics, always measured against the TRUE labels (never used for training)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def nmi(a, b) -> float:
    """Normalized mutual information with geometric normalization: I(a;b) / sqrt(H(a) H(b))."""
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


def confusion(pred, labels, num_slots: int) -> np.ndarray:
    """conf[k, c] = number of images of class c placed in slot k."""
    pred, labels = np.asarray(pred), np.asarray(labels)
    conf = np.zeros((num_slots, int(labels.max()) + 1))
    np.add.at(conf, (pred, labels), 1)
    return conf


def cluster_acc(pred, labels, num_slots: int) -> float:
    """Clustering accuracy with the optimal one-to-one matching between slots and classes (Hungarian)."""
    conf = confusion(pred, labels, num_slots)
    r, c = linear_sum_assignment(-conf)
    return float(conf[r, c].sum() / conf.sum())


def cluster_purity(pred, labels, num_slots: int) -> dict[int, tuple[int, int, float]]:
    """Per used slot: (size, dominant class, share of the dominant class)."""
    conf = confusion(pred, labels, num_slots)
    out = {}
    for k in range(num_slots):
        n = int(conf[k].sum())
        if n:
            c = int(conf[k].argmax())
            out[k] = (n, c, float(conf[k, c] / n))
    return out
