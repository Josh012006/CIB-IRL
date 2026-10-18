"""The evaluation protocol shared by every method (ours and the baselines): one partition, its metrics, the test modes.

A method is evaluated on a TRAIN partition (the unlabeled images it was fit on, transductive) and on TEST partitions made of
images it never saw, under several backgrounds (`TEST_MODES`). The labels only evaluate: they never reach a method.

    from eval.protocol import partition_metrics, is_right, TEST_MODES

`partition_metrics` is the function that `scripts/eval_heldout.py` used for the toy results (README Section 9.3); it moved
here so that the baselines (`scripts/run_baseline.py`) are scored by exactly the same code.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import cluster_acc, confusion, nmi

TEST_MODES = ("iid", "aligned", "inverted")
# iid       the nuisance is independent of the class, as in the training set of the toy
# aligned   the nuisance points to the RIGHT class  (a model that uses it looks better)
# inverted  the nuisance points to a WRONG class    (a model that uses it looks worse)


def partition_metrics(pred: np.ndarray, y: np.ndarray, bgid: np.ndarray | None, kept_slots: set[int] | None, K: int) -> dict:
    """Metrics of one partition.

    pred         cluster id of every image (any non-negative integers)
    y            true class of every image (evaluation only)
    bgid         a coarse id of the nuisance (background colour...), or None if the dataset has none
    kept_slots   the ids that hold a cluster; any other id is the group of images sent to "open a new cluster" (our model
                 only). None: every id is a cluster (the baselines)
    K            number of slots / clusters the method may use (size of the confusion matrix is max(K, largest id + 1))
    """
    pred, y = np.asarray(pred), np.asarray(y)
    K = max(int(K), int(pred.max()) + 1)
    bg = np.zeros_like(y) if bgid is None else np.asarray(bgid)
    conf = confusion(pred, y, K)
    rows, cols = linear_sum_assignment(-conf)
    slot_of_class = {int(c): int(r) for r, c in zip(rows, cols)}
    ids = np.unique(pred)
    worst, n_groups = 1.0, 0
    for c in np.unique(y):
        for b in np.unique(bg[y == c]):
            sel = (y == c) & (bg == b)
            if sel.sum() >= 10:
                n_groups += 1
                worst = min(worst, float((pred[sel] == slot_of_class.get(int(c), -1)).mean()))    # -1: fewer clusters than classes
    open_share = 0.0 if kept_slots is None else float(np.mean([p not in kept_slots for p in pred]))
    return {"acc": cluster_acc(pred, y, K), "nmi_class": nmi(pred, y),
            "nmi_background": None if bgid is None else nmi(pred, bg),
            "clusters": int(len(ids)), "clusters_ge10": int(sum(1 for q in ids if (pred == q).sum() >= 10)),
            "open_share": open_share, "worst_group_acc": worst, "n_groups": n_groups}


def is_right(m: dict, n_classes: int) -> bool:
    """A run is *right* when its output has exactly `n_classes` clusters (of at least 10 images) and ACC = 1."""
    return m["clusters_ge10"] == n_classes and m["acc"] > 0.999


def delta_acc(row: dict) -> float:
    """ACC(aligned) - ACC(inverted): how much the answer changes when the nuisance points to the right or to a wrong class.
    0 means the nuisance is not used."""
    return row["aligned"]["acc"] - row["inverted"]["acc"]
