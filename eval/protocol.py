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


def class_to_slot(pred: np.ndarray, y: np.ndarray, K: int) -> dict[int, int]:
    """The one-to-one matching class -> slot that maximises the accuracy of ONE partition (Hungarian).

    Computed on the TRAIN partition and then applied unchanged to the test partitions (`acc_fixed`): the labels only evaluate."""
    pred, y = np.asarray(pred), np.asarray(y)
    K = max(int(K), int(pred.max()) + 1)
    rows, cols = linear_sum_assignment(-confusion(pred, y, K))
    return {int(c): int(r) for r, c in zip(rows, cols)}


def partition_metrics(pred: np.ndarray, y: np.ndarray, bgid: np.ndarray | None, kept_slots: set[int] | None, K: int,
                      slot_of_class: dict[int, int] | None = None) -> dict:
    """Metrics of one partition.

    pred         cluster id of every image (any non-negative integers)
    y            true class of every image (evaluation only)
    bgid         a coarse id of the nuisance (background colour...), or None if the dataset has none
    kept_slots   the ids that hold a cluster; any other id is the group of images sent to "open a new cluster" (our model
                 only). None: every id is a cluster (the baselines)
    K            number of slots / clusters the method may use (size of the confusion matrix is max(K, largest id + 1))
    slot_of_class  matching class -> slot FIXED beforehand (`class_to_slot` of the train partition). `acc_fixed` is the share of
                 images sent to the slot of their class under this matching. None: the matching of this very partition, so
                 `acc_fixed` = `acc` (this is the case of the train partition).

    Why `acc_fixed`. `acc` re-matches slots and classes on every partition. In the test modes "aligned" and "inverted" the
    background colour is a one-to-one function of the class (the class itself, or the next one), so a method that clusters by
    colour alone has `acc` = 1 in BOTH modes and Delta ACC = 0: the Hungarian matching absorbs the permutation. With the matching
    fixed on the train partition, the same method scores 1 on "aligned" and about 0 on "inverted" (docs 5.33).
    """
    pred, y = np.asarray(pred), np.asarray(y)
    K = max(int(K), int(pred.max()) + 1)
    bg = np.zeros_like(y) if bgid is None else np.asarray(bgid)
    conf = confusion(pred, y, K)
    rows, cols = linear_sum_assignment(-conf)
    own = {int(c): int(r) for r, c in zip(rows, cols)}                 # the matching of this very partition
    fixed = own if slot_of_class is None else slot_of_class
    ids = np.unique(pred)
    worst, n_groups = 1.0, 0
    for c in np.unique(y):
        for b in np.unique(bg[y == c]):
            sel = (y == c) & (bg == b)
            if sel.sum() >= 10:
                n_groups += 1
                worst = min(worst, float((pred[sel] == own.get(int(c), -1)).mean()))    # -1: fewer clusters than classes
    open_share = 0.0 if kept_slots is None else float(np.mean([p not in kept_slots for p in pred]))
    acc_fixed = float(np.mean([fixed.get(int(c), -1) == p for c, p in zip(y, pred)]))
    return {"acc": cluster_acc(pred, y, K), "acc_fixed": acc_fixed, "nmi_class": nmi(pred, y),
            "nmi_background": None if bgid is None else nmi(pred, bg),
            "clusters": int(len(ids)), "clusters_ge10": int(sum(1 for q in ids if (pred == q).sum() >= 10)),
            "open_share": open_share, "worst_group_acc": worst, "n_groups": n_groups}


def is_right(m: dict, n_classes: int) -> bool:
    """A run is *right* when its output has exactly `n_classes` clusters (of at least 10 images) and ACC = 1."""
    return m["clusters_ge10"] == n_classes and m["acc"] > 0.999


def delta_acc(row: dict) -> float:
    """ACC(aligned) - ACC(inverted) with the class -> slot matching FIXED on the train partition (`acc_fixed`): how much the
    answer changes when the nuisance points to the right or to a wrong class. 0 means the nuisance is not used.
    Rows computed before docs 5.33 have no `acc_fixed`: they raise KeyError instead of silently using the other definition."""
    return row["aligned"]["acc_fixed"] - row["inverted"]["acc_fixed"]


def delta_acc_rematched(row: dict) -> float:
    """The former Delta ACC (docs 5.31): the matching is recomputed on each test partition. Blind to a method that clusters by
    background colour alone (see `partition_metrics`); kept as a diagnostic."""
    return row["aligned"]["acc"] - row["inverted"]["acc"]
