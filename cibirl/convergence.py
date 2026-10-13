"""When to stop training, WITHOUT looking at the labels.

Stopping on ACC / NMI would be selection on the test labels: the number of epochs would become a
hyper-parameter tuned on the answer. The rule below only uses what the method itself produces.

A run is declared converged at the end of an epoch (after `min_epochs`) when, over the last `window`
epochs, ALL of these hold:

1. partition stability   the argmax partition (policy with the memory frozen, the one that is
                         evaluated at the end) changes for at most `part_tol` of the images from one
                         epoch to the next, up to a renaming of the clusters (optimal matching: two
                         clusters exchanging their slot ids is not a change);
2. cluster count         the number of clusters in that partition is constant;
3. objective plateau     the mean twin accuracy and the mean expert loss (L_pos + L_neg) have moved by
                         less than `twin_tol` (absolute) and `loss_tol` (relative) compared with the
                         window before.

The run is otherwise stopped at `max_epochs` and flagged `converged = false`.

A converged run is a STABLE run, not necessarily a GOOD one (a collapsed solution is stable too):
always read `converged` together with the metrics and the diagnostics (m_std, n_clusters).
"""
from __future__ import annotations

import difflib
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .train import _coerce


@dataclass
class StopConfig:
    max_epochs: int = 300
    min_epochs: int = 40          # larger than the warm-ups (reward_warmup_iters, beta_warmup_iters) in epochs
    window: int = 10
    part_tol: float = 0.01
    twin_tol: float = 0.01
    loss_tol: float = 0.02
    enabled: bool = True          # False: run exactly max_epochs

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StopConfig":
        known = {f.name: f.type for f in fields(cls)}
        d = d or {}
        for key in d:
            if key not in known:
                hint = difflib.get_close_matches(key, known, n=1)
                raise KeyError(f"unknown stop key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        cfg = cls(**{k: _coerce(k, v, known[k]) for k, v in d.items()})
        if cfg.window < 1 or cfg.max_epochs < 1:
            raise ValueError("window and max_epochs must be >= 1")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def partition_change(a, b) -> float:
    """Share of images that do not keep their cluster from partition `a` to partition `b`, up to a renaming
    of the clusters (optimal one-to-one matching). A cluster that disappears or appears counts as changed."""
    a, b = np.asarray(a), np.asarray(b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    conf = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(conf, (ia, ib), 1)
    rows, cols = linear_sum_assignment(-conf)
    return float((len(a) - conf[rows, cols].sum()) / len(a))        # (n - matched) / n: exact for tolerances like 0.01


class ConvergenceMonitor:
    """Feed it one row per epoch (and the argmax partition): `observe` returns True when to stop."""

    def __init__(self, cfg: StopConfig):
        self.cfg = cfg
        self.rows: list[dict] = []
        self._prev: np.ndarray | None = None
        self.converged = False
        self.reason = "running"

    def observe(self, row: dict, pred) -> bool:
        """Adds `part_change`, `slot_change` and `n_clusters` to `row` (in place, so that they reach metrics.jsonl).

        `slot_change` is the same share WITHOUT the renaming (diagnostic: a large slot_change with a small
        part_change means that clusters are swapping slot ids)."""
        pred = np.asarray(pred)
        row["part_change"] = 1.0 if self._prev is None else partition_change(self._prev, pred)
        row["slot_change"] = 1.0 if self._prev is None else float(np.mean(pred != self._prev))
        row["n_clusters"] = int(len(np.unique(pred)))
        self._prev = pred
        self.rows.append(row)
        epochs_done = len(self.rows)
        if epochs_done >= self.cfg.max_epochs:
            if self.cfg.enabled and self._converged_now():
                self.converged, self.reason = True, "converged"
            else:
                self.reason = "max_epochs" if self.cfg.enabled else "fixed_epochs"
            return True
        if self.cfg.enabled and epochs_done >= self.cfg.min_epochs and self._converged_now():
            self.converged, self.reason = True, "converged"
            return True
        return False

    def _converged_now(self) -> bool:
        w = self.cfg.window
        if len(self.rows) < 2 * w + 1:           # two windows, and the first epoch has no predecessor
            return False
        last, prev = self.rows[-w:], self.rows[-2 * w:-w]
        if max(r["part_change"] for r in last) > self.cfg.part_tol:
            return False
        if len({r["n_clusters"] for r in last}) != 1:
            return False

        def mean(rows, key):
            return float(np.mean([r.get(key, math.nan) for r in rows]))

        def expert_loss(rows):
            return float(np.mean([r.get("L_pos", math.nan) + r.get("L_neg", math.nan) for r in rows]))

        d_twin = abs(mean(last, "twin_acc") - mean(prev, "twin_acc"))
        l_prev, l_last = expert_loss(prev), expert_loss(last)
        d_loss = abs(l_last - l_prev) / max(abs(l_prev), 1e-8)
        return bool(d_twin <= self.cfg.twin_tol and d_loss <= self.cfg.loss_tol)    # NaN -> False: keep training
