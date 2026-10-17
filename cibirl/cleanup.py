"""Last step after training, policy and encoder frozen: merge the memory slots that hold the same cluster twice.

Why it is needed (docs 5.23-5.28). Two slots of one class have the same summary up to the noise of the intents: the
reward cannot tell them apart, nothing closes them, and the memory only loses the duplicate by a slow neutral drift
(120 to 340 epochs). The output (argmax of the policy) is already right long before: it sends every image to the
bigger of the two twins. The summaries of twins are 1e-4 to 2e-3 apart, the summaries of two different clusters are
at least 3 apart (m is standardised), a gap of three orders of magnitude.

Rule, with no distance to choose. Sort the distances between the summaries of the non-empty slots. If the largest jump
between two consecutive sorted distances (a ratio) is at least `ratio`, every pair of slots below the jump is a pair of
twins: they are merged (single linkage). Else nothing is merged: there is no twin to find. `ratio` is a guard, not a
tuned value: the ratio is > 1000 when twins exist and < 3 between well-separated clusters.
"""
from __future__ import annotations

import numpy as np
import torch

from .memory import ClusterMemory

EPS = 1e-8           # distances are floored here so that two identical summaries give a finite ratio


def find_twins(summaries: torch.Tensor, counts: torch.Tensor, ratio: float = 10.0) -> tuple[dict[int, int], dict]:
    """Returns (`rep`, `info`): `rep[k]` is the slot that slot k is merged into (k itself if it is kept; the bigger slot of
    a group of twins is kept). `info` describes the decision: number of slots, the largest jump (`jump`), the distance below
    which slots are merged (`threshold`, None if nothing is merged) and the sorted distances (`distances`)."""
    slots = [int(k) for k in torch.nonzero(counts > 0).flatten()]
    rep = {k: k for k in slots}
    info = {"slots": len(slots), "jump": None, "threshold": None, "merged_pairs": 0, "distances": []}
    if len(slots) < 3:                       # one distance cannot be compared with another: no decision
        if len(slots) == 2:
            info["distances"] = [float((summaries[slots[0]] - summaries[slots[1]]).norm())]
        return rep, info
    pairs, dist = [], []
    for i, a in enumerate(slots):
        for b in slots[i + 1:]:
            pairs.append((a, b))
            dist.append(float((summaries[a] - summaries[b]).norm()))
    order = np.argsort(dist)
    d = np.maximum(np.asarray(dist)[order], EPS)
    jumps = d[1:] / d[:-1]
    t = int(np.argmax(jumps))
    info["jump"], info["distances"] = float(jumps[t]), [float(x) for x in np.asarray(dist)[order]]
    if jumps[t] < ratio:
        return rep, info
    info["threshold"] = float(d[t])
    parent = {k: k for k in slots}

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for j in order[:t + 1]:
        a, b = pairs[j]
        parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for k in slots:
        groups.setdefault(root(k), []).append(k)
    for members in groups.values():
        keep = max(members, key=lambda k: (float(counts[k]), -k))
        for k in members:
            rep[k] = keep
    info["merged_pairs"] = sum(1 for k in slots if rep[k] != k)
    return rep, info


@torch.no_grad()
def merge_twin_slots(memory: ClusterMemory, m_all: torch.Tensor, ratio: float = 10.0) -> dict:
    """Merge the twin slots of `memory` in place (members moved to the kept slot, summaries recomputed from `m_all`).
    Returns the `info` of `find_twins`, plus `rep`."""
    rep, info = find_twins(memory.summaries, memory.counts, ratio)
    if info["merged_pairs"]:
        lookup = torch.arange(memory.K, device=memory.assign.device)
        for k, r in rep.items():
            lookup[k] = r
        placed = memory.assign >= 0
        memory.assign[placed] = lookup[memory.assign[placed]]
        memory.recompute(m_all)
    info["rep"] = {int(k): int(r) for k, r in rep.items()}
    return info
