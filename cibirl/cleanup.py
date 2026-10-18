"""Last step after training, policy and encoder frozen: close the memory slots that hold the same cluster twice.

Three ways to do it (docs 5.28 and 5.29), all with the networks frozen; `TrainConfig.cleanup_mode` picks the one that gives the output:

    merge        merge the twin slots (below), then read the output again with the policy on the merged memory;
    relabel      (A) keep the output read BEFORE the merge, and only rename its slots with the merge: the policy is never
                 asked about a memory it has not seen;
    reintroduce  (C) sweep the images one by one: each is taken out of its slot and put back where the policy, on the current
                 memory, sends it (argmax, the summaries follow). The bigger twin attracts the images of the smaller one,
                 which empties. The output is the partition the sweeps end in. `reintroduce_no_open`: the same without the
                 action "open a new cluster" (the images are reintroduced INTO the existing clusters).

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

import copy
from contextlib import contextmanager

import numpy as np
import torch

from .memory import ClusterMemory, crp_log_prior

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


def relabel(pred: torch.Tensor, rep: dict[int, int]) -> torch.Tensor:
    """A: rename the slots of an output with the merge (`rep[k]` = slot that k is merged into). A slot that is not in `rep`
    (for instance the "outlier" id of the images sent to an empty slot) keeps its id."""
    size = max(int(pred.max()) if pred.numel() else 0, max(rep, default=0)) + 1
    lookup = torch.arange(size, device=pred.device)
    for k, r in rep.items():
        lookup[k] = r
    return lookup[pred]


@contextmanager
def use_memory(trainer, memory: ClusterMemory):
    """Temporarily read the output (`trainer.predict()`) with another memory."""
    keep = trainer.memory
    trainer.memory = memory
    try:
        yield
    finally:
        trainer.memory = keep


MAX_SWEEPS = 20      # safety cap of `reintroduce`; the sweeps stop earlier, when no image changes slot


@torch.no_grad()
def reintroduce(trainer, allow_open: bool = True, max_sweeps: int = MAX_SWEEPS, seed: int = 0) -> dict:
    """C: sweep the images one by one, in a random order, until a whole sweep moves no image. Same transition as a
    training rollout, with the policy frozen and ARGMAX instead of a sample: the image is withdrawn from its slot (the
    summaries follow), the policy reads the memory as it is now (CRP prior with the image withdrawn), and the image is put
    where the argmax says. At a tie between two twins the prior favours the bigger one, every move makes it bigger: the
    smaller twin is drained, then empty (the neutral drift of the training, made a ratchet).
    `allow_open=False` removes the action "open a new cluster" (the empty slots are not choosable).
    Works in place on `trainer.memory`. Returns {"sweeps", "moved" (per sweep), "slots" (non-empty at the end)}."""
    mem, cfg, N = trainer.memory, trainer.cfg, trainer.N
    bs = 512
    z = torch.cat([trainer.encoder(trainer.images[s:s + bs].to(trainer.device), sample=False).z for s in range(0, N, bs)])
    m = trainer.intent.mean(z)
    g = torch.Generator().manual_seed(int(seed))
    moved: list[int] = []
    for _ in range(max_sweeps):
        changes = 0
        for i in torch.randperm(N, generator=g).tolist():
            summ, empty = mem.state()
            lp0 = None
            if cfg.slot_prior == "crp":
                lp0 = crp_log_prior(mem.counts, empty, cfg.crp_alpha, int(mem.assign[i]), cfg.crp_power).unsqueeze(0)
            lp = trainer.policy.log_probs(z[i:i + 1], m[i:i + 1], summ.unsqueeze(0), empty.unsqueeze(0), lp0)
            if not allow_open and bool((~empty).any()):
                lp = lp.masked_fill(empty.unsqueeze(0), float("-inf"))
            changes += int(mem.place(i, m[i], int(lp.argmax(-1))))
        moved.append(changes)
        if changes == 0:
            break
    mem.recompute(m)
    return {"sweeps": len(moved), "moved": moved, "slots": mem.num_nonempty()}


def clone_memory(memory: ClusterMemory) -> ClusterMemory:
    return copy.deepcopy(memory)
