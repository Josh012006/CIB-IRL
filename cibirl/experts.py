"""Expert trajectory generator (README Section 5).

There are no labels, so the "expert" is a RULE built from image transformations:

* the views of one image form an ORBIT (the same thing seen through different nuisances);
* an episode shows the views of B images, n views each, shuffled, one after the other;
* the expert places each view in its OWN per-episode memory:
    - forced move   : a twin (same orbit) is already placed -> join the twin's slot (certain);
    - free move     : first view of its orbit -> open a new slot (no attraction is demonstrated);
    - repulsion     : every previously placed view of ANOTHER orbit says "do not join my slot"
                      (wrong with probability tau+, the chance that two random images share a concept).

Only the STRUCTURE is stored here (which image, which orbit, which order). The pixels are produced when a
batch is requested (views are re-drawn with the transformation family T) and the codes z, intents m and
memory summaries are recomputed by the trainer with the current encoder.

Phase 1 = every orbit is its own group (two different cat photos are two groups).
Phase 2 will only change how `group` is computed (see Phase2Relinker): nothing else moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

Transform = Callable[[Tensor], Tensor]   # [N, C, H, W] -> [N, C, H, W], independent randomness per sample


# --------------------------------------------------------------------------------------------------
# Views: the nuisance family T and orbits
# --------------------------------------------------------------------------------------------------
class OrbitSampler:
    """Draws a view of each image: chain d ~ U{0..d_max} calls of the transformation family T.

    d = 0 returns the image itself (it belongs to its own orbit). Bounding the cumulative strength of
    the chain is the job of `transform` itself; d_max bounds its length (long chains drift from the object).
    """

    def __init__(self, transform: Transform, d_max: int = 2, include_identity: bool = True):
        self.transform, self.d_max = transform, d_max
        self.low = 0 if include_identity else 1

    def __call__(self, x: Tensor) -> Tensor:
        d = torch.randint(self.low, self.d_max + 1, (x.shape[0],), device=x.device)
        out = x.clone()
        for step in range(1, self.d_max + 1):
            sel = d >= step                      # samples whose chain is at least `step` long
            if sel.any():
                out[sel] = self.transform(out[sel])
        return out


def split_expert_portion(num_items: int, fraction: float, seed: int = 0) -> Tensor:
    """Indices of the expert portion D_E (the learner's environment is the whole training set)."""
    g = torch.Generator().manual_seed(seed)
    n_expert = max(1, int(round(fraction * num_items)))
    return torch.randperm(num_items, generator=g)[:n_expert]


# --------------------------------------------------------------------------------------------------
# From groups to expert labels
# --------------------------------------------------------------------------------------------------
def assign_expert_slots(group: Tensor) -> tuple[Tensor, Tensor]:
    """The expert's slot of every view, from the group of every view.

    group [E, T] (any integer ids). The first view of a group opens the next free slot (0, 1, 2...),
    later views of the same group join it. Returns slot [E, T] and is_first [E, T] (free moves).
    Vectorized: the slot of a view = number of groups that appeared for the first time before its group.
    """
    E, T = group.shape
    pos = torch.arange(T, device=group.device)
    same = group.unsqueeze(2) == group.unsqueeze(1)                              # [E, T, T]
    first_idx = torch.where(same, pos.view(1, 1, T), torch.full_like(same, T, dtype=torch.long)).min(-1).values
    is_first = first_idx == pos.view(1, T)
    opened_before = is_first.unsqueeze(1) & (pos.view(1, 1, T) < first_idx.unsqueeze(-1))   # [E, T, T]
    return opened_before.sum(-1), is_first


def repulsion_mask(group: Tensor) -> Tensor:
    """neg_mask[e, t, v] = view v was placed before t AND belongs to another group: v repels t."""
    T = group.shape[1]
    earlier = torch.tril(torch.ones(T, T, dtype=torch.bool, device=group.device), diagonal=-1)
    return earlier.unsqueeze(0) & (group.unsqueeze(2) != group.unsqueeze(1))


# --------------------------------------------------------------------------------------------------
# Expert episodes
# --------------------------------------------------------------------------------------------------
@dataclass
class ExpertBatch:
    """Structure of E expert episodes of T = B*n views each (no pixels, no codes)."""
    image_ids: Tensor   # [E, T] dataset index of the image each view comes from
    orbit: Tensor       # [E, T] which image of the episode (0..B-1): the TRUE orbit, target of L_psi
    group: Tensor       # [E, T] what the expert considers "the same thing" (Phase 1: == orbit)
    slot: Tensor        # [E, T] the expert's slot for each view
    is_first: Tensor    # [E, T] True = free move (opens a slot); False = forced move (joins the twin)
    neg_mask: Tensor    # [E, T, T] repulsion pairs

    def to(self, device) -> "ExpertBatch":
        return ExpertBatch(*(t.to(device) for t in
                             (self.image_ids, self.orbit, self.group, self.slot, self.is_first, self.neg_mask)))


class ExpertEpisodes:
    """The dataset of expert demonstrations, built once before training and reused (README 5.2)."""

    def __init__(self, image_ids: Tensor, orbit: Tensor, num_slots: int):
        self.image_ids, self.orbit = image_ids, orbit            # [Etot, T] each
        self.num_episodes, self.T = image_ids.shape
        self.num_slots = num_slots

    @classmethod
    def build(cls, expert_ids: Tensor, num_episodes: int, num_orbits: int, views_per_orbit: int,
              num_slots: int, seed: int = 0) -> "ExpertEpisodes":
        """B = num_orbits distinct images per episode, n = views_per_orbit views each, shuffled."""
        B, n = num_orbits, views_per_orbit
        if B > num_slots:
            raise ValueError(f"B={B} orbits per episode must fit in K={num_slots} slots (B <= K)")
        if B > len(expert_ids):
            raise ValueError("not enough images in the expert portion")
        if n < 2:
            raise ValueError("need n >= 2 views per orbit: otherwise there is no forced move")
        g = torch.Generator().manual_seed(seed)
        ids_all, orb_all = [], []
        for start in range(0, num_episodes, 256):                 # chunks keep the sampling matrix small
            e = min(256, num_episodes - start)
            pick = torch.multinomial(torch.ones(e, len(expert_ids)), B, replacement=False, generator=g)
            ids = expert_ids[pick].repeat_interleave(n, dim=1)    # [e, T] orbit-major
            orb = torch.arange(B).repeat_interleave(n).expand(e, B * n)
            perm = torch.rand(e, B * n, generator=g).argsort(dim=1)   # shuffle the order of the views
            ids_all.append(ids.gather(1, perm))
            orb_all.append(orb.gather(1, perm))
        return cls(torch.cat(ids_all), torch.cat(orb_all), num_slots)

    def batch(self, episode_idx: Tensor, group: Tensor | None = None) -> ExpertBatch:
        """Labels for the chosen episodes. Phase 1: group = orbit. Phase 2 passes merged groups."""
        ids, orbit = self.image_ids[episode_idx], self.orbit[episode_idx]
        group = orbit if group is None else group
        slot, is_first = assign_expert_slots(group)
        return ExpertBatch(ids, orbit, group, slot, is_first, repulsion_mask(group))

    def linked_groups(self, episode_idx: Tensor, labels: Tensor, link_prob: float,
                      generator: torch.Generator | None = None) -> Tensor:
        """ORACLE DIAGNOSTIC (uses the true classes): groups in which orbits of the same class are linked.

        For every episode and every class present in it, with probability `link_prob` all the orbits of that
        class form ONE group (same expert slot, a forced move for the later ones, no repulsion between them);
        otherwise each orbit stays its own group, as in Phase 1. It plays the role that a perfect Phase 2
        relinker would play (with link_prob = 1) and answers one question only: is the expert's lack of
        cross-orbit attraction what keeps the learner from joining images of one class? Never a method.
        """
        ids, orbit = self.image_ids[episode_idx], self.orbit[episode_idx]
        cls = labels[ids]                                                        # [E, T]
        n_cls = int(labels.max()) + 1
        linked = torch.rand(len(episode_idx), n_cls, generator=generator) < link_prob   # [E, C]
        return torch.where(linked.gather(1, cls), cls, n_cls + orbit)            # ids: classes, then orbits

    def sample(self, batch_size: int, generator: torch.Generator | None = None,
               link_labels: Tensor | None = None, link_prob: float = 0.0) -> ExpertBatch:
        idx = torch.randint(0, self.num_episodes, (batch_size,), generator=generator)
        group = None
        if link_labels is not None and link_prob > 0:
            group = self.linked_groups(idx, link_labels, link_prob, generator)
        return self.batch(idx, group)


# --------------------------------------------------------------------------------------------------
# Phase 2 (not implemented yet)
# --------------------------------------------------------------------------------------------------
class Phase2Relinker:
    """Skeleton of the Phase 2 correction (README 5.5). It only has to produce `group` ids for
    `ExpertEpisodes.batch(..., group=...)`: two orbits linked together get the same group, hence the
    same expert slot, a forced move instead of a free move, and no repulsion between them.

    To do once Phase 1 is stable: mutual k-NN in the code space of an EMA copy of the encoder, confidence
    filter max_a pi >= gamma_c, cap of M links per epoch with ramp-up, persistence over several epochs,
    NO transitive closure, effective tau+ lowered as links remove false negatives, episodes enriched with
    neighbors (half random images, half of their mutual neighbors).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 2 comes after the Phase 1 baseline (README Section 5.5)")
