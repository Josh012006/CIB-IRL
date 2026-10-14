"""Cluster memories (README Sections 4.2 and 6.7) and the action space they define.

Two memories share one transition (the chosen slot absorbs the intent of the view):

    n_a <- n_a + 1,        c_a <- c_a + (m - c_a) / n_a            (running mean = arithmetic mean)

* `ClusterMemory`            the LEARNER's persistent memory C. Stateful, updated one image at a time
                             by the policy, summaries recomputed with the current encoder each epoch.
* `expert_memory_states`     the EXPERT's per-episode memory C^E. Functional and vectorized: since the
                             expert's slots are known in advance, the state before every step of every
                             episode is a cumulative mean, computed for all episodes at once.

Memories are constants for autograd: nothing in this file carries gradients.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

NEG_INF = float("-inf")


# --------------------------------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------------------------------
def valid_slot_mask(empty: Tensor) -> Tensor:
    """Which slots can be chosen: the occupied ones, plus ONE empty slot (the first) = "open a cluster".

    `empty`: [..., K] bool. Without this, K - (#occupied) identical empty slots would all be legal
    actions, and the probability of "opening a new cluster" would depend on how many empties are
    left, which changes during the episode. One canonical empty slot removes that artefact.
    """
    first_empty = (empty.long().cumsum(-1) == 1) & empty
    return (~empty) | first_empty


def masked_log_softmax(scores: Tensor, valid: Tensor) -> Tensor:
    """log softmax over the valid slots only. Invalid slots get -inf."""
    return scores.masked_fill(~valid, NEG_INF).log_softmax(-1)


def masked_entropy(logp: Tensor, valid: Tensor) -> Tensor:
    """Entropy of a distribution given by masked log-probs. Returns [...].

    `safe_logp` replaces -inf by 0 BEFORE the product: p * logp would be 0 * -inf = NaN in the
    backward pass even when the forward value is masked out.
    """
    safe_logp = logp.masked_fill(~valid, 0.0)
    return -(logp.exp() * safe_logp).sum(-1)


# --------------------------------------------------------------------------------------------------
# Learner's persistent memory
# --------------------------------------------------------------------------------------------------
class ClusterMemory:
    """Persistent memory C of the learner: one assignment per image, K slot summaries.

    Stored per image: `assign[i]` (slot, -1 if never placed) and `member_m[i]` (the intent used when
    it was placed, needed to withdraw it exactly). Per slot: `counts` and `summaries` (c_k).
    """

    def __init__(self, num_items: int, num_slots: int, m_dim: int, device: torch.device | str = "cpu"):
        self.K, self.d, self.device = num_slots, m_dim, torch.device(device)
        self.counts = torch.zeros(num_slots, device=self.device)
        self.summaries = torch.zeros(num_slots, m_dim, device=self.device)
        self.assign = torch.full((num_items,), -1, dtype=torch.long, device=self.device)
        self.member_m = torch.zeros(num_items, m_dim, device=self.device)

    # -- read ----------------------------------------------------------------------------------
    def state(self) -> tuple[Tensor, Tensor]:
        """(summaries [K, d], empty [K] bool). Copies: the caller can store them as a snapshot."""
        return self.summaries.clone(), (self.counts == 0)

    def num_nonempty(self) -> int:
        return int((self.counts > 0).sum())

    # -- write ---------------------------------------------------------------------------------
    @torch.no_grad()
    def withdraw(self, idx: int) -> None:
        """Remove image `idx` from its slot (exact inverse of the running mean)."""
        k = int(self.assign[idx])
        if k < 0:
            return
        n = self.counts[k]
        if n <= 1:
            self.counts[k] = 0
            self.summaries[k].zero_()
        else:
            self.summaries[k] = (self.summaries[k] * n - self.member_m[idx]) / (n - 1)
            self.counts[k] = n - 1
        self.assign[idx] = -1

    @torch.no_grad()
    def place(self, idx: int, m: Tensor, slot: int) -> bool:
        """Place image `idx` with intent `m` in `slot`. If it was placed before, the old placement is
        withdrawn first (an image belongs to at most one slot). Returns True if its slot changed."""
        prev = int(self.assign[idx])
        self.withdraw(idx)
        n = self.counts[slot] + 1
        self.summaries[slot] += (m - self.summaries[slot]) / n
        self.counts[slot] = n
        self.assign[idx] = slot
        self.member_m[idx] = m
        return prev >= 0 and prev != slot

    @torch.no_grad()
    def recompute(self, m_all: Tensor) -> None:
        """End of epoch: c_k <- mean of the CURRENT intents of the members of k (README 6.7).
        `m_all`: [N, d], intents of every image with the current encoder."""
        placed = self.assign >= 0
        sums = torch.zeros(self.K, self.d, device=self.device)
        sums.index_add_(0, self.assign[placed], m_all[placed])
        self.counts = torch.bincount(self.assign[placed], minlength=self.K).float()
        self.summaries = sums / self.counts.clamp(min=1).unsqueeze(-1)
        self.member_m = m_all.clone()

    @torch.no_grad()
    def subsample(self, m_all: Tensor, fraction: float) -> None:
        """Keep a random `fraction` of the members of every non-empty slot (at least one) and forget the others:
        their assignment goes back to -1 and the summaries are recomputed from the kept members only.
        Used once, early, as a burn-in: the memory is rebuilt from a sample of what the clusters contain."""
        keep = torch.zeros(self.assign.shape[0], dtype=torch.bool, device=self.device)
        for k in range(self.K):
            members = (self.assign == k).nonzero().flatten()
            if len(members):
                n_keep = max(1, math.ceil(fraction * len(members)))
                keep[members[torch.randperm(len(members), device=members.device)[:n_keep]]] = True
        self.assign = torch.where(keep, self.assign, torch.full_like(self.assign, -1))
        self.recompute(m_all)

    @torch.no_grad()
    def clear(self) -> None:
        """Forget everything: all images unplaced, all slots empty."""
        self.counts.zero_()
        self.summaries.zero_()
        self.assign.fill_(-1)
        self.member_m.zero_()

    def sizes(self) -> Tensor:
        return self.counts.clone()


# --------------------------------------------------------------------------------------------------
# Expert's per-episode memory
# --------------------------------------------------------------------------------------------------
def expert_memory_states(m: Tensor, slot: Tensor, num_slots: int) -> tuple[Tensor, Tensor]:
    """State of the expert's memory BEFORE each step, for a batch of episodes.

    m    [E, T, d]  intent of the view arriving at each step (detached by the caller)
    slot [E, T]     slot the expert puts each view in (known from the orbits)

    Returns summaries [E, T, K, d] and empty [E, T, K]: the state in which view t arrives, i.e. built
    from views 0..t-1 only. This is the transition applied t times, written as an exclusive cumsum.
    """
    onehot = F.one_hot(slot, num_slots).to(m.dtype)                    # [E, T, K]
    contrib = onehot.unsqueeze(-1) * m.unsqueeze(2)                    # [E, T, K, d]
    sums = contrib.cumsum(dim=1) - contrib                             # exclusive: views before t
    counts = onehot.cumsum(dim=1) - onehot                             # [E, T, K]
    summaries = sums / counts.clamp(min=1).unsqueeze(-1)
    return summaries, counts == 0
