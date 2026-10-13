"""Reward R_theta(z, m, C, a) and the choice model q_theta it induces (README Sections 5.2 and 6.3).

The reward is ONE network scoring "put the arriving image in slot a", given the clusters built so far.
It is permutation-equivariant over the slots: it never reads a slot's index or its count n_k, only
its content c_k. It is written as

    R(z, m, C, a) = g_theta( z, m, rel(m, c_a), pool_{k != a} rel(m, c_k), is_empty(a), has_others )

where rel(m, c_k) is a learned comparison between the arriving intent and a cluster summary. The part
on the chosen slot measures compatibility; the pooled part over the other slots measures distinctness.

`SlotScorer` is reused by policy.py (a separate network with its own weights, same input signature).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .memory import masked_log_softmax, valid_slot_mask


def mlp(sizes: list[int], last_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RelationEncoder(nn.Module):
    """rel(m, c_k): a learned comparison between the arriving intent m and each cluster summary c_k.

    Empty slots have no meaningful summary (zeros): a learned embedding takes its place.
    m [B, d], summaries [B, K, d], empty [B, K]  ->  [B, K, rel_dim]
    """

    def __init__(self, m_dim: int, hidden: int = 128, rel_dim: int = 64):
        super().__init__()
        self.empty_emb = nn.Parameter(torch.zeros(m_dim))
        nn.init.normal_(self.empty_emb, std=0.1)
        # Input: the two vectors, their product (agreement per dimension) and |difference| (distance).
        self.net = mlp([4 * m_dim, hidden, rel_dim], last_activation=True)
        self.rel_dim = rel_dim

    def forward(self, m: Tensor, summaries: Tensor, empty: Tensor) -> Tensor:
        B, K, d = summaries.shape
        c = torch.where(empty.unsqueeze(-1), self.empty_emb.expand(B, K, d), summaries)
        mm = m.unsqueeze(1).expand(B, K, d)
        return self.net(torch.cat([mm, c, mm * c, (mm - c).abs()], dim=-1))


class SlotScorer(nn.Module):
    """Permutation-equivariant scorer over slots: (z, m, summaries, empty) -> one score per slot.

    z [B, z_dim], m [B, m_dim], summaries [B, K, m_dim], empty [B, K] bool  ->  scores [B, K]
    """

    def __init__(self, z_dim: int, m_dim: int, hidden: int = 128, rel_dim: int = 64, use_z: bool = True):
        super().__init__()
        self.use_z = use_z
        self.rel_enc = RelationEncoder(m_dim, hidden, rel_dim)
        head_in = (z_dim if use_z else 0) + m_dim + 3 * rel_dim + 2
        self.head = mlp([head_in, hidden, hidden, 1])

    def forward(self, z: Tensor, m: Tensor, summaries: Tensor, empty: Tensor) -> Tensor:
        B, K, _ = summaries.shape
        rel = self.rel_enc(m, summaries, empty)                                     # [B, K, h]
        h = rel.shape[-1]

        # "Other occupied slots" of each slot a: others[b, a, k] = occupied(k) and k != a.
        eye = torch.eye(K, dtype=torch.bool, device=rel.device)
        others = (~empty).unsqueeze(1) & ~eye.unsqueeze(0)                          # [B, K(a), K(k)]
        n_others = others.sum(-1)                                                   # [B, K]
        has_others = n_others > 0

        rel_k = rel.unsqueeze(1).expand(B, K, K, h)                                 # [B, a, k, h]
        mean_o = (rel_k * others.unsqueeze(-1)).sum(2) / n_others.clamp(min=1).unsqueeze(-1)
        max_o = rel_k.masked_fill(~others.unsqueeze(-1), float("-inf")).max(2).values
        max_o = torch.where(has_others.unsqueeze(-1), max_o, torch.zeros_like(max_o))

        parts = [m.unsqueeze(1).expand(B, K, -1)]
        if self.use_z:
            parts.insert(0, z.unsqueeze(1).expand(B, K, -1))
        parts += [rel, mean_o, max_o, empty.float().unsqueeze(-1), has_others.float().unsqueeze(-1)]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)                      # [B, K]


class RewardModel(nn.Module):
    """R_theta. Learned by IRL on the expert's states (losses.expert_loss), applied on the learner's."""

    def __init__(self, z_dim: int, m_dim: int, hidden: int = 128, rel_dim: int = 64, use_z: bool = True):
        super().__init__()
        self.scorer = SlotScorer(z_dim, m_dim, hidden, rel_dim, use_z)

    def forward(self, z: Tensor, m: Tensor, summaries: Tensor, empty: Tensor) -> Tensor:
        """Raw rewards R_theta(s, a) for every slot: [B, K]."""
        return self.scorer(z, m, summaries, empty)

    @staticmethod
    def choice_log_probs(scores: Tensor, empty: Tensor) -> Tensor:
        """log q_theta(a | s) = log softmax_a R_theta(s, a), over the valid slots (README 5.2)."""
        return masked_log_softmax(scores, valid_slot_mask(empty))
