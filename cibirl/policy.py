"""RL policy pi_omega(a | z, m, C) and critic V_xi(s) (README Section 6.4).

Both are SEPARATE networks from the reward: the policy is trained by RL against the frozen reward and
never sees the expert, the orbits or the expert's actions. The policy has the same input signature and
the same permutation-equivariant structure as the reward (it reuses `SlotScorer`, with its own weights).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .memory import masked_entropy, masked_log_softmax, valid_slot_mask
from .reward import RelationEncoder, SlotScorer, mlp


class PolicyNet(nn.Module):
    def __init__(self, z_dim: int, m_dim: int, hidden: int = 128, rel_dim: int = 64, use_z: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.scorer = SlotScorer(z_dim, m_dim, hidden, rel_dim, use_z)
        self.temperature = float(temperature)

    def log_probs(self, z: Tensor, m: Tensor, summaries: Tensor, empty: Tensor,
                  log_prior: Tensor | None = None) -> Tensor:
        """log pi(a | s) over the valid slots: [B, K] (-inf on invalid slots).

        pi = softmax(scores / T + log_prior). T is part of the DEFINITION of the policy, so sampling, the PPO ratio and
        the entropy all use the same distribution. T < 1 sharpens it (less exploration); it never changes the argmax.
        `log_prior` [B, K] (memory.crp_log_prior) is a fixed additive offset, outside of T: the prior over partitions
        (docs 5.24). None = no prior, exactly the policy of before.
        """
        scores = self.scorer(z, m, summaries, empty)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        valid = valid_slot_mask(empty)
        if log_prior is not None:
            scores = scores + log_prior.masked_fill(~valid, 0.0)
        return masked_log_softmax(scores, valid)

    @torch.no_grad()
    def act(self, z: Tensor, m: Tensor, summaries: Tensor, empty: Tensor, greedy: bool = False,
            log_prior: Tensor | None = None):
        """Sample (or argmax) one action per row. Returns action [B], log-prob [B]."""
        logp = self.log_probs(z, m, summaries, empty, log_prior)
        a = logp.argmax(-1) if greedy else torch.distributions.Categorical(logits=logp).sample()
        return a, logp.gather(-1, a.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def entropy(logp: Tensor, empty: Tensor) -> Tensor:
        return masked_entropy(logp, valid_slot_mask(empty))


class ValueNet(nn.Module):
    """Critic V_xi(s): expected discounted return from state s = (z, m, C).

    The state is summarized in a permutation-invariant way: mean over the occupied slots of
    rel(m, c_k), plus z and m. Its own parameters (xi), not shared with the policy.
    """

    def __init__(self, z_dim: int, m_dim: int, hidden: int = 128, rel_dim: int = 64, use_z: bool = True):
        super().__init__()
        self.use_z = use_z
        self.rel = RelationEncoder(m_dim, hidden, rel_dim)
        self.head = mlp([(z_dim if use_z else 0) + m_dim + rel_dim + 1, hidden, 1])

    def forward(self, z: Tensor, m: Tensor, summaries: Tensor, empty: Tensor) -> Tensor:
        rel = self.rel(m, summaries, empty)                       # [B, K, h]
        occ = (~empty).unsqueeze(-1).float()
        pooled = (rel * occ).sum(1) / occ.sum(1).clamp(min=1)               # [B, h]
        any_occ = (occ.sum(1) > 0).float()                                   # [B, 1]
        parts = ([z] if self.use_z else []) + [m, pooled, any_occ]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)              # [B]
