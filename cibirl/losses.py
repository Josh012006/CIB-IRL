"""All the objectives (README Sections 4.4, 4.5, 6.1-6.4 and 6.7), as plain functions of tensors.

Reward step (updates phi, psi, theta):   L = L_psi + L_theta + beta * KL + lambda * L_inv
Policy step (updates omega, xi):         PPO surrogate + entropy bonus, value regression

Nothing here owns parameters or decides who gets gradients: the trainer routes them (train.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .memory import masked_log_softmax, valid_slot_mask


# --------------------------------------------------------------------------------------------------
# Information Bottleneck: compression term
# --------------------------------------------------------------------------------------------------
def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """E_x KL( N(mu, sigma^2) || N(0, I) ) = E_x 0.5 * sum_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2).

    Upper bound on I(z; x) (Deep VIB), with the fixed prior r(z) = N(0, I).
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()


# --------------------------------------------------------------------------------------------------
# Intent: L_psi
# --------------------------------------------------------------------------------------------------
def orbit_intent_target(mu: Tensor, orbit: Tensor) -> Tensor:
    """m*_i = sg[ mean over the views of orbit i of mu_psi(z_v) ], given back for every view.

    mu [E, T, d], orbit [E, T]  ->  [E, T, d]. The mean is taken among the views of the same
    episode that share the orbit id; sg = stop-gradient: it is a target, not a path for gradients.
    """
    same = (orbit.unsqueeze(2) == orbit.unsqueeze(1)).to(mu.dtype)         # [E, T, T]
    return torch.matmul(same, mu.detach()) / same.sum(-1, keepdim=True)


def intent_nll(mu: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """L_psi = -E log q_psi(m* | z_u), q_psi Gaussian (additive constant dropped).

    = 0.5 * sum_j [ log sigma_j^2 + (m*_j - mu_j)^2 / sigma_j^2 ]: a squared error weighted by the
    predicted precision; the model may claim uncertainty but pays log sigma^2 for it.
    """
    return 0.5 * (logvar + (target - mu).pow(2) / logvar.exp()).sum(-1).mean()


# --------------------------------------------------------------------------------------------------
# Reward: expert likelihood  L_theta = L_pos + L_neg - rho * H(marginal)
# --------------------------------------------------------------------------------------------------
def _log_one_minus_prob(scores: Tensor, valid: Tensor) -> Tensor:
    """log(1 - q(k)) for every slot k, computed without ever forming 1 - q (which loses precision
    when q -> 1):  log(1 - q_k) = log sum_{j valid, j != k} e^{s_j}  -  log sum_{j valid} e^{s_j}.

    Masked with a large negative number instead of -inf: a row with nothing left would give NaN
    gradients. The result is clamped at -30 (q = 1 - 1e-13).
    """
    K = scores.shape[-1]
    s = scores.masked_fill(~valid, -1e4)
    log_z = torch.logsumexp(s, dim=-1, keepdim=True)
    eye = torch.eye(K, dtype=torch.bool, device=scores.device)
    s_without_k = s.unsqueeze(-2).expand(*s.shape[:-1], K, K).masked_fill(eye, -1e4)   # [..., k, j]
    return (torch.logsumexp(s_without_k, dim=-1) - log_z).clamp(min=-30.0)


@dataclass
class ExpertLossOut:
    loss: Tensor
    l_pos: Tensor
    l_neg: Tensor
    twin_acc: Tensor      # fraction of forced moves where argmax q_theta is the twin's slot (monitoring)


def expert_loss(scores: Tensor, empty: Tensor, slot: Tensor, is_first: Tensor, neg_mask: Tensor,
                tau_plus: float, rho: float = 0.0, free_debias: float = 1.0) -> ExpertLossOut:
    """Maximum-likelihood IRL of the expert's relational decisions under q_theta = softmax(R_theta).

    scores [E, T, K]  raw rewards R_theta(s^E_t, a)
    empty  [E, T, K]  occupancy of the EXPERT's memory before each step
    slot, is_first [E, T];  neg_mask [E, T, T]  (see experts.py)

    L_pos   attraction : -log q(k+(u))                         for forced moves (twin placed in k+)
    L_neg   repulsion  : l(u,v) = -log(1 - q(k(v)))            for previously placed views v of other orbits,
                         debiased as in Chuang et al. 2020:
                         max{ ( E_v l(u,v) - tau+ * l(u,v+) ) / (1 - tau+), 0 }
    free_debias multiplies tau+ on the FREE moves only (1 = as above; 0 = plain repulsion -log(1-q) there, no
    correction): the correction cancels the repulsion as soon as q(wrong slot) < 1 - exp(-tau+ * l+), see
    docs/protocol-decisions.md 5.9.
    """
    E, T, K = scores.shape
    valid = valid_slot_mask(empty)
    logp = masked_log_softmax(scores, valid)                                   # log q_theta
    log1m = _log_one_minus_prob(scores, valid)                                 # log (1 - q_theta)
    forced = ~is_first

    # --- attraction ------------------------------------------------------------------------------
    logq_twin = logp.gather(-1, slot.unsqueeze(-1)).squeeze(-1)                # [E, T]; slot = twin's slot if forced
    l_pos = -logq_twin[forced].mean() if forced.any() else scores.sum() * 0.0

    # --- repulsion -------------------------------------------------------------------------------
    # ell[e, t, v] = -log(1 - q(slot of view v | state of step t))
    ell = -log1m.gather(-1, slot.unsqueeze(1).expand(E, T, T))
    n_neg = neg_mask.sum(-1)                                                   # [E, T]
    has_neg = n_neg > 0
    mean_ell = torch.where(neg_mask, ell, torch.zeros_like(ell)).sum(-1) / n_neg.clamp(min=1)

    # "positive" pair used for the correction: the twin if placed, else the batch average of forced moves.
    ell_twin = -log1m.gather(-1, slot.unsqueeze(-1)).squeeze(-1)               # [E, T]
    batch_plus = ell_twin[forced].mean() if forced.any() else torch.zeros((), device=scores.device)
    ell_plus = torch.where(forced, ell_twin, batch_plus.expand_as(ell_twin))

    if free_debias == 1.0:
        per_view = ((mean_ell - tau_plus * ell_plus) / (1.0 - tau_plus)).clamp(min=0.0)
    else:
        tau = torch.where(forced, torch.full_like(ell_plus, tau_plus), torch.full_like(ell_plus, tau_plus * free_debias))
        per_view = ((mean_ell - tau * ell_plus) / (1.0 - tau)).clamp(min=0.0)
    l_neg = per_view[has_neg].mean() if has_neg.any() else scores.sum() * 0.0

    # --- optional anti-collapse safeguard (rho = 0 by default) -------------------------------------
    loss = l_pos + l_neg
    if rho > 0:
        p_bar = logp.exp().mean(dim=(0, 1))
        loss = loss + rho * (p_bar * torch.log(p_bar + 1e-8)).sum()           # = - rho * H(p_bar)

    with torch.no_grad():
        acc = (logp.argmax(-1) == slot)[forced].float().mean() if forced.any() else torch.zeros(())
    return ExpertLossOut(loss, l_pos.detach(), l_neg.detach(), acc)


# --------------------------------------------------------------------------------------------------
# Invariance: L_inv
# --------------------------------------------------------------------------------------------------
def invariance_kl(logp_clean: Tensor, logp_pert: Tensor, valid: Tensor) -> Tensor:
    """E KL( pi(.|z, C, m) || pi(.|z~, C, m) ) between the policies on the clean and perturbed codes.

    Both log-probs are over the same valid slots (same memory snapshot). The caller keeps omega, psi
    and C frozen: only the encoder receives gradient (README 4.5 and 6.5).
    """
    safe_c = logp_clean.masked_fill(~valid, 0.0)
    safe_p = logp_pert.masked_fill(~valid, 0.0)
    return (logp_clean.exp() * (safe_c - safe_p)).sum(-1).mean()


# --------------------------------------------------------------------------------------------------
# RL: advantages and PPO losses
# --------------------------------------------------------------------------------------------------
def compute_gae(rewards: Tensor, values: Tensor, gamma: float, lam: float) -> tuple[Tensor, Tensor]:
    """Generalized advantage estimation over ONE episode (1-D tensors of length T).

    The episode ends after its last image (value 0 afterwards). With gamma = 0 the advantage is
    r_t - V(s_t): the policy is essentially greedy on the reward. Returns (advantages, returns).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last, next_value = 0.0, 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_value = values[t]
    return adv, adv + values


@dataclass
class PPOOut:
    policy_loss: Tensor   # L_omega: clipped surrogate - alpha * entropy
    value_loss: Tensor    # L_xi
    approx_kl: Tensor     # monitoring
    entropy: Tensor       # monitoring


def ppo_losses(logp_new: Tensor, logp_old: Tensor, adv: Tensor, entropy: Tensor,
               v_new: Tensor, returns: Tensor, clip: float, alpha: float) -> PPOOut:
    """L_omega = -E[min(r A, clip(r) A)] - alpha E[H(pi)],   L_xi = E[(V_xi(s) - G)^2]."""
    ratio = (logp_new - logp_old).exp()
    surrogate = torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)
    policy_loss = -surrogate.mean() - alpha * entropy.mean()
    value_loss = F.mse_loss(v_new, returns)
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean()
    return PPOOut(policy_loss, value_loss, approx_kl, entropy.mean().detach())
