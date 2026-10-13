"""Training loop: learner rollouts and the alternating reward / policy steps (README Sections 6.5 and 7).

One iteration (after a warm-up on the experts alone):

    1. Expert batch    E fixed-structure episodes (experts.py), views re-drawn with the family T.
    2. Learner rollout the CURRENT policy sweeps a minibatch of training images one by one and writes
                       into the persistent cluster memory. No label, no expert (no gradient either).
    3. Reward step     update phi, psi, theta:  L_psi + L_theta + beta * KL + lambda * L_inv
                       (L_psi and L_theta on the expert batch, L_inv on the learner's states).
    4. Policy step     update omega, xi with PPO against the reward, which is frozen.
    End of epoch       the cluster summaries are recomputed with the current encoder.

Who receives which gradient (README 6.5):
    L_psi   -> psi, and phi through z            L_theta -> theta, and phi through z (NOT psi)
    KL      -> phi                               L_inv   -> phi only (omega, psi and C frozen)
    L_omega -> omega, xi only (phi, psi, theta and C are inputs)
"""
from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from . import losses as L
from .encoder import build_encoder
from .experts import ExpertBatch, ExpertEpisodes, OrbitSampler, Transform, split_expert_portion
from .intent import IntentPredictor
from .memory import ClusterMemory, expert_memory_states, valid_slot_mask
from .policy import PolicyNet, ValueNet
from .reward import RewardModel


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # architecture
    num_slots: int = 10            # K, an UPPER bound on the number of clusters
    z_dim: int = 32
    m_dim: int = 16
    hidden: int = 128
    rel_dim: int = 64
    in_channels: int = 3
    image_size: int = 28
    use_z: bool = True             # False: reward and policy only see m (ablation)
    intent_standardize: bool = True

    # experts (Phase 1)
    phase: int = 1
    expert_fraction: float = 0.1   # share of the training set used to build the demonstrations
    num_expert_episodes: int = 4000
    orbits_per_episode: int = 5    # B  (B <= K)
    views_per_orbit: int = 2       # n  (n >= 2)
    episodes_per_batch: int = 32
    d_max: int = 2                 # longest chain of transformations in an orbit
    tau_plus: Optional[float] = None   # None -> 1/K
    rho: float = 0.0               # anti-collapse entropy term, off by default

    # compression and invariance
    beta: float = 1e-3
    beta_warmup_iters: int = 500
    lam: float = 1.0

    # RL
    gamma: float = 0.9
    gae_lambda: float = 0.95
    alpha: float = 0.01            # entropy bonus
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    value_coef: float = 0.5
    center_reward: bool = True     # r(s,a) = R(s,a) - mean over valid slots of R(s,.)

    # optimization
    lr_reward: float = 3e-4        # slower than the policy: the reward is the ruler
    lr_policy: float = 1e-3
    grad_clip: float = 5.0
    learner_batch: int = 128
    reward_warmup_iters: int = 200  # reward / encoder / intent on the experts alone, no learner rollout
    seed: int = 0


@dataclass
class Rollout:
    """What the learner did in one minibatch: the states it saw and the actions it took."""
    idx: Tensor          # [T] dataset indices, in the order they were presented
    x_view: Tensor       # [T, C, H, W] the view shown to the policy
    x_pert: Tensor       # [T, C, H, W] g(view), for L_inv
    z: Tensor            # [T, z_dim]  (deterministic codes)
    m: Tensor            # [T, m_dim]
    summaries: Tensor    # [T, K, m_dim]  snapshot of the memory BEFORE each placement
    empty: Tensor        # [T, K]
    actions: Tensor      # [T]
    logp_old: Tensor     # [T]


@contextmanager
def freeze(*modules: torch.nn.Module):
    """Parameters get no gradient inside the block, but gradients still flow THROUGH the module to its inputs."""
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


# --------------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: TrainConfig, images: Tensor, transform: Transform,
                 expert_ids: Tensor | None = None, device: str | torch.device = "cpu"):
        """images [N, C, H, W] float in [0, 1] (on any device); transform = the nuisance family T."""
        if cfg.phase != 1:
            raise NotImplementedError("Only Phase 1 is implemented (README Section 5.5)")
        self.cfg, self.device = cfg, torch.device(device)
        torch.manual_seed(cfg.seed)

        self.images, self.N = images, images.shape[0]
        self.transform = transform
        self.sampler = OrbitSampler(transform, cfg.d_max)
        K = cfg.num_slots
        self.tau = cfg.tau_plus if cfg.tau_plus is not None else 1.0 / K

        # networks
        dev = self.device
        self.encoder = build_encoder("small_cnn", cfg.z_dim, cfg.in_channels, cfg.image_size).to(dev)
        self.intent = IntentPredictor(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.intent_standardize).to(dev)
        self.reward = RewardModel(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.rel_dim, cfg.use_z).to(dev)
        self.policy = PolicyNet(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.rel_dim, cfg.use_z).to(dev)
        self.value = ValueNet(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.rel_dim, cfg.use_z).to(dev)

        # memory and experts
        self.memory = ClusterMemory(self.N, K, cfg.m_dim, dev)
        if expert_ids is None:
            expert_ids = split_expert_portion(self.N, cfg.expert_fraction, cfg.seed)
        self.experts = ExpertEpisodes.build(expert_ids, cfg.num_expert_episodes, cfg.orbits_per_episode,
                                            cfg.views_per_orbit, K, seed=cfg.seed)

        # two optimizers = the two steps of the alternation
        self.reward_params = list(itertools.chain(self.encoder.parameters(), self.intent.parameters(),
                                                  self.reward.parameters()))
        self.policy_params = list(itertools.chain(self.policy.parameters(), self.value.parameters()))
        self.opt_reward = torch.optim.Adam(self.reward_params, lr=cfg.lr_reward)
        self.opt_policy = torch.optim.Adam(self.policy_params, lr=cfg.lr_policy)

        self.iteration = 0
        self._changes = 0     # images that changed slot this epoch (numerator of the churn)
        self._revisits = 0    # images that had a previous placement this epoch (denominator)

    # ---------------------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------------------
    def _fetch(self, idx: Tensor) -> Tensor:
        return self.images[idx.to(self.images.device)].to(self.device)

    def _beta(self) -> float:
        """Linear warm-up of beta from 0: early on the reward is random and could push the bottleneck
        to throw away useful bits (README 6.5)."""
        w = self.cfg.beta_warmup_iters
        return self.cfg.beta * (1.0 if w <= 0 else min(1.0, self.iteration / w))

    def _clip_step(self, opt: torch.optim.Optimizer, params: list[Tensor], loss: Tensor) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        opt.step()

    # ---------------------------------------------------------------------------------------------
    # 2. Learner rollout
    # ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def learner_rollout(self, idx: Tensor) -> Rollout:
        """The policy places the images of `idx` one after the other. Each placement updates the
        persistent memory, so the next image sees a different state."""
        K, d = self.cfg.num_slots, self.cfg.m_dim
        T = len(idx)
        x_view = self.sampler(self._fetch(idx))
        x_pert = self.transform(x_view)
        z = self.encoder(x_view, sample=False).z
        m = self.intent.mean(z)

        summ_snap = torch.empty(T, K, d, device=self.device)
        empty_snap = torch.empty(T, K, dtype=torch.bool, device=self.device)
        actions = torch.empty(T, dtype=torch.long, device=self.device)
        logps = torch.empty(T, device=self.device)
        for t in range(T):
            summ, empty = self.memory.state()
            summ_snap[t], empty_snap[t] = summ, empty
            a, lp = self.policy.act(z[t:t + 1], m[t:t + 1], summ.unsqueeze(0), empty.unsqueeze(0))
            actions[t], logps[t] = a[0], lp[0]
            item = int(idx[t])
            had_slot = int(self.memory.assign[item]) >= 0
            changed = self.memory.place(item, m[t], int(a[0]))
            self._revisits += int(had_slot)
            self._changes += int(changed)
        return Rollout(idx, x_view, x_pert, z, m, summ_snap, empty_snap, actions, logps)

    # ---------------------------------------------------------------------------------------------
    # 3. Reward step (IRL): phi, psi, theta
    # ---------------------------------------------------------------------------------------------
    def _invariance(self, roll: Rollout) -> Tensor:
        """L_inv: the policy is a FIXED ruler (frozen), the memory and the intent too; the gradient
        only reaches the encoder, through the code of the clean view and of the perturbed view."""
        with freeze(self.policy):
            out_c = self.encoder(roll.x_view, sample=True)
            out_p = self.encoder(roll.x_pert, sample=True)
            with torch.no_grad():
                m_bar = self.intent.mean(out_c.z)          # intent of the CLEAN view, given to both
            logp_c = self.policy.log_probs(out_c.z, m_bar, roll.summaries, roll.empty)
            logp_p = self.policy.log_probs(out_p.z, m_bar, roll.summaries, roll.empty)
        return L.invariance_kl(logp_c, logp_p, valid_slot_mask(roll.empty))

    def reward_step(self, eb: ExpertBatch, roll: Rollout | None) -> dict:
        cfg, K = self.cfg, self.cfg.num_slots
        eb = eb.to(self.device)
        E, T = eb.slot.shape

        # views of the expert episodes, encoded with the CURRENT encoder
        x = self.sampler(self._fetch(eb.image_ids.flatten()))
        enc = self.encoder(x, sample=True)

        # intent: L_psi  (psi's own forward, batch statistics)
        mu_psi, lv_psi = self.intent(enc.z)
        m_star = L.orbit_intent_target(mu_psi.view(E, T, -1), eb.orbit)
        l_psi = L.intent_nll(mu_psi, lv_psi, m_star.reshape(E * T, -1))

        # the reward reads the intent as a CONSTANT: psi gets nothing from L_theta, and the expert's
        # memory is built from the same constants
        with torch.no_grad():
            m_state = self.intent.mean(enc.z)
        summ, empty = expert_memory_states(m_state.view(E, T, -1), eb.slot, K)
        scores = self.reward(enc.z, m_state, summ.reshape(E * T, K, -1), empty.reshape(E * T, K)).view(E, T, K)
        exp = L.expert_loss(scores, empty, eb.slot, eb.is_first, eb.neg_mask, self.tau, cfg.rho)

        kl = L.kl_standard_normal(enc.mu, enc.logvar)
        loss = l_psi + exp.loss + self._beta() * kl
        l_inv = torch.zeros((), device=self.device)
        if roll is not None and cfg.lam > 0:
            l_inv = self._invariance(roll)
            loss = loss + cfg.lam * l_inv

        self._clip_step(self.opt_reward, self.reward_params, loss)
        return {"L_psi": float(l_psi.detach()), "L_pos": float(exp.l_pos), "L_neg": float(exp.l_neg),
                "twin_acc": float(exp.twin_acc), "KL": float(kl.detach()), "L_inv": float(l_inv.detach()),
                "m_std": float(m_state.std(0).mean())}

    # ---------------------------------------------------------------------------------------------
    # 4. Policy step (RL): omega, xi
    # ---------------------------------------------------------------------------------------------
    def policy_step(self, roll: Rollout) -> dict:
        cfg = self.cfg
        valid = valid_slot_mask(roll.empty)
        with torch.no_grad():
            scores = self.reward(roll.z, roll.m, roll.summaries, roll.empty)          # reward frozen
            if cfg.center_reward:
                # softmax only identifies R up to a constant PER STATE; remove it so that the return
                # does not depend on an arbitrary offset (README 4.5 makes the same remark)
                n_valid = valid.sum(-1, keepdim=True)
                scores = scores - (scores * valid).sum(-1, keepdim=True) / n_valid
            r = scores.gather(1, roll.actions.unsqueeze(1)).squeeze(1)                # R(s_t, a_t)
            v_old = self.value(roll.z, roll.m, roll.summaries, roll.empty)
            adv, ret = L.compute_gae(r, v_old, cfg.gamma, cfg.gae_lambda)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(cfg.ppo_epochs):
            logp_all = self.policy.log_probs(roll.z, roll.m, roll.summaries, roll.empty)
            logp_new = logp_all.gather(1, roll.actions.unsqueeze(1)).squeeze(1)
            ent = PolicyNet.entropy(logp_all, roll.empty)
            v_new = self.value(roll.z, roll.m, roll.summaries, roll.empty)
            out = L.ppo_losses(logp_new, roll.logp_old, adv, ent, v_new, ret, cfg.ppo_clip, cfg.alpha)
            self._clip_step(self.opt_policy, self.policy_params, out.policy_loss + cfg.value_coef * out.value_loss)
        return {"return": float(r.sum()), "r_mean": float(r.mean()), "pi_entropy": float(out.entropy),
                "approx_kl": float(out.approx_kl), "L_xi": float(out.value_loss.detach())}

    # ---------------------------------------------------------------------------------------------
    # Loop
    # ---------------------------------------------------------------------------------------------
    def train_iteration(self, idx: Tensor) -> dict:
        eb = self.experts.sample(self.cfg.episodes_per_batch)
        if self.iteration < self.cfg.reward_warmup_iters:
            # warm-up: the reward learns from the forced moves before the policy acts alone
            stats = self.reward_step(eb, None)
        else:
            roll = self.learner_rollout(idx)
            stats = self.reward_step(eb, roll)
            stats.update(self.policy_step(roll))
        self.iteration += 1
        return stats

    @torch.no_grad()
    def recompute_memory(self, batch_size: int = 512) -> None:
        """End of epoch: the encoder moved, so the stored summaries are stale. Recompute them
        from the member lists with the current encoder (README 4.2)."""
        ms = []
        for s in range(0, self.N, batch_size):
            idx = torch.arange(s, min(s + batch_size, self.N))
            ms.append(self.intent.mean(self.encoder(self._fetch(idx), sample=False).z))
        self.memory.recompute(torch.cat(ms))

    def epoch_stats(self) -> dict:
        sizes = self.memory.sizes()
        p = sizes[sizes > 0] / sizes.sum().clamp(min=1)
        entropy = float(-(p * p.log()).sum()) if len(p) else 0.0
        return {"nonempty_slots": self.memory.num_nonempty(),
                "size_entropy": entropy, "K_eff": math.exp(entropy),
                "churn": self._changes / max(self._revisits, 1)}

    def fit(self, epochs: int, log_fn: Callable[[dict], None] = print) -> list[dict]:
        history = []
        for ep in range(epochs):
            perm = torch.randperm(self.N)
            self._changes = self._revisits = 0
            agg: dict[str, list[float]] = {}
            for s in range(0, self.N, self.cfg.learner_batch):
                idx = perm[s:s + self.cfg.learner_batch]
                if len(idx) < 2:
                    continue
                for k, v in self.train_iteration(idx).items():
                    agg.setdefault(k, []).append(v)
            self.recompute_memory()
            row = {"epoch": ep, "iter": self.iteration, **{k: sum(v) / len(v) for k, v in agg.items()},
                   **self.epoch_stats()}
            history.append(row)
            log_fn(row)
        return history

    # ---------------------------------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: Tensor | None = None, batch_size: int = 512) -> Tensor:
        """a = argmax_a pi(a | z, m, C) with the memory C frozen (README 6.7). Returns slot ids [N].

        An image whose argmax is the empty slot gets the id of the first empty slot: all those images
        end up in one "outlier" group, since nothing is written into the frozen memory.
        """
        images = self.images if images is None else images
        summ, empty = self.memory.state()
        out = []
        for s in range(0, images.shape[0], batch_size):
            x = images[s:s + batch_size].to(self.device)
            z = self.encoder(x, sample=False).z
            m = self.intent.mean(z)
            B = x.shape[0]
            logp = self.policy.log_probs(z, m, summ.unsqueeze(0).expand(B, -1, -1), empty.unsqueeze(0).expand(B, -1))
            out.append(logp.argmax(-1))
        return torch.cat(out)

    def save(self, path: str) -> None:
        torch.save({"cfg": asdict(self.cfg), "iteration": self.iteration,
                    "encoder": self.encoder.state_dict(), "intent": self.intent.state_dict(),
                    "reward": self.reward.state_dict(), "policy": self.policy.state_dict(),
                    "value": self.value.state_dict(), "assign": self.memory.assign,
                    "member_m": self.memory.member_m, "counts": self.memory.counts,
                    "summaries": self.memory.summaries}, path)
