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

import difflib
import itertools
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from . import losses as L
from .diagnostics import PlacementDiagnostics
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
    expert_oracle_link: float = 0.0  # ORACLE DIAGNOSTIC, uses the true classes (needs oracle_labels): probability
                                     # that the orbits of one class are linked in an expert episode. 0 = Phase 1.
    oracle_warm_memory: int = 0      # ORACLE DIAGNOSTIC (true classes): when the learner starts acting, the memory
                                     # already holds this many images of every class, slot c = class c. 0 = empty.
    burn_in_epochs: int = 0          # once the learner has been placing images for this many epochs (the reward
                                     # warm-up does not count), the memory is rebuilt from a random sample of the
                                     # members of each cluster (no labels). 0 = never.
    burn_in_fraction: float = 0.1    # share of each cluster kept by the burn-in (at least one member)
    recompute_every: int = 0         # also recompute the summaries from the current encoder every this many
                                     # iterations (the stored intents drift while the encoder trains). 0 = once per epoch.
    freeze_repr_after: int = 0       # once the learner has been placing images for this many epochs, the learning rate of
                                     # the encoder and of the intent predictor is multiplied by `repr_lr_scale`. 0 = never.
    repr_lr_scale: float = 0.0       # 0 = frozen (weights and BatchNorm statistics: m no longer moves), 0.1 = slowed

    # compression and invariance
    beta: float = 1e-4             # 1e-3 degraded the reward on the toy (docs/protocol-decisions.md, 3 and 5)
    beta_warmup_iters: int = 500
    lam: float = 1.0

    # RL
    gamma: float = 0.9
    gae_lambda: float = 0.95
    alpha: float = 0.01            # entropy bonus
    policy_temperature: float = 1.0  # pi = softmax(scores / T); T < 1: sharper policy, less exploration
    consolidate: bool = False      # end of epoch: rebuild the memory from the greedy partition (argmax of pi,
                                   # memory frozen) instead of the sampled placements of the epoch
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

    # ---------------------------------------------------------------------------------------------
    # Loading from YAML (the VALUES of an experiment live in configs/, this class is the schema and
    # the defaults). Several files are merged left to right, then `overrides` win:
    #     TrainConfig.from_yaml("configs/toy.yml", "configs/ablations/gamma0.yml", seed=3)
    # A file may start from others with `base: other.yml` (path relative to the file, or a list).
    # ---------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Strict: an unknown key is an error (with a suggestion), values are cast to the field type."""
        known = {f.name: f.type for f in fields(cls)}
        for key in d:
            if key not in known:
                hint = difflib.get_close_matches(key, known, n=1)
                raise KeyError(f"unknown config key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        return cls(**{k: _coerce(k, v, known[k]) for k, v in d.items()})

    @classmethod
    def _load_one(cls, path: str | Path, _chain: tuple = ()) -> dict[str, Any]:
        import yaml  # imported here so that the package does not need PyYAML unless a YAML is loaded
        path = Path(path).resolve()
        if path in _chain:
            raise ValueError(f"circular 'base' in {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}   # -sig: tolerate a BOM (Notepad)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: a config file must be a mapping key: value")
        bases = raw.pop("base", [])
        merged: dict[str, Any] = {}
        for b in ([bases] if isinstance(bases, str) else bases):
            merged.update(cls._load_one(path.parent / b, _chain + (path,)))
        merged.update(raw)
        return merged

    @classmethod
    def from_yaml(cls, *paths: str | Path, **overrides: Any) -> "TrainConfig":
        merged: dict[str, Any] = {}
        for p in paths:
            merged.update(cls._load_one(p))
        merged.update(overrides)
        return cls.from_dict(merged)

    def to_yaml(self, path: str | Path) -> None:
        """Write the RESOLVED config (defaults + files + overrides) next to a run, to reproduce it."""
        import yaml
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


def _coerce(name: str, value: Any, type_str: str) -> Any:
    """Cast a value read from YAML to the declared field type. Needed because YAML 1.1 reads `1e-3`
    (no dot) as a STRING, and because `True` would silently become 1 / 1.0 in an int / float field."""
    optional = type_str.startswith("Optional[")
    base = type_str[len("Optional["):-1] if optional else type_str
    if value is None:
        if optional:
            return None
        raise ValueError(f"config '{name}' cannot be null")
    if base == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"config '{name}' expects true/false, got {value!r}")
        return value
    if isinstance(value, bool):
        raise ValueError(f"config '{name}' expects a number, got {value!r}")
    try:
        if base == "float":
            return float(value)
        if base == "int":
            f = float(value)
            if f != int(f):
                raise ValueError
            return int(f)
    except (TypeError, ValueError):
        raise ValueError(f"config '{name}' expects {base}, got {value!r}") from None
    return value


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
                 expert_ids: Tensor | None = None, device: str | torch.device = "cpu",
                 oracle_labels: Tensor | None = None, eval_labels: Tensor | None = None):
        """images [N, C, H, W] float in [0, 1] (on any device); transform = the nuisance family T.

        `oracle_labels` [N] is read ONLY by the oracle diagnostics (cfg.expert_oracle_link > 0 or
        cfg.oracle_warm_memory > 0), which cheat on purpose. `eval_labels` [N] only feeds the `eval_*` placement
        statistics of cibirl/diagnostics.py: it never influences training (a test checks it)."""
        if cfg.phase != 1:
            raise NotImplementedError("Only Phase 1 is implemented (README Section 5.5)")
        if not 0.0 <= cfg.expert_oracle_link <= 1.0:
            raise ValueError(f"expert_oracle_link must be in [0, 1], got {cfg.expert_oracle_link}")
        if cfg.oracle_warm_memory < 0 or cfg.burn_in_epochs < 0 or not 0.0 < cfg.burn_in_fraction <= 1.0:
            raise ValueError("oracle_warm_memory, burn_in_epochs must be >= 0 and burn_in_fraction in (0, 1]")
        if cfg.recompute_every < 0 or cfg.freeze_repr_after < 0 or not 0.0 <= cfg.repr_lr_scale <= 1.0:
            raise ValueError("recompute_every, freeze_repr_after must be >= 0 and repr_lr_scale in [0, 1]")
        oracle_on = cfg.expert_oracle_link > 0 or cfg.oracle_warm_memory > 0
        if oracle_on and oracle_labels is None:
            raise ValueError("expert_oracle_link / oracle_warm_memory are oracle diagnostics: pass oracle_labels "
                             "to the Trainer")
        self._oracle_labels = oracle_labels.cpu().long() if oracle_on else None
        if cfg.oracle_warm_memory > 0 and int(self._oracle_labels.max()) + 1 > cfg.num_slots:
            raise ValueError("oracle_warm_memory needs one slot per class (num_slots >= number of classes)")
        self._warmed = False
        self._acting_epochs = 0
        self._diag = PlacementDiagnostics(eval_labels, cfg.num_slots) if eval_labels is not None else None
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
        self.policy = PolicyNet(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.rel_dim, cfg.use_z,
                                temperature=cfg.policy_temperature).to(dev)
        self.value = ValueNet(cfg.z_dim, cfg.m_dim, cfg.hidden, cfg.rel_dim, cfg.use_z).to(dev)

        # memory and experts
        self.memory = ClusterMemory(self.N, K, cfg.m_dim, dev)
        if expert_ids is None:
            expert_ids = split_expert_portion(self.N, cfg.expert_fraction, cfg.seed)
        self.experts = ExpertEpisodes.build(expert_ids, cfg.num_expert_episodes, cfg.orbits_per_episode,
                                            cfg.views_per_orbit, K, seed=cfg.seed)

        # two optimizers = the two steps of the alternation
        repr_params = list(itertools.chain(self.encoder.parameters(), self.intent.parameters()))
        self.reward_params = repr_params + list(self.reward.parameters())
        self.policy_params = list(itertools.chain(self.policy.parameters(), self.value.parameters()))
        # two groups (the representation, then the reward head) so that the representation alone can be frozen or slowed
        self.opt_reward = torch.optim.Adam([{"params": repr_params},
                                            {"params": list(self.reward.parameters())}], lr=cfg.lr_reward)
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
            prev = int(self.memory.assign[item])
            had_slot = prev >= 0
            if self._diag is not None:                       # evaluation only: reads the state before the update
                self._diag.record(item, int(a[0]), prev)
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
        eb = self.experts.sample(self.cfg.episodes_per_batch, link_labels=self._oracle_labels,
                                 link_prob=self.cfg.expert_oracle_link)
        if self.iteration < self.cfg.reward_warmup_iters:
            # warm-up: the reward learns from the forced moves before the policy acts alone
            stats = self.reward_step(eb, None)
        else:
            if self.cfg.oracle_warm_memory > 0 and not self._warmed:
                self._warm_memory()
            roll = self.learner_rollout(idx)
            stats = self.reward_step(eb, roll)
            stats.update(self.policy_step(roll))
        self.iteration += 1
        if self.cfg.recompute_every and self.iteration % self.cfg.recompute_every == 0 and bool((self.memory.assign >= 0).any()):
            self.memory.recompute(self.intents())
        return stats

    def _slow_representation(self) -> None:
        """Multiply the learning rate of the encoder and of the intent predictor by `repr_lr_scale`. At 0 the
        BatchNorm statistics of the intent are frozen too (momentum 0), otherwise m would keep moving."""
        self.opt_reward.param_groups[0]["lr"] = self.cfg.lr_reward * self.cfg.repr_lr_scale
        if self.cfg.repr_lr_scale == 0.0 and isinstance(self.intent.norm, torch.nn.BatchNorm1d):
            self.intent.norm.momentum = 0.0

    @torch.no_grad()
    def _warm_memory(self) -> None:
        """ORACLE DIAGNOSTIC: put `oracle_warm_memory` random images of every class into slot = class, so that the
        learner starts acting with an informative, complete memory instead of an empty one."""
        self._warmed = True
        labels = self._oracle_labels
        for c in range(int(labels.max()) + 1):
            ids = (labels == c).nonzero().flatten()
            pick = ids[torch.randperm(len(ids))[:self.cfg.oracle_warm_memory]]
            for i, m_i in zip(pick.tolist(), self.intents(self.images[pick])):
                self.memory.place(i, m_i, c)
        if self._diag is not None:
            self._diag.sync(self.memory.assign)

    @torch.no_grad()
    def intents(self, images: Tensor | None = None, batch_size: int = 512) -> Tensor:
        """m = mu_psi(mu_phi(x)) for every image, deterministic, with the current networks: [N, m_dim]."""
        images = self.images if images is None else images
        out = []
        for s in range(0, images.shape[0], batch_size):
            x = images[s:s + batch_size].to(self.device)
            out.append(self.intent.mean(self.encoder(x, sample=False).z))
        return torch.cat(out)

    @torch.no_grad()
    def recompute_memory(self, batch_size: int = 512) -> None:
        """End of epoch: the encoder moved, so the stored summaries are stale. Recompute them
        from the member lists with the current encoder (README 4.2).

        With `consolidate`, the member lists are first replaced by the greedy partition (argmax of pi against
        the memory just recomputed), then the summaries are recomputed once more: the sampled placements of
        the exploration are forgotten, the next epoch starts from the partition the model would output."""
        m_all = self.intents(batch_size=batch_size)
        self.memory.recompute(m_all)
        if self.cfg.consolidate:
            self.memory.assign = self.predict(batch_size=batch_size).to(self.memory.device)
            self.memory.recompute(m_all)

    def epoch_stats(self) -> dict:
        sizes = self.memory.sizes()
        p = sizes[sizes > 0] / sizes.sum().clamp(min=1)
        entropy = float(-(p * p.log()).sum()) if len(p) else 0.0
        return {"nonempty_slots": self.memory.num_nonempty(),
                "size_entropy": entropy, "K_eff": math.exp(entropy),
                "churn": self._changes / max(self._revisits, 1)}

    def fit(self, epochs: int, log_fn: Callable[[dict], Any] = print) -> list[dict]:
        """Up to `epochs` epochs. `log_fn(row)` is called after each one; if it returns True, training stops
        there (this is how a convergence monitor ends a run, see convergence.py)."""
        history = []
        for ep in range(epochs):
            perm = torch.randperm(self.N)
            self._changes = self._revisits = 0
            if self._diag is not None:
                self._diag.start_epoch(self.memory.assign)
            agg: dict[str, list[float]] = {}
            for s in range(0, self.N, self.cfg.learner_batch):
                idx = perm[s:s + self.cfg.learner_batch]
                if len(idx) < 2:
                    continue
                for k, v in self.train_iteration(idx).items():
                    agg.setdefault(k, []).append(v)
            report = {}
            if self._diag is not None:
                report = self._diag.report()                 # placements of this epoch (before the recompute)
            self.recompute_memory()
            if self._diag is not None:
                self._diag.sync(self.memory.assign)
                report["eval_purity"] = self._diag.purity()   # purity of the memory the next epoch starts from
            row = {"epoch": ep, "iter": self.iteration, **{k: sum(v) / len(v) for k, v in agg.items()},
                   **self.epoch_stats(), **report}
            history.append(row)
            if log_fn(row):
                break
            if bool((self.memory.assign >= 0).any()):
                self._acting_epochs += 1                      # epochs since the learner started to place images
            if self.cfg.burn_in_epochs and self._acting_epochs == self.cfg.burn_in_epochs:
                self.memory.subsample(self.intents(), self.cfg.burn_in_fraction)
            if self.cfg.freeze_repr_after and self._acting_epochs == self.cfg.freeze_repr_after:
                self._slow_representation()
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
