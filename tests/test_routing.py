"""Gradient routing (README 6.5): who may update what. Run: python tests/test_routing.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cibirl.losses as L                                # noqa: E402
from cibirl.train import TrainConfig, Trainer            # noqa: E402
from toy_data import background_redraw, make_toy         # noqa: E402


def make_trainer(**kw):
    x, _, _ = make_toy(n_per_class=60, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=200,
                      expert_fraction=0.5, learner_batch=32, episodes_per_batch=8, beta=1e-2,
                      beta_warmup_iters=0, **kw)
    return Trainer(cfg, x, background_redraw)


def snap(*mods):
    return [p.detach().clone() for m in mods for p in m.parameters()]


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def make_state():
    tr = make_trainer()
    tr.iteration = 10_000                                   # past all warm-ups
    roll = tr.learner_rollout(torch.arange(32))
    eb = tr.experts.sample(8)
    return tr, roll, eb


def test_reward_step_does_not_touch_policy():
    tr, roll, eb = make_state()
    pol_before = snap(tr.policy, tr.value)
    enc_before = snap(tr.encoder, tr.intent, tr.reward)
    tr.reward_step(eb, roll)
    assert same(pol_before, snap(tr.policy, tr.value)), "the reward step must not update omega / xi"
    assert not same(enc_before, snap(tr.encoder, tr.intent, tr.reward))
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in tr.policy.parameters()), "L_inv leaked into omega"


def test_policy_step_only_touches_policy_and_critic():
    tr, roll, _ = make_state()
    frozen_before = snap(tr.encoder, tr.intent, tr.reward)
    pol_before = snap(tr.policy, tr.value)
    tr.policy_step(roll)
    assert same(frozen_before, snap(tr.encoder, tr.intent, tr.reward)), "RL must never reach phi, psi, theta"
    assert not same(pol_before, snap(tr.policy, tr.value))


def test_reward_loss_does_not_update_intent():
    """Remove L_psi (zero loss, still connected to the graph): psi must not move at all."""
    tr, roll, eb = make_state()
    orig = L.intent_nll
    L.intent_nll = lambda mu, lv, t: (mu * 0).sum() + (lv * 0).sum()
    try:
        psi_before, enc_before, rew_before = snap(tr.intent), snap(tr.encoder), snap(tr.reward)
        tr.cfg.lam = 0.0
        tr.reward_step(eb, roll)
    finally:
        L.intent_nll = orig
    assert same(psi_before, snap(tr.intent)), "psi received a gradient from L_theta / KL"
    assert not same(enc_before, snap(tr.encoder)) and not same(rew_before, snap(tr.reward))


def test_invariance_loss_only_updates_the_encoder():
    """Keep only lambda * L_inv: only phi may move."""
    tr, roll, eb = make_state()
    orig_e, orig_i = L.expert_loss, L.intent_nll
    L.intent_nll = lambda mu, lv, t: (mu * 0).sum() + (lv * 0).sum()
    def zero_expert(scores, *a, **k):
        out = orig_e(scores, *a, **k)
        out.loss = (scores * 0).sum()
        return out
    L.expert_loss = zero_expert
    try:
        tr.cfg.beta = 0.0
        tr.cfg.lam = 1.0
        enc, psi, rew, pol = snap(tr.encoder), snap(tr.intent), snap(tr.reward), snap(tr.policy)
        stats = tr.reward_step(eb, roll)
    finally:
        L.expert_loss, L.intent_nll = orig_e, orig_i
    assert not same(enc, snap(tr.encoder)), "L_inv did not reach the encoder"
    assert same(psi, snap(tr.intent)) and same(rew, snap(tr.reward)) and same(pol, snap(tr.policy))
    assert stats["L_inv"] >= 0


if __name__ == "__main__":
    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t()
        print("ok ", t.__name__)
