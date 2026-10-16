"""The CRP prior on placements (cfg.slot_prior = "crp", docs 5.24). Run: python -m pytest tests/test_slot_prior.py"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cibirl.memory import crp_log_prior, valid_slot_mask   # noqa: E402
from cibirl.policy import PolicyNet                          # noqa: E402
from cibirl.train import TrainConfig, Trainer                # noqa: E402
from toy_data import background_redraw, make_toy             # noqa: E402

NEG_INF = float("-inf")


def make_trainer(**kw):
    x, _, _ = make_toy(n_per_class=60, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=200,
                      expert_fraction=0.5, learner_batch=32, episodes_per_batch=8, beta=1e-2,
                      beta_warmup_iters=0, reward_warmup_iters=2, **kw)
    return Trainer(cfg, x, background_redraw)


# -- configuration ---------------------------------------------------------------------------------
def test_defaults_keep_the_old_behaviour_and_the_config_validates():
    cfg = TrainConfig()
    assert cfg.slot_prior == "uniform" and cfg.crp_alpha == 1.0
    cfg = TrainConfig.from_dict({"slot_prior": "crp", "crp_alpha": 10})
    assert cfg.slot_prior == "crp" and cfg.crp_alpha == 10.0
    with pytest.raises(ValueError):
        make_trainer(slot_prior="dirichlet")
    with pytest.raises(ValueError):
        make_trainer(slot_prior="crp", crp_alpha=0.0)


# -- the prior itself ------------------------------------------------------------------------------
def test_masses_are_sizes_and_alpha_over_the_valid_slots():
    counts = torch.tensor([3.0, 1.0, 0.0, 0.0])
    empty = counts == 0
    lp = crp_log_prior(counts, empty, alpha=2.0)
    p = lp.exp()
    assert p.sum() == pytest.approx(1.0)
    assert p[:3].tolist() == pytest.approx([3 / 6, 1 / 6, 2 / 6])        # sizes 3, 1 and the open mass alpha = 2
    assert lp[3] == NEG_INF                                               # only ONE empty slot is a legal action


def test_the_image_is_withdrawn_from_its_own_slot():
    counts = torch.tensor([3.0, 1.0, 0.0, 0.0])
    empty = counts == 0
    in_big = crp_log_prior(counts, empty, 2.0, prev=0).exp()              # slot 0 loses the image: 3 -> 2
    assert in_big[:3].tolist() == pytest.approx([2 / 5, 1 / 5, 2 / 5])
    alone = crp_log_prior(counts, empty, 2.0, prev=1).exp()               # slot 1 had only this image: it is a new table
    assert alone[:3].tolist() == pytest.approx([3 / 7, 2 / 7, 2 / 7])
    nowhere = crp_log_prior(counts, empty, 2.0, prev=-1).exp()
    assert nowhere[:3].tolist() == pytest.approx([3 / 6, 1 / 6, 2 / 6])


def test_a_tie_is_broken_in_proportion_to_the_sizes():
    counts = torch.tensor([150.0, 50.0, 0.0, 0.0])
    p = crp_log_prior(counts, counts == 0, alpha=1.0).exp()
    assert (p[0] / p[1]).item() == pytest.approx(3.0)


def test_the_batch_version_matches_the_row_by_row_one():
    counts = torch.tensor([5.0, 2.0, 1.0, 0.0])
    empty = counts == 0
    prev = torch.tensor([0, 2, -1, 1])
    batch = crp_log_prior(counts, empty.unsqueeze(0).expand(4, -1), 1.5, prev)
    for i in range(4):
        row = crp_log_prior(counts, empty, 1.5, int(prev[i]))
        assert torch.allclose(batch[i], row)


def test_alpha_must_be_positive():
    with pytest.raises(ValueError):
        crp_log_prior(torch.tensor([1.0, 0.0]), torch.tensor([False, True]), alpha=0.0)


# -- the policy ------------------------------------------------------------------------------------
def _state(b=6, k=5, z_dim=5, m_dim=4):
    torch.manual_seed(0)
    z, m = torch.randn(b, z_dim), torch.randn(b, m_dim)
    summ = torch.randn(b, k, m_dim)
    empty = torch.tensor([[False, False, True, False, True]] * b)
    return z, m, summ, empty


def test_no_prior_is_exactly_the_old_policy_and_a_prior_is_an_additive_logit_offset():
    torch.manual_seed(1)
    net = PolicyNet(5, 4)
    z, m, summ, empty = _state()
    assert torch.equal(net.log_probs(z, m, summ, empty), net.log_probs(z, m, summ, empty, None))
    counts = torch.tensor([4.0, 9.0, 0.0, 2.0, 0.0])
    lp0 = crp_log_prior(counts, empty[0], 1.0).unsqueeze(0).expand(6, -1)
    with_prior = net.log_probs(z, m, summ, empty, lp0)
    valid = valid_slot_mask(empty)
    expected = (net.scorer(z, m, summ, empty) + lp0.masked_fill(~valid, 0.0)).masked_fill(~valid, NEG_INF).log_softmax(-1)
    assert torch.allclose(with_prior, expected)
    assert torch.isfinite(with_prior[valid]).all() and (with_prior[~valid] == NEG_INF).all()


def test_the_prior_gets_no_gradient_trouble_on_invalid_slots():
    net = PolicyNet(5, 4)
    z, m, summ, empty = _state()
    counts = torch.tensor([4.0, 9.0, 0.0, 2.0, 0.0])
    lp0 = crp_log_prior(counts, empty[0], 1.0).unsqueeze(0).expand(6, -1)
    loss = -net.log_probs(z, m, summ, empty, lp0).masked_fill(~valid_slot_mask(empty), 0.0).sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_where_the_scores_tie_the_policy_samples_in_proportion_to_the_sizes():
    torch.manual_seed(0)
    net = PolicyNet(5, 4)
    for p in net.parameters():
        p.data.zero_()                                    # all scores equal: the reward cannot tell the slots apart
    b = 4000
    z, m = torch.randn(b, 5), torch.randn(b, 4)
    summ = torch.zeros(b, 5, 4)
    empty = torch.tensor([[False, False, True, True, True]] * b)
    counts = torch.tensor([150.0, 50.0, 0.0, 0.0, 0.0])
    lp0 = crp_log_prior(counts, empty[0], 1.0).unsqueeze(0).expand(b, -1)
    a, _ = net.act(z, m, summ, empty, log_prior=lp0)
    share = (a == 0).float().mean().item()
    assert share == pytest.approx(150 / 201, abs=0.03)    # 150 / (150 + 50 + alpha)
    assert (a >= 2).float().mean().item() == pytest.approx(1 / 201, abs=0.01)


def test_a_sweep_of_resampling_keeps_the_proportions_instead_of_restoring_50_50():
    """The point of the prior: with scores that tie, a sweep in which every image is withdrawn and placed again keeps the
    sizes (a martingale), whereas a uniform tie-break pulls two duplicate slots back to 50/50."""
    def sweep(use_prior: bool, seed: int, n_a=100, n_b=20) -> float:
        g = torch.Generator().manual_seed(seed)
        slot = torch.tensor([0] * n_a + [1] * n_b)
        for i in torch.randperm(len(slot), generator=g).tolist():
            counts = torch.bincount(slot, minlength=3).float()
            empty = counts == 0
            if use_prior:
                lp = crp_log_prior(counts, empty, 1.0, int(slot[i]))
                probs = lp.exp()
            else:
                probs = valid_slot_mask(empty).float()
                probs = probs / probs.sum()
            probs[2:] = 0.0                               # only the two duplicates are candidates in this check
            slot[i] = torch.multinomial(probs / probs.sum(), 1, generator=g).item()
        return (slot == 0).float().mean().item()
    crp = sum(sweep(True, s) for s in range(40)) / 40
    uni = sum(sweep(False, s) for s in range(40)) / 40
    assert crp == pytest.approx(100 / 120, abs=0.06)
    assert uni == pytest.approx(0.5, abs=0.06)


# -- the trainer -----------------------------------------------------------------------------------
def test_the_rollout_carries_the_prior_only_when_asked():
    tr = make_trainer(slot_prior="crp", crp_alpha=2.0)
    roll = tr.learner_rollout(torch.arange(32))
    assert roll.log_prior.shape == (32, 4)
    valid = valid_slot_mask(roll.empty)
    assert torch.allclose(roll.log_prior.exp().mul(valid).sum(-1), torch.ones(32))
    assert make_trainer().learner_rollout(torch.arange(32)).log_prior is None


def test_a_few_iterations_run_with_the_prior_including_the_teacher_phase():
    for extra in ({}, {"policy_teacher_iters": 3}):
        tr = make_trainer(slot_prior="crp", crp_alpha=1.0, **extra)
        rows = tr.fit(2)
        assert len(rows) == 2 and all(math.isfinite(v) for v in rows[-1].values() if isinstance(v, float))


def test_predict_uses_the_prior_and_the_uniform_default_does_not():
    crp, uni = make_trainer(slot_prior="crp", crp_alpha=1.0), make_trainer()
    for tr in (crp, uni):
        for p in tr.policy.parameters():
            p.data.zero_()                                # tie between every slot: only the prior can decide
        n = tr.N
        tr.memory.assign = torch.tensor([1] * (3 * n // 4) + [0] * (n - 3 * n // 4))
        tr.memory.recompute(tr.intents())
    assert (crp.predict() == 1).all()                      # the bigger slot wins every image
    assert (uni.predict() == 0).all()                      # exact tie: argmax picks the first index, as before
