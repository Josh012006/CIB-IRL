"""The CRP prior on placements (cfg.slot_prior = "crp", docs 5.24). Run: python -m pytest tests/test_slot_prior.py"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cibirl.memory import crp_log_prior, gated_prior_offset, valid_slot_mask   # noqa: E402
from cibirl.policy import PolicyNet                          # noqa: E402
from cibirl.train import TrainConfig, Trainer                # noqa: E402
from toy_data import background_redraw, make_toy             # noqa: E402

NEG_INF = float("-inf")


def make_trainer(**kw):
    x, _, _ = make_toy(n_per_class=60, n_classes=4)
    kw.setdefault("num_slots", 4)
    cfg = TrainConfig(image_size=16, orbits_per_episode=3, num_expert_episodes=200,
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


# -- the exponent nu (docs 5.25) -------------------------------------------------------------------
def test_power_one_is_the_crp_and_power_two_squares_the_ratio_at_a_tie():
    counts = torch.tensor([150.0, 50.0, 0.0, 0.0])
    empty = counts == 0
    assert torch.equal(crp_log_prior(counts, empty, 1.0), crp_log_prior(counts, empty, 1.0, None, 1.0))
    for nu in (2.0, 3.0):
        p = crp_log_prior(counts, empty, 1.0, None, nu).exp()
        assert p.sum() == pytest.approx(1.0)
        assert (p[0] / p[1]).item() == pytest.approx(3.0 ** nu)
        assert p[3] == 0.0                                        # still one empty slot only


def test_power_must_be_positive_and_the_config_validates_it():
    with pytest.raises(ValueError):
        crp_log_prior(torch.tensor([1.0, 0.0]), torch.tensor([False, True]), 1.0, None, 0.0)
    assert TrainConfig().crp_power == 1.0 and TrainConfig.from_dict({"crp_power": 2}).crp_power == 2.0
    with pytest.raises(ValueError):
        make_trainer(slot_prior="crp", crp_power=0.0)


def test_nu_above_one_drifts_to_the_bigger_twin_where_nu_one_stays_put():
    """Resampling sweeps over two tied duplicates: nu = 1 keeps the share (martingale), nu = 2 pushes it to 1."""
    def sweep(nu: float, seed: int, n_a=70, n_b=50, sweeps=6) -> float:
        g = torch.Generator().manual_seed(seed)
        slot = torch.tensor([0] * n_a + [1] * n_b)
        for _ in range(sweeps):
            for i in torch.randperm(len(slot), generator=g).tolist():
                counts = torch.bincount(slot, minlength=3).float()
                probs = crp_log_prior(counts, counts == 0, 1.0, int(slot[i]), nu).exp()
                probs[2:] = 0.0
                slot[i] = torch.multinomial(probs / probs.sum(), 1, generator=g).item()
        return (slot == 0).float().mean().item()
    one = sum(sweep(1.0, s) for s in range(30)) / 30
    two = sum(sweep(2.0, s) for s in range(30)) / 30
    assert one == pytest.approx(70 / 120, abs=0.08)
    assert two > 0.8


def test_a_few_iterations_run_with_the_exponent():
    rows = make_trainer(slot_prior="crp", crp_alpha=1.0, crp_power=2.0).fit(2)
    assert len(rows) == 2 and all(math.isfinite(v) for v in rows[-1].values() if isinstance(v, float))


# -- the gate: the prior only among slots the scores cannot tell apart (docs 5.26) -------------------
def _gate_state():
    counts = torch.tensor([300.0, 100.0, 40.0, 0.0, 0.0])
    empty = counts == 0
    valid = valid_slot_mask(empty)
    lp = crp_log_prior(counts, empty, 1.0)
    return valid, lp


def test_no_gate_is_the_prior_everywhere_and_the_default_policy_has_no_gate():
    valid, lp = _gate_state()
    scores = torch.tensor([0.0, 0.0, -30.0, -5.0, 0.0])
    assert torch.equal(gated_prior_offset(scores, valid, lp, None), lp.masked_fill(~valid, 0.0))
    assert PolicyNet(5, 4).prior_gate is None and TrainConfig().crp_gate < 0
    assert make_trainer(slot_prior="crp").policy.prior_gate is None
    assert make_trainer(slot_prior="crp", crp_gate=1.0).policy.prior_gate == 1.0


def test_the_gate_redistributes_the_mass_of_the_tied_set_and_leaves_the_rest_alone():
    valid, lp = _gate_state()
    scores = torch.tensor([0.0, 0.3, -30.0, -5.0, 7.0])           # slots 0 and 1 tied (gap 0.3 <= 1); slot 4 is invalid
    off = gated_prior_offset(scores, valid, lp, gate=1.0)
    assert off[2] == 0.0 and off[3] == 0.0 and off[4] == 0.0       # outside the set (or invalid): untouched
    base = scores.masked_fill(~valid, float("-inf")).softmax(-1)
    new = (scores + off).masked_fill(~valid, float("-inf")).softmax(-1)
    assert new[:2].sum() == pytest.approx(base[:2].sum().item())     # the total mass of the set is unchanged
    assert new[2] == pytest.approx(base[2].item()) and new[3] == pytest.approx(base[3].item())
    assert (new[0] / new[1]).item() == pytest.approx((base[0] / base[1]).item() * 3.0, rel=1e-5)   # sizes 300 vs 100 inside the set


def test_exact_ties_are_broken_as_before_and_a_clearly_lower_open_slot_is_not_taxed_by_size():
    counts = torch.tensor([150.0, 50.0, 0.0, 0.0])
    empty = counts == 0
    valid = valid_slot_mask(empty)
    lp = crp_log_prior(counts, empty, 1.0)
    tie = torch.zeros(4)
    ungated = (tie + gated_prior_offset(tie, valid, lp, None)).masked_fill(~valid, float("-inf")).softmax(-1)
    gated = (tie + gated_prior_offset(tie, valid, lp, 0.5)).masked_fill(~valid, float("-inf")).softmax(-1)
    assert torch.allclose(ungated, gated)                            # everything is tied: the gate changes nothing
    # open slot (index 2) scores 1 nat below the two occupied ones: with the gate its odds are exp(-1), not exp(-1) * alpha / n
    scores = torch.tensor([0.0, 0.0, -1.0, 0.0])
    p = (scores + gated_prior_offset(scores, valid, lp, 0.5)).masked_fill(~valid, float("-inf")).softmax(-1)
    assert (p[2] / p[0]).item() == pytest.approx(math.exp(-1.0) / 1.5, rel=1e-5)   # slot 0 carries 3/4 of the tied set's mass 2 (sizes 150 vs 50)
    p_un = (scores + gated_prior_offset(scores, valid, lp, None)).masked_fill(~valid, float("-inf")).softmax(-1)
    assert p[2] > 20 * p_un[2]                                       # without the gate the size bias crushes the open slot


def test_the_gate_has_no_gradient_and_the_policy_trains_with_it():
    valid, lp = _gate_state()
    scores = torch.tensor([0.0, 0.3, -30.0, -5.0, 7.0], requires_grad=True)
    off = gated_prior_offset(scores, valid, lp, gate=1.0)
    assert not off.requires_grad
    for extra in ({}, {"policy_teacher_iters": 3}):
        rows = make_trainer(slot_prior="crp", crp_alpha=1.0, crp_gate=1.0, **extra).fit(2)
        assert len(rows) == 2 and all(math.isfinite(v) for v in rows[-1].values() if isinstance(v, float))


# -- advantage normalisation switch (docs 5.44) ------------------------------------------------------
def test_adv_norm_switch_default_on_and_policy_step_runs_without_it():
    assert TrainConfig().adv_norm is True
    tr = make_trainer(adv_norm=False)
    roll = tr.learner_rollout(torch.arange(32))
    out = tr.policy_step(roll)
    assert math.isfinite(out["pi_entropy"]) and math.isfinite(out["approx_kl"])


# -- resuming a state written before a field existed (docs 5.45) ---------------------------------------
def test_load_state_accepts_a_saved_config_without_a_new_field_at_its_default(tmp_path):
    tr = make_trainer()
    path = tmp_path / "state.pt"
    tr.save_state(str(path))
    st = torch.load(path, weights_only=False)
    del st["cfg"]["adv_norm"]                               # a state written before adv_norm existed
    torch.save(st, path)
    make_trainer().load_state(str(path))                    # default value: the same configuration

    with pytest.raises(ValueError, match="adv_norm"):       # not the default: another configuration
        make_trainer(adv_norm=False).load_state(str(path))


# -- separate clip of the policy and the value (docs 5.45) ---------------------------------------------
def test_clip_split_default_off_and_keeps_the_policy_gradient_out_of_the_value_clip():
    assert TrainConfig().clip_split is False
    applied = {}
    for split in (False, True):
        tr = make_trainer(clip_split=split)
        roll = tr.learner_rollout(torch.arange(32))
        logp = tr.policy.log_probs(roll.z, roll.m, roll.summaries, roll.empty, roll.log_prior)
        pol = logp.gather(1, roll.actions.unsqueeze(1)).mean()
        val = tr.value(roll.z, roll.m, roll.summaries, roll.empty).pow(2).mean() * 1e6     # a value loss that dominates the joint norm
        grads = torch.autograd.grad(pol, list(tr.policy.parameters()), retain_graph=True, allow_unused=True)
        alone = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
        tr._policy_clip_step(pol + val)
        applied[split] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in tr.policy.parameters() if p.grad is not None)))
        applied[("alone", split)] = alone
    alone = applied[("alone", True)]
    assert applied[False] < 1e-3 * alone                        # joint clip: the policy gradient is scaled down by the value norm
    assert applied[True] == pytest.approx(min(alone, tr.cfg.grad_clip), rel=1e-3)   # split: the policy gradient is untouched


# -- docs 5.48: decoy slots in the expert states and the "top2" pooling of the reward --------------------------
def test_top2_pool_matches_mean_with_one_other_slot_and_differs_with_several():
    from cibirl.reward import SlotScorer
    torch.manual_seed(0)
    mean, top2 = SlotScorer(8, 4, 16, 8, True, "mean"), SlotScorer(8, 4, 16, 8, True, "top2")
    top2.load_state_dict(mean.state_dict())
    z, m = torch.randn(5, 8), torch.randn(5, 4)
    summ = torch.randn(5, 6, 4)
    two = torch.tensor([False, False, True, True, True, True]).expand(5, 6)         # 2 occupied: each slot has ONE other
    four = torch.tensor([False, False, False, False, True, True]).expand(5, 6)
    assert torch.allclose(mean(z, m, summ, two), top2(z, m, summ, two), atol=1e-6)
    assert not torch.allclose(mean(z, m, summ, four), top2(z, m, summ, four), atol=1e-6)
    assert torch.isfinite(top2(z, m, summ, torch.ones(5, 6, dtype=torch.bool))).all()      # empty memory
    with pytest.raises(ValueError):
        SlotScorer(8, 4, 16, 8, True, "median")


def test_decoy_slots_keep_the_open_slot_of_the_expert_and_the_reward_step_runs():
    tr2 = make_trainer(decoy_slots=True, rho=0.1, num_slots=7)
    assert TrainConfig().decoy_slots is False and TrainConfig().rel_pool == "mean"
    eb = tr2.experts.sample(16)
    E, T = eb.slot.shape
    K, d = tr2.cfg.num_slots, tr2.cfg.m_dim
    m = torch.randn(E, T, d)
    slot, decoy = tr2._decoy_shift(m, eb)
    full, mask = decoy
    assert int(mask.sum(1).max()) <= K - tr2.cfg.orbits_per_episode and int(slot.max()) < K
    from cibirl.memory import expert_memory_states
    summ, empty = expert_memory_states(m, slot, K)
    summ, empty = tr2._fill_decoys(summ, empty, decoy)
    first_empty = (empty.long().cumsum(-1) == 1) & empty
    opening = first_empty.float().argmax(-1)
    assert torch.equal(opening[eb.is_first], slot[eb.is_first])                       # "open" is still the expert's slot
    forced = ~eb.is_first
    assert bool((~empty).gather(-1, slot.unsqueeze(-1)).squeeze(-1)[forced].all())     # the twin's slot is occupied
    assert not bool((~empty)[:, :, :].gather(-1, slot.unsqueeze(-1)).squeeze(-1)[eb.is_first].any())   # a new orbit is not in memory
    out = tr2.reward_step(eb, None)
    assert all(math.isfinite(v) for v in out.values())
    tr3 = make_trainer(rel_pool="top2", decoy_slots=True, num_slots=7)
    assert all(math.isfinite(v) for v in tr3.reward_step(tr3.experts.sample(16), None).values())


def test_nearest_and_attn_pools_are_finite_equivariant_and_learn_the_proximity():
    from cibirl.reward import SlotScorer
    torch.manual_seed(1)
    z, m, summ = torch.randn(6, 8), torch.randn(6, 4), torch.randn(6, 7, 4)
    for pool in ("nearest", "attn"):
        sc = SlotScorer(8, 4, 16, 8, True, pool)
        for occupied in (0, 1, 2, 5, 7):                           # includes the empty memory and a single slot
            empty = torch.arange(7).expand(6, 7) >= occupied
            out = sc(z, m, summ, empty)
            assert out.shape == (6, 7) and torch.isfinite(out).all()
        empty = torch.arange(7).expand(6, 7) >= 5
        perm = torch.tensor([3, 0, 4, 1, 2, 5, 6])                  # permute the occupied slots
        out = sc(z, m, summ, empty)
        out_p = sc(z, m, summ[:, perm], empty)
        assert torch.allclose(out[:, perm], out_p, atol=1e-5)       # permutation-equivariant
        sc.zero_grad(); sc(z, m, summ, empty).sum().backward()
        assert sc.prox.weight.grad is not None and float(sc.prox.weight.grad.abs().sum()) > 0   # the proximity gets a gradient


def test_local_pool_sees_only_the_nearest_slots_and_matches_mean_pool_on_small_memories():
    from cibirl.reward import SlotScorer
    torch.manual_seed(3)
    K, d = 8, 4
    mean, loc = SlotScorer(8, d, 16, 8, True, "mean"), SlotScorer(8, d, 16, 8, True, "local2")
    loc.load_state_dict(mean.state_dict())
    z, m, summ = torch.randn(6, 8), torch.randn(6, d), torch.randn(6, K, d)
    for pool in ("local2", "local3"):                                  # finite on the empty memory and a single slot, permutation-equivariant
        sc = SlotScorer(8, d, 16, 8, True, pool)
        for occupied in (0, 1, 2, 5, 8):
            empty = torch.arange(K).expand(6, K) >= occupied
            assert torch.isfinite(sc(z, m, summ, empty)).all()
        empty = torch.arange(K).expand(6, K) >= 5
        perm = torch.tensor([3, 0, 4, 1, 2, 5, 6, 7])
        assert torch.allclose(sc(z, m, summ, empty)[:, perm], sc(z, m, summ[:, perm], empty), atol=1e-5)
    # with 3 occupied slots every occupied slot has 2 other slots: local2 = mean + max over all the others = the "mean" pool
    empty3 = torch.arange(K).expand(6, K) >= 3
    assert torch.allclose(mean(z, m, summ, empty3)[:, :3], loc(z, m, summ, empty3)[:, :3], atol=1e-6)
    # the score of the OPEN slot does not depend on slots that are farther than the 2 nearest ones
    m0 = torch.zeros(6, d)
    near = 0.1 * torch.randn(6, K, d)
    few, many = torch.arange(K).expand(6, K) >= 2, torch.arange(K).expand(6, K) >= 6
    far = near.clone(); far[:, 2:] = 50.0 + torch.randn(6, K - 2, d)
    open_few, open_many = loc(z, m0, far, few)[:, 2], loc(z, m0, far, many)[:, 6]
    assert torch.allclose(open_few, open_many, atol=1e-5)
    assert not torch.allclose(mean(z, m0, far, few)[:, 2], mean(z, m0, far, many)[:, 6], atol=1e-3)   # the mean pool does see them
    with pytest.raises(ValueError):
        SlotScorer(8, d, 16, 8, True, "local4")


def test_decoy_exclude_makes_the_expert_loss_blind_to_the_scores_of_decoys():
    from cibirl.losses import expert_loss
    torch.manual_seed(2)
    E, T, K = 4, 6, 8
    scores = torch.randn(E, T, K)
    empty = torch.ones(E, T, K, dtype=torch.bool)
    slot = torch.randint(3, 6, (E, T)); is_first = torch.zeros(E, T, dtype=torch.bool); is_first[:, 0] = True
    empty[:, :, :6] = False                                     # slots 0..2 decoys, 3..5 the expert's
    neg = torch.zeros(E, T, T, dtype=torch.bool)
    decoy = torch.zeros(E, T, K, dtype=torch.bool); decoy[:, :, :3] = True
    a = expert_loss(scores, empty, slot, is_first, neg, 0.1, rho=0.1, decoy=decoy, decoy_exclude=True).loss
    changed = scores.clone(); changed[:, :, :3] += 7.0
    b = expert_loss(changed, empty, slot, is_first, neg, 0.1, rho=0.1, decoy=decoy, decoy_exclude=True).loss
    c = expert_loss(changed, empty, slot, is_first, neg, 0.1, rho=0.1, decoy=decoy, decoy_exclude=False).loss
    assert torch.allclose(a, b, atol=1e-6) and not torch.allclose(a, c, atol=1e-3)
    tr = make_trainer(decoy_slots=True, decoy_in_loss=False, rho=0.1, num_slots=7)
    assert all(math.isfinite(v) for v in tr.reward_step(tr.experts.sample(16), None).values())
    assert TrainConfig().decoy_in_loss is True
