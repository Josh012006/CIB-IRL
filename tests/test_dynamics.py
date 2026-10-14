"""Knobs on the assignment dynamics: policy_temperature and consolidate. Run: python -m pytest tests/test_dynamics.py"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cibirl.policy import PolicyNet                      # noqa: E402
from cibirl.train import TrainConfig, Trainer           # noqa: E402
from toy_data import background_redraw, make_toy         # noqa: E402


def make_trainer(**kw):
    x, _, _ = make_toy(n_per_class=60, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=200,
                      expert_fraction=0.5, learner_batch=32, episodes_per_batch=8, beta=1e-2,
                      beta_warmup_iters=0, reward_warmup_iters=2, **kw)
    return Trainer(cfg, x, background_redraw)


def _state(b=6, k=5, z_dim=5, m_dim=4):
    torch.manual_seed(0)
    z, m = torch.randn(b, z_dim), torch.randn(b, m_dim)
    summ = torch.randn(b, k, m_dim)
    empty = torch.tensor([[False, False, True, False, True]] * b)
    return z, m, summ, empty


def test_defaults_do_not_change_the_old_behaviour():
    cfg = TrainConfig()
    assert cfg.policy_temperature == 1.0 and cfg.consolidate is False


def test_config_reads_the_new_keys_from_yaml_values():
    cfg = TrainConfig.from_dict({"consolidate": True, "policy_temperature": 0.5, "alpha": 0})
    assert cfg.consolidate is True and cfg.policy_temperature == 0.5 and cfg.alpha == 0.0
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"consolidate": "yes"})                  # strict boolean


def test_temperature_one_is_the_identical_policy():
    torch.manual_seed(1)
    a = PolicyNet(5, 4)
    b = PolicyNet(5, 4, temperature=1.0)
    b.load_state_dict(a.state_dict())
    st = _state()
    assert torch.equal(a.log_probs(*st), b.log_probs(*st))


def test_low_temperature_sharpens_but_keeps_the_argmax_and_a_valid_distribution():
    torch.manual_seed(2)
    ref = PolicyNet(5, 4)
    cold = PolicyNet(5, 4, temperature=0.5)
    cold.load_state_dict(ref.state_dict())
    st = _state()
    lp_ref, lp_cold = ref.log_probs(*st), cold.log_probs(*st)
    assert torch.equal(lp_ref.argmax(-1), lp_cold.argmax(-1))
    assert torch.allclose(lp_cold.exp().sum(-1), torch.ones(st[0].shape[0]), atol=1e-5)
    assert (lp_cold.exp()[:, 4:] == 0).all()                           # 2nd empty slot is still not an action
    ent_ref, ent_cold = PolicyNet.entropy(lp_ref, st[3]), PolicyNet.entropy(lp_cold, st[3])
    assert (ent_cold < ent_ref).all()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        PolicyNet(5, 4, temperature=0.0)


def test_act_and_ppo_use_the_same_tempered_distribution():
    """logp returned by act() is the log-prob of the sampled action under log_probs(): the PPO ratio starts at 1."""
    torch.manual_seed(3)
    pol = PolicyNet(5, 4, temperature=0.5)
    st = _state(b=64)
    a, lp = pol.act(*st)
    assert torch.allclose(lp, pol.log_probs(*st).gather(-1, a.unsqueeze(-1)).squeeze(-1), atol=1e-6)


def test_recompute_keeps_sampled_placements_by_default():
    tr = make_trainer()
    K, N = tr.cfg.num_slots, tr.N
    tr.memory.assign = torch.randint(0, K, (N,))
    before = tr.memory.assign.clone()
    tr.recompute_memory()
    assert torch.equal(tr.memory.assign, before)


def test_consolidate_replaces_placements_by_the_greedy_partition():
    tr = make_trainer(consolidate=True)
    K, N = tr.cfg.num_slots, tr.N
    tr.memory.assign = torch.randint(0, K, (N,))
    tr.memory.recompute(tr.intents())                                  # the first half of recompute_memory()
    expected = tr.predict()                                            # greedy partition against that memory
    tr.recompute_memory()
    assert torch.equal(tr.memory.assign, expected)


def test_consolidated_memory_is_consistent_with_its_members():
    tr = make_trainer(consolidate=True)
    K, N = tr.cfg.num_slots, tr.N
    tr.memory.assign = torch.randint(0, K, (N,))
    tr.recompute_memory()
    mem, m = tr.memory, tr.intents()
    assert (mem.assign >= 0).all() and (mem.assign < K).all()
    assert torch.equal(mem.counts, torch.bincount(mem.assign, minlength=K).float())
    for k in range(K):
        sel = mem.assign == k
        if sel.any():
            assert torch.allclose(mem.summaries[k], m[sel].mean(0), atol=1e-5)


def test_a_consolidating_epoch_runs_end_to_end():
    tr = make_trainer(consolidate=True, policy_temperature=0.5, alpha=0.0)
    hist = tr.fit(2, log_fn=lambda row: False)
    assert len(hist) == 2 and all(0.0 <= r["churn"] <= 1.0 for r in hist)


# ---- oracle diagnostic: orbits of one class linked in the expert episodes ---------------------------------
from cibirl.experts import ExpertEpisodes                # noqa: E402


def _episodes():
    ids = torch.arange(60)
    labels = torch.arange(60) % 4                                      # class of image i
    ep = ExpertEpisodes.build(ids, 40, num_orbits=3, views_per_orbit=2, num_slots=4, seed=0)
    return ep, labels


def test_link_prob_zero_is_phase1():
    ep, labels = _episodes()
    torch.manual_seed(0)
    a = ep.sample(16)
    torch.manual_seed(0)
    b = ep.sample(16, link_labels=labels, link_prob=0.0)
    assert torch.equal(a.slot, b.slot) and torch.equal(a.group, b.group) and torch.equal(a.neg_mask, b.neg_mask)


def test_full_link_merges_orbits_of_the_same_class_only():
    ep, labels = _episodes()
    idx = torch.arange(40)
    group = ep.linked_groups(idx, labels, 1.0)
    cls = labels[ep.image_ids[idx]]
    same_class = cls.unsqueeze(2) == cls.unsqueeze(1)
    assert torch.equal(group.unsqueeze(2) == group.unsqueeze(1), same_class)     # group == class
    b = ep.batch(idx, group)
    # no repulsion between views of one class, slots shared inside a class
    assert not (b.neg_mask & same_class).any()
    assert torch.equal(b.slot.unsqueeze(2) == b.slot.unsqueeze(1), same_class)
    assert (cls[..., None] != cls[:, None]).any()                                # some episodes do mix classes
    assert (b.orbit.unsqueeze(2) != b.orbit.unsqueeze(1))[same_class].any()      # linked views come from other orbits


def test_partial_link_gives_both_kinds_of_episodes():
    ep, labels = _episodes()
    torch.manual_seed(1)
    group = ep.linked_groups(torch.arange(40), labels, 0.5)
    orbit = ep.orbit[torch.arange(40)]
    merged = (group != 4 + orbit).float().mean()                                  # views whose group is a class group
    assert 0.2 < float(merged) < 0.8


def test_oracle_link_needs_labels_and_valid_probability():
    x, y, _ = make_toy(n_per_class=20, n_classes=4)
    base = dict(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=50, expert_fraction=0.5)
    with pytest.raises(ValueError):
        Trainer(TrainConfig(expert_oracle_link=1.0, **base), x, background_redraw)         # no labels given
    with pytest.raises(ValueError):
        Trainer(TrainConfig(expert_oracle_link=1.5, **base), x, background_redraw, oracle_labels=y)
    tr = Trainer(TrainConfig(expert_oracle_link=1.0, **base), x, background_redraw, oracle_labels=y)
    assert tr._oracle_labels is not None
    assert Trainer(TrainConfig(**base), x, background_redraw, oracle_labels=y)._oracle_labels is None   # ignored when off


def test_an_oracle_epoch_runs():
    x, y, _ = make_toy(n_per_class=30, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=100, expert_fraction=0.5,
                      learner_batch=32, episodes_per_batch=8, reward_warmup_iters=2, expert_oracle_link=0.5)
    hist = Trainer(cfg, x, background_redraw, oracle_labels=y).fit(2, log_fn=lambda r: False)
    assert len(hist) == 2


# ---- placement diagnostics (evaluation only), oracle warm memory, burn-in --------------------------------
from cibirl.diagnostics import PlacementDiagnostics     # noqa: E402
from cibirl.memory import ClusterMemory                 # noqa: E402


def _tiny(**kw):
    x, y, _ = make_toy(n_per_class=30, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=100, expert_fraction=0.5,
                      learner_batch=32, episodes_per_batch=8, reward_warmup_iters=2, beta_warmup_iters=0,
                      **{k: v for k, v in kw.items() if k in TrainConfig.__dataclass_fields__})
    extra = {k: v for k, v in kw.items() if k not in TrainConfig.__dataclass_fields__}
    return Trainer(cfg, x, background_redraw, **extra), y


def test_diagnostics_classify_joins_and_opens():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    d = PlacementDiagnostics(y, num_slots=3)
    d.start_epoch(torch.full((6,), -1))
    d.record(0, 0, -1)          # open an empty slot, nothing occupied: right
    d.record(1, 0, -1)          # join slot 0 (class 0) with a class-0 image: right
    d.record(2, 0, -1)          # join slot 0 with a class-1 image: wrong join
    d.record(3, 1, -1)          # open although... class 1 is not dominant anywhere (slot 0 = 0,0,1): right
    d.record(4, 2, -1)          # open the empty slot for class 2: right
    d.record(5, 1, -1)          # join slot 1 (class 1) with a class-2 image: wrong join
    rep = d.report()["eval_place"]
    assert rep["0"] == [1, 0, 0] and rep["1"] == [2, 1, 0] and rep["2"] == [1, 0, 0] and rep["3"] == [0, 1, 0]
    d.record(0, 1, 0)           # image 0 (class 0) moves from slot 0 to slot 1: bookkeeping follows
    assert int(d.cc[0, 0]) == 1 and int(d.cc[1, 0]) == 1
    d.record(1, 2, 0)           # slot 0 loses its last class-0 member but still holds image 2: not depleted
    assert d.depleted == 0
    d.record(2, 2, 0)           # now slot 0 is empty
    assert d.depleted == 1


def test_a_wrong_open_is_detected():
    d = PlacementDiagnostics(torch.tensor([0, 0, 1]), num_slots=3)
    d.start_epoch(torch.full((3,), -1))
    d.record(0, 0, -1)
    d.record(1, 1, -1)          # opens a new slot for a class-0 image although slot 0 is class 0
    assert d.report()["eval_place"]["1"] == [0, 0, 1]


def test_diagnostics_do_not_change_training():
    x, y, _ = make_toy(n_per_class=30, n_classes=4)
    cfg = TrainConfig(num_slots=4, image_size=16, orbits_per_episode=3, num_expert_episodes=100, expert_fraction=0.5,
                      learner_batch=32, episodes_per_batch=8, reward_warmup_iters=2, beta_warmup_iters=0)
    plain = Trainer(cfg, x, background_redraw).fit(2, log_fn=lambda r: False)          # seeded in __init__
    with_diag = Trainer(cfg, x, background_redraw, eval_labels=y).fit(2, log_fn=lambda r: False)
    for ra, rc in zip(plain, with_diag):
        keys = [k for k in ra if not k.startswith("eval_")]
        assert keys and all(ra[k] == rc[k] for k in keys)
    assert "eval_place" in with_diag[0] and "eval_place" not in plain[0]
    placed = sum(sum(v) for v in with_diag[1]["eval_place"].values())     # epoch 0 has the reward warm-up
    assert placed >= x.shape[0] - 32                         # every image of the epoch but a possible short tail


def test_warm_memory_seeds_one_slot_per_class_and_needs_labels():
    with pytest.raises(ValueError):
        _tiny(oracle_warm_memory=2)
    x, y, _ = make_toy(n_per_class=30, n_classes=4)
    tr, _ = _tiny(oracle_warm_memory=3, oracle_labels=y)
    tr._warm_memory()
    mem = tr.memory
    assert torch.equal(mem.counts, torch.full((4,), 3.0))
    for c in range(4):
        assert (y[(mem.assign == c).nonzero().flatten()] == c).all()      # slot c holds only class c


def test_burn_in_keeps_a_sample_of_every_cluster():
    torch.manual_seed(0)
    mem = ClusterMemory(40, 3, 2)
    assign = torch.arange(40) % 3
    m = torch.randn(40, 2)
    mem.assign = assign.clone()
    mem.recompute(m)
    sizes = mem.counts.clone()
    mem.subsample(m, 0.25)
    assert (mem.counts >= 1).all() and (mem.counts < sizes).all()
    kept = mem.assign >= 0
    assert torch.equal(mem.assign[kept], assign[kept])                    # nobody changes slot
    assert torch.equal(mem.counts, torch.bincount(mem.assign[kept], minlength=3).float())
    for k in range(3):
        assert torch.allclose(mem.summaries[k], m[mem.assign == k].mean(0), atol=1e-6)


def test_burn_in_runs_in_the_loop_and_rejects_a_bad_fraction():
    tr, _ = _tiny(burn_in_epochs=1, burn_in_fraction=0.2)
    hist = tr.fit(2, log_fn=lambda r: False)
    assert len(hist) == 2
    with pytest.raises(ValueError):
        _tiny(burn_in_fraction=0.0)


# ---- stale summaries: recompute_every and the frozen / slowed representation ----------------------------------
def test_new_options_are_off_by_default_and_validated():
    cfg = TrainConfig()
    assert cfg.recompute_every == 0 and cfg.freeze_repr_after == 0 and cfg.repr_lr_scale == 0.0
    for bad in ({"recompute_every": -1}, {"freeze_repr_after": -1}, {"repr_lr_scale": 1.5}):
        with pytest.raises(ValueError):
            _tiny(**bad)


def test_recompute_every_keeps_the_summaries_current_with_the_encoder():
    def after_four_iterations(every):
        tr, _ = _tiny(recompute_every=every)       # warm-up: 2 iterations, then the learner places images
        for _ in range(4):
            tr.train_iteration(torch.arange(32))
        return tr
    tr = after_four_iterations(1)
    placed = tr.memory.assign >= 0
    assert placed.any()
    # the recompute ran after the last gradient step: stored intents = intents of the networks as they are now
    assert torch.allclose(tr.memory.member_m[placed], tr.intents()[placed], atol=1e-6)
    old = after_four_iterations(0)                  # default: the stored intents are those of placement time
    assert not torch.allclose(old.memory.member_m[old.memory.assign >= 0], old.intents()[old.memory.assign >= 0], atol=1e-6)


def test_frozen_representation_stops_moving_m():
    tr, _ = _tiny(freeze_repr_after=1, repr_lr_scale=0.0)
    tr.fit(2, log_fn=lambda r: False)         # epoch 0 = warm-up + first placements; frozen at its end
    assert tr._acting_epochs >= 1
    snap = [p.detach().clone() for p in list(tr.encoder.parameters()) + list(tr.intent.parameters())]
    bn = (tr.intent.norm.running_mean.clone(), tr.intent.norm.running_var.clone())
    m0 = tr.intents()
    tr.fit(2, log_fn=lambda r: False)
    for a, b in zip(snap, list(tr.encoder.parameters()) + list(tr.intent.parameters())):
        assert torch.equal(a, b)
    assert torch.equal(bn[0], tr.intent.norm.running_mean) and torch.equal(bn[1], tr.intent.norm.running_var)
    assert torch.equal(m0, tr.intents())


def test_slowed_representation_still_moves_but_less():
    tr, _ = _tiny(freeze_repr_after=1, repr_lr_scale=0.1)
    tr.fit(1, log_fn=lambda r: False)
    assert tr.opt_reward.param_groups[0]["lr"] == pytest.approx(0.1 * tr.cfg.lr_reward)
    assert tr.opt_reward.param_groups[1]["lr"] == pytest.approx(tr.cfg.lr_reward)


# ---- teacher phase and discarded memory (formation) ------------------------------------------------------------
def test_formation_options_are_off_by_default_and_validated():
    cfg = TrainConfig()
    assert cfg.policy_teacher_iters == 0 and cfg.memory_discard_epochs == 0
    for bad in ({"policy_teacher_iters": -1}, {"memory_discard_epochs": -1}):
        with pytest.raises(ValueError):
            _tiny(**bad)


def test_teacher_phase_acts_like_the_reward_and_distills_the_policy():
    tr, _ = _tiny(policy_teacher_iters=40)
    with torch.no_grad():
        for p in tr.reward.parameters():
            p.mul_(6.0)                                         # a sharp, non-uniform teacher
    roll = tr.learner_rollout(torch.arange(32), teacher=True)
    first = tr.distill_step(roll)["distill_kl"]
    for _ in range(25):
        last = tr.distill_step(roll)["distill_kl"]              # same states: the KL must go down
    assert first > 1e-3 and last < 0.5 * first and last >= -1e-6
    # the sampled actions are valid slots of the memory the policy saw
    valid = ~roll.empty | ((roll.empty.long().cumsum(-1) == 1) & roll.empty)
    assert valid.gather(1, roll.actions.unsqueeze(1)).all()


def test_teacher_phase_ends_and_ppo_resumes():
    tr, _ = _tiny(policy_teacher_iters=3)
    out = [tr.train_iteration(torch.randperm(tr.N)[:32]) for _ in range(8)]      # iterations 0-1 warm-up
    assert ["distill_kl" in o for o in out[2:]] == [True, True, True, False, False, False]
    assert all("pi_entropy" in o for o in out[5:])


def test_memory_clear_forgets_everything():
    mem = ClusterMemory(10, 3, 2)
    mem.place(0, torch.ones(2), 1)
    mem.place(1, torch.ones(2), 2)
    mem.clear()
    assert (mem.assign == -1).all() and (mem.counts == 0).all() and (mem.summaries == 0).all()


def test_discarded_epochs_empty_the_memory_and_then_it_persists():
    tr, _ = _tiny(memory_discard_epochs=1)
    tr.fit(1, log_fn=lambda r: False)             # epoch 0 places images (after 2 warm-up iterations), then is discarded
    assert (tr.memory.assign == -1).all() and tr.memory.num_nonempty() == 0
    tr.fit(1, log_fn=lambda r: False)             # second epoch of placements: kept
    assert (tr.memory.assign >= 0).sum() > 0


# ---- free_debias_scale: the correction of L_neg on free moves ---------------------------------------------------
def _expert_inputs(seed=0, E=6, T=6, K=5):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(E, T, K, generator=g) * 2
    empty = torch.ones(E, T, K, dtype=torch.bool)
    orbit = torch.arange(3).repeat_interleave(2).repeat(E, 1)                 # views 0,1 / 2,3 / 4,5
    is_first = torch.tensor([True, False] * 3).repeat(E, 1)
    slot = orbit.clone()
    for t in range(T):
        empty[:, t, :] = True
        placed = slot[:, :t]
        for e in range(E):
            empty[e, t, placed[e]] = False
    neg = (orbit.unsqueeze(2) != orbit.unsqueeze(1)) & (torch.arange(T).view(1, 1, T) < torch.arange(T).view(1, T, 1))
    return scores, empty, slot, is_first, neg


def test_free_debias_one_is_the_old_loss_and_zero_removes_the_dead_zone():
    from cibirl import losses as L
    s, empty, slot, first, neg = _expert_inputs()
    old = L.expert_loss(s, empty, slot, first, neg, 0.2)
    same = L.expert_loss(s, empty, slot, first, neg, 0.2, free_debias=1.0)
    assert torch.equal(old.loss, same.loss)
    # confident twin (large l+) puts the free-move repulsion in the dead zone; without the correction it comes back
    sharp = s.clone()
    for e in range(sharp.shape[0]):
        for t in (1, 3, 5):                                                   # forced moves: the twin's slot dominates
            sharp[e, t, slot[e, t]] += 12.0
    s_req = sharp.clone().requires_grad_(True)
    L.expert_loss(s_req, empty, slot, first, neg, 0.2, free_debias=1.0).loss.backward()
    g_old = s_req.grad[:, [0, 2, 4]].abs().sum()
    s_req2 = sharp.clone().requires_grad_(True)
    L.expert_loss(s_req2, empty, slot, first, neg, 0.2, free_debias=0.0).loss.backward()
    g_new = s_req2.grad[:, [0, 2, 4]].abs().sum()                             # gradient on free-move rows
    assert g_new > g_old


def test_free_debias_scale_is_validated_and_runs():
    with pytest.raises(ValueError):
        _tiny(free_debias_scale=1.5)
    tr, _ = _tiny(free_debias_scale=0.0)
    assert len(tr.fit(2, log_fn=lambda r: False)) == 2
