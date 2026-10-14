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
