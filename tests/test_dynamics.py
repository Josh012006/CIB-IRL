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
