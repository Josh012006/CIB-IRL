"""Unit checks of the pieces that are easy to get subtly wrong. Run: python -m pytest tests -q
(or simply: python tests/test_units.py)."""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cibirl import losses as L                                                  # noqa: E402
from cibirl.experts import assign_expert_slots, repulsion_mask, ExpertEpisodes   # noqa: E402
from cibirl.memory import ClusterMemory, expert_memory_states, valid_slot_mask, masked_log_softmax  # noqa: E402
from cibirl.reward import RewardModel                                           # noqa: E402
from cibirl.policy import PolicyNet                                             # noqa: E402

torch.manual_seed(0)


def test_valid_slot_mask_has_one_empty():
    empty = torch.tensor([[False, False, True, True, True], [True, True, True, True, True], [False] * 5])
    v = valid_slot_mask(empty)
    assert v.tolist() == [[True, True, True, False, False], [True, False, False, False, False], [True] * 5]


def test_memory_withdraw_is_exact_inverse():
    mem = ClusterMemory(10, 3, 4)
    ms = torch.randn(10, 4)
    for i in range(10):
        mem.place(i, ms[i], i % 3)
    s_before, c_before = mem.summaries.clone(), mem.counts.clone()
    mem.place(0, torch.randn(4), 1)          # move item 0 elsewhere...
    mem.place(0, ms[0], 0)                   # ...and put it back
    assert torch.allclose(mem.summaries, s_before, atol=1e-5) and torch.equal(mem.counts, c_before)
    for k in range(3):
        members = [i for i in range(10) if i % 3 == k]
        assert torch.allclose(mem.summaries[k], ms[members].mean(0), atol=1e-5)


def test_memory_recompute_matches_member_means():
    mem = ClusterMemory(8, 4, 3)
    for i in range(8):
        mem.place(i, torch.randn(3), i % 2)
    new_m = torch.randn(8, 3)
    mem.recompute(new_m)
    for k in (0, 1):
        assert torch.allclose(mem.summaries[k], new_m[[i for i in range(8) if i % 2 == k]].mean(0), atol=1e-5)
    assert mem.counts.tolist() == [4, 4, 0, 0] and (mem.summaries[2:] == 0).all()


def test_expert_slots_and_masks_match_brute_force():
    E, T = 6, 8
    group = torch.randint(0, 4, (E, T))
    slot, first = assign_expert_slots(group)
    neg = repulsion_mask(group)
    for e in range(E):
        seen = {}
        for t in range(T):
            g = int(group[e, t])
            if g not in seen:
                seen[g] = len(seen)
                assert first[e, t]
            else:
                assert not first[e, t]
            assert slot[e, t] == seen[g]
            for v in range(T):
                assert bool(neg[e, t, v]) == (v < t and group[e, v] != group[e, t])


def test_expert_memory_states_equal_sequential_transition():
    E, T, K, d = 3, 7, 5, 4
    group = torch.randint(0, 4, (E, T))
    slot, _ = assign_expert_slots(group)
    m = torch.randn(E, T, d)
    summ, empty = expert_memory_states(m, slot, K)
    for e in range(E):
        mem = ClusterMemory(T, K, d)
        for t in range(T):
            s, em = mem.state()
            assert torch.allclose(summ[e, t], s, atol=1e-5) and torch.equal(empty[e, t], em)
            mem.place(t, m[e, t], int(slot[e, t]))


def test_log_one_minus_prob_matches_naive():
    scores = torch.randn(4, 5, 6)
    empty = torch.rand(4, 5, 6) > 0.5
    valid = valid_slot_mask(empty)
    log1m = L._log_one_minus_prob(scores, valid)
    q = masked_log_softmax(scores, valid).exp()
    naive = torch.log((1 - q).clamp(min=1e-30))
    ok = valid & (q < 0.999)
    assert torch.allclose(log1m[ok], naive[ok], atol=1e-4)


def test_expert_loss_is_finite_and_has_gradient():
    E, T, K = 4, 6, 5
    group = torch.randint(0, 3, (E, T))
    slot, first = assign_expert_slots(group)
    neg = repulsion_mask(group)
    _, empty = expert_memory_states(torch.randn(E, T, 3), slot, K)
    scores = torch.randn(E, T, K, requires_grad=True)
    out = L.expert_loss(scores, empty, slot, first, neg, tau_plus=0.2)
    out.loss.backward()
    assert torch.isfinite(out.loss) and torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_expert_loss_prefers_the_right_scores():
    """Scores that put the twin's slot on top and the others low must have a smaller loss than the opposite."""
    E, T, K = 8, 6, 5
    group = torch.randint(0, 3, (E, T))
    slot, first = assign_expert_slots(group)
    neg = repulsion_mask(group)
    _, empty = expert_memory_states(torch.randn(E, T, 3), slot, K)
    good = torch.full((E, T, K), -2.0).scatter(-1, slot.unsqueeze(-1), 3.0)
    bad = -good
    assert L.expert_loss(good, empty, slot, first, neg, 0.2).loss < L.expert_loss(bad, empty, slot, first, neg, 0.2).loss


def test_slot_scorer_is_permutation_equivariant():
    K, d, B = 6, 4, 3
    net = RewardModel(z_dim=5, m_dim=d)
    z, m = torch.randn(B, 5), torch.randn(B, d)
    summ = torch.randn(B, K, d)
    empty = torch.tensor([[False, False, True, False, True, True]] * B)
    perm = torch.randperm(K)
    s1 = net(z, m, summ, empty)
    s2 = net(z, m, summ[:, perm], empty[:, perm])
    assert torch.allclose(s1[:, perm], s2, atol=1e-5)


def test_policy_probabilities_sum_to_one_on_valid_slots():
    pol = PolicyNet(5, 4)
    z, m = torch.randn(3, 5), torch.randn(3, 4)
    summ = torch.randn(3, 6, 4)
    empty = torch.tensor([[False, False, True, False, True, True]] * 3)
    logp = pol.log_probs(z, m, summ, empty)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3), atol=1e-5)
    assert (logp.exp()[:, 4:] == 0).all()              # the 2nd and 3rd empty slots are not actions


def test_gae_gamma_zero_is_reward_minus_value():
    r, v = torch.randn(7), torch.randn(7)
    adv, ret = L.compute_gae(r, v, gamma=0.0, lam=0.95)
    assert torch.allclose(adv, r - v, atol=1e-6) and torch.allclose(ret, r, atol=1e-6)


def test_episode_builder_constraints():
    ids = torch.arange(100)
    ep = ExpertEpisodes.build(ids, 20, num_orbits=4, views_per_orbit=2, num_slots=5, seed=1)
    b = ep.sample(10)
    assert b.image_ids.shape == (10, 8)
    for e in range(10):
        assert len(set(b.image_ids[e].tolist())) == 4                       # distinct images
        assert (b.is_first[e].sum() == 4) and (b.slot[e].max() == 3)
    try:
        ExpertEpisodes.build(ids, 5, num_orbits=6, views_per_orbit=2, num_slots=5)
    except ValueError:
        pass
    else:
        raise AssertionError("B > K must be rejected")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
