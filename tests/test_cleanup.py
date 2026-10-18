"""The last step after training: close the memory slots that hold one cluster twice (cibirl/cleanup.py, docs 5.28, 5.29):
merge, relabel (A) and reintroduce (C).
Run: python -m pytest tests -q"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cibirl.cleanup import clone_memory, find_twins, merge_twin_slots, reintroduce, relabel, use_memory   # noqa: E402
from cibirl.memory import ClusterMemory, masked_log_softmax, valid_slot_mask                       # noqa: E402
from cibirl.train import TrainConfig                                                               # noqa: E402

CENTRES = torch.tensor([[4., 0, 0, 0], [0, 4., 0, 0], [0, 0, 4., 0], [0, 0, 0, 4.], [-3., -3., 0, 0]])


def state(twins=(), sizes=None):
    """Five clusters in slots 0-4, plus twins: (slot, copy_of, offset, size)."""
    S, C = torch.zeros(10, 4), torch.zeros(10)
    S[:5], C[:5] = CENTRES, 300.
    for slot, of, off, n in twins:
        S[slot] = CENTRES[of] + off
        C[slot], C[of] = n, C[of] - n
    return S, C


def test_twins_are_merged_into_the_bigger_slot_and_nothing_else():
    S, C = state(twins=[(5, 0, 1e-3, 40), (6, 2, 2e-3, 120)])
    rep, info = find_twins(S, C)
    assert rep[5] == 0 and rep[6] == 2 and all(rep[k] == k for k in (0, 1, 2, 3, 4))
    assert info["merged_pairs"] == 2 and info["jump"] > 1000 and info["threshold"] < 0.01


def test_the_smaller_twin_is_the_one_that_goes():
    S, C = state(twins=[(5, 0, 1e-3, 200)])          # slot 5 has 200, slot 0 has 100
    rep, _ = find_twins(S, C)
    assert rep[0] == 5 and rep[5] == 5


def test_a_chain_of_twins_is_one_cluster():
    S, C = state(twins=[(5, 0, 1e-3, 50), (6, 0, 2e-3, 50)])
    rep, _ = find_twins(S, C)
    assert len({rep[0], rep[5], rep[6]}) == 1


def test_no_twin_no_merge_even_if_clusters_are_close_to_one_another():
    S, C = state()
    rep, info = find_twins(S, C)
    assert all(rep[k] == k for k in rep) and info["threshold"] is None and info["jump"] < 3
    S[4] = torch.tensor([3.2, 0.9, 0, 0])             # one pair of clusters much closer than the others: still not twins
    rep, info = find_twins(S, C)
    assert all(rep[k] == k for k in rep)


def test_two_slots_or_one_slot_cannot_be_compared():
    S, C = torch.zeros(4, 3), torch.zeros(4)
    C[0] = C[1] = 10
    assert find_twins(S, C)[0] == {0: 0, 1: 1}
    C[1] = 0
    assert find_twins(S, C)[0] == {0: 0}
    assert find_twins(S, torch.zeros(4))[0] == {}


def test_identical_summaries_do_not_break_the_ratio():
    S, C = state(twins=[(5, 0, 0.0, 40)])
    rep, info = find_twins(S, C)
    assert rep[5] == 0 and info["jump"] > 1e6


def test_the_ratio_is_a_guard():
    S, C = state(twins=[(5, 0, 0.05, 40)])            # jump ~ 4 / 0.05 = 80
    assert find_twins(S, C, ratio=10.0)[1]["merged_pairs"] == 1
    assert find_twins(S, C, ratio=1000.0)[1]["merged_pairs"] == 0


def test_merge_moves_the_members_and_recomputes_the_summaries():
    N = 30
    m = torch.zeros(N, 4)
    m[:10], m[10:20], m[20:25], m[25:] = CENTRES[0], CENTRES[1], CENTRES[2], CENTRES[0] + 1e-3
    mem = ClusterMemory(N, 10, 4)
    mem.assign = torch.tensor([0] * 10 + [1] * 10 + [2] * 5 + [5] * 5)
    mem.recompute(m)
    info = merge_twin_slots(mem, m)
    assert info["merged_pairs"] == 1 and info["rep"][5] == 0
    assert set(mem.assign.tolist()) == {0, 1, 2} and mem.counts[0] == 15 and mem.counts[5] == 0
    assert torch.allclose(mem.summaries[0], m[[*range(10), *range(25, 30)]].mean(0), atol=1e-6)
    before = mem.assign.clone()
    assert merge_twin_slots(mem, m)["merged_pairs"] == 0 and torch.equal(mem.assign, before)      # idempotent


# ---------------------------------------------------------------------------------------------------------------------
# A: relabel, C: reintroduce
# ---------------------------------------------------------------------------------------------------------------------
def test_relabel_renames_the_slots_of_an_output_and_keeps_the_others():
    pred = torch.tensor([0, 5, 6, 3, 9, 5])                        # 9: an "outlier" id, not in the merge
    rep = {0: 0, 5: 0, 6: 2, 2: 2, 3: 3}
    assert relabel(pred, rep).tolist() == [0, 0, 2, 3, 9, 0]
    assert pred.tolist() == [0, 5, 6, 3, 9, 5]                     # the input is not modified
    assert relabel(pred, {}).tolist() == pred.tolist()


class _Id:
    """Encoder and intent of a toy trainer: the image IS its intent."""
    def __call__(self, x, sample=False):
        return SimpleNamespace(z=x)

    def mean(self, z):
        return z


class _NearestPolicy:
    """score(slot) = - squared distance of m to the summary; the empty slot (open) scores `open_score(m)`; plus the prior."""
    def __init__(self, open_score=lambda m: torch.full((m.shape[0],), -50.0)):
        self.open_score = open_score

    def log_probs(self, z, m, summaries, empty, log_prior=None):
        scores = -((m.unsqueeze(1) - summaries) ** 2).sum(-1)
        scores = torch.where(empty, self.open_score(m).unsqueeze(1).expand_as(scores), scores)
        valid = valid_slot_mask(empty)
        if log_prior is not None:
            scores = scores + log_prior.masked_fill(~valid, 0.0)
        return masked_log_softmax(scores, valid)


def toy_trainer(split=(0.6, 0.4), policy=None):
    """Three clusters (A, B, C) of 100 identical intents; A is held by two twin slots (0 and 5) in proportion `split`."""
    N, d = 300, 4
    m = torch.cat([CENTRES[i].expand(100, d) for i in range(3)]).clone()
    cfg = TrainConfig(num_slots=10, m_dim=d, slot_prior="crp", crp_alpha=1.0)
    mem = ClusterMemory(N, 10, d)
    nA = int(100 * split[0])
    mem.assign = torch.tensor([0] * nA + [5] * (100 - nA) + [1] * 100 + [2] * 100)
    mem.recompute(m)
    return SimpleNamespace(encoder=_Id(), intent=_Id(), policy=policy or _NearestPolicy(), memory=mem, cfg=cfg,
                           images=m, N=N, device=torch.device("cpu")), m


def test_reintroduce_drains_the_smaller_twin_and_touches_nothing_else():
    tr, m = toy_trainer()
    info = reintroduce(tr)
    mem = tr.memory
    assert mem.num_nonempty() == 3 and mem.counts[5] == 0 and mem.counts[0] == 100 and info["slots"] == 3
    assert info["moved"][-1] == 0 and info["moved"][0] == 40 and info["sweeps"] == len(info["moved"]) >= 2
    assert set(mem.assign[:100].tolist()) == {0} and set(mem.assign[100:200].tolist()) == {1} and set(mem.assign[200:].tolist()) == {2}
    assert torch.allclose(mem.summaries[0], CENTRES[0]) and torch.equal(mem.member_m, m)


def test_reintroduce_is_deterministic_and_the_input_copy_is_untouched():
    tr, _ = toy_trainer(split=(0.5, 0.5))
    before = clone_memory(tr.memory)
    a = clone_memory(tr.memory)
    with use_memory(tr, a):
        reintroduce(tr, seed=3)
    b = clone_memory(before)
    with use_memory(tr, b):
        reintroduce(tr, seed=3)
    assert torch.equal(a.assign, b.assign) and a.num_nonempty() == 3                      # an exact tie also ends in one slot
    assert torch.equal(tr.memory.assign, before.assign)                                   # use_memory restored the memory


def test_reintroduce_leaves_a_memory_without_twins_alone():
    tr, _ = toy_trainer(split=(1.0, 0.0))
    before = tr.memory.assign.clone()
    info = reintroduce(tr)
    assert info["moved"] == [0] and torch.equal(tr.memory.assign, before)


def test_reintroduce_no_open_never_uses_an_empty_slot():
    # a policy that wants to open a new cluster for the images of cluster C (it ranks "open" above the join)
    pol = _NearestPolicy(open_score=lambda m: torch.where((m - CENTRES[2]).norm(dim=-1) < 1, 100.0, -50.0))
    tr, _ = toy_trainer(policy=pol)
    with_open = clone_memory(tr.memory)
    with use_memory(tr, with_open):
        reintroduce(tr, allow_open=True)
    assert set(with_open.assign[200:].tolist()) - {2}                                     # C left its slot for a new one
    no_open = clone_memory(tr.memory)
    with use_memory(tr, no_open):
        reintroduce(tr, allow_open=False)
    assert set(no_open.assign[200:].tolist()) == {2} and no_open.num_nonempty() == 3
