"""The last step after training: merge the memory slots that hold one cluster twice (cibirl/cleanup.py, docs 5.28).
Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cibirl.cleanup import find_twins, merge_twin_slots   # noqa: E402
from cibirl.memory import ClusterMemory                   # noqa: E402

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
