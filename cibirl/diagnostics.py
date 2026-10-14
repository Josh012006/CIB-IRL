"""EVALUATION-ONLY bookkeeping of the learner's placements (never used to train, to stop or to choose anything).

The true classes are used to ask, for every placement made during an epoch, whether the policy took the right
KIND of decision given the memory it saw:

* join  a slot that is already occupied : right iff the dominant class of that slot is the class of the image;
* open  the empty slot                  : right iff no occupied slot has the image's class as its dominant class.

A wrong join means "put it with something it does not belong to" (contamination); a wrong open means "split a
class that already had a slot". Both are tallied by the number of occupied slots the policy saw (a memory that
is still being built has few), together with the purity of the slots and the number of times a slot was emptied.

Everything is reported under `eval_*` keys, the same convention as the other label-based monitors. Plain numpy
(no torch ops) because it is called once per placement.
"""
from __future__ import annotations

import numpy as np
from torch import Tensor

CORRECT, WRONG_JOIN, WRONG_OPEN = 0, 1, 2


class PlacementDiagnostics:
    def __init__(self, labels: Tensor, num_slots: int):
        self.y = labels.cpu().long().numpy()
        self.K, self.C = num_slots, int(self.y.max()) + 1
        self.cc = np.zeros((num_slots, self.C), dtype=np.int64)          # class counts of every slot (mirror)
        self.size = np.zeros(num_slots, dtype=np.int64)
        self.table = np.zeros((num_slots + 1, 3), dtype=np.int64)
        self.depleted = 0

    # -- bookkeeping ---------------------------------------------------------------------------------
    def sync(self, assign: Tensor) -> None:
        """Rebuild the class counts from the memory's assignment (after any change made outside of record())."""
        a = assign.cpu().numpy()
        placed = a >= 0
        self.cc[:] = 0
        np.add.at(self.cc, (a[placed], self.y[placed]), 1)
        self.size = self.cc.sum(1)

    def start_epoch(self, assign: Tensor) -> None:
        self.table[:] = 0
        self.depleted = 0
        self.sync(assign)

    def record(self, item: int, slot: int, prev_slot: int) -> None:
        """Call BEFORE the memory is updated: the state is the one the policy saw (the image still sits in prev_slot)."""
        y = int(self.y[item])
        occupied = self.size > 0
        n_occ = int(occupied.sum())
        if occupied[slot]:
            outcome = CORRECT if int(self.cc[slot].argmax()) == y else WRONG_JOIN
        else:
            exists = bool(((self.cc.argmax(1) == y) & occupied).any())
            outcome = WRONG_OPEN if exists else CORRECT
        self.table[n_occ, outcome] += 1
        if prev_slot != slot:
            if prev_slot >= 0:
                self.cc[prev_slot, y] -= 1
                self.size[prev_slot] -= 1
                if self.size[prev_slot] == 0:
                    self.depleted += 1
            self.cc[slot, y] += 1
            self.size[slot] += 1

    # -- report --------------------------------------------------------------------------------------
    def purity(self) -> float:
        """Share of the placed images that belong to the dominant class of their slot."""
        total = int(self.cc.sum())
        return float(self.cc.max(1).sum()) / total if total else 0.0

    def report(self) -> dict:
        place = {str(n): [int(v) for v in self.table[n]] for n in range(self.K + 1) if int(self.table[n].sum()) > 0}
        total = int(self.table.sum())
        wrong = int(self.table[:, WRONG_JOIN].sum() + self.table[:, WRONG_OPEN].sum())
        return {"eval_place": place,                       # {n_occupied: [correct, wrong_join, wrong_open]}
                "eval_place_error": wrong / total if total else 0.0,
                "eval_depleted": self.depleted,            # slots emptied during the epoch
                "eval_purity": self.purity()}              # purity of the memory at the end of the epoch
