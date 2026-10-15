"""The kNN measure of scripts/probe_representation.py. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_representation import knn_accuracy      # noqa: E402


def test_knn_is_perfect_on_separated_classes_and_chance_on_noise():
    g = torch.Generator().manual_seed(0)
    y = np.repeat(np.arange(4), 50)
    centres = torch.tensor([[0.0, 0], [20, 0], [0, 20], [20, 20]])
    f = centres[torch.as_tensor(y)] + torch.randn(200, 2, generator=g)
    assert knn_accuracy(f, y, 4) == 1.0
    noise = torch.randn(2000, 8, generator=g)
    y2 = np.random.default_rng(0).integers(0, 4, 2000)
    assert abs(knn_accuracy(noise, y2, 4) - 0.25) < 0.05


def test_knn_never_uses_the_point_itself():
    f = torch.tensor([[0.0], [0.0], [50.0], [50.0]])
    assert knn_accuracy(f, np.array([0, 0, 1, 1]), 2, k=1) == 1.0       # the neighbour is the twin at distance 0, not oneself
    assert knn_accuracy(f, np.array([0, 1, 0, 1]), 2, k=1) == 0.0       # if it voted for itself this would be 1.0
