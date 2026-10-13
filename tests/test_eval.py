"""Metrics and figures. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import visualize as viz                                  # noqa: E402
from eval.metrics import cluster_acc, cluster_purity, nmi          # noqa: E402


def test_nmi_extremes():
    y = np.repeat(np.arange(4), 50)
    assert abs(nmi(y, y) - 1) < 1e-9
    assert abs(nmi(y, (y + 2) % 4) - 1) < 1e-9                      # relabeling does not matter
    rng = np.random.default_rng(0)
    assert nmi(y, rng.integers(0, 4, size=len(y))) < 0.1            # independent labels


def test_cluster_acc_is_invariant_to_slot_permutation():
    y = np.repeat(np.arange(3), 40)
    pred = np.array([2, 0, 1])[y]                                   # perfect clustering, shuffled slot ids
    assert cluster_acc(pred, y, num_slots=5) == 1.0
    pred2 = pred.copy()
    pred2[:10] = 4                                                  # 10 images in another slot
    assert abs(cluster_acc(pred2, y, 5) - (120 - 10) / 120) < 1e-9


def test_purity():
    y = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1])
    out = cluster_purity(pred, y, 3)
    assert out[0] == (2, 0, 1.0) and out[1][0] == 4 and abs(out[1][2] - 0.75) < 1e-9 and 2 not in out


def test_figures_are_written(tmp_path):
    n = 60
    images = torch.rand(n, 3, 8, 8)
    labels = np.repeat(np.arange(3), 20)
    pred = np.array([1, 0, 2])[labels]
    m = np.random.default_rng(0).normal(size=(n, 6))
    history = [{"epoch": e, "L_pos": 1.0 / (e + 1), "L_neg": 0.5, "twin_acc": 0.5, "L_psi": -1.0, "KL": 100.0,
                "nonempty_slots": 3, "K_eff": 2.5, "churn": 0.1, "pi_entropy": 0.5, "m_std": 1.0} for e in range(5)]
    viz.plot_curves(history, tmp_path / "curves.png")
    viz.plot_cluster_grid(images, pred, tmp_path / "grid.png", num_slots=4, labels=labels)
    viz.plot_cluster_grid(images, pred, tmp_path / "grid_nolabels.png", num_slots=4)
    viz.plot_confusion(pred, labels, tmp_path / "conf.png", num_slots=4)
    viz.plot_embedding(m, pred, tmp_path / "emb.png", labels=labels, background=labels)
    for f in ("curves", "grid", "grid_nolabels", "conf", "emb"):
        assert (tmp_path / f"{f}.png").stat().st_size > 1000
