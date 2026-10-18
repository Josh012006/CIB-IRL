"""The toy data and its test partition (scripts/toy_data.py, README Section 8.4).
Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from toy_data import make_toy, make_toy_test       # noqa: E402


def test_make_toy_is_deterministic_and_has_the_expected_shapes():
    x, y, bg = make_toy(seed=3)
    x2, y2, bg2 = make_toy(seed=3)
    assert x.shape == (1500, 3, 16, 16) and y.shape == (1500,) and bg.shape == (1500,)
    assert torch.equal(x, x2) and torch.equal(y, y2) and torch.equal(bg, bg2)
    assert torch.bincount(y).tolist() == [300] * 5


@pytest.mark.parametrize("mode", ["iid", "aligned", "inverted"])
def test_test_partition_has_the_shape_of_the_training_set_but_new_images(mode):
    x, y, _ = make_toy(seed=4)
    xt, yt, bgt = make_toy_test(4, 4, mode)
    assert xt.shape == x.shape and torch.equal(yt, y) and bgt.shape == y.shape
    assert not torch.equal(xt, x)                           # new images
    assert float(xt.min()) >= 0.0 and float(xt.max()) <= 1.0


def test_test_partition_keeps_the_classes_of_the_seed():
    """Same shapes: the white blocks of a class are the same pixels in the training and the test images."""
    x, y, _ = make_toy(seed=5)
    xt, yt, _ = make_toy_test(5, 5, "iid")
    for c in range(5):
        white = (x[y == c][0] > 0.999).all(0)               # block pixels of the class (white in the three channels)
        white_t = (xt[yt == c][0] > 0.999).all(0)
        assert torch.equal(white, white_t)


def test_test_partition_depends_on_its_own_generator_not_on_the_training_data():
    a, _, _ = make_toy_test(6, 1, "iid")
    b, _, _ = make_toy_test(6, 2, "iid")
    c, _, _ = make_toy_test(6, 1, "iid")
    assert not torch.equal(a, b) and torch.equal(a, c)


def test_aligned_and_inverted_backgrounds_follow_the_palette():
    """Aligned: the background of a class is constant (its palette colour, small noise); inverted: it is that of the next class."""
    xa, ya, _ = make_toy_test(7, 7, "aligned")
    xi, yi, _ = make_toy_test(7, 7, "inverted")
    corner_a = xa[:, :, 0, 0]                               # the corner pixel is always background
    corner_i = xi[:, :, 0, 0]
    mean_a = torch.stack([corner_a[ya == c].mean(0) for c in range(5)])
    mean_i = torch.stack([corner_i[yi == c].mean(0) for c in range(5)])
    assert float(corner_a[ya == 0].std(0).max()) < 0.06
    assert torch.allclose(mean_i, mean_a.roll(-1, 0), atol=0.02)    # class c shows the colour of class c + 1


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        make_toy_test(0, 0, "other")
