"""The MNIST dataset builder (on small fake idx files: no network) and the rotation operation of the `digits` bank.
Run: python -m pytest tests -q"""
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets                                    # noqa: E402
from cibirl import transforms as T                 # noqa: E402


def idx(arr: np.ndarray) -> bytes:
    """gzip-compressed idx file of a uint8 array."""
    header = bytes([0, 0, 8, arr.ndim]) + b"".join(int(s).to_bytes(4, "big") for s in arr.shape)
    return gzip.compress(header + arr.astype(np.uint8).tobytes())


@pytest.fixture
def fake_mnist(tmp_path, monkeypatch):
    """40 'images' per digit 0..9 whose pixels all equal 10 * digit + 1: the digit can be read back from the pixels."""
    labs = np.repeat(np.arange(10), 40)
    imgs = (10 * labs + 1).astype(np.uint8)[:, None, None] * np.ones((1, 28, 28), dtype=np.uint8)
    files = {"train-images-idx3-ubyte.gz": idx(imgs), "train-labels-idx1-ubyte.gz": idx(labs)}
    for name, blob in files.items():
        (tmp_path / name).write_bytes(blob)
    monkeypatch.setattr(datasets, "MNIST_FILES", {n: hashlib.md5(b).hexdigest() for n, b in files.items()})
    return tmp_path


def test_idx_roundtrip():
    a = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.gz"
        p.write_bytes(idx(a))
        assert np.array_equal(datasets.read_idx(p), a)
        p.write_bytes(idx(a[0, 0]))
        assert np.array_equal(datasets.read_idx(p), a[0, 0])


def test_build_mnist_shapes_labels_and_seeds(fake_mnist):
    d = datasets.build("mnist", seed=0, digits=[3, 1, 6], n_per_class=25, root=str(fake_mnist))
    assert d.images.shape == (75, 1, 28, 28) and d.images.dtype == torch.float32
    assert d.images.min() >= 0 and d.images.max() <= 1 and d.background is None
    assert torch.equal(d.labels, torch.arange(3).repeat_interleave(25))          # classes numbered in the order of `digits`
    pixel = (d.images[:, 0, 0, 0] * 255).round().long()
    assert torch.equal((pixel - 1) // 10, torch.tensor([3, 1, 6]).repeat_interleave(25))   # the right digits were drawn
    again = datasets.build("mnist", seed=0, digits=[3, 1, 6], n_per_class=25, root=str(fake_mnist))
    other = datasets.build("mnist", seed=1, digits=[3, 1, 6], n_per_class=25, root=str(fake_mnist))
    assert torch.equal(d.images, again.images)                                    # same seed, same sample
    assert d.images.shape == other.images.shape                                   # (each seed draws its own images from the pool of 40)
    assert d.info == {"digits": [3, 1, 6], "n_per_class": 25}


def test_build_mnist_errors(fake_mnist):
    with pytest.raises(ValueError, match="distinct"):
        datasets.build("mnist", seed=0, digits=[1, 1], root=str(fake_mnist))
    with pytest.raises(ValueError, match="only 40"):
        datasets.build("mnist", seed=0, digits=[1], n_per_class=41, root=str(fake_mnist))
    with pytest.raises(ValueError, match="dataset 'mnist'"):
        datasets.build("mnist", seed=0, colour="red")


def test_wrong_checksum_is_not_accepted(tmp_path, monkeypatch):
    (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(b"not mnist")
    monkeypatch.setattr(datasets, "MNIST_MIRRORS", ())                            # no network in the tests
    with pytest.raises(RuntimeError, match="could not download"):
        datasets.fetch_mnist(tmp_path)


def test_mnist_has_a_default_config_that_matches_the_data():
    from cibirl.train import TrainConfig
    cfg = TrainConfig.from_yaml(ROOT / datasets.DEFAULT_CONFIG["mnist"])
    assert (cfg.in_channels, cfg.image_size, cfg.num_slots) == (1, 28, 6)
    assert cfg.orbits_per_episode <= cfg.num_slots


# ---- rotation ----------------------------------------------------------------------------------------
def test_zero_rotation_is_identity_and_range_is_kept():
    x = torch.rand(8, 1, 28, 28)
    assert torch.allclose(T.from_spec({"ops": [{"random_rotation": {"max_degrees": 0}}]})(x), x, atol=1e-5)
    y = T.from_spec({"ops": [{"random_rotation": {"max_degrees": 15}}]})(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_rotation_turns_a_bar_by_a_bounded_angle():
    x = torch.zeros(200, 1, 28, 28)
    x[:, :, 4:24, 13:15] = 1.0                                                    # a vertical bar
    y = T.from_spec({"ops": [{"random_rotation": {"max_degrees": 15}}]})(x)
    ys, xs = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
    angles = []
    for img in y[:, 0]:
        w = img / img.sum()
        cx, cy = (w * xs).sum(), (w * ys).sum()
        cov = torch.stack([((xs - cx) ** 2 * w).sum(), ((xs - cx) * (ys - cy) * w).sum()])
        var_y = ((ys - cy) ** 2 * w).sum()
        angles.append(torch.rad2deg(torch.atan(cov[1] / var_y)).abs())             # tilt of the bar
    angles = torch.stack(angles)
    assert angles.max() < 16.5 and angles.max() > 10 and angles.mean() > 3        # bounded by 15 degrees, and really used


def test_rotation_is_independent_per_sample_and_validated():
    x = torch.zeros(1, 1, 16, 16)
    x[..., 2:14, 7:9] = 1.0
    y = T.from_spec({"ops": ["random_rotation"]})(x.expand(64, -1, -1, -1).contiguous())
    assert len({tuple(v.flatten()[:64].tolist()) for v in y}) > 30
    with pytest.raises(ValueError, match="max_degrees"):
        T.from_spec({"ops": [{"random_rotation": {"max_degrees": 120}}]})
    with pytest.raises(ValueError, match="square"):
        T.from_spec({"ops": ["random_rotation"]})(torch.rand(2, 1, 16, 12))


def test_digits_bank_is_the_documented_one():
    bank = T.build(ROOT / "configs" / "transforms" / "digits.yml")
    assert [n for n, _ in bank.ops] == ["random_resized_crop", "random_rotation", "color_jitter"]
    x = torch.rand(8, 1, 28, 28)                                                   # works on the one-channel data
    assert bank(x).shape == x.shape
