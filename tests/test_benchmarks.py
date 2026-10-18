"""The benchmark abstraction (train builder + test builder + modes) for the toy, Colored MNIST and Waterbirds, on small fake data
(no network, no download). Run: python -m pytest tests -q"""
import csv
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
from eval.metrics import nmi                       # noqa: E402
from eval.protocol import TEST_MODES               # noqa: E402


def test_registry_and_errors():
    assert set(datasets.BENCHMARKS) == {"toy", "colored_mnist", "waterbirds"}
    assert all(b.modes == TEST_MODES for b in datasets.BENCHMARKS.values())
    with pytest.raises(KeyError, match="no test partitions"):
        datasets.build_test("mnist", 0, "iid")
    with pytest.raises(ValueError, match="unknown mode"):
        datasets.build_test("toy", 0, "sideways")
    with pytest.raises(ValueError, match="benchmark 'toy'"):
        datasets.build_test("toy", 0, "iid", colour="red")                           # a wrong parameter is named, not swallowed


# ---- toy ----------------------------------------------------------------------------------------------
def test_toy_test_modes_have_the_documented_backgrounds():
    tr = datasets.build("toy", 0, n_per_class=60, source="generate")
    iid, ali, inv = (datasets.build_test("toy", 0, m, n_per_class=60, source="generate") for m in TEST_MODES)
    for t in (iid, ali, inv):
        assert t.images.shape == tr.images.shape and torch.equal(t.labels, tr.labels)
    assert nmi(iid.background.numpy(), iid.labels.numpy()) < 0.1                    # independent of the class
    assert nmi(ali.background.numpy(), ali.labels.numpy()) > 0.7                    # points to the right class
    assert nmi(inv.background.numpy(), inv.labels.numpy()) > 0.7                    # still a function of the class ...
    assert not torch.equal(ali.background, inv.background)                           # ... but another one: the next class
    assert not torch.allclose(tr.images, iid.images)                                 # the test images were never seen in training
    again = datasets.build_test("toy", 0, "aligned", n_per_class=60, source="generate")
    assert torch.equal(again.images, ali.images)                                     # deterministic


def test_toy_spurious_is_a_training_shortcut_only():
    clean = datasets.build("toy", 1, n_per_class=100, spurious=0.0, source="generate")
    spur = datasets.build("toy", 1, n_per_class=100, spurious=0.9, source="generate")
    assert nmi(clean.background.numpy(), clean.labels.numpy()) < 0.15
    assert nmi(spur.background.numpy(), spur.labels.numpy()) > 0.3                  # 0.47 against < 0.05 without shortcut
    a = datasets.build_test("toy", 1, "iid", n_per_class=100, spurious=0.9, source="generate")       # accepted and ignored
    b = datasets.build_test("toy", 1, "iid", n_per_class=100, source="generate")
    assert torch.equal(a.images, b.images)


# ---- Colored MNIST ------------------------------------------------------------------------------------
def idx(arr):
    header = bytes([0, 0, 8, arr.ndim]) + b"".join(int(s).to_bytes(4, "big") for s in arr.shape)
    return gzip.compress(header + arr.astype(np.uint8).tobytes())


@pytest.fixture
def fake_mnist(tmp_path, monkeypatch):
    """60 images per digit in train and in t10k, all pixels equal to 20 * digit + 40 (the digit can be read from the pixels)."""
    files = {}
    for prefix, name in (("train", "train"), ("t10k", "t10k")):
        labs = np.repeat(np.arange(10), 60)
        imgs = (20 * labs + 40).astype(np.uint8)[:, None, None] * np.ones((1, 28, 28), dtype=np.uint8)
        files[f"{name}-images-idx3-ubyte.gz"], files[f"{name}-labels-idx1-ubyte.gz"] = idx(imgs), idx(labs)
    for n, blob in files.items():
        (tmp_path / n).write_bytes(blob)
    md5 = {n: hashlib.md5(b).hexdigest() for n, b in files.items()}
    monkeypatch.setattr(datasets, "MNIST_FILES", {n: m for n, m in md5.items() if n.startswith("train")})
    monkeypatch.setattr(datasets, "MNIST_TEST_FILES", {n: m for n, m in md5.items() if n.startswith("t10k")})
    monkeypatch.setattr(datasets, "MNIST_DIR", tmp_path)
    return tmp_path


def cm_train(root, **kw):
    return datasets.build("colored_mnist", 0, n_per_class=50, root=str(root), **kw)


def cm_test(root, mode, **kw):
    return datasets.build_test("colored_mnist", 0, mode, n_per_class=50, root=str(root), **kw)


def test_colored_mnist_shapes_and_the_colour_is_a_nuisance(fake_mnist):
    b = cm_train(fake_mnist)
    assert b.images.shape == (500, 3, 28, 28) and b.labels.shape == (500,) and b.images.min() >= 0 and b.images.max() <= 1
    assert nmi(b.background.numpy(), b.labels.numpy()) < 0.1                         # spurious 0: the colour says nothing on the class
    # the pixels of the digit are painted: the strongest channel is as bright as the grey level of the digit
    gray = (20 * b.labels.float() + 40) / 255
    assert torch.allclose(b.images.amax(dim=(1, 2, 3)), gray, atol=1e-5) or (b.images.amax(dim=(1, 2, 3)) <= gray + 1e-5).all()


def test_colored_mnist_spurious_makes_the_colour_a_shortcut_in_training_only(fake_mnist):
    spur = cm_train(fake_mnist, spurious=1.0)
    assert nmi(spur.background.numpy(), spur.labels.numpy()) > 0.8
    iid = cm_test(fake_mnist, "iid", spurious=1.0)                                    # ignored by the test builder
    assert nmi(iid.background.numpy(), iid.labels.numpy()) < 0.1


def test_colored_mnist_test_modes(fake_mnist):
    ali, inv = cm_test(fake_mnist, "aligned"), cm_test(fake_mnist, "inverted")
    assert nmi(ali.background.numpy(), ali.labels.numpy()) > 0.8 and nmi(inv.background.numpy(), inv.labels.numpy()) > 0.8
    # digit 0 (class 0) has the palette hue 0 (red) when aligned: no green or blue; inverted, it has the hue of class 1
    red = ali.images[ali.labels == 0]
    assert red[:, 1:].max() < 0.2 * red[:, 0].max() + 1e-6
    assert not torch.equal(ali.images, inv.images)
    train = cm_train(fake_mnist)
    assert not torch.equal(ali.images[:1], train.images[:1])                          # test digits come from the t10k set


def test_colored_mnist_has_the_right_default_config():
    from cibirl.train import TrainConfig
    cfg = TrainConfig.from_yaml(ROOT / datasets.DEFAULT_CONFIG["colored_mnist"])
    assert (cfg.in_channels, cfg.image_size) == (3, 28)


# ---- Waterbirds ---------------------------------------------------------------------------------------
@pytest.fixture
def fake_waterbirds(tmp_path):
    from PIL import Image
    rows, rng = [], np.random.default_rng(0)
    for i in range(48):
        split, y, place = (0 if i < 24 else 2), i % 2, (i // 2) % 2                   # all four (y, place) groups in each split
        name = f"img{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(tmp_path / name)
        rows.append({"img_id": i, "img_filename": name, "y": y, "split": split, "place": place})
    with open(tmp_path / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_waterbirds_train_split_and_resize(fake_waterbirds):
    b = datasets.build("waterbirds", 0, size=32, root=str(fake_waterbirds))
    assert b.images.shape == (24, 3, 32, 32) and b.images.min() >= 0 and b.images.max() <= 1
    assert set(b.labels.tolist()) == {0, 1} and set(b.background.tolist()) == {0, 1}
    small = datasets.build("waterbirds", 0, size=16, max_images=10, root=str(fake_waterbirds))
    assert small.images.shape == (10, 3, 16, 16)
    again = datasets.build("waterbirds", 0, size=16, max_images=10, root=str(fake_waterbirds))
    assert torch.equal(small.images, again.images)                                    # the subset depends on the seed only


def test_waterbirds_test_modes_select_matching_or_mismatching_backgrounds(fake_waterbirds):
    def t(mode):
        return datasets.build_test("waterbirds", 0, mode, size=16, root=str(fake_waterbirds))
    iid, ali, inv = t("iid"), t("aligned"), t("inverted")
    assert len(iid.labels) == 24 and len(ali.labels) + len(inv.labels) == 24
    assert (ali.labels == ali.background).all() and (inv.labels != inv.background).all()
    assert not torch.equal(iid.images[:1], datasets.build("waterbirds", 0, size=16, root=str(fake_waterbirds)).images[:1])  # test split


def test_waterbirds_errors_tell_what_to_do(tmp_path, fake_waterbirds):          # tmp_path holds the fake dataset too
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        datasets.build("waterbirds", 0, root=str(tmp_path / "empty"))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.csv").write_text("img_filename,y\na.jpg,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        datasets.build("waterbirds", 0, root=str(bad))
    with pytest.raises(ValueError, match="no image in split"):
        datasets.build_test("waterbirds", 0, "iid", split=1, root=str(fake_waterbirds))
