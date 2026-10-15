"""The datasets the experiment runner can build, and the dataset-specific parts of the nuisance family.

A dataset is a function `build_<name>(seed, **params) -> DataBundle`. Labels (and the optional background
id) are for EVALUATION and figures only: they never reach the model.

To add a dataset: write its builder, add it to DATASETS, give it a default config file.
"""
from __future__ import annotations

import gzip
import hashlib
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cibirl import transforms                      # noqa: E402
from toy_data import background_redraw, make_toy   # noqa: E402


@dataclass
class DataBundle:
    images: Tensor                    # [N, C, H, W] float in [0, 1]
    labels: Tensor                    # [N] true classes, evaluation only
    background: Tensor | None         # [N] a nuisance id (e.g. background colour), evaluation only
    info: dict = field(default_factory=dict)   # the parameters it was built with, recorded in the run


# --------------------------------------------------------------------------------------------------
# Toy: white block-shapes on a coloured background
# --------------------------------------------------------------------------------------------------
@transforms.register("toy_background_redraw")
def _toy_background_redraw():
    """Oracle T of the toy data (scripts/toy_data.py): redraw the background colour exactly."""
    return background_redraw


def build_toy(seed: int, n_classes: int = 5, n_per_class: int = 300, size: int = 16, spurious: float = 0.0) -> DataBundle:
    images, labels, background = make_toy(n_per_class=n_per_class, n_classes=n_classes, size=size,
                                          spurious=spurious, seed=seed)
    return DataBundle(images, labels, background,
                      {"n_classes": n_classes, "n_per_class": n_per_class, "size": size, "spurious": spurious})


# --------------------------------------------------------------------------------------------------
# MNIST: a chosen subset of the digits, grey 28 x 28, one channel
# --------------------------------------------------------------------------------------------------
MNIST_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"          # not tracked by git (see .gitignore)
MNIST_FILES = {                                                              # file -> md5 of the official MNIST file
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
MNIST_MIRRORS = (                                                            # tried in order; the md5 decides if a copy is good
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_mnist(root: Path = MNIST_DIR) -> Path:
    """Make sure the two MNIST training files are in `root` (downloaded once, md5-checked); returns `root`."""
    root.mkdir(parents=True, exist_ok=True)
    for fname, md5 in MNIST_FILES.items():
        path = root / fname
        if path.exists() and _md5(path) == md5:
            continue
        errors = []
        for base in MNIST_MIRRORS:
            try:
                with urllib.request.urlopen(base + fname, timeout=60) as r:
                    path.write_bytes(r.read())
                if _md5(path) == md5:
                    break
                errors.append(f"{base}: wrong checksum")
            except OSError as e:
                errors.append(f"{base}: {e}")
        else:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {fname}:\n  " + "\n  ".join(errors)
                               + f"\nPut the official file in {root} by hand.")
    return root


def read_idx(path: Path) -> np.ndarray:
    """The idx format of MNIST (gzip): big-endian header, then the raw bytes. Images [N, 28, 28] or labels [N]."""
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def build_mnist(seed: int, digits: list[int] | None = None, n_per_class: int = 300, root: str | None = None) -> DataBundle:
    """`n_per_class` images of each digit in `digits` (default 1..6), drawn at random from the 60 000 MNIST training
    images with the run's seed (each seed has its own sample). Labels are for evaluation only."""
    digits = list(range(1, 7)) if digits is None else [int(d) for d in digits]
    if len(set(digits)) != len(digits) or not digits or not all(0 <= d <= 9 for d in digits):
        raise ValueError(f"digits must be distinct values in 0..9, got {digits}")
    folder = fetch_mnist(Path(root)) if root else fetch_mnist()
    imgs, labs = read_idx(folder / "train-images-idx3-ubyte.gz"), read_idx(folder / "train-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(seed)
    picked, labels = [], []
    for c, d in enumerate(digits):
        pool = np.flatnonzero(labs == d)
        if len(pool) < n_per_class:
            raise ValueError(f"digit {d} has only {len(pool)} training images, n_per_class={n_per_class}")
        picked.append(rng.choice(pool, n_per_class, replace=False))
        labels += [c] * n_per_class
    x = torch.from_numpy(imgs[np.concatenate(picked)].astype(np.float32) / 255.0).unsqueeze(1)   # [N, 1, 28, 28]
    return DataBundle(x, torch.tensor(labels), None, {"digits": digits, "n_per_class": n_per_class})


DATASETS = {"toy": build_toy, "mnist": build_mnist}
DEFAULT_CONFIG = {"toy": "configs/toy.yml", "mnist": "configs/mnist.yml"}   # TrainConfig file used when a run does not give one


def build(name: str, seed: int, **params) -> DataBundle:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    try:
        return DATASETS[name](seed, **params)
    except TypeError as e:
        raise ValueError(f"dataset '{name}': {e}") from None
