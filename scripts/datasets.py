"""The datasets the experiment runner can build, and the dataset-specific parts of the nuisance family.

A dataset is a function `build_<name>(seed, **params) -> DataBundle`. Labels (and the optional background
id) are for EVALUATION and figures only: they never reach the model.

A BENCHMARK is a dataset that also has TEST partitions (images the model never saw, under several nuisances): a pair
`train(seed, **params)` / `test(seed, mode, **params)` and the list of its test modes. The three benchmarks of the paper:

    toy            5 classes of white blocks on a coloured background        modes iid, aligned, inverted
    colored_mnist  MNIST digits painted in a colour tied to the class        modes iid, aligned, inverted
    waterbirds     landbirds / waterbirds on land / water backgrounds        modes iid, aligned, inverted (groups of the test split)

All methods (ours and the baselines of `baselines/`) read them through `build` and `build_test`, and are scored by
`eval/protocol.py`, so a new method or a new benchmark only touches one side.

To add a dataset: write its builder, add it to DATASETS, give it a default config file. For a benchmark, also write its
test builder and add it to BENCHMARKS.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cibirl import transforms                      # noqa: E402
from toy_data import background_redraw, load_toy, make_toy, make_toy_test, split_name, TEST_MODES   # noqa: E402


@dataclass
class DataBundle:
    images: Tensor                    # [N, C, H, W] float in [0, 1]
    labels: Tensor                    # [N] true classes, evaluation only
    background: Tensor | None         # [N] a nuisance id (e.g. background colour), evaluation only
    info: dict = field(default_factory=dict)   # the parameters it was built with, recorded in the run
    source: str = "generated"         # where the pixels came from: "generated", "file" (data/toy/...) or a folder name


# --------------------------------------------------------------------------------------------------
# Toy: white block-shapes on a coloured background
# --------------------------------------------------------------------------------------------------
@transforms.register("toy_background_redraw")
def _toy_background_redraw():
    """Oracle T of the toy data (scripts/toy_data.py): redraw the background colour exactly."""
    return background_redraw


def _toy_is_default(n_classes, n_per_class, size) -> bool:
    return (n_classes, n_per_class, size) == (5, 300, 16)


def build_toy(seed: int, n_classes: int = 5, n_per_class: int = 300, size: int = 16, spurious: float = 0.0,
              source: str = "auto") -> DataBundle:
    """The training images of a seed. `spurious` = probability that the background is the class's palette colour (a shortcut).

    source  "auto"      read data/toy/seedNNN.npz when it holds this configuration, else generate (the images are the same)
            "file"      the file or an error
            "generate"  always generate (what the results of the README were produced with before the files existed)
    """
    if source not in ("auto", "file", "generate"):
        raise ValueError(f"source must be auto, file or generate, got '{source}'")
    triple, where = None, "generated"
    if source != "generate" and _toy_is_default(n_classes, n_per_class, size):
        try:
            triple = load_toy(seed, split_name(spurious))
        except KeyError:
            triple = None
        if triple is not None:
            where = "file"
    if triple is None:
        if source == "file":
            raise FileNotFoundError(f"no file for toy seed {seed}, spurious {spurious} in data/toy/ "
                                    f"(python scripts/make_toy_dataset.py --seeds {seed} --spurious {spurious:g})")
        triple = make_toy(n_per_class=n_per_class, n_classes=n_classes, size=size, spurious=spurious, seed=seed)
    images, labels, background = triple
    return DataBundle(images, labels, background,
                      {"n_classes": n_classes, "n_per_class": n_per_class, "size": size, "spurious": spurious}, where)


def build_toy_test(seed: int, mode: str = "iid", n_classes: int = 5, n_per_class: int = 300, size: int = 16,
                   spurious: float | None = None, source: str = "auto") -> DataBundle:
    """A test partition for the model trained on `build_toy(seed)`: same classes, new images, background `mode`
    (iid, aligned, inverted). `spurious` is accepted and ignored: the test partition does not depend on the training shortcut."""
    if mode not in TEST_MODES:
        raise ValueError(f"unknown mode '{mode}': {', '.join(TEST_MODES)}")
    triple, where = None, "generated"
    if source != "generate" and _toy_is_default(n_classes, n_per_class, size):
        triple = load_toy(seed, f"test_{mode}")
        where = "file" if triple is not None else where
    if triple is None:
        if source == "file":
            raise FileNotFoundError(f"no file for toy seed {seed} in data/toy/ (python scripts/make_toy_dataset.py --seeds {seed})")
        triple = make_toy_test(seed, seed, mode, n_per_class=n_per_class, n_classes=n_classes, size=size)
    images, labels, background = triple
    return DataBundle(images, labels, background, {"n_classes": n_classes, "n_per_class": n_per_class, "size": size, "mode": mode},
                      where)


# --------------------------------------------------------------------------------------------------
# MNIST: a chosen subset of the digits, grey 28 x 28, one channel
# --------------------------------------------------------------------------------------------------
MNIST_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"          # not tracked by git (see .gitignore)
MNIST_FILES = {                                                              # file -> md5 of the official MNIST file
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}
MNIST_TEST_FILES = {                                                         # the 10 000 official test images (Colored MNIST test sets)
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MNIST_MIRRORS = (                                                            # tried in order; the md5 decides if a copy is good
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_mnist(root: Path = MNIST_DIR, files: dict | None = None) -> Path:
    """Make sure the MNIST files (default: the two training files) are in `root` (downloaded once, md5-checked); returns `root`."""
    root.mkdir(parents=True, exist_ok=True)
    for fname, md5 in (files or MNIST_FILES).items():
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


def _sample_digits(imgs: np.ndarray, labs: np.ndarray, digits: list[int] | None, n_per_class: int, rng) -> tuple[Tensor, Tensor, list[int]]:
    """`n_per_class` random images of each digit of `digits` (default 1..6), classes numbered in the order of `digits`."""
    digits = list(range(1, 7)) if digits is None else [int(d) for d in digits]
    if len(set(digits)) != len(digits) or not digits or not all(0 <= d <= 9 for d in digits):
        raise ValueError(f"digits must be distinct values in 0..9, got {digits}")
    picked, labels = [], []
    for c, d in enumerate(digits):
        pool = np.flatnonzero(labs == d)
        if len(pool) < n_per_class:
            raise ValueError(f"digit {d} has only {len(pool)} training images, n_per_class={n_per_class}")
        picked.append(rng.choice(pool, n_per_class, replace=False))
        labels += [c] * n_per_class
    x = torch.from_numpy(imgs[np.concatenate(picked)].astype(np.float32) / 255.0).unsqueeze(1)   # [N, 1, 28, 28]
    return x, torch.tensor(labels), digits


def build_mnist(seed: int, digits: list[int] | None = None, n_per_class: int = 300, root: str | None = None) -> DataBundle:
    """`n_per_class` images of each digit in `digits` (default 1..6), drawn at random from the 60 000 MNIST training
    images with the run's seed (each seed has its own sample). Labels are for evaluation only."""
    if digits is not None and (len(set(digits)) != len(digits) or not digits or not all(0 <= int(d) <= 9 for d in digits)):
        raise ValueError(f"digits must be distinct values in 0..9, got {digits}")
    folder = fetch_mnist(Path(root)) if root else fetch_mnist()
    imgs, labs = read_idx(folder / "train-images-idx3-ubyte.gz"), read_idx(folder / "train-labels-idx1-ubyte.gz")
    x, y, digits = _sample_digits(imgs, labs, digits, n_per_class, np.random.default_rng(seed))
    return DataBundle(x, y, None, {"digits": digits, "n_per_class": n_per_class})


# --------------------------------------------------------------------------------------------------
# Colored MNIST: the digit is painted in a colour; the colour is the nuisance (and, in training, a shortcut)
# --------------------------------------------------------------------------------------------------
def _hue_to_rgb(h: Tensor) -> Tensor:
    """Hue in [0, 1) at full saturation and value -> RGB [N, 3]."""
    h6 = (h % 1.0) * 6
    return torch.stack([(h6 - 3).abs() - 1, 2 - (h6 - 2).abs(), 2 - (h6 - 4).abs()], 1).clamp(0, 1)


def _paint(gray: Tensor, hue: Tensor) -> Tensor:
    """gray digits [N,1,H,W] -> [N,3,H,W]: the strokes in the colour `hue`, the background black."""
    return gray * _hue_to_rgb(hue).view(-1, 3, 1, 1)


def _class_hues(n_classes: int) -> Tensor:
    return torch.arange(n_classes).float() / n_classes          # the palette: one hue per class, evenly spaced


def _hue_id(hue: Tensor, bins: int = 8) -> Tensor:
    return ((hue % 1.0) * bins).long().clamp(max=bins - 1)    # a coarse colour label for NMI(cluster, colour) and worst groups


def build_colored_mnist(seed: int, digits: list[int] | None = None, n_per_class: int = 300, spurious: float = 0.0,
                        root: str | None = None) -> DataBundle:
    """MNIST digits (default 0..9) painted in a colour. The colour is the nuisance.

    spurious = probability that the colour is the palette colour of the digit's class (a shortcut) instead of a random hue;
    0 = the colour carries no information on the class. Same convention as the toy. Labels and the colour id are for
    evaluation only. Training images come from the MNIST training set, the test partitions from the MNIST test set."""
    digits = list(range(10)) if digits is None else digits
    folder = fetch_mnist(Path(root)) if root else fetch_mnist()
    imgs, labs = read_idx(folder / "train-images-idx3-ubyte.gz"), read_idx(folder / "train-labels-idx1-ubyte.gz")
    x, y, digits = _sample_digits(imgs, labs, digits, n_per_class, np.random.default_rng(seed))
    g = torch.Generator().manual_seed(seed + 10_007)
    hue = torch.rand(len(y), generator=g)
    use = torch.rand(len(y), generator=g) < spurious
    hue = torch.where(use, (_class_hues(len(digits))[y] + 0.01 * torch.randn(len(y), generator=g)) % 1.0, hue)
    return DataBundle(_paint(x, hue), y, _hue_id(hue), {"digits": digits, "n_per_class": n_per_class, "spurious": spurious})


def build_colored_mnist_test(seed: int, mode: str = "iid", digits: list[int] | None = None, n_per_class: int = 300,
                             spurious: float | None = None, root: str | None = None) -> DataBundle:
    """Test partition: MNIST TEST images of the same digits, painted by `mode` (iid: random hue; aligned: the palette
    colour of the class; inverted: the palette colour of the next class). `spurious` is accepted and ignored."""
    if mode not in TEST_MODES:
        raise ValueError(f"unknown mode '{mode}': {', '.join(TEST_MODES)}")
    digits = list(range(10)) if digits is None else digits
    folder = fetch_mnist(Path(root) if root else MNIST_DIR, MNIST_TEST_FILES)
    imgs, labs = read_idx(folder / "t10k-images-idx3-ubyte.gz"), read_idx(folder / "t10k-labels-idx1-ubyte.gz")
    x, y, digits = _sample_digits(imgs, labs, digits, n_per_class, np.random.default_rng(1_000_003 * (seed + 1) + 7))
    g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + 11)
    if mode == "iid":
        hue = torch.rand(len(y), generator=g)
    else:
        src = y if mode == "aligned" else (y + 1) % len(digits)
        hue = (_class_hues(len(digits))[src] + 0.01 * torch.randn(len(y), generator=g)) % 1.0
    return DataBundle(_paint(x, hue), y, _hue_id(hue), {"digits": digits, "n_per_class": n_per_class, "mode": mode})


# --------------------------------------------------------------------------------------------------
# Waterbirds: landbirds / waterbirds on land / water backgrounds (the correlation is in the training split)
# --------------------------------------------------------------------------------------------------
WATERBIRDS_DIR = Path(__file__).resolve().parents[1] / "data" / "waterbirds"     # not tracked by git
WATERBIRDS_HELP = ("Waterbirds is not downloaded by the code (about 1.8 GB). Get the official release (WILDS `waterbirds_v1.0`, "
                   "or the Group-DRO `waterbird_complete95_forest2water2`, https://github.com/kohpangwei/group_DRO), extract it "
                   "in {root}: the folder must contain metadata.csv (columns img_id, img_filename, y, split, place) and the images.")


def _waterbirds_meta(root: Path) -> tuple[Path, list[dict]]:
    """(folder of the images, rows of metadata.csv) found in `root` or in one of its sub-folders."""
    root = Path(root)
    hits = [p for p in [root / "metadata.csv", *sorted(root.glob("*/metadata.csv"))] if p.exists()]
    if not hits:
        raise FileNotFoundError(WATERBIRDS_HELP.format(root=root))
    with open(hits[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"img_filename", "y", "split", "place"} - set(rows[0]) if rows else {"rows"}
    if missing:
        raise ValueError(f"{hits[0]}: missing column(s) {sorted(missing)}")
    return hits[0].parent, rows


def _load_waterbirds(root: Path, split: int, size: int, sel=None, seed: int = 0, max_images: int | None = None) -> DataBundle:
    """Images of one official split (0 train, 1 val, 2 test), resized to size x size, float in [0, 1]. `sel(row)` keeps a subset;
    `max_images` draws that many at random (with `seed`). y: 0 landbird, 1 waterbird. background: place, 0 land, 1 water."""
    from PIL import Image                                              # only Waterbirds needs Pillow
    folder, rows = _waterbirds_meta(root)
    keep = [r for r in rows if int(r["split"]) == split and (sel is None or sel(r))]
    if not keep:
        raise ValueError(f"no image in split {split} with this selection in {folder / 'metadata.csv'}")
    if max_images is not None and len(keep) > max_images:
        keep = [keep[i] for i in sorted(np.random.default_rng(seed).choice(len(keep), max_images, replace=False))]
    xs = []
    for r in keep:
        with Image.open(folder / r["img_filename"]) as im:
            xs.append(np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    y, place = torch.tensor([int(r["y"]) for r in keep]), torch.tensor([int(r["place"]) for r in keep])
    return DataBundle(x, y, place, {}, str(folder.name))


def build_waterbirds(seed: int, size: int = 64, max_images: int | None = None, root: str | None = None) -> DataBundle:
    """The official TRAINING split (95 % of the landbirds on land, of the waterbirds on water): the shortcut is in the data,
    as it is in the benchmark. `max_images` draws a random subset with the run's seed. Labels and place are for evaluation only."""
    b = _load_waterbirds(Path(root) if root else WATERBIRDS_DIR, 0, size, None, seed, max_images)
    b.info = {"size": size, "max_images": max_images}
    return b


def build_waterbirds_test(seed: int, mode: str = "iid", size: int = 64, max_images: int | None = None, split: int = 2,
                          root: str | None = None, spurious: float | None = None) -> DataBundle:
    """Test partition of Waterbirds (official test split by default; `split=1` is the validation split):
    iid       the whole split (the background is independent of the class inside each class)
    aligned   only the images whose background matches the bird (landbird on land, waterbird on water)
    inverted  only the images whose background does not match (landbird on water, waterbird on land)
    `spurious` is accepted and ignored."""
    if mode not in TEST_MODES:
        raise ValueError(f"unknown mode '{mode}': {', '.join(TEST_MODES)}")
    sel = None if mode == "iid" else ((lambda r: r["y"] == r["place"]) if mode == "aligned" else (lambda r: r["y"] != r["place"]))
    b = _load_waterbirds(Path(root) if root else WATERBIRDS_DIR, split, size, sel, seed, max_images)
    b.info = {"size": size, "max_images": max_images, "mode": mode, "split": split}
    return b


DATASETS = {"toy": build_toy, "mnist": build_mnist, "colored_mnist": build_colored_mnist, "waterbirds": build_waterbirds}
DEFAULT_CONFIG = {"toy": "configs/toy.yml", "mnist": "configs/mnist.yml",     # TrainConfig file used when a run does not give one
                  "colored_mnist": "configs/colored_mnist.yml", "waterbirds": "configs/waterbirds.yml"}


@dataclass(frozen=True)
class Benchmark:
    """A dataset with test partitions. `test(seed, mode, **params)` takes the same params as `train` (it ignores `spurious`)."""
    name: str
    train: Callable
    test: Callable
    modes: tuple = TEST_MODES


BENCHMARKS = {"toy": Benchmark("toy", build_toy, build_toy_test),
              "colored_mnist": Benchmark("colored_mnist", build_colored_mnist, build_colored_mnist_test),
              "waterbirds": Benchmark("waterbirds", build_waterbirds, build_waterbirds_test)}


def build(name: str, seed: int, **params) -> DataBundle:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    try:
        return DATASETS[name](seed, **params)
    except TypeError as e:
        raise ValueError(f"dataset '{name}': {e}") from None


def build_test(name: str, seed: int, mode: str, **params) -> DataBundle:
    """The test partition `mode` of a benchmark, for the model trained on `build(name, seed, **params)`."""
    if name not in BENCHMARKS:
        raise KeyError(f"dataset '{name}' has no test partitions. Benchmarks: {sorted(BENCHMARKS)}")
    try:
        return BENCHMARKS[name].test(seed, mode, **params)
    except TypeError as e:
        raise ValueError(f"benchmark '{name}': {e}") from None
