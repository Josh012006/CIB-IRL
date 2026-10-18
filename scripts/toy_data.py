"""Tiny synthetic data used by the smoke test: a class = a block-shape drawn in white on a random
background colour. The background is the nuisance; the transformation family T re-draws it."""
import torch


def _prototypes(seed, n_classes, size):
    """The shapes of the classes and their palette colours, drawn from the generator of `seed`. Returns
    (masks [C,H,W], palette [C,3], generator): the generator is returned in the state training data continue from."""
    g = torch.Generator().manual_seed(seed)
    protos = []
    while len(protos) < n_classes:
        blocks = (torch.rand(4, 4, generator=g) > 0.5).float()
        blocks[0, 0] = 0.0                                   # the corner pixel is always background
        if not any(torch.equal(blocks, p) for p in protos) and 4 <= blocks.sum() <= 11:
            protos.append(blocks)
    masks = torch.stack([b.repeat_interleave(size // 4, 0).repeat_interleave(size // 4, 1) for b in protos])
    palette = torch.rand(n_classes, 3, generator=g) * 0.6
    return masks, palette, g


def _render(masks, labels, bg):
    m = masks[labels].unsqueeze(1)                           # [N,1,H,W]
    img = m * 1.0 + (1 - m) * bg.view(len(labels), 3, 1, 1)
    return img.clamp(0, 1)


def _background_id(bg):
    return bg.argmax(1) * 2 + (bg.sum(1) > 0.9).long()       # a coarse background label


def make_toy(n_per_class=300, n_classes=5, size=16, spurious=0.0, seed=0):
    """Returns images [N,3,size,size], labels [N], background-colour id [N] (for leakage checks).

    spurious = probability that the background colour is the class's palette colour (a shortcut)
    instead of a random colour.
    """
    masks, palette, g = _prototypes(seed, n_classes, size)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    n = len(labels)
    bg = torch.rand(n, 3, generator=g) * 0.6
    use_pal = torch.rand(n, generator=g) < spurious
    bg = torch.where(use_pal.unsqueeze(-1), palette[labels] + 0.03 * torch.randn(n, 3, generator=g), bg)
    return _render(masks, labels, bg), labels, _background_id(bg)


def make_toy_test(seed, test_seed, mode="iid", n_per_class=300, n_classes=5, size=16):
    """A TEST partition for the model trained on `make_toy(seed=seed)`: the same classes (same shapes, same palette), but
    images that were never seen, with backgrounds drawn from their own generator (`test_seed`).

    mode "iid"       random backgrounds, as in training (spurious = 0)
    mode "aligned"   the background of a class is its palette colour (the shortcut points to the right class)
    mode "inverted"  the background of class c is the palette colour of class (c + 1) mod C (the shortcut points to a wrong class)

    The training data are not used. Returns images, labels, background-colour id (evaluation only)."""
    masks, palette, _ = _prototypes(seed, n_classes, size)
    g = torch.Generator().manual_seed(1_000_003 * (test_seed + 1) + 7)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    n = len(labels)
    if mode == "iid":
        bg = torch.rand(n, 3, generator=g) * 0.6
    elif mode in ("aligned", "inverted"):
        src = labels if mode == "aligned" else (labels + 1) % n_classes
        bg = (palette[src] + 0.03 * torch.randn(n, 3, generator=g)).clamp(0, 1)
    else:
        raise ValueError(f"unknown mode '{mode}': iid, aligned or inverted")
    return _render(masks, labels, bg), labels, _background_id(bg)


def background_redraw(x):
    """The nuisance family T for the toy data: new random background colour + a little noise.
    Background pixels are the ones close to the corner pixel (always background)."""
    n = x.shape[0]
    bg = x[:, :, 0:1, 0:1]
    is_bg = ((x - bg).abs().sum(1, keepdim=True) < 0.25).float()
    new = torch.rand(n, 3, 1, 1, device=x.device) * 0.6
    out = is_bg * new + (1 - is_bg) * x
    return (out + 0.02 * torch.randn_like(out)).clamp(0, 1)


# --------------------------------------------------------------------------------------------------
# The toy data as FILES (data/toy/), so that anybody can load the exact images used for the results
# without regenerating them. The generator above stays the source of truth: `scripts/make_toy_dataset.py`
# writes the files from it and checks them against it (`--verify`), and tests/test_toy_files.py does the same.
#
#   data/toy/seed000.npz     one file per seed, float32 images in [0, 1], int64 labels and background ids:
#       train_x/y/bg            the 1500 unlabeled images the model clusters   (make_toy(seed, spurious=0))
#       train_s90_x/y/bg        the same classes with a colour shortcut         (make_toy(seed, spurious=0.9)), if written
#       test_iid_x/y/bg         the three TEST partitions                       (make_toy_test(seed, seed, mode))
#       test_aligned_x/y/bg
#       test_inverted_x/y/bg
#   data/toy/manifest.json   generator parameters, seeds written, sha256 of every file
# --------------------------------------------------------------------------------------------------
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

TOY_DIR = Path(__file__).resolve().parents[1] / "data" / "toy"
TOY_DEFAULTS = {"n_classes": 5, "n_per_class": 300, "size": 16}      # the files hold exactly this configuration
TEST_MODES = ("iid", "aligned", "inverted")


def split_name(spurious: float = 0.0) -> str:
    """Name of a training split in the files: 'train' (no shortcut), 'train_s90' (spurious = 0.9)."""
    return "train" if not spurious else f"train_s{round(100 * spurious):02d}"


def toy_path(seed: int, root: Path | str | None = None) -> Path:
    return Path(root or TOY_DIR) / f"seed{seed:03d}.npz"


def toy_arrays(seed: int, spurious_levels=(0.9,)) -> dict:
    """Every array of the file of `seed`, generated (not read): the content `save_toy` writes."""
    out = {}

    def put(name, triple):
        x, y, bg = triple
        out[f"{name}_x"], out[f"{name}_y"], out[f"{name}_bg"] = x.numpy().astype(np.float32), y.numpy(), bg.numpy()

    put("train", make_toy(seed=seed, spurious=0.0, **TOY_DEFAULTS))
    for s in spurious_levels:
        put(split_name(s), make_toy(seed=seed, spurious=s, **TOY_DEFAULTS))
    for mode in TEST_MODES:
        put(f"test_{mode}", make_toy_test(seed, seed, mode, **TOY_DEFAULTS))
    return out


def save_toy(seed: int, root: Path | str | None = None, spurious_levels=(0.9,)) -> Path:
    path = toy_path(seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **toy_arrays(seed, spurious_levels))
    return path


def load_toy(seed: int, split: str = "train", root: Path | str | None = None):
    """(images [N,3,16,16] float32 tensor, labels [N], background id [N]) of one split of the file of `seed`:
    'train', 'train_s90', 'test_iid', 'test_aligned' or 'test_inverted'. Returns None if the file does not exist; raises
    KeyError if the file exists but lacks that split."""
    path = toy_path(seed, root)
    if not path.exists():
        return None
    with np.load(path) as f:
        if f"{split}_x" not in f.files:
            have = sorted({k[:-2] for k in f.files if k.endswith("_x")})
            raise KeyError(f"{path.name} has no split '{split}' (it has {have})")
        return torch.from_numpy(f[f"{split}_x"]), torch.from_numpy(f[f"{split}_y"]), torch.from_numpy(f[f"{split}_bg"])


def content_sha256(path: Path | str) -> str:
    """Hash of the ARRAYS of a file (names, dtypes, shapes, bytes), not of the zip: a rewrite of identical arrays gives the
    same digest, whatever the timestamps stored by the zip format."""
    h = hashlib.sha256()
    with np.load(path) as f:
        for k in sorted(f.files):
            a = f[k]
            h.update(f"{k}|{a.dtype}|{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
