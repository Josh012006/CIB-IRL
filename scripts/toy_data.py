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
