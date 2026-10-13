"""Tiny synthetic data used by the smoke test: a class = a block-shape drawn in white on a random
background colour. The background is the nuisance; the transformation family T re-draws it."""
import torch


def make_toy(n_per_class=300, n_classes=5, size=16, spurious=0.0, seed=0):
    """Returns images [N,3,size,size], labels [N], background-colour id [N] (for leakage checks).

    spurious = probability that the background colour is the class's palette colour (a shortcut)
    instead of a random colour.
    """
    g = torch.Generator().manual_seed(seed)
    protos = []
    while len(protos) < n_classes:
        blocks = (torch.rand(4, 4, generator=g) > 0.5).float()
        blocks[0, 0] = 0.0                                   # the corner pixel is always background
        if not any(torch.equal(blocks, p) for p in protos) and 4 <= blocks.sum() <= 11:
            protos.append(blocks)
    masks = torch.stack([b.repeat_interleave(size // 4, 0).repeat_interleave(size // 4, 1) for b in protos])
    palette = torch.rand(n_classes, 3, generator=g) * 0.6
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    n = len(labels)
    bg = torch.rand(n, 3, generator=g) * 0.6
    use_pal = torch.rand(n, generator=g) < spurious
    bg = torch.where(use_pal.unsqueeze(-1), palette[labels] + 0.03 * torch.randn(n, 3, generator=g), bg)
    m = masks[labels].unsqueeze(1)                           # [N,1,H,W]
    img = m * 1.0 + (1 - m) * bg.view(n, 3, 1, 1)
    bg_id = (bg.argmax(1) * 2 + (bg.sum(1) > 0.9).long())    # a coarse background label
    return img.clamp(0, 1), labels, bg_id


def background_redraw(x):
    """The nuisance family T for the toy data: new random background colour + a little noise.
    Background pixels are the ones close to the corner pixel (always background)."""
    n = x.shape[0]
    bg = x[:, :, 0:1, 0:1]
    is_bg = ((x - bg).abs().sum(1, keepdim=True) < 0.25).float()
    new = torch.rand(n, 3, 1, 1, device=x.device) * 0.6
    out = is_bg * new + (1 - is_bg) * x
    return (out + 0.02 * torch.randn_like(out)).clamp(0, 1)
