"""The transformation family T (README Section 5.1) as a registry of small operations and a YAML bank.

T is not a robustness knob: it DEFINES the invariance we impose. Whatever T changes is declared a
nuisance (the views of one image stay one orbit), whatever it preserves is declared identity.
Too rich a T destroys signal; too poor a T leaves the nuisance in the intent m. That is why a bank
is a written, versioned object (configs/transforms/*.yml) and not code scattered in a script.

Protocol (see configs/transforms/README.md):

* `generic`         the core bank, applied unchanged to every dataset;
* `natural`         generic + horizontal flip: natural images only (a flipped digit is another digit);
* `informed_color`  generic + a full hue redraw: a DECLARED side information, used to measure how much a
                    result depends on a prior about the nuisance (e.g. the spurious colour of Colored MNIST).

An operation is admitted to a bank only if domain knowledge, written down BEFORE looking at any result,
says it preserves identity. Never choose it by looking at labels.

Conventions
-----------
* every operation maps a float batch [N, C, H, W] in [0, 1] to a batch of the same shape and range;
* randomness is drawn independently PER SAMPLE (two copies of one image get two different views);
* operations never modify their input, work on any device, and use only torch (no torchvision);
* chaining (`OrbitSampler`, d_max) applies the whole bank again to its own output: strengths compound.
"""
from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .experts import Transform

_REGISTRY: dict[str, Callable[..., Transform]] = {}
_RGB_WEIGHTS = (0.299, 0.587, 0.114)


def register(name: str):
    """Decorator: make a factory `f(**params) -> Transform` usable by name in a bank file."""
    def deco(factory):
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"transform '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def _check(x: Tensor, name: str, channels: int | None = None) -> None:
    if x.dim() != 4:
        raise ValueError(f"{name}: expected a batch [N, C, H, W], got shape {tuple(x.shape)}")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"{name}: needs {channels} channels (RGB), the data has {x.shape[1]}")


def _uniform(n: int, lo: float, hi: float, like: Tensor) -> Tensor:
    return lo + (hi - lo) * torch.rand(n, device=like.device, dtype=like.dtype)


def _blend(x: Tensor, y: Tensor, p: float) -> Tensor:
    """y for a random share p of the samples, x for the others (independent draw per sample)."""
    if p >= 1.0:
        return y
    take = torch.rand(x.shape[0], device=x.device) < p
    return torch.where(take.view(-1, 1, 1, 1), y, x)


def _gray(x: Tensor) -> Tensor:
    if x.shape[1] == 3:
        w = torch.tensor(_RGB_WEIGHTS, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * w).sum(1, keepdim=True)
    return x.mean(1, keepdim=True)


def _pair(v: Sequence[float], name: str) -> tuple[float, float]:
    if len(v) != 2 or v[0] > v[1]:
        raise ValueError(f"{name} must be [low, high] with low <= high, got {list(v)}")
    return float(v[0]), float(v[1])


def _prob(p: float, name: str = "p") -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return float(p)


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------
@register("random_resized_crop")
def random_resized_crop(scale: Sequence[float] = (0.7, 1.0), ratio: Sequence[float] = (0.8, 1.25)) -> Transform:
    """Crop a random window (area share in `scale`, aspect ratio in `ratio`) and resize it back to H x W.

    Done for the whole batch with one affine grid: no loop, per-sample windows.
    """
    s_lo, s_hi = _pair(scale, "scale")
    r_lo, r_hi = _pair(ratio, "ratio")
    if not 0.0 < s_lo or s_hi > 1.0 or r_lo <= 0.0:
        raise ValueError("scale must lie in (0, 1] and ratio must be positive")

    def op(x: Tensor) -> Tensor:
        _check(x, "random_resized_crop")
        n = x.shape[0]
        area = _uniform(n, s_lo, s_hi, x)
        r = _uniform(n, math.log(r_lo), math.log(r_hi), x).exp()
        w = (area * r).sqrt().clamp(max=1.0)                 # window size as a share of the image
        h = (area / r).sqrt().clamp(max=1.0)
        cx = (2 * torch.rand_like(w) - 1) * (1 - w)          # centre, so that the window stays inside
        cy = (2 * torch.rand_like(h) - 1) * (1 - h)
        theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w, cx, h, cy
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    return op


@register("hflip")
def hflip(p: float = 0.5) -> Transform:
    """Mirror left-right with probability p. Natural images only: it changes the identity of digits."""
    p = _prob(p)

    def op(x: Tensor) -> Tensor:
        _check(x, "hflip")
        return _blend(x, x.flip(-1), p)

    return op


# --------------------------------------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------------------------------------
@register("color_jitter")
def color_jitter(brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, p: float = 1.0) -> Transform:
    """Random brightness / contrast / saturation factors in [1 - s, 1 + s], applied in this fixed order.

    The HUE is left alone: this is what separates `generic` from `informed_color`.
    """
    p = _prob(p)
    for name, s in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
        if s < 0:
            raise ValueError(f"{name} must be >= 0, got {s}")

    def op(x: Tensor) -> Tensor:
        _check(x, "color_jitter", 3 if saturation > 0 else None)
        n = x.shape[0]
        out = x
        if brightness > 0:
            out = out * _uniform(n, 1 - brightness, 1 + brightness, x).view(-1, 1, 1, 1)
        if contrast > 0:
            mean = _gray(out).mean(dim=(1, 2, 3), keepdim=True)
            out = (out - mean) * _uniform(n, 1 - contrast, 1 + contrast, x).view(-1, 1, 1, 1) + mean
        if saturation > 0:
            g = _gray(out)
            out = (out - g) * _uniform(n, 1 - saturation, 1 + saturation, x).view(-1, 1, 1, 1) + g
        return _blend(x, out.clamp(0.0, 1.0), p)

    return op


@register("grayscale")
def grayscale(p: float = 0.1) -> Transform:
    """Drop the colour (luma replicated on the 3 channels) with probability p."""
    p = _prob(p)

    def op(x: Tensor) -> Tensor:
        _check(x, "grayscale", 3)
        return _blend(x, _gray(x).expand_as(x), p)

    return op


def _hue_matrix(theta: Tensor) -> Tensor:
    """Rotation by `theta` (radians, shape [N]) around the grey axis (1,1,1)/sqrt(3) of RGB space: [N, 3, 3].

    Greys are fixed points; 120 degrees maps red to green to blue. Circulant matrix [[a,b,d],[d,a,b],[b,d,a]].
    """
    c, s = theta.cos(), theta.sin()
    a = c + (1 - c) / 3
    b = (1 - c) / 3 - s / math.sqrt(3)
    d = (1 - c) / 3 + s / math.sqrt(3)
    return torch.stack([a, b, d, d, a, b, b, d, a], dim=-1).view(-1, 3, 3)


@register("hue_rotate")
def hue_rotate(max_degrees: float = 180.0, p: float = 1.0) -> Transform:
    """Rotate the hue by an angle drawn uniformly in [-max_degrees, +max_degrees].

    With 180 the hue is fully redrawn: the original colour carries no information about the view.
    It is a rotation in RGB space (then clipped to [0, 1]), an approximation of a perceptual hue shift.
    """
    p = _prob(p)
    if not 0.0 <= max_degrees <= 180.0:
        raise ValueError(f"max_degrees must be in [0, 180], got {max_degrees}")
    max_rad = math.radians(max_degrees)

    def op(x: Tensor) -> Tensor:
        _check(x, "hue_rotate", 3)
        rot = _hue_matrix(_uniform(x.shape[0], -max_rad, max_rad, x))
        out = torch.einsum("nij,njhw->nihw", rot, x).clamp(0.0, 1.0)
        return _blend(x, out, p)

    return op


@register("channel_permute")
def channel_permute(p: float = 1.0) -> Transform:
    """Shuffle the RGB channels (6 possible colour re-assignments) with probability p."""
    p = _prob(p)

    def op(x: Tensor) -> Tensor:
        _check(x, "channel_permute", 3)
        perm = torch.rand(x.shape[0], 3, device=x.device).argsort(dim=1)
        out = x.gather(1, perm.view(-1, 3, 1, 1).expand_as(x))
        return _blend(x, out, p)

    return op


# --------------------------------------------------------------------------------------------------
# Banks
# --------------------------------------------------------------------------------------------------
class Compose:
    """Operations applied one after the other. Keeps the spec it was built from, for logs and run folders."""

    def __init__(self, ops: list[tuple[str, Transform]], spec: dict[str, Any] | None = None):
        self.ops, self.spec = ops, spec or {}

    def __call__(self, x: Tensor) -> Tensor:
        for _, op in self.ops:
            x = op(x)
        return x

    @property
    def name(self) -> str:
        return str(self.spec.get("name", "unnamed"))

    def __repr__(self) -> str:
        return f"Compose({self.name}: " + " -> ".join(n for n, _ in self.ops) + ")"


def from_spec(spec: dict[str, Any]) -> Compose:
    """Build a bank from {name, description, ops: [op | {op: {params}}, ...]}. Strict about every name."""
    extra = set(spec) - {"name", "description", "ops"}
    if extra:
        raise KeyError(f"unknown key(s) in a transform bank: {sorted(extra)} (allowed: name, description, ops)")
    if not isinstance(spec.get("ops"), list):
        raise ValueError("a transform bank needs 'ops: [...]'")
    ops: list[tuple[str, Transform]] = []
    for item in spec["ops"]:
        if isinstance(item, str):
            name, params = item, {}
        elif isinstance(item, dict) and len(item) == 1:
            (name, params), = item.items()
            params = params or {}
        else:
            raise ValueError(f"each op is a name or a one-key mapping {{name: params}}, got {item!r}")
        if name not in _REGISTRY:
            hint = difflib.get_close_matches(name, _REGISTRY, n=1)
            raise KeyError(f"unknown transform '{name}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")
                           + f". Available: {available()}")
        try:
            ops.append((name, _REGISTRY[name](**params)))
        except TypeError as e:
            raise ValueError(f"transform '{name}': {e}") from None
        except ValueError as e:
            raise ValueError(f"transform '{name}': {e}") from None
    return Compose(ops, spec)


def load_bank(path: str | Path) -> dict[str, Any]:
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))     # -sig: tolerate a BOM (Notepad)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a transform bank is a mapping (name, ops)")
    raw.setdefault("name", Path(path).stem)
    return raw


def build(path_or_spec: str | Path | dict[str, Any]) -> Compose:
    """A bank from a YAML file or from an already loaded spec."""
    spec = path_or_spec if isinstance(path_or_spec, dict) else load_bank(path_or_spec)
    return from_spec(spec)
