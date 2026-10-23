"""The transformation family: every operation, every shipped bank, and the property that matters
scientifically (the informed bank erases the hue, the generic bank does not). Run: python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets                                    # noqa: E402,F401  (registers the toy operation)
from cibirl import transforms as T                 # noqa: E402

BANKS = sorted((ROOT / "configs" / "transforms").glob("*.yml"))


def batch(n=16, c=3, size=16, seed=0):
    return torch.rand(n, c, size, size, generator=torch.Generator().manual_seed(seed))


def hue_angle(x):
    """Hue (radians) of the mean colour of each image, and its chroma."""
    r, g, b = x.mean((2, 3)).unbind(1)
    a, bb = r - (g + b) / 2, math.sqrt(3) / 2 * (g - b)
    return torch.atan2(bb, a), (a ** 2 + bb ** 2).sqrt()


def hue_preservation(bank, n=600):
    """mean cos(hue_out - hue_in) over solid-colour images: 1 = hue kept, 0 = hue erased."""
    torch.manual_seed(0)
    hue = torch.rand(n) * 2 * math.pi
    base = torch.stack([0.5 + 0.35 * torch.cos(hue), 0.5 + 0.35 * torch.cos(hue - 2 * math.pi / 3),
                        0.5 + 0.35 * torch.cos(hue + 2 * math.pi / 3)], 1)
    x = base.view(n, 3, 1, 1).expand(n, 3, 12, 12).contiguous()
    h_in, _ = hue_angle(x)
    h_out, chroma = hue_angle(T.build(bank)(x))
    keep = chroma > 1e-3                                           # grayscale views have no hue
    return torch.cos(h_out - h_in)[keep].mean().item()


# ---- operations ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", [
    ("random_resized_crop", {}), ("hflip", {}), ("color_jitter", {"brightness": .3, "contrast": .3, "saturation": .3}),
    ("grayscale", {"p": 1.0}), ("hue_rotate", {}), ("channel_permute", {})])
def test_every_op_keeps_shape_range_and_input(name, params):
    x = batch()
    before = x.clone()
    y = T.from_spec({"ops": [{name: params}]})(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert y.min() >= 0 and y.max() <= 1
    assert torch.equal(x, before)                                  # the input is not modified


def test_probability_zero_is_identity_and_one_is_applied():
    x = batch()
    for name in ("hflip", "grayscale", "hue_rotate", "channel_permute"):
        assert torch.equal(T.from_spec({"ops": [{name: {"p": 0.0}}]})(x), x), name
    assert torch.equal(T.from_spec({"ops": [{"hflip": {"p": 1.0}}]})(x), x.flip(-1))


def test_randomness_is_independent_per_sample():
    one = batch(1)
    x = one.expand(64, -1, -1, -1).contiguous()                    # 64 copies of the same image
    for name, at_least in (("random_resized_crop", 30), ("hue_rotate", 30), ("channel_permute", 5)):   # 6 permutations exist
        y = T.from_spec({"ops": [name]})(x)
        distinct = {tuple(v[:, :2, :2].flatten().tolist()) for v in y}
        assert len(distinct) >= at_least, f"{name}: the copies got the same view"


def test_full_crop_is_identity():
    x = batch()
    y = T.from_spec({"ops": [{"random_resized_crop": {"scale": [1, 1], "ratio": [1, 1]}}]})(x)
    assert torch.allclose(x, y, atol=1e-5)


def test_crop_stays_inside_the_image():
    x = torch.ones(32, 3, 16, 16)
    y = T.from_spec({"ops": [{"random_resized_crop": {"scale": [0.3, 0.6]}}]})(x)
    assert torch.allclose(y, x, atol=1e-5)                         # a constant image stays constant: no border artefact


def test_hue_matrix_properties():
    th = torch.tensor([0.0, 2 * math.pi / 3, 1.0])
    rot = T._hue_matrix(th)
    assert torch.allclose(rot[0], torch.eye(3), atol=1e-6)         # angle 0
    red = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(rot[1] @ red, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)    # 120 degrees: red -> green
    grey = torch.ones(3)
    assert torch.allclose(rot @ grey, grey.expand(3, 3), atol=1e-6)                  # greys are fixed
    assert torch.allclose(rot @ rot.transpose(1, 2), torch.eye(3).expand(3, 3, 3), atol=1e-5)   # it is a rotation


def test_channel_permute_keeps_the_pixel_values():
    x = batch()
    y = T.from_spec({"ops": ["channel_permute"]})(x)
    assert torch.allclose(x.sort(dim=1).values, y.sort(dim=1).values)


def test_colour_ops_refuse_non_rgb_data():
    x = batch(c=1)
    for op in ("hue_rotate", "grayscale", "channel_permute", {"color_jitter": {"saturation": 0.2}}):
        with pytest.raises(ValueError, match="RGB"):
            T.from_spec({"ops": [op]})(x)
    T.from_spec({"ops": [{"color_jitter": {"brightness": 0.2}}]})(x)   # brightness alone is fine on 1 channel


# ---- specs -----------------------------------------------------------------------------------------
def test_spec_errors_are_explicit():
    with pytest.raises(KeyError, match="did you mean 'hflip'"):
        T.from_spec({"ops": ["hflp"]})
    with pytest.raises(ValueError, match="hflip"):
        T.from_spec({"ops": [{"hflip": {"q": 0.5}}]})              # unknown parameter
    with pytest.raises(ValueError, match="probability|p must"):
        T.from_spec({"ops": [{"hflip": {"p": 1.5}}]})
    with pytest.raises(ValueError, match="low <= high"):
        T.from_spec({"ops": [{"random_resized_crop": {"scale": [1.0, 0.5]}}]})
    with pytest.raises(KeyError, match="unknown key"):
        T.from_spec({"ops": [], "opps": []})
    with pytest.raises(ValueError, match="ops"):
        T.from_spec({"name": "x"})


def test_bank_file_with_bom(tmp_path):
    p = tmp_path / "b.yml"
    p.write_bytes("name: accentué\nops:\n  - hflip: {p: 1.0}\n".encode("utf-8-sig"))
    bank = T.build(p)
    assert bank.name == "accentué" and len(bank.ops) == 1


# ---- shipped banks ---------------------------------------------------------------------------------
def test_there_are_banks():
    assert {"generic", "natural", "informed_color", "toy_oracle"} <= {p.stem for p in BANKS}


@pytest.mark.parametrize("path", BANKS, ids=lambda p: p.stem)
def test_every_shipped_bank_builds_and_runs(path):
    bank = T.build(path)
    assert bank.name == path.stem                                   # the name field matches the file name
    x = batch()
    y = bank(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_banks_differ_exactly_as_documented():
    ops = {p.stem: [n for n, _ in T.build(p).ops] for p in BANKS}
    assert ops["natural"] == ["random_resized_crop", "hflip", "color_jitter", "grayscale"]
    assert "hflip" not in ops["generic"] and "hue_rotate" not in ops["generic"]
    assert "hflip" not in ops["informed_color"] and "hue_rotate" in ops["informed_color"]
    assert sorted(n for n in ops["informed_color"] if n != "hue_rotate") == sorted(ops["generic"])


def test_informed_bank_erases_the_hue_and_generic_keeps_it():
    keep_generic = hue_preservation(ROOT / "configs" / "transforms" / "generic.yml")
    keep_informed = hue_preservation(ROOT / "configs" / "transforms" / "informed_color.yml")
    assert keep_generic > 0.8, keep_generic                         # a colour shortcut survives the generic bank
    assert abs(keep_informed) < 0.2, keep_informed                  # and is gone with the declared side information


def test_chain_through_the_orbit_sampler():
    from cibirl.experts import OrbitSampler
    sampler = OrbitSampler(T.build(ROOT / "configs" / "transforms" / "generic.yml"), d_max=2)
    x = batch(32)
    y = sampler(x)
    assert y.shape == x.shape and not torch.equal(x, y)


# ---- digit operations: random_affine, random_stroke_width -----------------------------------------------
def test_affine_defaults_are_the_identity_and_range_is_kept():
    x = batch()
    assert torch.allclose(T.from_spec({"ops": [{"random_affine": {}}]})(x), x, atol=1e-5)
    y = T.from_spec({"ops": [{"random_affine": {"shear_degrees": 15, "translate": 0.07, "scale": [0.9, 1.1]}}]})(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and not torch.allclose(y, x)


def test_affine_shears_a_vertical_bar():
    img = torch.zeros(1, 1, 28, 28)
    img[..., 6:22, 13:15] = 1.0                                    # a vertical stroke, centred
    for _ in range(20):
        y = T.from_spec({"ops": [{"random_affine": {"shear_degrees": 30}}]})(img)
        top, bottom = y[0, 0, 7].argmax().item(), y[0, 0, 21].argmax().item()
        assert abs(top - bottom) <= 10                             # the bar leans, it does not move away
    ys = [T.from_spec({"ops": [{"random_affine": {"shear_degrees": 30}}]})(img) for _ in range(30)]
    leans = [abs(v[0, 0, 7].argmax().item() - v[0, 0, 21].argmax().item()) for v in ys]
    assert max(leans) >= 3                                         # and it does lean for some draws


def test_stroke_width_thickens_or_thins_and_p_zero_is_identity():
    img = torch.zeros(200, 1, 28, 28)
    img[..., 10:18, 12:16] = 1.0
    y = T.from_spec({"ops": [{"random_stroke_width": {"max_strength": 1.0, "p": 1.0}}]})(img)
    mass = y.sum(dim=(1, 2, 3)) / img[0].sum()
    assert (mass > 1.01).any() and (mass < 0.99).any()             # both directions occur
    assert y.min() >= 0 and y.max() <= 1
    assert torch.equal(T.from_spec({"ops": [{"random_stroke_width": {"p": 0.0}}]})(img), img)
    with pytest.raises(ValueError):
        T.from_spec({"ops": [{"random_stroke_width": {"max_strength": 0.0}}]})


# ---- digit operations: elastic_distortion, gaussian_blur, gaussian_noise, random_erasing ----------------
def _bar():
    img = torch.zeros(64, 1, 28, 28)
    img[..., 6:22, 12:16] = 1.0
    return img


def test_elastic_moves_the_strokes_gently_and_p_zero_is_identity():
    img = _bar()
    y = T.from_spec({"ops": [{"elastic_distortion": {"sigma": 3.5, "strength": 1.2}}]})(img)
    assert y.shape == img.shape and y.min() >= 0 and y.max() <= 1 and torch.isfinite(y).all()
    assert not torch.allclose(y, img)
    mass = y.sum(dim=(1, 2, 3)) / img[0].sum()
    assert 0.8 < mass.mean() < 1.2 and mass.min() > 0.5            # the ink is moved, not created nor destroyed
    assert torch.equal(T.from_spec({"ops": [{"elastic_distortion": {"p": 0.0}}]})(img), img)
    with pytest.raises(ValueError):
        T.from_spec({"ops": [{"elastic_distortion": {"strength": 0.0}}]})


def test_blur_spreads_the_ink_without_changing_its_mass():
    img = _bar()
    y = T.from_spec({"ops": [{"gaussian_blur": {"sigma": [0.5, 0.5]}}]})(img)
    assert y.max() <= 1 and y.min() >= 0
    assert torch.allclose(y.sum(dim=(1, 2, 3)), img.sum(dim=(1, 2, 3)), rtol=1e-3)
    assert (y > 0).sum() > (img > 0).sum()                         # it spreads
    assert torch.equal(T.from_spec({"ops": [{"gaussian_blur": {"p": 0.0}}]})(img), img)
    with pytest.raises(ValueError):
        T.from_spec({"ops": [{"gaussian_blur": {"sigma": [0.0, 5.0]}}]})


def test_noise_is_small_clipped_and_p_zero_is_identity():
    img = _bar()
    y = T.from_spec({"ops": [{"gaussian_noise": {"max_std": 0.04}}]})(img)
    assert y.min() >= 0 and y.max() <= 1 and not torch.equal(y, img)
    assert (y - img).abs().mean() < 0.04
    assert torch.equal(T.from_spec({"ops": [{"gaussian_noise": {"p": 0.0}}]})(img), img)
    with pytest.raises(ValueError):
        T.from_spec({"ops": [{"gaussian_noise": {"max_std": 0.0}}]})


def test_erasing_blacks_out_exactly_one_square():
    img = torch.ones(200, 1, 28, 28)
    y = T.from_spec({"ops": [{"random_erasing": {"size": 4, "p": 1.0}}]})(img)
    assert torch.all((y == 0).sum(dim=(1, 2, 3)) == 16)            # one 4 x 4 square per image, fully inside the image
    z = T.from_spec({"ops": [{"random_erasing": {"size": 4, "p": 0.5}}]})(img)
    n_erased = ((z == 0).sum(dim=(1, 2, 3)) == 16).sum().item()
    assert 60 < n_erased < 140                                     # about half of the images
    assert torch.equal(T.from_spec({"ops": [{"random_erasing": {"p": 0.0}}]})(img), img)
    with pytest.raises(ValueError):
        T.from_spec({"ops": [{"random_erasing": {"size": 40}}]})(img)


def test_digits_zoomshift_stroke_elastic_bank_and_dmax_one():
    from cibirl.experts import OrbitSampler
    bank = T.build(ROOT / "configs" / "transforms" / "digits_zoomshift_stroke_elastic.yml")
    x = torch.rand(64, 1, 28, 28)
    for d_max in (1, 2):
        y = OrbitSampler(bank, d_max=d_max)(x)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and torch.isfinite(y).all()
    # d_max = 1: a view is the image itself or the image after exactly one application (about half untouched)
    y1 = OrbitSampler(bank, d_max=1)(x)
    untouched = (y1 == x).flatten(1).all(1).float().mean().item()
    assert 0.25 < untouched < 0.75


def test_digits_full_bank_builds_and_keeps_the_range():
    from cibirl.experts import OrbitSampler
    bank = T.build(ROOT / "configs" / "transforms" / "digits_full.yml")
    x = torch.rand(16, 1, 28, 28)
    y = OrbitSampler(bank, d_max=2)(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and torch.isfinite(y).all()
