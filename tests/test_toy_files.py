"""The toy dataset as files (data/toy/): exact round trip, the loader, the committed files and `source` of the builder.
Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets                                    # noqa: E402
import make_toy_dataset                            # noqa: E402
import toy_data                                    # noqa: E402


@pytest.fixture(scope="module")
def toy_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    toy_data.save_toy(3, d)
    return d


def test_file_holds_exactly_what_the_generator_makes(toy_dir):
    x, y, bg = toy_data.load_toy(3, "train", toy_dir)
    gx, gy, gbg = toy_data.make_toy(seed=3, spurious=0.0, **toy_data.TOY_DEFAULTS)
    assert x.dtype == torch.float32 and torch.allclose(x, gx, atol=1e-6)
    assert torch.equal(y, gy) and torch.equal(bg, gbg)
    for mode in toy_data.TEST_MODES:
        x, y, bg = toy_data.load_toy(3, f"test_{mode}", toy_dir)
        gx, gy, gbg = toy_data.make_toy_test(3, 3, mode, **toy_data.TOY_DEFAULTS)
        assert torch.allclose(x, gx, atol=1e-6) and torch.equal(y, gy) and torch.equal(bg, gbg)
    x, _, _ = toy_data.load_toy(3, "train_s90", toy_dir)
    assert torch.allclose(x, toy_data.make_toy(seed=3, spurious=0.9, **toy_data.TOY_DEFAULTS)[0], atol=1e-6)


def test_loader_reports_missing_file_and_missing_split(toy_dir, tmp_path):
    assert toy_data.load_toy(3, "train", tmp_path) is None                          # no file: None, the caller decides
    with pytest.raises(KeyError, match="no split 'train_s50'"):
        toy_data.load_toy(3, "train_s50", toy_dir)
    assert toy_data.split_name(0.0) == "train" and toy_data.split_name(0.9) == "train_s90"


def test_content_hash_ignores_the_zip_but_not_the_arrays(toy_dir, tmp_path):
    toy_data.save_toy(3, tmp_path)
    assert toy_data.content_sha256(toy_data.toy_path(3, tmp_path)) == toy_data.content_sha256(toy_data.toy_path(3, toy_dir))
    toy_data.save_toy(4, tmp_path)
    assert toy_data.content_sha256(toy_data.toy_path(4, tmp_path)) != toy_data.content_sha256(toy_data.toy_path(3, tmp_path))


def test_parse_seeds_and_verify_detects_a_change(toy_dir, tmp_path, capsys):
    assert make_toy_dataset.parse_seeds(["0-2", "5", "7-8,10"]) == [0, 1, 2, 5, 7, 8, 10]
    assert make_toy_dataset.verify([3], toy_dir, (0.9,)) == 0
    assert make_toy_dataset.verify([99], toy_dir, (0.9,)) == 1                      # missing file
    toy_data.save_toy(3, tmp_path)
    arrays = dict(np.load(toy_data.toy_path(3, tmp_path)))
    arrays["train_x"] = arrays["train_x"] + 0.01                                     # tamper with one array
    np.savez_compressed(toy_data.toy_path(3, tmp_path), **arrays)
    assert make_toy_dataset.verify([3], tmp_path, (0.9,)) == 1
    assert "train_x" in capsys.readouterr().out


def test_cli_writes_files_and_manifest(tmp_path):
    make_toy_dataset.main(["--seeds", "0-1", "--out", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seeds"] == [0, 1] and set(manifest["content_sha256"]) == {"seed000.npz", "seed001.npz"}
    assert manifest["content_sha256"]["seed001.npz"] == toy_data.content_sha256(tmp_path / "seed001.npz")
    assert make_toy_dataset.verify([0, 1], tmp_path, (0.9,)) == 0


# ---- the builder: `source` ------------------------------------------------------------------------------
@pytest.fixture
def toy_root(toy_dir, monkeypatch):
    monkeypatch.setattr(toy_data, "TOY_DIR", toy_dir)
    return toy_dir


def test_builder_reads_the_file_or_generates_the_same_images(toy_root):
    f = datasets.build("toy", 3, source="file")
    g = datasets.build("toy", 3, source="generate")
    a = datasets.build("toy", 3)                                                     # auto
    assert (f.source, g.source, a.source) == ("file", "generated", "file")
    assert torch.allclose(f.images, g.images, atol=1e-6) and torch.equal(f.labels, g.labels)
    fs = datasets.build("toy", 3, spurious=0.9, source="file")
    assert torch.allclose(fs.images, datasets.build("toy", 3, spurious=0.9, source="generate").images, atol=1e-6)
    for mode in datasets.TEST_MODES:
        t = datasets.build_test("toy", 3, mode, source="file")
        assert t.source == "file" and torch.allclose(t.images, datasets.build_test("toy", 3, mode, source="generate").images, atol=1e-6)


def test_builder_falls_back_to_the_generator_and_source_file_is_strict(toy_root):
    b = datasets.build("toy", 8)                                                     # no file for seed 8
    assert b.source == "generated" and b.images.shape == (1500, 3, 16, 16)
    with pytest.raises(FileNotFoundError, match="make_toy_dataset.py --seeds 8"):
        datasets.build("toy", 8, source="file")
    with pytest.raises(FileNotFoundError, match="make_toy_dataset.py --seeds 8"):
        datasets.build_test("toy", 8, "iid", source="file")
    with pytest.raises(FileNotFoundError, match="--spurious 0.5"):
        datasets.build("toy", 3, spurious=0.5, source="file")                        # file exists, split does not
    assert datasets.build("toy", 3, spurious=0.5).source == "generated"              # auto: generate
    assert datasets.build("toy", 3, n_per_class=20).source == "generated"            # another configuration is never read from a file
    assert datasets.build("toy", 3, n_per_class=20).images.shape[0] == 100
    with pytest.raises(ValueError, match="source must be"):
        datasets.build("toy", 3, source="disk")


# ---- the files that are committed ----------------------------------------------------------------------
COMMITTED = sorted((ROOT / "data" / "toy").glob("seed*.npz"))


def test_committed_manifest_matches_the_committed_files():
    manifest_path = ROOT / "data" / "toy" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("data/toy is not generated (python scripts/make_toy_dataset.py)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seeds"] == sorted(int(p.name[4:7]) for p in COMMITTED)
    for p in COMMITTED:
        assert toy_data.content_sha256(p) == manifest["content_sha256"][p.name], p.name


@pytest.mark.skipif(not COMMITTED, reason="data/toy is not generated")
def test_committed_files_equal_the_generator():
    assert make_toy_dataset.verify([int(COMMITTED[0].name[4:7]), int(COMMITTED[-1].name[4:7])], ROOT / "data" / "toy", (0.9,)) == 0
