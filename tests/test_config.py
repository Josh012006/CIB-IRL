"""Config loading: strictness, casting, merging, inheritance, and that every committed YAML is valid.
Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cibirl.train import TrainConfig   # noqa: E402


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_every_committed_config_loads():
    files = sorted((ROOT / "configs").rglob("*.yml"))
    assert files, "no config found"
    for f in files:
        TrainConfig.from_yaml(f)


def test_every_ablation_applies_on_top_of_toy():
    for f in sorted((ROOT / "configs" / "ablations").glob("*.yml")):
        cfg = TrainConfig.from_yaml(ROOT / "configs" / "toy.yml", f)
        assert cfg.num_slots == 5                      # the experiment config is kept
        assert cfg != TrainConfig.from_yaml(ROOT / "configs" / "toy.yml"), f"{f.name} changes nothing"


def test_defaults_when_no_file():
    assert TrainConfig.from_yaml() == TrainConfig()


def test_unknown_key_is_an_error_with_a_hint(tmp_path):
    p = write(tmp_path, "bad.yml", "gama: 0.5\n")
    with pytest.raises(KeyError, match="did you mean 'gamma'"):
        TrainConfig.from_yaml(p)


def test_yaml_scientific_notation_without_dot_is_cast_to_float(tmp_path):
    p = write(tmp_path, "sci.yml", "beta: 1e-3\nlr_policy: 3e-4\nnum_expert_episodes: 2e3\n")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.beta == 1e-3 and cfg.lr_policy == 3e-4 and cfg.num_expert_episodes == 2000
    assert isinstance(cfg.num_expert_episodes, int)


def test_wrong_types_are_rejected(tmp_path):
    for text in ("use_z: 1\n", "num_slots: 2.5\n", "gamma: true\n", "num_slots: null\n"):
        with pytest.raises(ValueError):
            TrainConfig.from_yaml(write(tmp_path, "t.yml", text))


def test_optional_field_accepts_null_and_zero(tmp_path):
    assert TrainConfig.from_yaml(write(tmp_path, "a.yml", "tau_plus: null\n")).tau_plus is None
    assert TrainConfig.from_yaml(write(tmp_path, "b.yml", "tau_plus: 0\n")).tau_plus == 0.0


def test_merge_order_and_overrides(tmp_path):
    a = write(tmp_path, "a.yml", "gamma: 0.5\nseed: 1\n")
    b = write(tmp_path, "b.yml", "gamma: 0.0\n")
    cfg = TrainConfig.from_yaml(a, b, seed=7)
    assert cfg.gamma == 0.0 and cfg.seed == 7          # later file wins, overrides win over files


def test_base_inheritance_and_cycle(tmp_path):
    write(tmp_path, "root.yml", "num_slots: 8\nbeta: 0.01\n")
    child = write(tmp_path, "child.yml", "base: root.yml\nbeta: 0.5\n")
    cfg = TrainConfig.from_yaml(child)
    assert cfg.num_slots == 8 and cfg.beta == 0.5
    write(tmp_path, "x.yml", "base: y.yml\n")
    y = write(tmp_path, "y.yml", "base: x.yml\n")
    with pytest.raises(ValueError, match="circular"):
        TrainConfig.from_yaml(y)


def test_to_yaml_round_trip(tmp_path):
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "toy.yml", seed=5)
    out = tmp_path / "resolved.yml"
    cfg.to_yaml(out)
    assert TrainConfig.from_yaml(out) == cfg


def test_yaml_with_accents_and_bom_is_read_as_utf8(tmp_path):
    """Windows editors (Notepad) save UTF-8 with a BOM, and French comments contain accents."""
    p = tmp_path / "fr.yml"
    p.write_bytes("﻿# Réglages du test — été\nnum_slots: 7\n".encode("utf-8"))
    assert TrainConfig.from_yaml(p).num_slots == 7
