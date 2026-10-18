"""The retained configurations (configs/benchmarks/) and the first-design configs of the other datasets.
Run: python -m pytest tests -q"""
import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets                                    # noqa: E402
import run_suite                                   # noqa: E402
from cibirl.train import TrainConfig               # noqa: E402

BENCH = ROOT / "configs" / "benchmarks"
FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def test_toy_benchmark_writes_every_field_explicitly():
    raw = yaml.safe_load((BENCH / "toy.yml").read_text(encoding="utf-8"))
    assert set(raw) <= FIELDS
    assert FIELDS - set(raw) <= {"seed"}, f"not written out: {sorted(FIELDS - set(raw))}"          # seed is set by the run


@pytest.mark.parametrize("folder", ["toy_cov_k10_final_ac", "toy_cov_k10_final_ac_val"])
def test_toy_benchmark_is_the_configuration_of_the_stored_runs(folder):
    runs = sorted((ROOT / "results" / folder).glob("final_seed*/config.yml"))
    if not runs:
        pytest.skip(f"results/{folder} is not in this checkout")
    ours = TrainConfig.from_yaml(BENCH / "toy.yml")
    for path in runs[:5] + runs[-5:]:
        stored = TrainConfig.from_yaml(path)
        diff = {f: (getattr(stored, f), getattr(ours, f)) for f in FIELDS - {"seed"} if getattr(stored, f) != getattr(ours, f)}
        assert not diff, f"{path.parent.name}: {diff}"


def test_toy_final_suite_uses_the_benchmark_config_and_the_documented_stop_rule():
    suite = run_suite.load_suite(ROOT / "configs" / "suites" / "toy_final.yml")
    exp = suite["experiments"][0]
    assert exp["configs"] == ["configs/benchmarks/toy.yml"] and exp["transform"] == "toy_oracle"
    assert exp["stop"] == {"max_epochs": 300, "min_epochs": 40} and exp["data_params"] == {"spurious": 0.0}
    assert suite["seeds"] == list(range(200))


@pytest.mark.parametrize("name", ["colored_mnist", "waterbirds", "toy", "mnist"])
def test_default_config_of_each_dataset_matches_its_data_shape(name):
    cfg = TrainConfig.from_yaml(ROOT / datasets.DEFAULT_CONFIG[name])
    shape = {"toy": (3, 16), "mnist": (1, 28), "colored_mnist": (3, 28), "waterbirds": (3, 64)}[name]
    assert (cfg.in_channels, cfg.image_size) == shape
    assert cfg.orbits_per_episode <= cfg.num_slots


def test_every_dataset_default_config_exists_and_every_benchmark_has_one():
    for name, path in datasets.DEFAULT_CONFIG.items():
        assert (ROOT / path).exists(), name
    assert set(datasets.BENCHMARKS) <= set(datasets.DEFAULT_CONFIG)


def test_baseline_3_overlay_is_a_partial_config():
    raw = yaml.safe_load((ROOT / "configs" / "baselines" / "irl_noib.yml").read_text(encoding="utf-8"))
    assert raw == {"beta": 0.0, "lam": 0.0}
