"""Baselines 1 and 2 (K-Means / GMM on features, SimCLR, IIC), the shared evaluation protocol, the runner, the suite of the
comparison and the comparison table. All on tiny runs. Run: python -m pytest tests -q"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_methods                             # noqa: E402
import datasets                                    # noqa: E402
import run_baseline                                # noqa: E402
import run_suite                                   # noqa: E402
from baselines import BaselineConfig, build_clusterer      # noqa: E402
from baselines.contrastive import iic_loss, nt_xent        # noqa: E402
from cibirl import transforms                              # noqa: E402
from cibirl.train import TrainConfig                       # noqa: E402
from eval import aggregate                                 # noqa: E402
from eval.protocol import TEST_MODES, delta_acc, is_right, partition_metrics   # noqa: E402

BANK = ROOT / "configs" / "transforms" / "toy_oracle.yml"


# ---- the shared protocol ------------------------------------------------------------------------------
def test_partition_metrics_of_a_perfect_and_of_a_nuisance_partition():
    y = np.repeat(np.arange(3), 20)
    bg = np.tile(np.arange(2), 30)
    m = partition_metrics(y.copy(), y, bg, None, 3)
    assert m["acc"] == 1 and m["clusters"] == 3 and m["clusters_ge10"] == 3 and m["nmi_class"] == pytest.approx(1)
    assert m["nmi_background"] < 0.05 and m["worst_group_acc"] == 1 and m["open_share"] == 0
    assert is_right(m, 3) and not is_right(m, 4)
    byb = partition_metrics(bg.copy(), y, bg, None, 3)                                 # clusters = background
    assert byb["nmi_background"] == pytest.approx(1) and byb["acc"] < 0.5 and not is_right(byb, 3)
    assert partition_metrics(y, y, None, None, 3)["nmi_background"] is None            # a dataset without a nuisance id


def test_open_share_counts_images_outside_the_kept_slots():
    y = np.repeat(np.arange(2), 20)
    pred = np.where(np.arange(40) < 30, y, 5)
    m = partition_metrics(pred, y, None, {0, 1}, 6)
    assert m["open_share"] == pytest.approx(10 / 40)


def test_delta_acc_and_modes():
    assert TEST_MODES == ("iid", "aligned", "inverted")
    assert delta_acc({"aligned": {"acc": 0.9}, "inverted": {"acc": 0.4}}) == pytest.approx(0.5)


def test_eval_heldout_uses_the_shared_protocol():
    import eval_heldout
    assert eval_heldout.partition_metrics is partition_metrics and eval_heldout.MODES == TEST_MODES


# ---- config -------------------------------------------------------------------------------------------
def test_baseline_config_is_strict(tmp_path):
    assert BaselineConfig.from_dict({"method": "simclr", "epochs": "7"}).epochs == 7
    with pytest.raises(KeyError, match="did you mean 'num_clusters'"):
        BaselineConfig.from_dict({"num_cluster": 3})
    with pytest.raises(ValueError, match="method must be"):
        BaselineConfig(method="dbscan")
    with pytest.raises(ValueError, match="features must be"):
        BaselineConfig(features="vit")
    with pytest.raises(ValueError, match="num_clusters"):
        BaselineConfig(num_clusters=1)
    p = tmp_path / "c.yml"
    BaselineConfig(method="iic", seed=4).to_yaml(p)
    assert BaselineConfig.from_yaml(p, seed=9).seed == 9 and BaselineConfig.from_yaml(p).method == "iic"


# ---- the losses ---------------------------------------------------------------------------------------
def test_nt_xent_prefers_matching_views():
    z = torch.randn(32, 16)
    assert nt_xent(z, z + 0.01 * torch.randn_like(z), 0.5) < nt_xent(z, torch.randn(32, 16), 0.5)
    assert math.isfinite(float(nt_xent(z, z, 0.5)))


def test_iic_loss_is_minus_log_k_for_balanced_deterministic_clusters_and_zero_for_independent_ones():
    K = 4
    p = torch.eye(K).repeat(10, 1)
    assert float(iic_loss(p, p)) == pytest.approx(-math.log(K), abs=1e-5)             # I = log K, the maximum
    uniform = torch.full((40, K), 1 / K)
    assert float(iic_loss(uniform, uniform)) == pytest.approx(0, abs=1e-5)             # no information
    shuffled = p[torch.randperm(40)]
    assert float(iic_loss(p, shuffled)) > float(iic_loss(p, p))


# ---- the clusterers -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy():
    return datasets.build("toy", 0, n_per_class=40, source="generate"), datasets.build_test("toy", 0, "aligned", n_per_class=40, source="generate")


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_feature_baselines_fit_the_train_images_and_predict_new_ones(toy, method):
    train, test = toy
    c = build_clusterer(BaselineConfig(method=method, num_clusters=5, seed=0)).fit(train.images)
    p_train, p_test = c.predict(train.images), c.predict(test.images)
    assert p_train.shape == (200,) and p_test.shape == (200,) and p_train.dtype == np.int64
    assert 1 < len(np.unique(p_train)) <= 5
    assert (c.predict(train.images) == p_train).all()                                  # deterministic once fit


def test_pixel_kmeans_is_not_given_labels_or_a_bank(toy):
    train, _ = toy
    c = build_clusterer(BaselineConfig(method="kmeans", num_clusters=3)).fit(train.images, bank=None)
    assert c.history == []


@pytest.mark.parametrize("method", ["simclr", "iic"])
def test_contrastive_baselines_train_with_the_bank_and_predict(toy, method):
    train, test = toy
    bank = transforms.build(BANK)
    cfg = BaselineConfig(method=method, num_clusters=4, epochs=3, batch_size=64, embed_dim=16, proj_dim=8, in_channels=3, image_size=16)
    c = build_clusterer(cfg, bank).fit(train.images)
    assert len(c.history) == 3 and all(math.isfinite(h["loss"]) for h in c.history)
    p = c.predict(test.images)
    assert p.shape == (200,) and p.min() >= 0 and p.max() < 4
    assert c.embed(train.images).shape == (200, 16)


def test_contrastive_baselines_need_a_bank_and_the_right_method():
    cfg = BaselineConfig(method="simclr")
    with pytest.raises(ValueError, match="needs a transformation bank"):
        build_clusterer(cfg, None)
    from baselines import ContrastiveClusterer
    with pytest.raises(ValueError, match="simclr and iic"):
        ContrastiveClusterer(BaselineConfig(method="kmeans"), transforms.build(BANK))


def test_contrastive_training_is_seeded(toy):
    train, _ = toy
    bank = transforms.build(BANK)
    cfg = BaselineConfig(method="iic", num_clusters=4, epochs=1, batch_size=64, embed_dim=8, seed=2)
    torch.manual_seed(0)
    a = build_clusterer(cfg, bank).fit(train.images).embed(train.images)
    torch.manual_seed(0)
    b = build_clusterer(cfg, bank).fit(train.images).embed(train.images)
    assert np.allclose(a, b)                                                           # same init (cfg.seed) and same draws


def test_resnet_features_shape_without_downloading_weights():
    pytest.importorskip("torchvision")
    from baselines.features import make_extractor
    f = make_extractor("resnet18", pretrained=False)(torch.rand(3, 1, 28, 28))        # one-channel images are repeated; resized to 224
    assert f.shape == (3, 512)


# ---- the runner ---------------------------------------------------------------------------------------
def run(tmp_path, *args):
    return run_baseline.main(["--out-root", str(tmp_path), "--data", "toy", "--data-set", "n_per_class=30", *args])


def test_run_baseline_writes_everything_the_comparison_reads(tmp_path):
    held = run(tmp_path, "--method", "kmeans", "--set", "num_clusters=5", "seed=1", "--name", "km_seed1")
    out = tmp_path / "km_seed1"
    for f in ("config.yml", "run.yml", "final_metrics.json", "heldout.json", "preds.npy"):
        assert (out / f).exists(), f
    assert set(held) >= {"run", "suite", "seed", "method", "train", *TEST_MODES}
    assert json.loads((out / "heldout.json").read_text(encoding="utf-8"))["iid"]["acc"] == held["iid"]["acc"]
    meta = yaml.safe_load((out / "run.yml").read_text(encoding="utf-8"))
    assert meta["uses_labels_for_training"] is False and meta["data"] == "toy" and meta["data_params"]["n_per_class"] == 30
    assert np.load(out / "preds.npy").shape == (150,)
    groups = aggregate.collect(tmp_path)                                               # the same folder works with eval.aggregate
    assert list(groups) == ["km"] and groups["km"][0]["acc"] == pytest.approx(held["train"]["acc"])
    with pytest.raises(SystemExit, match="already exists"):
        run(tmp_path, "--method", "kmeans", "--name", "km_seed1")
    run(tmp_path, "--method", "kmeans", "--name", "km_seed1", "--overwrite", "--no-test")
    assert "iid" not in json.loads((out / "heldout.json").read_text(encoding="utf-8"))


def test_run_baseline_simclr_and_iic_end_to_end(tmp_path):
    for m in ("simclr", "iic"):
        held = run(tmp_path, "--method", m, "--transform", "toy_oracle", "--set", "num_clusters=4", "epochs=2", "batch_size=64",
                   "embed_dim=16", "--name", f"{m}_seed0")
        assert held["train"]["clusters"] <= 4 and (tmp_path / f"{m}_seed0" / "history.jsonl").exists()
        final = json.loads((tmp_path / f"{m}_seed0" / "final_metrics.json").read_text(encoding="utf-8"))
        assert final["epochs_run"] == 2 and final["transform"] == "toy_oracle"
    with pytest.raises(SystemExit, match="needs --transform"):
        run(tmp_path, "--method", "simclr", "--name", "x")


def test_run_baseline_spurious_shortcut_reaches_the_train_images_only(tmp_path):
    clean = run(tmp_path, "--method", "kmeans", "--set", "num_clusters=5", "--name", "c")
    spur = run(tmp_path, "--method", "kmeans", "--set", "num_clusters=5", "--data-set", "spurious=0.9", "--name", "s")
    assert yaml.safe_load((tmp_path / "s" / "run.yml").read_text(encoding="utf-8"))["data_params"]["spurious"] == 0.9
    assert clean["train"]["nmi_background"] != spur["train"]["nmi_background"]         # the train images changed ...
    assert clean["iid"]["acc"] >= 0 and spur["iid"]["acc"] >= 0                        # ... the test images are scored the same way


def test_run_baseline_rejects_unknown_benchmark_and_keys(tmp_path):
    with pytest.raises(SystemExit, match="unknown benchmark"):
        run_baseline.main(["--method", "kmeans", "--data", "mnist", "--out-root", str(tmp_path)])
    with pytest.raises(KeyError, match="unknown baseline config key"):
        run(tmp_path, "--method", "kmeans", "--set", "clusters=3")


# ---- baseline 3 = our trainer without the bottleneck ------------------------------------------------------
def test_baseline_3_changes_only_beta_and_lambda():
    base = TrainConfig.from_yaml(ROOT / "configs" / "benchmarks" / "toy.yml")
    noib = TrainConfig.from_yaml(ROOT / "configs" / "benchmarks" / "toy.yml", ROOT / "configs" / "baselines" / "irl_noib.yml")
    diff = {k for k in vars(base) if getattr(base, k) != getattr(noib, k)}
    assert diff == {"beta", "lam"} and (noib.beta, noib.lam) == (0.0, 0.0) and base.beta > 0


# ---- the comparison suite and the table --------------------------------------------------------------------
def test_comparison_suite_covers_four_methods_in_two_regimes():
    suite = run_suite.load_suite(ROOT / "configs" / "suites" / "toy_compare.yml")
    exps = {e["name"]: e for e in suite["experiments"]}
    assert len(exps) == 20 and len(suite["seeds"]) == 200
    for regime, s in (("clean", 0.0), ("spur90", 0.9)):
        for stem in ("kmeans_k5", "kmeans_k10", "gmm_k5", "gmm_k10", "simclr_k5", "simclr_k10", "iic_k5", "iic_k10", "irl_noib", "final"):
            assert exps[f"{stem}_{regime}"]["data_params"]["spurious"] == s
    assert exps["final_clean"]["configs"] == ["configs/benchmarks/toy.yml"]
    assert exps["irl_noib_spur90"]["configs"][-1] == "configs/baselines/irl_noib.yml"
    assert exps["simclr_k10_clean"]["transform"] == "toy_oracle" and "transform" not in exps["kmeans_k5_clean"]


def test_comparison_suite_dry_run_builds_the_right_commands():
    res = run_suite_dry("configs/suites/toy_compare.yml", "--only", "kmeans_k5_clean", "iic_k10_spur90", "irl_noib_clean", "--seeds", "0", "1")
    assert "6 runs would be launched" in res.stdout, res.stdout + res.stderr
    assert "run_baseline.py --method kmeans" in res.stdout and "run_baseline.py --method iic" in res.stdout
    assert "run_experiment.py" in res.stdout and "configs/baselines/irl_noib.yml" in res.stdout


def run_suite_dry(*args):
    import subprocess
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_suite.py"), *args, "--dry-run"], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT)


def test_suite_rejects_a_bank_less_contrastive_baseline_and_unknown_method(tmp_path):
    p = tmp_path / "s.yml"
    p.write_text("experiments:\n  - {name: a, method: simclr}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="needs a 'name' and a 'transform'"):
        run_suite.load_suite(p)
    p.write_text("experiments:\n  - {name: a, method: dbscan, transform: generic}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown method"):
        run_suite.load_suite(p)
    p.write_text("experiments:\n  - {name: a, method: kmeans}\n", encoding="utf-8")                       # kmeans needs no bank
    assert run_suite.load_suite(p)["experiments"][0]["method"] == "kmeans"


def fake_row(run_name, accs, **extra):
    row = {"run": run_name}
    for part, (acc, clusters_ge10) in accs.items():
        row[part] = {"acc": acc, "clusters": clusters_ge10, "clusters_ge10": clusters_ge10, "nmi_background": 0.01}
    return {**row, **extra}


def test_compare_methods_table(tmp_path):
    full = {"train": (1.0, 5), "iid": (1.0, 5), "aligned": (1.0, 5), "inverted": (0.6, 7)}
    rows = [fake_row("ours_seed0", full), fake_row("ours_seed1", {**full, "inverted": (1.0, 5)}),
            fake_row("km_seed0", {"train": (0.5, 5), "iid": (0.5, 5), "aligned": (0.7, 5), "inverted": (0.3, 5)})]
    f = tmp_path / "ours.json"
    f.write_text(json.dumps(rows[:2]), encoding="utf-8")
    d = tmp_path / "base" / "km_seed0"
    d.mkdir(parents=True)
    (d / "heldout.json").write_text(json.dumps(rows[2]), encoding="utf-8")
    got = compare_methods.summarize(compare_methods.collect_rows([f, tmp_path / "base"]))
    by = {s["method"]: s for s in got}
    assert set(by) == {"ours", "km"} and by["ours"]["runs"] == 2
    assert by["ours"]["train"]["right"] == 2 and by["ours"]["inverted"]["right"] == 1           # 7 clusters is not right
    assert by["ours"]["delta_acc"]["mean"] == pytest.approx(0.2) and by["km"]["delta_acc"]["mean"] == pytest.approx(0.4)
    table = compare_methods.to_markdown(got)
    lines = table.strip().splitlines()
    assert len({l.count("|") for l in lines}) == 1 and lines[2].startswith("| `km`")        # a regular table, sorted by method
    out = tmp_path / "t.csv"
    compare_methods.to_csv(got, out)
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("method,runs,train_right")
    with pytest.raises(SystemExit, match="no such file"):
        compare_methods.collect_rows([tmp_path / "nothing"])


def test_compare_methods_handles_rows_without_all_parts():
    rows = [fake_row("a_seed0", {"train": (1.0, 5)})]
    s = compare_methods.summarize(rows)[0]
    assert "delta_acc" not in s and compare_methods.to_markdown([s]).count("-") > 0


def test_fewer_clusters_than_classes_is_scored_not_a_crash():
    y = np.repeat(np.arange(5), 20)
    m = partition_metrics(y % 3, y, np.zeros_like(y), None, 3)                         # 3 clusters for 5 classes (K = 3 baselines)
    assert m["clusters"] == 3 and 0 < m["acc"] < 1 and m["worst_group_acc"] == 0     # a class without a cluster has accuracy 0
    assert not is_right(m, 5)


# ---- our model and baseline 3 through the same held-out evaluation, then the table ----------------------------
def tiny_cibirl_run(out_root, experiment, extra_configs=()):
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--data", "toy", "--data-set", "n_per_class=40", "spurious=0.0",
           "--transform", "toy_oracle", "--config", "configs/benchmarks/toy.yml", *extra_configs, "--stop-set", "max_epochs=3", "min_epochs=1",
           "--set", "reward_warmup_iters=3", "num_expert_episodes=100", "seed=0", "--name", f"{experiment}_seed0", "--out-root", str(out_root)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert res.returncode == 0, res.stderr[-2000:]


def test_ours_and_baseline_3_and_baseline_1_end_to_end_into_one_table(tmp_path):
    import eval_heldout
    suite = tmp_path / "suite"
    tiny_cibirl_run(suite, "final")
    tiny_cibirl_run(suite, "irl_noib", ["configs/baselines/irl_noib.yml"])
    assert yaml.safe_load((suite / "irl_noib_seed0" / "config.yml").read_text(encoding="utf-8"))["beta"] == 0.0     # the overlay was applied
    for exp in ("final", "irl_noib"):
        eval_heldout.main([str(suite), "--experiment", exp, "--out", str(tmp_path / f"held_{exp}"), "--workers", "1"])
    rows = json.loads((tmp_path / "held_irl_noib" / "heldout_metrics.json").read_text(encoding="utf-8"))
    assert rows[0]["run"] == "irl_noib_seed0" and rows[0]["train_reproduces_preds"] and set(rows[0]) >= {"train", *TEST_MODES}
    run_baseline.main(["--method", "kmeans", "--data", "toy", "--data-set", "n_per_class=40", "spurious=0.0", "--set", "num_clusters=5",
                       "--out-root", str(tmp_path / "base"), "--name", "kmeans_k5_clean_seed0"])
    got = {s["method"]: s for s in compare_methods.summarize(compare_methods.collect_rows(
        [tmp_path / "held_final" / "heldout_metrics.json", tmp_path / "held_irl_noib" / "heldout_metrics.json", tmp_path / "base"]))}
    assert set(got) == {"final", "irl_noib", "kmeans_k5_clean"}
    assert all(set(s["parts"]) == {"train", *TEST_MODES} and "delta_acc" in s for s in got.values())             # the same columns for every method
