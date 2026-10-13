"""The generalized runner: one run until convergence, the suite files, the parallel suite and the aggregation.
Run: python -m pytest tests -q"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets                                    # noqa: E402,F401
from cibirl import transforms                      # noqa: E402
from cibirl.train import TrainConfig               # noqa: E402
from eval import aggregate                         # noqa: E402

TINY = ["--data-set", "n_per_class=40", "--set", "reward_warmup_iters=3", "num_expert_episodes=100"]


def run(args, cwd=None):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", cwd=cwd)


# ---- the committed suites --------------------------------------------------------------------------
def suite_files():
    return sorted((ROOT / "configs" / "suites").glob("*.yml"))


@pytest.mark.parametrize("path", suite_files(), ids=lambda p: p.stem)
def test_every_committed_suite_is_valid(path):
    import run_suite
    suite = run_suite.load_suite(path)
    bank_dir = ROOT / "configs" / "transforms"
    for exp in suite["experiments"]:
        assert (bank_dir / f"{exp['transform']}.yml").exists(), exp["name"]
        transforms.build(bank_dir / f"{exp['transform']}.yml")
        assert exp.get("data", "toy") in datasets.DATASETS
        for c in exp.get("configs", []):
            TrainConfig.from_yaml(ROOT / c, **exp["set"])
        from cibirl.convergence import StopConfig
        StopConfig.from_dict(exp["stop"])


def test_suite_dry_run_lists_every_run():
    res = run([ROOT / "scripts" / "run_suite.py", "configs/suites/toy_regimes.yml", "--seeds", "0", "1", "--dry-run"], cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "12 runs would be launched" in res.stdout                       # 6 experiments x 2 seeds
    assert "--transform informed_color" in res.stdout and "spurious=0.9" in res.stdout


def test_suite_merge_rules_and_errors(tmp_path):
    import run_suite
    p = tmp_path / "s.yml"
    p.write_text("common: {data: toy, set: {gamma: 0.5}, stop: {max_epochs: 9}}\n"
                 "experiments:\n  - {name: a, transform: generic, set: {beta: 0.1}}\n", encoding="utf-8")
    exp = run_suite.load_suite(p)["experiments"][0]
    assert exp["set"] == {"gamma": 0.5, "beta": 0.1} and exp["stop"] == {"max_epochs": 9} and exp["data"] == "toy"
    cmd = run_suite.command(exp, 3, "a_seed3", tmp_path, max_epochs=None)
    assert cmd[cmd.index("--set") + 1] == "seed=3" and "gamma=0.5" in cmd and "max_epochs=9" in cmd
    for text, msg in [("experiments: []\n", "non-empty"),
                      ("experiments:\n  - {name: a, transform: generic, seeds: 3}\n", "unknown key"),
                      ("experiments:\n  - {name: a}\n", "transform"),
                      ("experiments:\n  - {name: a, transform: g}\n  - {name: a, transform: g}\n", "duplicate"),
                      ("oops: 1\nexperiments:\n  - {name: a, transform: g}\n", "unknown key")]:
        p.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit, match=msg):
            run_suite.load_suite(p)


# ---- one run ---------------------------------------------------------------------------------------
def test_run_until_convergence_stops_early_and_records_it(tmp_path):
    # tolerances so loose that the rule fires at its earliest possible epoch: 2 windows + 1 = 3 epochs
    res = run([ROOT / "scripts" / "run_experiment.py", "--data", "toy", "--transform", "generic", "--name", "c",
               "--out-root", tmp_path, "--stop-set", "max_epochs=12", "min_epochs=3", "window=1", "part_tol=1.0",
               "twin_tol=1.0", "loss_tol=1e9", *TINY])
    assert res.returncode == 0, res.stderr[-2000:]
    run_dir = tmp_path / "c"
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["epochs_run"] == 3 and final["converged"] is True and final["stop_reason"] == "converged"
    assert final["transform"] == "generic" and {"acc", "nmi_class", "nmi_background", "clusters_used"} <= set(final)
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and {"part_change", "n_clusters"} <= set(rows[0])
    for f in ("config.yml", "transform.yml", "run.yml", "preds.npy", "intents.npy", "checkpoint.pt", "figures/curves.png",
              "figures/clusters.png", "figures/confusion.png", "figures/embedding.png"):
        assert (run_dir / f).exists() and (run_dir / f).stat().st_size > 0, f
    assert yaml.safe_load((run_dir / "transform.yml").read_text(encoding="utf-8"))["name"] == "generic"
    assert yaml.safe_load((run_dir / "run.yml").read_text(encoding="utf-8"))["stop"]["window"] == 1


def test_run_hits_max_epochs_when_not_converged(tmp_path):
    res = run([ROOT / "scripts" / "run_experiment.py", "--data", "toy", "--transform", "toy_oracle", "--name", "m",
               "--out-root", tmp_path, "--stop-set", "max_epochs=2", "min_epochs=50", *TINY])
    assert res.returncode == 0, res.stderr[-2000:]
    final = json.loads((tmp_path / "m" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["epochs_run"] == 2 and final["converged"] is False and final["stop_reason"] == "max_epochs"


def test_fixed_epochs_disables_the_convergence_test(tmp_path):
    res = run([ROOT / "scripts" / "run_experiment.py", "--data", "toy", "--transform", "generic", "--name", "f",
               "--out-root", tmp_path, "--fixed-epochs", "2", *TINY])
    assert res.returncode == 0, res.stderr[-2000:]
    final = json.loads((tmp_path / "f" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["epochs_run"] == 2 and final["converged"] is False and final["stop_reason"] == "fixed_epochs"


def test_run_errors_are_explicit(tmp_path):
    base = [ROOT / "scripts" / "run_experiment.py", "--out-root", tmp_path]
    assert "unknown transform bank" in (run([*base, "--transform", "nope"]).stderr)
    assert "unknown dataset" in (run([*base, "--transform", "generic", "--data", "nope"]).stderr)
    res = run([*base, "--transform", "generic", "--set", "image_size=32"])
    assert res.returncode != 0 and "image_size" in res.stderr
    res = run([*base, "--transform", "generic", "--stop-set", "windw=3"])
    assert res.returncode != 0 and "did you mean 'window'" in res.stderr


# ---- parallel suite --------------------------------------------------------------------------------
def test_parallel_suite_with_banks_resumes_and_aggregates(tmp_path):
    suite = tmp_path / "tiny.yml"
    suite.write_text(yaml.safe_dump({
        "name": "tiny", "seeds": [0],
        "common": {"data": "toy", "data_params": {"n_per_class": 40}, "set": {"reward_warmup_iters": 3, "num_expert_episodes": 100},
                   "fixed_epochs": 1},
        "experiments": [{"name": "gen", "transform": "generic"}, {"name": "inf", "transform": "informed_color"}]}), encoding="utf-8")
    cmd = [ROOT / "scripts" / "run_suite.py", suite, "--out-root", tmp_path / "out", "--cpu-count", "2"]
    res = run(cmd)
    assert res.returncode == 0, (res.stdout + res.stderr)[-2000:]
    for name in ("gen_seed0", "inf_seed0"):
        assert (tmp_path / "out" / name / "final_metrics.json").exists()
        assert (tmp_path / "out" / "_logs" / f"{name}.log").exists()
    assert "gen" in res.stdout and "inf" in res.stdout and "converged" in res.stdout      # the table was printed
    again = run([*cmd, "--skip-existing"])
    assert again.returncode == 0 and again.stdout.count("skip ") == 2


def test_a_failing_run_is_reported_and_the_others_go_on(tmp_path):
    suite = tmp_path / "bad.yml"
    suite.write_text(yaml.safe_dump({
        "name": "bad", "seeds": [0], "common": {"data": "toy", "data_params": {"n_per_class": 40}, "fixed_epochs": 1,
                                                 "set": {"reward_warmup_iters": 3, "num_expert_episodes": 100}},
        "experiments": [{"name": "ok", "transform": "generic"}, {"name": "ko", "transform": "no_such_bank"}]}), encoding="utf-8")
    res = run([ROOT / "scripts" / "run_suite.py", suite, "--out-root", tmp_path / "out", "--cpu-count", "2"])
    assert res.returncode == 1 and "FAIL" in res.stdout and "ko_seed0" in res.stdout
    assert (tmp_path / "out" / "ok_seed0" / "final_metrics.json").exists()


# ---- aggregation -----------------------------------------------------------------------------------
def test_aggregate_handles_old_and_new_runs(tmp_path, capsys):
    def write(name, **metrics):
        d = tmp_path / name
        d.mkdir()
        (d / "final_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    write("a_seed0", acc=1.0, nmi_class=1.0, nmi_background=0.0, clusters_used=5, epochs=30)                  # legacy file
    write("a_seed1", acc=0.5, nmi_class=0.5, nmi_background=0.2, clusters_used=3, epochs_run=50, converged=True)
    write("b_seed0", acc=0.8, nmi_class=0.9, clusters_used=4, epochs_run=60, converged=False)                # no background
    aggregate.main(str(tmp_path))
    out = capsys.readouterr().out
    line_a = next(l for l in out.splitlines() if l.startswith("a "))
    line_b = next(l for l in out.splitlines() if l.startswith("b "))
    assert "0.750 +- 0.250" in line_a and "40.000 +- 10.000" in line_a and line_a.rstrip().endswith("1/1")
    assert "n/a" in line_b and line_b.rstrip().endswith("0/1")
