"""The command line, end to end, in a fresh process (this is what runs on Windows, Linux and macOS in CI)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_toy_writes_everything(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "smoke_toy.py"), "--epochs", "1", "--out-root", str(tmp_path),
           "--name", "t", "--set", "reward_warmup_iters=3", "num_expert_episodes=100"]
    res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr[-2000:]
    run = tmp_path / "t"
    for f in ("config.yml", "metrics.jsonl", "final_metrics.json", "preds.npy", "intents.npy", "checkpoint.pt",
              "figures/curves.png", "figures/clusters.png", "figures/confusion.png", "figures/embedding.png"):
        assert (run / f).exists() and (run / f).stat().st_size > 0, f
    final = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
    assert {"acc", "nmi_class", "nmi_background", "clusters_used"} <= set(final)


def test_existing_run_is_not_overwritten_silently(tmp_path):
    (tmp_path / "t").mkdir()
    cmd = [sys.executable, str(ROOT / "scripts" / "smoke_toy.py"), "--epochs", "1", "--out-root", str(tmp_path), "--name", "t"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0 and "already exists" in (res.stdout + res.stderr)


def test_suite_runner_help():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_toy_suite.py"), "--help"],
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0 and "--seeds" in res.stdout and "--cpu-count" in res.stdout


def test_suite_runs_in_parallel_and_resumes(tmp_path):
    suite = [sys.executable, str(ROOT / "scripts" / "run_toy_suite.py"), "--seeds", "0", "--epochs", "1",
             "--out-root", str(tmp_path), "--cpu-count", "2", "--skip-existing"]
    res = subprocess.run(suite, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, (res.stdout + res.stderr)[-2000:]
    for name in ("toy", "toy_spur0.9", "toy_no_standardize"):
        assert (tmp_path / f"{name}_seed0" / "final_metrics.json").exists(), name
        assert (tmp_path / "_logs" / f"{name}_seed0.log").exists(), name
    again = subprocess.run(suite, capture_output=True, text=True, encoding="utf-8")   # everything is done: nothing relaunched
    assert again.returncode == 0 and again.stdout.count("skip ") == 3


def test_suite_rejects_bad_cpu_count():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_toy_suite.py"), "--cpu-count", "0"],
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0 and "--cpu-count" in res.stderr
