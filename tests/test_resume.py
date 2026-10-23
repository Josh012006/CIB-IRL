"""A run stopped and continued with --resume is the same run as one that was not stopped."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--data", "toy", "--transform", "toy_oracle", "--config", "configs/benchmarks/toy.yml",
        "--set", "reward_warmup_iters=3", "num_expert_episodes=100", "seed=0", "--device", "cpu", "--snapshot-every", "0"]
KEYS = ("epoch", "iter", "L_psi", "L_pos", "L_neg", "twin_acc", "KL", "n_clusters", "eval_acc")


def run(tmp_path, name, stop, *extra):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), *BASE, "--out-root", str(tmp_path), "--name", name,
           "--stop-set", f"max_epochs={stop}", "min_epochs=1000", *extra]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, (res.stdout + res.stderr)[-3000:]


def rows(tmp_path, name):
    lines = (tmp_path / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [{k: json.loads(l)[k] for k in KEYS} for l in lines]


def test_resume_is_the_same_run(tmp_path):
    run(tmp_path, "straight", 4, "--ckpt-every", "1000")
    run(tmp_path, "cut", 2, "--ckpt-every", "1000")
    assert (tmp_path / "cut" / "resume_state.pt").exists()
    with open(tmp_path / "cut" / "metrics.jsonl", "a", encoding="utf-8") as f:       # epochs written after the last state (a crash)
        f.write('{"epoch": 2, "iter": 1, "junk": true}\n')
    run(tmp_path, "cut", 4, "--ckpt-every", "1000", "--resume")
    assert rows(tmp_path, "cut") == rows(tmp_path, "straight")
    assert (tmp_path / "cut" / "at_ep2" / "final_metrics.json").exists()      # the first stop is kept
    final = json.loads((tmp_path / "cut" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["epochs_run"] == 4


def test_resume_without_state_fails(tmp_path):
    run(tmp_path, "nostate", 1)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), *BASE, "--out-root", str(tmp_path), "--name", "nostate",
           "--stop-set", "max_epochs=2", "--resume"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0 and "resume_state.pt" in (res.stdout + res.stderr)
