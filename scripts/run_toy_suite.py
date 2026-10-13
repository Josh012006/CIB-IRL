"""The three toy experiments over several seeds, then the table mean +- std. Works on any OS.

    python scripts/run_toy_suite.py                              # 3 seeds, 30 epochs, one run at a time
    python scripts/run_toy_suite.py --seeds 0 1 2 3 4 --epochs 100 --cpu-count 3
    python scripts/run_toy_suite.py --skip-existing              # resume after an interruption

The seed changes the initialization AND the toy dataset (class shapes, backgrounds).
Each run goes to results/<name>/ (see scripts/smoke_toy.py); its console output goes to
results/_logs/<name>.log. The table is printed at the end and can be printed again with:
    python -m eval.aggregate results

Parallelism: --cpu-count N runs N independent runs at the same time, each in its own Python process
(subprocess), so it behaves the same on Windows, macOS and Linux. To avoid N processes fighting over
the same cores, every process is limited to (machine cores // N) torch threads, or --threads-per-run.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.aggregate import main as aggregate   # noqa: E402

# (run name prefix, extra arguments of smoke_toy.py)
EXPERIMENTS = [
    ("toy", []),
    ("toy_spur0.9", ["--spurious", "0.9"]),
    ("toy_no_standardize", ["--config", "configs/toy.yml", "configs/ablations/no_standardize.yml"]),
]

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_one(name: str, cmd: list[str], log_path: Path, env: dict) -> tuple[str, int, float]:
    """Run one smoke_toy.py in a subprocess, output to its own log file. Returns (name, exit code, seconds)."""
    t0 = time.time()
    say(f"start  {name}")
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    say(f"{'done ' if code == 0 else 'FAIL '}  {name}  ({dt / 60:.1f} min)" + ("" if code == 0 else f"  -> see {log_path}"))
    return name, code, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have final_metrics.json")
    ap.add_argument("--cpu-count", type=int, default=1, metavar="N", help="number of runs executed in parallel (default 1)")
    ap.add_argument("--threads-per-run", type=int, default=None,
                    help="torch threads of each run (default: machine cores // cpu-count, at least 1)")
    args = ap.parse_args()
    if args.cpu_count < 1:
        ap.error("--cpu-count must be >= 1")

    machine = os.cpu_count() or 1
    if args.cpu_count > machine:
        say(f"warning: --cpu-count {args.cpu_count} > {machine} cores on this machine")
    threads = args.threads_per_run or max(1, machine // args.cpu_count)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUTF8="1")

    out_root = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    (out_root / "_logs").mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in args.seeds:
        for prefix, extra in EXPERIMENTS:
            name = f"{prefix}_seed{seed}"
            if args.skip_existing and (out_root / name / "final_metrics.json").exists():
                say(f"skip   {name} (already done)")
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "smoke_toy.py"), "--epochs", str(args.epochs),
                   "--set", f"seed={seed}", "--name", name, "--out-root", str(out_root), "--overwrite", *extra]
            jobs.append((name, cmd, out_root / "_logs" / f"{name}.log"))

    say(f"{len(jobs)} runs, {args.cpu_count} in parallel, {threads} torch thread(s) each ({machine} cores)")
    t0 = time.time()
    failed = []
    pool = ThreadPoolExecutor(max_workers=args.cpu_count)   # threads only wait on subprocesses
    try:
        futures = [pool.submit(run_one, name, cmd, log, env) for name, cmd, log in jobs]
        for fut in as_completed(futures):
            name, code, _ = fut.result()
            if code != 0:
                failed.append(name)
    except KeyboardInterrupt:
        say("interrupted: pending runs cancelled, running ones stop with Ctrl-C (resume with --skip-existing)")
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)
    pool.shutdown()

    say(f"\nall runs finished in {(time.time() - t0) / 60:.1f} min")
    aggregate(str(out_root))
    if failed:
        say(f"\n{len(failed)} run(s) failed: {', '.join(failed)} (logs in {out_root / '_logs'})")
        sys.exit(1)
