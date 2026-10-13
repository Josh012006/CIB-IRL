"""The three toy experiments (oracle nuisance family) over several seeds, then the table mean +- std.

    python scripts/run_toy_suite.py                              # 3 seeds, 30 epochs, one run at a time
    python scripts/run_toy_suite.py --seeds 0 1 2 3 4 --epochs 100 --cpu-count 3
    python scripts/run_toy_suite.py --skip-existing              # resume after an interruption

Fixed number of epochs. For runs until convergence and for other transformation banks use run_suite.py.
The seed changes the initialization AND the toy dataset. Each run goes to results/<name>/ (see run_experiment.py);
its console output goes to results/_logs/<name>.log. The table can be printed again with:
    python -m eval.aggregate results

--cpu-count N runs N independent runs at the same time, each in its own process, on any OS. Every process is
limited to (machine cores // N) torch threads, or --threads-per-run.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _parallel import Job, run_jobs, say          # noqa: E402
from eval.aggregate import main as aggregate      # noqa: E402

# (run name prefix, extra arguments of smoke_toy.py)
EXPERIMENTS = [
    ("toy", []),
    ("toy_spur0.9", ["--spurious", "0.9"]),
    ("toy_no_standardize", ["--config", "configs/toy.yml", "configs/ablations/no_standardize.yml"]),
]

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

    out_root = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    jobs = []
    for seed in args.seeds:
        for prefix, extra in EXPERIMENTS:
            name = f"{prefix}_seed{seed}"
            if args.skip_existing and (out_root / name / "final_metrics.json").exists():
                say(f"skip   {name} (already done)")
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "smoke_toy.py"), "--epochs", str(args.epochs),
                   "--set", f"seed={seed}", "--name", name, "--out-root", str(out_root), "--overwrite", *extra]
            jobs.append(Job(name, cmd, out_root / "_logs" / f"{name}.log"))

    failed = run_jobs(jobs, ROOT, args.cpu_count, args.threads_per_run)
    print()
    aggregate(str(out_root))
    if failed:
        say(f"\n{len(failed)} run(s) failed: {', '.join(failed)} (logs in {out_root / '_logs'})")
        sys.exit(1)
