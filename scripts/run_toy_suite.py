"""The three toy experiments over several seeds, then the table mean +- std. Works on any OS.

    python scripts/run_toy_suite.py                              # 3 seeds, 30 epochs
    python scripts/run_toy_suite.py --seeds 0 1 2 3 4 --epochs 40
    python scripts/run_toy_suite.py --skip-existing              # resume after an interruption

The seed changes the initialization AND the toy dataset (class shapes, backgrounds).
Each run goes to results/<name>/ (see scripts/smoke_toy.py); the table is printed at the end and can be
printed again with:  python -m eval.aggregate results
"""
import argparse
import subprocess
import sys
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have final_metrics.json")
    args = ap.parse_args()

    out_root = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    for seed in args.seeds:
        for prefix, extra in EXPERIMENTS:
            name = f"{prefix}_seed{seed}"
            if args.skip_existing and (out_root / name / "final_metrics.json").exists():
                print(f"skip {name} (already done)", flush=True)
                continue
            cmd = [sys.executable, str(ROOT / "scripts" / "smoke_toy.py"), "--epochs", str(args.epochs),
                   "--set", f"seed={seed}", "--name", name, "--out-root", str(out_root), "--overwrite", *extra]
            print("\n>>> " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
    print()
    aggregate(str(out_root))
