"""Toy experiment with the oracle nuisance family (exact background redraw), for a FIXED number of epochs.

    python scripts/smoke_toy.py                                       # random background
    python scripts/smoke_toy.py --spurious 0.9                        # background correlated with the class
    python scripts/smoke_toy.py --config configs/toy.yml configs/ablations/no_standardize.yml
    python scripts/smoke_toy.py --set seed=3 gamma=0 --epochs 40 --name my_run

Kept so that older commands still work. It is scripts/run_experiment.py with `--data toy`,
`--transform toy_oracle` and `--fixed-epochs`; use run_experiment.py directly to choose the transformation
bank or to train until convergence. The output folder is described in run_experiment.py.
"""
import argparse
from pathlib import Path

from run_experiment import default_name, main as run_main, parse_kv

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/toy.yml"],
                    help="YAML files merged left to right (experiment first, then ablations)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. --set gamma=0 seed=3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--spurious", type=float, default=0.0, help="prob. that the background colour follows the class")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=10)
    args = ap.parse_args()

    seed = parse_kv(args.set).get("seed", 0)
    data_params = {"n_classes": args.classes, "spurious": args.spurious}
    name = args.name
    if name is None:                      # same default name as before: toy[_spur0.9][_ablations]_seedN
        parts = ["toy"] + ([f"spur{args.spurious:g}"] if args.spurious else []) + [Path(c).stem for c in args.config[1:]]
        name = "_".join(parts + [f"seed{seed}"])
    argv = ["--data", "toy", "--data-set", *(f"{k}={v}" for k, v in data_params.items()),
            "--transform", "toy_oracle", "--config", *args.config, "--fixed-epochs", str(args.epochs),
            "--device", args.device, "--out-root", args.out_root, "--name", name,
            "--snapshot-every", str(args.snapshot_every)]
    if args.set:
        argv += ["--set", *args.set]
    if args.overwrite:
        argv.append("--overwrite")
    run_main(argv)
