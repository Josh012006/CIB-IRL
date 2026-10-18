"""Train and evaluate ONE baseline (README Section 9.2, items 1 and 2) on one benchmark and one seed.

    python scripts/run_baseline.py --method kmeans --data toy --set num_clusters=5
    python scripts/run_baseline.py --method kmeans --data toy --set num_clusters=10
    python scripts/run_baseline.py --method simclr --data toy --transform toy_oracle --set num_clusters=10 epochs=100
    python scripts/run_baseline.py --method iic    --data toy --transform toy_oracle --data-set spurious=0.9 --set seed=3
    python scripts/run_baseline.py --method kmeans --data colored_mnist --set features=resnet18 num_clusters=10

Baseline 3 (standard multi-intention IRL) is OUR trainer without the Information Bottleneck: it runs through
scripts/run_experiment.py with configs/baselines/irl_noib.yml. Only items 1 and 2 are here.

The method is fit on the unlabeled TRAIN images of the benchmark (`--data-set` gives the shortcut level: spurious=0.9), then each
partition is scored with eval/protocol.py: the train images and every test mode of the benchmark (iid, aligned, inverted).
The labels only evaluate. Writes <out-root>/<name>/ :
    config.yml           the resolved BaselineConfig
    run.yml              dataset, parameters, transformation bank, command line
    final_metrics.json   ACC, NMI(cluster, class), NMI(cluster, background), clusters_used (read by eval/aggregate.py)
    heldout.json         the same metrics on the train images and on every test partition (same layout as eval_heldout.py)
    preds.npy            the cluster of every train image
    history.jsonl        the training loss per epoch (simclr, iic)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from baselines import BaselineConfig, build_clusterer, METHODS      # noqa: E402
from cibirl import transforms                                       # noqa: E402
from eval.protocol import partition_metrics                         # noqa: E402
from run_experiment import parse_kv, resolve_bank, resolve_file     # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data", default="toy", help=f"benchmark: {sorted(datasets.BENCHMARKS)}")
    ap.add_argument("--data-set", nargs="*", default=[], metavar="KEY=VALUE", help="dataset parameters, e.g. spurious=0.9")
    ap.add_argument("--transform", default=None, help="transformation bank (name in configs/transforms/ or a .yml); needed by simclr and iic")
    ap.add_argument("--config", nargs="+", default=[], help="BaselineConfig YAML files merged left to right")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="BaselineConfig overrides, e.g. num_clusters=10 seed=3")
    ap.add_argument("--device", default=None, help="cpu (default) or cuda")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="only score the train images")
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    if args.data not in datasets.BENCHMARKS:
        raise SystemExit(f"unknown benchmark '{args.data}'. Available: {sorted(datasets.BENCHMARKS)}")
    bench = datasets.BENCHMARKS[args.data]
    overrides = parse_kv(args.set)
    if args.device:
        overrides["device"] = args.device
    cfg = BaselineConfig.from_yaml(*[resolve_file(c) for c in args.config], method=args.method, **overrides)
    if args.method in ("simclr", "iic") and not args.transform:
        raise SystemExit(f"{args.method} needs --transform (the two views are two draws of the bank)")
    bank = transforms.build(resolve_bank(args.transform)) if args.transform else None

    data_params = parse_kv(args.data_set)
    train = datasets.build(args.data, cfg.seed, **data_params)
    cfg.in_channels, cfg.image_size = int(train.images.shape[1]), int(train.images.shape[2])
    y, bg = train.labels.numpy(), (train.background.numpy() if train.background is not None else None)

    tag = "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in data_params.items())
    name = args.name or "_".join(p for p in (args.method, f"k{cfg.num_clusters}", args.data, tag, f"seed{cfg.seed}") if p)
    out = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    out = out / name
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} already exists: choose --name, or pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg.to_yaml(out / "config.yml")
    (out / "run.yml").write_text(yaml.safe_dump(
        {"data": args.data, "data_params": train.info, "transform": bank.name if bank else None, "method": cfg.method,
         "uses_labels_for_training": False, "command": " ".join(sys.argv)}, sort_keys=False, allow_unicode=True), encoding="utf-8")

    t0 = time.time()
    clusterer = build_clusterer(cfg, bank).fit(train.images)
    pred = clusterer.predict(train.images)
    print(f"{name}: fit in {time.time() - t0:.1f}s", flush=True)

    K = cfg.num_clusters
    held = {"run": name, "suite": out.parent.name, "seed": cfg.seed, "method": cfg.method,
            "train": partition_metrics(pred, y, bg, None, K)}
    if not args.no_test:
        for mode in bench.modes:
            test = datasets.build_test(args.data, cfg.seed, mode, **train.info)
            held[mode] = partition_metrics(clusterer.predict(test.images), test.labels.numpy(),
                                           test.background.numpy() if test.background is not None else None, None, K)
    final = {"name": name, "seed": cfg.seed, "data": args.data, **train.info, "method": cfg.method, "num_clusters": K,
             "features": cfg.features if cfg.method in ("kmeans", "gmm") else None,
             "transform": bank.name if bank else None, "epochs_run": len(clusterer.history) or None, "converged": None,
             "acc": held["train"]["acc"], "nmi_class": held["train"]["nmi_class"], "clusters_used": held["train"]["clusters"]}
    if bg is not None:
        final["nmi_background"] = held["train"]["nmi_background"]
    (out / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out / "heldout.json").write_text(json.dumps(held, indent=1), encoding="utf-8")
    np.save(out / "preds.npy", pred)
    if clusterer.history:
        (out / "history.jsonl").write_text("\n".join(json.dumps(r) for r in clusterer.history) + "\n", encoding="utf-8")
    print("  ".join(f"{p} ACC {held[p]['acc']:.3f} ({held[p]['clusters']} clusters)" for p in held if isinstance(held[p], dict)), flush=True)
    return held


if __name__ == "__main__":
    main()
