"""One run of the whole pipeline: a dataset, a transformation bank, a config, and a stopping rule.

    python scripts/run_experiment.py --data toy --transform generic
    python scripts/run_experiment.py --data toy --data-set spurious=0.9 --transform informed_color --set seed=3
    python scripts/run_experiment.py --data toy --transform toy_oracle --config configs/toy.yml configs/ablations/no_inv.yml
    python scripts/run_experiment.py --data toy --transform generic --fixed-epochs 30        # no convergence test
    python scripts/run_experiment.py --data toy --transform generic --stop-set max_epochs=500 min_epochs=60

By default training runs UNTIL CONVERGENCE (cibirl/convergence.py: stable argmax partition, constant number of
clusters, plateau of the expert objective; no label is used) or until `max_epochs`.

Writes <out-root>/<name>/ :
    config.yml           resolved TrainConfig (defaults + files + --set)
    transform.yml        the transformation bank as it was used
    run.yml              dataset and parameters, stopping rule, command line
    metrics.jsonl        one line per epoch (losses, twin_acc, churn, part_change, n_clusters...; eval_* = labels,
                         for monitoring only)
    final_metrics.json   ACC, NMI(cluster, class), NMI(cluster, background), epochs_run, converged, stop_reason
    preds.npy            slot of every image under the policy with the memory frozen
    intents.npy          m of every image
    checkpoint.pt        networks + memory
    figures/             curves.png, clusters.png (+ clusters_epXXX.png snapshots), confusion.png, embedding.png

The model that is evaluated is the one of the LAST epoch: no epoch is picked by looking at the labels.
"""
import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from cibirl import transforms                                       # noqa: E402
from cibirl.convergence import ConvergenceMonitor, StopConfig       # noqa: E402
from cibirl.train import TrainConfig, Trainer                       # noqa: E402
from eval import visualize as viz                                   # noqa: E402
from eval.metrics import cluster_acc, nmi                           # noqa: E402


def parse_kv(items) -> dict:
    """['a=1', 'b=[1, 2]', 'c=true'] -> {'a': 1, 'b': [1, 2], 'c': True} (values read as YAML)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve_file(p: str) -> Path:
    """cwd first, then the repo root (so that commands work from any folder)."""
    return Path(p) if Path(p).exists() else ROOT / p


def resolve_bank(arg: str) -> Path:
    """A bank is a file path or a name in configs/transforms/."""
    if arg.endswith((".yml", ".yaml")) or Path(arg).exists():
        return resolve_file(arg)
    path = ROOT / "configs" / "transforms" / f"{arg}.yml"
    if not path.exists():
        names = sorted(p.stem for p in (ROOT / "configs" / "transforms").glob("*.yml"))
        raise SystemExit(f"unknown transform bank '{arg}'. Available: {names}")
    return path


def default_name(data: str, data_params: dict, bank: str, configs: list[str], seed: int) -> str:
    parts = [data] + [f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in data_params.items()]
    parts.append(bank)
    parts += [Path(c).stem for c in configs[1:]]                    # ablation files
    parts.append(f"seed{seed}")
    return "_".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="toy", help=f"dataset: {sorted(datasets.DATASETS)}")
    ap.add_argument("--data-set", nargs="*", default=[], metavar="KEY=VALUE", help="dataset parameters, e.g. spurious=0.9")
    ap.add_argument("--transform", required=True, help="transformation bank: a name in configs/transforms/ or a .yml path")
    ap.add_argument("--config", nargs="+", default=None,
                    help="TrainConfig YAML files merged left to right (default: the dataset's default config)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="TrainConfig overrides, e.g. gamma=0 seed=3")
    ap.add_argument("--stop-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="stopping rule, e.g. max_epochs=500 min_epochs=60 (see cibirl/convergence.py)")
    ap.add_argument("--fixed-epochs", type=int, default=None, help="run exactly N epochs, no convergence test")
    ap.add_argument("--device", default="cpu", help="cpu (default: faster on tiny models) or cuda")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--name", default=None, help="run folder name (default built from the options)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    ap.add_argument("--snapshot-every", type=int, default=10, help="save a cluster grid every N epochs (0 = never)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    data_params, overrides, stop_overrides = parse_kv(args.data_set), parse_kv(args.set), parse_kv(args.stop_set)
    if args.data not in datasets.DATASETS:
        raise SystemExit(f"unknown dataset '{args.data}'. Available: {sorted(datasets.DATASETS)}")
    config_files = [resolve_file(c) for c in (args.config or [datasets.DEFAULT_CONFIG[args.data]])]
    cfg = TrainConfig.from_yaml(*config_files, **overrides)
    stop = StopConfig.from_dict(stop_overrides)
    if args.fixed_epochs is not None:
        stop = replace(stop, max_epochs=args.fixed_epochs, enabled=False)
    bank = transforms.build(resolve_bank(args.transform))
    data = datasets.build(args.data, cfg.seed, **data_params)
    images, y = data.images, data.labels.numpy()
    bg = data.background.numpy() if data.background is not None else None
    if images.shape[1] != cfg.in_channels or images.shape[2] != cfg.image_size or images.shape[3] != cfg.image_size:
        raise SystemExit(f"data are {tuple(images.shape[1:])} but the config says in_channels={cfg.in_channels}, "
                         f"image_size={cfg.image_size}")

    name = args.name or default_name(args.data, data_params, bank.name, args.config or [""], cfg.seed)
    out = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    out = out / name
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} already exists: choose --name, or pass --overwrite")
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    cfg.to_yaml(out / "config.yml")
    (out / "transform.yml").write_text(yaml.safe_dump(bank.spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "run.yml").write_text(yaml.safe_dump(
        {"data": args.data, "data_params": data.info, "transform": bank.name, "stop": stop.to_dict(),
         "device": args.device, "command": " ".join(sys.argv)}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"run folder: {out}\n{bank!r}\nstop: {stop}", flush=True)

    tr = Trainer(cfg, images, bank, device=args.device)
    monitor = ConvergenceMonitor(stop)
    history: list[dict] = []
    metrics_file = open(out / "metrics.jsonl", "w", encoding="utf-8")
    t0 = time.time()

    def log(row: dict) -> bool:
        pred = tr.predict().cpu().numpy()
        stop_now = monitor.observe(row, pred)                       # adds part_change and n_clusters to the row
        # EVALUATION ONLY: the labels are used to follow the dynamics, never to train nor to stop (the monitor
        # does not read these keys)
        row["eval_acc"], row["eval_nmi_class"] = cluster_acc(pred, y, cfg.num_slots), nmi(pred, y)
        if bg is not None:
            row["eval_nmi_background"] = nmi(pred, bg)
        history.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        print(f"ep {row['epoch']:3d} it {row['iter']:5d} | L_pos {row.get('L_pos', 0):5.3f} L_neg {row.get('L_neg', 0):5.3f} "
              f"twin {row.get('twin_acc', 0):.2f} KL {row.get('KL', 0):6.1f} L_inv {row.get('L_inv', 0):.3f} "
              f"m_std {row.get('m_std', 0):.2f} | ent {row.get('pi_entropy', 0):.2f} | clusters {row['n_clusters']} "
              f"K_eff {row['K_eff']:.2f} churn {row['churn']:.2f} dpart {row['part_change']:.3f} "
              f"| acc* {row['eval_acc']:.2f} | {time.time() - t0:5.0f}s", flush=True)
        e = args.snapshot_every
        if e and not stop_now and row["epoch"] % e == e - 1:
            viz.plot_cluster_grid(images, pred, out / "figures" / f"clusters_ep{row['epoch'] + 1:03d}.png", cfg.num_slots,
                                  labels=y, title=f"Clusters formed by the policy, after epoch {row['epoch'] + 1}")
        return stop_now

    tr.fit(stop.max_epochs, log_fn=log)
    metrics_file.close()
    print(f"\nstopped after {len(history)} epochs: {monitor.reason}", flush=True)

    # ---- final clustering: argmax of the policy with the memory frozen (README 6.7) ----------------
    pred = tr.predict().cpu().numpy()
    m = tr.intents().cpu().numpy()
    final = {"name": out.name, "seed": cfg.seed, "data": args.data, **data.info, "transform": bank.name,
             "epochs_run": len(history), "converged": monitor.converged, "stop_reason": monitor.reason,
             "acc": cluster_acc(pred, y, cfg.num_slots), "nmi_class": nmi(pred, y)}
    if bg is not None:
        final["nmi_background"] = nmi(pred, bg)
        final["nmi_class_background_in_data"] = nmi(y, bg)
    final["clusters_used"] = int(len(np.unique(pred)))
    (out / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    np.save(out / "preds.npy", pred)
    np.save(out / "intents.npy", m)
    tr.save(str(out / "checkpoint.pt"))

    fig = out / "figures"
    viz.plot_curves(history, fig / "curves.png")
    viz.plot_cluster_grid(images, pred, fig / "clusters.png", cfg.num_slots, labels=y,
                          title="Clusters formed by the policy (argmax, memory frozen)")
    viz.plot_confusion(pred, y, fig / "confusion.png", cfg.num_slots)
    viz.plot_embedding(m, pred, fig / "embedding.png", labels=y, background=bg)

    print("\n" + "\n".join(f"{k:>30s}: {v:.3f}" if isinstance(v, float) else f"{k:>30s}: {v}" for k, v in final.items()))
    print(f"\nfigures and metrics in {out}")


if __name__ == "__main__":
    main()
