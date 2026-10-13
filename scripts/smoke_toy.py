"""End-to-end run of the Phase 1 pipeline on synthetic data, with everything saved under results/.

    python scripts/smoke_toy.py                                       # fond aleatoire
    python scripts/smoke_toy.py --spurious 0.9                        # fond correle aux classes (shortcut)
    python scripts/smoke_toy.py --config configs/toy.yml configs/ablations/no_standardize.yml
    python scripts/smoke_toy.py --set seed=3 gamma=0 --epochs 40 --name my_run

Writes results/<name>/ :
    config.yml           resolved config (defaults + files + --set)
    metrics.jsonl        one line per epoch (losses, twin_acc, nonempty_slots, churn...)
    final_metrics.json   ACC, NMI(cluster, class), NMI(cluster, background), clusters used
    preds.npy            slot of every image under the policy with the memory frozen
    intents.npy          m of every image
    checkpoint.pt        networks + memory
    figures/             curves.png, clusters.png (+ clusters_epXXX.png snapshots), confusion.png, embedding.png
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cibirl.train import TrainConfig, Trainer                       # noqa: E402
from eval import visualize as viz                                   # noqa: E402
from eval.metrics import cluster_acc, nmi                           # noqa: E402
from toy_data import background_redraw, make_toy                    # noqa: E402


def default_name(args, cfg) -> str:
    parts = ["toy"]
    if args.spurious:
        parts.append(f"spur{args.spurious:g}")
    parts += [Path(c).stem for c in args.config[1:]]          # ablation files
    parts.append(f"seed{cfg.seed}")
    return "_".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/toy.yml"],
                    help="YAML files merged left to right (experiment first, then ablations)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. --set gamma=0 seed=3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--spurious", type=float, default=0.0, help="prob. that the background colour follows the class")
    ap.add_argument("--device", default="cpu", help="cpu (default: faster on this tiny model) or cuda")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--name", default=None, help="run folder name (default built from the options)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    ap.add_argument("--snapshot-every", type=int, default=10, help="save a cluster grid every N epochs (0 = never)")
    args = ap.parse_args()

    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    config_files = [Path(c) if Path(c).exists() else ROOT / c for c in args.config]   # cwd first, then repo root
    cfg = TrainConfig.from_yaml(*config_files, **overrides)

    out = ROOT / args.out_root / (args.name or default_name(args, cfg))
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} already exists: choose --name, or pass --overwrite")
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    cfg.to_yaml(out / "config.yml")
    print(f"run folder: {out}", flush=True)

    images, labels, background = make_toy(n_per_class=300, n_classes=args.classes, spurious=args.spurious, seed=cfg.seed)
    y, bg = labels.numpy(), background.numpy()
    tr = Trainer(cfg, images, background_redraw, device=args.device)
    history: list[dict] = []
    metrics_file = open(out / "metrics.jsonl", "w", encoding="utf-8")

    def log(row: dict) -> None:
        history.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        print(f"ep {row['epoch']:2d} it {row['iter']:4d} | L_pos {row.get('L_pos', 0):5.3f} L_neg {row.get('L_neg', 0):5.3f} "
              f"twin {row.get('twin_acc', 0):.2f} KL {row.get('KL', 0):6.1f} L_inv {row.get('L_inv', 0):.3f} "
              f"m_std {row.get('m_std', 0):.2f} | ent {row.get('pi_entropy', 0):.2f} | slots {row['nonempty_slots']} "
              f"K_eff {row['K_eff']:.2f} churn {row['churn']:.2f}", flush=True)
        e = args.snapshot_every
        if e and row["epoch"] % e == e - 1 and row["epoch"] != args.epochs - 1:
            p = tr.predict().cpu().numpy()
            viz.plot_cluster_grid(images, p, out / "figures" / f"clusters_ep{row['epoch'] + 1:03d}.png", cfg.num_slots,
                                  labels=y, title=f"Clusters formed by the policy, after epoch {row['epoch'] + 1}")

    tr.fit(args.epochs, log_fn=log)
    metrics_file.close()

    # ---- final clustering: argmax of the policy with the memory frozen (README 6.7) ----------------
    pred = tr.predict().cpu().numpy()
    m = tr.intents().cpu().numpy()
    final = {"name": out.name, "seed": cfg.seed, "epochs": args.epochs, "spurious": args.spurious,
             "acc": cluster_acc(pred, y, cfg.num_slots), "nmi_class": nmi(pred, y), "nmi_background": nmi(pred, bg),
             "nmi_class_background_in_data": nmi(y, bg), "clusters_used": int(len(np.unique(pred)))}
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
