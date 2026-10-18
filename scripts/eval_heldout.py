"""Evaluate trained runs on a TEST partition: images the model never saw (docs 5.31).

Training clusters the 1500 images of its own training set (transductive, no label). Here the same trained model, frozen,
assigns NEW images of the same classes: the memory is not touched, the output is the argmax of the policy against the
memory of the end of training (README Section 5.4, with the prior), and the slots are renamed by the final clean-up (mode `relabel`,
docs 5.29). The labels only EVALUATE.

Test sets (`datasets.build_test`; for the toy, scripts/toy_data.py `make_toy_test`: same shapes and palette as the training set
of the seed, new images). The dataset, its parameters (e.g. a colour shortcut `spurious`) and the transformation bank are read
from the run folder (run.yml, transform.yml), so a run trained on any benchmark or with a shortcut is scored on ITS test sets:
    iid       the nuisance is independent of the class, as in the toy training
    aligned   the nuisance points to the right class (toy: the background of a class is its palette colour)
    inverted  the nuisance points to a wrong class (toy: the background of class c is the palette colour of class c+1)
Metrics of README Section 9.3: ACC (Hungarian), NMI(cluster, class), NMI(cluster, background) (spurious leakage), the drop
Delta ACC = ACC(aligned) - ACC(inverted), the worst-group ACC (groups = class x coarse background colour, on `iid`), plus the
number of clusters and the share of images sent to "open a new cluster" (an empty slot).

Usage:
    python scripts/eval_heldout.py results/toy_cov_k10_final_ac results/toy_cov_k10_final_ac_val --out results/toy_final_report
    python scripts/eval_heldout.py results/toy_compare --experiment final_spur90 --out results/toy_compare/heldout_final_spur90
Writes <out>/heldout_metrics.json (one row per run) and heldout_metrics.csv. Needs the checkpoint.pt and memory_pre.pt of each run.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from cibirl import transforms                                       # noqa: E402
from cibirl.cleanup import find_twins, relabel                      # noqa: E402
from cibirl.train import TrainConfig, Trainer                       # noqa: E402
from eval.protocol import TEST_MODES as MODES, is_right, partition_metrics   # noqa: E402,F401  (partition_metrics: kept importable from here)


def evaluate_run(run_dir: str) -> dict:
    torch.set_num_threads(1)
    run = Path(run_dir)
    cfg = TrainConfig.from_yaml(run / "config.yml")
    if cfg.cleanup_mode != "relabel" or not cfg.final_cleanup:
        raise SystemExit(f"{run}: this script reads the output of mode 'relabel' (final_cleanup true)")
    meta = yaml.safe_load((run / "run.yml").read_text(encoding="utf-8"))
    name, params = meta["data"], meta["data_params"]                # e.g. toy, {n_classes: 5, ..., spurious: 0.9}
    data = datasets.build(name, cfg.seed, **params)
    y = data.labels.numpy()
    bg = data.background.numpy() if data.background is not None else None
    bank = transforms.build(run / "transform.yml")
    tr = Trainer(cfg, data.images, bank, device="cpu", eval_labels=data.labels)
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    for k in ("encoder", "intent", "reward", "policy", "value"):
        getattr(tr, k).load_state_dict(ck[k])
        getattr(tr, k).eval()
    pre = torch.load(run / "memory_pre.pt", map_location="cpu", weights_only=False)      # the memory the output is read against
    tr.memory.assign, tr.memory.member_m, tr.memory.counts, tr.memory.summaries = pre["assign"], pre["member_m"], pre["counts"], pre["summaries"]
    rep, _ = find_twins(tr.memory.summaries, tr.memory.counts, cfg.cleanup_ratio)
    kept = set(rep.values())

    def read(images=None):
        return relabel(tr.predict(images), rep).numpy()

    official = np.load(run / "preds.npy")
    pred_train = read()
    row = {"run": run.name, "suite": run.parent.name, "seed": cfg.seed, "method": "cibirl", "data": name,
           "n_classes": int(len(np.unique(y))), "spurious": params.get("spurious"),
           "train_reproduces_preds": bool((pred_train == official).all())}
    row["train"] = partition_metrics(pred_train, y, bg, kept, cfg.num_slots)
    for mode in datasets.BENCHMARKS[name].modes:
        test = datasets.build_test(name, cfg.seed, mode, **params)
        row[mode] = partition_metrics(read(test.images), test.labels.numpy(),
                                      test.background.numpy() if test.background is not None else None, kept, cfg.num_slots)
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", help="suite folders (results/<suite>) holding <experiment>_seed*/ runs")
    ap.add_argument("--experiment", default="final", help="name of the experiment in the suites: runs are <experiment>_seed<k> (default final)")
    ap.add_argument("--out", default="results/toy_final_report")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (quick check)")
    args = ap.parse_args(argv)
    runs = []
    for s in args.suites:
        runs += sorted(glob.glob(os.path.join(s, f"{args.experiment}_seed*")), key=lambda p: int(p.rsplit("seed", 1)[1]))
    runs = runs[:args.limit] if args.limit else runs
    with mp.Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(evaluate_run, runs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(runs):
                print(f"{i}/{len(runs)} runs", flush=True)
    rows.sort(key=lambda r: r["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_metrics.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    cols = ["acc", "nmi_class", "nmi_background", "clusters", "clusters_ge10", "open_share", "worst_group_acc"]
    with open(out / "heldout_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suite", "seed", "train_reproduces_preds"] + [f"{p}_{c}" for p in ("train",) + MODES for c in cols])
        for r in rows:
            w.writerow([r["suite"], r["seed"], r["train_reproduces_preds"]] + [r[p][c] for p in ("train",) + MODES for c in cols])
    ok = sum(r["train_reproduces_preds"] for r in rows)
    print(f"the frozen-memory reading reproduces the saved output on the training images in {ok}/{len(rows)} runs")
    for p in ("train",) + MODES:
        right = sum(1 for r in rows if is_right(r[p], r["n_classes"]))
        print(f"{p:>9s}: right {right}/{len(rows)}  ACC {np.mean([r[p]['acc'] for r in rows]):.4f}  NMI {np.mean([r[p]['nmi_class'] for r in rows]):.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
