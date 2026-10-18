"""The comparison table: one row per method, the same columns for every method (README Section 11.1.5).

    python scripts/compare_methods.py results/toy_compare                     # baselines (heldout.json in each run) + our rows
    python scripts/compare_methods.py results/toy_compare/heldout_final_spur90/heldout_metrics.json results/toy_compare
    python scripts/compare_methods.py results/toy_compare --out results/toy_compare/table.md --csv results/toy_compare/table.csv

Each input is either a folder (every <run>/heldout.json under it, written by scripts/run_baseline.py) or a JSON file holding a
list of such rows (heldout_metrics.json, written by scripts/eval_heldout.py for our model and for baseline 3). Rows are grouped by
run name without the `_seed<k>` suffix: that name is the row label (see the names in configs/suites/toy_compare.yml).

Columns: runs, then for the train images and each test mode the number of RIGHT outputs (exactly the true number of clusters
and ACC = 1) and the mean ACC, then Delta ACC = ACC(aligned) - ACC(inverted) (0 = the nuisance is not used), NMI(cluster,
nuisance) on the iid test set and the mean number of clusters. The labels only evaluate.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.protocol import TEST_MODES, is_right         # noqa: E402

PARTS = ("train",) + TEST_MODES


def collect_rows(paths) -> list[dict]:
    rows: list[dict] = []
    for p in map(Path, paths):
        if p.is_dir():
            rows += [json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("*/heldout.json"))]
        elif p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            rows += data if isinstance(data, list) else [data]
        else:
            raise SystemExit(f"{p}: no such file or folder")
    return rows


def label_of(row: dict) -> str:
    return re.sub(r"_seed\d+$", "", row["run"])


def n_classes_of(row: dict) -> int:
    """True number of classes; older rows (results/toy_final_report) did not store it: the toy has 5."""
    return int(row.get("n_classes", 5))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[label_of(r)].append(r)
    out = []
    for label, rs in groups.items():
        parts = [p for p in PARTS if all(p in r for r in rs)]
        s = {"method": label, "runs": len(rs), "parts": parts}
        for p in parts:
            acc = np.array([r[p]["acc"] for r in rs])
            s[p] = {"right": int(sum(is_right(r[p], n_classes_of(r)) for r in rs)), "acc_mean": float(acc.mean()),
                    "acc_std": float(acc.std()), "clusters": float(np.mean([r[p]["clusters"] for r in rs])),
                    "nmi_background": (float(np.mean([r[p]["nmi_background"] for r in rs]))
                                       if all(r[p].get("nmi_background") is not None for r in rs) else None)}
        if "aligned" in parts and "inverted" in parts:
            d = np.array([r["aligned"]["acc"] - r["inverted"]["acc"] for r in rs])
            s["delta_acc"] = {"mean": float(d.mean()), "std": float(d.std())}
        out.append(s)
    return sorted(out, key=lambda s: s["method"])


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(summary: list[dict]) -> str:
    head = ["Method", "Runs"] + [f"{p}: right / ACC" for p in PARTS] + ["$\\Delta$ACC", "NMI(cluster, nuisance), iid", "Clusters (train)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summary:
        cells = [f"`{s['method']}`", str(s["runs"])]
        for p in PARTS:
            cells.append(f"{s[p]['right']} / {s['runs']}, {s[p]['acc_mean']:.3f} $\\pm$ {s[p]['acc_std']:.3f}" if p in s else "-")
        cells.append(f"{s['delta_acc']['mean']:.3f} $\\pm$ {s['delta_acc']['std']:.3f}" if "delta_acc" in s else "-")
        cells.append(fmt(s["iid"]["nmi_background"], 4) if "iid" in s else "-")
        cells.append(fmt(s["train"]["clusters"], 2) if "train" in s else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_csv(summary: list[dict], path: Path) -> None:
    cols = ["method", "runs"] + [f"{p}_{k}" for p in PARTS for k in ("right", "acc_mean", "acc_std", "clusters", "nmi_background")] \
        + ["delta_acc_mean", "delta_acc_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in summary:
            row = [s["method"], s["runs"]]
            for p in PARTS:
                row += [s[p][k] if p in s else "" for k in ("right", "acc_mean", "acc_std", "clusters", "nmi_background")]
            row += [s["delta_acc"]["mean"], s["delta_acc"]["std"]] if "delta_acc" in s else ["", ""]
            w.writerow(row)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="folders with <run>/heldout.json, or heldout_metrics.json files")
    ap.add_argument("--out", default=None, help="write the Markdown table here")
    ap.add_argument("--csv", default=None, help="write the numbers as CSV here")
    args = ap.parse_args(argv)
    rows = collect_rows(args.paths)
    if not rows:
        raise SystemExit("no result found (run scripts/run_baseline.py / scripts/eval_heldout.py first)")
    summary = summarize(rows)
    table = to_markdown(summary)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    if args.csv:
        to_csv(summary, Path(args.csv))


if __name__ == "__main__":
    main()
