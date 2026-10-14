"""Mean +- std over seeds of the final metrics written by the runs.

    python -m eval.aggregate results
    python -m eval.aggregate results/toy_regimes

Reads <root>/*/final_metrics.json, groups runs by name with the `_seed<k>` suffix removed.
`epochs` is the number of epochs actually run; `converged` counts the runs that stopped on the label-free
convergence rule (the others hit max_epochs, or ran a fixed number of epochs).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["acc", "nmi_class", "nmi_background", "clusters_used", "epochs_run"]
HEADERS = {"epochs_run": "epochs"}
LEGACY = {"epochs_run": "epochs"}          # runs written before convergence existed called it `epochs`


def collect(root: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(root.glob("*/final_metrics.json")):
        groups[re.sub(r"_seed\d+$", "", f.parent.name)].append(json.loads(f.read_text(encoding="utf-8")))
    return groups


def _values(runs: list[dict], key: str) -> np.ndarray:
    vals = [r[key] if key in r else r.get(LEGACY.get(key, ""), None) for r in runs]
    return np.array([v for v in vals if v is not None], dtype=float)


def summary_rows(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name, runs in groups.items():
        row = {"experiment": name, "seeds": len(runs)}
        for k in KEYS:
            v = _values(runs, k)
            row[k] = (float(v.mean()), float(v.std())) if len(v) else None
        flags = [r["converged"] for r in runs if "converged" in r]
        row["converged"] = (sum(flags), len(flags)) if flags else None
        rows.append(row)
    return rows


def main(root: str = "results") -> None:
    groups = collect(Path(root))
    if not groups:
        print(f"no final_metrics.json under {root}/*/")
        return
    width = max(30, max(len(n) for n in groups) + 2)
    print(f"{'experiment':{width}s} {'seeds':>5s} " + " ".join(f"{HEADERS.get(k, k):>18s}" for k in KEYS) + f" {'converged':>10s}")
    for row in summary_rows(groups):
        cells = [f"{row[k][0]:8.3f} +- {row[k][1]:5.3f}" if row[k] else "n/a" for k in KEYS]
        conv = f"{row['converged'][0]}/{row['converged'][1]}" if row["converged"] else "-"
        print(f"{row['experiment']:{width}s} {row['seeds']:5d} " + " ".join(f"{c:>18s}" for c in cells) + f" {conv:>10s}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
