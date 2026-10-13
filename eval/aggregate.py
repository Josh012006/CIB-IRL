"""Mean +- std over seeds of the final metrics written by the runs.

    python -m eval.aggregate results

Reads results/*/final_metrics.json, groups runs by name with the `_seed<k>` suffix removed.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["acc", "nmi_class", "nmi_background", "clusters_used"]


def collect(root: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(root.glob("*/final_metrics.json")):
        groups[re.sub(r"_seed\d+$", "", f.parent.name)].append(json.loads(f.read_text(encoding="utf-8")))
    return groups


def main(root: str = "results") -> None:
    groups = collect(Path(root))
    if not groups:
        print(f"no final_metrics.json under {root}/*/")
        return
    print(f"{'experiment':44s} {'seeds':>5s} " + " ".join(f"{k:>18s}" for k in KEYS))
    for name, runs in groups.items():
        cells = []
        for k in KEYS:
            v = np.array([r[k] for r in runs], dtype=float)
            cells.append(f"{v.mean():8.3f} +- {v.std():5.3f}")
        print(f"{name:44s} {len(runs):5d} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main(*sys.argv[1:2])
