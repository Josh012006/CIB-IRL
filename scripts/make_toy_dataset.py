"""Write (or check) the toy dataset files in data/toy/.

    python scripts/make_toy_dataset.py                       # seeds 0-9, with the colour-shortcut training split (spurious 0.9)
    python scripts/make_toy_dataset.py --seeds 0-199         # every seed of the results (about 90 MB)
    python scripts/make_toy_dataset.py --verify              # regenerate and compare with the files (exit 1 on a difference)
    python scripts/make_toy_dataset.py --spurious 0.9 0.5    # other shortcut levels

One file per seed (see scripts/toy_data.py for the content). The files are float32 and exact: loading one gives the very
same tensors as `make_toy` / `make_toy_test` (checked by --verify and by tests/test_toy_files.py), so an experiment can read
the files or regenerate the images and get the same numbers. The generator stays the source of truth.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_data import TOY_DEFAULTS, TOY_DIR, content_sha256, save_toy, toy_arrays, toy_path   # noqa: E402


def parse_seeds(spec: list[str]) -> list[int]:
    """['0-9', '12', '20-22'] -> [0..9, 12, 20, 21, 22]"""
    out: list[int] = []
    for item in spec:
        for part in item.split(","):
            lo, _, hi = part.partition("-")
            out += list(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def verify(seeds: list[int], root: Path, spurious) -> int:
    """Regenerate every array and compare it with the file. Returns the number of problems."""
    bad = 0
    for s in seeds:
        path = toy_path(s, root)
        if not path.exists():
            print(f"MISSING  {path}")
            bad += 1
            continue
        fresh = toy_arrays(s, spurious)
        with np.load(path) as f:
            for k, v in fresh.items():
                if k not in f.files:
                    print(f"seed {s}: '{k}' is missing in the file")
                    bad += 1
                elif v.dtype.kind == "f":
                    if not np.allclose(f[k], v, atol=1e-6):
                        print(f"seed {s}: '{k}' differs from the generator (max diff {np.abs(f[k] - v).max():.2e})")
                        bad += 1
                elif not np.array_equal(f[k], v):
                    print(f"seed {s}: '{k}' differs from the generator")
                    bad += 1
    print("all files match the generator" if not bad else f"{bad} problem(s)")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=["0-9"], help="e.g. 0-9 or 0-199 or '3 7 12-15' (default 0-9)")
    ap.add_argument("--spurious", nargs="*", type=float, default=[0.9],
                    help="levels of the colour shortcut for the extra training splits (default 0.9; none: pass no value)")
    ap.add_argument("--out", default=str(TOY_DIR), help="folder (default data/toy)")
    ap.add_argument("--verify", action="store_true", help="compare the files with the generator instead of writing")
    args = ap.parse_args(argv)
    seeds, root = parse_seeds(args.seeds), Path(args.out)
    if args.verify:
        sys.exit(1 if verify(seeds, root, tuple(args.spurious)) else 0)
    for s in seeds:
        save_toy(s, root, tuple(args.spurious))
    files = {p.name: content_sha256(p) for p in sorted(root.glob("seed*.npz"))}
    manifest = {"generator": "scripts/toy_data.py (make_toy, make_toy_test)", "parameters": TOY_DEFAULTS,
                "spurious_levels": args.spurious, "test_modes": ["iid", "aligned", "inverted"],
                "seeds": sorted(int(n[4:7]) for n in files), "content_sha256": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in root.glob("seed*.npz")) / 1e6
    print(f"wrote {len(seeds)} file(s) in {root} ({len(files)} seeds in total, {size:.1f} MB)")


if __name__ == "__main__":
    main()
