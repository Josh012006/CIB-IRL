"""Does a transformation bank make instances easy to tell apart, in PIXEL space? (diagnostic; labels = evaluation only)

For every image, two views A_i and B_i are drawn with the bank (applied `--chain` times in a row, like an orbit of the
expert). In pixel space:

    d_view     median distance between the two views of one image
    d_same     median distance between views of two different images of the SAME class
    d_diff     median distance between views of two images of DIFFERENT classes
    twin       share of images whose view A has the view B of the same image as its nearest neighbour among all B
    same_class share of images whose view A has a view B of the same class as its nearest neighbour

d_view << d_same means that an instance is recognised from its pixels whatever the bank does (identity of the instance,
not of the class, is what the orbits teach); d_view close to d_same means that the bank blurs the instance.

    python scripts/probe_bank.py --transform digits digits_strong generic --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from cibirl import transforms                                       # noqa: E402


def bank_geometry(images: torch.Tensor, labels: torch.Tensor, bank, chain: int = 1, seed: int = 0) -> dict:
    n = images.shape[0]
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        views = []
        for _ in range(2):
            v = images
            for _ in range(chain):
                v = bank(v)
            views.append(v.flatten(1))
        a, b = views
        dist = torch.cdist(a, b)                                    # dist[i, j] = |A_i - B_j|
        eye = torch.eye(n, dtype=torch.bool)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        top = dist.argmin(1)
        return {"d_view": dist[eye].median().item(), "d_same": dist[same & ~eye].median().item(),
                "d_diff": dist[~same].median().item(),
                "twin": (top == torch.arange(n)).float().mean().item(),
                "same_class": (labels[top] == labels).float().mean().item()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--data-set", nargs="*", default=["n_per_class=300"], metavar="KEY=VALUE")
    ap.add_argument("--transform", nargs="+", default=["digits"], help="banks: names in configs/transforms/ or .yml paths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain", type=int, nargs="+", default=[1, 2], help="how many times the bank is applied in a row")
    ap.add_argument("--out", default=None, help="write the numbers to this JSON file")
    args = ap.parse_args(argv)
    from run_experiment import parse_kv, resolve_bank

    data = datasets.build(args.data, args.seed, **parse_kv(args.data_set))
    out = {}
    print(f"{'bank':>16s} {'chain':>5s} {'d_view':>7s} {'d_same':>7s} {'d_diff':>7s} {'view/same':>9s} {'twin':>6s} {'same_class':>10s}")
    for name in args.transform:
        bank = transforms.build(resolve_bank(name))
        for chain in args.chain:
            r = bank_geometry(data.images, data.labels, bank, chain, args.seed)
            out[f"{bank.name}_chain{chain}"] = r
            print(f"{bank.name:>16s} {chain:5d} {r['d_view']:7.2f} {r['d_same']:7.2f} {r['d_diff']:7.2f} "
                  f"{r['d_view'] / r['d_same']:9.2f} {r['twin']:6.3f} {r['same_class']:10.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
