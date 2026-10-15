"""How good are the code z and the intent m, epoch after epoch? (diagnostic of a run; labels used for EVALUATION only)

Trains one run exactly as scripts/run_experiment.py would (same seed, same data, same config: the numbers of a probed run
are identical to the unprobed one, because the probe restores the torch random state and takes no gradient) and, at some
epochs, measures:

    knn_z, knn_m          leave-one-out accuracy of the 10 nearest neighbours (labels = true classes) on z and on m
    kmeans_m              ACC of k-means (k = number of classes) on m
    twin_retrieval_m      share of images whose view A has the view B of the SAME image as nearest neighbour in m
    nn_view_same_class    share of images whose view A has a view B of the same CLASS as nearest neighbour in m
    acc_policy            ACC of the clusters the policy forms (argmax, memory frozen); meaningless before the learner acts
    twin_acc_reward, KL, L_psi   the training statistics of the epoch

    python scripts/probe_representation.py --seed 0 --last 49
    python scripts/probe_representation.py --seed 0 --last 19 --at 0 9 19 --set beta=1e-3 --out-root results/mnist_digits/probe_beta1e-3

Writes <out-root>/seed<k>.jsonl, one line per probed epoch. The pixels' own kNN accuracy is printed as a reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from cibirl import transforms                                       # noqa: E402
from cibirl.train import TrainConfig, Trainer                       # noqa: E402
from eval.metrics import cluster_acc                                # noqa: E402


def knn_accuracy(f: torch.Tensor, y: np.ndarray, n_classes: int, k: int = 10) -> float:
    d = torch.cdist(f, f)
    d.fill_diagonal_(float("inf"))
    nn_idx = d.topk(k, largest=False).indices.numpy()
    pred = np.array([np.bincount(v, minlength=n_classes).argmax() for v in y[nn_idx]])
    return float((pred == y).mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--data-set", nargs="*", default=["n_per_class=300"], metavar="KEY=VALUE")
    ap.add_argument("--transform", default="digits")
    ap.add_argument("--config", default="configs/mnist.yml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="TrainConfig overrides, e.g. beta=1e-3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--last", type=int, default=49, help="last epoch (included)")
    ap.add_argument("--at", type=int, nargs="*", default=[0, 2, 4, 6, 7, 8, 9, 11, 14, 19, 24, 29, 39, 49, 79, 99],
                    help="epochs probed")
    ap.add_argument("--out-root", default="results/mnist_digits/probe")
    args = ap.parse_args(argv)
    from sklearn.cluster import KMeans
    from run_experiment import parse_kv, resolve_bank, resolve_file

    cfg = TrainConfig.from_yaml(resolve_file(args.config), **{"seed": args.seed, **parse_kv(args.set)})
    bank = transforms.build(resolve_bank(args.transform))
    data = datasets.build(args.data, args.seed, **parse_kv(args.data_set))
    y = data.labels.numpy()
    yt, n_cls, n = data.labels, int(data.labels.max()) + 1, len(y)
    tr = Trainer(cfg, data.images, bank, device="cpu", eval_labels=data.labels)
    print(f"kNN(10) accuracy of the pixels: {knn_accuracy(data.images.flatten(1), y, n_cls):.3f}", flush=True)
    out_dir = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out = open(out_dir / f"seed{args.seed}.jsonl", "w", encoding="utf-8")

    def enc(x):
        return torch.cat([tr.encoder(x[s:s + 512], sample=False).z for s in range(0, len(x), 512)])

    def log(row: dict) -> bool:
        e = row["epoch"]
        if e in args.at:
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(123)
                z = enc(data.images)
                m = tr.intent.mean(z)
                ma, mb = tr.intent.mean(enc(bank(data.images))), tr.intent.mean(enc(bank(data.images)))
                top = torch.cdist(ma, mb).argmin(1)
                res = {"epoch": e, "twin_acc_reward": row["twin_acc"], "KL": row["KL"], "L_psi": row.get("L_psi"),
                       "acc_policy": cluster_acc(tr.predict().numpy(), y, cfg.num_slots),
                       "knn_z": knn_accuracy(z, y, n_cls), "knn_m": knn_accuracy(m, y, n_cls),
                       "kmeans_m": cluster_acc(KMeans(n_cls, n_init=5, random_state=0).fit_predict(m.numpy()), y, n_cls),
                       "twin_retrieval_m": float((top == torch.arange(n)).float().mean()),
                       "nn_view_same_class": float((yt[top] == yt).float().mean())}
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        return e >= args.last

    tr.fit(args.last + 1, log_fn=log)
    out.close()


if __name__ == "__main__":
    main()
