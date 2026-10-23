"""Real pictures of what the method sees and returns on the MNIST runs of docs/protocol-decisions.md 5.36. Nothing is trained.

    python scripts/show_mnist10_runs.py results/mnist10_final --seeds 0 1 --out figures_dir

For every seed: (1) `expert_views_seed<s>.png`: real expert episodes (the first ones of the 2 000, not chosen). An episode = 5 images of the expert
portion, 2 views of each (drawn with the training bank and the orbit sampler, d in {0, 1, 2}), shuffled. The two views of an image are
drawn side by side under their slot; #n is the order in which the expert sees a view (opens = first view of its image, joins =
second); the digit of a slot is the TRUE one (evaluation only, the method never sees it). Red title: another slot of the episode
holds the same digit. (2) `clusters_seed<s>.png`: random images of the clusters of the final partition that merge several digits (frames
coloured by true digit) and of the fragments of one digit. Also prints descriptors (ink, size, slant) and the share of expert
episodes that contain two images of the same digit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cibirl import transforms                                             # noqa: E402
from cibirl.experts import ExpertEpisodes, OrbitSampler, split_expert_portion   # noqa: E402
from cibirl.train import TrainConfig                                      # noqa: E402
from scripts import datasets                                              # noqa: E402

TAB = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def descriptors(x):
    """x [N, 28, 28]: ink mass, bounding box height / width (pixels above 0.3), slant = cov(x, y) / var(y) of the ink."""
    out = []
    ys, xs = np.mgrid[0:28, 0:28]
    for im in x:
        m = im.sum()
        cx, cy = (im * xs).sum() / m, (im * ys).sum() / m
        vy = (im * (ys - cy) ** 2).sum() / m
        slant = (im * (xs - cx) * (ys - cy)).sum() / m / max(vy, 1e-6)
        on = np.argwhere(im > 0.3)
        h, w = (on[:, 0].max() - on[:, 0].min() + 1, on[:, 1].max() - on[:, 1].min() + 1) if len(on) else (0, 0)
        out.append((m, h, w, slant))
    return np.array(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="figures_mnist10")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        run = Path(a.root) / f"final_seed{s}"
        cfg = TrainConfig.from_yaml(run / "config.yml")
        data = datasets.build("mnist", s, digits=list(range(10)), n_per_class=300)
        imgs, y = data.images, data.labels.numpy()
        pred = np.load(run / "preds.npy")
        torch.manual_seed(100 + s)
        # ---- (1) expert episodes, rebuilt exactly as in the trainer (same seed -> same episodes) ----------------------
        ids = split_expert_portion(len(imgs), cfg.expert_fraction, cfg.seed)
        ep = ExpertEpisodes.build(ids, cfg.num_expert_episodes, cfg.orbits_per_episode, cfg.views_per_orbit, cfg.num_slots, seed=cfg.seed)
        eb = ep.batch(torch.arange(ep.num_episodes))
        cls = torch.from_numpy(y)[eb.image_ids]                                      # [E, T] true digit of every view
        n_orb = cfg.orbits_per_episode
        same = 0
        pair_same = pair_89 = pairs = 0
        for e in range(ep.num_episodes):
            first = {int(o): int(c) for o, c in zip(eb.orbit[e], cls[e])}          # digit of every orbit of the episode
            ds = list(first.values())
            same += int(len(set(ds)) < len(ds))
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    pairs += 1
                    pair_same += int(ds[i] == ds[j])
                    pair_89 += int({ds[i], ds[j]} == {8, 9})
        print(f"\n=== seed {s} ===\nexpert episodes: {ep.num_episodes}, {n_orb} images x {cfg.views_per_orbit} views;"
              f" episodes with two images of the same digit: {same / ep.num_episodes:.0%};"
              f" pairs of images told to be DIFFERENT that are the same digit: {pair_same / pairs:.1%} (8-9 pairs: {pair_89 / pairs:.1%})")
        bank = transforms.build(ROOT / "configs" / "transforms" / f"{yaml_bank(run)}.yml")
        sampler = OrbitSampler(bank, cfg.d_max)
        views = sampler(imgs[eb.image_ids[:a.episodes].flatten()]).view(a.episodes, -1, 28, 28)
        T = views.shape[1]
        n_slot = cfg.orbits_per_episode
        per = cfg.views_per_orbit
        gap = 0.45
        fig, ax = plt.subplots(a.episodes, n_slot * per + n_slot - 1, figsize=(1.15 * n_slot * per + 0.6 * n_slot, 1.55 * a.episodes + 0.9),
                               gridspec_kw={"width_ratios": ([1] * per + [gap]) * (n_slot - 1) + [1] * per})
        for e in range(a.episodes):
            slot_digit = {}
            for sl in range(n_slot):
                ts = [t for t in range(T) if int(eb.slot[e, t]) == sl]
                slot_digit[sl] = int(cls[e, ts[0]])
            for sl in range(n_slot):
                ts = [t for t in range(T) if int(eb.slot[e, t]) == sl]
                twin = [o for o in range(n_slot) if o != sl and slot_digit[o] == slot_digit[sl]]
                for j, t in enumerate(ts):
                    k = ax[e, sl * (per + 1) + j]
                    k.imshow(views[e, t], cmap="gray", vmin=0, vmax=1)
                    k.set_xticks([])
                    k.set_yticks([])
                    k.text(1, 4.5, f"#{t + 1} " + ("opens" if bool(eb.is_first[e, t]) else "joins"), color="yellow", fontsize=6.5,
                           fontweight="bold")
                    if j == 0:
                        k.set_title(f"slot {sl}   digit {slot_digit[sl]}", fontsize=8, loc="left", fontweight="bold",
                                    color="#c00000" if twin else "black")
                if sl < n_slot - 1:
                    ax[e, sl * (per + 1) + per].axis("off")
            same = sorted({d for d in slot_digit.values() if list(slot_digit.values()).count(d) > 1})
            ax[e, 0].set_ylabel(f"episode {e + 1}" + (f"\n(repeated digit: {', '.join(map(str, same))})" if same else ""), fontsize=7)
        fig.suptitle(f"seed {s}: real expert episodes. One pair of views = ONE image = ONE expert slot (always the same digit).\n"
                     f"#n = order in which the expert sees the view, opens = first view of its image (new slot), joins = second view. "
                     f"The digit is the true one (evaluation only).\nRed title: another slot of the episode holds the same digit, "
                     f"yet the expert treats the two images as different.", fontsize=7.5)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        plt.savefig(out / f"expert_views_seed{s}.png", dpi=110)
        plt.close()
        # ---- (2) final clusters -------------------------------------------------------------------------------------
        x = imgs[:, 0].numpy()
        desc = descriptors(x)
        clusters = sorted(set(pred.tolist()))
        info = []
        for k in clusters:
            members = np.where(pred == k)[0]
            counts = np.bincount(y[members], minlength=10)
            info.append((k, members, counts))
        merged = [i for i in info if (i[2] >= 0.15 * i[2].sum()).sum() >= 2 and i[2].sum() >= 100]
        by_digit = {}
        for k, members, counts in info:
            if counts.max() >= 0.8 * counts.sum() and counts.sum() >= 10:
                by_digit.setdefault(int(counts.argmax()), []).append((k, members))
        frag = {d: v for d, v in by_digit.items() if len(v) >= 2}
        print("digit descriptors, all images of the digit:  ink  height  width  slant")
        for d in range(10):
            print(f"  digit {d}: " + "  ".join(f"{v:6.2f}" for v in desc[y == d].mean(0)))
        rng = np.random.default_rng(s)
        rows = []                                                                    # (title, list of (index, digit))
        for k, members, counts in merged:
            sel = rng.choice(members, min(48, len(members)), replace=False)
            rows.append((f"cluster {k}: {len(members)} images, digits " + ", ".join(f"{d}:{c}" for d, c in enumerate(counts) if c >= 0.05 * counts.sum()),
                         [(i, y[i]) for i in sel], "merged"))
            for d in np.where(counts >= 0.15 * counts.sum())[0]:
                print(f"  merged cluster {k}, digit {d} ({counts[d]} images): " + "  ".join(f"{v:6.2f}" for v in desc[members[y[members] == d]].mean(0)))
        for d, lst in frag.items():
            for k, members in lst:
                sel = rng.choice(members, min(16, len(members)), replace=False)
                rows.append((f"digit {d} alone in cluster {k} ({len(members)} images)", [(i, y[i]) for i in sel], "fragment"))
                print(f"  fragment of {d}, cluster {k} ({len(members)}): " + "  ".join(f"{v:6.2f}" for v in desc[members].mean(0)))
        if not rows:
            continue
        ncol = 16
        nr = sum(int(np.ceil(len(r[1]) / ncol)) + 1 for r in rows)
        fig = plt.figure(figsize=(ncol * 0.85, nr * 0.9))
        gs = fig.add_gridspec(nr, ncol, hspace=0.12, wspace=0.05)
        r0 = 0
        for title, items, kind in rows:
            ax0 = fig.add_subplot(gs[r0, :])
            ax0.axis("off")
            ax0.text(0, 0.2, title, fontsize=9, fontweight="bold")
            r0 += 1
            for n, (i, d) in enumerate(items):
                k = fig.add_subplot(gs[r0 + n // ncol, n % ncol])
                k.imshow(x[i], cmap="gray", vmin=0, vmax=1)
                k.set_xticks([])
                k.set_yticks([])
                for sp in k.spines.values():
                    sp.set_edgecolor(TAB[int(d)])
                    sp.set_linewidth(3)
                k.text(1, 5, str(int(d)), color="yellow", fontsize=7, fontweight="bold")
            r0 += int(np.ceil(len(items) / ncol))
        fig.suptitle(f"seed {s}: random images of the final clusters (frame colour and yellow digit = true digit)", fontsize=10)
        plt.savefig(out / f"clusters_seed{s}.png", dpi=90, bbox_inches="tight")
        plt.close()


def yaml_bank(run):
    import yaml
    return yaml.safe_load((run / "run.yml").read_text(encoding="utf-8"))["transform"]


if __name__ == "__main__":
    main()
