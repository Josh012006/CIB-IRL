"""Figures and tables of the README Section 10 (Results, toy environment). Everything is read from files already produced:

    results/<suite>/final_seed*/{final_metrics.json, metrics.jsonl, preds.npy, intents.npy}   (the training runs)
    <out>/heldout_metrics.json                                                                  (scripts/eval_heldout.py)
    <out>/trace_seed<S>.npz                                                                     (--trace, below)

Usage (from the repo root):
    python scripts/eval_heldout.py results/toy_cov_k10_final_ac results/toy_cov_k10_final_ac_val --out results/toy_final_report
    python scripts/make_report.py results/toy_cov_k10_final_ac results/toy_cov_k10_final_ac_val --out results/toy_final_report
and, once, to record how the clusters of one run evolve (replays the run with a checkpoint per epoch, which does not change its training):
    python scripts/run_experiment.py ... --name trace_seed20 --out-root <scratch> --save-epochs 1-100     (same command as the suite, see run.yml)
    python scripts/make_report.py --trace <scratch>/trace_seed20 --out results/toy_final_report

Labels are used here only to EVALUATE and to colour the pictures: nothing below feeds back into a training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Rectangle                             # noqa: E402

from eval.metrics import cluster_acc, nmi                            # noqa: E402
from toy_data import make_toy, make_toy_test                         # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Look: one system for every figure (reference palette of the dataviz skill, light surface; identity is also carried by labels)
# ---------------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # classes 0-4 (categorical slots 1-5)
BLUE, ORANGE, AQUA, RED = C[0], C[1], C[2], "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 170,
    "lines.linewidth": 2.0, "font.family": "DejaVu Sans"})
CLASS_NAMES = [f"class {c}" for c in range(5)]


def fr(x, nd=2) -> str:
    """A number with a fixed number of decimals."""
    return f"{x:.{nd}f}"


def save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def hwc(x, scale=1):
    a = np.asarray(x).transpose(1, 2, 0) if np.asarray(x).ndim == 3 else np.asarray(x)
    return np.kron(a, np.ones((scale, scale, 1))) if scale > 1 else a


def load_runs(suites):
    runs = []
    for s in suites:
        for d in sorted(glob.glob(os.path.join(s, "final_seed*")), key=lambda p: int(p.rsplit("seed", 1)[1])):
            fm = json.load(open(os.path.join(d, "final_metrics.json"), encoding="utf-8"))
            rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"), encoding="utf-8")]
            runs.append({"dir": d, "seed": fm["seed"], "final": fm, "rows": rows})
    runs.sort(key=lambda r: r["seed"])
    return runs


def right(m) -> bool:
    return m["clusters_ge10"] == 5 and m["acc"] > 0.999


def clopper_pearson(k: int, n: int, a: float = 0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------------------------------
# Figure 1: the task
# ---------------------------------------------------------------------------------------------------------------------
def fig_task(out: Path, seed: int = 100) -> None:
    from toy_data import background_redraw
    import torch
    x, y, _ = make_toy(seed=seed)
    x = x.numpy()
    y = y.numpy()
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    n_cols, scale, gap = 9, 5, 3
    cell = 16 * scale
    canvas = np.ones((5 * (cell + gap), n_cols * (cell + gap), 3))
    for c in range(5):
        idx = rng.choice(np.where(y == c)[0], n_cols, replace=False)
        for j, i in enumerate(idx):
            canvas[c * (cell + gap):c * (cell + gap) + cell, j * (cell + gap):j * (cell + gap) + cell] = hwc(x[i], scale)
    ax.imshow(canvas, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([(c + 0.5) * (cell + gap) - gap / 2 for c in range(5)])
    ax.set_yticklabels(CLASS_NAMES)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Five classes: a pattern of white blocks on a random background colour", pad=22)
    ax.text(0, 1.0, "", transform=ax.transAxes)
    # (b) an orbit: the same image under the transformation family T (the background is redrawn)
    ax2 = fig.add_subplot(gs[1])
    torch.manual_seed(0)
    picks = [int(np.where(y == c)[0][3]) for c in (0, 2, 4)]
    n_views = 4
    canvas2 = np.ones((len(picks) * (cell + gap), (n_views + 1) * (cell + gap) + 12, 3))
    for r, i in enumerate(picks):
        xi = torch.from_numpy(x[i:i + 1])
        canvas2[r * (cell + gap):r * (cell + gap) + cell, :cell] = hwc(x[i], scale)
        for v in range(n_views):
            xv = background_redraw(xi)[0].numpy()
            x0 = (v + 1) * (cell + gap) + 12
            canvas2[r * (cell + gap):r * (cell + gap) + cell, x0:x0 + cell] = hwc(xv, scale)
    ax2.imshow(canvas2, interpolation="nearest")
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.axvline(cell + gap + 5, color=GRID, lw=1.5)
    ax2.set_title("One orbit: the same image under $\\mathcal{T}$", pad=22)
    ax2.text(cell / 2, -5, "image", ha="center", va="bottom", color=INK2, fontsize=9)
    ax2.text(cell * 3 + 40, -5, "views (background redrawn)", ha="center", va="bottom", color=INK2, fontsize=9)
    save(fig, out / "fig1_task.png")


# ---------------------------------------------------------------------------------------------------------------------
# Figure 2: why it is hard (pixels are organised by the background, not by the class)
# ---------------------------------------------------------------------------------------------------------------------
def fig_difficulty(out: Path, seed: int) -> dict:
    """Left: in pixel space every class is a long segment, stretched by the background luminance (the nuisance).
    Right: K-means on pixels with K = 10 (our upper bound) cuts the classes along that luminance."""
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    x, y, bgid = make_toy(seed=seed)
    xs = x.numpy()
    flat = x.reshape(len(x), -1).numpy()
    y, bgid = y.numpy(), bgid.numpy()
    pca = PCA(2, random_state=0).fit_transform(flat)
    km10 = KMeans(10, n_init=10, random_state=0).fit_predict(flat)
    lum = np.array([img[:, 0, 0].mean() for img in xs])                    # luminance of the background (corner pixel)
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
    fig = plt.figure(figsize=(11.8, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    sc = ax.scatter(pca[:, 0], pca[:, 1], c=lum, cmap=cmap, s=11, linewidths=0)
    for c in range(5):
        cx, cy = np.median(pca[y == c], 0)
        ax.text(cx, cy, f" class {c}", fontsize=9, color=INK, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("1st principal component of the pixels")
    ax.set_ylabel("2nd principal component")
    ax.set_title("Pixels: each class is stretched\nalong the brightness of its background", pad=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("background brightness", color=INK2)
    cb.outline.set_visible(False)
    ax2 = fig.add_subplot(gs[1])
    gallery(ax2, xs, y, km10, "K-means on pixels, K = 10: each class\nis split into two clusters", seed=seed)
    handles = [Rectangle((0, 0), 1, 1, color=C[c]) for c in range(5)]
    fig.legend(handles, CLASS_NAMES, loc="lower right", ncol=5, fontsize=8.5, bbox_to_anchor=(0.93, -0.02), handlelength=1.1, columnspacing=0.9)
    save(fig, out / "fig2_difficulty.png")
    return {}


def pixel_kmeans_all(seeds) -> dict:
    """K-means on the pixels of the training images of every seed, with K = 5 (the true number) and K = 10 (our upper bound)."""
    from sklearn.cluster import KMeans
    res = {}
    for K in (5, 10):
        accs, nmis, nmib = [], [], []
        for s in seeds:
            x, y, bgid = make_toy(seed=s)
            km = KMeans(K, n_init=5, random_state=0).fit_predict(x.reshape(len(x), -1).numpy())
            accs.append(cluster_acc(km, y.numpy(), K))
            nmis.append(nmi(km, y.numpy()))
            nmib.append(nmi(km, bgid.numpy()))
        res[f"K{K}"] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "nmi_class": float(np.mean(nmis)),
                        "nmi_background": float(np.mean(nmib)), "seeds": len(accs), "right": int(sum(a > 0.999 for a in accs))}
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Trace of one run (clusters at the start and at the end)
# ---------------------------------------------------------------------------------------------------------------------
def extract_trace(trace_dir: str, out: Path) -> None:
    """Reads the per-epoch checkpoints of a replayed run and stores, for every epoch, the output partition and the memory
    partition (small arrays: this is what the figure needs)."""
    import torch
    from cibirl import transforms
    from cibirl.train import TrainConfig, Trainer
    import datasets
    torch.set_num_threads(1)
    d = Path(trace_dir)
    cfg = TrainConfig.from_yaml(d / "config.yml")
    data = datasets.build("toy", cfg.seed, spurious=0.0)
    bank = transforms.build(str(ROOT / "configs" / "transforms" / "toy_oracle.yml"))
    tr = Trainer(cfg, data.images, bank, device="cpu", eval_labels=data.labels)
    files = sorted((d / "ckpts").glob("ep*.pt"))
    from sklearn.decomposition import PCA
    preds, mems, pcs = [], [], []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        for k in ("encoder", "intent", "reward", "policy", "value"):
            getattr(tr, k).load_state_dict(ck[k])
            getattr(tr, k).eval()
        tr.memory.assign, tr.memory.member_m, tr.memory.counts, tr.memory.summaries = ck["assign"], ck["member_m"], ck["counts"], ck["summaries"]
        preds.append(tr.predict().numpy().astype(np.int16))
        mems.append(tr.memory.assign.numpy().astype(np.int16))
        pcs.append(PCA(2, random_state=0).fit_transform(tr.intents().numpy()).astype(np.float32))
    fm = json.load(open(d / "final_metrics.json", encoding="utf-8"))
    (out / f"trace_seed{cfg.seed}_metrics.jsonl").write_text((d / "metrics.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    m_final = np.load(d / "intents.npy")
    np.savez_compressed(out / f"trace_seed{cfg.seed}.npz", pred=np.stack(preds), memory=np.stack(mems), m_pca=np.stack(pcs),
                        m_pca_final=PCA(2, random_state=0).fit_transform(m_final).astype(np.float32), epochs=np.arange(1, len(files) + 1),
                        final=np.load(d / "preds.npy").astype(np.int16), mem_pre=np.load(d / "variants.npz")["mem_pre"].astype(np.int16),
                        epochs_run=fm["epochs_run"], seed=cfg.seed)
    print("wrote trace for seed", cfg.seed, "epochs", len(files))


def gallery(ax, images, y, part, title, rows_max=10, per_row=10, scale=2, gap=2, seed=0):
    """One column of clusters: a row per cluster (largest first), `per_row` sample images, the class composition as a bar."""
    rng = np.random.default_rng(seed)
    ids, cnt = np.unique(part, return_counts=True)
    order = ids[np.argsort(-cnt)][:rows_max]
    cell = 16 * scale
    bar_w = 7 * cell // 2
    H = rows_max * (cell + gap)
    W = per_row * (cell + gap) + 10 + bar_w
    canvas = np.ones((H, per_row * (cell + gap), 3))
    ax.set_xlim(-62, W)
    ax.set_ylim(H, -22)
    for r, k in enumerate(order):
        sel = np.where(part == k)[0]
        pick = rng.choice(sel, min(per_row, len(sel)), replace=False)
        for j, i in enumerate(pick):
            canvas[r * (cell + gap):r * (cell + gap) + cell, j * (cell + gap):j * (cell + gap) + cell] = hwc(images[i], scale)
        comp = np.bincount(y[sel], minlength=5) / len(sel)
        x0 = per_row * (cell + gap) + 10
        for c in range(5):
            if comp[c] > 0:
                ax.add_patch(Rectangle((x0, r * (cell + gap)), comp[c] * bar_w, cell, color=C[c], lw=0))
                x0 += comp[c] * bar_w
        ax.text(-6, r * (cell + gap) + cell / 2, f"{len(sel)}", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.imshow(canvas, extent=(0, canvas.shape[1], H, 0), interpolation="nearest", zorder=1)
    ax.set_xlim(-62, W)
    ax.set_ylim(H, -22)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10.5, pad=8, loc="left")
    ax.text(-6, -10, "n", fontsize=8.5, ha="right", va="center", color=MUTED)
    ax.text(per_row * (cell + gap) + 10, -10, "true classes", fontsize=8.5, va="center", color=MUTED)


def fig_trace(out: Path, seed: int, stages, name: str, headline: str) -> None:
    tz = np.load(out / f"trace_seed{seed}.npz")
    x, y, _ = make_toy(seed=seed)
    x, y = x.numpy(), y.numpy()
    fig = plt.figure(figsize=(5.2 * len(stages), 5.6))
    gs = fig.add_gridspec(2, len(stages), wspace=0.08, hspace=0.12, height_ratios=[1.0, 1.25])
    for j, (kind, ep, label) in enumerate(stages):
        part = tz["final"].astype(int) if kind == "final" else tz["pred"][ep - 1].astype(int)
        pc = tz["m_pca_final"] if kind == "final" else tz["m_pca"][ep - 1]
        acc = cluster_acc(part, y, 16) if part.max() < 16 else float("nan")
        n_cl = len(np.unique(part))
        ax0 = fig.add_subplot(gs[0, j])
        for c in range(5):
            ax0.scatter(pc[y == c, 0], pc[y == c, 1], s=7, color=C[c], linewidths=0)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.grid(False)
        plural = "s" if n_cl > 1 else ""
        ax0.set_title(f"{label}\n{n_cl} cluster{plural}, ACC {acc:.2f}", fontsize=10.5, pad=8)
        if j == 0:
            ax0.set_ylabel("intent $m$\n(first 2 components)", fontsize=9)
        gallery(fig.add_subplot(gs[1, j]), x, y, part, "", rows_max=6, seed=seed)
    fig.suptitle(headline, x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995)
    handles = [Rectangle((0, 0), 1, 1, color=C[c]) for c in range(5)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    save(fig, out / name)


def fig_trace_curves(out: Path, seed: int, name: str, headline: str) -> None:
    tz = np.load(out / f"trace_seed{seed}.npz")
    x, y, _ = make_toy(seed=seed)
    y = y.numpy()
    ep = tz["epochs"]
    n_out = [len(np.unique(p)) for p in tz["pred"]]
    n_mem = [len(np.unique(m[m >= 0])) for m in tz["memory"]]
    acc = [cluster_acc(p.astype(int), y, 16) for p in tz["pred"]]
    rows = [json.loads(l) for l in open(out / f"trace_seed{seed}_metrics.jsonl", encoding="utf-8")]
    pc = [r["part_change"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3), gridspec_kw={"wspace": 0.28})
    axs[0].step(ep, n_out, where="post", color=BLUE, label="clusters of the output")
    axs[0].step(ep, n_mem, where="post", color=ORANGE, label="occupied memory slots")
    axs[0].axhline(5, color=MUTED, lw=1, ls=":")
    axs[0].text(ep[-1], 5.2, "5 classes", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    axs[0].set_title("Number of clusters")
    axs[0].legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    axs[1].plot(ep, acc, color=BLUE)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("ACC of the output (evaluation)")
    axs[2].semilogy(np.arange(1, len(pc) + 1)[1:], np.maximum(pc[1:], 1e-4), color=BLUE)
    axs[2].axhline(0.01, color=MUTED, lw=1, ls=":")
    axs[2].text(len(pc), 0.0115, "threshold 0.01", ha="right", fontsize=8.5, color=MUTED)
    axs[2].set_title("Images changing cluster (per epoch)")
    for a in axs:
        a.set_xlabel("epoch")
        a.axvspan(0, 10, color=GRID, alpha=0.5, lw=0)
    axs[0].text(5, axs[0].get_ylim()[1] * 0.93, "reward only", ha="center", fontsize=8, color=MUTED)
    fig.suptitle(headline, x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
    save(fig, out / name)


# ---------------------------------------------------------------------------------------------------------------------
# Figure: mean training dynamics over the 200 seeds
# ---------------------------------------------------------------------------------------------------------------------
def curve_matrix(runs, key, horizon):
    """[runs, horizon]: the value at every epoch, the last value carried on after the run stopped (the final state persists)."""
    M = np.full((len(runs), horizon), np.nan)
    for i, r in enumerate(runs):
        v = np.array([row.get(key, np.nan) if not isinstance(row.get(key), dict) else np.nan for row in r["rows"]], float)
        v = v[:horizon]
        M[i, :len(v)] = v
        M[i, len(v):] = v[-1]
    return M


def band(ax, M, color, label, central="mean", lo=10, hi=90, x0=1):
    x = np.arange(x0, M.shape[1] + x0)
    c = np.nanmean(M, 0) if central == "mean" else np.nanmedian(M, 0)
    ax.fill_between(x, np.nanpercentile(M, lo, 0), np.nanpercentile(M, hi, 0), color=color, alpha=0.16, lw=0)
    ax.plot(x, c, color=color, label=label)


def fig_dynamics(runs, out: Path, horizon: int = 70) -> None:
    M = {k: curve_matrix(runs, k, horizon) for k in ("n_clusters", "nonempty_slots", "eval_acc", "eval_nmi_class", "part_change",
                                                      "twin_acc", "L_pos", "L_neg", "mem_gap", "KL", "L_cov", "eval_nmi_background")}
    stop = np.array([r["final"]["epochs_run"] for r in runs])
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.0), gridspec_kw={"hspace": 0.42, "wspace": 0.27})
    ax = axs[0, 0]
    band(ax, M["n_clusters"], BLUE, "output (argmax of the policy)", "median")
    band(ax, M["nonempty_slots"], ORANGE, "memory (occupied slots)", "median")
    ax.axhline(5, color=MUTED, lw=1, ls=":")
    ax.set_title("Number of clusters")
    ax.set_ylim(0, 10.6)
    ax.legend(fontsize=8.5, loc="center right", bbox_to_anchor=(1.0, 0.33))
    ax = axs[0, 1]
    band(ax, M["eval_acc"], BLUE, "ACC", "mean")
    band(ax, M["eval_nmi_class"], ORANGE, "NMI(cluster, class)", "mean")
    ax.set_ylim(0, 1.03)
    ax.set_title("Quality of the output")
    ax.legend(fontsize=8.5, loc="lower right")
    ax = axs[0, 2]
    pc = np.maximum(M["part_change"], 1e-4)
    x = np.arange(1, horizon + 1)
    ax.fill_between(x[1:], np.nanpercentile(pc, 10, 0)[1:], np.nanpercentile(pc, 90, 0)[1:], color=BLUE, alpha=0.16, lw=0)
    ax.semilogy(x[1:], np.nanmedian(pc, 0)[1:], color=BLUE)
    ax.axhline(0.01, color=MUTED, lw=1, ls=":")
    ax.text(horizon, 0.0112, "tolerance 0.01", ha="right", fontsize=8.5, color=MUTED)
    ax.set_ylabel("share of images (floor $10^{-4}$)", fontsize=8.5)
    ax.set_title("Images changing cluster")
    ax = axs[1, 0]
    band(ax, M["twin_acc"], BLUE, "twin_acc (sur les experts)", "mean")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Expert decisions reproduced (twin_acc)")
    ax = axs[1, 1]
    band(ax, M["L_pos"] + M["L_neg"], BLUE, "$L_{pos}+L_{neg}$", "mean")
    ax.set_title("Expert loss of the reward")
    ax = axs[1, 2]
    band(ax, M["mem_gap"], ORANGE, "memory / output gap", "mean")
    ax.set_ylim(0, None)
    ax.set_title("Memory / output gap")
    for a in axs.ravel():
        a.set_xlabel("epoch")
        a.axvspan(1, 10, color=GRID, alpha=0.5, lw=0)
        a.axvline(np.median(stop), color=INK2, lw=1, ls="--")
    axs[0, 0].text(np.median(stop) + 0.8, 0.5, f"median stop\n(epoch {np.median(stop):.0f})", fontsize=8, color=INK2, va="bottom")
    axs[0, 0].text(5.5, 9.4, "reward\nonly", ha="center", fontsize=8, color=MUTED)
    fig.suptitle(f"Training dynamics over {len(runs)} seeds: line = median (clusters, changes) or mean, band = 10th to 90th percentile",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.97)
    fig.text(0.01, 0.925, "After its stop, a run keeps its last value; grey area: reward only (the policy does not act yet).",
             fontsize=9, color=INK2)
    save(fig, out / "fig5_dynamics_200seeds.png")


# ---------------------------------------------------------------------------------------------------------------------
# Figure: results over the 200 seeds
# ---------------------------------------------------------------------------------------------------------------------
def fig_summary(runs, held, out: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.6), gridspec_kw={"hspace": 0.42, "wspace": 0.22})
    ep = np.array([r["final"]["epochs_run"] for r in runs])
    ax = axs[0, 0]
    bins = np.arange(38, 104, 4)
    ax.hist(ep, bins=bins, color=BLUE, edgecolor=SURF, linewidth=1.5)
    ax.axvline(np.median(ep), color=INK2, ls="--", lw=1)
    ax.text(np.median(ep) + 1, ax.get_ylim()[1] * 0.92, f"median {fr(np.median(ep), 1)}", fontsize=8.5, color=INK2)
    ax.set_title("Stopping epoch (convergence of the output)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("number of seeds")
    ax = axs[0, 1]
    before = np.array([r["final"]["before_cleanup"]["memory_clusters"] for r in runs])
    after = np.array([r["final"]["memory_clusters"] for r in runs])
    ks = np.arange(3, 11)
    w = 0.38
    ax.bar(ks - w / 2, [(before == k).sum() for k in ks], w * 0.92, color=ORANGE, label=f"at the stop (mean {fr(before.mean())})")
    ax.bar(ks + w / 2, [(after == k).sum() for k in ks], w * 0.92, color=BLUE, label=f"after the clean-up (mean {fr(after.mean())})")
    ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(ks)
    ax.set_title("Occupied memory slots")
    ax.set_xlabel("number of slots (5 classes)")
    ax.set_ylabel("number of seeds")
    ax.legend(fontsize=8.5)
    ax = axs[1, 0]
    names = [("train", "train\n(unlabeled)"), ("iid", "test\niid"), ("aligned", "test\naligned"), ("inverted", "test\ninverted")]
    rng = np.random.default_rng(0)
    for j, (k, lab) in enumerate(names):
        v = np.array([h[k]["acc"] for h in held])
        ax.scatter(j + rng.uniform(-0.28, 0.28, len(v)), v, s=14, color=BLUE, alpha=0.55, linewidths=0)
        ax.hlines(v.mean(), j - 0.34, j + 0.34, color=INK, lw=1.4)
        ax.text(j, 0.52, f"{(v > 0.999).sum()}/{len(v)} right", ha="center", fontsize=9, color=INK2)
        ax.text(j, 1.012, f"mean {fr(v.mean(), 3)}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n[1] for n in names])
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("ACC")
    ax.set_title("ACC per seed (one dot = one seed)")
    ax = axs[1, 1]
    below, above = [], []
    for r in runs:
        cl = r["final"]["cleanup"]
        d, t = np.array(cl["slot_distances"]), cl["threshold"]
        if t is None:
            above += list(d)
        else:
            below += list(d[d <= t])
            above += list(d[d > t])
    below, above = np.maximum(np.array(below), 1e-6), np.array(above)
    bins = np.logspace(-6, 1, 50)
    ax.hist(below, bins=bins, color=ORANGE, label=f"merged twins: {len(below)} pairs, max {fr(below.max(), 3)}")
    ax.hist(above, bins=bins, color=BLUE, label=f"distinct clusters: {len(above)} pairs, min {fr(np.sort(above)[1])}\n(+ 1 pair at {fr(np.sort(above)[0])}, seed 102)")
    ax.set_xscale("log")
    ax.set_title("Distance between slot summaries (200 seeds)")
    ax.set_xlabel("distance between two summaries $\\|c_a - c_b\\|$")
    ax.set_ylabel("number of pairs")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, out / "fig6_results_200seeds.png")


def fig_failures(runs, out: Path, seeds) -> None:
    sel = [r for r in runs if r["seed"] in seeds]
    fig, axs = plt.subplots(1, len(sel), figsize=(2.55 * len(sel), 3.1), gridspec_kw={"wspace": 0.5})
    for ax, r in zip(np.atleast_1d(axs), sel):
        x, y, _ = make_toy(seed=r["seed"])
        y = y.numpy()
        pred = np.load(os.path.join(r["dir"], "preds.npy"))
        ids = np.unique(pred)
        conf = np.array([[((pred == k) & (y == c)).sum() for c in range(5)] for k in ids]) / 300.0
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(5))
        ax.set_xticklabels(range(5), fontsize=8)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([f"cluster {i + 1}" for i in range(len(ids))], fontsize=8)
        ax.set_xlabel("class")
        for i in range(len(ids)):
            for c in range(5):
                if conf[i, c] > 0.02:
                    ax.text(c, i, f"{conf[i, c]:.0%}", ha="center", va="center", fontsize=6.5, color="white" if conf[i, c] > 0.55 else INK)
        ax.set_title(f"seed {r['seed']} ({len(ids)} clusters)", fontsize=9.5)
    save(fig, out / "fig7_failures.png")


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------
def tables(runs, held, out: Path, kmeans: dict | None) -> dict:
    res = {"n_runs": len(runs)}
    fresh = [h for h in held if h["seed"] >= 100]
    cols = ("train", "iid", "aligned", "inverted")

    def stat(rows, part, key):
        v = np.array([h[part][key] for h in rows])
        return float(v.mean()), float(v.std())
    for tag, rows in (("all", held), ("fresh", fresh)):
        block = {}
        for p in cols:
            n_right = sum(1 for h in rows if right(h[p]))
            lo, hi = clopper_pearson(len(rows) - n_right, len(rows))
            block[p] = {"right": n_right, "n": len(rows), "failure_ci95": [lo, hi]}
            for k in ("acc", "nmi_class", "nmi_background", "clusters", "open_share", "worst_group_acc"):
                block[p][k] = stat(rows, p, k)
        d = np.array([h["aligned"]["acc"] - h["inverted"]["acc"] for h in rows])           # matching recomputed on each partition
        block["delta_acc_rematched"] = [float(d.mean()), float(d.std())]
        if all("acc_fixed" in h[m] for h in rows for m in ("aligned", "inverted")):        # matching fixed on the train partition (docs 5.33)
            d = np.array([h["aligned"]["acc_fixed"] - h["inverted"]["acc_fixed"] for h in rows])
            block["delta_acc"] = [float(d.mean()), float(d.std())]
        res[tag] = block
    ep = np.array([r["final"]["epochs_run"] for r in runs])
    res["epochs"] = {"min": int(ep.min()), "median": float(np.median(ep)), "mean": float(ep.mean()), "max": int(ep.max()),
                     "converged": int(sum(r["final"]["converged"] for r in runs))}
    pre = np.array([r["final"]["before_cleanup"]["memory_clusters"] for r in runs])
    post = np.array([r["final"]["memory_clusters"] for r in runs])
    res["memory_slots"] = {"before_mean": float(pre.mean()), "after_mean": float(post.mean()), "before_range": [int(pre.min()), int(pre.max())],
                           "after_range": [int(post.min()), int(post.max())]}
    res["merges"] = {"total": int(sum(r["final"]["cleanup"]["merged_slots"] for r in runs)),
                     "across_classes": int(sum(r["final"]["cleanup"]["merges_across_classes"] for r in runs))}
    res["failures"] = [{"seed": r["seed"], "clusters": r["final"]["clusters_used"], "acc": r["final"]["acc"]} for r in runs if not right(r["final"] | {"clusters_ge10": r["final"]["clusters_used"]})]
    allbelow, allabove = [], []
    for r in runs:
        cl = r["final"]["cleanup"]
        d, t = np.array(cl["slot_distances"]), cl["threshold"]
        allabove += list(d if t is None else d[d > t])
        allbelow += [] if t is None else list(d[d <= t])
    sa = np.sort(allabove)
    res["distances"] = {"twin_max": float(max(allbelow)), "distinct_min": float(sa[0]), "distinct_second_min": float(sa[1]),
                        "n_twin": len(allbelow), "n_distinct": len(sa)}
    res["train_reproduces_preds"] = int(sum(h["train_reproduces_preds"] for h in held))
    if kmeans:
        res["kmeans_pixels"] = kmeans
    (out / "report_numbers.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", help="suite folders with the 200 training runs")
    ap.add_argument("--out", default="results/toy_final_report")
    ap.add_argument("--trace", default=None, help="folder of a replayed run with ckpts/: stores its per-epoch partitions in <out>")
    ap.add_argument("--only", nargs="*", default=None, help="figures to draw (task, difficulty, trace, dynamics, summary, failures, tables)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.trace:
        extract_trace(args.trace, out)
        if not args.suites:
            return
    want = lambda n: args.only is None or n in args.only                  # noqa: E731
    runs = load_runs(args.suites)
    held = json.load(open(out / "heldout_metrics.json", encoding="utf-8"))
    by_seed = {r["seed"]: r for r in runs}
    km = None
    if want("task"):
        fig_task(out)
    if want("difficulty"):
        fig_difficulty(out, 100)
    if want("trace"):
        for seed, stages, headline in ((20, [("epoch", 12, "Start: epoch 12"), ("epoch", 13, "Epoch 13"), ("final", None, "End: output at the stop (epoch 42)")],
                                        "Seed 20 (replay): from a single cluster to five pure clusters"),):
            if (out / f"trace_seed{seed}.npz").exists():
                fig_trace(out, seed, stages, f"fig3_clusters_seed{seed}.png", headline)
                fig_trace_curves(out, seed, f"fig4_trace_seed{seed}.png", f"Seed {seed}: evolution during training")
    if want("dynamics"):
        fig_dynamics(runs, out)
    if want("summary"):
        fig_summary(runs, held, out)
    if want("failures"):
        fig_failures(runs, out, [12, 30, 59, 94, 122, 152])
    if want("tables"):
        km = pixel_kmeans_all(sorted(by_seed))
        r = tables(runs, held, out, km)
        print(json.dumps({k: r[k] for k in ("epochs", "memory_slots", "merges", "failures", "kmeans_pixels")}, indent=1))


if __name__ == "__main__":
    main()
