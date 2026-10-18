"""Figures: training curves, the clusters formed by the policy, confusion matrix, intent space.

All functions take plain arrays / tensors and a path, write a PNG, and return nothing.
Labels (classes, backgrounds) are only used to COLOR or ANNOTATE: they never reach the model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # no display needed (cluster, CI)
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import sklearn

from .metrics import cluster_purity, confusion  # noqa: E402


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _hwc(img) -> np.ndarray:
    a = np.asarray(img.detach().cpu() if hasattr(img, "detach") else img, dtype=float)
    a = np.clip(np.moveaxis(a, 0, -1), 0, 1)
    return a[..., 0] if a.shape[-1] == 1 else a


# --------------------------------------------------------------------------------------------------
def plot_curves(history: list[dict], path) -> None:
    """Training diagnostics per epoch (README 5.7 and 8.3)."""
    panels = [
        ("Expert losses", ["L_pos", "L_neg"]),
        ("Twin accuracy (forced moves)", ["twin_acc"]),
        ("Intent L_psi and KL / 100", ["L_psi", "KL_scaled"]),
        ("Clusters: non-empty slots, argmax clusters, K_eff", ["nonempty_slots", "n_clusters", "K_eff"]),
        ("Churn (sampled) and argmax partition change", ["churn", "part_change", "slot_change"]),
        ("Policy entropy and m_std (collapse check)", ["pi_entropy", "m_std"]),
        ("EVALUATION ONLY: uses labels, never used to train or to stop", ["eval_acc", "eval_nmi_class", "eval_nmi_background"]),
    ]
    rows = [dict(r, KL_scaled=r.get("KL", 0.0) / 100.0) for r in history]
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    flat = axes.ravel()
    for ax, (title, keys) in zip(flat, panels):
        for k in keys:
            ax.plot(ep, [r.get(k, np.nan) for r in rows], label=k, marker="o", ms=2.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------------------------------
def plot_cluster_grid(images, pred, path, num_slots: int, per_cluster: int = 14, labels=None,
                      seed: int = 0, title: str | None = None) -> None:
    """One row per cluster formed by the policy: a random sample of the images placed in it.

    images [N,C,H,W] (CPU), pred [N] slot ids. With `labels` (display only), each row says which true
    class dominates it and how pure it is. Empty slots are not drawn.
    """
    pred = np.asarray(pred)
    rng = np.random.default_rng(seed)
    purity = cluster_purity(pred, labels, num_slots) if labels is not None else {}
    used = [k for k in range(num_slots) if (pred == k).any()]
    fig, axes = plt.subplots(len(used), per_cluster, figsize=(per_cluster * 0.62 + 2.4, len(used) * 0.75 + 0.6),
                             squeeze=False)
    for row, k in enumerate(used):
        members = np.flatnonzero(pred == k)
        pick = rng.choice(members, size=min(per_cluster, len(members)), replace=False)
        for col in range(per_cluster):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pick):
                ax.imshow(_hwc(images[int(pick[col])]), interpolation="nearest")
        text = f"slot {k}\nn={len(members)}"
        if k in purity:
            _, c, share = purity[k]
            text += f"\nclass {c}: {share:.0%}"
        axes[row, 0].text(-0.15, 0.5, text, transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=7)
    fig.suptitle(title or "Clusters formed by the policy", fontsize=11)
    _save(fig, path)


# --------------------------------------------------------------------------------------------------
def plot_confusion(pred, labels, path, num_slots: int) -> None:
    """Slots (rows) against true classes (columns): how clean is each cluster?"""
    conf = confusion(pred, labels, num_slots)
    used = np.flatnonzero(conf.sum(1) > 0)
    conf = conf[used]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * conf.shape[1], 1.0 + 0.6 * len(used)))
    im = ax.imshow(conf / conf.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if conf[i, j] / conf[i].sum() > 0.6 else "black")
    ax.set_xticks(range(conf.shape[1]))
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels([f"slot {k}" for k in used])
    ax.set_xlabel("true class")
    ax.set_title("Images per (slot, class); colour = share of the row", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, path)


# --------------------------------------------------------------------------------------------------
def pca_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def embed_2d(x, method: str = "auto", seed: int = 0) -> tuple[np.ndarray, str]:
    """2-D projection of the intents. "auto" = t-SNE when scikit-learn is installed, else PCA.
    PCA keeps global distances but can stack several groups on one point; t-SNE separates groups."""
    x = np.asarray(x, dtype=float)
    if method in ("auto", "tsne"):
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, len(x) // 4))
            return TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(x), "t-SNE"
        except ImportError:
            if method == "tsne":
                raise
    return pca_2d(x), "PCA"


def plot_embedding(m, pred, path, labels=None, background=None, max_points: int = 2000, seed: int = 0,
                   method: str = "auto") -> None:
    """The intent space m in 2-D, coloured by the policy's cluster, then by the true class and by the
    background label if given. A nuisance-free m groups by cluster / class and NOT by background."""
    m, pred = np.asarray(m), np.asarray(pred)
    idx = np.random.default_rng(seed).permutation(len(m))[:max_points]
    xy, name_2d = embed_2d(m[idx], method, seed)
    panels = [("policy cluster", pred[idx])]
    if labels is not None:
        panels.append(("true class", np.asarray(labels)[idx]))
    if background is not None:
        panels.append(("background label (nuisance)", np.asarray(background)[idx]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, (name, c) in zip(axes[0], panels):
        ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="tab10", s=8, alpha=0.7)
        ax.set_title(f"intent space m ({name_2d}), coloured by {name}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    _save(fig, path)
