"""Baseline 2: pure contrastive / mutual-information training, then clustering (README Section 9.2, item 2).

The question this baseline answers: do contrastive augmentations ALONE discard the background, with no reward, no policy and no
Information Bottleneck? It receives exactly what our model's experts receive: the same transformation bank, applied twice to an
image to make two views; no label; the same convolutional backbone (`cibirl.encoder.SmallCNN`).

    simclr   NT-Xent loss between the projections of two views (Chen et al. 2020); the representation (backbone output, before the
             projection head) is clustered by K-Means after training.
    iic      Invariant Information Clustering (Ji et al. 2019): a softmax head with K outputs, trained to maximize the mutual
             information between the cluster probabilities of two views; the cluster of an image is the argmax of the head.

Both are trained for a FIXED number of epochs (`BaselineConfig.epochs`): unlike our model they have no label-free stopping rule.
MoCo v3 is not implemented (README lists it with SimCLR; it differs by a momentum encoder and a queue).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cibirl.encoder import SmallCNN

from .config import BaselineConfig


def nt_xent(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    """SimCLR loss. z1[i] and z2[i] are the projections of two views of image i (the positive pair); the other 2B - 2 projections
    of the batch are the negatives."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


def iic_loss(p1: Tensor, p2: Tensor, eps: float = 1e-8) -> Tensor:
    """Negative mutual information of the cluster assignments of two views. p1, p2: [B, K] probabilities (rows sum to 1).
    The joint P[i, j] = mean over images of p1[i] p2[j] is symmetrized."""
    P = p1.t() @ p2 / p1.shape[0]
    P = ((P + P.t()) / 2).clamp(min=eps)
    Pi, Pj = P.sum(1, keepdim=True), P.sum(0, keepdim=True)
    return -(P * (P.log() - Pi.log() - Pj.log())).sum()


class ContrastiveClusterer:
    def __init__(self, cfg: BaselineConfig, bank):
        if cfg.method not in ("simclr", "iic"):
            raise ValueError(f"ContrastiveClusterer handles simclr and iic, not '{cfg.method}'")
        if bank is None:
            raise ValueError(f"{cfg.method} needs a transformation bank (the two views are two draws of it)")
        self.cfg, self.bank = cfg, bank
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.backbone = SmallCNN(cfg.in_channels, 32, cfg.embed_dim, cfg.image_size).to(self.device)
        out = cfg.proj_dim if cfg.method == "simclr" else cfg.num_clusters
        self.head = (nn.Sequential(nn.Linear(cfg.embed_dim, cfg.embed_dim), nn.ReLU(), nn.Linear(cfg.embed_dim, out))
                     if cfg.method == "simclr" else nn.Linear(cfg.embed_dim, out)).to(self.device)
        self.kmeans = None
        self.history: list[dict] = []

    def _loss(self, x: Tensor) -> Tensor:
        v1, v2 = self.bank(x), self.bank(x)                      # two independent views of every image
        h1, h2 = self.head(self.backbone(v1)), self.head(self.backbone(v2))
        if self.cfg.method == "simclr":
            return nt_xent(h1, h2, self.cfg.temperature)
        return iic_loss(F.softmax(h1, 1), F.softmax(h2, 1))

    def fit(self, images: Tensor, bank=None) -> "ContrastiveClusterer":
        cfg = self.cfg
        opt = torch.optim.Adam(list(self.backbone.parameters()) + list(self.head.parameters()), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        N = images.shape[0]
        self.backbone.train()
        for ep in range(cfg.epochs):
            perm, losses = torch.randperm(N), []
            for s in range(0, N, cfg.batch_size):
                idx = perm[s:s + cfg.batch_size]
                if len(idx) < 2:
                    continue
                loss = self._loss(images[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.history.append({"epoch": ep, "loss": float(np.mean(losses))})
        self.backbone.eval()
        if cfg.method == "simclr":
            from sklearn.cluster import KMeans
            self.kmeans = KMeans(cfg.num_clusters, n_init=cfg.n_init, random_state=cfg.seed).fit(self.embed(images))
        return self

    @torch.no_grad()
    def embed(self, images: Tensor, batch_size: int = 512) -> np.ndarray:
        """The representation (backbone output, before the projection head): [N, embed_dim]."""
        return torch.cat([self.backbone(images[s:s + batch_size].to(self.device)).cpu()
                          for s in range(0, images.shape[0], batch_size)]).numpy()

    @torch.no_grad()
    def predict(self, images: Tensor) -> np.ndarray:
        if self.cfg.method == "simclr":
            return self.kmeans.predict(self.embed(images)).astype(np.int64)
        return np.concatenate([self.head(self.backbone(images[s:s + 512].to(self.device))).argmax(1).cpu().numpy()
                               for s in range(0, images.shape[0], 512)]).astype(np.int64)
