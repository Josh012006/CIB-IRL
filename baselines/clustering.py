"""Baseline 1: geometric / statistical clustering of fixed features (README Section 9.2, item 1).

K-Means or a diagonal GMM on pixels or on ImageNet-ResNet features. Nothing is trained and no label or transformation is used.
`fit` learns the centres on the training images; `predict` assigns ANY images (the test partitions) to the nearest centre /
most probable component, so the baseline is evaluated on the same held-out partitions as our model.
"""
from __future__ import annotations

import numpy as np
import sklearn
from torch import Tensor

from .config import BaselineConfig
from .features import make_extractor


class FeatureClusterer:
    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.extract = make_extractor(cfg.features, cfg.pretrained, cfg.device)
        self.model = None
        self.history: list[dict] = []

    def fit(self, images: Tensor, bank=None) -> "FeatureClusterer":
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        f = self.extract(images)
        if self.cfg.method == "kmeans":
            self.model = KMeans(self.cfg.num_clusters, n_init=self.cfg.n_init, random_state=self.cfg.seed).fit(f)
        elif self.cfg.method == "gmm":
            self.model = GaussianMixture(self.cfg.num_clusters, covariance_type="diag", reg_covar=self.cfg.gmm_reg,
                                         n_init=1, random_state=self.cfg.seed).fit(f)
        else:
            raise ValueError(f"FeatureClusterer handles kmeans and gmm, not '{self.cfg.method}'")
        return self

    def predict(self, images: Tensor) -> np.ndarray:
        return self.model.predict(self.extract(images)).astype(np.int64)
