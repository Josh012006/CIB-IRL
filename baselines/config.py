"""Settings of the baselines (README Section 9.2), in one strict dataclass loaded from YAML like `cibirl.train.TrainConfig`.

    BaselineConfig.from_yaml("configs/baselines/simclr.yml", num_clusters=10, seed=3)

Only the keys a method reads matter; the others keep their defaults and are ignored. An unknown key is an error (with a
suggestion), so a typo cannot silently run another experiment.
"""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

METHODS = ("kmeans", "gmm", "simclr", "iic")
FEATURES = ("pixels", "resnet18", "resnet50")


@dataclass
class BaselineConfig:
    method: str = "kmeans"          # kmeans | gmm (baseline 1) | simclr | iic (baseline 2)
    num_clusters: int = 10          # K given to the baseline. 10 = the upper bound of our model (it does not know the number
                                    # of classes), or the true number of classes for the "known K" reference
    seed: int = 0

    # baseline 1: clustering of fixed features (also the clustering step after SimCLR, which then uses its own embedding)
    features: str = "pixels"        # pixels | resnet18 | resnet50 (torchvision, ImageNet weights)
    pretrained: bool = True         # resnet: ImageNet weights (downloaded by torchvision at first use; False = random weights, tests only)
    n_init: int = 10                # K-Means restarts
    gmm_reg: float = 1e-3           # variance floor of the diagonal GMM (a flat background has zero variance)

    # baseline 2: contrastive training (SimCLR: NT-Xent between two views; IIC: mutual information of two views' cluster
    # probabilities). The two views are two independent draws of the SAME transformation bank as our model's experts.
    embed_dim: int = 128            # size of the representation (output of the CNN) that is clustered
    proj_dim: int = 64              # SimCLR projection head
    temperature: float = 0.5        # SimCLR
    epochs: int = 100               # fixed number of epochs: there is no label-free stopping rule for these methods
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"

    # filled from the data by the runner
    in_channels: int = 3
    image_size: int = 16

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got '{self.method}'")
        if self.features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}, got '{self.features}'")
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be >= 2")
        if self.epochs < 1 or self.batch_size < 2 or self.temperature <= 0:
            raise ValueError("epochs >= 1, batch_size >= 2 and temperature > 0 are required")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaselineConfig":
        known = {f.name: f for f in fields(cls)}
        for key in d:
            if key not in known:
                hint = difflib.get_close_matches(key, known, n=1)
                raise KeyError(f"unknown baseline config key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        default = cls()
        return cls(**{k: type(getattr(default, k))(v) if not isinstance(getattr(default, k), bool) else bool(v)
                      for k, v in d.items()})

    @classmethod
    def from_yaml(cls, *paths: str | Path, **overrides: Any) -> "BaselineConfig":
        merged: dict[str, Any] = {}
        for p in paths:
            raw = yaml.safe_load(Path(p).read_text(encoding="utf-8-sig")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"{p}: a config file must be a mapping key: value")
            merged.update(raw)
        merged.update(overrides)
        return cls.from_dict(merged)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
