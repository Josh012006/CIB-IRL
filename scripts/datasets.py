"""The datasets the experiment runner can build, and the dataset-specific parts of the nuisance family.

A dataset is a function `build_<name>(seed, **params) -> DataBundle`. Labels (and the optional background
id) are for EVALUATION and figures only: they never reach the model.

To add a dataset: write its builder, add it to DATASETS, give it a default config file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cibirl import transforms                      # noqa: E402
from toy_data import background_redraw, make_toy   # noqa: E402


@dataclass
class DataBundle:
    images: Tensor                    # [N, C, H, W] float in [0, 1]
    labels: Tensor                    # [N] true classes, evaluation only
    background: Tensor | None         # [N] a nuisance id (e.g. background colour), evaluation only
    info: dict = field(default_factory=dict)   # the parameters it was built with, recorded in the run


# --------------------------------------------------------------------------------------------------
# Toy: white block-shapes on a coloured background
# --------------------------------------------------------------------------------------------------
@transforms.register("toy_background_redraw")
def _toy_background_redraw():
    """Oracle T of the toy data (scripts/toy_data.py): redraw the background colour exactly."""
    return background_redraw


def build_toy(seed: int, n_classes: int = 5, n_per_class: int = 300, size: int = 16, spurious: float = 0.0) -> DataBundle:
    images, labels, background = make_toy(n_per_class=n_per_class, n_classes=n_classes, size=size,
                                          spurious=spurious, seed=seed)
    return DataBundle(images, labels, background,
                      {"n_classes": n_classes, "n_per_class": n_per_class, "size": size, "spurious": spurious})


DATASETS = {"toy": build_toy}
DEFAULT_CONFIG = {"toy": "configs/toy.yml"}      # TrainConfig file used when a run does not give one


def build(name: str, seed: int, **params) -> DataBundle:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    try:
        return DATASETS[name](seed, **params)
    except TypeError as e:
        raise ValueError(f"dataset '{name}': {e}") from None
