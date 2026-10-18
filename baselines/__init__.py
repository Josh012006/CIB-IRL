"""The baselines of README Section 9.2 that run outside the CIB-IRL trainer, behind one interface.

    from baselines import build_clusterer, BaselineConfig
    clusterer = build_clusterer(BaselineConfig(method="simclr", num_clusters=10), bank).fit(train_images)
    clusterer.predict(any_images)          # numpy array of cluster ids; labels are never used

    1  kmeans, gmm   fixed features (pixels or ImageNet ResNet) + K-Means / diagonal GMM          baselines/clustering.py
    2  simclr, iic   contrastive / mutual-information training with OUR transformation bank,
                     then clustering                                                              baselines/contrastive.py
    3  standard multi-intention IRL: it is OUR trainer without the causal Information Bottleneck
       (`beta = 0`, `lam = 0`, configs/baselines/irl_noib.yml), so it runs through scripts/run_experiment.py and shares every
       other line of code with the model; it is not re-implemented here.

`scripts/run_baseline.py` trains one baseline on one benchmark and scores it on the train and test partitions with
`eval/protocol.py`, exactly as `scripts/eval_heldout.py` scores our model.
"""
from .clustering import FeatureClusterer
from .config import BaselineConfig, METHODS
from .contrastive import ContrastiveClusterer


def build_clusterer(cfg: BaselineConfig, bank=None):
    """The baseline described by `cfg`. `bank` (the transformation family T) is only read by simclr and iic."""
    if cfg.method in ("kmeans", "gmm"):
        return FeatureClusterer(cfg)
    return ContrastiveClusterer(cfg, bank)


__all__ = ["BaselineConfig", "METHODS", "FeatureClusterer", "ContrastiveClusterer", "build_clusterer"]
