"""Fixed feature extractors for baseline 1 (K-Means / GMM on features). They never see a label and never train.

    pixels     the raw pixels, flattened (what README Table 2 used on the toy)
    resnet18   torchvision ResNet, ImageNet weights, global-pooled features (512-d); resnet50 gives 2048-d.
               For the real datasets (Colored MNIST, Waterbirds). torchvision is imported only here.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pixel_features(images: Tensor) -> np.ndarray:
    return images.reshape(images.shape[0], -1).cpu().numpy().astype(np.float32)


class ResNetFeatures:
    """ImageNet ResNet used as a frozen feature extractor. Images are made 3-channel, resized to `resolution` (224 by default,
    the size the network was trained at) and normalized with the ImageNet statistics."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True, device: str = "cpu", resolution: int = 224):
        try:
            import torchvision
        except ImportError as e:                                    # pragma: no cover - depends on the environment
            raise ImportError("resnet features need torchvision (pip install torchvision)") from e
        ctor = getattr(torchvision.models, name)
        weights = None
        if pretrained:
            weights = getattr(torchvision.models, f"{name.replace('resnet', 'ResNet')}_Weights").IMAGENET1K_V1
        net = ctor(weights=weights)
        net.fc = torch.nn.Identity()
        self.net, self.device, self.resolution = net.eval().to(device), torch.device(device), resolution
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: Tensor, batch_size: int = 128) -> np.ndarray:
        out = []
        for s in range(0, images.shape[0], batch_size):
            x = images[s:s + batch_size].to(self.device).float()
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            if x.shape[-1] != self.resolution:
                x = torch.nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            out.append(self.net((x - self.mean) / self.std).cpu())
        return torch.cat(out).numpy().astype(np.float32)


def make_extractor(kind: str, pretrained: bool = True, device: str = "cpu"):
    """A function images [N, C, H, W] in [0, 1] -> features [N, D] (numpy)."""
    if kind == "pixels":
        return pixel_features
    if kind in ("resnet18", "resnet50"):
        return ResNetFeatures(kind, pretrained, device)
    raise ValueError(f"unknown features '{kind}'")
