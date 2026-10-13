"""Causal encoder f_phi: the only component that sees pixels (README Section 6.1).

It maps an image x to a *stochastic* code  z = mu_phi(x) + sigma_phi(x) * eps,  eps ~ N(0, I).
Everything downstream (intent, reward, policy) only sees z. The compression pressure is the
KL(p_phi(z|x) || N(0, I)) computed in losses.py; this file only produces (z, mu, logvar).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Bounds on log sigma^2. They keep exp(logvar) finite and stop the KL from exploding early on.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


class SmallCNN(nn.Module):
    """Backbone for small images (Colored MNIST, 28x28). Returns a feature vector of size `out_dim`.

    GroupNorm instead of BatchNorm on purpose: the same encoder is called on expert batches,
    learner batches and perturbed views, which have different statistics. A BatchNorm would
    mix them and behave differently in train and eval mode.
    """

    def __init__(self, in_channels: int = 3, width: int = 32, out_dim: int = 128, image_size: int = 28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GroupNorm(4, width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.GroupNorm(4, 2 * width), nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.GroupNorm(4, 2 * width), nn.ReLU(),
        )
        # Flattened size found with a dummy pass, so any image_size works.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.conv(x))


@dataclass
class EncoderOutput:
    z: Tensor        # sampled (or mean) code, the one the rest of the model uses
    mu: Tensor       # mean of p_phi(z|x)
    logvar: Tensor   # log sigma^2 of p_phi(z|x), clamped


class CausalEncoder(nn.Module):
    """Backbone + VIB head. `backbone` is any module with an `out_dim` attribute (SmallCNN, a ResNet wrapper...)."""

    def __init__(self, backbone: nn.Module, z_dim: int = 32):
        super().__init__()
        self.backbone = backbone
        feat = backbone.out_dim
        self.head_mu = nn.Linear(feat, z_dim)
        self.head_logvar = nn.Linear(feat, z_dim)
        # Start with small noise (sigma ~ 0.14): at the beginning the utility terms must be able to
        # shape z before the KL term (beta) starts to erase bits.
        nn.init.constant_(self.head_logvar.bias, -4.0)
        nn.init.normal_(self.head_logvar.weight, std=1e-3)
        self.z_dim = z_dim

    def forward(self, x: Tensor, sample: bool = True) -> EncoderOutput:
        h = self.backbone(x)
        mu = self.head_mu(h)
        logvar = self.head_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        if sample:                                   # training: reparametrization trick
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:                                        # inference / memory recompute: deterministic code
            z = mu
        return EncoderOutput(z=z, mu=mu, logvar=logvar)


def build_encoder(kind: str = "small_cnn", z_dim: int = 32, in_channels: int = 3, image_size: int = 28,
                  backbone: nn.Module | None = None) -> CausalEncoder:
    """Factory. For Waterbirds, pass a ResNet wrapper through `backbone` (must expose `out_dim`)."""
    if backbone is not None:
        return CausalEncoder(backbone, z_dim)
    if kind == "small_cnn":
        return CausalEncoder(SmallCNN(in_channels=in_channels, image_size=image_size), z_dim)
    raise ValueError(f"unknown encoder kind: {kind}")
