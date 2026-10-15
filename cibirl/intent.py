"""Intent predictor q_psi(m | z) (README Section 6.2).

It turns a code z into a Gaussian over an "intent" m: what the image is about, in a space where
clusters are compared. The memory stores the *average intent* of the members of each cluster, so
the reward and the policy always compare m with the summaries c_k.

psi is trained ONLY by L_psi (orbit consistency, see losses.py). The reward never sends it gradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0


class BatchWhiten(nn.Module):
    """Decorrelates AND standardizes the dimensions: x -> (x - mean) L^{-T}, with L the Cholesky factor of the batch
    covariance (+ eps I). In train mode the statistics of the batch are used (and the running ones updated); otherwise,
    and in `apply_running`, the running ones. A BatchNorm without scale only sets each variance to 1, so every
    dimension may be a copy of the same variable; this layer makes that impossible (the covariance of the output is I).
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-4):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_cov", torch.eye(dim))

    @staticmethod
    def _whiten(x: Tensor, mean: Tensor, cov: Tensor, eps: float) -> Tensor:
        chol = torch.linalg.cholesky(cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype))
        return torch.linalg.solve_triangular(chol, (x - mean).T, upper=False).T

    def apply_running(self, x: Tensor) -> Tensor:
        return self._whiten(x, self.running_mean, self.running_cov, self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return self.apply_running(x)
        mean = x.mean(0)
        xc = x - mean
        cov = xc.T @ xc / max(x.shape[0] - 1, 1)
        with torch.no_grad():
            self.running_mean.lerp_(mean.detach(), self.momentum)
            self.running_cov.lerp_(cov.detach(), self.momentum)
        return self._whiten(x, mean, cov, self.eps)


class IntentPredictor(nn.Module):
    def __init__(self, z_dim: int, m_dim: int = 16, hidden: int = 128, standardize: bool = True, whiten: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head_mu = nn.Linear(hidden, m_dim)
        self.head_logvar = nn.Linear(hidden, m_dim)
        # Implementation choice (NOT in the README, see the note below): a BatchNorm without learnable
        # scale on the mean. It forces the intents of a batch to keep unit variance across images.
        if standardize and whiten:
            self.norm = BatchWhiten(m_dim)
        else:
            self.norm = nn.BatchNorm1d(m_dim, affine=False) if standardize else nn.Identity()
        self.m_dim = m_dim

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mu_psi(z), log sigma_psi^2(z)). z must be 2-D: [N, z_dim].

        In train mode the standardization uses the statistics of the batch (and updates the running
        ones). This is the forward used by L_psi.
        """
        h = self.trunk(z)
        mu = self.norm(self.head_mu(h))
        logvar = self.head_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def mean(self, z: Tensor) -> Tensor:
        """The intent m = mu_psi(z) used as STATE: arriving image, memory content, inference.

        It always uses the running statistics, whatever the module mode. So a given z gets the same
        m in an expert batch, a learner batch, a memory recompute and at test time; otherwise the
        intents stored in the memory and the intents that arrive would not be comparable.
        """
        mu = self.head_mu(self.trunk(z))
        if isinstance(self.norm, nn.BatchNorm1d):
            mu = F.batch_norm(mu, self.norm.running_mean, self.norm.running_var, training=False, eps=self.norm.eps)
        elif isinstance(self.norm, BatchWhiten):
            mu = self.norm.apply_running(mu)
        return mu

# Note on `standardize` (collapse of m).
# L_psi asks that all the views of an orbit give the same m, and its target is the mean of the
# model's OWN outputs over the orbit. A constant m (or an output that shrinks to 0) gives L_psi ~ 0
# without having to ignore any nuisance, and since the reward does not train psi nothing pushes
# back. The cluster summaries c_k would then be all equal and the comparison with the clusters would
# carry no information. The unit-variance constraint over the batch makes this shortcut impossible:
# to reduce L_psi the model has to be consistent across views *while* keeping the images apart.
# Switch it off (standardize=False) to follow the README to the letter and to run the ablation.
