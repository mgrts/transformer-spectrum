"""
Custom loss functions for training.

This module provides loss functions including:
- SGTLoss: Skewed Generalized T-distribution based loss
- CauchyLoss: Cauchy (Lorentzian) loss for heavy-tailed errors
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from scipy.special import beta


class SGTLoss(nn.Module):
    """
    Skewed Generalized T-distribution negative log-likelihood loss.

    This loss function is based on the SGT distribution, which generalizes
    the Student's t-distribution with additional skewness and shape parameters.
    It is particularly useful for regression tasks with heavy-tailed or
    asymmetric residual distributions.

    Args:
        p: Shape parameter controlling tail heaviness (default: 2.0)
        q: Shape parameter; larger values -> lighter tails (default: 2.0)
        lam: Skewness parameter in (-1, 1); 0 = symmetric (default: 0.0)
        sigma: Scale parameter; must be > 0 (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-6)

    References:
        Hansen, C., McDonald, J. B., & Newey, W. K. (2010).
        "Instrumental variables estimation with flexible distributions."
        Journal of Business & Economic Statistics.
    """

    def __init__(
        self,
        p: float = 2.0,
        q: float = 2.0,
        lam: float = 0.0,
        sigma: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # Validate parameters
        if not (p > 0):
            raise ValueError(f"p must be positive, got {p}")
        if not (q > 0):
            raise ValueError(f"q must be positive, got {q}")
        if not (-1 < lam < 1):
            raise ValueError(f"lam must be in (-1, 1), got {lam}")
        if not (sigma > 0):
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.p = p
        self.q = q
        self.lam = lam
        self.sigma = sigma
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the SGT loss.

        Args:
            y_pred: Predicted values of shape (batch, seq_len, features)
            y_true: Target values of shape (batch, seq_len, features)

        Returns:
            Scalar loss value (mean over all elements)
        """
        p = self.p
        q = self.q
        lam = self.lam
        sigma = self.sigma
        eps = self.eps

        device = y_true.device
        dtype = y_true.dtype

        # Precompute beta functions (as tensors on correct device)
        B1 = torch.tensor(beta(1.0 / p, q), dtype=dtype, device=device)
        B2 = torch.tensor(beta(2.0 / p, q - 1.0 / p), dtype=dtype, device=device)
        B3 = torch.tensor(beta(3.0 / p, q - 2.0 / p), dtype=dtype, device=device)

        # Compute v (scaling factor)
        v_numer = q ** (-1.0 / p)
        v_denom = torch.sqrt(
            (1 + 3 * lam ** 2) * (B3 / B1) - 4 * lam ** 2 * (B2 / B1) ** 2 + eps
        )
        v = v_numer / (v_denom + eps)

        sigma_t = torch.tensor(sigma, dtype=dtype, device=device)

        # Compute mode adjustment
        m = lam * v * sigma_t * (2 * (q ** (1.0 / p)) * B2 / B1)

        # Compute loss (negative log-likelihood)
        diff = y_true - y_pred + m
        scaled = torch.abs(diff / (sigma_t * v + eps)) ** p

        skew_term = (1 + lam * torch.sign(diff)) ** p
        ratio = scaled / (q * skew_term + eps)

        # Use log1p for numerical stability when ratio is small
        loss = (1 / p + q) * torch.log1p(ratio)

        return loss.mean()

    def extra_repr(self) -> str:
        return f"p={self.p}, q={self.q}, lam={self.lam}, sigma={self.sigma}"


class CauchyLoss(nn.Module):
    """
    Cauchy (Lorentzian) loss function.

    This loss is more robust to outliers than MSE, as it has heavier tails.
    The loss is: gamma * log(1 + (pred - target)^2 / gamma)

    Args:
        gamma: Scale parameter controlling outlier sensitivity (default: 1.0)
        reduction: Reduction mode ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        gamma: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        if not (gamma > 0):
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cauchy loss.

        Args:
            y_pred: Predicted values
            y_true: Target values

        Returns:
            Loss value (reduced according to self.reduction)
        """
        diffs = y_pred - y_true
        # Use log1p for numerical stability
        loss = self.gamma * torch.log1p(diffs ** 2 / self.gamma)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction='{self.reduction}'"


def get_loss_function(
    loss_type: str,
    sgt_lambda: float = 0.0,
    sgt_q: float = 2.0,
    sgt_sigma: float = 1.0,
    cauchy_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function to create loss function by name.

    Args:
        loss_type: One of 'mse', 'mae', 'cauchy', 'sgt'
        sgt_lambda: Skewness for SGT loss
        sgt_q: Shape parameter q for SGT loss
        sgt_sigma: Scale for SGT loss
        cauchy_gamma: Scale for Cauchy loss

    Returns:
        Configured loss module

    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "sgt":
        return SGTLoss(eps=1e-6, sigma=sgt_sigma, p=2.0, q=sgt_q, lam=sgt_lambda)
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "cauchy":
        return CauchyLoss(gamma=cauchy_gamma)
    else:
        raise ValueError(
            f"Unsupported loss type: '{loss_type}'. "
            f"Choose from: 'mse', 'mae', 'cauchy', 'sgt'"
        )

