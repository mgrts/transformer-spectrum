"""
Utility functions for model training.

Includes seed setting, model/loss factories, and training utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from transformer_spectrum.modeling.loss_functions import get_loss_function, CauchyLoss, SGTLoss
from transformer_spectrum.modeling.models import TransformerWithPE, LSTM

if TYPE_CHECKING:
    from transformer_spectrum.settings import ExperimentConfig, LossConfig, ModelConfig


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    This function is deprecated. Use `transformer_spectrum.settings.set_seed` instead.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic CUDA operations
    """
    from transformer_spectrum.settings import set_seed as _set_seed
    _set_seed(seed, deterministic)


def get_model(
    model_type: str,
    in_dim: int,
    out_dim: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    device: torch.device | str,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function to create a model by type.

    Args:
        model_type: Either 'transformer' or 'lstm'
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        embed_dim: Embedding/hidden dimension
        num_heads: Number of attention heads (transformer only)
        num_layers: Number of layers
        device: Device to place model on
        dropout: Dropout rate

    Returns:
        Configured model on specified device

    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower().strip()

    if model_type == "transformer":
        model = TransformerWithPE(
            in_dim=in_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            out_dim=out_dim,
            dropout=dropout,
        )
    elif model_type == "lstm":
        model = LSTM(
            input_dim=in_dim,
            hidden_dim=embed_dim,
            num_layers=num_layers,
            output_dim=out_dim,
        )
    else:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Choose from: 'transformer', 'lstm'"
        )

    return model.to(device)


def get_model_from_config(config: ModelConfig, in_dim: int, out_dim: int, device: torch.device) -> nn.Module:
    """
    Create model from a ModelConfig object.

    Args:
        config: Model configuration
        in_dim: Input dimension
        out_dim: Output dimension
        device: Device to place model on

    Returns:
        Configured model
    """
    return get_model(
        model_type=config.type.value,
        in_dim=in_dim,
        out_dim=out_dim,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        device=device,
        dropout=config.dropout,
    )


def get_loss_from_config(config: LossConfig) -> nn.Module:
    """
    Create loss function from a LossConfig object.

    Args:
        config: Loss configuration

    Returns:
        Configured loss module
    """
    return get_loss_function(
        loss_type=config.type.value,
        sgt_lambda=config.sgt.lam,
        sgt_q=config.sgt.q,
        sgt_sigma=config.sgt.sigma,
        cauchy_gamma=config.cauchy_gamma,
    )


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience: How many epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' if lower is better, 'max' if higher is better
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode.lower()

        if self.mode == "min":
            self.best_value = float("inf")
            self._is_better = lambda new, best: new < best - self.min_delta
        elif self.mode == "max":
            self.best_value = float("-inf")
            self._is_better = lambda new, best: new > best + self.min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.counter = 0
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: The current validation metric

        Returns:
            True if training should stop
        """
        if self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"min_delta={self.min_delta}, mode='{self.mode}')"
        )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device: 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

