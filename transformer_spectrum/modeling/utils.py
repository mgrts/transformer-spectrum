import numpy as np
import torch

from transformer_spectrum.modeling.loss_functions import CauchyLoss, SGTLoss
from transformer_spectrum.modeling.models import TransformerWithPE, LSTM


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loss_function(loss_type: str, sgt_loss_lambda: float = 0.0, sgt_loss_q: float = 2.0,
                      sgt_loss_sigma: float = 1.0):
    loss_type = loss_type.lower()
    if loss_type == 'sgt':
        return SGTLoss(eps=0.0, sigma=sgt_loss_sigma, p=2.0, q=sgt_loss_q, lam=sgt_loss_lambda)
    elif loss_type == 'mse':
        return torch.nn.MSELoss()
    elif loss_type == 'mae':
        return torch.nn.L1Loss()
    elif loss_type == 'cauchy':
        return CauchyLoss(gamma=2.0)
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')


def get_model(model_type: str, in_dim: int, out_dim: int, embed_dim: int, num_heads: int, num_layers: int, device):
    if model_type == 'transformer':
        model = TransformerWithPE(
            in_dim=in_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            out_dim=out_dim,
        ).to(device)
    elif model_type == 'lstm':
        model = LSTM(
            input_dim=in_dim,
            hidden_dim=embed_dim,
            num_layers=num_layers,
            output_dim=out_dim,
        ).to(device)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return model


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """
        Call this after each validation step.

        Args:
            current_value (float): The current validation loss or monitored metric.

        Returns:
            bool: True if training should stop.
        """
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
