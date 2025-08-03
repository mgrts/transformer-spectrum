import numpy as np
import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
