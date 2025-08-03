from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, input_len: int, output_len: int):
        super().__init__()
        self.inputs = data[:, :input_len, :]
        self.targets = data[:, input_len:input_len + output_len, :]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def create_dataloaders(
    data: np.ndarray,
    input_len: int,
    output_len: int,
    batch_size: int,
    test_split: float,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    total_len = input_len + output_len
    data = data[:, :total_len, :]

    train_data, val_data = train_test_split(data, test_size=test_split, random_state=seed)

    train_loader = DataLoader(SequenceDataset(train_data, input_len, output_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_data, input_len, output_len), batch_size=batch_size)

    return train_loader, val_loader
