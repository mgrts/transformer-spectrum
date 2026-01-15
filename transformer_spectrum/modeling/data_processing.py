"""
Data processing utilities for sequence datasets.

This module provides dataset classes and data loading utilities
for time series autoregression tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence prediction.

    Splits each sequence into input and target portions for
    autoregressive training.

    Args:
        data: Array of shape (num_sequences, sequence_length, num_features)
        input_len: Length of input sequence
        output_len: Length of target sequence (prediction horizon)

    Raises:
        ValueError: If shapes are inconsistent
    """

    def __init__(
        self,
        data: NDArray[np.floating],
        input_len: int,
        output_len: int,
    ) -> None:
        super().__init__()

        # Validate inputs
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D data (num_sequences, seq_len, features), got shape {data.shape}"
            )

        num_sequences, seq_len, num_features = data.shape

        if input_len <= 0:
            raise ValueError(f"input_len must be positive, got {input_len}")
        if output_len <= 0:
            raise ValueError(f"output_len must be positive, got {output_len}")
        if input_len + output_len > seq_len:
            raise ValueError(
                f"input_len ({input_len}) + output_len ({output_len}) = {input_len + output_len} "
                f"exceeds sequence length ({seq_len})"
            )

        self.inputs = data[:, :input_len, :].astype(np.float32)
        self.targets = data[:, input_len:input_len + output_len, :].astype(np.float32)
        self.num_features = num_features

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (input, target) tensors
        """
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx]),
        )


def create_dataloaders(
    data: NDArray[np.floating],
    input_len: int,
    output_len: int,
    batch_size: int,
    test_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from sequence data.

    Args:
        data: Array of shape (num_sequences, total_seq_len, num_features)
        input_len: Length of input sequence
        output_len: Length of target sequence
        batch_size: Training batch size
        test_split: Fraction of data for validation (0 < test_split < 1)
        seed: Random seed for train/val split
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not (0 < test_split < 1):
        raise ValueError(f"test_split must be in (0, 1), got {test_split}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total_len = input_len + output_len

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D data (num_sequences, seq_len, features), got shape {data.shape}"
        )

    if data.shape[1] < total_len:
        raise ValueError(
            f"Sequence length ({data.shape[1]}) must be >= input_len + output_len ({total_len})"
        )

    # Truncate to required length
    data = data[:, :total_len, :]

    # Split data
    train_data, val_data = train_test_split(
        data,
        test_size=test_split,
        random_state=seed,
    )

    # Create datasets
    train_dataset = SequenceDataset(train_data, input_len, output_len)
    val_dataset = SequenceDataset(val_data, input_len, output_len)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def load_dataset(path: str | Path, mmap_mode: str | None = None) -> NDArray[np.floating]:
    """
    Load a numpy dataset file.

    Args:
        path: Path to .npy file
        mmap_mode: Memory-map mode ('r', 'r+', 'w+', 'c') or None

    Returns:
        Loaded array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If loaded data is not 3D
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, mmap_mode=mmap_mode)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D data (num_sequences, seq_len, features), got shape {data.shape}"
        )

    return data

