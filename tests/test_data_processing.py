"""
Tests for data processing utilities.
"""

import pytest
import numpy as np
import torch

from transformer_spectrum.modeling.data_processing import (
    SequenceDataset,
    create_dataloaders,
)


class TestSequenceDataset:
    """Tests for SequenceDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        data = np.random.randn(100, 300, 1).astype(np.float32)
        dataset = SequenceDataset(data, input_len=240, output_len=60)

        assert len(dataset) == 100

    def test_shapes(self):
        """Test that input and target shapes are correct."""
        n_sequences = 50
        seq_len = 300
        n_features = 3
        input_len = 200
        output_len = 100

        data = np.random.randn(n_sequences, seq_len, n_features).astype(np.float32)
        dataset = SequenceDataset(data, input_len=input_len, output_len=output_len)

        input_seq, target_seq = dataset[0]

        assert input_seq.shape == (input_len, n_features)
        assert target_seq.shape == (output_len, n_features)

    def test_dtypes(self):
        """Test that outputs are float32 tensors."""
        data = np.random.randn(10, 100, 1).astype(np.float64)  # Input as float64
        dataset = SequenceDataset(data, input_len=60, output_len=40)

        input_seq, target_seq = dataset[0]

        assert input_seq.dtype == torch.float32
        assert target_seq.dtype == torch.float32

    def test_non_overlapping(self):
        """Test that inputs and targets don't overlap."""
        data = np.arange(1000).reshape(10, 100, 1).astype(np.float32)
        input_len = 60
        output_len = 40

        dataset = SequenceDataset(data, input_len=input_len, output_len=output_len)

        input_seq, target_seq = dataset[0]

        # Input should be [0:60], target should be [60:100]
        assert input_seq[0, 0].item() == 0
        assert input_seq[-1, 0].item() == input_len - 1
        assert target_seq[0, 0].item() == input_len
        assert target_seq[-1, 0].item() == input_len + output_len - 1

    def test_indexing(self):
        """Test indexing different samples."""
        data = np.random.randn(20, 100, 2).astype(np.float32)
        dataset = SequenceDataset(data, input_len=60, output_len=40)

        # Test different indices
        for idx in [0, 5, 10, 19]:
            input_seq, target_seq = dataset[idx]

            # Check that data matches original
            np.testing.assert_allclose(
                input_seq.numpy(),
                data[idx, :60, :],
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                target_seq.numpy(),
                data[idx, 60:100, :],
                rtol=1e-5,
            )

    def test_validation_3d_input(self):
        """Test that 2D input raises error."""
        data = np.random.randn(100, 300)  # Missing feature dimension

        with pytest.raises(ValueError, match="Expected 3D"):
            SequenceDataset(data, input_len=240, output_len=60)

    def test_validation_lengths(self):
        """Test that invalid lengths raise errors."""
        data = np.random.randn(10, 100, 1)

        with pytest.raises(ValueError, match="input_len must be positive"):
            SequenceDataset(data, input_len=0, output_len=50)

        with pytest.raises(ValueError, match="output_len must be positive"):
            SequenceDataset(data, input_len=50, output_len=0)

        with pytest.raises(ValueError, match="exceeds sequence length"):
            SequenceDataset(data, input_len=60, output_len=60)  # 120 > 100


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_basic_creation(self):
        """Test basic dataloader creation."""
        data = np.random.randn(100, 300, 1)

        train_loader, val_loader = create_dataloaders(
            data=data,
            input_len=240,
            output_len=60,
            batch_size=16,
            test_split=0.2,
            seed=42,
        )

        assert train_loader is not None
        assert val_loader is not None

    def test_split_ratio(self):
        """Test that split ratio is approximately correct."""
        n_samples = 100
        test_split = 0.2
        data = np.random.randn(n_samples, 300, 1)

        train_loader, val_loader = create_dataloaders(
            data=data,
            input_len=240,
            output_len=60,
            batch_size=16,
            test_split=test_split,
            seed=42,
        )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        assert n_train + n_val == n_samples
        assert abs(n_val / n_samples - test_split) < 0.05

    def test_batch_sizes(self):
        """Test batch sizes are correct."""
        data = np.random.randn(100, 300, 1)
        batch_size = 16

        train_loader, val_loader = create_dataloaders(
            data=data,
            input_len=240,
            output_len=60,
            batch_size=batch_size,
            test_split=0.2,
            seed=42,
        )

        # Get first batch
        batch = next(iter(train_loader))
        assert batch[0].shape[0] == batch_size

    def test_reproducibility(self):
        """Test that same seed gives same split."""
        data = np.random.randn(100, 300, 1)

        train1, val1 = create_dataloaders(
            data=data, input_len=240, output_len=60,
            batch_size=16, test_split=0.2, seed=42,
        )

        train2, val2 = create_dataloaders(
            data=data, input_len=240, output_len=60,
            batch_size=16, test_split=0.2, seed=42,
        )

        # Check that datasets have same data
        np.testing.assert_allclose(
            train1.dataset.inputs,
            train2.dataset.inputs,
        )

    def test_different_seeds(self):
        """Test that different seeds give different splits."""
        data = np.random.randn(100, 300, 1)

        train1, _ = create_dataloaders(
            data=data, input_len=240, output_len=60,
            batch_size=16, test_split=0.2, seed=42,
        )

        train2, _ = create_dataloaders(
            data=data, input_len=240, output_len=60,
            batch_size=16, test_split=0.2, seed=123,
        )

        # Splits should be different
        assert not np.allclose(
            train1.dataset.inputs,
            train2.dataset.inputs,
        )

    def test_validation_test_split(self):
        """Test that invalid test_split raises error."""
        data = np.random.randn(100, 300, 1)

        with pytest.raises(ValueError, match="test_split must be in"):
            create_dataloaders(
                data=data, input_len=240, output_len=60,
                batch_size=16, test_split=0.0,
            )

        with pytest.raises(ValueError, match="test_split must be in"):
            create_dataloaders(
                data=data, input_len=240, output_len=60,
                batch_size=16, test_split=1.0,
            )

    def test_data_truncation(self):
        """Test that data is truncated to input_len + output_len."""
        data = np.random.randn(50, 1000, 1)  # Long sequences
        input_len = 200
        output_len = 100

        train_loader, _ = create_dataloaders(
            data=data, input_len=input_len, output_len=output_len,
            batch_size=16, test_split=0.2,
        )

        # Check that dataset uses truncated data
        dataset = train_loader.dataset
        assert dataset.inputs.shape[1] == input_len
        assert dataset.targets.shape[1] == output_len


class TestDataloaderIteration:
    """Tests for iterating through dataloaders."""

    def test_iterate_train(self):
        """Test iterating through training loader."""
        data = np.random.randn(100, 300, 1)

        train_loader, _ = create_dataloaders(
            data=data, input_len=240, output_len=60,
            batch_size=16, test_split=0.2,
        )

        total_samples = 0
        for inputs, targets in train_loader:
            total_samples += inputs.shape[0]

            assert inputs.ndim == 3
            assert targets.ndim == 3
            assert inputs.shape[1] == 240
            assert targets.shape[1] == 60

        assert total_samples == len(train_loader.dataset)

    def test_shuffle_train(self):
        """Test that training data is shuffled."""
        data = np.arange(1000).reshape(10, 100, 1).astype(np.float32)

        train_loader, _ = create_dataloaders(
            data=data, input_len=60, output_len=40,
            batch_size=10, test_split=0.2, seed=42,
        )

        # Get data from two epochs
        epoch1 = [batch[0].clone() for batch in train_loader]
        epoch2 = [batch[0].clone() for batch in train_loader]

        # Data order should be different (with high probability)
        # Note: This could theoretically fail with very low probability
        is_different = not all(
            torch.allclose(b1, b2) for b1, b2 in zip(epoch1, epoch2)
        )
        assert is_different or len(epoch1) == 1  # Allow single batch case

    def test_no_shuffle_val(self):
        """Test that validation data is not shuffled."""
        data = np.arange(1000).reshape(10, 100, 1).astype(np.float32)

        _, val_loader = create_dataloaders(
            data=data, input_len=60, output_len=40,
            batch_size=10, test_split=0.5, seed=42,
        )

        # Get data from two epochs
        epoch1 = [batch[0].clone() for batch in val_loader]
        epoch2 = [batch[0].clone() for batch in val_loader]

        # Should be same order
        for b1, b2 in zip(epoch1, epoch2):
            assert torch.allclose(b1, b2)
