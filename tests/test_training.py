"""
Smoke tests for training pipeline.

These tests verify that the training pipeline works end-to-end
on tiny synthetic data. They are not meant to test model quality.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from transformer_spectrum.modeling.models import TransformerWithPE, LSTM
from transformer_spectrum.modeling.loss_functions import get_loss_function
from transformer_spectrum.modeling.data_processing import create_dataloaders
from transformer_spectrum.modeling.utils import get_model, EarlyStopping
from transformer_spectrum.settings import set_seed


class TestModels:
    """Tests for model architectures."""

    def test_transformer_forward(self):
        """Test transformer forward pass."""
        batch_size = 4
        src_len = 60
        tgt_len = 20
        in_dim = 1
        out_dim = 1

        model = TransformerWithPE(
            in_dim=in_dim,
            out_dim=out_dim,
            embed_dim=32,
            num_heads=2,
            num_layers=2,
        )

        src = torch.randn(batch_size, src_len, in_dim)
        tgt = torch.randn(batch_size, tgt_len, out_dim)

        output = model(src, tgt)

        assert output.shape == (batch_size, tgt_len, out_dim)

    def test_transformer_infer(self):
        """Test transformer inference (autoregressive)."""
        batch_size = 4
        src_len = 60
        tgt_len = 20

        model = TransformerWithPE(
            in_dim=1,
            out_dim=1,
            embed_dim=32,
            num_heads=2,
            num_layers=2,
        )

        src = torch.randn(batch_size, src_len, 1)

        output = model.infer(src, tgt_len=tgt_len)

        assert output.shape == (batch_size, tgt_len, 1)

    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        batch_size = 4
        src_len = 60
        tgt_len = 20

        model = LSTM(
            input_dim=1,
            hidden_dim=32,
            num_layers=2,
            output_dim=1,
        )

        src = torch.randn(batch_size, src_len, 1)
        tgt = torch.randn(batch_size, tgt_len, 1)

        output = model(src, tgt)

        assert output.shape == (batch_size, tgt_len, 1)

    def test_lstm_infer(self):
        """Test LSTM inference."""
        batch_size = 4
        src_len = 60
        tgt_len = 20

        model = LSTM(
            input_dim=1,
            hidden_dim=32,
            num_layers=2,
            output_dim=1,
        )

        src = torch.randn(batch_size, src_len, 1)

        output = model.infer(src, tgt_len=tgt_len)

        assert output.shape == (batch_size, tgt_len, 1)

    def test_get_model_factory(self):
        """Test model factory function."""
        model_transformer = get_model(
            model_type="transformer",
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
            device="cpu",
        )
        assert isinstance(model_transformer, TransformerWithPE)

        model_lstm = get_model(
            model_type="lstm",
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
            device="cpu",
        )
        assert isinstance(model_lstm, LSTM)


class TestTrainingStep:
    """Tests for single training steps."""

    @pytest.fixture
    def tiny_data(self):
        """Create tiny synthetic dataset."""
        set_seed(42)
        # 20 sequences, 100 timesteps, 1 feature
        data = np.random.randn(20, 100, 1).astype(np.float32)
        return data

    @pytest.fixture
    def tiny_loaders(self, tiny_data):
        """Create dataloaders from tiny data."""
        return create_dataloaders(
            data=tiny_data,
            input_len=60,
            output_len=40,
            batch_size=8,
            test_split=0.2,
            seed=42,
        )

    def test_single_training_step(self, tiny_loaders):
        """Test a single forward-backward pass."""
        train_loader, _ = tiny_loaders

        model = TransformerWithPE(
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Get one batch
        src, tgt = next(iter(train_loader))

        # Forward pass
        model.train()
        output = model(src, tgt)
        loss = criterion(output, tgt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is finite
        assert torch.isfinite(loss)

        # Verify gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_decreases_loss(self, tiny_loaders):
        """Test that training decreases loss."""
        train_loader, _ = tiny_loaders

        set_seed(42)
        model = TransformerWithPE(
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []

        # Train for a few steps
        for epoch in range(5):
            epoch_loss = 0.0
            model.train()
            for src, tgt in train_loader:
                output = model(src, tgt)
                loss = criterion(output, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(train_loader))

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_different_loss_functions(self, tiny_loaders):
        """Test training with different loss functions."""
        train_loader, _ = tiny_loaders

        for loss_type in ["mse", "mae", "cauchy", "sgt"]:
            set_seed(42)
            model = TransformerWithPE(
                in_dim=1, out_dim=1,
                embed_dim=16, num_heads=2, num_layers=1,
            )

            criterion = get_loss_function(loss_type)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # One training step
            src, tgt = next(iter(train_loader))
            model.train()
            output = model(src, tgt)
            loss = criterion(output, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss), f"Loss not finite for {loss_type}"


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience epochs."""
        es = EarlyStopping(patience=3)

        # Improving losses
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.8)

        # Stagnating losses
        assert not es.step(0.85)  # Counter = 1
        assert not es.step(0.85)  # Counter = 2
        assert es.step(0.85)      # Counter = 3, stop

    def test_early_stopping_resets(self):
        """Test that counter resets on improvement."""
        es = EarlyStopping(patience=3)

        es.step(1.0)
        es.step(1.1)  # Counter = 1
        es.step(1.1)  # Counter = 2
        es.step(0.5)  # Improvement, counter = 0

        assert es.counter == 0
        assert not es.early_stop


class TestModelSaveLoad:
    """Tests for model serialization."""

    def test_save_and_load(self):
        """Test saving and loading model weights."""
        model = TransformerWithPE(
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
        )

        # Get initial predictions
        src = torch.randn(2, 30, 1)
        tgt = torch.randn(2, 10, 1)

        model.eval()
        with torch.no_grad():
            pred_before = model(src, tgt).clone()

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            save_path = f.name

        # Create new model and load weights
        model2 = TransformerWithPE(
            in_dim=1, out_dim=1,
            embed_dim=32, num_heads=2, num_layers=2,
        )
        model2.load_state_dict(torch.load(save_path), strict=False)

        # Predictions should match
        model2.eval()
        with torch.no_grad():
            pred_after = model2(src, tgt)

        assert torch.allclose(pred_before, pred_after)

        # Cleanup
        Path(save_path).unlink()


class TestDeterminism:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self):
        """Test that setting seed gives reproducible results."""
        def run_training():
            set_seed(42)

            model = TransformerWithPE(
                in_dim=1, out_dim=1,
                embed_dim=16, num_heads=2, num_layers=1,
            )

            data = np.random.randn(10, 50, 1).astype(np.float32)
            train_loader, _ = create_dataloaders(
                data=data, input_len=30, output_len=20,
                batch_size=4, test_split=0.2, seed=42,
            )

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train for one epoch
            total_loss = 0.0
            model.train()
            for src, tgt in train_loader:
                output = model(src, tgt)
                loss = criterion(output, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            return total_loss, list(model.parameters())[0].clone()

        # Run twice
        loss1, params1 = run_training()
        loss2, params2 = run_training()

        # Should be identical
        assert abs(loss1 - loss2) < 1e-6
        assert torch.allclose(params1, params2)
