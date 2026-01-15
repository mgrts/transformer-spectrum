"""
Tests for loss functions.
"""

import pytest
import torch
import numpy as np

from transformer_spectrum.modeling.loss_functions import (
    SGTLoss,
    CauchyLoss,
    get_loss_function,
)


class TestSGTLoss:
    """Tests for SGT loss function."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = SGTLoss()
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)

        loss = loss_fn(y_pred, y_true)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert torch.isfinite(loss)

    def test_zero_error_small_loss(self):
        """Test that zero prediction error gives small loss."""
        loss_fn = SGTLoss(sigma=1.0, q=2.0)
        y = torch.randn(8, 10, 1)

        loss = loss_fn(y, y)  # Same prediction and target

        # With zero error, loss should be close to zero (but not exactly due to mode adjustment)
        assert loss.item() < 1.0

    def test_larger_error_larger_loss(self):
        """Test that larger errors give larger losses."""
        loss_fn = SGTLoss()
        y_true = torch.zeros(8, 10, 1)

        loss_small = loss_fn(torch.ones(8, 10, 1) * 0.1, y_true)
        loss_large = loss_fn(torch.ones(8, 10, 1) * 10.0, y_true)

        assert loss_large > loss_small

    def test_different_q_values(self):
        """Test behavior with different q (tail heaviness) values."""
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)

        # Light tails (close to Gaussian)
        loss_fn_light = SGTLoss(q=100.0, sigma=0.707)
        loss_light = loss_fn_light(y_pred, y_true)

        # Heavy tails
        loss_fn_heavy = SGTLoss(q=1.001, sigma=15.0)
        loss_heavy = loss_fn_heavy(y_pred, y_true)

        # Both should give valid losses
        assert torch.isfinite(loss_light)
        assert torch.isfinite(loss_heavy)

    def test_skewness_parameter(self):
        """Test with different skewness values."""
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)

        for lam in [-0.5, 0.0, 0.5]:
            loss_fn = SGTLoss(lam=lam)
            loss = loss_fn(y_pred, y_true)
            assert torch.isfinite(loss)

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            SGTLoss(sigma=-1.0)

        with pytest.raises(ValueError, match="lam must be in"):
            SGTLoss(lam=1.5)

        with pytest.raises(ValueError, match="p must be positive"):
            SGTLoss(p=-1.0)

    def test_device_handling(self):
        """Test that loss works on different devices."""
        loss_fn = SGTLoss()

        # CPU
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)
        loss_cpu = loss_fn(y_pred, y_true)
        assert loss_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()
            loss_cuda = loss_fn(y_pred, y_true)
            assert loss_cuda.device.type == "cuda"

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = SGTLoss()
        y_pred = torch.randn(8, 10, 1, requires_grad=True)
        y_true = torch.randn(8, 10, 1)

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.isfinite(y_pred.grad).all()


class TestCauchyLoss:
    """Tests for Cauchy loss function."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = CauchyLoss()
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)

        loss = loss_fn(y_pred, y_true)

        assert loss.ndim == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_zero_error(self):
        """Test that zero error gives zero loss."""
        loss_fn = CauchyLoss()
        y = torch.randn(8, 10, 1)

        loss = loss_fn(y, y)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        y_pred = torch.randn(8, 10, 1)
        y_true = torch.randn(8, 10, 1)

        loss_mean = CauchyLoss(reduction="mean")(y_pred, y_true)
        loss_sum = CauchyLoss(reduction="sum")(y_pred, y_true)
        loss_none = CauchyLoss(reduction="none")(y_pred, y_true)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == y_pred.shape

    def test_robustness_to_outliers(self):
        """Test that Cauchy loss is robust to outliers compared to MSE."""
        y_true = torch.zeros(10)
        y_pred_normal = torch.randn(10) * 0.1
        y_pred_outlier = y_pred_normal.clone()
        y_pred_outlier[0] = 100.0  # Large outlier

        cauchy_loss = CauchyLoss()
        mse_loss = torch.nn.MSELoss()

        # Compute losses
        cauchy_normal = cauchy_loss(y_pred_normal, y_true)
        cauchy_outlier = cauchy_loss(y_pred_outlier, y_true)

        mse_normal = mse_loss(y_pred_normal, y_true)
        mse_outlier = mse_loss(y_pred_outlier, y_true)

        # Cauchy should be less affected by outliers (relative increase is smaller)
        cauchy_ratio = cauchy_outlier / cauchy_normal
        mse_ratio = mse_outlier / mse_normal

        assert cauchy_ratio < mse_ratio

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            CauchyLoss(gamma=-1.0)


class TestGetLossFunction:
    """Tests for the loss function factory."""

    def test_mse(self):
        """Test MSE loss creation."""
        loss_fn = get_loss_function("mse")
        assert isinstance(loss_fn, torch.nn.MSELoss)

    def test_mae(self):
        """Test MAE loss creation."""
        loss_fn = get_loss_function("mae")
        assert isinstance(loss_fn, torch.nn.L1Loss)

    def test_cauchy(self):
        """Test Cauchy loss creation."""
        loss_fn = get_loss_function("cauchy", cauchy_gamma=3.0)
        assert isinstance(loss_fn, CauchyLoss)
        assert loss_fn.gamma == 3.0

    def test_sgt(self):
        """Test SGT loss creation."""
        loss_fn = get_loss_function("sgt", sgt_lambda=0.1, sgt_q=5.0, sgt_sigma=2.0)
        assert isinstance(loss_fn, SGTLoss)
        assert loss_fn.lam == 0.1
        assert loss_fn.q == 5.0
        assert loss_fn.sigma == 2.0

    def test_case_insensitive(self):
        """Test that loss type is case-insensitive."""
        loss_fn_lower = get_loss_function("mse")
        loss_fn_upper = get_loss_function("MSE")
        loss_fn_mixed = get_loss_function("MsE")

        assert type(loss_fn_lower) == type(loss_fn_upper) == type(loss_fn_mixed)

    def test_invalid_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unsupported loss type"):
            get_loss_function("invalid")


class TestLossNumericalStability:
    """Tests for numerical stability of loss functions."""

    def test_sgt_large_values(self):
        """Test SGT loss with large values."""
        loss_fn = SGTLoss()
        y_pred = torch.randn(8, 10, 1) * 1000
        y_true = torch.randn(8, 10, 1) * 1000

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss)

    def test_sgt_small_values(self):
        """Test SGT loss with small values."""
        loss_fn = SGTLoss()
        y_pred = torch.randn(8, 10, 1) * 1e-6
        y_true = torch.randn(8, 10, 1) * 1e-6

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss)

    def test_cauchy_large_values(self):
        """Test Cauchy loss with large values."""
        loss_fn = CauchyLoss()
        y_pred = torch.randn(8, 10, 1) * 1000
        y_true = torch.randn(8, 10, 1) * 1000

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss)
