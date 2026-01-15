"""
Tests for spectral metrics.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from transformer_spectrum.metrics.spectral_metrics import (
    spectral_entropy,
    alpha_exponent,
    stable_rank,
    get_spectral_metrics,
    _esd,
    _hill_alpha,
    _ks_alpha,
)


class TestSpectralEntropy:
    """Tests for spectral entropy computation."""

    def test_identity_matrix(self):
        """Test entropy of identity matrix (uniform singular values)."""
        W = np.eye(10)
        entropy = spectral_entropy(W)

        # All singular values are 1, so entropy should be log(10)
        expected = np.log(10)  # Maximum entropy for 10 singular values
        assert_allclose(entropy, expected, rtol=1e-5)

    def test_rank_one_matrix(self):
        """Test entropy of rank-1 matrix (minimal entropy)."""
        # Rank-1 matrix has one non-zero singular value
        W = np.outer(np.ones(10), np.ones(10))
        entropy = spectral_entropy(W)

        # Only one non-zero singular value -> entropy = 0
        assert_allclose(entropy, 0.0, atol=1e-10)

    def test_diagonal_matrix(self):
        """Test entropy of diagonal matrix with known singular values."""
        s = np.array([1.0, 0.5, 0.25, 0.125])
        W = np.diag(s)

        entropy = spectral_entropy(W)

        # Compute expected entropy
        s2 = s ** 2
        p = s2 / s2.sum()
        expected = -np.sum(p * np.log(p))

        assert_allclose(entropy, expected, rtol=1e-5)

    def test_empty_matrix(self):
        """Test handling of edge cases."""
        W = np.array([[]])
        entropy = spectral_entropy(W)
        assert np.isnan(entropy)

    def test_rectangular_matrix(self):
        """Test with non-square matrix."""
        W = np.random.randn(10, 20)
        entropy = spectral_entropy(W)

        assert np.isfinite(entropy)
        assert entropy >= 0

    def test_1d_input(self):
        """Test that 1D input returns nan."""
        W = np.array([1, 2, 3])
        entropy = spectral_entropy(W)
        assert np.isnan(entropy)


class TestAlphaExponent:
    """Tests for alpha exponent (power-law decay)."""

    def test_power_law_singular_values(self):
        """Test with matrix having power-law singular values."""
        # Create matrix with known power-law singular value decay
        n = 50
        alpha_true = 1.5
        s = np.arange(1, n + 1) ** (-alpha_true)

        # Construct matrix with these singular values
        U = np.eye(n)
        V = np.eye(n)
        W = U @ np.diag(s) @ V.T

        alpha_est = alpha_exponent(W)

        # Should recover approximately the true alpha
        # (not exact due to fitting range and edge effects)
        assert abs(alpha_est - alpha_true) < 0.5

    def test_uniform_singular_values(self):
        """Test with uniform singular values (alpha ~ 0)."""
        W = np.eye(50)
        alpha = alpha_exponent(W)

        # Uniform SVs should give small alpha
        assert alpha < 0.5

    def test_small_matrix(self):
        """Test with matrix too small for fitting."""
        W = np.random.randn(4, 4)
        alpha = alpha_exponent(W)

        # Should return nan for small matrices
        assert np.isnan(alpha)

    def test_custom_fit_range(self):
        """Test with custom fit range."""
        W = np.random.randn(50, 50)

        alpha_default = alpha_exponent(W)
        alpha_custom = alpha_exponent(W, fit_range=(5, 30))

        # Both should be finite
        assert np.isfinite(alpha_default)
        assert np.isfinite(alpha_custom)


class TestStableRank:
    """Tests for stable rank computation."""

    def test_identity_matrix(self):
        """Test stable rank of identity matrix."""
        n = 10
        W = np.eye(n)

        sr = stable_rank(W)

        # Identity matrix has stable rank = n
        assert_allclose(sr, n, rtol=1e-5)

    def test_rank_one_matrix(self):
        """Test stable rank of rank-1 matrix."""
        W = np.outer(np.ones(10), np.ones(10))

        sr = stable_rank(W)

        # Rank-1 matrix has stable rank = 1
        assert_allclose(sr, 1.0, rtol=1e-5)

    def test_stable_rank_bounds(self):
        """Test that stable rank is between 1 and matrix dimension."""
        for _ in range(10):
            n = 20
            W = np.random.randn(n, n)

            sr = stable_rank(W)

            assert 1.0 <= sr <= n

    def test_rectangular_matrix(self):
        """Test stable rank of rectangular matrix."""
        W = np.random.randn(10, 30)

        sr = stable_rank(W)

        # Stable rank <= min(m, n)
        assert 1.0 <= sr <= 10


class TestESD:
    """Tests for eigenspectral density computation."""

    def test_esd_positive(self):
        """Test that ESD values are non-negative."""
        W = np.random.randn(20, 20)
        esd = _esd(W)

        assert np.all(esd >= 0)

    def test_esd_count(self):
        """Test that ESD has correct number of values."""
        W = np.random.randn(10, 20)
        esd = _esd(W)

        # Should have min(m, n) eigenvalues
        assert len(esd) == 10


class TestHillAlpha:
    """Tests for Hill estimator."""

    def test_known_distribution(self):
        """Test Hill estimator on samples from known power-law."""
        np.random.seed(42)

        # Generate power-law samples
        alpha_true = 2.5
        n = 1000
        u = np.random.uniform(0, 1, n)
        x = (1 - u) ** (-1 / (alpha_true - 1))  # Inverse CDF method

        alpha_est = _hill_alpha(x)

        # Should be close to true alpha
        assert abs(alpha_est - alpha_true) < 0.5

    def test_small_sample(self):
        """Test that small samples return nan."""
        x = np.array([1, 2, 3])
        alpha = _hill_alpha(x)

        assert np.isnan(alpha)

    def test_custom_k(self):
        """Test with custom k parameter."""
        x = np.random.pareto(2.0, 100) + 1

        alpha_k10 = _hill_alpha(x, k=10)
        alpha_k20 = _hill_alpha(x, k=20)

        # Both should be valid
        assert np.isfinite(alpha_k10)
        assert np.isfinite(alpha_k20)


class TestKSAlpha:
    """Tests for KS-based alpha estimator."""

    def test_known_distribution(self):
        """Test KS estimator on samples from known power-law."""
        np.random.seed(42)

        # Generate power-law samples
        alpha_true = 2.5
        n = 1000
        u = np.random.uniform(0, 1, n)
        x = (1 - u) ** (-1 / (alpha_true - 1))

        alpha_est = _ks_alpha(x)

        # Should be close to true alpha
        assert abs(alpha_est - alpha_true) < 1.0

    def test_small_sample(self):
        """Test that small samples return nan."""
        x = np.array([1, 2, 3, 4, 5])
        alpha = _ks_alpha(x)

        assert np.isnan(alpha)


class TestGetSpectralMetrics:
    """Tests for combined spectral metrics."""

    def test_all_metrics_present(self):
        """Test that all expected metrics are returned."""
        W = np.random.randn(30, 30)
        metrics = get_spectral_metrics(W)

        expected_keys = [
            "spectral_entropy",
            "alpha_exponent",
            "stable_rank",
            "pl_alpha_hill",
            "pl_alpha_ks",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_metrics_finite(self):
        """Test that metrics are finite for reasonable matrices."""
        W = np.random.randn(50, 50)
        metrics = get_spectral_metrics(W)

        # All should be finite
        assert np.isfinite(metrics["spectral_entropy"])
        assert np.isfinite(metrics["stable_rank"])
        # alpha exponent may be nan for some matrices

    def test_consistent_types(self):
        """Test that all metrics are floats."""
        W = np.random.randn(30, 30)
        metrics = get_spectral_metrics(W)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float"


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_ill_conditioned_matrix(self):
        """Test with ill-conditioned matrix."""
        # Create ill-conditioned matrix
        n = 30
        s = np.logspace(0, -10, n)  # Condition number ~ 10^10
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        W = U @ np.diag(s) @ V.T

        metrics = get_spectral_metrics(W)

        # Should still get reasonable results
        assert np.isfinite(metrics["spectral_entropy"])
        assert np.isfinite(metrics["stable_rank"])

    def test_near_zero_matrix(self):
        """Test with near-zero matrix."""
        W = np.random.randn(20, 20) * 1e-15

        # Should not crash
        metrics = get_spectral_metrics(W)
        # Results may be nan, but should not crash

    def test_large_values(self):
        """Test with large matrix values."""
        W = np.random.randn(20, 20) * 1e10

        metrics = get_spectral_metrics(W)

        # Should handle large values
        assert np.isfinite(metrics["spectral_entropy"])
        assert np.isfinite(metrics["stable_rank"])
