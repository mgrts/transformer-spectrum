"""
Spectral metrics for analyzing weight matrix distributions.

This module provides functions to compute spectral properties of weight matrices,
which are useful for analyzing the conditioning and expressivity of neural networks.

Metrics included:
- Spectral entropy: Information-theoretic measure of singular value distribution
- Alpha exponent: Power-law decay rate of singular values
- Stable rank: Effective rank of the matrix
- Power-law alpha (Hill & KS estimators): Heavy-tail index of eigenspectrum

References:
    Martin, C. H., & Mahoney, M. W. (2021).
    "Implicit Self-Regularization in Deep Neural Networks."
    JMLR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray


def spectral_entropy(weight_matrix: NDArray[np.floating]) -> float:
    """
    Compute spectral entropy of a weight matrix.

    Spectral entropy measures how uniformly distributed the singular values are.
    A higher entropy indicates more uniform distribution (more "democratic" spectrum),
    while lower entropy indicates concentration of variance in few singular values.

    Args:
        weight_matrix: 2D weight matrix of shape (m, n)

    Returns:
        Spectral entropy in nats. Returns NaN if computation fails.

    Note:
        The entropy is computed on the normalized squared singular values,
        which represent the fraction of variance explained by each component.
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    if s.size == 0 or not np.isfinite(s).any():
        return np.nan

    # Filter to valid positive singular values
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return np.nan

    # Compute probability distribution from squared singular values
    p = (s ** 2).astype(np.float64)
    Z = p.sum()

    if Z <= 0 or not np.isfinite(Z):
        return np.nan

    p /= Z
    return float(entropy(p))


def _choose_fit_window(m: int) -> tuple[int, int]:
    """
    Choose fitting window for power-law estimation.

    Selects indices [start, end) for fitting, avoiding edge effects.
    """
    if m < 8:
        return (0, 0)

    start = max(1, int(0.10 * m))
    end = max(start + 6, int(0.60 * m))
    end = min(end, m)

    if end - start < 6:
        return (0, 0)

    return (start, end)


def alpha_exponent(
    weight_matrix: NDArray[np.floating],
    fit_range: tuple[int, int] | None = None,
) -> float:
    """
    Estimate the power-law exponent (alpha) from singular value decay.

    Fits a power law σ_i ∝ i^(-α) in log-log space. A larger alpha indicates
    faster decay of singular values, suggesting lower effective rank.

    Args:
        weight_matrix: 2D weight matrix
        fit_range: Optional (start, end) indices for fitting. If None,
                   automatically chosen to avoid edge effects.

    Returns:
        Estimated alpha exponent. Returns NaN if estimation fails.

    Note:
        The fit is performed in log-log space using least squares.
        Edge singular values are typically excluded to avoid noise.
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    s = s[np.isfinite(s) & (s > 0)]
    s = np.sort(s)[::-1]  # Sort in descending order
    m = s.size

    if m == 0:
        return np.nan

    if fit_range is None:
        start, end = _choose_fit_window(m)
        if end - start < 2:
            return np.nan
    else:
        start, end = fit_range
        if end > m or end - start < 2:
            return np.nan

    ranks = np.arange(1, m + 1, dtype=np.float64)
    log_x = np.log(ranks[start:end])
    log_y = np.log(s[start:end])

    try:
        slope, _ = np.polyfit(log_x, log_y, 1)
        return float(-slope)
    except Exception:
        return np.nan


def stable_rank(weight_matrix: NDArray[np.floating]) -> float:
    """
    Compute the stable rank of a weight matrix.

    Stable rank is defined as ||W||_F^2 / ||W||_2^2, which equals
    sum(s^2) / max(s)^2 where s are singular values.

    Unlike matrix rank, stable rank is continuous and measures the
    "effective" number of significant singular values.

    Args:
        weight_matrix: 2D weight matrix

    Returns:
        Stable rank (always >= 1 for non-zero matrices). Returns NaN if fails.

    References:
        Rudelson, M., & Vershynin, R. (2007).
        "Sampling from large matrices: An approach through geometric functional analysis."
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    s = s[np.isfinite(s) & (s >= 0)]
    if s.size == 0:
        return np.nan

    s_max = s.max()
    if s_max <= 0 or not np.isfinite(s_max):
        return np.nan

    num = float(np.sum(s ** 2))
    den = float(s_max ** 2)

    return num / den


def _esd(weight_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the Empirical Spectral Density (eigenvalue spectrum).

    For a weight matrix W, computes the squared singular values,
    which are eigenvalues of W^T W.
    """
    try:
        s = svd(weight_matrix, compute_uv=False)
        return (s * s).astype(np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def _hill_alpha(lambdas: NDArray[np.floating], k: int | None = None) -> float:
    """
    Estimate power-law exponent using the Hill estimator.

    The Hill estimator is a maximum likelihood estimator for the tail
    index of a power-law distribution.

    Args:
        lambdas: Array of eigenvalues (squared singular values)
        k: Number of order statistics to use. If None, uses ~10% of data.

    Returns:
        Estimated alpha. Returns NaN if estimation fails.
    """
    x = np.asarray(lambdas, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size

    if n < 8:
        return float("nan")

    if k is None:
        k = max(5, int(0.10 * n))
        k = min(k, max(5, n - 1))

    # Get the k largest values
    tail = np.sort(x)[::-1][:k]
    xmin = tail[-1]

    if xmin <= 0 or np.any(tail <= 0):
        return float("nan")

    logs = np.log(tail / xmin)
    H = logs.mean()

    if H <= 0 or not np.isfinite(H):
        return float("nan")

    return float(1.0 + 1.0 / H)


def _ks_alpha(lambdas: NDArray[np.floating]) -> float:
    """
    Estimate power-law exponent using KS-minimization.

    Searches for the xmin that minimizes the Kolmogorov-Smirnov
    statistic between the empirical CDF and the fitted power-law.

    Args:
        lambdas: Array of eigenvalues

    Returns:
        Estimated alpha. Returns NaN if estimation fails.

    References:
        Clauset, A., Shalizi, C. R., & Newman, M. E. (2009).
        "Power-law distributions in empirical data." SIAM review.
    """
    x = np.asarray(lambdas, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size

    if n < 20:
        return float("nan")

    xs = np.sort(x)
    best_alpha, best_ks = float("nan"), float("inf")

    # Start search from median to avoid too few tail samples
    start = n // 2

    for i in range(start, n - 5):
        xmin = xs[i]
        tail = xs[i:]
        m = tail.size

        if m < 5 or xmin <= 0:
            continue

        denom = np.sum(np.log(tail / xmin))
        if denom <= 0 or not np.isfinite(denom):
            continue

        # MLE for power-law exponent
        alpha = 1.0 + m / denom

        # KS statistic
        F_emp = np.arange(1, m + 1, dtype=np.float64) / m
        F_model = 1.0 - (tail / xmin) ** (1.0 - alpha)
        ks = np.max(np.abs(F_emp - F_model))

        if np.isfinite(ks) and ks < best_ks:
            best_ks, best_alpha = ks, alpha

    return float(best_alpha) if np.isfinite(best_alpha) else float("nan")


def get_spectral_metrics(weight_matrix: NDArray[np.floating]) -> dict[str, float]:
    """
    Compute all spectral metrics for a weight matrix.

    Args:
        weight_matrix: 2D weight matrix

    Returns:
        Dictionary containing:
        - spectral_entropy: Shannon entropy of normalized singular values
        - alpha_exponent: Power-law decay rate from linear fit
        - stable_rank: Effective rank of the matrix
        - pl_alpha_hill: Power-law alpha via Hill estimator
        - pl_alpha_ks: Power-law alpha via KS-minimization
    """
    lam = _esd(weight_matrix)

    return {
        "spectral_entropy": spectral_entropy(weight_matrix),
        "alpha_exponent": alpha_exponent(weight_matrix, fit_range=None),
        "stable_rank": stable_rank(weight_matrix),
        "pl_alpha_hill": _hill_alpha(lam),
        "pl_alpha_ks": _ks_alpha(lam),
    }
