import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy


def spectral_entropy(weight_matrix: np.ndarray) -> float:
    """
    Compute spectral entropy based on singular values.
    Higher entropy → more spread-out spectrum.
    """
    singular_values = svd(weight_matrix, compute_uv=False)
    power_spectrum = singular_values ** 2
    power_spectrum /= np.sum(power_spectrum)
    return entropy(power_spectrum)


def alpha_exponent(weight_matrix: np.ndarray, fit_range: tuple = (5, 50)) -> float:
    """
    Estimate power-law alpha exponent from log-log plot of singular values.
    Lower alpha → heavier tail.
    """
    singular_values = svd(weight_matrix, compute_uv=False)

    if len(singular_values) < fit_range[1]:
        # Not enough singular values to compute slope in this range
        return np.nan

    log_x = np.log(np.arange(1, len(singular_values) + 1))
    log_y = np.log(singular_values)

    start, end = fit_range
    slope, _ = np.polyfit(log_x[start:end], log_y[start:end], 1)
    return -slope


def stable_rank(weight_matrix: np.ndarray) -> float:
    """
    Compute the stable rank: ratio of Frobenius norm squared to spectral norm squared.
    """
    singular_values = svd(weight_matrix, compute_uv=False)
    return np.sum(singular_values ** 2) / (singular_values[0] ** 2)


def get_spectral_metrics(weight_matrix: np.ndarray) -> dict:
    """
    Return all spectral metrics as a dictionary.
    """
    return {
        "spectral_entropy": spectral_entropy(weight_matrix),
        "alpha_exponent": alpha_exponent(weight_matrix),
        "stable_rank": stable_rank(weight_matrix)
    }
