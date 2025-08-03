import numpy as np
from loguru import logger


def mean_abs_deviation(x: np.ndarray) -> float:
    return np.mean(np.abs(x - np.mean(x)))


def generate_n_sample(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    sample_size = len(x)

    # Generate a (n, sample_size) matrix of resampled indices
    resampled_indices = np.random.randint(0, sample_size, size=(n, sample_size))
    resampled_samples = x[resampled_indices]

    # Sum across the n samples
    return np.sum(resampled_samples, axis=0)


def kappa(x: np.ndarray, n: int) -> float:
    x = np.asarray(x)

    if n <= 1:
        raise ValueError('n must be greater than 1')

    m1 = mean_abs_deviation(x)
    mn = mean_abs_deviation(generate_n_sample(x, n))

    if mn <= 0 or m1 <= 0:
        raise ValueError('MAD of sample resulted in zero or negative, cannot compute log ratio.')

    return 2 - (np.log(n) / np.log(mn / m1))


def estimate_kappa_exponent(X: np.ndarray, n: int):
    S_1 = X
    S_n = generate_n_sample(X, n)

    M_1 = mean_abs_deviation(S_1)
    M_n = mean_abs_deviation(S_n)

    numerator = np.log(n)
    denominator = np.log(M_n / M_1)
    K_1n = 2 - (numerator / denominator)

    return numerator, denominator, K_1n, M_1, M_n


def compute_dispersion_scaling_series(X: np.ndarray, num_values: int = 100):
    metric_array = np.zeros((num_values, 5))

    for i in range(num_values):
        try:
            values = estimate_kappa_exponent(X, i + 1)
            metric_array[i] = values
        except Exception as e:
            logger.warning(f'Failed to compute dispersion scaling at n={i + 1}: {e}')
            metric_array[i] = np.nan

    return metric_array
