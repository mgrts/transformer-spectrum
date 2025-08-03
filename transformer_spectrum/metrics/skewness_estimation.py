import numpy as np


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)

    if std == 0:
        raise ValueError('Standard deviation is zero; skewness is undefined.')

    return np.mean((x - mean) ** 3) / (std ** 3)


def skewness_of_diff(x: np.ndarray) -> float:
    if x.ndim != 3 or x.shape[2] != 1:
        raise ValueError("Input array must have shape (n_sequences, sequence_length, 1)")

    diffs = np.diff(x[:, :, 0], axis=1)
    flat_diffs = diffs.flatten()
    return skewness(flat_diffs)
