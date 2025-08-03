from pathlib import Path

import numpy as np
import typer
from loguru import logger
from scipy.special import beta
from scipy.stats import t

from transformer_spectrum.config import PROCESSED_DATA_DIR, SEED, SEQUENCE_LENGTH

app = typer.Typer(pretty_exceptions_show_locals=False)


def safe_normalize(kernel):
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError('Kernel sum is zero, normalization failed.')
    return kernel / kernel_sum


def sine_kernel(size, period):
    x = np.linspace(0, 2 * np.pi * size / period, size)
    kernel = (np.sin(x) + 1) / 2
    return safe_normalize(kernel)


def cosine_kernel(size, period):
    x = np.linspace(0, 2 * np.pi * size / period, size)
    kernel = (np.cos(x) + 1) / 2
    return safe_normalize(kernel)


def gaussian_kernel(size, sigma):
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return safe_normalize(kernel)


def combined_cosine_gaussian_kernel(size, sigma, period):
    cosine_k = cosine_kernel(size, period)
    gaussian_k = gaussian_kernel(size, sigma)
    return safe_normalize(cosine_k * gaussian_k)


def filter1d_with_kernel(data, kernel):
    return np.convolve(data, kernel, mode='same')


def smooth_sequence(sequence, smoothing_type, kernel_size, sigma, period):
    sequence = sequence.flatten()

    if smoothing_type == 'gaussian':
        kernel = gaussian_kernel(kernel_size, sigma)
    elif smoothing_type == 'sine':
        kernel = sine_kernel(kernel_size, period)
    elif smoothing_type == 'cosine':
        kernel = cosine_kernel(kernel_size, period)
    elif smoothing_type == 'combined_cosine_gaussian':
        kernel = combined_cosine_gaussian_kernel(kernel_size, sigma, period)
    else:
        raise ValueError(f'Unsupported smoothing type: {smoothing_type}')

    smoothed = filter1d_with_kernel(sequence, kernel)
    return smoothed.reshape(-1)


class SkewedGeneralizedT:
    def __init__(self, mu=0.0, sigma=1.0, lam=0.0, p=2.0, q=2.0):
        assert -1 < lam < 1, "lam must be in (-1, 1)"
        assert sigma > 0, "sigma must be positive"
        assert p > 0 and q > 0, "p and q must be positive"

        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.p = p
        self.q = q

        # Compute beta functions
        B1 = beta(1.0 / p, q)
        B2 = beta(2.0 / p, q - 1.0 / p)
        B3 = beta(3.0 / p, q - 2.0 / p)

        # Compute v and m as per the provided formulas
        self.v = q ** (-1.0 / p) / np.sqrt((1 + 3 * lam ** 2) * (B3 / B1) - 4 * lam ** 2 * (B2 / B1) ** 2)
        self.m = lam * self.v * sigma * (2 * q ** (1.0 / p) * B2 / B1)

        # Normalizing constant
        self.norm_const = p / (2 * self.v * sigma * q**(1.0 / p) * B1)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = x - self.mu + self.m
        sgn_z = np.sign(z)
        denom = self.q * (self.sigma * self.v)**self.p * (1 + self.lam * sgn_z)**self.p
        bracket = 1 + (np.abs(z)**self.p) / denom
        return self.norm_const * bracket ** (-(1 / self.p + self.q))

    def rvs(self, size=1):
        samples = []
        # Use pdf at mu - m as a rough upper bound for rejection sampling
        max_pdf = self.pdf(self.mu - self.m)
        while len(samples) < size:
            x_candidate = t.rvs(df=2 * self.q, size=1) * self.sigma + self.mu
            y = np.random.uniform(0, max_pdf)
            if y < self.pdf(x_candidate):
                samples.append(x_candidate[0])
        return np.array(samples)

    def generate_sequences(self, n_sequences, sequence_length, n_features=1):
        total_samples = n_sequences * sequence_length * n_features
        samples = self.rvs(size=total_samples)
        return samples.reshape(n_sequences, sequence_length, n_features)


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy',
    n_sequences: int = 1000,
    sequence_length: int = SEQUENCE_LENGTH,
    n_features: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    lam: float = 0.0,
    p: float = 2.0,
    q: float = 2.0,
    apply_smoothing: bool = True,
    smoothing_type: str = 'combined_cosine_gaussian',
    kernel_size: int = 99,
    kernel_sigma: float = 10.0,
    period: float = 30.0,
    seed: int = SEED,
):
    np.random.seed(seed)
    logger.info('Generating synthetic sequences with SGT distribution...')
    logger.info(f'Params: mu={mu}, sigma={sigma}, lambda={lam}, p={p}, q={q}, smoothing={smoothing_type}')

    sgt = SkewedGeneralizedT(mu=mu, sigma=sigma, lam=lam, p=p, q=q)
    raw = sgt.generate_sequences(n_sequences, sequence_length, n_features)

    dataset = np.zeros_like(raw)
    for i in range(n_sequences):
        for j in range(n_features):
            if apply_smoothing:
                dataset[i, :, j] = smooth_sequence(raw[i, :, j], smoothing_type, kernel_size, kernel_sigma, period)
            else:
                dataset[i, :, j] = raw[i, :, j]

    logger.info(f'Saving dataset to {output_path} with shape {dataset.shape}')

    np.save(output_path, dataset)

    logger.success('Dataset generation complete.')


if __name__ == '__main__':
    app()
