import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy


def spectral_entropy(weight_matrix: np.ndarray) -> float:
    s = svd(weight_matrix, compute_uv=False)
    if s.size == 0 or not np.isfinite(s).any():
        return np.nan
    p = (s ** 2).astype(float)
    Z = p.sum()
    if Z <= 0 or not np.isfinite(Z):
        return np.nan
    p /= Z
    return float(entropy(p))


def _choose_fit_window(m: int) -> tuple[int, int]:
    if m < 8:
        return (0, 0)
    start = max(1, int(0.10 * m))
    end = max(start + 6, int(0.60 * m))
    end = min(end, m)
    if end - start < 6:
        return (0, 0)
    return (start, end)


def alpha_exponent(weight_matrix: np.ndarray, fit_range: tuple[int, int] | None = None) -> float:
    s = svd(weight_matrix, compute_uv=False)
    s = s[np.isfinite(s) & (s > 0)]
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

    ranks = np.arange(1, m + 1, dtype=float)
    log_x = np.log(ranks[start:end])
    log_y = np.log(s[start:end])

    try:
        slope, _ = np.polyfit(log_x, log_y, 1)
        return float(-slope)
    except Exception:
        return np.nan


def stable_rank(weight_matrix: np.ndarray) -> float:
    s = svd(weight_matrix, compute_uv=False)
    s = s[np.isfinite(s) & (s >= 0)]
    if s.size == 0:
        return np.nan
    num = float(np.sum(s ** 2))
    den = float((s.max() ** 2)) if s.size else np.nan
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return num / den


def _esd(weight_matrix: np.ndarray) -> np.ndarray:
    s = svd(weight_matrix, compute_uv=False)
    return (s * s).astype(float)


def _hill_alpha(lambdas: np.ndarray, k: int | None = None) -> float:
    x = np.asarray(lambdas, float)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < 8:
        return float("nan")

    if k is None:
        k = max(5, int(0.10 * n))
        k = min(k, max(5, n - 1))

    tail = np.sort(x)[::-1][:k]
    xmin = tail[-1]
    if xmin <= 0 or np.any(tail <= 0):
        return float("nan")

    logs = np.log(tail / xmin)
    H = logs.mean()
    if H <= 0 or not np.isfinite(H):
        return float("nan")
    return float(1.0 + 1.0 / H)


def _ks_alpha(lambdas: np.ndarray) -> float:
    x = np.asarray(lambdas, float)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < 20:
        return float("nan")

    xs = np.sort(x)
    best_alpha, best_ks = float("nan"), float("inf")
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

        alpha = 1.0 + m / denom
        F_emp = np.arange(1, m + 1, dtype=float) / m
        F_model = 1.0 - (tail / xmin) ** (1.0 - alpha)
        ks = np.max(np.abs(F_emp - F_model))
        if np.isfinite(ks) and ks < best_ks:
            best_ks, best_alpha = ks, alpha

    return float(best_alpha) if np.isfinite(best_alpha) else float("nan")


def get_spectral_metrics(weight_matrix: np.ndarray) -> dict:
    lam = _esd(weight_matrix)
    return {
        "spectral_entropy": spectral_entropy(weight_matrix),
        "alpha_exponent": alpha_exponent(weight_matrix, fit_range=None),
        "stable_rank": stable_rank(weight_matrix),
        "pl_alpha_hill": _hill_alpha(lam),
        "pl_alpha_ks": _ks_alpha(lam),
    }
