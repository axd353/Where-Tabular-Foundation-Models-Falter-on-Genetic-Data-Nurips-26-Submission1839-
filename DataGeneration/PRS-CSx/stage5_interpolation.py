"""
Stage 5: Ancestry-continuous interpolation with per-SNP bandwidths.

Each SNP j gets its own bandwidth h_j ~ Uniform(h_floor, h),
producing per-variant kernel weights.
"""

from typing import Tuple

import numpy as np


H_FLOOR = 0.01  # minimum bandwidth to avoid division by zero


def sample_per_snp_bandwidths(
    rng: np.random.Generator,
    N: int,
    h: float,
    h_floor: float = H_FLOOR,
) -> np.ndarray:
    """
    Stage 5a: draw per-SNP bandwidths.

    h_j ~ Uniform(h_floor, h)

    Returns
    -------
    h_per_snp : (N,)
    """
    return rng.uniform(h_floor, max(h, h_floor + 1e-12), size=N)


def compute_interpolation_weights(
    A_mat: np.ndarray,
    centres: np.ndarray,
    h_per_snp: np.ndarray,
) -> np.ndarray:
    """
    Stage 5b: per-variant interpolation weights.

    w_tilde_{k,j}(a_i) = exp(-||a_i - c_k||^2 / (2 * h_j^2))
    w_{k,j}(a_i) = w_tilde / sum_k' w_tilde

    Parameters
    ----------
    A_mat      : (M, A) ancestry coordinates
    centres    : (Kc, A) cluster centres
    h_per_snp  : (N,) per-SNP bandwidths

    Returns
    -------
    weights : (M, Kc, N) normalised interpolation weights
    """
    M, A = A_mat.shape
    Kc = centres.shape[0]
    N = h_per_snp.shape[0]

    # Squared distances: (M, Kc)
    # ||a_i - c_k||^2 for each (i, k)
    diff = A_mat[:, np.newaxis, :] - centres[np.newaxis, :, :]  # (M, Kc, A)
    sq_dist = np.sum(diff ** 2, axis=2)  # (M, Kc)

    # Expand for per-SNP bandwidths: (M, Kc, N)
    h2 = h_per_snp ** 2  # (N,)
    # log_w = -sq_dist / (2 * h_j^2) for each SNP j
    log_w = -sq_dist[:, :, np.newaxis] / (2.0 * h2[np.newaxis, np.newaxis, :])  # (M, Kc, N)

    # Stable softmax over Kc dimension
    log_w_max = np.max(log_w, axis=1, keepdims=True)  # (M, 1, N)
    w = np.exp(log_w - log_w_max)  # (M, Kc, N)
    w_sum = np.sum(w, axis=1, keepdims=True)  # (M, 1, N)
    weights = w / w_sum

    return weights


def interpolate_effects(
    weights: np.ndarray,
    beta_pop: np.ndarray,
) -> np.ndarray:
    """
    Stage 5c: per-individual effects via weighted combination.

    beta_{i,j} = sum_k w_{k,j}(a_i) * beta_pop[k, j]

    Parameters
    ----------
    weights  : (M, Kc, N) interpolation weights
    beta_pop : (Kc, N) per-population effects

    Returns
    -------
    beta : (M, N) per-individual effects
    """
    # weights: (M, Kc, N), beta_pop: (Kc, N) -> broadcast (1, Kc, N)
    # element-wise multiply then sum over Kc
    return np.sum(weights * beta_pop[np.newaxis, :, :], axis=1)
