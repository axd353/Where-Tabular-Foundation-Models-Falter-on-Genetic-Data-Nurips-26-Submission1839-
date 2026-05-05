"""
Stage 2: Per-population effect sizes via MVN with per-SNP cross-population
genetic correlation r_{g,j} and shared local shrinkage psi_j.
"""

from typing import Optional

import numpy as np


def build_cross_pop_correlation_matrix(
    Kc: int,
    r_g: float,
    sigma_beta_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build the cross-population correlation/scale matrix C for a single r_g value.

    C_kk = sigma_beta_k[k]^2
    C_kl = r_g * sigma_beta_k[k] * sigma_beta_k[l]   (k != l)

    Returns
    -------
    C : (Kc, Kc) symmetric positive-definite matrix
    """
    if sigma_beta_k is None:
        sigma_beta_k = np.ones(Kc)
    sigma_beta_k = np.asarray(sigma_beta_k, dtype=float)

    C = r_g * np.outer(sigma_beta_k, sigma_beta_k)
    np.fill_diagonal(C, sigma_beta_k ** 2)
    return C


def sample_per_snp_rg(
    rng: np.random.Generator,
    N: int,
    r_g: float,
    r_g_floor: float = 0.04,
) -> np.ndarray:
    """
    Stage 2a: draw per-SNP genetic correlations.

    r_{g,j} ~ Uniform(r_g_floor, r_g)

    Parameters
    ----------
    rng       : numpy random generator
    N         : number of SNPs
    r_g       : upper bound (task-level genetic correlation)
    r_g_floor : lower bound (default 0.04)

    Returns
    -------
    r_g_per_snp : (N,)
    """
    lo = min(r_g_floor, r_g)
    hi = max(r_g_floor, r_g)
    return rng.uniform(lo, max(hi, lo + 1e-12), size=N)


def sample_population_effects(
    rng: np.random.Generator,
    Kc: int,
    N: int,
    psi: np.ndarray,
    causal_mask: np.ndarray,
    r_g_per_snp: np.ndarray,
    sigma_beta_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 2: draw per-population effect sizes jointly.

    For causal SNP j:
        beta_{.,j} ~ MVN(0, psi_j * C_j)
    where C_j uses per-SNP genetic correlation r_{g,j}.

    For non-causal SNP j:
        beta_{.,j} = 0

    Implementation: per-SNP Cholesky of C_j, then scale by sqrt(psi_j).

    Returns
    -------
    beta_pop : (Kc, N) per-population effects
    """
    if sigma_beta_k is None:
        sigma_beta_k = np.ones(Kc)
    sigma_beta_k = np.asarray(sigma_beta_k, dtype=float)

    beta_pop = np.zeros((Kc, N), dtype=float)

    causal_idx = np.where(causal_mask)[0]
    n_causal = causal_idx.size
    if n_causal == 0:
        return beta_pop

    # Draw standard normals for all causal SNPs at once
    U = rng.standard_normal((Kc, n_causal))  # (Kc, n_causal)

    for ci, j in enumerate(causal_idx):
        C_j = build_cross_pop_correlation_matrix(Kc, float(r_g_per_snp[j]), sigma_beta_k)
        C_j[np.diag_indices_from(C_j)] += 1e-10
        L_j = np.linalg.cholesky(C_j)  # (Kc, Kc)
        beta_pop[:, j] = np.sqrt(psi[j]) * (L_j @ U[:, ci])

    return beta_pop
