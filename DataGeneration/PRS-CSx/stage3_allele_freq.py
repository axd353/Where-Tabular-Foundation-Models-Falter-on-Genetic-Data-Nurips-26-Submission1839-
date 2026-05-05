"""
Stage 3: Allele frequencies via Balding-Nichols model + dosage genotypes.
"""

from typing import Tuple

import numpy as np


def sample_ancestral_mafs(
    rng: np.random.Generator,
    N: int,
    maf_range: Tuple[float, float] = (0.05, 0.5),
) -> np.ndarray:
    """Stage 3a: ancestral MAFs ~ Uniform(p_min, p_max)."""
    return rng.uniform(maf_range[0], maf_range[1], size=N)


def sample_cluster_mafs_balding_nichols(
    rng: np.random.Generator,
    p_anc: np.ndarray,
    F_ST_k: np.ndarray,
) -> np.ndarray:
    """
    Stage 3b: cluster-specific MAFs via Balding-Nichols.

    For cluster k, SNP j:
        alpha_kj = p_anc[j] * (1 - F) / F
        beta_kj  = (1 - p_anc[j]) * (1 - F) / F
        p_{k,j} ~ Beta(alpha_kj, beta_kj)

    Parameters
    ----------
    p_anc  : (N,) ancestral MAFs
    F_ST_k : (Kc,) per-cluster F_ST values

    Returns
    -------
    maf_by_cluster : (Kc, N)
    """
    Kc = F_ST_k.shape[0]
    N = p_anc.shape[0]
    maf_by_cluster = np.empty((Kc, N), dtype=float)

    for k in range(Kc):
        F = F_ST_k[k]
        ratio = (1.0 - F) / F
        a_param = p_anc * ratio          # (N,)
        b_param = (1.0 - p_anc) * ratio  # (N,)
        # Clamp to avoid degenerate Beta(~0, ~0)
        a_param = np.maximum(a_param, 1e-6)
        b_param = np.maximum(b_param, 1e-6)
        maf_by_cluster[k] = rng.beta(a_param, b_param)

    # Clamp to valid MAF range to avoid 0/1 extremes
    np.clip(maf_by_cluster, 1e-6, 1.0 - 1e-6, out=maf_by_cluster)
    return maf_by_cluster


def sample_dosage_genotypes(
    rng: np.random.Generator,
    maf_by_cluster: np.ndarray,
    cluster_ids: np.ndarray,
    M: int,
    N: int,
) -> np.ndarray:
    """
    Stage 3c: dosage genotypes G_{i,j} ~ Binomial(2, p_{k_i, j}).

    Returns
    -------
    G : (M, N) int in {0, 1, 2}
    """
    p = maf_by_cluster[cluster_ids]  # (M, N)
    return rng.binomial(2, p, size=(M, N)).astype(int)
