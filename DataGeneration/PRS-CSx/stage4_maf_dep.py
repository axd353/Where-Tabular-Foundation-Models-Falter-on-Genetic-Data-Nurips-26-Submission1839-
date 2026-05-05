"""
Stage 4: MAF-dependent architecture adjustment.

Rescales effect sizes by [2 * p * (1-p)]^{alpha/2} per cluster/SNP.
"""

import numpy as np


def apply_maf_dependent_scaling(
    beta_pop: np.ndarray,
    maf_by_cluster: np.ndarray,
    causal_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Stage 4: in-place MAF-dependent rescaling of population effects.

    beta_pop[k, j] *= [2 * p_{k,j} * (1 - p_{k,j})]^{alpha/2}

    When alpha=0 this is a no-op.

    Parameters
    ----------
    beta_pop       : (Kc, N) per-population effects (modified in place)
    maf_by_cluster : (Kc, N) cluster-specific MAFs
    causal_mask    : (N,) bool
    alpha          : scalar exponent (0 or negative)

    Returns
    -------
    beta_pop : (Kc, N) modified array (same object)
    """
    if alpha == 0.0:
        return beta_pop

    causal_idx = np.where(causal_mask)[0]
    if causal_idx.size == 0:
        return beta_pop

    p = maf_by_cluster[:, causal_idx]  # (Kc, n_causal)
    heterozygosity = 2.0 * p * (1.0 - p)
    # Avoid log(0) for degenerate MAFs
    heterozygosity = np.maximum(heterozygosity, 1e-12)
    scale = np.power(heterozygosity, alpha / 2.0)
    beta_pop[:, causal_idx] *= scale

    return beta_pop
