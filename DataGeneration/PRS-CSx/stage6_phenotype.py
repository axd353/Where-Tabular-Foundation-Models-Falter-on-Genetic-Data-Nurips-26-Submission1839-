"""
Stage 6: Phenotype generation — heritability-calibrated liability model
with balanced binarisation.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.special import expit as _sigmoid

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from make_data_ver3 import _choose_balanced_threshold


def compute_genetic_values(G: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Stage 6a: g_i = sum_j G_{i,j} * beta_{i,j}."""
    return np.sum(G * beta, axis=1)


def calibrate_noise_sd(
    g: np.ndarray,
    cluster_ids: np.ndarray,
    h_sq_k: np.ndarray,
) -> np.ndarray:
    """
    Stage 6b: per-cluster noise SD calibrated to target heritability.

    Uses the **population-wide** genetic variance as the common baseline:

        sigma_eps_k^2 = Var(g) * (1 - h^2_k) / h^2_k

    where Var(g) is computed over ALL M individuals, not per-cluster.
    This ensures h^2_k is the sole knob differentiating noise levels
    across clusters.  Since h^2_major >= h^2_minority by design,
    majority always gets strictly less noise, making its labels cleaner
    and its Fisher discriminant ratio in G-space higher.

    Previously, per-cluster Var_k(g) was used, which caused majority
    (the largest cluster with the most genetic diversity) to receive
    disproportionately large absolute noise even with higher h^2_k,
    inadvertently making minority classification easier.

    Returns
    -------
    sigma_eps : (Kc,) noise SD per cluster
    """
    Kc = h_sq_k.shape[0]
    sigma_eps = np.zeros(Kc, dtype=float)
    # Pool variance over the full cohort — shared reference for all clusters
    var_g_global = np.var(g, ddof=1) if len(g) >= 2 else 1.0
    for k in range(Kc):
        h2 = float(np.clip(h_sq_k[k], 0.01, 0.99))
        sigma_eps[k] = np.sqrt(max(var_g_global * (1.0 - h2) / h2, 1e-12))
    return sigma_eps


def generate_labels(
    rng: np.random.Generator,
    g: np.ndarray,
    cluster_ids: np.ndarray,
    sigma_eps: np.ndarray,
    intercept: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stages 6c–6d: liability + sigmoid + balanced threshold binarisation.

    Returns
    -------
    y     : (M,) int binary labels
    eta   : (M,) latent liability
    probs : (M,) sigmoid probabilities
    """
    M = g.shape[0]
    eps = np.empty(M, dtype=float)
    for i in range(M):
        eps[i] = rng.normal(0.0, sigma_eps[cluster_ids[i]])

    eta = intercept + g + eps

    probs = _sigmoid(eta)

    # Per-cluster balanced threshold: each cluster independently gets
    # ~50% prevalence, regardless of where the cluster's probs sit
    # relative to the global distribution.  A single global threshold
    # calibrated mainly to the majority (largest cluster) would leave
    # minority prevalence free to vary widely (22%–87% observed), giving
    # a "free accuracy" shortcut to classifiers that identify ancestry.
    Kc = int(cluster_ids.max()) + 1
    y = np.empty(M, dtype=int)
    for k in range(Kc):
        mask = cluster_ids == k
        if mask.sum() == 0:
            continue
        t_k = _choose_balanced_threshold(probs[mask])
        y[mask] = (probs[mask] >= t_k).astype(int)

    return y, eta, probs
