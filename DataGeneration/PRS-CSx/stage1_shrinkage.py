"""
Stage 1: Global–local continuous shrinkage prior (Strawderman–Berger).

Gamma–Gamma hierarchy: phi -> delta_j -> psi_j
Plus causal/null SNP mask.
"""

from typing import Tuple

import numpy as np


def sample_shrinkage_prior(
    rng: np.random.Generator,
    N: int,
    phi: float,
    a: float = 1.0,
    b: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1a: draw per-SNP local variance psi_j via Gamma-Gamma cascade.

    delta_j ~ Gamma(shape=b, rate=phi)   =>  scale = 1/phi
    psi_j   ~ Gamma(shape=a, rate=delta_j) =>  scale = 1/delta_j

    Returns
    -------
    delta : (N,) intermediate gamma values
    psi   : (N,) per-SNP local shrinkage variances
    """
    # numpy uses shape/scale parametrisation: Gamma(shape, scale=1/rate)
    delta = rng.gamma(shape=b, scale=1.0 / phi, size=N)
    # Avoid delta=0 which would make scale infinite
    delta = np.maximum(delta, 1e-30)
    psi = rng.gamma(shape=a, scale=1.0 / delta, size=N)
    return delta, psi


def sample_causal_mask(
    rng: np.random.Generator,
    N: int,
    pi_causal: float,
) -> np.ndarray:
    """
    Stage 1b: Bernoulli mask for causal SNPs.

    Returns
    -------
    causal_mask : (N,) bool — True for causal SNPs.
    """
    return rng.random(N) < pi_causal
