"""
Stage 0: Cluster centres & individual ancestry coordinates.

Reuses helpers from the HGP pipeline (make_data_ver3 / make_data_ver2).
"""

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

# Add parent directory so we can import from make_data_ver3 / make_data_ver2
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from make_data_ver2 import _scale_centers_to_target_mean_distance
from make_data_ver3 import _assign_cluster_sds, _make_imbalanced_cluster_ids


def generate_cluster_centres(
    rng: np.random.Generator,
    Kc: int,
    A: int,
    L_scale1: float,
) -> np.ndarray:
    """Stage 0a: sample Kc centres in R^A, rescale to target mean distance."""
    raw = rng.standard_normal((Kc, A))
    return _scale_centers_to_target_mean_distance(raw, L_scale1)


def generate_cluster_ids(
    rng: np.random.Generator,
    M: int,
    Kc: int,
    ancestry_group_cutpoints: Sequence[int],
    ancestry_group_size_divisors: Sequence[float],
    noise_percent: float,
) -> np.ndarray:
    """Stage 0b: imbalanced cluster assignments."""
    return _make_imbalanced_cluster_ids(
        rng, M, Kc,
        ancestry_group_cutpoints=ancestry_group_cutpoints,
        ancestry_group_size_divisors=ancestry_group_size_divisors,
        noise_percent=noise_percent,
    )


def generate_ancestry_coordinates(
    rng: np.random.Generator,
    centres: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_sd_mult: float,
    L_scale1: float,
    indiv_sd_choices: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stages 0c–0d: per-cluster ancestry spread + individual coordinates.

    Returns
    -------
    A_mat : (M, A) ancestry coordinates
    cluster_sd_multipliers : (Kc,) assigned SD multipliers
    """
    Kc = centres.shape[0]
    A = centres.shape[1]
    M = cluster_ids.shape[0]

    L_scale2 = cluster_sd_mult * L_scale1

    sd_multipliers = _assign_cluster_sds(rng, Kc, indiv_sd_choices)
    cluster_sds = sd_multipliers * L_scale2  # (Kc,)

    A_mat = np.empty((M, A), dtype=float)
    for i in range(M):
        k = cluster_ids[i]
        A_mat[i] = rng.normal(centres[k], cluster_sds[k], size=A)

    return A_mat, sd_multipliers
