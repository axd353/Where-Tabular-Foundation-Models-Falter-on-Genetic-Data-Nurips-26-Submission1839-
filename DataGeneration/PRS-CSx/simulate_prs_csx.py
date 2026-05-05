"""
PRS-CSx–inspired synthetic data simulator.

Generates one task (dataset + latents) following the 7 stages
described in DATA_GENERATION.md.  Drop-in replacement for
    simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw
in the HGP pipeline.

Usage:
    sim_out = simulate_prs_csx(**sim_cfg)
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from stage0_ancestry import (
    generate_ancestry_coordinates,
    generate_cluster_centres,
    generate_cluster_ids,
)
from stage1_shrinkage import sample_causal_mask, sample_shrinkage_prior
from stage2_effects import sample_per_snp_rg, sample_population_effects
from stage3_allele_freq import (
    sample_ancestral_mafs,
    sample_cluster_mafs_balding_nichols,
    sample_dosage_genotypes,
)
from stage4_maf_dep import apply_maf_dependent_scaling
from stage5_interpolation import (
    compute_interpolation_weights,
    interpolate_effects,
    sample_per_snp_bandwidths,
)
from stage6_phenotype import (
    calibrate_noise_sd,
    compute_genetic_values,
    generate_labels,
)


def simulate_prs_csx(
    M: int = 5000,
    *,
    # ---- ancestry & clustering (Stage 0) ----
    A: int = 16,
    num_clusters: int = 4,
    L_scale1: float = 1.0,
    cluster_sd_mult: float = 0.04,
    indiv_sd_choices: Sequence[float] = (1.0,),
    ancestry_group_cutpoints: Sequence[int] = (1, 3),
    ancestry_group_size_divisors: Sequence[float] = (4,),
    noise_percent: float = 10.0,
    # ---- shrinkage prior (Stage 1) ----
    N: int = 60,
    phi: float = 0.1,
    a: float = 1.0,
    b: float = 0.5,
    pi_causal: float = 1.0,
    # ---- cross-population effects (Stage 2) ----
    r_g: float = 0.85,
    sigma_beta_k: Optional[Sequence[float]] = None,
    # ---- allele frequencies (Stage 3) ----
    maf_range: Tuple[float, float] = (0.05, 0.5),
    F_ST_k: Optional[Sequence[float]] = None,
    # ---- MAF-dependent architecture (Stage 4) ----
    alpha: float = 0.0,
    # ---- interpolation (Stage 5) ----
    h: float = 1.0,
    # ---- phenotype (Stage 6) ----
    h_sq_k: Optional[Sequence[float]] = None,
    intercept: float = 0.0,
    # ---- RNG ----
    seed: int = 42,
) -> Dict:
    """
    Generate one PRS-CSx task: M individuals, N SNPs, num_clusters clusters.

    Returns a dict with the same keys expected by the HGP pipeline's
    downstream code (A, G, y, beta, cluster_ids) plus PRS-CSx-specific
    latents (beta_pop, psi, delta, causal_mask, etc.).
    """
    Kc = int(num_clusters)
    N = int(N)
    M = int(M)
    rng = np.random.default_rng(seed)

    # ---- defaults for per-cluster arrays ----
    if F_ST_k is None:
        F_ST_k = rng.uniform(0.01, 0.25, size=Kc)                          # (Kc,)
    F_ST_k = np.asarray(F_ST_k, dtype=float)                                # (Kc,)

    if h_sq_k is None:
        h_sq_major = rng.uniform(0.3, 0.6)                                  # scalar
        h_sq_k_arr = np.empty(Kc, dtype=float)                              # (Kc,)
        h_sq_k_arr[0] = h_sq_major
        for k in range(1, Kc):
            h_sq_k_arr[k] = h_sq_major * rng.uniform(0.5, 1.0)
        h_sq_k = h_sq_k_arr
    h_sq_k = np.asarray(h_sq_k, dtype=float)                                # (Kc,)

    if sigma_beta_k is not None:
        sigma_beta_k = np.asarray(sigma_beta_k, dtype=float)                # (Kc,)

    indiv_sd_choices = tuple(float(x) for x in indiv_sd_choices)
    if len(indiv_sd_choices) < Kc:
        indiv_sd_choices = indiv_sd_choices + (1.0,) * (Kc - len(indiv_sd_choices))

    # ==================================================================
    # Stage 0: Cluster centres & individual ancestry
    # Reuses from HGP: _scale_centers_to_target_mean_distance (ver2),
    #   _make_imbalanced_cluster_ids, _assign_cluster_sds (ver3)
    # ==================================================================
    centres = generate_cluster_centres(rng, Kc, A, L_scale1)                # (Kc, A)

    cluster_ids = generate_cluster_ids(                                      # (M,) int
        rng, M, Kc,
        ancestry_group_cutpoints=ancestry_group_cutpoints,
        ancestry_group_size_divisors=ancestry_group_size_divisors,
        noise_percent=noise_percent,
    )

    A_mat, cluster_sd_multipliers = generate_ancestry_coordinates(           # (M, A), (Kc,)
        rng, centres, cluster_ids,
        cluster_sd_mult=cluster_sd_mult,
        L_scale1=L_scale1,
        indiv_sd_choices=indiv_sd_choices,
    )

    # ==================================================================
    # Stage 1: Global–local shrinkage prior
    # ==================================================================
    delta, psi = sample_shrinkage_prior(rng, N, phi, a=a, b=b)              # (N,), (N,)
    causal_mask = sample_causal_mask(rng, N, pi_causal)                      # (N,) bool

    # ==================================================================
    # Stage 2: Per-population effect sizes (per-SNP r_{g,j})
    # ==================================================================
    r_g_per_snp = sample_per_snp_rg(rng, N, r_g)                            # (N,)
    beta_pop = sample_population_effects(                                    # (Kc, N)
        rng, Kc, N, psi, causal_mask, r_g_per_snp,
        sigma_beta_k=sigma_beta_k,
    )

    # ==================================================================
    # Stage 3: Allele frequencies + genotypes
    # ==================================================================
    p_anc = sample_ancestral_mafs(rng, N, maf_range=maf_range)              # (N,)
    maf_by_cluster = sample_cluster_mafs_balding_nichols(rng, p_anc, F_ST_k) # (Kc, N)
    G = sample_dosage_genotypes(rng, maf_by_cluster, cluster_ids, M, N)     # (M, N) int

    # ==================================================================
    # Stage 4: MAF-dependent architecture
    # ==================================================================
    beta_pop = apply_maf_dependent_scaling(                                  # (Kc, N) in-place
        beta_pop, maf_by_cluster, causal_mask, alpha,
    )

    # ==================================================================
    # Stage 5: Ancestry-continuous interpolation (per-SNP bandwidths)
    # ==================================================================
    h_per_snp = sample_per_snp_bandwidths(rng, N, h)                        # (N,)
    interp_weights = compute_interpolation_weights(A_mat, centres, h_per_snp) # (M, Kc, N)
    beta = interpolate_effects(interp_weights, beta_pop)                     # (M, N)

    # ==================================================================
    # Stage 6: Phenotype generation
    # Reuses from HGP: _choose_balanced_threshold (ver3)
    # ==================================================================
    g = compute_genetic_values(G, beta)                                      # (M,)
    sigma_eps = calibrate_noise_sd(g, cluster_ids, h_sq_k)                   # (Kc,)
    y, eta, probs = generate_labels(rng, g, cluster_ids, sigma_eps, intercept) # (M,), (M,), (M,)

    # ==================================================================
    # Assemble output (compatible with HGP downstream + PRS-CSx latents)
    # ==================================================================
    sim_out = {
        # --- same keys as HGP sim_out ---
        "A": A_mat,                          # (M, A)
        "G": G,                              # (M, N)
        "y": y,                              # (M,)
        "beta": beta,                        # (M, N)
        "cluster_ids": cluster_ids,          # (M,)
        # --- PRS-CSx specific latents ---
        "beta_pop": beta_pop,                # (Kc, N)
        "psi": psi,                          # (N,)
        "delta": delta,                      # (N,)
        "causal_mask": causal_mask,          # (N,)  bool
        "maf_by_cluster": maf_by_cluster,    # (Kc, N)
        "maf_ancestral": p_anc,              # (N,)
        "centers": centres,                  # (Kc, A)
        "r_g_per_snp": r_g_per_snp,          # (N,)
        "h_per_snp": h_per_snp,              # (N,)
        "interp_weights": interp_weights,    # (M, Kc, N)
        "h_sq_k": h_sq_k,                   # (Kc,)
        "sigma_eps": sigma_eps,              # (Kc,)
        "F_ST_k": F_ST_k,                   # (Kc,)
        "cluster_sd_multipliers": cluster_sd_multipliers,  # (Kc,)
        "g": g,                              # (M,)
        "eta": eta,                          # (M,)
        "probs": probs,                      # (M,)
    }
    return sim_out
