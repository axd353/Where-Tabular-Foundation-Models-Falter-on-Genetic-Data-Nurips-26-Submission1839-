import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
from make_data_ver2 import _pairwise_sq_dists, _rbf_kernel, _scale_centers_to_target_mean_distance


# ================================================================
# NEW helpers (A): imbalanced clusters + per-cluster 

# Cluster assignment: ver2 uses balanced cluster sizes (equal-ish counts per cluster). See balanced assignment in make_data_ver2.py:146-174. ver3 switches to imbalanced clusters via _make_imbalanced_cluster_ids, with configurable ancestry group ratios and noise. See make_data_ver3.py:9-120 and usage in make_data_ver3.py:159-176.
# Ancestry spread: ver2 samples per-individual SDs from indiv_sd_choices, independent of cluster. See make_data_ver2.py:176-214. ver3 assigns per-cluster SD multipliers and forces the majority cluster to be twice as wide as the largest non-majority cluster. See helper in make_data_ver3.py:122-156 and usage in make_data_ver3.py:178-193.
# Genotypes (MAFs): ver2 uses a single MAF per SNP shared across all individuals (dosage sampled from that MAF). See make_data_ver2.py:216-228. ver3 makes MAFs cluster-dependent via per-cluster offsets, so different clusters have different MAFs for the same SNP. See make_data_ver3.py:159-206 and usage in make_data_ver3.py:195-200.
# Outputs/metadata: ver3 includes cluster_ids in the output and adds extra metadata fields like cluster_sd_multipliers_assigned, maf_by_cluster, and majority_is_cluster_index in kernel_info. See make_data_ver3.py:246-299. ver2 doesn’t return cluster_ids and has a smaller kernel_info. See make_data_ver2.py:256-323.
# Pipeline script: ver2 includes a runnable __main__ pipeline that builds payloads, splits train/val/test, writes latents, and creates a manifest CSV. See make_data_ver2.py:327-515. ver3 does not include that end-to-end payload/manifest pipeline; it only defines the simulator functions. The file ends after the balanced wrapper. See make_data_ver3.py:302-608.

# ================================================================

def _make_imbalanced_cluster_ids(
    rng: np.random.Generator,
    M: int,
    Kc: int,
    *,
    ancestry_group_cutpoints: Sequence[int] = (1, 3),
    ancestry_group_size_divisors: Sequence[float] = (9,12),
    noise_percent: float = 10.0,
) -> np.ndarray:
    """
    Create imbalanced cluster assignments with user-controllable ancestry group sizes.

    Intuition
    ---------
    We think of clusters as arranged in *ancestry groups* with different typical sizes,
    expressed as ratios relative to the *major* group.

    Defaults:
      ancestry_group_cutpoints = (1, 3)
      ancestry_group_size_divisors = (4.0, 8.0)

    For Kc clusters, this means:
      - Group 0: clusters [0 : 1)        → major ancestry group
          size ∝ 1.0 (baseline)
      - Group 1: clusters [1 : 3)        → next 2 ancestry groups
          size ∝ 1 / 4.0    (¼ of major)
      - Group 2: clusters [3 : Kc)       → remaining ancestry groups
          size ∝ 1 / 8.0    (⅛ of major)

    In general:
      - ancestry_group_cutpoints[g] is the *end index* (exclusive) of group g,
        interpreted on [0, Kc). The first group always starts at 0.
      - ancestry_group_size_divisors[g] > 0 controls relative size of group g+1
        vs the major group, via weight ≈ 1 / divisor.

    Noise
    -----
    Let base sizes sum to approximately (1 - noise_fraction) * M, where
      noise_fraction = noise_percent / 100.
    The remaining individuals are assigned by sampling cluster indices uniformly
    at random (with replacement) and incrementing their sizes by +1.

    This keeps:
      - approximate ratios between ancestry groups, and
      - exactly M total individuals.

    Parameters
    ----------
    M : int
        Total number of individuals.
    Kc : int
        Number of clusters.
    ancestry_group_cutpoints : sequence of int, default=(1, 3)
        Monotone increasing “end indices” defining ancestry groups along 0..Kc.
        Only values within [1, Kc] are used.
    ancestry_group_size_divisors : sequence of float, default=(4.0, 8.0)
        Positive divisors controlling group sizes *relative to the major group*.
        With defaults:
          group 0 (major) weight = 1.0
          group 1        weight = 1 / 4.0
          group 2+       weight = 1 / 8.0
    noise_percent : float, default=10.0
        Percent of individuals (~10%) assigned as random noise on top
        of the base pattern.

    Returns
    -------
    cluster_ids : (M,) int
        Cluster label for each individual, shuffled.

    Notes
    -----
    • Cluster 0 is the majority ancestry group by construction.
    • Every cluster gets at least 1 individual.
    """
    if Kc <= 0:
        raise ValueError("num_clusters must be ≥ 1.")
    if Kc == 1:
        return np.zeros(M, dtype=int)

    # ---- sanitize group configuration
    cutpoints = list(ancestry_group_cutpoints)
    divisors = list(ancestry_group_size_divisors)
    if len(cutpoints) == 0:
        cutpoints = [1]
    # use only as many divisors as we have transitions
    G = min(len(cutpoints), len(divisors))
    cutpoints = cutpoints[:G]
    divisors = divisors[:G]

    # clamp cutpoints into [1, Kc] and make them non-decreasing
    cutpoints = [max(1, min(Kc, int(cp))) for cp in cutpoints]
    cutpoints = sorted(set(cutpoints))  # ensure unique & sorted
    G = len(cutpoints)

    # ---- assign relative weights per cluster
    weights = np.zeros(Kc, dtype=float)

    # Group 0: major ancestry group, clusters [0 : cutpoints[0])
    major_end = cutpoints[0]
    weights[0:major_end] = 1.0

    prev_end = major_end
    # Groups 1 .. G-1 (use divisors[g-1])
    for g in range(1, G):
        group_end = cutpoints[g]
        if group_end > prev_end:
            w = 1.0 / float(divisors[g - 1])
            weights[prev_end:group_end] = w
        prev_end = group_end

    # Remaining clusters [prev_end : Kc) use last divisor (if any), or 1.0 as fallback
    if prev_end < Kc:
        if G > 0:
            w_last = 1.0 / float(divisors[G - 1])
        else:
            w_last = 1.0
        weights[prev_end:Kc] = w_last

    # If anything ended up non-positive (edge cases), fall back to equal weights
    if not np.all(weights > 0):
        weights[:] = 1.0

    # ---- build base sizes from weights
    noise_frac = max(0.0, min(float(noise_percent) / 100.0, 0.95))
    base_frac = 1.0 - noise_frac
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        # degenerate, fallback to equal split
        sizes = [max(1, M // Kc)] * Kc
    else:
        lam = max(1.0, np.floor(base_frac * M / w_sum))
        base_sizes = [max(1, int(np.floor(lam * w))) for w in weights]
        base_sum = int(np.sum(base_sizes))
        remainder = M - base_sum

        if remainder < 0:
            # Too many assigned in base; trim from largest clusters
            sizes = base_sizes[:]
            idx_order = np.argsort(sizes)[::-1]  # largest first
            need = -remainder
            for idx in idx_order:
                take = min(need, sizes[idx] - 1)
                sizes[idx] -= take
                need -= take
                if need <= 0:
                    break
        else:
            sizes = base_sizes[:]
            if remainder > 0:
                # Add +1 remainder times to random cluster indices
                bumps = rng.integers(0, Kc, size=remainder)
                for b in bumps:
                    sizes[b] += 1

    # ---- Build ids and shuffle
    cluster_ids = np.concatenate(
        [np.full(s, idx, dtype=int) for idx, s in enumerate(sizes)]
    )
    # Guard against any off-by-one from pathological rounding:
    if cluster_ids.shape[0] > M:
        cluster_ids = cluster_ids[:M]
    elif cluster_ids.shape[0] < M:
        # pad with majority (index 0)
        pad = np.zeros(M - cluster_ids.shape[0], dtype=int)
        cluster_ids = np.concatenate([cluster_ids, pad])

    rng.shuffle(cluster_ids)
    return cluster_ids

def _assign_cluster_sds(
    rng: np.random.Generator,
    Kc: int,
    sd_choices: Sequence[float],
    *,
    majority_cluster_index: int = 0,
) -> np.ndarray:
    """
    Select a width (multiplier) per cluster from sd_choices.
    Make the majority cluster's width = 2 * max(widths of other clusters).
    Returns per-cluster multipliers (no L_scale2 yet).
    """
    sd_choices = np.asarray(sd_choices, dtype=float)
    if np.any(sd_choices <= 0):
        raise ValueError("indiv_sd_choices must be positive.")

    # sample one sd choice per *non-majority* cluster
    per_cluster = np.empty(Kc, dtype=float)
    if Kc == 1:
        # trivial: just pick the max choice, then double
        per_cluster[0] = 2.0 * float(np.max(sd_choices))
        return per_cluster

    other_idxs = [i for i in range(Kc) if i != majority_cluster_index]
    sampled = rng.choice(sd_choices, size=len(other_idxs), replace=True)
    max_other = float(np.max(sampled))
    # write others
    for i, c_idx in enumerate(other_idxs):
        per_cluster[c_idx] = float(sampled[i])
    # majority set to twice the largest other
    per_cluster[majority_cluster_index] = 2.0 * max_other
    return per_cluster

# ================================================================
# NEW helper (B): genotypes with cluster-dependent MAFs
# ================================================================

def _simulate_dosage_by_cluster(
    rng: np.random.Generator,
    M: int,
    N: int,
    cluster_ids: np.ndarray,
    maf_range: Tuple[float, float],
    *,
    Kc: Optional[int] = None,
    cluster_maf_offset: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dosage matrix G with MAFs varying by cluster.

    For SNP j:
      - Draw a base MAF ~ Uniform(lo, hi).
      - For each cluster k, set maf_{k,j} = clip(base + delta_k, lo, hi),
        with delta_k ~ Uniform(-cluster_maf_offset, cluster_maf_offset).
      - Individual i from cluster k uses maf_{k,j}.

    Returns:
      G: (M, N) int in {0,1,2}
      maf_by_cluster: (Kc, N) floats in [lo, hi]
    """
    lo, hi = float(maf_range[0]), float(maf_range[1])
    if not (0.0 < lo <= hi <= 0.5):
        raise ValueError("maf_range must satisfy 0 < low ≤ high ≤ 0.5 for MAF.")

    if Kc is None:
        Kc = int(np.max(cluster_ids)) + 1

    base_mafs = rng.uniform(lo, hi, size=N)  # (N,)
    # one offset per cluster (shared across SNPs) keeps clusters distinct genome-wide
    deltas = rng.uniform(-cluster_maf_offset, cluster_maf_offset, size=Kc)  # (Kc,)

    maf_by_cluster = np.clip(base_mafs[None, :] + deltas[:, None], lo, hi)  # (Kc, N)

    # Build per-individual p matrix by indexing cluster rows
    p = maf_by_cluster[cluster_ids]  # (M, N)
    G = rng.binomial(2, p, size=(M, N)).astype(int)
    return G, maf_by_cluster


def _choose_balanced_threshold(
    p: np.ndarray,
    *,
    lower_frac: float = 0.45,
    upper_frac: float = 0.55,
    num_candidates: int = 21,
    margin_window: float = 0.02,
) -> float:
    """
    Choose a threshold t for probabilities p that keeps the fraction below t in
    [lower_frac, upper_frac] and avoids crowded regions of p.

    Method
    ------
    1) Sort p and identify feasible counts k in [ceil(lower_frac*M), floor(upper_frac*M)].
    2) Sample well-spread candidate k values across this range and map each to a
       midpoint threshold t between sorted values (p[k-1], p[k]) when distinct.
    3) Score each candidate by the local density of p near t, measured as the
       fraction of samples within a symmetric window |p - t| <= w, where
       w = margin_window * (max(p) - min(p)).
    4) Return the t with minimal local density (ties broken by closeness to 50% below).

    If no candidate interval is usable (e.g., all p are identical), fall back to
    the median of p.
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        raise ValueError("p must be non-empty.")

    M = int(p.size)
    k_min = int(np.ceil(float(lower_frac) * M))
    k_max = int(np.floor(float(upper_frac) * M))

    p_sorted = np.sort(p)
    # Need a split between p_sorted[k-1] and p_sorted[k]
    k_min = max(1, k_min)
    k_max = min(M - 1, k_max)

    if k_min > k_max:
        return float(np.median(p_sorted))

    k_vals = np.unique(
        np.round(np.linspace(k_min, k_max, num=max(2, int(num_candidates)))).astype(int)
    )

    p_min = float(p_sorted[0])
    p_max = float(p_sorted[-1])
    window = max(float(margin_window) * (p_max - p_min), 1e-12)

    best_t = None
    best_density = None
    best_frac_dist = None

    for k in k_vals:
        lo = float(p_sorted[k - 1])
        hi = float(p_sorted[k])
        if hi == lo:
            # No interval; skip to avoid ambiguous fraction below
            continue
        t = 0.5 * (lo + hi)
        frac_below = float(np.mean(p < t))
        if not (lower_frac <= frac_below <= upper_frac):
            continue
        density = float(np.mean(np.abs(p - t) <= window))
        frac_dist = abs(frac_below - 0.5)
        if best_density is None or density < best_density or (density == best_density and frac_dist < best_frac_dist):
            best_t = t
            best_density = density
            best_frac_dist = frac_dist

    if best_t is None:
        return float(np.median(p_sorted))
    return float(best_t)


# ================================================================
# New clustered, all-nonstationary GP simulator (clustered MAFs)
# ================================================================

def simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw(
    M: int,
    *,
    # ---- ancestry & clustering ----
    A: int = 4,
    num_clusters: int = 5,
    L_scale1: float = 1.0,
    cluster_sd_mult: float = 0.004,              # L_scale2 = cluster_sd_mult * L_scale1
    indiv_sd_choices: Sequence[float] = (.5, 0.75, 1.0, 1.5),
    ancestry_group_cutpoints: Sequence[int] = (1, 3),         # used in _make_imbalanced_cluster_ids
    ancestry_group_size_divisors: Sequence[float] = (9,12),  # used in _make_imbalanced_cluster_ids
    noise_percent: float = 10.0,# used in _make_imbalanced_cluster_ids
    # ---- genotype generation (dosage in {0,1,2}) ----
    N: int = 60,
    maf_range: Tuple[float, float] = (0.1, 0.5),
    cluster_maf_offset: float = 0.08,            # NEW: cluster MAF perturbation strength
    # ---- GP kernel & coefficients ----
    kernel: str = "rbf",
    amplitude: float = 1.0,
    drift_b: Union[float, Sequence[float]] = 0.0,
    jitter: float = 1e-6,
    # All SNPs vary: per-SNP lengthscale choices (in units of L_scale1)
    ls_multipliers: Sequence[float] = (100.0, 0.5, 0.008),
    ls_choice_probs: Optional[Sequence[float]] = None,
    # ---- label generation ----
    intercept: float = 0.0,
    logit_noise_sd: float = 0.0,
    # ---- reproducibility ----
    seed: Optional[int] = 1990,
    # ---- extras ----
    return_latent_eta: bool = False,
) -> Dict[str, np.ndarray]:
    r"""
    NEW variant of simulate_binary_gp_with_clusters_all_nonstationary_raw
      (A) Cluster sizes are imbalanced with a **majority cluster ≈ 4×** others.
          Per-cluster width is assigned, and the **majority cluster is twice as wide**
          as the largest other cluster.
      (B) Genotype MAFs **depend on cluster** (not allele alone).

    The rest of the behavior mirrors your original function for easy diffing.

    Notes on imbalanced sizes:
    • ancestry_group_cutpoints (default=(1, 3)):
          Defines ancestry groups as index ranges on [0, Kc):
            - Group 0: [0 : 1)   → major ancestry group.
            - Group 1: [1 : 3)   → next ancestry groups.
            - Group 2: [3 : Kc)  → remaining groups.

      • ancestry_group_size_divisors (default=(4.0, 8.0)):
          Controls relative sizes vs the major group:
            - Group 0: weight ∝ 1.0
            - Group 1: weight ∝ 1 / 4.0  (¼ of major)
            - Group 2+: weight ∝ 1 / 8.0 (⅛ of major)

      • noise_percent (default=10.0):
          We allocate approximately (1 - noise_percent/100) * M individuals
          according to the above ratios, then assign the remaining ~noise_percent%
          by sampling cluster indices uniformly at random.

    The downstream contract (returned keys/shapes) is unchanged.

    Parameters
    ----------
    M : int
        Number of individuals.
    A : int, default=4
        Dimensionality of the continuous ancestry space. Each individual has ancestry
        coordinate $a_i \in \mathbb{R}^A$.
    num_clusters : int, default=5
        Number of ancestry clusters $K_c$.
    L_scale1 : float, default=1.0
        Target mean inter-cluster distance; also the reference lengthscale for $K_{aa}$.
    cluster_sd_mult : float, default=0.004
        Sets per-cluster SD scale via $L_{\text{scale2}} = \text{cluster\_sd\_mult} \cdot L_{\text{scale1}}$.
        This is a scalar that defines the base SD scale shared by all clusters.
    indiv_sd_choices : sequence of floats, default=(0.5, 0.75, 1.0, 1.5)
        Candidate per-cluster width multipliers. After assignment, cluster $k$ uses
        SD $\sigma_k = m_k \cdot L_{\text{scale2}}$, where $m_k$ is the chosen multiplier.
        A per-cluster multiplier $m_k$ is sampled from this list to form
        $\sigma_k$, so the final per-cluster SDs have shape $(K_c,)$.
    ancestry_group_cutpoints : sequence of int, default=(1, 3)
        Defines ancestry groups along cluster indices [0, Kc). Each cutpoint is an
        exclusive end index. Used to create imbalanced cluster sizes.
    ancestry_group_size_divisors : sequence of float, default=(9, 12)
        Relative size divisors for each ancestry group beyond the major group.
        For a group with divisor $d$, its clusters have weight $\propto 1/d$.
    noise_percent : float, default=10.0
        Fraction (in percent) of individuals assigned as noise after base sizes are set.
        Remaining individuals are assigned by uniform random cluster sampling.
    N : int, default=60
        Number of SNPs.
    maf_range : (float, float), 
        Base MAF range for sampling; must satisfy $0 < \text{low} \le \text{high} \le 0.5$.
    cluster_maf_offset : float, 
        Cluster-specific MAF perturbation strength. For SNP $j$, a base MAF $p_j$ is
        drawn, then each cluster $k$ uses $p_{k,j} = \text{clip}(p_j + \delta_k, \text{low}, \text{high})$ with
        $\delta_k \sim \mathcal{U}(-\text{cluster\_maf\_offset}, \text{cluster\_maf\_offset})$.
    kernel : {"rbf"}, default="rbf"
        Kernel family over ancestry for GP coefficients. Only RBF is implemented here.
    amplitude : float, default=1.0
        Kernel signal std $\sigma$ (variance $\sigma^2$).
    drift_b : float or (A,), default=0.0
        Drift vector $b$ for the GP mean: $m(a_i) = a_i^T b$. If scalar, broadcast to (A,).
    jitter : float, default=1e-6
        Diagonal jitter added to kernels for numerical stability.
    ls_multipliers : sequence of floats, default=(100.0, 0.5, 0.008)
        Per-SNP lengthscale choices as multiples of $L_{\text{scale1}}$.
    ls_choice_probs : sequence of floats or None, default=None
        Optional probabilities for choosing each lengthscale multiplier. Uniform if None.
    intercept : float, default=0.0
        Logit intercept added to $\eta_i$.
    logit_noise_sd : float, default=0.0
        Gaussian noise SD added to the logit: $\eta_i \leftarrow \eta_i + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0,\,\text{logit\_noise\_sd}^2)$.
    seed : int or None, default=1990
        RNG seed for reproducibility.
    return_latent_eta : bool, default=False
        If True, include the latent $\eta$ in the output dict.

    Returns
    -------
    out : dict
        Keys and shapes:
          - "A": (M, A) ancestry coordinates.
          - "G": (M, N) dosage genotypes in {0,1,2} with cluster-specific MAFs.
          - "beta": (M, N) realized per-individual SNP effects $\beta_j(a_i)$.
          - "varying_mask": (N,) all True (all SNPs vary with ancestry).
          - "y": (M,) binary labels in {0,1}.
          - "K_aa": (M, M) reference ancestry kernel at $L_{\text{scale1}}$.
          - "kernel_info": dict with metadata (per-SNP lengthscales, cluster widths, MAFs).
          - "drift_b": (A,) drift vector used.
          - "cluster_ids": (M,) integer cluster label per individual.
          - "eta": (M,) latent logit, if return_latent_eta=True.
    """
    rng = np.random.default_rng(seed)

    # ---- 1) Cluster centers, scaled to mean distance ≈ L_scale1 (unchanged)
    Kc = int(num_clusters)
    if Kc <= 0:
        raise ValueError("num_clusters must be ≥ 1.")
    raw_centers = rng.normal(loc=0.0, scale=1.0, size=(Kc, A))
    centers = _scale_centers_to_target_mean_distance(raw_centers, target=L_scale1)  # (Kc, A)

    # ---- 2) Assign individuals to clusters (IMBALANCED) then shuffle (NEW helper)
    cluster_ids = _make_imbalanced_cluster_ids(rng,M, Kc, ancestry_group_cutpoints=ancestry_group_cutpoints,
        ancestry_group_size_divisors=ancestry_group_size_divisors, noise_percent=noise_percent,)  # (M,)

    # ---- 3) Sample ancestry per individual from its cluster (per-cluster widths) (NEW helper)
    L_scale2 = float(cluster_sd_mult) * float(L_scale1)
    # choose per-cluster multipliers from choices; majority cluster index = 0 by convention
    cluster_width_mults = _assign_cluster_sds(rng, Kc, indiv_sd_choices, majority_cluster_index=0)  # (Kc,)
    # actual per-cluster SDs in ancestry space
    cluster_sds = cluster_width_mults * L_scale2  # (Kc,)
    # sample ancestry
    A_mat = np.empty((M, A), dtype=float)
    for i in range(M):
        c_idx = cluster_ids[i]
        c = centers[c_idx]
        sd = float(cluster_sds[c_idx])
        A_mat[i] = rng.normal(loc=c, scale=sd, size=(A,))

    # ---- 4) Genotypes: dosage in {0,1,2} with cluster-dependent MAFs (NEW helper)
    G, maf_by_cluster = _simulate_dosage_by_cluster(
        rng, M, N, cluster_ids, maf_range, Kc=Kc, cluster_maf_offset=float(cluster_maf_offset)
    )  # (M,N), (Kc,N)

    # ---- 5) Per-SNP GP lengthscales (unchanged)
    ls_mults = np.asarray(ls_multipliers, dtype=float)
    if np.any(ls_mults <= 0):
        raise ValueError("ls_multipliers must be positive.")
    if ls_choice_probs is None:
        probs = None
    else:
        probs = np.asarray(ls_choice_probs, dtype=float)
        if probs.shape[0] != ls_mults.shape[0]:
            raise ValueError("ls_choice_probs must match the number of ls_multipliers.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("ls_choice_probs must sum to > 0.")
        probs = probs / total

    choices_idx = rng.choice(ls_mults.size, size=N, replace=True, p=probs)
    per_snp_ls = (ls_mults[choices_idx] * float(L_scale1)).astype(float)  # (N,)

    # ---- 6) Drift mean m(a) = a^T b (unchanged)
    if np.isscalar(drift_b):
        b = np.full(A, float(drift_b), dtype=float)
    else:
        b = np.asarray(drift_b, dtype=float)
        if b.shape != (A,):
            raise ValueError("drift_b must be scalar or shape (A,).")
    mean_vec = A_mat @ b  # (M,)

    # ---- 7) Draw ancestry-varying betas for all SNPs (unchanged)
    if kernel != "rbf":
        raise NotImplementedError("Only 'rbf' kernel is implemented in this version.")

    unique_ls, inverse = np.unique(per_snp_ls, return_inverse=True)
    beta = np.empty((M, N), dtype=float)
    Ks_by_ls = {}
    Ls_by_ls = {}

    for ls in unique_ls:
        K = _rbf_kernel(A_mat, lengthscale=float(ls), amplitude=float(amplitude), jitter=float(jitter))
        Ks_by_ls[float(ls)] = K
        try:
            L_chol = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K.copy()
            K.flat[:: K.shape[0] + 1] += 1e-8
            L_chol = np.linalg.cholesky(K)
            Ks_by_ls[float(ls)] = K
        Ls_by_ls[float(ls)] = L_chol

    for group_idx, ls in enumerate(unique_ls):
        cols = np.where(inverse == group_idx)[0]
        if cols.size == 0:
            continue
        L_chol = Ls_by_ls[float(ls)]
        Z = rng.normal(size=(M, cols.size))
        draws = (L_chol @ Z) + mean_vec[:, None]
        beta[:, cols] = draws

    # ---- 8) Labels (threshold chosen to keep ~balanced classes)
    eta = intercept + np.sum(G * beta, axis=1)
    if logit_noise_sd > 0:
        eta = eta + rng.normal(0.0, float(logit_noise_sd), size=M)
    p = 1.0 / (1.0 + np.exp(-eta))
    thresh = _choose_balanced_threshold(p)
    y = (p >= thresh).astype(int)

    # ---- 9) Outputs (downstream-compatible) (unchanged keys; extra metadata added)
    varying_mask = np.ones(N, dtype=bool)

    K_aa_reference = _rbf_kernel(A_mat, lengthscale=float(L_scale1), amplitude=float(amplitude), jitter=float(jitter))

    stationarity_difficulty = float(np.mean(per_snp_ls) / float(L_scale1))

    kernel_info = {
        "kernel": kernel,                                   # str     kernel family ("rbf")
        "amplitude": float(amplitude),                      # float   kernel signal std σ
        "jitter": float(jitter),                            # float   diagonal jitter added to kernels
        "L_scale1": float(L_scale1),                        # float   reference lengthscale (target mean inter-center distance)
        "cluster_sd_mult": float(cluster_sd_mult),          # float   sets L_scale2 via L_scale2 = cluster_sd_mult * L_scale1
        "L_scale2": float(L_scale2),                        # float   base per-cluster SD scale before multipliers
        "num_clusters": int(num_clusters),                  # int     number of clusters Kc
        "centers": centers,                                 # (Kc, A) cluster centers in ancestry space
        "indiv_sd_choices": tuple(float(x) for x in indiv_sd_choices),  # tuple  candidate per-cluster width multipliers
        "cluster_sd_multipliers_assigned": cluster_width_mults,         # (Kc,)  chosen width multiplier per cluster (before * L_scale2)
        "per_snp_lengthscales": per_snp_ls,                 # (N,)    absolute per-SNP lengthscales ℓ_j
        "ls_multipliers": tuple(float(x) for x in ls_mults),# tuple   discrete multipliers (choices) used to form ℓ_j = m * L_scale1
        "ls_choice_probs": None if probs is None else tuple(float(x) for x in probs),  # tuple|None  probs over ls_multipliers
        # "K_aa_by_lengthscale": {float(ls): Ks_by_ls[float(ls)] for ls in unique_ls},   # dict[ℓ]->(M,M) ancestry kernel at ℓ 
        #                                                                                ##TODO needs fixing after sampling these will change
        "stationarity_difficulty": stationarity_difficulty, # float   mean(ℓ_j)/L_scale1; smaller ⇒ harder (faster variation)
        "maf_by_cluster": maf_by_cluster,                   # (Kc, N) cluster-specific MAFs actually used for genotype sampling
        "majority_is_cluster_index": 0,                     # int     index of majority cluster (by convention)
    }

    out = {
        "A": A_mat,                    # (M, A)  ancestry coordinates
        "G": G,                        # (M, N)  dosage genotypes in {0,1,2}
        "beta": beta,                  # (M, N)  realized per-individual SNP effects β_j(a_i)
        "varying_mask": varying_mask,  # (N,)    boolean mask (all True; all SNPs vary)
        "y": y,                        # (M,)    binary labels {0,1}
        "K_aa": K_aa_reference,        # (M, M)  ancestry kernel at reference lengthscale L_scale1
        "kernel_info": kernel_info,    # (metadata) per-run constants and per-SNP details (no M-dependent arrays)
        "drift_b": b,                  # (A,)    drift vector b used in m(a)=a^T b
        "cluster_ids": cluster_ids,    # (M,)    cluster label (int) per individual
    }
    if return_latent_eta:
        out["eta"] = eta               # (M,)    latent logit η if requested
    return out


if __name__ == "__main__":
    sim_cfg = dict(
        M=2200,
        A=16,
        num_clusters=4,
        L_scale1=1,
        cluster_sd_mult=0.004,
        indiv_sd_choices=(0.5, 0.75, 1.0, 1.5),
        ancestry_group_cutpoints=(1, 3),
        ancestry_group_size_divisors=(3, 5),
        noise_percent=10.0,
        N=10,
        maf_range=(0.15, 0.5),
        cluster_maf_offset=0.05,
        kernel="rbf",
        amplitude=1.0,
        drift_b=0.0,
        jitter=1e-6,
        ls_multipliers=(100.0, 0.5, 0.008),
        ls_choice_probs=None,
        intercept=0.0,
        logit_noise_sd=0.0,
        seed=1990,
        return_latent_eta=False,
    )

    sim_out = simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw(**sim_cfg)

    out_dir = Path("Validation/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clusteredmafs_latents.pkl"

    latents = {
        "A": np.asarray(sim_out["A"]),
        "G": np.asarray(sim_out["G"]),
        "y": np.asarray(sim_out["y"]).astype(int),
        "beta": np.asarray(sim_out["beta"]),
        "kernel_info": sim_out.get("kernel_info", {}),
        "cluster_ids": np.asarray(sim_out.get("cluster_ids", [])),
    }

    with out_path.open("wb") as f:
        pickle.dump(latents, f, protocol=pickle.HIGHEST_PROTOCOL)

    y = latents["y"]
    p_ones = float(np.mean(y))
    print("[ok] ran simulator")
    print(f"saved: {out_path}")
    print(f"M={latents['A'].shape[0]}, A={latents['A'].shape[1]}, N={latents['G'].shape[1]}")
    print(f"y=1 fraction: {p_ones:.3f}")
    




