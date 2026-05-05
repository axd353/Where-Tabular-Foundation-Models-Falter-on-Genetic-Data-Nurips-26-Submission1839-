import numpy as np
from typing import Dict, Tuple, Sequence, Optional, Union, List
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ================================================================
# Utilities
# ================================================================

def _pairwise_sq_dists(Z: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances.

    Parameters
    ----------
    Z : (M, A) ndarray
        Row i is an A-dim vector.

    Returns
    -------
    r2 : (M, M) ndarray
        r2[i, j] = ||Z[i] - Z[j]||^2
    """
    s = np.sum(Z * Z, axis=1, keepdims=True)  # (M,1)
    return s + s.T - 2.0 * (Z @ Z.T)


def _rbf_kernel(A_mat: np.ndarray, lengthscale: float, amplitude: float, jitter: float) -> np.ndarray:
    """
    RBF (squared-exponential) kernel on ancestry coordinates.

    K[i, j] = amplitude^2 * exp(-0.5 * ||(a_i - a_j)/ℓ||^2) + jitter * δ_{ij}

    Parameters
    ----------
    A_mat : (M, A) ndarray
        Ancestry coordinates.
    lengthscale : float
        Positive scalar ℓ; larger ⇒ smoother across ancestry.
    amplitude : float
        Kernel signal std σ (variance = σ^2).
    jitter : float
        Small diagonal term for numerical stability.

    Returns
    -------
    K : (M, M) ndarray
    """
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive.")
    Z = A_mat / float(lengthscale)
    r2 = _pairwise_sq_dists(Z)
    K = (amplitude ** 2) * np.exp(-0.5 * r2)
    # add jitter to the diagonal
    K.flat[:: K.shape[0] + 1] += jitter
    return K


def _scale_centers_to_target_mean_distance(C: np.ndarray, target: float) -> np.ndarray:
    """
    Rescale cluster centers so their mean pairwise distance ≈ target.

    Parameters
    ----------
    C : (K, A) ndarray
        Raw cluster centers (any spread).
    target : float
        Desired mean inter-center distance (≈ L_scale1).

    Returns
    -------
    C_scaled : (K, A) ndarray
    """
    if C.shape[0] < 2:
        return C
    D2 = _pairwise_sq_dists(C)
    D = np.sqrt(np.maximum(D2, 0.0))
    iu = np.triu_indices_from(D, k=1)
    current_mean = float(np.mean(D[iu])) if iu[0].size else 1.0
    if current_mean <= 0:
        return C
    s = target / current_mean
    return C * s


# ================================================================
# New clustered, all-nonstationary GP simulator with dosage 0/1/2
# ================================================================

def simulate_binary_gp_with_clusters_all_nonstationary_raw(
    M: int,
    *,
    # ---- ancestry & clustering ----
    A: int = 4,
    num_clusters: int = 5,
    L_scale1: float = 1.0,
    cluster_sd_mult: float = 0.004,              # L_scale2 = cluster_sd_mult * L_scale1
    indiv_sd_choices: Sequence[float] = (.5, 0.75, 1.0, 1.5), #how spread out are the clusters in acetsral space
    # ---- genotype generation (now dosage in {0,1,2}) ----
    N: int = 60,
    maf_range: Tuple[float, float] = (0.05, 0.5),
    # ---- GP kernel & coefficients ----
    kernel: str = "rbf",                         # Only "rbf" implemented in this version
    amplitude: float = 1.0,
    drift_b: Union[float, Sequence[float]] = 0.0,
    jitter: float = 1e-6,
    # All SNPs vary: per-SNP lengthscale choices (in units of L_scale1)
    ls_multipliers: Sequence[float] = (100.0, 0.5, 0.008),
    ls_choice_probs: Optional[Sequence[float]] = None,  # None => uniform over choices
    # ---- label generation ----
    intercept: float = 0.0,
    logit_noise_sd: float = 0.0,
    # ---- reproducibility ----
    seed: Optional[int] = 1990,
    # ---- extras ----
    return_latent_eta: bool = False,
) -> Dict[str, np.ndarray]:
    r"""
    Simulate a **binary phenotype** with **dosage genotypes in {0,1,2}** and
    ancestry-varying effects for **all** SNPs. Subjects occupy clusters in ancestry space.

    Overview
    --------
    1) **Ancestry clusters**:   
       - Draw `num_clusters` centers in A-dim space, then rescale their positions so the
         **mean inter-center distance ≈ L_scale1**.
       - Within each cluster, subjects are sampled from an isotropic normal:
         a_i ~ 𝓝(center_k, (i⋅L_scale2)^2 I_A), where L_scale2 = cluster_sd_mult⋅L_scale1
         and i ∈ {0.75, 1.0, 1.5} chosen per individual.

    2) **Genotypes**:
       - For each SNP: MAF ~ Uniform(maf_range).
       - Dosage G_{ij} ~ Binomial(n=2, p=MAF_j) ⇒ G ∈ {0,1,2}.

    3) **Per-SNP GP effects (all non-stationary)**:
       - Each SNP j draws a lengthscale ℓ_j from the set
         { m⋅L_scale1 | m ∈ ls_multipliers } (optionally with probabilities).
       - For each unique ℓ, we compute K_aa(ℓ) via an RBF kernel and sample
         β_j(·) ~ 𝓝(m, K_aa(ℓ)), where m_i = a_i^T drift_b (scalar drift).
       - This yields β of shape (M, N). **All SNPs vary with ancestry**:
         `varying_mask = np.ones(N, dtype=bool)`.

    4) **Labels**:
       - η_i = intercept + Σ_j G_{ij} β_j(a_i) + ε_i,   ε_i ~ 𝓝(0, logit_noise_sd^2)
       - y_i ~ Bernoulli(sigmoid(η_i)).

    Notes on difficulty / stationarity
    ----------------------------------
    - A handy summary of “how non-stationary” a dataset is:
      `stationarity_difficulty = mean(ℓ_j) / L_scale1`.
      Smaller ratios ⇒ **harder** (faster variation across ancestry).

    Downstream Contract (unchanged)
    -------------------------------
    Returns a dict with the same keys and shapes as your prior simulator:
      - "A": (M, A) ancestry coordinates
      - "G": (M, N) dosage genotypes in {0,1,2}   **(changed from {0,1})**
      - "beta": (M, N) realized per-individual coefficients
      - "varying_mask": (N,) all True
      - "y": (M,) integer labels {0,1}
      - "K_aa": (M, M) **reference kernel at L_scale1** (kept for compatibility)
      - "kernel_info": dict with metadata (includes per-SNP ℓ_j, centers, cluster ids)
      - "drift_b": (A,) drift vector actually used
      - "eta": (M,) if return_latent_eta=True

    Parameters
    ----------
    M : int
        Number of individuals.
    A : int, default=4
        Dimensionality of the continuous ancestry space.
    num_clusters : int, default=5
        Number of ancestry clusters.
    L_scale1 : float, default=1.0
        Target mean inter-cluster distance; also the reference lengthscale for K_aa.
    cluster_sd_mult : float, default=0.004
        Sets per-cluster SD: L_scale2 = cluster_sd_mult * L_scale1.
    indiv_sd_choices : sequence of floats, default=(0.75, 1.0, 1.5)
        Per-individual multiplier on L_scale2.
    N : int, default=60
        Number of SNPs.
    maf_range : (float, float), default=(0.05, 0.5) -- varies for every SNP
        Minor allele frequency range for dosage generation (Binomial(n=2,p)).
        Must satisfy 0 < low ≤ high ≤ 0.5.
    kernel : {"rbf"}, default="rbf"
        Kernel over ancestry for GP coefficients. Only RBF implemented here.
    amplitude : float, default=1.0
        Kernel signal std (σ).
    drift_b : float or (A,), default=0.0
        Drift vector b in m(a) = a^T b.
    jitter : float, default=1e-6
        Diagonal jitter added to kernels.
    ls_multipliers : sequence of floats, default=(100.0, 0.5, 0.004)
        Per-SNP lengthscale choices as multiples of L_scale1.
    ls_choice_probs : sequence of floats or None, default=None
        Optional probabilities for choosing each multiplier. Uniform if None.
    intercept : float, default=0.0
        Logit intercept.
    logit_noise_sd : float, default=0.0
        Gaussian noise SD added to the logit before the sigmoid.
    seed : int or None, default=1990
        RNG seed for reproducibility.
    return_latent_eta : bool, default=False
        If True, include the latent η in the output dict.

    Returns
    -------
    out : dict
        See “Downstream Contract (unchanged)” above.
    """
    rng = np.random.default_rng(seed)

    # ---- 1) Cluster centers, scaled to mean distance ≈ L_scale1
    Kc = int(num_clusters)
    if Kc <= 0:
        raise ValueError("num_clusters must be ≥ 1.")
    raw_centers = rng.normal(loc=0.0, scale=1.0, size=(Kc, A))
    centers = _scale_centers_to_target_mean_distance(raw_centers, target=L_scale1)  # (Kc, A)

    # ---- 2) Assign individuals to clusters (balanced as possible), then shuffle
    base = M // Kc
    extra = M % Kc
    sizes = np.array([base + (1 if i < extra else 0) for i in range(Kc)], dtype=int)  # (Kc,)
    cluster_ids = np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])  # (M,)
    rng.shuffle(cluster_ids)

    # ---- 3) Sample ancestry for each individual from its cluster
    L_scale2 = float(cluster_sd_mult) * float(L_scale1)
    sd_choices = np.asarray(indiv_sd_choices, dtype=float)
    if np.any(sd_choices <= 0):
        raise ValueError("indiv_sd_choices must be positive.")
    sd_idx = rng.integers(0, sd_choices.size, size=M)
    indiv_sds = sd_choices[sd_idx] * L_scale2  # (M,)

    A_mat = np.empty((M, A), dtype=float)
    for i in range(M):
        c = centers[cluster_ids[i]]
        sd = indiv_sds[i]
        A_mat[i] = rng.normal(loc=c, scale=sd, size=(A,))

    # ---- 4) Genotypes: dosage in {0,1,2}
    lo, hi = float(maf_range[0]), float(maf_range[1])
    if not (0.0 < lo <= hi <= 0.5):
        raise ValueError("maf_range must satisfy 0 < low ≤ high ≤ 0.5 for MAF.")
    mafs = rng.uniform(lo, hi, size=N)                                # (N,)
    # Dosage per SNP per individual: Binomial with n=2
    G = rng.binomial(2, mafs[None, :], size=(M, N)).astype(int)        # (M, N) in {0,1,2}

    # ---- 5) Per-SNP GP lengthscales
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

    # ---- 6) Drift mean m(a) = a^T b
    if np.isscalar(drift_b):
        b = np.full(A, float(drift_b), dtype=float)
    else:
        b = np.asarray(drift_b, dtype=float)
        if b.shape != (A,):
            raise ValueError("drift_b must be scalar or shape (A,).")
    mean_vec = A_mat @ b  # (M,)

    # ---- 7) Draw ancestry-varying betas for all SNPs
    if kernel != "rbf":
        raise NotImplementedError("Only 'rbf' kernel is implemented in this version.")

    unique_ls, inverse = np.unique(per_snp_ls, return_inverse=True)
    beta = np.empty((M, N), dtype=float)
    Ks_by_ls = {}
    Ls_by_ls = {}

    # Precompute one K and its Cholesky per unique lengthscale
    for ls in unique_ls:
        K = _rbf_kernel(A_mat, lengthscale=float(ls), amplitude=float(amplitude), jitter=float(jitter))
        Ks_by_ls[float(ls)] = K
        try:
            L_chol = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add a tiny bit more jitter if needed
            K = K.copy()
            K.flat[:: K.shape[0] + 1] += 1e-8
            L_chol = np.linalg.cholesky(K)
            Ks_by_ls[float(ls)] = K
        Ls_by_ls[float(ls)] = L_chol

    # Sample independent GPs for SNP groups sharing the same ℓ
    for group_idx, ls in enumerate(unique_ls):
        cols = np.where(inverse == group_idx)[0]     # SNP indices with this ℓ
        if cols.size == 0:
            continue
        L_chol = Ls_by_ls[float(ls)]                # (M, M)
        Z = rng.normal(size=(M, cols.size))         # (M, k)
        draws = (L_chol @ Z) + mean_vec[:, None]    # (M, k)
        beta[:, cols] = draws

    # ---- 8) Labels
    eta = intercept + np.sum(G * beta, axis=1)      # (M,)
    if logit_noise_sd > 0:
        eta = eta + rng.normal(0.0, float(logit_noise_sd), size=M)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(int)

    # ---- 9) Outputs (downstream-compatible)
    varying_mask = np.ones(N, dtype=bool)  # all SNPs vary

    # Reference kernel (for compatibility / inspection) at the dataset scale L_scale1
    K_aa_reference = _rbf_kernel(A_mat, lengthscale=float(L_scale1), amplitude=float(amplitude), jitter=float(jitter))

    # Difficulty summary: smaller means harder (faster ancestry variation)
    stationarity_difficulty = float(np.mean(per_snp_ls) / float(L_scale1))

    kernel_info = {
        "kernel": kernel,
        "amplitude": float(amplitude),
        "jitter": float(jitter),
        "L_scale1": float(L_scale1),
        "cluster_sd_mult": float(cluster_sd_mult),
        "L_scale2": float(L_scale2),
        "num_clusters": int(num_clusters),
        "centers": centers,                     # (Kc, A)
        "cluster_ids": cluster_ids,             # (M,)
        "indiv_sd_choices": tuple(float(x) for x in indiv_sd_choices),
        "per_snp_lengthscales": per_snp_ls,     # (N,)
        "ls_multipliers": tuple(float(x) for x in ls_mults), #The discrete set of lengthscale multipliers we sample from (e.g., (100.0, 0.5, 0.004)). Each SNP’s actual lengthscale is multiplier * L_scale1. K = number of choices.
        "ls_choice_probs": None if probs is None else tuple(float(x) for x in probs),#Optional probabilities for choosing each multiplier in ls_multipliers. None means uniform. Sums to 1.0 if provided.
        "K_aa_by_lengthscale": {float(ls): Ks_by_ls[float(ls)] for ls in unique_ls},  # For each unique absolute lengthscale ℓ (in the current dataset), stores the RBF ancestry kernel matrix computed at that ℓ. Keys are the numeric ℓ, values are the corresponding (M, M) kernels you can inspect for stability/numerics.
        "stationarity_difficulty": stationarity_difficulty,
    }

    out = {
        "A": A_mat,                    # (M, A)
        "G": G,                        # (M, N) in {0,1,2}
        "beta": beta,                  # (M, N)
        "varying_mask": varying_mask,  # (N,)
        "y": y,                        # (M,)
        "K_aa": K_aa_reference,        # (M, M)
        "kernel_info": kernel_info,
        "drift_b": b,                  # (A,)
    }
    if return_latent_eta:
        out["eta"] = eta
    return out


def sim_to_Xy(sim_out: dict):
    """
    Convert simulator output to (X, y, meta), keeping the same contract your pipeline expects.
    X = [A | G]  -> shape (M, A+N), with G as dosage {0,1,2}.
    """
    A = np.asarray(sim_out["A"])     # (M, A)
    G = np.asarray(sim_out["G"])     # (M, N) in {0,1,2}
    y = np.asarray(sim_out["y"]).astype(int)  # (M,)
    X = np.concatenate([A, G], axis=1)
    meta = {
        "A_dim": int(A.shape[1]),
        "N_snps": int(G.shape[1]),
        "M": int(A.shape[0]),
        "varying_mask": np.asarray(sim_out["varying_mask"]),
        "kernel_info": sim_out.get("kernel_info", {}),
        "drift_b": np.asarray(sim_out.get("drift_b", [])),
    }
    return X, y, meta


if __name__ == "__main__":
    out_dir = Path("gp_payloads_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    role = "test"  # just a label baked into filenames
    seed = 1990
    sim_cfg = dict(
        # ---- ancestry & clustering ----
        M=1200,
        A=16,
        num_clusters=6,
        L_scale1=0.5,         # reference lengthscale (also mean inter-center distance)
        cluster_sd_mult=0.004, # L_scale2 = cluster_sd_mult * L_scale1
        indiv_sd_choices=(0.75, 1.0, 1.5),

        # ---- genotypes (dosage) ----
        N=20,
        maf_range=(0.05, 0.5),

        # ---- GP kernel & coeffs ----
        kernel="rbf",
        amplitude=1.0,
        drift_b=0.0,
        jitter=1e-6,
        ls_multipliers=(100.0, 0.5, 0.004),  # each SNP picks ℓ_j from these × L_scale1
        ls_choice_probs=None,                # or e.g. (0.2, 0.6, 0.2)

        # ---- labels ----
        intercept=0.0,
        logit_noise_sd=0.0,

        # ---- reproducibility ----
        seed=seed,

        # ---- extras ----
        return_latent_eta=False,
    )

    # --------------------------
    # Run simulator
    # --------------------------
    sim_out = simulate_binary_gp_with_clusters_all_nonstationary_raw(**sim_cfg)

    # --------------------------
    # Build payload (same schema)
    # --------------------------
    X_all, y_all, meta = sim_to_Xy(sim_out)
    M = X_all.shape[0]

    # Split knobs (adjust as desired)
    trainval_fraction = 0.85
    valid_set_ratio = 0.2

    rng = np.random.default_rng(seed)
    n_trainval = int(np.floor(M * float(trainval_fraction)))
    all_idx = np.arange(M)
    trainval_idx = rng.choice(all_idx, size=n_trainval, replace=False)
    mask = np.zeros(M, dtype=bool)
    mask[trainval_idx] = True

    X_pool, y_pool = X_all[mask], y_all[mask]
    X_test, y_test = X_all[~mask], y_all[~mask]

    strat = y_pool if np.unique(y_pool).size >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=float(valid_set_ratio),
        random_state=seed,
        stratify=strat
    )

    payload = {
        "train": [X_train, y_train],
        "gradient_signal": [X_val, y_val],
        "HPO": [None, None],
        "holdout": [X_test, y_test],
    }

    # --------------------------
    # File naming
    # --------------------------
    M_val = int(sim_cfg["M"])
    A_dim_val = int(sim_cfg["A"])
    N_val = int(sim_cfg["N"])
    # all SNPs vary now; keep N_ctrl column compatible by setting to N
    N_ctrl_val = N_val

    amp_val = float(sim_cfg.get("amplitude", 1.0))
    L_scale1 = float(sim_cfg.get("L_scale1", 1.0))
    kernel_name = sim_cfg.get("kernel", "rbf")

    base_name = (
        f"gp_M{M_val}_Adim{A_dim_val}_N{N_val}_Nctrl{N_ctrl_val}_"
        f"{kernel_name}_{role}_A{amp_val:g}_l{L_scale1:g}_seed{seed}"
    )

    payload_path = out_dir / f"{base_name}.pkl"

    # --------------------------
    # Save payload
    # --------------------------
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --------------------------
    # Save latents (same keys as old flow)
    # --------------------------
    latents = {
        "beta": np.asarray(sim_out["beta"]),                 # (M, N)
        "K_aa": np.asarray(sim_out["K_aa"]),                 # (M, M) reference at L_scale1
        "drift_b": np.asarray(sim_out["drift_b"]),           # (A,)
        "varying_mask": np.asarray(sim_out["varying_mask"]), # (N,)
        "kernel_info": sim_out.get("kernel_info", {}),
        "intercept": float(sim_cfg.get("intercept", 0.0)),
        "logit_noise_sd": float(sim_cfg.get("logit_noise_sd", 0.0)),
        "seed": sim_cfg.get("seed", None),
        "ancestral-codes": np.asarray(sim_out["A"]),         # (M, A)
    }
    latents_path = payload_path.with_suffix("")
    latents_path = Path(str(latents_path) + "-latents.pkl")
    with open(latents_path, "wb") as f:
        pickle.dump(latents, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --------------------------
    # Save single-row manifest CSV
    #   A (amplitude) and l (lengthscale) mirror your prior manifest;
    #   l now comes from L_scale1; N_ctrl is set to N (all SNPs non-stationary).
    # --------------------------
    manifest_row = {
        "role": role,
        "path_to_dataset": str(payload_path),
        "path_to_latents": str(latents_path),
        "A": amp_val,                                  # amplitude
        "l": L_scale1,                                 # reference lengthscale
        "A_dim": A_dim_val,
        "N_snps": N_val,
        "M": M_val,
        "kernel": kernel_name,
        "N_ctrl": N_ctrl_val,                          # compatibility: all SNPs vary
        "drift_present": bool(np.any(np.asarray(sim_out["drift_b"]) != 0)),
        "seed": seed,
        "trainval_fraction": float(trainval_fraction),
        "valid_set_ratio": float(valid_set_ratio),
        # keep full config for traceability (swap L_scale1 for 'lengthscales' semantics)
        "p_ns": 1.0,                                   # all variants non-stationary
        "config_json": json.dumps(sim_cfg, default=float),
    }
    df_manifest = pd.DataFrame([manifest_row])
    manifest_csv = out_dir / "manifest_single.csv"
    df_manifest.to_csv(manifest_csv, index=False)

    print(f"[ok] Wrote payload: {payload_path}")
    print(f"[ok] Wrote latents: {latents_path}")
    print(f"[ok] Wrote manifest: {manifest_csv}")


def simulate_binary_gp_with_clusters_all_nonstationary(
    M: int,
    *,
    # pass-through core knobs (subset shown; others may be in **kwargs)
    A: int = 16,
    num_clusters: int = 5,
    L_scale1: float = 0.05,
    cluster_sd_mult: float = 0.004,
    indiv_sd_choices: Sequence[float] = (0.75, 1.0, 1.5),
    N: int = 60,
    maf_range: Tuple[float, float] = (0.05, 0.5),
    kernel: str = "rbf",
    amplitude: float = 1.0,
    drift_b: Union[float, Sequence[float]] = 0.0,
    jitter: float = 1e-6,
    ls_multipliers: Sequence[float] = (100.0, 0.5, 0.004),
    ls_choice_probs: Optional[Sequence[float]] = None,
    intercept: float = 0.0,
    logit_noise_sd: float = 0.0,
    seed: Optional[int] = 1990,
    return_latent_eta: bool = False,
    # --- balancing controls (keyword-only) ---
    range_accepted: Tuple[float, float] = (0.45, 0.55),
    oversample_factor: int = 2,
    max_retries: int = 25,
    # extras to carry through (ignored by raw sim but useful in manifests)
    combo_hash: Optional[str] = None,
):
    """
    Balanced-version wrapper around the RAW clustered simulator.

    Strategy:
      - Generate a POOL of size M_pool = ceil(M * oversample_factor).
      - Compute y_pool fraction; choose k ∈ [lo*M, hi*M] positives if feasible.
      - Subselect exactly M rows (k positives, M-k negatives); retry with seed+attempt if needed.

    Parameters
    ----------
    range_accepted : (float, float)
        Target fraction range for y=1 in the final dataset (inclusive).
    oversample_factor : int
        Pool size multiplier; larger gives more flexibility to hit the range.
    max_retries : int
        Number of re-simulate attempts with seed+attempt if balancing fails.

    Returns
    -------
    sim_out_balanced : dict
        Same keys as RAW simulator: "A","G","beta","varying_mask","y","K_aa","kernel_info","drift_b"[,"eta"].
        (K_aa is subset to the selected individuals.)
    """
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be >= 1")
    lo, hi = float(range_accepted[0]), float(range_accepted[1])
    k_min = int(np.ceil(lo * M))
    k_max = int(np.floor(hi * M))
    if k_min > k_max:
        raise ValueError("range_accepted too narrow for given M (k_min > k_max).")

    def _pick_k(pos_pool: int, neg_pool: int) -> Optional[int]:
        # choose k in [k_min,k_max] feasible for pool, closest to M/2
        cands = [k for k in range(k_min, k_max + 1) if pos_pool >= k and neg_pool >= (M - k)]
        if not cands:
            return None
        mid = 0.5 * M
        return min(cands, key=lambda z: abs(z - mid))

    for attempt in range(int(max_retries)):
        seed_i = None if seed is None else int(seed) + attempt
        M_pool = max(M + 2, int(np.ceil(M * oversample_factor)))

        sim_out = simulate_binary_gp_with_clusters_all_nonstationary_raw(
            M=M_pool,
            A=A,
            num_clusters=num_clusters,
            L_scale1=L_scale1,
            cluster_sd_mult=cluster_sd_mult,
            indiv_sd_choices=indiv_sd_choices,
            N=N,
            maf_range=maf_range,
            kernel=kernel,
            amplitude=amplitude,
            drift_b=drift_b,
            jitter=jitter,
            ls_multipliers=ls_multipliers,
            ls_choice_probs=ls_choice_probs,
            intercept=intercept,
            logit_noise_sd=logit_noise_sd,
            seed=seed_i,
            return_latent_eta=return_latent_eta,
        )

        y_pool = np.asarray(sim_out["y"]).astype(int)
        pos_idx = np.where(y_pool == 1)[0]
        neg_idx = np.where(y_pool == 0)[0]

        k = _pick_k(pos_idx.size, neg_idx.size)
        if k is None:
            continue

        rng = np.random.default_rng(seed_i)
        take_pos = rng.choice(pos_idx, size=k, replace=False)
        take_neg = rng.choice(neg_idx, size=(M - k), replace=False)
        sel = np.concatenate([take_pos, take_neg], axis=0)
        rng.shuffle(sel)

        # subset arrays
        A_sub    = np.asarray(sim_out["A"])[sel, :]
        G_sub    = np.asarray(sim_out["G"])[sel, :]
        beta_sub = np.asarray(sim_out["beta"])[sel, :]
        y_sub    = y_pool[sel]
        Kaa_sub  = np.asarray(sim_out["K_aa"])[np.ix_(sel, sel)]

        out = {
            "A": A_sub,
            "G": G_sub,
            "beta": beta_sub,
            "varying_mask": np.asarray(sim_out["varying_mask"]),
            "y": y_sub,
            "K_aa": Kaa_sub,
            "kernel_info": sim_out["kernel_info"],  # keep as-is (contains per-SNP ℓ, centers, etc.)
            "drift_b": np.asarray(sim_out["drift_b"]),
        }
        if return_latent_eta and ("eta" in sim_out):
            out["eta"] = np.asarray(sim_out["eta"])[sel]
        return out

    raise RuntimeError(
        f"Could not assemble size-{M} with y∈[{lo:.2f},{hi:.2f}] after {max_retries} retries. "
        f"Try larger oversample_factor, a wider range_accepted, or adjust intercept/amplitude."
    )
