"""
Generate train/eval families of distribution-shifted datasets using
`simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw`.

Artifacts per accepted setting
-----------------------------
1) Train pickle: `<stem>_train.pkl`
2) Test pickle: `<stem>_test.pkl`
3) Latents pickle: `<stem>_latents.pkl`
4) One manifest row in either `train_manifest.csv` or `eval_manifest.csv`

Manifest columns (both train/eval)
----------------------------------
- setting_id
- role
- seed
- path_to_train
- path_to_test
- path_to_latents
- M
- A
- N
- num_clusters
- cluster_sd_mult
- indiv_sd_choices
- ancestry_group_cutpoints
- ancestry_group_size_divisors
- ls_multipliers
- ls_choice_probs
- setting_json
- sim_config_json

Train/Test pickle contents
--------------------------
Each train/test pickle is a dict with:
- table: pandas DataFrame with columns
    - A_0..A_{A-1}
    - G_0..G_{N-1}
    - y
    - cluster_id
    - sample_index
- columns: ordered list of DataFrame column names

Latents pickle contents
-----------------------
Each latents pickle is a dict with keys:
- A, G, y, beta, cluster_ids, kernel_info, drift_b
- train_indices, test_indices
- sim_config, setting, seed

`train_indices` and `test_indices` are integer row indices into the FULL simulated
population arrays (`A`, `G`, `y`, `beta`, `cluster_ids`) before split. They preserve
the exact membership mapping used to build train/test tables.


How to link train/test rows back to latents/effect sizes:

For any row in train/test table, take i = sample_index.
Then index latents arrays at i:
effect sizes: beta[i, :]
ancestry: A[i, :]
genotype: G[i, :]
cluster: cluster_ids[i]
label: y[i]
Also, row-order alignment is available:
train table row k corresponds to train_indices[k]
test table row k corresponds to test_indices[k]


run as 
python test_train_dist_shift.py /lustre/isaac24/scratch/adas23/PostR01SubmissionFeb2026/test_train_dist_shift/data/data2_300_training_100_testing --num_train_settings 300 --num_eval_settings 100 --base_seed 1990

"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from make_data_ver3 import simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _sample_ancestry_group_structure(
    rng: np.random.Generator,
    num_clusters: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Sample ancestry group cutpoints/divisors compatible with num_clusters.

    Rules implemented:
    - One major cluster (group 0): cluster index 0 only.
    - Second group has 1 or 2 clusters when possible.
    - Remaining clusters (if any) are the third group.
    - Group size divisors:
        second group divisor = 4
        third group divisor (if present) in {5, 6, 7}, equal probability.

    Returns
    -------
    ancestry_group_cutpoints : tuple[int, ...]
        End indices (exclusive) for group boundaries.
    ancestry_group_size_divisors : tuple[int, ...]
        Divisors expected by the simulator helper.
    """
    if num_clusters < 2:
        raise ValueError("num_clusters must be >= 2 for this pipeline.")

    max_second_group = min(2, num_clusters - 1)
    second_group_size = int(rng.choice(np.arange(1, max_second_group + 1)))

    second_group_end = 1 + second_group_size
    remaining = num_clusters - second_group_end

    cutpoints = (1, second_group_end)
    if remaining > 0:
        third_div = int(rng.choice([5, 6, 7]))
        divisors = (4, third_div)
    else:
        divisors = (4,)
    return cutpoints, divisors


def sample_setting(rng: np.random.Generator, *, A: int = 16) -> Dict:
    """
    Sample one dataset setting under the requested distributional constraints.

    Sampling rules:
    - M from {10_000, 15_000, 20_000, 25_000}
    - cluster_sd_mult from {0.004, 0.04, 0.2}
    - num_clusters from {2,3,4,5,6}, with higher probability on {3,4}
    - indiv_sd_choices is length num_clusters, mostly 1.0 with occasional 0.75/1.25
    - ancestry_group_cutpoints / ancestry_group_size_divisors follow requested structure
    - N from {40,60,100}
    - ls_multipliers: length 3 or 4, unique from {100,200,2,4,0.01,0.05}
    - ls_choice_probs: softmax(random logits), positive and sums to 1
    """
    M = int(rng.choice([10_000, 8_000  ]))
    # M = int(rng.choice([8000, 4000]))   # smaller M for testing
    num_clusters = int(rng.choice([2, 3, 4, 5], p=[0.20, 0.35, 0.35, 0.10]))
    cluster_sd_mult = float(rng.choice([0.004, 0.04, 0.2]))

    indiv_sd_choices = np.ones(num_clusters, dtype=float)
    n_special = int(rng.choice([0, 1, 2], p=[0.65, 0.25, 0.10]))
    n_special = min(n_special, num_clusters)
    if n_special > 0:
        special_idx = rng.choice(num_clusters, size=n_special, replace=False)
        special_vals = rng.choice([0.75, 1.25], size=n_special, replace=True)
        indiv_sd_choices[special_idx] = special_vals

    cutpoints, divisors = _sample_ancestry_group_structure(rng, num_clusters)

    N = int(rng.choice([30, 40, 60, 20]))

    ls_pool = np.array([100.0, 200.0, 50, 10, 2.0, 4.0, 0.5,0.1, 0.04, 0.01, 0.05, .02], dtype=float)
    ls_len = int(rng.choice([3, 4]))
    ls_multipliers = tuple(float(x) for x in rng.choice(ls_pool, size=ls_len, replace=False))
    logits = rng.uniform(0.0, 1.0, size=ls_len)
    ls_choice_probs = tuple(float(x) for x in _softmax(logits))

    setting = {
        "M": M,
        "A": int(A),
        "N": int(N),
        "num_clusters": num_clusters,
        "cluster_sd_mult": cluster_sd_mult,
        "indiv_sd_choices": tuple(float(x) for x in indiv_sd_choices),
        "ancestry_group_cutpoints": tuple(int(x) for x in cutpoints),
        "ancestry_group_size_divisors": tuple(int(x) for x in divisors),
        "ls_multipliers": ls_multipliers,
        "ls_choice_probs": ls_choice_probs,
    }
    return setting


def _make_train_test_split_indices(
    rng: np.random.Generator,
    cluster_ids: np.ndarray,
    *,
    test_per_cluster: int = 50,
    min_train_to_test_ratio: float = 2.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build train/test split indices with fixed test size per cluster.

        Returns None if any cluster cannot satisfy:
    - at least `test_per_cluster` samples for test
    - train count >= min_train_to_test_ratio * test_per_cluster

        Notes
        -----
        - Cluster-wise sampling is used to pick test members.
        - Final `test_idx` and `train_idx` are both shuffled, so downstream tables are
            not grouped by cluster.
    """
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)

    test_idx_parts: List[np.ndarray] = []
    for cluster in unique_clusters:
        idx = np.where(cluster_ids == cluster)[0]
        if idx.size < test_per_cluster:
            return None
        rng.shuffle(idx)
        test_idx_parts.append(idx[:test_per_cluster])

    test_idx = np.concatenate(test_idx_parts)
    rng.shuffle(test_idx)
    test_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    test_mask[test_idx] = True
    train_idx = np.where(~test_mask)[0]
    rng.shuffle(train_idx)

    min_train_required = int(np.ceil(min_train_to_test_ratio * test_per_cluster))
    train_clusters, train_counts = np.unique(cluster_ids[train_idx], return_counts=True)
    train_count_map = {int(c): int(n) for c, n in zip(train_clusters, train_counts)}

    for cluster in unique_clusters:
        if train_count_map.get(int(cluster), 0) < min_train_required:
            return None

    return train_idx, test_idx


def _build_feature_table(
    A_mat: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    row_indices: np.ndarray,
) -> pd.DataFrame:
    A_slice = np.asarray(A_mat[row_indices])
    G_slice = np.asarray(G[row_indices])
    y_slice = np.asarray(y[row_indices]).astype(int)
    c_slice = np.asarray(cluster_ids[row_indices]).astype(int)

    A_cols = [f"A_{i}" for i in range(A_slice.shape[1])]
    G_cols = [f"G_{j}" for j in range(G_slice.shape[1])]

    X = np.concatenate([A_slice, G_slice], axis=1)
    df = pd.DataFrame(X, columns=A_cols + G_cols)
    df["y"] = y_slice
    df["cluster_id"] = c_slice
    df["sample_index"] = row_indices.astype(int)
    return df


def _save_pickle(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _generate_one_triplet(
    *,
    output_dir: str,
    setting_idx: int,
    role: str,
    seed: int,
    max_attempts: int = 30,
) -> Optional[Dict]:
    """
    Generate one train/test/latent triplet. Retries up to max_attempts
    if the cluster/split constraints are not met.

    Returns a manifest row dict, or None on total failure.
    """
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        rng = np.random.default_rng(attempt_seed)
        setting = sample_setting(rng)

        sim_cfg = dict(
            M=setting["M"],
            A=setting["A"],
            num_clusters=setting["num_clusters"],
            L_scale1=1.0,
            cluster_sd_mult=setting["cluster_sd_mult"],
            indiv_sd_choices=setting["indiv_sd_choices"],
            ancestry_group_cutpoints=setting["ancestry_group_cutpoints"],
            ancestry_group_size_divisors=setting["ancestry_group_size_divisors"],
            noise_percent=10.0,
            N=setting["N"],
            maf_range=(0.1, 0.5),
            cluster_maf_offset=0.08,
            kernel="rbf",
            amplitude=1.0,
            drift_b=0.0,
            jitter=1e-6,
            ls_multipliers=setting["ls_multipliers"],
            ls_choice_probs=setting["ls_choice_probs"],
            intercept=0.0,
            logit_noise_sd=0.0,
            seed=attempt_seed,
            return_latent_eta=False,
        )

        sim_out = simulate_binary_gp_with_clusters_all_nonstationary_clusteredmafs_raw(**sim_cfg)

        split = _make_train_test_split_indices(
            rng,
            np.asarray(sim_out["cluster_ids"]),
            test_per_cluster=50,
            min_train_to_test_ratio=2.0,
        )
        if split is None:
            continue

        train_idx, test_idx = split
        out_dir = Path(output_dir)

        train_df = _build_feature_table(
            np.asarray(sim_out["A"]),
            np.asarray(sim_out["G"]),
            np.asarray(sim_out["y"]),
            np.asarray(sim_out["cluster_ids"]),
            train_idx,
        )
        test_df = _build_feature_table(
            np.asarray(sim_out["A"]),
            np.asarray(sim_out["G"]),
            np.asarray(sim_out["y"]),
            np.asarray(sim_out["cluster_ids"]),
            test_idx,
        )

        stem = f"distshift_{role}_{setting_idx:04d}_seed{attempt_seed}"
        train_path = out_dir / f"{stem}_train.pkl"
        test_path = out_dir / f"{stem}_test.pkl"
        latents_path = out_dir / f"{stem}_latents.pkl"

        train_payload = {"table": train_df, "columns": list(train_df.columns)}
        test_payload = {"table": test_df, "columns": list(test_df.columns)}
        latents_payload = {
            "A": np.asarray(sim_out["A"]),
            "G": np.asarray(sim_out["G"]),
            "y": np.asarray(sim_out["y"]).astype(int),
            "beta": np.asarray(sim_out["beta"]),
            "cluster_ids": np.asarray(sim_out["cluster_ids"]).astype(int),
            "kernel_info": sim_out.get("kernel_info", {}),
            "drift_b": np.asarray(sim_out.get("drift_b", [])),
            "train_indices": np.asarray(train_idx).astype(int),
            "test_indices": np.asarray(test_idx).astype(int),
            "sim_config": sim_cfg,
            "setting": setting,
            "seed": int(attempt_seed),
        }

        _save_pickle(train_path, train_payload)
        _save_pickle(test_path, test_payload)
        _save_pickle(latents_path, latents_payload)

        row = {
            "setting_id": int(setting_idx),
            "role": role,
            "seed": int(attempt_seed),
            "path_to_train": str(train_path),
            "path_to_test": str(test_path),
            "path_to_latents": str(latents_path),
            "M": int(setting["M"]),
            "A": int(setting["A"]),
            "N": int(setting["N"]),
            "num_clusters": int(setting["num_clusters"]),
            "cluster_sd_mult": float(setting["cluster_sd_mult"]),
            "indiv_sd_choices": json.dumps(setting["indiv_sd_choices"]),
            "ancestry_group_cutpoints": json.dumps(setting["ancestry_group_cutpoints"]),
            "ancestry_group_size_divisors": json.dumps(setting["ancestry_group_size_divisors"]),
            "ls_multipliers": json.dumps(setting["ls_multipliers"]),
            "ls_choice_probs": json.dumps(setting["ls_choice_probs"]),
            "setting_json": json.dumps(setting),
            "sim_config_json": json.dumps(sim_cfg),
        }
        return row

    # all attempts failed
    return None


def _worker(args_tuple):
    """
    Top-level picklable worker called by Pool.imap_unordered.
    Returns (setting_idx, role, row_or_none, error_string_or_none).
    """
    output_dir, setting_idx, role, seed, max_attempts = args_tuple
    try:
        row = _generate_one_triplet(
            output_dir=output_dir,
            setting_idx=setting_idx,
            role=role,
            seed=seed,
            max_attempts=max_attempts,
        )
        return (setting_idx, role, row, None)
    except Exception:
        import traceback
        return (setting_idx, role, None, traceback.format_exc())


def generate_train_eval_family(
    *,
    output_dir: str,
    num_train_settings: int,
    num_eval_settings: int,
    base_seed: int = 1990,
    max_attempts_per_setting: int = 30,
    n_proc: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate train/eval setting families and write two manifests.

    Inputs:
    - output_dir
    - num_train_settings
    - num_eval_settings
    - n_proc: number of worker processes (1 = serial)

    Outputs:
    - train manifest saved to: <output_dir>/train_manifest.csv
    - eval manifest saved to:  <output_dir>/eval_manifest.csv
    - generation log saved to: <output_dir>/generation.log
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── logging to file and stdout ───────────────────────────────────
    log_path = out_dir / "generation.log"
    logger = logging.getLogger("distshift_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("distshift orchestrator starting")
    logger.info(f"  output_dir            = {output_dir}")
    logger.info(f"  num_train_settings    = {num_train_settings}")
    logger.info(f"  num_eval_settings     = {num_eval_settings}")
    logger.info(f"  base_seed             = {base_seed}")
    logger.info(f"  max_attempts/setting  = {max_attempts_per_setting}")
    logger.info(f"  n_proc                = {n_proc}")

    # ── build work items ─────────────────────────────────────────────
    work_items = []
    for idx in range(num_train_settings):
        seed = int(base_seed + idx * 1000)
        work_items.append((str(out_dir), idx, "train", seed, max_attempts_per_setting))
    for idx in range(num_eval_settings):
        seed = int(base_seed + 500_000 + idx * 1000)
        work_items.append((str(out_dir), idx, "eval", seed, max_attempts_per_setting))

    total_tasks = len(work_items)
    logger.info(f"Total tasks: {total_tasks} ({num_train_settings} train + {num_eval_settings} eval)")

    # ── dispatch ─────────────────────────────────────────────────────
    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    n_done = 0
    n_failed = 0
    t0 = time.perf_counter()

    if n_proc <= 1:
        for item in work_items:
            setting_idx, role, row, err = _worker(item)
            n_done += 1
            if err is not None:
                n_failed += 1
                logger.error(f"EXCEPTION {role} #{setting_idx}: {err}")
            elif row is None:
                n_failed += 1
                logger.warning(f"FAILED {role} #{setting_idx}: could not satisfy split constraints after all attempts")
            else:
                if role == "train":
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
                logger.info(f"[{n_done}/{total_tasks}] OK {role} #{setting_idx}  (seed={row['seed']}, Kc={row['num_clusters']}, N={row['N']})")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_proc) as pool:
            for setting_idx, role, row, err in pool.imap_unordered(_worker, work_items):
                n_done += 1
                if err is not None:
                    n_failed += 1
                    logger.error(f"EXCEPTION {role} #{setting_idx}: {err}")
                elif row is None:
                    n_failed += 1
                    logger.warning(f"FAILED {role} #{setting_idx}: could not satisfy split constraints after all attempts")
                else:
                    if role == "train":
                        train_rows.append(row)
                    else:
                        eval_rows.append(row)
                    logger.info(f"[{n_done}/{total_tasks}] OK {role} #{setting_idx}  (seed={row['seed']}, Kc={row['num_clusters']}, N={row['N']})")

    elapsed = time.perf_counter() - t0

    # ── sort rows by setting_id for deterministic manifests ──────────
    train_rows.sort(key=lambda r: r["setting_id"])
    eval_rows.sort(key=lambda r: r["setting_id"])

    train_manifest = pd.DataFrame(train_rows)
    eval_manifest = pd.DataFrame(eval_rows)

    train_manifest_path = out_dir / "train_manifest.csv"
    eval_manifest_path = out_dir / "eval_manifest.csv"

    train_manifest.to_csv(train_manifest_path, index=False)
    eval_manifest.to_csv(eval_manifest_path, index=False)

    logger.info(f"Wrote train manifest: {len(train_manifest)} rows -> {train_manifest_path}")
    logger.info(f"Wrote eval  manifest: {len(eval_manifest)} rows -> {eval_manifest_path}")
    logger.info(f"Failed: {n_failed}/{total_tasks}")
    logger.info(f"Elapsed: {elapsed:.1f}s  ({elapsed/max(n_done,1):.2f}s/task)")
    if n_failed > 0:
        logger.warning(f"{n_failed} task(s) failed — check log for details.")

    return train_manifest, eval_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/eval dist-shift datasets with paired train/test/latents manifests."
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--num_train_settings", type=int, required=True)
    parser.add_argument("--num_eval_settings", type=int, required=True)
    parser.add_argument("--base_seed", type=int, default=1990)
    parser.add_argument("--max_attempts_per_setting", type=int, default=30)
    parser.add_argument(
        "--n_proc", type=int, default=0,
        help="Number of worker processes. 0 = auto (os.cpu_count()-1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve n_proc: 0 means auto-detect; reserve 1 CPU for the main process
    n_proc = args.n_proc
    if n_proc <= 0:
        n_proc = max(1, (os.cpu_count() or 1) - 1)
    # Cap to at most total tasks so we don't over-allocate
    total = args.num_train_settings + args.num_eval_settings
    n_proc = min(n_proc, total)

    print(f"Detected {os.cpu_count()} CPUs, using {n_proc} worker processes (1 reserved for main)")

    train_manifest, eval_manifest = generate_train_eval_family(
        output_dir=args.output_dir,
        num_train_settings=args.num_train_settings,
        num_eval_settings=args.num_eval_settings,
        base_seed=args.base_seed,
        max_attempts_per_setting=args.max_attempts_per_setting,
        n_proc=n_proc,
    )
    print(f"[done] train={len(train_manifest)} eval={len(eval_manifest)}")