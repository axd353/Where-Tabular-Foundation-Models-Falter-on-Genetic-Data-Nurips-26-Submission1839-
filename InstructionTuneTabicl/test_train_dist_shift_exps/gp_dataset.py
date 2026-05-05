"""
gp_dataset.py
=============
IterableDataset backed by the GP-simulation tabular datasets stored on disk.

Each `__next__` call ("batch") constructs `batch_size` tabular in-context
learning tasks:

  For each task (one manifest row):
    - Load TEST pickle  → sample 90 % without replacement → N_tst rows
    - Load TRAIN pickle → sample 10 × N_tst rows (with replacement if
                          the train set is smaller)
    - Concatenate sequence: [ train_context (10·N_tst) | test_query (N_tst) ]
    - Drop 'cluster_id' and 'sample_index'; 'y' is the label.
    - Remap y labels to 0-indexed integers.
    - Pad features to self.max_features with zeros.

Returned tensors match the interface expected by tabicl's Trainer:
  X          : NestedTensor  (B, T_i, max_features)   dtype=float32
  y          : NestedTensor  (B, T_i)                 dtype=float32  (int vals)
  d          : LongTensor    (B,)                     actual # features
  seq_lens   : LongTensor    (B,)
  train_sizes: LongTensor    (B,)

NestedTensors are used because different manifest rows can produce
different sequence lengths.  Trainer.run_batch() calls
`t.to_padded_tensor(padding=0.0)` which pads them to the batch-maximum
before splitting into micro-batches.  Trainer.align_micro_batch() then
trims each micro-batch back to its actual (seq_len, n_features).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: str) -> dict:
    """Load a pickle file and return its contents."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (everything except 'y', 'cluster_id', 'sample_index')."""
    exclude = {"y", "cluster_id", "sample_index"}
    return [c for c in df.columns if c not in exclude]


def _normalize_labels(y_arr: np.ndarray) -> np.ndarray:
    """Remap arbitrary integer labels to 0-indexed contiguous integers."""
    unique_vals = np.unique(y_arr)
    mapping = {v: i for i, v in enumerate(unique_vals)}
    return np.array([mapping[v] for v in y_arr], dtype=np.int64)


def _build_task(
    train_pkl_path: str,
    test_pkl_path: str,
    max_features: int,
    rng: np.random.Generator,
    pkl_cache: Optional[Dict[str, dict]] = None,
) -> Tuple[Tensor, Tensor, int, int, int]:
    """
    Build one in-context learning task from a train/test pickle pair.

    Parameters
    ----------
    train_pkl_path, test_pkl_path : str
        Paths to the train and test pickles for one manifest row.
    max_features : int
        Global maximum number of features; shorter rows are zero-padded.
    rng : np.random.Generator
        Numpy RNG to use for sampling.
    pkl_cache : dict, optional
        In-memory cache keyed by path string to avoid repeated disk reads.

    Returns
    -------
    X_i      : float32 Tensor  (T, max_features)  – full concatenated sequence
    y_i      : float32 Tensor  (T,)               – labels (0-indexed, as float)
    n_feat   : int             – actual number of features
    seq_len  : int             – T = train_size + test_size
    train_sz : int             – number of context (train) rows
    """

    # ---- Load pickles (with caching) ----------------------------------------
    def _cached_load(path: str) -> dict:
        if pkl_cache is not None:
            if path not in pkl_cache:
                pkl_cache[path] = _load_pkl(path)
            return pkl_cache[path]
        return _load_pkl(path)

    train_data = _cached_load(train_pkl_path)
    test_data  = _cached_load(test_pkl_path)

    train_df: pd.DataFrame = train_data["table"]
    test_df:  pd.DataFrame = test_data["table"]

    feat_cols = _get_feature_cols(test_df)   # same schema for train and test
    n_feat = len(feat_cols)

    # ---- Sample test rows (90 % without replacement) ------------------------
    N_test_total = len(test_df)
    N_tst = max(1, int(0.9 * N_test_total))
    test_idx = rng.choice(N_test_total, size=N_tst, replace=False)
    test_sample = test_df.iloc[test_idx]

    # ---- Sample train rows (10 × N_tst, with replacement if too small) ------
    N_train_needed = 10 * N_tst
    N_train_total  = len(train_df)
    replace_flag   = N_train_needed > N_train_total
    if replace_flag:
        warnings.warn(
            f"[GPPriorDataset] Sampling with replacement: need {N_train_needed} train rows "
            f"but only {N_train_total} available.\n"
            f"  train pkl : {train_pkl_path}\n"
            f"  test  pkl : {test_pkl_path}\n"
            f"  test  rows available (before 90% subsample): {N_test_total}",
            stacklevel=2,
        )
    train_idx = rng.choice(N_train_total, size=N_train_needed, replace=replace_flag)
    train_sample = train_df.iloc[train_idx]

    # ---- Extract features and labels ----------------------------------------
    X_train_np = train_sample[feat_cols].values.astype(np.float32)
    y_train_np = train_sample["y"].values

    X_test_np  = test_sample[feat_cols].values.astype(np.float32)
    y_test_np  = test_sample["y"].values

    # Remap labels: fit mapping on train labels, apply to both
    all_labels = np.concatenate([y_train_np, y_test_np])
    all_labels_norm = _normalize_labels(all_labels)
    y_train_np = all_labels_norm[:N_train_needed]
    y_test_np  = all_labels_norm[N_train_needed:]

    # ---- Build full sequence: [train_context | test_query] ------------------
    T = N_train_needed + N_tst

    # Pad feature dimension to max_features
    X_i = np.zeros((T, max_features), dtype=np.float32)
    X_i[:N_train_needed, :n_feat] = X_train_np
    X_i[N_train_needed:, :n_feat] = X_test_np

    y_i = np.concatenate([y_train_np, y_test_np]).astype(np.float32)

    return (
        torch.from_numpy(X_i),           # (T, max_features)
        torch.from_numpy(y_i),           # (T,)
        n_feat,
        T,
        N_train_needed,
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class GPPriorDataset(IterableDataset):
    """
    Infinite IterableDataset over GP-simulation tabular datasets.

    Each call to `__next__` returns one *batch* of in-context learning
    tasks constructed from randomly sampled manifest rows.

    Parameters
    ----------
    data_dir : str | Path
        Root directory that contains `train_manifest.csv` and the per-setting
        pickle files.
    batch_size : int
        Number of tasks (manifest rows) per batch.
    max_features : int | None
        Maximum number of features across all datasets.  If None, computed
        from the manifest using max(A + N).
    manifest_filename : str
        CSV filename inside data_dir (default: "train_manifest.csv").
    seed : int
        Base seed for numpy RNG used during sampling.  The actual seed is
        incremented every batch so that consecutive batches differ.
    cache_pickles : bool
        If True, keep loaded pickles in RAM.  Speeds up training but
        increases memory usage.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        max_features: Optional[int] = None,
        manifest_filename: str = "train_manifest.csv",
        seed: int = 42,
        cache_pickles: bool = True,
    ):
        super().__init__()

        self.data_dir   = Path(data_dir)
        self.batch_size = batch_size
        self.seed       = seed

        # ---- Load manifest --------------------------------------------------
        manifest_path = self.data_dir / manifest_filename
        self.manifest: pd.DataFrame = pd.read_csv(manifest_path)

        # ---- Determine max_features -----------------------------------------
        if max_features is None:
            # Columns A and N hold the ancestry and genotype feature counts
            self.max_features = int((self.manifest["A"] + self.manifest["N"]).max())
        else:
            self.max_features = int(max_features)

        # ---- Optional pickle cache ------------------------------------------
        self._pkl_cache: Optional[Dict[str, dict]] = {} if cache_pickles else None

        # ---- RNG (will be advanced per batch) --------------------------------
        self._rng = np.random.default_rng(seed)
        self._batch_count = 0  # used to advance seed across batches

    # ------------------------------------------------------------------
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample `batch_size` manifest rows and construct one training batch.

        Returns
        -------
        X          : NestedTensor (B, T_i, max_features)  float32
        y          : NestedTensor (B, T_i)                float32
        d          : LongTensor   (B,)   – actual feature counts
        seq_lens   : LongTensor   (B,)
        train_sizes: LongTensor   (B,)
        """
        B = batch_size if batch_size is not None else self.batch_size

        # Use a fresh RNG per batch so batches are reproducible by batch index
        rng = np.random.default_rng(self.seed + self._batch_count)
        self._batch_count += 1

        # Sample B rows from the manifest (with replacement if B > len(manifest))
        row_indices = rng.choice(len(self.manifest), size=B, replace=(B > len(self.manifest)))

        X_list, y_list, d_list, seq_len_list, train_size_list = [], [], [], [], []

        for row_idx in row_indices:
            row = self.manifest.iloc[int(row_idx)]

            train_pkl = str(row["path_to_train"])
            test_pkl  = str(row["path_to_test"])

            X_i, y_i, n_feat, seq_len, train_sz = _build_task(
                train_pkl_path=train_pkl,
                test_pkl_path=test_pkl,
                max_features=self.max_features,
                rng=rng,
                pkl_cache=self._pkl_cache,
            )

            X_list.append(X_i)
            y_list.append(y_i)
            d_list.append(n_feat)
            seq_len_list.append(seq_len)
            train_size_list.append(train_sz)

        # NestedTensor allows variable T_i across the batch.
        # Trainer.run_batch() calls .to_padded_tensor(0.0) to align.
        X_nested = nested_tensor(X_list, dtype=torch.float32)
        y_nested = nested_tensor(y_list, dtype=torch.float32)

        d          = torch.tensor(d_list,          dtype=torch.long)
        seq_lens   = torch.tensor(seq_len_list,    dtype=torch.long)
        train_sizes = torch.tensor(train_size_list, dtype=torch.long)

        return X_nested, y_nested, d, seq_lens, train_sizes

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self) -> "GPPriorDataset":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"GPPriorDataset(\n"
            f"  data_dir:     {self.data_dir}\n"
            f"  manifest rows:{len(self.manifest)}\n"
            f"  batch_size:   {self.batch_size}\n"
            f"  max_features: {self.max_features}\n"
            f"  cache_pickles:{self._pkl_cache is not None}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Eval-batch preparation (fixed samples, used every step)
# ---------------------------------------------------------------------------

def prepare_eval_batch(
    data_dir: str,
    n_eval: int = 10,
    eval_manifest_filename: str = "eval_manifest.csv",
    max_features: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[Tensor], List[Tensor], List[int], List[int], List[int]]:
    """
    Pre-generate a fixed evaluation batch from the first `n_eval` rows of
    `eval_manifest.csv`.  The same samples are reused every step so that
    evaluation metrics are comparable across steps.

    Parameters
    ----------
    data_dir : str
        Root directory containing eval_manifest.csv and pickles.
    n_eval : int
        Number of manifest rows to evaluate (first n_eval entries).
    eval_manifest_filename : str
        Filename of the eval manifest CSV.
    max_features : int | None
        Must match GPPriorDataset.max_features.  If None, computed from
        eval manifest using max(A + N).
    seed : int
        Fixed seed – guarantees identical samples on every evaluation call.

    Returns
    -------
    List of (X_i, y_i, n_feat, seq_len, train_size) per eval dataset.
    Each X_i is a plain 2-D float32 Tensor (T, max_features) without
    batch dimension; they are consumed individually by the eval loop.
    """
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / eval_manifest_filename)

    # Only use first n_eval rows
    manifest = manifest.head(n_eval)

    if max_features is None:
        max_features = int((manifest["A"] + manifest["N"]).max())

    rng = np.random.default_rng(seed)
    results = []
    pkl_cache: Dict[str, dict] = {}

    for _, row in manifest.iterrows():
        X_i, y_i, n_feat, seq_len, train_sz = _build_task(
            train_pkl_path=str(row["path_to_train"]),
            test_pkl_path=str(row["path_to_test"]),
            max_features=max_features,
            rng=rng,
            pkl_cache=pkl_cache,
        )
        results.append((X_i, y_i, n_feat, seq_len, train_sz))

    return results
