"""
Generate mixed HGP + PRS-CSx train/eval families of distribution-shifted
datasets.

Half the settings are drawn from the Hierarchical Gaussian Process pipeline
(``test_train_dist_shift.py``) and the other half from the PRS-CSx pipeline
(``PRS-CSx/test_train_dist_shift_prs_csx.py``).

A *single* process pool dispatches work items for both families so the total
number of workers never exceeds ``n_proc - 1``.

Manifest extras vs. the single-family scripts
----------------------------------------------
- ``task_family`` column: ``"HGP"`` or ``"PRS-CSx"``
- Rows are **shuffled** (seeded) before saving so that HGP and PRS-CSx
  settings are interleaved rather than contiguous.

Example
-------
python test_train_dist_shift_mixed.py /lustre/isaac24/scratch/adas23/PostR01SubmissionFeb2026/test_train_dist_shift/data/PrsCsx_HGP_mixed/400_train_90_eval_tasks --num_train_settings 400 --num_eval_settings 90 --base_seed 1990 --n_proc 39
when resuming:
python test_train_dist_shift_mixed.py /lustre/isaac24/scratch/adas23/PostR01SubmissionFeb2026/test_train_dist_shift/data/PrsCsx_HGP_mixed/400_train_90_eval_tasks --num_train_settings 400 --num_eval_settings 90 --base_seed 1990 --n_proc 25 --resume

"""

import argparse
import importlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths needed by worker processes to import the two family modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
_PRSCSX_DIR = str(Path(__file__).resolve().parent / "PRS-CSx")
_TRIPLET_RE = re.compile(
    r"^(distshift|prscsx)_(train|eval)_(\d{4})_seed(\d+)_(train|test|latents)\.pkl$"
)


def _load_pickle(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_triplet_name(path: Path) -> Optional[Tuple[str, str, int, int, str]]:
    match = _TRIPLET_RE.match(path.name)
    if match is None:
        return None

    prefix, role, setting_idx, seed, artifact_kind = match.groups()
    family = "HGP" if prefix == "distshift" else "PRS-CSx"
    return family, role, int(setting_idx), int(seed), artifact_kind


def _build_manifest_row_from_existing_triplet(
    *,
    latents_path: Path,
    family: str,
    role: str,
    setting_idx: int,
    seed_from_name: int,
) -> Optional[Dict]:
    prefix = "distshift" if family == "HGP" else "prscsx"
    stem = f"{prefix}_{role}_{setting_idx:04d}_seed{seed_from_name}"
    train_path = latents_path.with_name(f"{stem}_train.pkl")
    test_path = latents_path.with_name(f"{stem}_test.pkl")

    if not train_path.exists() or not test_path.exists():
        return None

    latents_payload = _load_pickle(latents_path)
    setting = dict(latents_payload["setting"])
    sim_cfg = dict(latents_payload["sim_config"])
    seed = int(latents_payload.get("seed", seed_from_name))

    row = {
        "setting_id": int(setting_idx),
        "role": role,
        "seed": seed,
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
        "setting_json": json.dumps(setting),
        "sim_config_json": json.dumps(sim_cfg),
        "task_family": family,
    }

    if family == "HGP":
        row["ls_multipliers"] = json.dumps(setting["ls_multipliers"])
        row["ls_choice_probs"] = json.dumps(setting["ls_choice_probs"])
    else:
        row["phi"] = float(setting["phi"])
        row["r_g"] = float(setting["r_g"])
        row["pi_causal"] = float(setting["pi_causal"])
        row["alpha"] = float(setting["alpha"])
        row["h"] = float(setting["h"])
        row["h_sq_k"] = json.dumps(setting["h_sq_k"])
        row["F_ST_k"] = json.dumps(setting["F_ST_k"])

    return row


def _shuffle_manifest(df: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _write_manifest_checkpoints(
    *,
    out_dir: Path,
    train_rows: List[Dict],
    eval_rows: List[Dict],
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_manifest = _shuffle_manifest(pd.DataFrame(train_rows), random_state=base_seed)
    eval_manifest = _shuffle_manifest(pd.DataFrame(eval_rows), random_state=base_seed + 1)

    train_manifest.to_csv(out_dir / "train_manifest.csv", index=False)
    eval_manifest.to_csv(out_dir / "eval_manifest.csv", index=False)
    return train_manifest, eval_manifest


def _recover_existing_rows(
    *,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[Dict], Set[Tuple[str, str, int]]]:
    chosen_latents: Dict[Tuple[str, str, int], Path] = {}
    duplicates = 0

    for latents_path in out_dir.glob("*_latents.pkl"):
        parsed = _parse_triplet_name(latents_path)
        if parsed is None:
            continue
        family, role, setting_idx, _seed, artifact_kind = parsed
        if artifact_kind != "latents":
            continue

        key = (family, role, setting_idx)
        existing = chosen_latents.get(key)
        if existing is None or latents_path.stat().st_mtime > existing.stat().st_mtime:
            if existing is not None:
                duplicates += 1
            chosen_latents[key] = latents_path

    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    completed_keys: Set[Tuple[str, str, int]] = set()
    incomplete_triplets = 0

    for (family, role, setting_idx), latents_path in sorted(chosen_latents.items()):
        parsed = _parse_triplet_name(latents_path)
        if parsed is None:
            continue

        row = _build_manifest_row_from_existing_triplet(
            latents_path=latents_path,
            family=family,
            role=role,
            setting_idx=setting_idx,
            seed_from_name=parsed[3],
        )
        if row is None:
            incomplete_triplets += 1
            continue

        completed_keys.add((family, role, setting_idx))
        if role == "train":
            train_rows.append(row)
        else:
            eval_rows.append(row)

    logger.info(
        "Recovered %d completed triplet(s) already on disk: %d train, %d eval",
        len(completed_keys),
        len(train_rows),
        len(eval_rows),
    )
    if duplicates > 0:
        logger.warning(
            "Found %d duplicate completed setting(s); kept the newest latents file for each key.",
            duplicates,
        )
    if incomplete_triplets > 0:
        logger.warning(
            "Found %d incomplete triplet(s) with missing train/test siblings; they will be regenerated.",
            incomplete_triplets,
        )

    return train_rows, eval_rows, completed_keys


# ====================================================================
# Worker (top-level, picklable for mp "spawn" context)
# ====================================================================

def _worker_mixed(args_tuple):
    """
    Dispatch to either HGP or PRS-CSx ``_generate_one_triplet``.

    args_tuple: (output_dir, setting_idx, role, seed, max_attempts, family)
    Returns:  (setting_idx, role, row_or_none, error_string_or_none, family)
    """
    output_dir, setting_idx, role, seed, max_attempts, family = args_tuple

    # Ensure both source directories are importable in spawned workers
    import sys as _sys
    if _PROJECT_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJECT_ROOT)
    if _PRSCSX_DIR not in _sys.path:
        _sys.path.insert(0, _PRSCSX_DIR)

    try:
        module_name = "test_train_dist_shift" if family == "HGP" else "test_train_dist_shift_prs_csx"
        _generate_one_triplet = importlib.import_module(module_name)._generate_one_triplet

        row = _generate_one_triplet(
            output_dir=output_dir,
            setting_idx=setting_idx,
            role=role,
            seed=seed,
            max_attempts=max_attempts,
        )
        if row is not None:
            row["task_family"] = family
        return (setting_idx, role, row, None, family)

    except Exception:
        import traceback
        return (setting_idx, role, None, traceback.format_exc(), family)


# ====================================================================
# Main orchestrator
# ====================================================================

def generate_mixed_train_eval(
    *,
    output_dir: str,
    num_train_settings: int,
    num_eval_settings: int,
    base_seed: int = 1990,
    max_attempts_per_setting: int = 30,
    n_proc: int = 1,
    resume: bool = False,
    rebuild_manifests_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mixed HGP + PRS-CSx train/eval families and write manifests.

    Returns (train_manifest, eval_manifest) DataFrames.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── logging to file and stdout ───────────────────────────────────
    log_path = out_dir / "generation.log"
    logger = logging.getLogger("mixed_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # ── split counts: half HGP, half PRS-CSx ─────────────────────────
    n_hgp_train = num_train_settings // 2
    n_prscsx_train = num_train_settings - n_hgp_train
    n_hgp_eval = num_eval_settings // 2
    n_prscsx_eval = num_eval_settings - n_hgp_eval

    logger.info("mixed HGP + PRS-CSx orchestrator starting")
    logger.info(f"  output_dir            = {output_dir}")
    logger.info(f"  num_train_settings    = {num_train_settings}  (HGP={n_hgp_train}, PRS-CSx={n_prscsx_train})")
    logger.info(f"  num_eval_settings     = {num_eval_settings}  (HGP={n_hgp_eval}, PRS-CSx={n_prscsx_eval})")
    logger.info(f"  base_seed             = {base_seed}")
    logger.info(f"  max_attempts/setting  = {max_attempts_per_setting}")
    logger.info(f"  n_proc                = {n_proc}")
    logger.info(f"  resume                = {resume}")
    logger.info(f"  rebuild_manifests_only= {rebuild_manifests_only}")

    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    completed_keys: Set[Tuple[str, str, int]] = set()

    if resume or rebuild_manifests_only:
        train_rows, eval_rows, completed_keys = _recover_existing_rows(
            out_dir=out_dir,
            logger=logger,
        )
        train_manifest, eval_manifest = _write_manifest_checkpoints(
            out_dir=out_dir,
            train_rows=train_rows,
            eval_rows=eval_rows,
            base_seed=base_seed,
        )
        logger.info(
            "Wrote recovered manifest checkpoint: %d train row(s), %d eval row(s)",
            len(train_manifest),
            len(eval_manifest),
        )
    else:
        train_manifest = pd.DataFrame()
        eval_manifest = pd.DataFrame()

    if rebuild_manifests_only:
        logger.info("Manifest rebuild requested; skipping new dataset generation.")
        return train_manifest, eval_manifest

    # ── build work items ─────────────────────────────────────────────
    #   Seed ranges are non-overlapping across family × role:
    #     HGP      train : base_seed + idx * 1000
    #     HGP      eval  : base_seed +   500_000 + idx * 1000
    #     PRS-CSx  train : base_seed + 1_000_000 + idx * 1000
    #     PRS-CSx  eval  : base_seed + 1_500_000 + idx * 1000
    work_items = []

    # HGP train
    for idx in range(n_hgp_train):
        if ("HGP", "train", idx) in completed_keys:
            continue
        seed = int(base_seed + idx * 1000)
        work_items.append(
            (str(out_dir), idx, "train", seed, max_attempts_per_setting, "HGP")
        )
    # PRS-CSx train  (setting_idx continues from HGP)
    for idx in range(n_prscsx_train):
        setting_idx = n_hgp_train + idx
        if ("PRS-CSx", "train", setting_idx) in completed_keys:
            continue
        seed = int(base_seed + 1_000_000 + idx * 1000)
        work_items.append(
            (str(out_dir), setting_idx, "train", seed, max_attempts_per_setting, "PRS-CSx")
        )
    # HGP eval
    for idx in range(n_hgp_eval):
        if ("HGP", "eval", idx) in completed_keys:
            continue
        seed = int(base_seed + 500_000 + idx * 1000)
        work_items.append(
            (str(out_dir), idx, "eval", seed, max_attempts_per_setting, "HGP")
        )
    # PRS-CSx eval  (setting_idx continues from HGP)
    for idx in range(n_prscsx_eval):
        setting_idx = n_hgp_eval + idx
        if ("PRS-CSx", "eval", setting_idx) in completed_keys:
            continue
        seed = int(base_seed + 1_500_000 + idx * 1000)
        work_items.append(
            (str(out_dir), setting_idx, "eval", seed, max_attempts_per_setting, "PRS-CSx")
        )

    total_tasks = len(work_items)
    logger.info(
        f"Total tasks: {total_tasks}  "
        f"({n_hgp_train}+{n_prscsx_train} train, "
        f"{n_hgp_eval}+{n_prscsx_eval} eval)"
    )
    logger.info("Pending tasks after resume scan: %d", total_tasks)

    if total_tasks == 0:
        logger.info("All requested settings already exist on disk; nothing to generate.")
        return train_manifest, eval_manifest

    # ── dispatch ─────────────────────────────────────────────────────
    n_done = len(completed_keys)
    n_failed = 0
    t0 = time.perf_counter()

    if n_proc <= 1:
        for item in work_items:
            setting_idx, role, row, err, family = _worker_mixed(item)
            n_done += 1
            if err is not None:
                n_failed += 1
                logger.error(f"EXCEPTION {family}/{role} #{setting_idx}: {err}")
            elif row is None:
                n_failed += 1
                logger.warning(
                    f"FAILED {family}/{role} #{setting_idx}: "
                    "could not satisfy split constraints after all attempts"
                )
            else:
                (train_rows if role == "train" else eval_rows).append(row)
                train_manifest, eval_manifest = _write_manifest_checkpoints(
                    out_dir=out_dir,
                    train_rows=train_rows,
                    eval_rows=eval_rows,
                    base_seed=base_seed,
                )
                logger.info(
                    f"[{n_done}/{len(completed_keys) + total_tasks}] OK {family}/{role} #{setting_idx}  "
                    f"(seed={row['seed']}, Kc={row['num_clusters']}, N={row['N']}, M={row['M']})"
                )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_proc) as pool:
            for setting_idx, role, row, err, family in pool.imap_unordered(
                _worker_mixed, work_items
            ):
                n_done += 1
                if err is not None:
                    n_failed += 1
                    logger.error(f"EXCEPTION {family}/{role} #{setting_idx}: {err}")
                elif row is None:
                    n_failed += 1
                    logger.warning(
                        f"FAILED {family}/{role} #{setting_idx}: "
                        "could not satisfy split constraints after all attempts"
                    )
                else:
                    (train_rows if role == "train" else eval_rows).append(row)
                    train_manifest, eval_manifest = _write_manifest_checkpoints(
                        out_dir=out_dir,
                        train_rows=train_rows,
                        eval_rows=eval_rows,
                        base_seed=base_seed,
                    )
                    logger.info(
                        f"[{n_done}/{len(completed_keys) + total_tasks}] OK {family}/{role} #{setting_idx}  "
                        f"(seed={row['seed']}, Kc={row['num_clusters']}, N={row['N']}, M={row['M']})"
                    )

    elapsed = time.perf_counter() - t0

    train_manifest, eval_manifest = _write_manifest_checkpoints(
        out_dir=out_dir,
        train_rows=train_rows,
        eval_rows=eval_rows,
        base_seed=base_seed,
    )

    train_manifest_path = out_dir / "train_manifest.csv"
    eval_manifest_path = out_dir / "eval_manifest.csv"

    logger.info(f"Wrote train manifest: {len(train_manifest)} rows -> {train_manifest_path}")
    logger.info(f"Wrote eval  manifest: {len(eval_manifest)} rows -> {eval_manifest_path}")
    logger.info(f"Failed: {n_failed}/{total_tasks}")
    logger.info(f"Elapsed: {elapsed:.1f}s  ({elapsed / max(n_done, 1):.2f}s/task)")

    if n_failed > 0:
        logger.warning(f"{n_failed} task(s) failed — check log for details.")

    return train_manifest, eval_manifest


# ====================================================================
# CLI
# ====================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixed HGP + PRS-CSx train/eval dist-shift datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", type=str,
                        help="Directory for pickles, manifests, and log file.")
    parser.add_argument("--num_train_settings", type=int, required=True,
                        help="Total train settings (split ~50/50 between HGP and PRS-CSx).")
    parser.add_argument("--num_eval_settings", type=int, required=True,
                        help="Total eval settings (split ~50/50 between HGP and PRS-CSx).")
    parser.add_argument("--base_seed", type=int, default=1990)
    parser.add_argument("--max_attempts_per_setting", type=int, default=40)
    parser.add_argument(
        "--n_proc", type=int, default=0,
        help="Number of worker processes.  0 = auto (os.cpu_count()-1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed triplets already present in output_dir and generate only missing settings.",
    )
    parser.add_argument(
        "--rebuild_manifests_only",
        action="store_true",
        help="Recover train/eval manifests from completed triplets already on disk, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve n_proc: 0 → auto; reserve 1 CPU for the main process
    n_proc = args.n_proc
    if n_proc <= 0:
        n_proc = max(1, (os.cpu_count() or 1) - 1)
    total = args.num_train_settings + args.num_eval_settings
    n_proc = min(n_proc, total)

    print(
        f"Detected {os.cpu_count()} CPUs, using {n_proc} worker processes "
        f"(1 reserved for main)"
    )

    t0 = time.perf_counter()
    train_manifest, eval_manifest = generate_mixed_train_eval(
        output_dir=args.output_dir,
        num_train_settings=args.num_train_settings,
        num_eval_settings=args.num_eval_settings,
        base_seed=args.base_seed,
        max_attempts_per_setting=args.max_attempts_per_setting,
        n_proc=n_proc,
        resume=args.resume,
        rebuild_manifests_only=args.rebuild_manifests_only,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[done] train={len(train_manifest)} eval={len(eval_manifest)}  "
        f"elapsed={elapsed:.1f}s"
    )
