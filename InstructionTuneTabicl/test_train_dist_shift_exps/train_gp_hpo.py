#!/usr/bin/env python
"""
train_gp_hpo.py
===============
Hyperparameter sweep launcher for GP Stage-3 fine-tuning.

Overview
--------
This script has **two operating modes**, selected via the ``--single_run`` flag:

1. **Orchestrator mode** (default – no ``--single_run``):
   Called once by ``run_gp_hpo.sh``.  It reads the HPO search-space lists
   (``--lr_values``, ``--max_steps_values``, etc.), builds the full Cartesian
   product of settings, creates a shared datetime main run directory under both
   CHECKPOINT_DIR and WANDB_DIR, writes ``hpo_runs_manifest.csv``, and then
   spawns one **child process per setting** by re-invoking *this same script*
   with ``--single_run`` and a concrete set of hyperparameter values.  It
   manages a pool of concurrent child processes, assigning each one a GPU slot
   via ``CUDA_VISIBLE_DEVICES``.

2. **Worker mode** (``--single_run``):
   One invocation per HPO setting.  It imports ``_build_tabicl_config`` from
   ``train_gp.py`` and ``GPTrainer`` from ``gp_trainer.py``, builds a config
   namespace, and runs the full training loop for that single setting.  All
   output (stdout + stderr) is redirected to ``<run_wandb_dir>/runner.log``.

Execution flow
--------------
::

   run_gp_hpo.sh
       └─► train_gp_hpo.py  (orchestrator / _main_hpo)
               creates  CHECKPOINT_DIR/<run_id>/
               creates  WANDB_DIR/<run_id>/
               writes   hpo_runs_manifest.csv
               for each setting:
                   subprocess  train_gp_hpo.py --single_run ...  (_single_run)
                       └─► _build_tabicl_config()  (from train_gp.py)
                       └─► GPTrainer.train()        (from gp_trainer.py)
                               checkpoint  → CHECKPOINT_DIR/<run_id>/<run_name>/
                               wandb logs  → WANDB_DIR/<run_id>/<run_name>/

Design goals
------------
- Reuse existing training code from train_gp.py and GPTrainer without modifying them.
- Create one shared datetime run directory under both checkpoint and wandb roots.
- Create one sub-directory per hyperparameter setting inside each main run dir.
- Support concurrent runs across multiple GPUs, and multiple runs per GPU.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Ensure local experiment modules (hpo_stage2_eval, etc.) are importable
_EXP_DIR = str(Path(__file__).resolve().parent)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _detect_gpu_ids() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        return ids

    try:
        import torch

        n = torch.cuda.device_count()
        return [str(i) for i in range(n)]
    except Exception:
        return []


def _parse_float_list(raw: str) -> List[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value in list.")
    return [float(x) for x in vals]


def _parse_int_list(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one int value in list.")
    return [int(x) for x in vals]


def _fmt_float(v: float) -> str:
    return f"{v:.12g}"


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "")


@dataclass
class RunSetting:
    run_name: str
    lr: float
    max_steps: int
    batch_size: int
    weight_decay: float
    proximal_wd: bool
    run_checkpoint_dir: str
    run_wandb_dir: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch HPO sweeps for GP Stage-3 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--single_run", action="store_true", default=False)

    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--wandb_dir", required=True)
    p.add_argument("--checkpoint_path", default=None)

    p.add_argument("--lr_values", default="2e-6")
    p.add_argument("--max_steps_values", default="100")
    p.add_argument("--batch_size_values", default="8")
    p.add_argument("--weight_decay_values", default="0")
    p.add_argument(
        "--proximal_wd_values",
        default="false",
        help=(
            "Comma-separated list of true/false values to sweep over. "
            "'false' = standard AdamW decay-to-zero; "
            "'true' = proximal/anchor regularisation toward the initial checkpoint. "
            "Use 'false,true' to sweep both modes."
        ),
    )

    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--save_perm_every", type=int, default=20)
    p.add_argument("--save_temp_every", type=int, default=1)

    p.add_argument("--n_eval", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--train_manifest", default="train_manifest.csv")
    p.add_argument("--eval_manifest", default="eval_manifest.csv")

    p.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    p.add_argument("--max_features", type=int, default=None)
    p.add_argument("--no_cache_pickles", action="store_true", default=False)

    p.add_argument("--wandb_log", action="store_true", default=False)
    p.add_argument("--wandb_project", default="TabICL")
    p.add_argument("--wandb_name", default="GP_stage3_hpo")
    p.add_argument(
        "--wandb_mode", default="offline", choices=["online", "offline", "disabled"]
    )

    p.add_argument("--np_seed", type=int, default=42)
    p.add_argument("--torch_seed", type=int, default=42)

    p.add_argument(
        "--max_concurrent",
        type=int,
        default=0,
        help="0 means auto (detected_gpus * runs_per_gpu when using cuda, else 1).",
    )
    p.add_argument(
        "--runs_per_gpu",
        type=int,
        default=1,
        help="Maximum simultaneous runs to place on each detected GPU.",
    )
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--main_run_id", default=None)

    p.add_argument("--run_name", default=None)
    p.add_argument("--run_checkpoint_dir", default=None)
    p.add_argument("--run_wandb_dir", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--proximal_wd", action="store_true", default=False,
                   help="Worker-mode: proximal weight decay flag (set by orchestrator).")
    p.add_argument("--scheduler", default="cosine_warmup",
                   help="LR scheduler: constant | cosine_warmup | cosine_with_restarts | linear_warmup | polynomial_decay_warmup.")
    p.add_argument("--warmup_proportion", type=float, default=0.1,
                   help="Fraction of max_steps used for linear warmup (0.0–1.0). Use -1 to switch to --warmup_steps.")

    p.add_argument(
        "--num_additional_checkpoints",
        type=int,
        default=0,
        help=(
            "Number of periodic step-*.ckpt checkpoints nearest to the best-eval "
            "checkpoint that are also evaluated during Stage-2.  "
            "The reported stage2_eval_acc / stage2_eval_ce are the MEDIAN over "
            "all (num_additional_checkpoints + 1) checkpoints.  "
            "0 = evaluate the best checkpoint only (original behaviour).  "
            "Increasing this value smooths out evaluation noise caused by "
            "checkpoint-to-checkpoint training variance, at the cost of "
            "(num_additional_checkpoints × inference time) extra GPU compute "
            "after training completes."
        ),
    )

    return p


def _single_run(args: argparse.Namespace) -> None:
    """
    **Worker mode** – train one HPO setting in the current process.

    This function is called when the script is invoked with ``--single_run``.
    It is always launched as a *child process* by the orchestrator
    (``_main_hpo`` via ``_launch_run``), never directly by the user.

    It:
      1. Creates the per-setting checkpoint and wandb output directories.
      2. Imports ``_build_tabicl_config`` from ``train_gp.py`` and
         ``GPTrainer`` from ``gp_trainer.py``  (reusing all existing training
         logic without modification).
      3. Builds a ``tabicl``-compatible config namespace for the given
         hyperparameter values.
      4. Instantiates ``GPTrainer`` and calls ``trainer.train()``.

    Stdout and stderr are redirected to ``<run_wandb_dir>/runner.log`` by
    the parent process before this function is entered.
    """
    required = [
        args.run_name,
        args.run_checkpoint_dir,
        args.run_wandb_dir,
        args.lr,
        args.max_steps,
        args.batch_size,
        args.weight_decay,
    ]
    if any(v is None for v in required):
        raise ValueError("single_run mode requires run_name/dirs/lr/max_steps/batch_size/weight_decay")

    os.makedirs(args.run_checkpoint_dir, exist_ok=True)
    os.makedirs(args.run_wandb_dir, exist_ok=True)

    from torch.multiprocessing import set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    from train_gp import _build_tabicl_config
    from gp_trainer import GPTrainer

    one = argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proximal_wd=args.proximal_wd,
        save_perm_every=args.save_perm_every,
        save_temp_every=args.save_temp_every,
        n_eval=args.n_eval,
        eval_seed=args.eval_seed,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        wandb_name=f"{args.wandb_name}_{args.run_name}",
        wandb_mode=args.wandb_mode,
        device=args.device,
        np_seed=args.np_seed,
        torch_seed=args.torch_seed,
        max_features=args.max_features,
        no_cache_pickles=args.no_cache_pickles,
        scheduler=args.scheduler,
        warmup_proportion=args.warmup_proportion,
    )

    print("=" * 72)
    print(f"[RUN START] {args.run_name}")
    print(f"  checkpoint_dir : {args.run_checkpoint_dir}")
    print(f"  wandb_dir      : {args.run_wandb_dir}")
    print(f"  lr             : {args.lr}")
    print(f"  max_steps      : {args.max_steps}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  weight_decay   : {args.weight_decay}")
    print(f"  proximal_wd    : {args.proximal_wd}")
    print(f"  device         : {args.device}")
    print("=" * 72)

    config = _build_tabicl_config(one, ckpt_dir=args.run_checkpoint_dir, wandb_dir=args.run_wandb_dir)
    trainer = GPTrainer(config)
    trainer.train()

    print(f"[RUN DONE] {args.run_name}")


def _write_manifest(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "lr",
        "max_steps",
        "batch_size",
        "weight_decay",
        "proximal_wd",
        "checkpoint_dir",
        "wandb_dir",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _launch_run(
    script_path: Path,
    args: argparse.Namespace,
    setting: RunSetting,
    gpu_id: Optional[str],
) -> tuple[subprocess.Popen, object]:
    """
    Spawn a child worker process for one HPO setting.

    The child process is *this same script* (``train_gp_hpo.py``) re-invoked
    with ``--single_run`` and a concrete set of hyperparameters.  Re-using the
    same script keeps the import path and sys.path setup identical for every
    worker, and means no separate entry-point file is needed.

    Parameters
    ----------
    script_path : Path
        Absolute path to this file (``train_gp_hpo.py``).  The child is
        launched as ``python <script_path> --single_run ...``.
    args : argparse.Namespace
        The parent-process parsed arguments (fixed options shared across all
        runs, e.g. data_dir, wandb_project, micro_batch_size).
    setting : RunSetting
        Hyperparameters and output directories specific to this worker.
    gpu_id : Optional[str]
        GPU index string to assign via ``CUDA_VISIBLE_DEVICES``.  If None the
        child inherits the parent environment unchanged (CPU or whatever the
        parent has set).

    Returns
    -------
    (proc, log_f)
        ``proc``  – the ``subprocess.Popen`` handle for the child process.
        ``log_f`` – the open file handle for ``<run_wandb_dir>/runner.log``
                    (stdout + stderr of the child are redirected here).
                    The caller is responsible for closing it when the process
                    finishes.
    """
    env = os.environ.copy()
    run_device = args.device
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        run_device = "cuda"

    cmd = [
        args.python_exe,
        str(script_path),
        "--single_run",
        "--data_dir",
        args.data_dir,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--wandb_dir",
        args.wandb_dir,
        "--run_name",
        setting.run_name,
        "--run_checkpoint_dir",
        setting.run_checkpoint_dir,
        "--run_wandb_dir",
        setting.run_wandb_dir,
        "--lr",
        str(setting.lr),
        "--max_steps",
        str(setting.max_steps),
        "--batch_size",
        str(setting.batch_size),
        "--weight_decay",
        str(setting.weight_decay),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--save_perm_every",
        str(args.save_perm_every),
        "--save_temp_every",
        str(args.save_temp_every),
        "--n_eval",
        str(args.n_eval),
        "--eval_seed",
        str(args.eval_seed),
        "--train_manifest",
        args.train_manifest,
        "--eval_manifest",
        args.eval_manifest,
        "--device",
        run_device,
        "--wandb_project",
        args.wandb_project,
        "--wandb_name",
        args.wandb_name,
        "--wandb_mode",
        args.wandb_mode,
        "--np_seed",
        str(args.np_seed),
        "--torch_seed",
        str(args.torch_seed),
        "--scheduler",
        args.scheduler,
        "--warmup_proportion",
        str(args.warmup_proportion),
    ]

    if args.checkpoint_path:
        cmd.extend(["--checkpoint_path", args.checkpoint_path])
    if args.max_features is not None:
        cmd.extend(["--max_features", str(args.max_features)])
    if args.no_cache_pickles:
        cmd.append("--no_cache_pickles")
    if args.wandb_log:
        cmd.append("--wandb_log")
    if setting.proximal_wd:
        cmd.append("--proximal_wd")

    log_path = Path(setting.run_wandb_dir) / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def _write_fixed_config_csv(
    args: argparse.Namespace,
    main_wandb_dir: Path,
    main_ckpt_dir: Path,
) -> None:
    """
    Write ``hpo_fixed_config.csv`` recording every configuration value that was
    held *constant* across all HPO settings.

    Two categories of parameters are recorded:

    1. **Run-level fixed args** – values supplied via ``run_gp_hpo.sh`` /
       ``train_gp_hpo.py`` CLI that are not part of the HPO search space
       (which is ``lr × max_steps × batch_size × weight_decay``).

    2. **Model architecture constants** – values hard-coded inside
       ``_build_tabicl_config`` and passed directly to ``GPTrainer``.

    The CSV has three columns:

    ``config_name``
        The argument or parameter name (matches ``argparse`` / ``tabicl`` keys).
    ``value``
        The value used during this HPO sweep.
    ``description``
        A short human-readable explanation of the parameter.

    The file is written into both ``main_wandb_dir`` and ``main_ckpt_dir`` so
    that a self-contained record exists alongside the run logs and checkpoints.
    """
    rows = [
        # ------------------------------------------------------------------ #
        # Run-level fixed training / eval / setup args                        #
        # ------------------------------------------------------------------ #
        ("data_dir",           str(getattr(args, "data_dir", "")),
         "Root directory containing train/eval manifests and pickle files."),
        ("init_checkpoint_path", str(getattr(args, "checkpoint_path", "") or ""),
         "Pre-trained TabICL checkpoint used to initialise all fine-tuning runs. "
         "Empty string means the public HuggingFace release was used."),
        ("proximal_wd_values",  str(getattr(args, "proximal_wd_values", "false")),
         "Swept proximal_wd values. false = standard AdamW decay-to-zero; "
         "true = proximal/anchor regularisation toward the initial checkpoint "
         "(lambda/2)*||theta-theta_0||^2. Per-run value is in hpo_runs_manifest.csv."),
        ("micro_batch_size",   str(args.micro_batch_size),
         "Gradient-accumulation micro-batch size (1 = no accumulation)."),
        ("save_perm_every",    str(args.save_perm_every),
         "Save a permanent checkpoint every N gradient steps."),
        ("save_temp_every",    str(args.save_temp_every),
         "Save a temp (rolling) checkpoint every N gradient steps."),
        ("n_eval",             str(args.n_eval),
         "Number of eval-manifest rows used for per-step evaluation (stage-1)."),
        ("eval_seed",          str(args.eval_seed),
         "Fixed RNG seed for constructing the stage-1 evaluation batch."),
        ("train_manifest",     str(args.train_manifest),
         "Filename of the training manifest CSV inside data_dir."),
        ("eval_manifest",      str(args.eval_manifest),
         "Filename of the evaluation manifest CSV inside data_dir."),
        ("device",             str(args.device),
         "PyTorch device string (cuda / cpu)."),
        ("max_features",       str(args.max_features) if args.max_features is not None else "auto",
         "Max feature dimension; 'auto' means computed from the manifests at runtime."),
        ("no_cache_pickles",   str(args.no_cache_pickles),
         "Disable in-memory pickle caching when True (reduces RAM at cost of I/O)."),
        ("wandb_project",      str(args.wandb_project),
         "W&B project name."),
        ("wandb_name",         str(args.wandb_name),
         "W&B run-name base (per-setting suffix appended at runtime)."),
        ("wandb_mode",         str(args.wandb_mode),
         "W&B logging mode: online | offline | disabled."),
        ("np_seed",            str(args.np_seed),
         "NumPy global RNG seed."),
        ("torch_seed",         str(args.torch_seed),
         "PyTorch global RNG seed."),
        ("runs_per_gpu",       str(args.runs_per_gpu),
         "Maximum simultaneous HPO runs placed on each detected GPU."),
        ("max_concurrent",     str(args.max_concurrent),
         "Hard cap on total concurrent child processes (0 = auto from gpu count × runs_per_gpu)."),
        # ------------------------------------------------------------------ #
        # Model architecture constants (hard-coded in _build_tabicl_config)   #
        # These define the Stage-3 TabICL architecture and are NOT exposed as  #
        # CLI arguments – they are fixed for all HPO runs.                    #
        # ------------------------------------------------------------------ #
        ("arch_embed_dim",     "128",
         "Token embedding dimension shared by column- and row-transformers."),
        ("arch_col_num_blocks","3",
         "Number of transformer blocks in the column embedder."),
        ("arch_col_nhead",     "4",
         "Attention heads in the column embedder."),
        ("arch_col_num_inds",  "128",
         "Number of inducing points for the column embedder."),
        ("arch_freeze_col",    "True",
         "Column embedder weights are frozen during Stage-3 fine-tuning."),
        ("arch_row_num_blocks","3",
         "Number of transformer blocks in the row interactor."),
        ("arch_row_nhead",     "8",
         "Attention heads in the row interactor."),
        ("arch_row_num_cls",   "4",
         "Number of CLS tokens in the row interactor."),
        ("arch_row_rope_base", "100000",
         "RoPE base frequency for the row interactor."),
        ("arch_freeze_row",    "True",
         "Row interactor weights are frozen during Stage-3 fine-tuning."),
        ("arch_icl_num_blocks","12",
         "Number of transformer blocks in the ICLearning (tf_icl) head."),
        ("arch_icl_nhead",     "4",
         "Attention heads in the ICLearning head."),
        ("arch_ff_factor",     "2",
         "Feed-forward hidden-dim multiplier relative to embed_dim."),
        ("arch_norm_first",    "True",
         "Use pre-LayerNorm (True) rather than post-LayerNorm architecture."),
        ("arch_scheduler",     args.scheduler,
         "Learning-rate scheduler type."),
        ("arch_warmup_proportion", str(args.warmup_proportion),
         "Fraction of max_steps used for linear LR warmup."),
        ("arch_gradient_clipping", "1.0",
         "Max gradient norm for gradient clipping."),
        ("arch_dtype",         "float32",
         "Floating-point precision used for training."),
        ("arch_only_load_model","True",
         "Only load model weights from checkpoint, resetting training state."),
    ]

    fieldnames = ["config_name", "value", "description"]
    for out_dir in (main_wandb_dir, main_ckpt_dir):
        csv_path = out_dir / "hpo_fixed_config.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for name, val, desc in rows:
                w.writerow({"config_name": name, "value": val, "description": desc})
        print(f"Fixed config saved : {csv_path}")


def _main_hpo(args: argparse.Namespace) -> None:
    """
    **Orchestrator mode** – build the sweep grid and manage child processes.

    This is the default entry point when the script is called without
    ``--single_run`` (i.e. directly from ``run_gp_hpo.sh``).

    Steps
    -----
    1. Parse the comma-separated HPO search-space lists into Python lists.
    2. Build the full Cartesian product of (lr × max_steps × batch_size ×
       weight_decay) as a list of ``RunSetting`` objects.
    3. Create one shared datetime main run directory under both
       ``CHECKPOINT_DIR`` and ``WANDB_DIR``.
    4. Write ``hpo_runs_manifest.csv`` (maps run names to hyperparameters and
       output directories) into both main run directories.
    5. Detect available GPUs and build a ``free_gpu_pool`` – a list of GPU ID
       strings with ``runs_per_gpu`` tokens each, so the same GPU ID appears
       multiple times if more than one run is allowed per GPU.
    6. Run a scheduler loop: while there are pending settings or active child
       processes, pop a GPU token and launch a worker (``_launch_run``) until
       ``max_concurrent`` is reached.  As workers finish, their GPU tokens are
       returned to the pool and new workers are started.
    7. Report failures and raise if any worker exited non-zero.
    """
    lr_values = _parse_float_list(args.lr_values)
    max_steps_values = _parse_int_list(args.max_steps_values)
    batch_size_values = _parse_int_list(args.batch_size_values)
    weight_decay_values = _parse_float_list(args.weight_decay_values)
    proximal_wd_values = [
        v.strip().lower() == "true"
        for v in args.proximal_wd_values.split(",")
        if v.strip()
    ]
    if not proximal_wd_values:
        raise ValueError("--proximal_wd_values must contain at least one of: false, true")

    main_run_id = args.main_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    main_ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve() / main_run_id
    main_wandb_dir = Path(args.wandb_dir).expanduser().resolve() / main_run_id
    main_ckpt_dir.mkdir(parents=True, exist_ok=True)
    main_wandb_dir.mkdir(parents=True, exist_ok=True)

    settings: List[RunSetting] = []
    idx = 0
    for lr, max_steps, batch_size, wd, pwd in itertools.product(
        lr_values, max_steps_values, batch_size_values, weight_decay_values, proximal_wd_values
    ):
        idx += 1
        run_name = _safe_name(
            f"run_{idx:03d}__lr_{_fmt_float(lr)}__max_steps_{max_steps}"
            f"__batch_size_{batch_size}__weight_decay_{_fmt_float(wd)}"
            f"__proximal_wd_{pwd}"
        )
        settings.append(
            RunSetting(
                run_name=run_name,
                lr=lr,
                max_steps=max_steps,
                batch_size=batch_size,
                weight_decay=wd,
                proximal_wd=pwd,
                run_checkpoint_dir=str(main_ckpt_dir / run_name),
                run_wandb_dir=str(main_wandb_dir / run_name),
            )
        )

    manifest_rows = [
        {
            "run_name": s.run_name,
            "lr": _fmt_float(s.lr),
            "max_steps": str(s.max_steps),
            "batch_size": str(s.batch_size),
            "weight_decay": _fmt_float(s.weight_decay),
            "proximal_wd": str(s.proximal_wd),
            "checkpoint_dir": s.run_checkpoint_dir,
            "wandb_dir": s.run_wandb_dir,
        }
        for s in settings
    ]
    _write_manifest(manifest_rows, main_wandb_dir / "hpo_runs_manifest.csv")
    _write_manifest(manifest_rows, main_ckpt_dir / "hpo_runs_manifest.csv")

    use_cuda = args.device.startswith("cuda") and _cuda_available()
    gpu_ids = _detect_gpu_ids() if use_cuda else []

    if args.runs_per_gpu < 1:
        raise ValueError("--runs_per_gpu must be >= 1")

    if args.max_concurrent > 0:
        max_concurrent = args.max_concurrent
    else:
        max_concurrent = (len(gpu_ids) * args.runs_per_gpu) if gpu_ids else 1

    print("=" * 72)
    print("GP HPO launcher")
    print(f"main_run_id    : {main_run_id}")
    print(f"checkpoint_root: {main_ckpt_dir}")
    print(f"wandb_root     : {main_wandb_dir}")
    print(f"num_settings   : {len(settings)}")
    print(f"device_base    : {args.device}")
    print(f"detected_gpus  : {gpu_ids if gpu_ids else 'none'}")
    print(f"runs_per_gpu   : {args.runs_per_gpu}")
    print(f"max_concurrent : {max_concurrent}")
    print("=" * 72)

    script_path = Path(__file__).resolve()
    pending = settings.copy()
    active: Dict[subprocess.Popen, Dict[str, object]] = {}
    failures: List[str] = []

    free_gpu_pool: List[str] = []
    for gpu_id in gpu_ids:
        free_gpu_pool.extend([gpu_id] * args.runs_per_gpu)

    while pending or active:
        while pending and len(active) < max_concurrent:
            setting = pending[0]

            if gpu_ids:
                if not free_gpu_pool:
                    break
                gpu_id = free_gpu_pool.pop(0)
            else:
                gpu_id = None

            pending.pop(0)
            proc, log_f = _launch_run(script_path, args, setting, gpu_id)
            active[proc] = {"setting": setting, "gpu_id": gpu_id, "log_f": log_f}

            print(
                f"[LAUNCH] {setting.run_name} "
                f"(gpu={gpu_id if gpu_id is not None else 'cpu'}, pid={proc.pid})"
            )

        done_procs = []
        for proc, meta in active.items():
            rc = proc.poll()
            if rc is None:
                continue
            done_procs.append((proc, rc, meta))

        for proc, rc, meta in done_procs:
            setting = meta["setting"]
            gpu_id = meta["gpu_id"]
            log_f = meta["log_f"]
            log_f.close()

            if gpu_id is not None:
                free_gpu_pool.append(gpu_id)

            if rc == 0:
                print(f"[DONE]   {setting.run_name} (rc=0)")
            else:
                print(f"[FAILED] {setting.run_name} (rc={rc})")
                failures.append(setting.run_name)
                # Print last 40 lines of the runner log so the error is visible
                log_path = Path(setting.run_wandb_dir) / "runner.log"
                if log_path.is_file():
                    lines = log_path.read_text(errors="replace").splitlines()
                    tail = lines[-40:] if len(lines) > 40 else lines
                    print(f"  --- last {len(tail)} lines of {log_path} ---")
                    for ln in tail:
                        print(f"  | {ln}")
                    print(f"  --- end of log ---")

            del active[proc]

        if active:
            time.sleep(2.0)

    if failures:
        print("\nSome runs failed:")
        for r in failures:
            print(f"  - {r}")
        raise RuntimeError(f"{len(failures)} run(s) failed.")

    print("\nAll HPO runs completed successfully.")
    print(f"Manifest: {main_wandb_dir / 'hpo_runs_manifest.csv'}")

    # Write a record of every fixed (non-swept) config parameter so that
    # the run directory is self-contained and fully reproducible.
    _write_fixed_config_csv(args, main_wandb_dir, main_ckpt_dir)

    # ---- Stage-2 evaluation ----------------------------------------------
    # Run post-hoc stage-2 eval on the held-out eval-manifest slice for
    # every HPO setting's best checkpoint, then update the manifest CSV.
    print("\n" + "=" * 72)
    print("Starting Stage-2 evaluation ...")
    manifest_csv = main_wandb_dir / "hpo_runs_manifest.csv"
    from hpo_stage2_eval import run_stage2_eval
    run_stage2_eval(
        manifest_path=str(manifest_csv),
        n_eval=args.n_eval,
        data_dir=args.data_dir,
        device=args.device if args.device.startswith("cuda") else None,
        max_features=args.max_features,
        eval_seed=args.eval_seed,
        eval_manifest_filename=args.eval_manifest,
        num_additional_checkpoints=args.num_additional_checkpoints,
    )
    print("Stage-2 evaluation complete.")


def main() -> None:
    args = _build_parser().parse_args()
    if args.single_run:
        _single_run(args)
    else:
        _main_hpo(args)


if __name__ == "__main__":
    main()
